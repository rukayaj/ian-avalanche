import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
import copy
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Any, Optional, Tuple, Set
import re
import subprocess

from PIL import Image
import fitz

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional for some utilities
    pd = None
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - allows offline utilities to import
    OpenAI = None  # type: ignore[assignment]

from .batch_builder import (
    build_detect_regions_request,
    build_extract_all_request,
    build_extract_precip_request,
    build_extract_temperature_request,
    build_extract_wind_request,
    write_jsonl,
)
from .parse_batch_results import (
    parse_custom_id,
    parse_detection_pages,
    parse_jsonl_file,
    extract_structured_output,
)
from .pdf_render import render_pdf_to_images
from .utils import crop_normalized_box
from .aggregation import (
    DEFAULT_NUMERIC_TOLERANCES,
    normalize_location_string,
    normalize_precip_type,
    normalize_text_value,
    format_numeric_value,
    harmonize_locations,
    aggregate_runs,
)


DEFAULT_KINDS = ("wind", "precipitation", "temperature")
SECTION_KIND_MAP = {
    "Wind": "wind",
    "Precip": "precipitation",
    "Temperature": "temperature",
}
KIND_REPROMPT_RULES = {
    "wind": (
        "Wind graph rules:\n"
        "- Extract exactly 24 hours (18:00→17:00) with hour_index 0..23.\n"
        "- Wind speed (blue) and gust (pink) must be integer mph values aligned with the axis ticks. "
        "Check the 18:00 point directly on the grid and keep gusts ≥ speeds for every hour.\n"
        "- Wind direction is the text above each hour, using only the 16-point compass list."
    ),
    "precipitation": (
        "Precipitation graph rules:\n"
        "- Extract exactly 24 hourly entries (18:00→17:00) with hour_index 0..23.\n"
        "- Rainfall (blue bars) is reported to the nearest 0.1 mm (0.0 when absent); snowfall (white bars) to one decimal place. "
        "Treat blue and white bars as separate values for the same hour—do not add them.\n"
        "- Precipitation type text must match what is printed above each hour.\n"
        "- Calibrate using the 18:00 bars before filling in the remaining hours."
    ),
    "temperature": (
        "Temperature graph rules:\n"
        "- Extract 24 hours (18:00→17:00) with hour_index 0..23.\n"
        "- Air temperature (blue) comes from the left °C axis to one decimal place.\n"
        "- Freezing level (pink) and wet bulb freezing level (yellow) come from the right-hand axis, rounded to the nearest 10 m. "
        "Ignore static summit or elevation labels printed on the background.\n"
        "- Calibrate on the 18:00 point for each line before completing the series."
    ),
}


def extract_page9_rows(pdf_path: Path) -> List[Dict[str, Any]]:
    """Parse page 9 table for cumulative rain/snow and forecast start date."""
    rows: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return rows
    try:
        if doc.page_count < 9:
            return rows
    finally:
        doc.close()

    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-f", "9", "-l", "9", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return rows

    text = result.stdout.splitlines()
    date_str = ""
    for line in text:
        m = re.search(r"1800 on (.+?) until", line)
        if m:
            date_str = m.group(1).strip()
            break
    if date_str:
        rows.append(
            {
                "SourceFile": pdf_path.name,
                "Page": 9,
                "Location": "",
                "Section": "Meta",
                "Measurement": "ForecastWindowStart",
                "MeasurementType": "Date",
                "Units": "",
                "ForecastWindowStartLocal": "",
                "HourLabel": "",
                "HourIndex": "",
                "TimestampLocal": "",
                "ValueNumeric": "",
                "ValueText": date_str,
                "Notes": "",
            }
        )

    row_re = re.compile(r"\s*([A-Za-z' ]{3,})\s+([0-9]+\.?[0-9]*)\s+([0-9]+\.?[0-9]*)\s*$")
    for line in text:
        m = row_re.match(line)
        if not m:
            continue
        site = m.group(1).strip()
        snow_val = float(m.group(2))
        rain_val = float(m.group(3))
        base = {
            "SourceFile": pdf_path.name,
            "Page": 9,
            "Location": site,
            "Section": "Precip",
            "ForecastWindowStartLocal": "",
            "HourLabel": "",
            "HourIndex": "",
            "TimestampLocal": "",
            "Notes": "",
        }
        rows.append(
            {
                **base,
                "Measurement": "Accumulation",
                "MeasurementType": "SnowTotal_cm",
                "Units": "cm",
                "ValueNumeric": snow_val,
                "ValueText": "",
            }
        )
        rows.append(
            {
                **base,
                "Measurement": "Accumulation",
                "MeasurementType": "RainTotal_mm",
                "Units": "mm",
                "ValueNumeric": rain_val,
                "ValueText": "",
            }
        )
    return rows


@dataclass(frozen=True)
class BatchPaths:
    detect_jsonl: Path
    detect_results: Path
    extract_jsonl: Path
    extract_results: Path
    out_csv: Path
    disagreement_report: Optional[Path] = None
def discover_pdfs(explicit: Sequence[str] | None, input_dir: Path) -> List[Path]:
    """Return the list of PDFs to process, preferring explicit arguments over directory scanning."""
    if explicit:
        pdfs = [Path(p).expanduser() for p in explicit]
    else:
        search_dir = input_dir if input_dir else Path(".")
        candidates = sorted(search_dir.rglob("*.pdf"))
        # If the target directory is empty, fall back to PDFs in CWD for convenience
        if not candidates and search_dir != Path("."):
            candidates = sorted(Path(".").glob("*.pdf"))
        pdfs = candidates
    disabled_dir = (input_dir / "disabled").resolve()

    def is_disabled(path: Path) -> bool:
        try:
            path.resolve().relative_to(disabled_dir)
            return True
        except ValueError:
            return False

    pdfs = [p for p in pdfs if not is_disabled(p)]
    pdfs = [p for p in pdfs if p.suffix.lower() == ".pdf" and p.is_file()]
    if not pdfs:
        target = input_dir if not explicit else Path(".")
        raise SystemExit(
            f"No PDF files found to process. "
            f"Provide --input or place PDFs in {target.resolve()}."
        )
    resolved: List[Path] = []
    seen_paths: set[Path] = set()
    seen_names: Dict[str, Path] = {}
    for pdf in pdfs:
        abs_pdf = pdf.resolve()
        if abs_pdf in seen_paths:
            continue
        seen_paths.add(abs_pdf)
        name = abs_pdf.name
        if name in seen_names and seen_names[name] != abs_pdf:
            raise SystemExit(f"Duplicate PDF file name detected: {name}. Paths: {seen_names[name]} and {abs_pdf}")
        seen_names[name] = abs_pdf
        resolved.append(abs_pdf)
    return resolved


def build_pdf_lookup(pdfs: Sequence[Path]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for pdf in pdfs:
        lookup[pdf.name] = pdf
    return lookup


def build_detection_jsonl(model: str, pdfs: Sequence[Path], out_jsonl: Path, dpi: int, max_page_px: int) -> Path:
    items: List[Dict] = []
    for pdf in pdfs:
        pages = render_pdf_to_images(str(pdf), dpi=dpi)
        total = len(pages)
        for page_index, img in enumerate(pages, start=1):
            if total > 2 and page_index in (1, total):
                continue
            req = build_detect_regions_request(model, img, max_side=max_page_px)
            req["custom_id"] = f"detect::{pdf.name}::p{page_index}"
            items.append(req)
    write_jsonl(items, str(out_jsonl))
    print(f"Wrote {len(items)} detection batch items -> {out_jsonl}")
    return out_jsonl


def build_extraction_jsonl_from_detect(
    model: str,
    detection_map: Dict[str, Dict[int, Dict[str, object]]],
    pdf_lookup: Dict[str, Path],
    out_jsonl: Path,
    dpi: int,
    max_crop_px: int,
    kinds: Iterable[str],
    *,
    only_graphs: Optional[Iterable[Tuple[str, int, str]]] = None,
    prompt_overrides: Optional[Dict[Tuple[str, int, str], str]] = None,
    prefer_combined: bool = True,
) -> Path:
    kind_set = {k for k in kinds if k}
    if not kind_set:
        raise SystemExit("At least one graph kind must be specified for extraction.")
    items: List[Dict] = []
    rendered_cache: Dict[str, List[Image.Image]] = {}
    allowed_graphs: Optional[Set[Tuple[str, int, str]]] = None
    if only_graphs:
        allowed_graphs = {tuple(g) for g in only_graphs}
    override_map = prompt_overrides or {}
    for file_name, pages in detection_map.items():
        pdf_path = pdf_lookup.get(file_name)
        if not pdf_path:
            print(f"Warning: detection results reference {file_name}, but no matching PDF was provided.")
            continue
        images = rendered_cache.get(file_name)
        if images is None:
            images = render_pdf_to_images(str(pdf_path), dpi=dpi)
            rendered_cache[file_name] = images
        total_pages = len(images)
        for page_num, page_data in sorted(pages.items()):
            if not (1 <= page_num <= total_pages):
                print(f"Warning: skipping page {page_num} for {file_name}; out of range.")
                continue
            page_img = images[page_num - 1]
            crops_by_kind: Dict[str, any] = {}
            graphs = page_data.get("graphs") or []
            for graph in graphs:
                kind = graph.get("kind")
                bbox = graph.get("bbox")
                if kind not in kind_set or not bbox:
                    continue
                try:
                    crop = crop_normalized_box(page_img, tuple(bbox))
                except Exception as exc:
                    print(f"Warning: failed to crop {kind} on {file_name} page {page_num}: {exc}")
                    continue
                crops_by_kind[kind] = crop
            if not crops_by_kind:
                continue
            selected_crops = {
                k: v
                for k, v in crops_by_kind.items()
                if (allowed_graphs is None or (file_name, page_num, k) in allowed_graphs)
            }
            if not selected_crops:
                continue
            have_all_requested = (
                prefer_combined
                and {"wind", "precipitation", "temperature"}.issubset(kind_set)
                and {"wind", "precipitation", "temperature"}.issubset(selected_crops.keys())
            )
            has_override = any(
                override_map.get((file_name, page_num, ck)) for ck in ("wind", "precipitation", "temperature")
            )
            if have_all_requested and not has_override:
                req = build_extract_all_request(model, selected_crops, max_side=max_crop_px)
                req["custom_id"] = f"extract::{file_name}::p{page_num}::combined"
                items.append(req)
                continue
            if "wind" in selected_crops and "wind" in kind_set:
                wind_prompt = override_map.get((file_name, page_num, "wind"))
                req = build_extract_wind_request(
                    model,
                    selected_crops["wind"],
                    prompt=wind_prompt,
                    max_side=max_crop_px,
                )
                req["custom_id"] = f"extract::{file_name}::p{page_num}::wind"
                items.append(req)
            if "precipitation" in selected_crops and "precipitation" in kind_set:
                precip_prompt = override_map.get((file_name, page_num, "precipitation"))
                req = build_extract_precip_request(
                    model,
                    selected_crops["precipitation"],
                    prompt=precip_prompt,
                    max_side=max_crop_px,
                )
                req["custom_id"] = f"extract::{file_name}::p{page_num}::precipitation"
                items.append(req)
            if "temperature" in selected_crops and "temperature" in kind_set:
                temp_prompt = override_map.get((file_name, page_num, "temperature"))
                req = build_extract_temperature_request(
                    model,
                    selected_crops["temperature"],
                    prompt=temp_prompt,
                    max_side=max_crop_px,
                )
                req["custom_id"] = f"extract::{file_name}::p{page_num}::temperature"
                items.append(req)
    write_jsonl(items, str(out_jsonl))
    print(f"Wrote {len(items)} extraction batch items -> {out_jsonl}")
    return out_jsonl


def _series_all_zero(kind: str, payload: Dict[str, Any]) -> bool:
    hours = payload.get("hours") or []
    if not hours:
        return True
    if kind == "wind":
        return all(
            (h.get("wind_speed_mph") or 0) == 0 and (h.get("wind_gust_mph") or 0) == 0
            for h in hours
        )
    if kind == "precipitation":
        return all(
            (h.get("rain_mm") or 0) == 0 and (h.get("snow_cm") or 0) == 0
            for h in hours
        )
    if kind == "temperature":
        return all(
            (h.get("air_temp_c") or 0) == 0
            and (h.get("freezing_level_m") or 0) == 0
            and (h.get("wet_bulb_freezing_level_m") or 0) == 0
            for h in hours
        )
    return False


def _normalize_series(kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = copy.deepcopy(payload)
    data["location"] = normalize_location_string(data.get("location"))
    hours = data.get("hours") or []
    if kind == "precipitation":
        for h in hours:
            h["precip_type"] = normalize_precip_type(h.get("precip_type"))
    return data


def _validate_series(kind: str, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    normalized = _normalize_series(kind, payload)
    hours = normalized.get("hours") or []
    if not hours:
        return False, normalized, "empty_hours"
    if _series_all_zero(kind, normalized):
        return False, normalized, "all_zero_series"
    return True, normalized, ""


def load_structured_results(jsonl_path: Path) -> Tuple[Dict[str, Dict[int, Dict[str, Any]]], List[Dict[str, Any]]]:
    results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    invalid_entries: List[Dict[str, Any]] = []
    if not Path(jsonl_path).exists():
        return results, invalid_entries
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("error"):
                continue
            resp_wrapper = obj.get("response") or {}
            body = resp_wrapper.get("body") or {}
            parsed = extract_structured_output(body)
            if not parsed:
                continue
            meta = parse_custom_id(obj.get("custom_id", ""))
            file_name = meta.get("file") or ""
            page_num = meta.get("page") or 0
            kind = meta.get("kind") or ""
            if not file_name or page_num <= 0:
                continue
            if kind == "combined":
                for sub_kind in ("wind", "precipitation", "temperature"):
                    sub_payload = parsed.get(sub_kind)
                    if sub_payload:
                        ok, normalized_payload, reason = _validate_series(sub_kind, sub_payload)
                        if ok:
                            results.setdefault(file_name, {}).setdefault(page_num, {})[sub_kind] = normalized_payload
                        else:
                            invalid_entries.append(
                                {
                                    "file": file_name,
                                    "page": page_num,
                                    "kind": sub_kind,
                                    "reason": reason,
                                    "line": line_number,
                                }
                            )
            elif kind in ("wind", "precipitation", "temperature"):
                ok, normalized_payload, reason = _validate_series(kind, parsed)
                if ok:
                    results.setdefault(file_name, {}).setdefault(page_num, {})[kind] = normalized_payload
                else:
                    invalid_entries.append(
                        {
                            "file": file_name,
                            "page": page_num,
                            "kind": kind,
                            "reason": reason,
                            "line": line_number,
                        }
                    )
    return results, invalid_entries


def dataframe_to_row_map(df: pd.DataFrame) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    row_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    if df.empty:
        return row_map
    for row in df.to_dict(orient="records"):
        if isinstance(row.get("ValueNumeric"), float) and math.isnan(row.get("ValueNumeric")):
            row["ValueNumeric"] = ""
        if isinstance(row.get("ValueText"), float) and math.isnan(row.get("ValueText")):
            row["ValueText"] = ""
        if isinstance(row.get("Location"), float) and math.isnan(row.get("Location")):
            row["Location"] = ""
        row["Location"] = normalize_location_string(row.get("Location"))
        key = (
            row.get("SourceFile"),
            int(row.get("Page")),
            row.get("Section"),
            row.get("Measurement"),
            row.get("MeasurementType"),
            int(row.get("HourIndex")),
            row.get("HourLabel"),
        )
        row["ValueText"] = normalize_text_value(row.get("Section"), row.get("MeasurementType"), row.get("ValueText"))
        row_map[key] = row
    return row_map


def summarize_payload_for_prompt(kind: str, payload: Dict[str, Any]) -> str:
    hours = payload.get("hours") or []
    if not hours:
        return "(no data)"
    if kind == "wind":
        summary = {
            "wind_speed_mph": [h.get("wind_speed_mph") for h in hours],
            "wind_gust_mph": [h.get("wind_gust_mph") for h in hours],
            "wind_direction": [h.get("wind_direction") for h in hours],
        }
    elif kind == "precipitation":
        summary = {
            "rain_mm": [h.get("rain_mm") for h in hours],
            "snow_cm": [h.get("snow_cm") for h in hours],
            "precip_type": [h.get("precip_type") for h in hours],
        }
    elif kind == "temperature":
        summary = {
            "air_temp_c": [h.get("air_temp_c") for h in hours],
            "freezing_level_m": [h.get("freezing_level_m") for h in hours],
            "wet_bulb_freezing_level_m": [h.get("wet_bulb_freezing_level_m") for h in hours],
        }
    else:
        summary = {"hours": hours}
    return json.dumps(summary, separators=(",", ":"))


def build_rerun_prompt(kind: str, payloads: List[Dict[str, Any]]) -> str:
    rules = KIND_REPROMPT_RULES.get(kind, "")
    prompt = (
        "Previous extractions for this graph disagreed. Re-read the graph carefully and provide a corrected 24-hour series. "
        "Do not average or merge earlier outputs—use the plotted data and axis gridlines.\n"
        f"{rules}\n"
    )
    if payloads:
        prompt += "Earlier runs (for reference only):"
        for idx, payload in enumerate(payloads, start=1):
            prompt += f"\nRun {idx}: {summarize_payload_for_prompt(kind, payload)}"
    prompt += "\nReturn the full series following the schema exactly."
    return prompt


def submit_and_wait(client: OpenAI, jsonl_path: Path, out_results_path: Path, poll_interval: int = 20) -> Path:
    jsonl_path = Path(jsonl_path)
    out_results_path = Path(out_results_path)
    with open(jsonl_path, "rb") as f:
        up = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    print(f"Submitted batch {batch.id}; status={batch.status}")
    while True:
        b = client.batches.retrieve(batch.id)
        print(f"status={b.status}")
        if b.status in ("completed", "failed", "expired", "canceled"):
            break
        time.sleep(poll_interval)
    if b.status != "completed":
        raise SystemExit(f"Batch {b.id} ended with status {b.status}")
    fid = b.output_file_id
    if not fid:
        if b.error_file_id:
            content = client.files.content(b.error_file_id)
            out_results_path.parent.mkdir(parents=True, exist_ok=True)
            err_path = out_results_path.with_suffix(".errors.jsonl")
            with open(err_path, "wb") as f:
                f.write(content.read())
            raise SystemExit(f"Batch completed with errors; details saved to {err_path}")
        raise SystemExit("No output_file_id on completed batch")
    content = client.files.content(fid)
    out_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_results_path, "wb") as f:
        f.write(content.read())
    print(f"Saved results to {out_results_path}")
    return out_results_path


def run_pipeline(
    client: OpenAI,
    model: str,
    pdfs: Sequence[Path],
    paths: BatchPaths,
    dpi: int,
    max_page_px: int,
    max_crop_px: int,
    poll_interval: int,
    kinds: Iterable[str],
    extraction_repeats: int = 2,
    default_numeric_tolerance: float = 1.0,
    tolerance_overrides: Optional[Dict[Tuple[str, str], float]] = None,
    rerun_on_disagreement: bool = True,
) -> Path:
    if pd is None:
        raise ImportError("pandas is required to run the batch pipeline. Please install pandas.")
    if OpenAI is None:
        raise ImportError("openai package is required to run the batch pipeline.")
    pdf_lookup = build_pdf_lookup(pdfs)
    build_detection_jsonl(model, pdfs, paths.detect_jsonl, dpi=dpi, max_page_px=max_page_px)
    submit_and_wait(client, paths.detect_jsonl, paths.detect_results, poll_interval=poll_interval)

    detection_map = parse_detection_pages([paths.detect_results])
    detection_locations = {
        (file_name, page_num): page_data.get("location") or ""
        for file_name, pages in detection_map.items()
        for page_num, page_data in pages.items()
        if page_data.get("location")
    } or None

    repeat_total = max(1, extraction_repeats)
    run_maps: List[Dict[Tuple[Any, ...], Dict[str, Any]]] = []
    structured_runs: List[Dict[str, Dict[int, Dict[str, Any]]]] = []
    invalid_summaries: List[Dict[str, Any]] = []

    base_extract_jsonl = paths.extract_jsonl
    base_extract_results = paths.extract_results

    for run_index in range(repeat_total):
        suffix = "" if run_index == 0 else f"_r{run_index + 1}"
        extract_jsonl_path = (
            base_extract_jsonl
            if run_index == 0
            else base_extract_jsonl.with_name(f"{base_extract_jsonl.stem}{suffix}{base_extract_jsonl.suffix}")
        )
        extract_results_path = (
            base_extract_results
            if run_index == 0
            else base_extract_results.with_name(f"{base_extract_results.stem}{suffix}{base_extract_results.suffix}")
        )

        build_extraction_jsonl_from_detect(
            model,
            detection_map,
            pdf_lookup,
            extract_jsonl_path,
            dpi=dpi,
            max_crop_px=max_crop_px,
            kinds=kinds,
        )
        submit_and_wait(client, extract_jsonl_path, extract_results_path, poll_interval=poll_interval)

        structured, invalid_entries = load_structured_results(extract_results_path)
        structured_runs.append(structured)
        if invalid_entries:
            reason_counts = Counter(entry["reason"] for entry in invalid_entries)
            invalid_summaries.append(
                {
                    "run": run_index + 1,
                    "total_invalid": len(invalid_entries),
                    "reasons": dict(reason_counts),
                }
            )
        else:
            invalid_summaries.append({"run": run_index + 1, "total_invalid": 0, "reasons": {}})

        rows = parse_jsonl_file(str(extract_results_path), detection_locations=detection_locations)
        df = pd.DataFrame(rows)
        run_maps.append(dataframe_to_row_map(df))

        per_run_csv = paths.out_csv.with_name(f"{paths.out_csv.stem}_run{run_index + 1}.csv")
        per_run_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(per_run_csv, index=False)

    tolerance_map = dict(DEFAULT_NUMERIC_TOLERANCES)
    if tolerance_overrides:
        tolerance_map.update(tolerance_overrides)

    final_rows_map, disagreements = aggregate_runs(run_maps, tolerance_map, default_numeric_tolerance)

    report: Dict[str, Any] = {
        "run_count": repeat_total,
        "disagreements": disagreements,
    }

    flagged_graphs = {
        (entry["graph"]["SourceFile"], entry["graph"]["Page"], entry["graph"]["Section"])
        for entry in disagreements
    }
    flagged_graphs = {g for g in flagged_graphs if SECTION_KIND_MAP.get(g[2])}
    if flagged_graphs:
        report["flagged_graphs"] = [
            {"SourceFile": src, "Page": page, "Section": section, "kind": SECTION_KIND_MAP.get(section)}
            for (src, page, section) in sorted(flagged_graphs)
        ]
    if invalid_summaries:
        report["invalid_runs"] = invalid_summaries

    rerun_info: Dict[str, Any] = {}
    if flagged_graphs and rerun_on_disagreement:
        kinds_subset = sorted({SECTION_KIND_MAP[g[2]] for g in flagged_graphs if SECTION_KIND_MAP.get(g[2])})
        if kinds_subset:
            only_graphs: List[Tuple[str, int, str]] = []
            prompt_overrides: Dict[Tuple[str, int, str], str] = {}
            for source_file, page, section in flagged_graphs:
                kind = SECTION_KIND_MAP.get(section)
                if not kind:
                    continue
                payloads: List[Dict[str, Any]] = []
                for structured in structured_runs:
                    payload = structured.get(source_file, {}).get(page, {}).get(kind)
                    if payload:
                        payloads.append(payload)
                prompt_overrides[(source_file, page, kind)] = build_rerun_prompt(kind, payloads)
                only_graphs.append((source_file, page, kind))
            if only_graphs:
                rerun_suffix = "_rerun"
                rerun_jsonl = base_extract_jsonl.with_name(
                    f"{base_extract_jsonl.stem}{rerun_suffix}{base_extract_jsonl.suffix}"
                )
                rerun_results = base_extract_results.with_name(
                    f"{base_extract_results.stem}{rerun_suffix}{base_extract_results.suffix}"
                )
                build_extraction_jsonl_from_detect(
                    model,
                    detection_map,
                    pdf_lookup,
                    rerun_jsonl,
                    dpi=dpi,
                    max_crop_px=max_crop_px,
                    kinds=kinds_subset,
                    only_graphs=only_graphs,
                    prompt_overrides=prompt_overrides,
                    prefer_combined=False,
                )
                submit_and_wait(client, rerun_jsonl, rerun_results, poll_interval=poll_interval)

                structured_rerun, rerun_invalid = load_structured_results(rerun_results)
                structured_runs.append(structured_rerun)
                if rerun_invalid:
                    reason_counts = Counter(entry["reason"] for entry in rerun_invalid)
                    invalid_summaries.append(
                        {
                            "run": f"rerun",
                            "total_invalid": len(rerun_invalid),
                            "reasons": dict(reason_counts),
                        }
                    )

                rerun_rows = parse_jsonl_file(str(rerun_results), detection_locations=detection_locations)
                rerun_df = pd.DataFrame(rerun_rows)
                rerun_map = dataframe_to_row_map(rerun_df)
                for key, row in rerun_map.items():
                    final_rows_map[key] = row
                harmonize_locations(final_rows_map)

                rerun_csv = paths.out_csv.with_name(f"{paths.out_csv.stem}{rerun_suffix}.csv")
                rerun_csv.parent.mkdir(parents=True, exist_ok=True)
                rerun_df.to_csv(rerun_csv, index=False)

                rerun_info = {
                    "graphs": [
                        {
                            "SourceFile": src,
                            "Page": page,
                            "Section": section,
                            "kind": SECTION_KIND_MAP.get(section),
                        }
                        for (src, page, section) in sorted(flagged_graphs)
                    ],
                    "jsonl": str(rerun_jsonl),
                    "results": str(rerun_results),
                    "csv": str(rerun_csv),
                }

    if rerun_info:
        report["rerun"] = rerun_info

    final_rows = list(final_rows_map.values())

    # Append page 9 accumulations and forecast start date (parsed directly from PDF)
    for pdf_path in pdfs:
        try:
            extra_rows = extract_page9_rows(pdf_path)
        except Exception:
            extra_rows = []
        final_rows.extend(extra_rows)

    final_df = pd.DataFrame(final_rows)
    if not final_df.empty:
        final_df = final_df.sort_values(
            ["SourceFile", "Page", "Section", "HourIndex", "MeasurementType"]
        ).reset_index(drop=True)

    paths.out_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(paths.out_csv, index=False)
    print(f"Wrote {len(final_df)} rows to {paths.out_csv}")

    if paths.disagreement_report:
        paths.disagreement_report.parent.mkdir(parents=True, exist_ok=True)
        with open(paths.disagreement_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved disagreement report to {paths.disagreement_report}")

    return paths.out_csv


def main():
    parser = argparse.ArgumentParser(description="End-to-end batch pipeline: detection -> extraction -> CSV")
    parser.add_argument("--input", nargs="*", help="Optional explicit PDF paths")
    parser.add_argument("--input-dir", default="in", help="Directory to scan for PDFs when --input is not supplied")
    parser.add_argument("--output-dir", default=os.getenv("OUT_DIR", "out"), help="Directory for intermediate + final outputs")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-5"), help="OpenAI model to use")
    parser.add_argument("--dpi", type=int, default=150, help="PDF render DPI")
    parser.add_argument("--max_page_px", type=int, default=900, help="Max page side for detection requests")
    parser.add_argument("--max_crop_px", type=int, default=720, help="Max crop side for extraction requests")
    parser.add_argument("--poll_interval", type=int, default=20, help="Seconds between batch status polls")
    parser.add_argument("--out_csv", default=None, help="Override output CSV path (default: <output-dir>/batch_results.csv)")
    parser.add_argument("--detect_jsonl", default=None, help="Override detection JSONL path")
    parser.add_argument("--detect_results", default=None, help="Override detection results JSONL path")
    parser.add_argument("--extract_jsonl", default=None, help="Override extraction JSONL path")
    parser.add_argument("--extract_results", default=None, help="Override extraction results JSONL path")
    parser.add_argument("--extract_repeats", type=int, default=int(os.getenv("EXTRACT_REPEATS", "2")), help="Number of extraction repetitions to run for variance checking (default: 2)")
    parser.add_argument("--default_numeric_tolerance", type=float, default=float(os.getenv("DEFAULT_NUMERIC_TOLERANCE", "1.0")), help="Default tolerance used when comparing numeric values across runs")
    parser.add_argument(
        "--tolerance_override",
        action="append",
        default=[],
        help="Override numeric tolerance for a specific Section:MeasurementType, e.g. 'Wind:Speed=2.5'. May be repeated.",
    )
    parser.add_argument("--disable_rerun", action="store_true", help="Skip targeted re-run when runs disagree")
    parser.add_argument(
        "--disagreement_report",
        default=None,
        help="Path to write disagreement report JSON (default: <output-dir>/disagreement_report.json)",
    )
    parser.add_argument(
        "--no_disagreement_report",
        action="store_true",
        help="Disable writing the disagreement report JSON even if disagreements are detected.",
    )
    parser.add_argument(
        "--kinds",
        default=",".join(DEFAULT_KINDS),
        help="Comma-separated list of graph kinds to extract (default: wind,precipitation,temperature)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created input directory at {input_dir}; add PDFs there or pass --input.")

    pdfs = discover_pdfs(args.input, input_dir)
    print(f"Found {len(pdfs)} PDF(s):")
    for pdf in pdfs:
        try:
            display = pdf.relative_to(Path.cwd())
        except ValueError:
            display = pdf
        print(f" - {display}")

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]

    detect_jsonl = Path(args.detect_jsonl) if args.detect_jsonl else output_dir / "batch_detect.jsonl"
    detect_results = Path(args.detect_results) if args.detect_results else output_dir / "batch_detect_results.jsonl"
    extract_jsonl = Path(args.extract_jsonl) if args.extract_jsonl else output_dir / "batch_extract.jsonl"
    extract_results = Path(args.extract_results) if args.extract_results else output_dir / "batch_extract_results.jsonl"
    out_csv = Path(args.out_csv) if args.out_csv else output_dir / "batch_results.csv"

    if args.no_disagreement_report:
        disagreement_report = None
    else:
        disagreement_report = Path(args.disagreement_report).resolve() if args.disagreement_report else output_dir / "disagreement_report.json"

    tolerance_overrides: Dict[Tuple[str, str], float] = {}
    for override in args.tolerance_override:
        try:
            key_part, value_part = override.split("=", 1)
            section, measurement = key_part.split(":", 1)
            tolerance_overrides[(section.strip(), measurement.strip())] = float(value_part.strip())
        except ValueError as exc:
            raise SystemExit(f"Invalid --tolerance_override '{override}': {exc}")

    paths = BatchPaths(
        detect_jsonl=detect_jsonl,
        detect_results=detect_results,
        extract_jsonl=extract_jsonl,
        extract_results=extract_results,
        out_csv=out_csv,
        disagreement_report=disagreement_report,
    )

    client = OpenAI()
    run_pipeline(
        client=client,
        model=args.model,
        pdfs=pdfs,
        paths=paths,
        dpi=args.dpi,
        max_page_px=args.max_page_px,
        max_crop_px=args.max_crop_px,
        poll_interval=args.poll_interval,
        kinds=kinds or DEFAULT_KINDS,
        extraction_repeats=args.extract_repeats,
        default_numeric_tolerance=args.default_numeric_tolerance,
        tolerance_overrides=tolerance_overrides or None,
        rerun_on_disagreement=not args.disable_rerun,
    )


if __name__ == "__main__":
    main()

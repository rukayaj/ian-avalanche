import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image

import pandas as pd
from openai import OpenAI

from .batch_builder import (
    build_detect_regions_request,
    build_extract_all_request,
    build_extract_precip_request,
    build_extract_temperature_request,
    build_extract_wind_request,
    write_jsonl,
)
from .parse_batch_results import parse_detection_pages, parse_jsonl_file
from .pdf_render import render_pdf_to_images
from .utils import crop_normalized_box


DEFAULT_KINDS = ("wind", "precipitation", "temperature")


@dataclass(frozen=True)
class BatchPaths:
    detect_jsonl: Path
    detect_results: Path
    extract_jsonl: Path
    extract_results: Path
    out_csv: Path


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
) -> Path:
    kind_set = {k for k in kinds if k}
    if not kind_set:
        raise SystemExit("At least one graph kind must be specified for extraction.")
    items: List[Dict] = []
    rendered_cache: Dict[str, List[Image.Image]] = {}
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
            have_all_requested = {"wind", "precipitation", "temperature"}.issubset(kind_set)
            if have_all_requested and all(k in crops_by_kind for k in ("wind", "precipitation", "temperature")):
                req = build_extract_all_request(model, crops_by_kind, max_side=max_crop_px)
                req["custom_id"] = f"extract::{file_name}::p{page_num}::combined"
                items.append(req)
                continue
            if "wind" in crops_by_kind and "wind" in kind_set:
                req = build_extract_wind_request(model, crops_by_kind["wind"], max_side=max_crop_px)
                req["custom_id"] = f"extract::{file_name}::p{page_num}::wind"
                items.append(req)
            if "precipitation" in crops_by_kind and "precipitation" in kind_set:
                req = build_extract_precip_request(model, crops_by_kind["precipitation"], max_side=max_crop_px)
                req["custom_id"] = f"extract::{file_name}::p{page_num}::precipitation"
                items.append(req)
            if "temperature" in crops_by_kind and "temperature" in kind_set:
                req = build_extract_temperature_request(model, crops_by_kind["temperature"], max_side=max_crop_px)
                req["custom_id"] = f"extract::{file_name}::p{page_num}::temperature"
                items.append(req)
    write_jsonl(items, str(out_jsonl))
    print(f"Wrote {len(items)} extraction batch items -> {out_jsonl}")
    return out_jsonl


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
) -> Path:
    pdf_lookup = build_pdf_lookup(pdfs)
    build_detection_jsonl(model, pdfs, paths.detect_jsonl, dpi=dpi, max_page_px=max_page_px)
    submit_and_wait(client, paths.detect_jsonl, paths.detect_results, poll_interval=poll_interval)

    detection_map = parse_detection_pages([paths.detect_results])
    build_extraction_jsonl_from_detect(
        model,
        detection_map,
        pdf_lookup,
        paths.extract_jsonl,
        dpi=dpi,
        max_crop_px=max_crop_px,
        kinds=kinds,
    )
    submit_and_wait(client, paths.extract_jsonl, paths.extract_results, poll_interval=poll_interval)

    detection_locations = {
        (file_name, page_num): page_data.get("location") or ""
        for file_name, pages in detection_map.items()
        for page_num, page_data in pages.items()
        if page_data.get("location")
    } or None

    rows = parse_jsonl_file(str(paths.extract_results), detection_locations=detection_locations)
    df = pd.DataFrame(rows)
    paths.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.out_csv, index=False)
    print(f"Wrote {len(df)} rows to {paths.out_csv}")
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

    paths = BatchPaths(
        detect_jsonl=detect_jsonl,
        detect_results=detect_results,
        extract_jsonl=extract_jsonl,
        extract_results=extract_results,
        out_csv=out_csv,
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
    )


if __name__ == "__main__":
    main()

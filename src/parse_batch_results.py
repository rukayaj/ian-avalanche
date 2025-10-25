import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .extractors import series_to_rows


def extract_structured_output(resp: Dict) -> Optional[Dict]:
    # Prefer 'output_parsed' when present
    if "output_parsed" in resp and resp["output_parsed"] is not None:
        return resp["output_parsed"]
    # Else scan output list
    out = resp.get("output") or []
    for item in out:
        contents = item.get("content") or []
        for c in contents:
            if "parsed" in c and isinstance(c["parsed"], dict):
                return c["parsed"]
            if "json" in c and isinstance(c["json"], dict):
                return c["json"]
            # Try to parse text as JSON
            if c.get("type") in ("output_text", "text"):
                txt = c.get("text", "")
                try:
                    return json.loads(txt)
                except Exception:
                    continue
    return None


def parse_custom_id(custom_id: str) -> Dict:
    # Format: extract::<file name>::p{page}::{kind}
    parts = custom_id.split("::")
    data = {"kind": "", "file": "", "page": 0}
    if len(parts) >= 4 and parts[0] == "extract":
        data["file"] = parts[1]
        # page like p2
        p = parts[2]
        if p and p[0].lower() == "p":
            try:
                data["page"] = int(p[1:])
            except Exception:
                data["page"] = 0
        data["kind"] = parts[3]
    return data


def parse_detection_pages(jsonl_paths: Sequence[os.PathLike[str] | str]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    pages_by_file: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for path_like in jsonl_paths:
        path = Path(path_like)
        if not path.exists():
            print(f"Warning: detection results file not found: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Warning: could not parse detection JSONL line {line_number} in {path}: {exc}")
                    continue
                if obj.get("error"):
                    continue
                resp_wrapper = obj.get("response") or {}
                body = resp_wrapper.get("body") or {}
                parsed = extract_structured_output(body)
                if not parsed or parsed.get("page_type") != "area_graphs":
                    continue
                cid = obj.get("custom_id", "")
                parts = cid.split("::")
                if len(parts) < 3 or parts[0] != "detect":
                    continue
                file_name = parts[1]
                page_token = parts[2]
                page_num = 0
                if page_token.lower().startswith("p"):
                    try:
                        page_num = int(page_token[1:])
                    except ValueError:
                        page_num = 0
                if page_num <= 0:
                    continue
                pages_by_file.setdefault(file_name, {})[page_num] = {
                    "location": parsed.get("location") or "",
                    "graphs": parsed.get("graphs") or [],
                }
    return pages_by_file


def parse_jsonl_file(path: str, detection_locations: Optional[Dict[Tuple[str, int], str]] = None) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("error"):
                continue
            resp_wrapper = obj.get("response") or {}
            body = resp_wrapper.get("body") or {}
            parsed = extract_structured_output(body)
            if not parsed:
                continue
            custom_id = obj.get("custom_id", "")
            meta = parse_custom_id(custom_id)
            file_name = meta.get("file") or ""
            page_index = max(0, meta.get("page", 1) - 1)
            detection_location = ""
            if detection_locations:
                detection_location = detection_locations.get((file_name, page_index + 1), "") or ""
            location = parsed.get("location") or detection_location
            kind = meta.get("kind") or ""
            # Support both per-graph and combined outputs
            if kind == "combined" and all(k in (parsed or {}) for k in ("wind", "precipitation", "temperature")):
                for sub_kind in ("wind", "precipitation", "temperature"):
                    sub_payload = parsed.get(sub_kind) or {}
                    loc = (
                        location
                        or sub_payload.get("location")
                        or detection_location
                        or parsed.get(sub_kind, {}).get("location")
                        or ""
                    )
                    rows.extend(series_to_rows(file_name, page_index, loc, sub_kind, sub_payload))
                continue
            if kind not in ("wind", "precipitation", "temperature"):
                continue
            rows.extend(series_to_rows(file_name, page_index, location, kind, parsed))
    return rows


def _fill_missing_locations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Location" not in df.columns:
        return df
    # For each (SourceFile, Page), if Location is empty for some rows but present for others, fill with the most common non-empty string.
    def fill_group(g: pd.DataFrame) -> pd.DataFrame:
        non_empty = [s for s in g["Location"].astype(str).tolist() if s.strip()]
        repl = None
        if non_empty:
            # Choose the most frequent
            repl = pd.Series(non_empty).mode().iloc[0]
        if repl:
            g.loc[g["Location"].astype(str).str.strip() == "", "Location"] = repl
        return g
    return df.groupby(["SourceFile", "Page"], group_keys=False).apply(fill_group)


def main():
    parser = argparse.ArgumentParser(description="Parse and merge OpenAI Batch results JSONLs into a single CSV of rows")
    parser.add_argument("--input_jsonl", nargs="+", help="One or more Batch results JSONL files")
    parser.add_argument("--out_csv", default="out/batch_results.csv")
    parser.add_argument(
        "--detect_results",
        nargs="*",
        help="Optional detection results JSONL files for location fallback",
    )
    args = parser.parse_args()

    all_rows: List[Dict] = []
    detection_locations: Optional[Dict[Tuple[str, int], str]] = None
    if args.detect_results:
        detection_map = parse_detection_pages(args.detect_results)
        detection_locations = {
            (file, page): page_data.get("location") or ""
            for file, pages in detection_map.items()
            for page, page_data in pages.items()
            if page_data.get("location")
        } or None

    for path in args.input_jsonl:
        if not os.path.exists(path):
            print(f"Warning: missing {path}")
            continue
        rows = parse_jsonl_file(path, detection_locations=detection_locations)
        all_rows.extend(rows)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df = _fill_missing_locations(df)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(all_rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()

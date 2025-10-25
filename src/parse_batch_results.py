import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .extractors import series_to_rows


def _extract_parsed_from_response(resp: Dict) -> Optional[Dict]:
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


def parse_jsonl_file(path: str) -> List[Dict]:
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
            parsed = _extract_parsed_from_response(body)
            if not parsed:
                continue
            custom_id = obj.get("custom_id", "")
            meta = parse_custom_id(custom_id)
            location = parsed.get("location") or ""
            page_index = max(0, meta.get("page", 1) - 1)
            kind = meta.get("kind") or ""
            # Support both per-graph and combined outputs
            if kind == "combined" and all(k in (parsed or {}) for k in ("wind", "precipitation", "temperature")):
                for sub_kind in ("wind", "precipitation", "temperature"):
                    loc = location or parsed.get(sub_kind, {}).get("location") or ""
                    sub_payload = parsed.get(sub_kind) or {}
                    src_file = meta.get("file") or ""
                    rows.extend(series_to_rows(src_file, page_index, loc, sub_kind, sub_payload))
                continue
            if kind not in ("wind", "precipitation", "temperature"):
                continue
            src_file = meta.get("file") or ""
            rows.extend(series_to_rows(src_file, page_index, location, kind, parsed))
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
    args = parser.parse_args()

    all_rows: List[Dict] = []
    for path in args.input_jsonl:
        if not os.path.exists(path):
            print(f"Warning: missing {path}")
            continue
        rows = parse_jsonl_file(path)
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

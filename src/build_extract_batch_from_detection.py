import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from PIL import Image

from .pdf_render import render_pdf_to_images
from .utils import crop_normalized_box
from .batch_builder import (
    build_extract_wind_request,
    build_extract_precip_request,
    build_extract_temperature_request,
    write_jsonl,
)


def parse_custom_id(custom_id: str) -> Dict:
    # detect::<file>::p{page}
    parts = custom_id.split("::")
    data = {"file": "", "page": 0}
    if len(parts) >= 3 and parts[0] == "detect":
        data["file"] = parts[1]
        p = parts[2]
        if p and p[0].lower() == "p":
            try:
                data["page"] = int(p[1:])
            except Exception:
                pass
    return data


def main():
    parser = argparse.ArgumentParser(description="Build extraction batch JSONL from detection batch results JSONL")
    parser.add_argument("--detect_results", nargs="+", help="Detection results JSONL file(s)")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-5"))
    parser.add_argument("--out_jsonl", default="out/batch_extract_from_detection.jsonl")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max_crop_px", type=int, default=720)
    parser.add_argument("--kinds", type=str, default="wind,precipitation,temperature")
    args = parser.parse_args()

    kinds_set = {k.strip() for k in args.kinds.split(",") if k.strip()}

    # Build a mapping from file -> pages -> graphs
    pages_by_file: Dict[str, Dict[int, List[Dict]]] = {}
    for path in args.detect_results:
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
                resp = resp_wrapper.get("body") or {}
                # parsed detection result
                parsed = None
                if "output_parsed" in resp and resp["output_parsed"] is not None:
                    parsed = resp["output_parsed"]
                else:
                    outs = resp.get("output") or []
                    for o in outs:
                        for c in o.get("content") or []:
                            if "parsed" in c and isinstance(c["parsed"], dict):
                                parsed = c["parsed"]
                                break
                            if c.get("type") in ("output_text", "text"):
                                try:
                                    parsed = json.loads(c.get("text", ""))
                                except Exception:
                                    pass
                        if parsed:
                            break
                if not parsed:
                    continue
                cid = obj.get("custom_id", "")
                meta = parse_custom_id(cid)
                file = meta.get("file")
                page = meta.get("page")
                if not file or not page:
                    continue
                if parsed.get("page_type") != "area_graphs":
                    continue
                pages_by_file.setdefault(file, {})[page] = parsed.get("graphs", [])

    items: List[Dict] = []
    # For each file-page, render and crop using detected bboxes, then create per-graph requests
    for file, pages in pages_by_file.items():
        # Resolve file path in current directory
        candidates = list(Path(".").glob(file)) or list(Path(".").glob(f"**/{file}"))
        if not candidates:
            print(f"Warning: cannot find PDF for {file}")
            continue
        pdf_path = str(candidates[0])
        all_pages = render_pdf_to_images(pdf_path, dpi=args.dpi)
        for pnum, graphs in pages.items():
            if pnum < 1 or pnum > len(all_pages):
                continue
            img = all_pages[pnum - 1]
            by_kind: Dict[str, Image.Image] = {}
            for g in graphs:
                kind = g.get("kind")
                if kind not in kinds_set:
                    continue
                bbox = g.get("bbox")
                crop = crop_normalized_box(img, tuple(bbox))
                by_kind[kind] = crop
            if "wind" in by_kind:
                req = build_extract_wind_request(args.model, by_kind["wind"], max_side=args.max_crop_px)
                req["custom_id"] = f"extract::{file}::p{pnum}::wind"
                items.append(req)
            if "precipitation" in by_kind:
                req = build_extract_precip_request(args.model, by_kind["precipitation"], max_side=args.max_crop_px)
                req["custom_id"] = f"extract::{file}::p{pnum}::precipitation"
                items.append(req)
            if "temperature" in by_kind:
                req = build_extract_temperature_request(args.model, by_kind["temperature"], max_side=args.max_crop_px)
                req["custom_id"] = f"extract::{file}::p{pnum}::temperature"
                items.append(req)

    out_path = args.out_jsonl
    write_jsonl(items, out_path)
    print(f"Wrote {len(items)} batch items to {out_path}")


if __name__ == "__main__":
    main()

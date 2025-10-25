import argparse
import os
from pathlib import Path
from typing import Dict

from PIL import Image

from .batch_builder import (
    build_extract_precip_request,
    build_extract_temperature_request,
    build_extract_wind_request,
    write_jsonl,
)
from .local_detect import detect_graphs_on_page_local
from .pdf_render import render_pdf_to_images
from .run_batch_pipeline import discover_pdfs
from .utils import crop_normalized_box


def main():
    parser = argparse.ArgumentParser(description="Build OpenAI Batch JSONL for per-graph extraction using local detection")
    parser.add_argument("--input", nargs="*", help="Input PDF files (defaults to --input-dir)")
    parser.add_argument("--input-dir", default=".", help="Directory to scan for PDFs when --input is omitted")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-5"))
    parser.add_argument("--out_jsonl", default="out/batch_extract.jsonl")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max_page_px", type=int, default=900)
    parser.add_argument("--max_crop_px", type=int, default=720)
    parser.add_argument("--kinds", type=str, default="wind,precipitation,temperature")
    parser.add_argument("--mode", choices=["per-graph", "combined"], default="per-graph", help="Extract per graph or as a single combined request per page")
    args = parser.parse_args()

    pdfs = discover_pdfs(args.input, Path(args.input_dir).resolve())
    items = []
    kinds_set = {k.strip() for k in args.kinds.split(",") if k.strip()}

    for pdf in pdfs:
        pages = render_pdf_to_images(str(pdf), dpi=args.dpi)
        total = len(pages)
        for i, img in enumerate(pages, start=1):
            if i in (1, total):
                continue
            regions = detect_graphs_on_page_local(img)
            if regions.get("page_type") != "area_graphs":
                continue
            graphs = regions.get("graphs", [])
            by_kind: Dict[str, Image.Image] = {}
            for g in graphs:
                k = g.get("kind")
                if k not in kinds_set:
                    continue
                crop = crop_normalized_box(img, tuple(g.get("bbox")))
                by_kind[k] = crop

            if args.mode == "combined":
                # Only create a combined request when we have all three crops
                if all(k in by_kind for k in ("wind", "precipitation", "temperature")):
                    from .batch_builder import build_extract_all_request
                    req = build_extract_all_request(args.model, by_kind, max_side=args.max_crop_px)
                    req["custom_id"] = f"extract::{pdf.name}::p{i}::combined"
                    items.append(req)
                continue

            # Per-graph mode
            if "wind" in by_kind:
                req = build_extract_wind_request(args.model, by_kind["wind"], max_side=args.max_crop_px)
                req["custom_id"] = f"extract::{pdf.name}::p{i}::wind"
                items.append(req)
            if "precipitation" in by_kind:
                req = build_extract_precip_request(args.model, by_kind["precipitation"], max_side=args.max_crop_px)
                req["custom_id"] = f"extract::{pdf.name}::p{i}::precipitation"
                items.append(req)
            if "temperature" in by_kind:
                req = build_extract_temperature_request(args.model, by_kind["temperature"], max_side=args.max_crop_px)
                req["custom_id"] = f"extract::{pdf.name}::p{i}::temperature"
                items.append(req)

    out_jsonl = Path(args.out_jsonl)
    write_jsonl(items, str(out_jsonl))
    print(f"Wrote {len(items)} batch items to {out_jsonl}")


if __name__ == "__main__":
    main()

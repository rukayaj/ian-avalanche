import argparse
import os
from pathlib import Path
from typing import List

from PIL import Image

from .pdf_render import render_pdf_to_images
from .batch_builder import build_detect_regions_request, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Build OpenAI Batch JSONL for region detection")
    parser.add_argument("--input", nargs="*", help="Input PDF files")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-5"))
    parser.add_argument("--out_jsonl", default="out/batch_detect.jsonl")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max_page_px", type=int, default=900)
    args = parser.parse_args()

    files = args.input or [str(p) for p in Path(".").glob("*.pdf")]
    items = []
    for pdf in files:
        pages = render_pdf_to_images(pdf, dpi=args.dpi)
        total = len(pages)
        for i, img in enumerate(pages, start=1):
            # skip page 1 and last
            if i in (1, total):
                continue
            req = build_detect_regions_request(args.model, img, max_side=args.max_page_px)
            # Encode file and page in custom_id for mapping later
            from pathlib import Path as _P
            req["custom_id"] = f"detect::{_P(pdf).name}::p{i}"
            items.append(req)

    write_jsonl(items, args.out_jsonl)
    print(f"Wrote {len(items)} batch items to {args.out_jsonl}")


if __name__ == "__main__":
    main()

import argparse
import os
from pathlib import Path

from .run_batch_pipeline import (
    build_extraction_jsonl_from_detect,
    build_pdf_lookup,
    discover_pdfs,
)
from .parse_batch_results import parse_detection_pages


def main():
    parser = argparse.ArgumentParser(description="Build extraction batch JSONL from detection batch results JSONL")
    parser.add_argument("--detect_results", nargs="+", required=True, help="Detection results JSONL file(s)")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-5"), help="OpenAI model name")
    parser.add_argument("--out_jsonl", default="out/batch_extract_from_detection.jsonl", help="Output extraction JSONL path")
    parser.add_argument("--pdf", nargs="*", help="Optional explicit PDF files referenced by detection results")
    parser.add_argument("--pdf-dir", default=".", help="Directory to scan for PDFs when --pdf is omitted")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for re-rendering PDFs before cropping")
    parser.add_argument("--max_crop_px", type=int, default=720, help="Max crop side length for extraction requests")
    parser.add_argument("--kinds", type=str, default="wind,precipitation,temperature", help="Comma-separated kinds to include")
    args = parser.parse_args()

    detect_paths = [Path(p).resolve() for p in args.detect_results]
    pdfs = discover_pdfs(args.pdf, Path(args.pdf_dir).resolve())
    pdf_lookup = build_pdf_lookup(pdfs)
    detection_map = parse_detection_pages(detect_paths)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()] or ["wind", "precipitation", "temperature"]
    out_jsonl = Path(args.out_jsonl)
    build_extraction_jsonl_from_detect(
        args.model,
        detection_map,
        pdf_lookup,
        out_jsonl,
        dpi=args.dpi,
        max_crop_px=args.max_crop_px,
        kinds=kinds,
    )


if __name__ == "__main__":
    main()

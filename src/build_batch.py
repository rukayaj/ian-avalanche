import argparse
import os
from pathlib import Path

from .run_batch_pipeline import build_detection_jsonl, discover_pdfs


def main():
    parser = argparse.ArgumentParser(description="Build OpenAI Batch JSONL for region detection")
    parser.add_argument("--input", nargs="*", help="Input PDF files (defaults to --input-dir)")
    parser.add_argument("--input-dir", default=".", help="Directory to scan for PDFs when --input is omitted")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-5"), help="OpenAI model name")
    parser.add_argument("--out_jsonl", default="out/batch_detect.jsonl", help="Output JSONL path")
    parser.add_argument("--dpi", type=int, default=200, help="PDF render DPI")
    parser.add_argument("--max_page_px", type=int, default=900, help="Max page side in pixels when sending to the model")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    pdfs = discover_pdfs(args.input, input_dir)
    out_jsonl = Path(args.out_jsonl)
    build_detection_jsonl(args.model, pdfs, out_jsonl, dpi=args.dpi, max_page_px=args.max_page_px)


if __name__ == "__main__":
    main()

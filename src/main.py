import argparse
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from .pdf_render import render_pdf_to_images
from .extractors import (
    detect_graphs_on_page,
    extract_graph_series,
    series_to_rows,
    rows_to_dataframe,
)
from .llm_client import LLMClient
from .local_detect import detect_graphs_on_page_local
from .utils import crop_normalized_box, ensure_rgb, pil_to_data_url
from .metrics import time_block, log_event


def process_pdf(
    pdf_path: str,
    out_dir: str,
    model: str,
    dpi: int = 300,
    only_pages: List[int] | None = None,
    kinds_filter: List[str] | None = None,
    max_page_px: int | None = 1400,
    max_crop_px: int | None = 1000,
    detect_method: str = "local",
) -> pd.DataFrame:
    client = LLMClient(model=model)
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    pages = render_pdf_to_images(pdf_path, dpi=dpi, metrics_path=metrics_path)
    all_rows: List[Dict] = []

    # Heuristic for this document structure: pages 2..N-1 are area pages
    # but we still detect page_type via LLM to be safe.
    selected_pages = set(only_pages) if only_pages else None
    kinds_set = set(kinds_filter) if kinds_filter else None
    total_pages = len(pages)
    for i, img in enumerate(tqdm(pages, desc=f"Pages {os.path.basename(pdf_path)}")):
        # Skip first/last by default (front page + back page not needed)
        if not selected_pages and (i + 1 in (1, total_pages)):
            continue
        if selected_pages and (i + 1) not in selected_pages:
            continue
        # Skip page 1 and last if you want; but ask the model to confirm
        with time_block("page_processing", metrics_path, file=pdf_path, page=i + 1):
            # detect method is passed via closure variable 'detect_method' set by main()
            if detect_method == "local":
                regions = detect_graphs_on_page_local(img)
            else:
                regions = detect_graphs_on_page(
                    img, client, desired_kinds=kinds_set, metrics_path=metrics_path, page_num=i + 1, source_file=pdf_path, max_page_px=max_page_px
                )
        if regions.get("page_type") != "area_graphs":
            continue

        location = regions.get("location") or ""
        graphs = regions.get("graphs", [])
        # Optionally save debug with crops
        page_out = Path(out_dir) / f"{Path(pdf_path).stem}_page_{i+1}"
        page_out.mkdir(parents=True, exist_ok=True)

        # For best per-graph accuracy, use individual per-graph extraction calls (slower but higher fidelity)
        for g in graphs:
            kind = g.get("kind")
            if kinds_set and kind not in kinds_set:
                continue
            bbox = g.get("bbox")
            with time_block("crop_graph", metrics_path, file=pdf_path, page=i + 1, kind=kind):
                crop = crop_normalized_box(img, tuple(bbox))
            crop.save(page_out / f"crop_{kind}.png")
            payload, meta = extract_graph_series(
                kind, crop, client, metrics_path=metrics_path, page_num=i + 1, source_file=pdf_path, max_crop_px=max_crop_px
            )
            if payload.get("location"):
                location = payload["location"]
            rows = series_to_rows(pdf_path, i, location, kind, payload)
            all_rows.extend(rows)
            log_event(metrics_path, {"type": "rows_added", "file": pdf_path, "page": i + 1, "kind": kind, "rows": len(rows), "retry": bool(meta.get("retry"))})

    return rows_to_dataframe(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Extract graphs from avalanche forecast PDFs to CSV")
    parser.add_argument("--input", nargs="*", help="Input PDF files (default: all PDFs in CWD)")
    parser.add_argument("--out_csv", default="out/extracted.csv", help="Output CSV path")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-5"), help="OpenAI model")
    parser.add_argument("--out_dir", default="out", help="Output directory for crops/debug")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default: 300)")
    parser.add_argument("--max_page_px", type=int, default=1400, help="Max page image side for detection (default: 1400)")
    parser.add_argument("--max_crop_px", type=int, default=1000, help="Max crop image side for extraction (default: 1000)")
    parser.add_argument("--detect", choices=["local", "llm"], default="llm", help="Region detection method (default: llm)")
    parser.add_argument(
        "--pages",
        type=str,
        default="",
        help="Comma-separated 1-based page numbers to process (e.g., '2' or '2,3,4')",
    )
    parser.add_argument(
        "--kinds",
        type=str,
        default="",
        help="Comma-separated kinds to process: wind,precipitation,temperature",
    )
    args = parser.parse_args()

    inputs = args.input
    if not inputs:
        inputs = [str(p) for p in Path(".").glob("*.pdf")]

    frames: List[pd.DataFrame] = []
    # Parse page and kind filters
    only_pages: List[int] | None = None
    if args.pages:
        pages_list: List[int] = []
        for tok in args.pages.split(","):
            tok = tok.strip()
            if tok:
                try:
                    pages_list.append(int(tok))
                except ValueError:
                    pass
        if pages_list:
            only_pages = pages_list

    kinds_filter: List[str] | None = None
    if args.kinds:
        kinds_list = [k.strip() for k in args.kinds.split(",") if k.strip()]
        if kinds_list:
            kinds_filter = kinds_list

    for pdf in inputs:
        if not pdf.lower().endswith(".pdf"):
            continue
        df = process_pdf(
            pdf,
            args.out_dir,
            args.model,
            dpi=args.dpi,
            only_pages=only_pages,
            kinds_filter=kinds_filter,
            max_page_px=args.max_page_px,
            max_crop_px=args.max_crop_px,
            detect_method=args.detect,
        )
        frames.append(df)
        # Write per-PDF CSV chunk for long runs
        stem = Path(pdf).stem
        per_pdf_csv = Path(args.out_dir) / f"{stem}_extracted.csv"
        df.to_csv(per_pdf_csv, index=False)
        print(f"Wrote per-PDF rows: {len(df)} to {per_pdf_csv}")

    if frames:
        out_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    else:
        out_df = pd.DataFrame()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out_csv}")
    # Print metrics summary
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    try:
        from .metrics import summarize_metrics

        print(summarize_metrics(metrics_path))
    except Exception as e:
        print(f"Metrics summary unavailable: {e}")


if __name__ == "__main__":
    main()

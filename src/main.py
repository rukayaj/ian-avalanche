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
from .utils import crop_normalized_box, ensure_rgb, pil_to_data_url


def process_pdf(
    pdf_path: str,
    out_dir: str,
    model: str,
    dpi: int = 300,
    only_pages: List[int] | None = None,
    kinds_filter: List[str] | None = None,
) -> pd.DataFrame:
    client = LLMClient(model=model)
    os.makedirs(out_dir, exist_ok=True)
    pages = render_pdf_to_images(pdf_path, dpi=dpi)
    all_rows: List[Dict] = []

    # Heuristic for this document structure: pages 2..N-1 are area pages
    # but we still detect page_type via LLM to be safe.
    selected_pages = set(only_pages) if only_pages else None
    kinds_set = set(kinds_filter) if kinds_filter else None
    for i, img in enumerate(tqdm(pages, desc=f"Pages {os.path.basename(pdf_path)}")):
        if selected_pages and (i + 1) not in selected_pages:
            continue
        # Skip page 1 and last if you want; but ask the model to confirm
        regions = detect_graphs_on_page(img, client, desired_kinds=kinds_set)
        if regions.get("page_type") != "area_graphs":
            continue

        location = regions.get("location") or ""
        graphs = regions.get("graphs", [])
        # Optionally save debug with crops
        page_out = Path(out_dir) / f"{Path(pdf_path).stem}_page_{i+1}"
        page_out.mkdir(parents=True, exist_ok=True)

        # process each detected graph
        for g in graphs:
            kind = g.get("kind")
            if kinds_set and kind not in kinds_set:
                continue
            bbox = g.get("bbox")
            crop = crop_normalized_box(img, tuple(bbox))
            # Save crop for debugging
            crop.save(page_out / f"crop_{kind}.png")
            payload = extract_graph_series(kind, crop, client)
            # if location comes through in payload, prefer it
            if payload.get("location"):
                location = payload["location"]
            rows = series_to_rows(pdf_path, i, location, kind, payload)
            all_rows.extend(rows)

    return rows_to_dataframe(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Extract graphs from avalanche forecast PDFs to CSV")
    parser.add_argument("--input", nargs="*", help="Input PDF files (default: all PDFs in CWD)")
    parser.add_argument("--out_csv", default="out/extracted.csv", help="Output CSV path")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-5"), help="OpenAI model")
    parser.add_argument("--out_dir", default="out", help="Output directory for crops/debug")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default: 300)")
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


if __name__ == "__main__":
    main()

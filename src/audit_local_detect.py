import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

from PIL import Image

from .pdf_render import render_pdf_to_images
from .local_detect import detect_graphs_on_page_local
from .overlay import draw_bboxes


def is_valid_graphs(graphs: List[Dict]) -> bool:
    if len(graphs) != 3:
        return False
    # vertical order and plausible widths/heights
    ys = [g["bbox"][1] for g in graphs]
    if not (ys[0] <= ys[1] <= ys[2]):
        # allow small disorder; sort and keep
        graphs.sort(key=lambda g: g["bbox"][1])
    for g in graphs:
        x, y, w, h = g["bbox"]
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            return False
        if not (0.3 <= w <= 0.95):
            return False
        if not (0.15 <= h <= 0.6):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Audit local region detection across PDFs and save overlays + CSV")
    parser.add_argument("--input", nargs="*", help="Input PDF files (default: all PDFs in CWD)")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out_dir", default="out/local_detect_audit")
    args = parser.parse_args()

    files = args.input or [str(p) for p in Path(".").glob("*.pdf")]
    os.makedirs(args.out_dir, exist_ok=True)
    summary_rows: List[List] = []

    for pdf in files:
        pages = render_pdf_to_images(pdf, dpi=args.dpi)
        total = len(pages)
        for i, img in enumerate(pages, start=1):
            # we expect pages 2..N-1 to be area pages
            expected_area = (i not in (1, total))
            res = detect_graphs_on_page_local(img)
            page_type = res.get("page_type")
            graphs = res.get("graphs", [])
            ok = (page_type == "area_graphs") if expected_area else (page_type != "area_graphs")
            # Overlay for inspection
            overlay_path = os.path.join(args.out_dir, f"{Path(pdf).stem}_p{i}.png")
            try:
                if graphs:
                    draw_bboxes(img, graphs, overlay_path)
            except Exception:
                pass
            valid_graphs = is_valid_graphs(graphs) if expected_area else page_type != "area_graphs"
            summary_rows.append([pdf, i, page_type, ok, valid_graphs, len(graphs)])

    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "page", "page_type", "ok_expected", "valid_graphs_shape", "num_graphs"])
        w.writerows(summary_rows)
    print(f"Wrote audit to {csv_path}; overlays in {args.out_dir}")


if __name__ == "__main__":
    main()


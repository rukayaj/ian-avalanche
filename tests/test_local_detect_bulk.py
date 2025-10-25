import os
from pathlib import Path
import pytest

from src.pdf_render import render_pdf_to_images
from src.local_detect import detect_graphs_on_page_local


@pytest.mark.slow
def test_local_detect_all_pdfs_middle_pages():
    # Limit workload with env vars if desired
    max_pdfs = int(os.getenv("MAX_PDFS", "999"))
    dpi = int(os.getenv("TEST_DPI", "150"))

    pdf_root = Path(os.getenv("PDF_DIR", "in"))
    if not pdf_root.exists():
        pdf_root = Path(".")
    pdfs = list(pdf_root.rglob("*.pdf"))[:max_pdfs]
    assert pdfs, f"No PDFs found for testing in {pdf_root}"
    total_checked = 0
    total_ok = 0
    for pdf in pdfs:
        pages = render_pdf_to_images(str(pdf), dpi=dpi)
        assert len(pages) >= 3, f"Expected at least 3 pages in {pdf}"
        for i in range(1, len(pages)-1):  # 1-based pages 2..N-1
            res = detect_graphs_on_page_local(pages[i])
            total_checked += 1
            if res.get('page_type') != 'area_graphs':
                continue
            graphs = res.get('graphs', [])
            if len(graphs) != 3:
                continue
            # basic bbox sanity
            ok = True
            ys = [g['bbox'][1] for g in graphs]
            if not (ys[0] <= ys[1] <= ys[2]):
                ok = False
            for g in graphs:
                x, y, w, h = g['bbox']
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    ok = False
                if not (0.3 <= w <= 0.95 and 0.12 <= h <= 0.65):
                    ok = False
            if ok:
                total_ok += 1
    # We expect strong coverage; allow occasional misses
    assert total_ok / max(1, total_checked) > 0.8, f"Local detection success ratio too low: {total_ok}/{total_checked}"

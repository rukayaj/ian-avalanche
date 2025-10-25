import pytest
from pathlib import Path

from PIL import Image

from src.pdf_render import render_pdf_to_images
from src.local_detect import detect_graphs_on_page_local


def test_local_detect_returns_three_graphs_for_page_two():
    pdf = next(Path('.').glob('Scottish Avalanche Information Service-5.pdf'))
    pages = render_pdf_to_images(str(pdf), dpi=150)
    img = pages[1]  # page 2 (0-based index)
    out = detect_graphs_on_page_local(img)
    assert out.get('page_type') == 'area_graphs'
    graphs = out.get('graphs', [])
    kinds = {g.get('kind') for g in graphs}
    assert kinds == {'wind', 'precipitation', 'temperature'}
    for g in graphs:
        x, y, w, h = g['bbox']
        assert 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1

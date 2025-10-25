from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image
from .metrics import log_event, time_block


def render_pdf_to_images(path: str, dpi: int = 350, metrics_path: Optional[str] = None) -> List[Image.Image]:
    """Render all pages of a PDF to PIL Images at the given DPI."""
    images: List[Image.Image] = []
    with time_block("render_pdf", metrics_path, file=path, dpi=dpi):
        doc = fitz.open(path)
        try:
            for idx, page in enumerate(doc, start=1):
                with time_block("render_page", metrics_path, file=path, dpi=dpi, page=idx):
                    zoom = dpi / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    mode = "RGB" if pix.alpha == 0 else "RGBA"
                    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    images.append(img)
        finally:
            doc.close()
    log_event(metrics_path, {"type": "render_summary", "file": path, "pages": len(images), "dpi": dpi})
    return images

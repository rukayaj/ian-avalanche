from typing import List

import fitz  # PyMuPDF
from PIL import Image


def render_pdf_to_images(path: str, dpi: int = 350) -> List[Image.Image]:
    """Render all pages of a PDF to PIL Images at the given DPI."""
    images: List[Image.Image] = []
    doc = fitz.open(path)
    try:
        for page in doc:
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
    return images


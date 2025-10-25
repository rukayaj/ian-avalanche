import base64
from io import BytesIO
from typing import Tuple, Optional

from PIL import Image


def pil_to_data_url(img: Image.Image, format: str = "PNG", quality: Optional[int] = None) -> str:
    buf = BytesIO()
    save_kwargs = {}
    if format.upper() == "JPEG":
        # Ensure no alpha for JPEG
        if img.mode in ("RGBA", "LA"):
            img = img.convert("RGB")
        if quality is not None:
            save_kwargs["quality"] = max(1, min(quality, 95))
        save_kwargs["optimize"] = True
        save_kwargs["progressive"] = True
    img.save(buf, format=format, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{format.lower()};base64,{b64}"


def crop_normalized_box(img: Image.Image, box: Tuple[float, float, float, float]) -> Image.Image:
    """
    Crop using normalized bbox: (x, y, w, h) in [0,1] relative to image size.
    Returns a new PIL Image.
    """
    W, H = img.size
    x, y, w, h = box
    left = max(0, min(W, int(round(x * W))))
    top = max(0, min(H, int(round(y * H))))
    right = max(0, min(W, int(round((x + w) * W))))
    bottom = max(0, min(H, int(round((y + h) * H))))
    if right <= left or bottom <= top:
        # Fall back to whole image if bbox is invalid
        return img.copy()
    return img.crop((left, top, right, bottom))


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    """Resize image so that the larger side is <= max_side, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    if w >= h:
        new_w = max_side
        new_h = int(h * (max_side / w))
    else:
        new_h = max_side
        new_w = int(w * (max_side / h))
    return img.resize((new_w, new_h), Image.LANCZOS)

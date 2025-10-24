import base64
from io import BytesIO
from typing import Tuple

from PIL import Image


def pil_to_data_url(img: Image.Image, format: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=format)
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


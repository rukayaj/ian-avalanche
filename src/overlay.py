from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "wind": (30, 144, 255),          # dodgerblue
    "precipitation": (70, 130, 180), # steelblue
    "temperature": (255, 99, 71),    # tomato
}


def draw_bboxes(img: Image.Image, graphs: List[Dict], out_path: str) -> None:
    im = img.convert("RGB").copy()
    W, H = im.size
    draw = ImageDraw.Draw(im)
    for g in graphs:
        kind = g.get("kind", "")
        color = COLORS.get(kind, (255, 215, 0))
        x, y, w, h = g.get("bbox", [0, 0, 1, 1])
        L = int(x * W)
        T = int(y * H)
        R = int((x + w) * W)
        B = int((y + h) * H)
        draw.rectangle([L, T, R, B], outline=color, width=4)
        label = kind or "graph"
        draw.text((L + 6, T + 6), label, fill=color)
    im.save(out_path)


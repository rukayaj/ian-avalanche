from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def _to_gray_np(img: Image.Image, max_side: int = 1400) -> Tuple[np.ndarray, float, float]:
    im = img.convert("L")
    w, h = im.size
    scale = 1.0
    if max(w, h) > max_side:
        if w >= h:
            new_w = max_side
            new_h = int(h * (max_side / w))
        else:
            new_h = max_side
            new_w = int(w * (max_side / h))
        im = im.resize((new_w, new_h), Image.BILINEAR)
        scale = new_w / w
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr, im.size[0] / img.size[0], im.size[1] / img.size[1]


def _vertical_profile(arr: np.ndarray) -> np.ndarray:
    # Edge-like measure: sum abs vertical and horizontal diffs
    dv = np.abs(np.diff(arr, axis=0, prepend=arr[[0], :]))
    dh = np.abs(np.diff(arr, axis=1, prepend=arr[:, [0]]))
    e = dv + dh
    prof = e.mean(axis=1)
    # Smooth with simple moving average
    k = 15
    if arr.shape[0] > k:
        kernel = np.ones(k) / k
        prof = np.convolve(prof, kernel, mode="same")
    return prof


def _find_segments(prof: np.ndarray, min_seg_h: int) -> List[Tuple[int, int]]:
    n = len(prof)
    def high_mask(pctl: float) -> List[Tuple[int, int]]:
        thr = np.percentile(prof, pctl)
        high = prof > thr
        segs: List[Tuple[int, int]] = []
        in_seg = False
        start = 0
        for i, v in enumerate(high):
            if v and not in_seg:
                in_seg = True
                start = i
            elif not v and in_seg:
                if i - start >= min_seg_h:
                    segs.append((start, i))
                in_seg = False
        if in_seg and n - start >= min_seg_h:
            segs.append((start, n))
        return segs

    # Try a few thresholds to get at least 3 high-activity bands
    for p in (70, 65, 60, 55, 50, 45, 40):
        segs = high_mask(p)
        if len(segs) >= 3:
            segs.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
            return segs[:3]

    # Fallback: split by low-activity separators
    thr_low = np.percentile(prof, 35)
    low = prof < thr_low
    gaps: List[Tuple[int, int]] = []
    in_gap = False
    start = 0
    for i, v in enumerate(low):
        if v and not in_gap:
            in_gap = True
            start = i
        elif not v and in_gap:
            gaps.append((start, i))
            in_gap = False
    if in_gap:
        gaps.append((start, n))
    segs: List[Tuple[int, int]] = []
    prev = 0
    for (g0, g1) in gaps:
        if g0 - prev >= min_seg_h:
            segs.append((prev, g0))
        prev = g1
    if n - prev >= min_seg_h:
        segs.append((prev, n))
    segs.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
    return segs[:3]


def _horizontal_bounds(arr: np.ndarray, y0: int, y1: int) -> Tuple[int, int]:
    band = arr[y0:y1, :]
    col_prof = np.mean(np.abs(np.diff(band, axis=0, prepend=band[[0], :])), axis=0)
    thr = np.percentile(col_prof, 40)
    mask = col_prof > thr
    xs = np.where(mask)[0]
    if xs.size == 0:
        return 0, arr.shape[1]
    left = max(0, xs.min() - 5)
    right = min(arr.shape[1], xs.max() + 5)
    return left, right


def detect_graphs_on_page_local(img: Image.Image) -> Dict:
    # Convert to grayscale and shrink for speed
    arr, sx, sy = _to_gray_np(img, max_side=1400)
    h, w = arr.shape
    prof = _vertical_profile(arr)
    # Expect sizable segments; require at least ~12% of page height
    min_seg_h = max(40, int(0.12 * h))
    segs = _find_segments(prof, min_seg_h)
    if len(segs) < 3:
        # Heuristic: choose cutlines near 1/3 and 2/3 of height, snap to local minima
        def local_min_around(center: int, radius: int) -> int:
            a = max(0, center - radius)
            b = min(h - 1, center + radius)
            sl = prof[a:b]
            return a + int(np.argmin(sl))

        c1 = local_min_around(int(h * 0.33), int(h * 0.1))
        c2 = local_min_around(int(h * 0.66), int(h * 0.1))
        cuts = sorted([c1, c2])
        segs = [(0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], h)]
        # enforce minimum segment height
        segs = [s for s in segs if s[1] - s[0] >= min_seg_h]
        if len(segs) < 3:
            return {"page_type": "other", "location": "", "graphs": []}
    # Sort by vertical position (top to bottom)
    segs_sorted = sorted(segs, key=lambda ab: ab[0])[:3]
    kinds = ["wind", "precipitation", "temperature"]
    graphs = []
    for kind, (y0, y1) in zip(kinds, segs_sorted):
        x0, x1 = _horizontal_bounds(arr, y0, y1)
        # Normalize to original page
        nx = x0 / w
        ny = y0 / h
        nw = (x1 - x0) / w
        nh = (y1 - y0) / h
        graphs.append({"kind": kind, "bbox": [float(nx), float(ny), float(nw), float(nh)]})
    return {"page_type": "area_graphs", "location": "", "graphs": graphs}

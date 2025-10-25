import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(path: Optional[str], event: Dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    evt = {"ts": _now_iso(), **event}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(evt) + "\n")


@contextmanager
def time_block(name: str, metrics_path: Optional[str] = None, **fields):
    start = time.perf_counter()
    try:
        yield
    finally:
        dur = time.perf_counter() - start
        log_event(metrics_path, {"type": name, "duration_s": round(dur, 3), **fields})


def summarize_metrics(path: str) -> str:
    if not os.path.exists(path):
        return "No metrics found."
    import collections

    counts = collections.Counter()
    totals = collections.defaultdict(float)
    per_page = collections.defaultdict(float)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                evt = json.loads(line)
            except Exception:
                continue
            t = evt.get("type")
            d = float(evt.get("duration_s", 0.0) or 0.0)
            counts[t] += 1
            totals[t] += d
            if t in ("page_processing",):
                key = (evt.get("file"), evt.get("page"))
                per_page[key] += d

    def fmt(s: float) -> str:
        return f"{s:.1f}s"

    lines = ["Metrics summary:"]
    for t, c in counts.most_common():
        total = totals[t]
        avg = total / c if c else 0.0
        lines.append(f"- {t}: n={c}, total={fmt(total)}, avg={fmt(avg)}")

    # Top slow pages
    slow_pages = sorted(per_page.items(), key=lambda kv: kv[1], reverse=True)[:5]
    if slow_pages:
        lines.append("Top slow pages:")
        for (fname, page), dur in slow_pages:
            lines.append(f"- {fname} p{page}: {fmt(dur)}")

    return "\n".join(lines)

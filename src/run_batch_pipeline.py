import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from .pdf_render import render_pdf_to_images
from .utils import crop_normalized_box, ensure_rgb, resize_max_side, pil_to_data_url
from .batch_builder import (
    build_detect_regions_request,
    build_extract_wind_request,
    build_extract_precip_request,
    build_extract_temperature_request,
    write_jsonl,
)
from .parse_batch_results import parse_jsonl_file


def submit_and_wait(client: OpenAI, jsonl_path: str, out_results_path: str, poll_interval: int = 20) -> str:
    with open(jsonl_path, "rb") as f:
        up = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    print(f"Submitted batch {batch.id}; status={batch.status}")
    while True:
        b = client.batches.retrieve(batch.id)
        print(f"status={b.status}")
        if b.status in ("completed", "failed", "expired", "canceled"):
            break
        import time

        time.sleep(poll_interval)
    if b.status != "completed":
        raise SystemExit(f"Batch {b.id} ended with status {b.status}")
    fid = b.output_file_id
    if not fid:
        # When errors, output_file_id can be None; download error_file_id for debugging
        if b.error_file_id:
            content = client.files.content(b.error_file_id)
            Path(out_results_path).parent.mkdir(parents=True, exist_ok=True)
            err_path = str(Path(out_results_path).with_suffix(".errors.jsonl"))
            with open(err_path, "wb") as f:
                f.write(content.read())
            raise SystemExit(f"Batch completed with no output_file_id; errors saved to {err_path}")
        raise SystemExit("No output_file_id on completed batch")
    content = client.files.content(fid)
    Path(out_results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_results_path, "wb") as f:
        f.write(content.read())
    print(f"Saved results to {out_results_path}")
    return out_results_path


def build_detection_jsonl(model: str, pdfs: List[str], out_jsonl: str, dpi: int, max_page_px: int) -> str:
    items: List[Dict] = []
    for pdf in pdfs:
        pages = render_pdf_to_images(pdf, dpi=dpi)
        total = len(pages)
        for i, img in enumerate(pages, start=1):
            if i in (1, total):
                continue
            req = build_detect_regions_request(model, img, max_side=max_page_px)
            req["custom_id"] = f"detect::{Path(pdf).name}::p{i}"
            items.append(req)
    write_jsonl(items, out_jsonl)
    print(f"Wrote {len(items)} batch items to {out_jsonl}")
    return out_jsonl


def build_extraction_jsonl_from_detect(model: str, detect_results_jsonl: str, out_jsonl: str, dpi: int, max_crop_px: int, kinds: List[str]) -> str:
    # Parse detection results
    pages_by_file: Dict[str, Dict[int, List[Dict]]] = {}
    with open(detect_results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("error"):
                continue
            resp_wrapper = obj.get("response") or {}
            body = resp_wrapper.get("body") or {}
            # Extract JSON from text content
            parsed = None
            outs = body.get("output") or []
            for o in outs:
                for c in o.get("content") or []:
                    if c.get("type") in ("output_text", "text"):
                        try:
                            parsed = json.loads(c.get("text", ""))
                        except Exception:
                            pass
                if parsed:
                    break
            if not parsed:
                continue
            cid = obj.get("custom_id", "")
            parts = cid.split("::")
            if len(parts) < 3:
                continue
            file = parts[1]
            pstr = parts[2]
            page = int(pstr[1:]) if pstr.startswith("p") else 0
            if parsed.get("page_type") != "area_graphs":
                continue
            pages_by_file.setdefault(file, {})[page] = parsed.get("graphs", [])

    items: List[Dict] = []
    for file, pages in pages_by_file.items():
        candidates = list(Path(".").glob(file)) or list(Path(".").glob(f"**/{file}"))
        if not candidates:
            print(f"Warning: cannot find PDF for {file}")
            continue
        pdf_path = str(candidates[0])
        all_pages = render_pdf_to_images(pdf_path, dpi=dpi)
        for pnum, graphs in pages.items():
            if pnum < 1 or pnum > len(all_pages):
                continue
            img = all_pages[pnum - 1]
            by_kind: Dict[str, any] = {}
            for g in graphs:
                kind = g.get("kind")
                if kind not in kinds:
                    continue
                bbox = g.get("bbox")
                crop = crop_normalized_box(img, tuple(bbox))
                by_kind[kind] = crop
            if "wind" in by_kind:
                req = build_extract_wind_request(model, by_kind["wind"], max_side=max_crop_px)
                req["custom_id"] = f"extract::{file}::p{pnum}::wind"
                items.append(req)
            if "precipitation" in by_kind:
                req = build_extract_precip_request(model, by_kind["precipitation"], max_side=max_crop_px)
                req["custom_id"] = f"extract::{file}::p{pnum}::precipitation"
                items.append(req)
            if "temperature" in by_kind:
                req = build_extract_temperature_request(model, by_kind["temperature"], max_side=max_crop_px)
                req["custom_id"] = f"extract::{file}::p{pnum}::temperature"
                items.append(req)
    write_jsonl(items, out_jsonl)
    print(f"Wrote {len(items)} batch items to {out_jsonl}")
    return out_jsonl


def main():
    parser = argparse.ArgumentParser(description="End-to-end batch pipeline: detection -> extraction -> CSV")
    parser.add_argument("--input", nargs="+", help="Input PDF files")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-5"))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max_page_px", type=int, default=900)
    parser.add_argument("--max_crop_px", type=int, default=720)
    parser.add_argument("--poll_interval", type=int, default=20)
    parser.add_argument("--out_csv", default="out/batch_results.csv")
    # Optional override paths
    parser.add_argument("--detect_jsonl", default="out/batch_detect.jsonl")
    parser.add_argument("--detect_results", default="out/batch_detect_results_structured.jsonl")
    parser.add_argument("--extract_jsonl", default="out/batch_extract.jsonl")
    parser.add_argument("--extract_results", default="out/batch_extract_results.jsonl")
    args = parser.parse_args()

    Path("out").mkdir(exist_ok=True)
    client = OpenAI()

    # 1) Detection JSONL
    build_detection_jsonl(args.model, args.input, args.detect_jsonl, args.dpi, args.max_page_px)

    # 2) Submit detection and wait
    submit_and_wait(client, args.detect_jsonl, args.detect_results, poll_interval=args.poll_interval)

    # 3) Build extraction JSONL from detection results
    kinds = ["wind", "precipitation", "temperature"]
    build_extraction_jsonl_from_detect(args.model, args.detect_results, args.extract_jsonl, args.dpi, args.max_crop_px, kinds)

    # 4) Submit extraction and wait
    submit_and_wait(client, args.extract_jsonl, args.extract_results, poll_interval=args.poll_interval)

    # 5) Parse results to CSV
    rows = parse_jsonl_file(args.extract_results)
    import pandas as pd

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()


import argparse
import os
import time
from pathlib import Path

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="Submit a JSONL to OpenAI Batch API and download results JSONL")
    parser.add_argument("--jsonl", required=True, help="Input JSONL with requests")
    parser.add_argument("--endpoint", default="/v1/responses")
    parser.add_argument("--completion_window", default="24h")
    parser.add_argument("--out_results", required=True, help="Output results JSONL path")
    parser.add_argument("--poll_interval", type=int, default=30)
    args = parser.parse_args()

    client = OpenAI()

    # Upload file
    with open(args.jsonl, "rb") as f:
        up = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint=args.endpoint,
        completion_window=args.completion_window,
    )
    print(f"Submitted batch {batch.id}; status={batch.status}")

    # Poll
    while True:
        b = client.batches.retrieve(batch.id)
        print(f"status={b.status}")
        if b.status in ("completed", "failed", "expired", "canceled"):
            break
        time.sleep(args.poll_interval)

    if b.status != "completed":
        raise SystemExit(f"Batch {b.id} ended with status {b.status}")

    # Download results
    fid = b.output_file_id
    if not fid:
        raise SystemExit("No output_file_id on completed batch")
    content = client.files.content(fid)
    out_path = Path(args.out_results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(content.read())
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()

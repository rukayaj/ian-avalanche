Overview
========

This project extracts hourly forecast series from sportscotland/SAIS avalanche PDFs using OpenAI's multimodal Responses + Batch APIs. The pipeline renders each PDF, locates the three area graphs (wind, precipitation, temperature), crops them, and asks the model for strict JSON output that is flattened into a CSV.


Quick Start
-----------

1. Drop every PDF you want to process into the `in/` directory (nested folders are allowed).
2. Create `.env.local` containing your API key and any overrides, e.g.:
   ```
   OPENAI_API_KEY=sk-...
   MODEL_NAME=gpt-5
   ```
3. Run the full pipeline:
   ```
   docker compose up --build
   ```
   The container renders PDFs, submits detection + extraction batches, polls for completion, and writes `out/batch_results.csv`. Intermediate JSONL artefacts are stored alongside the CSV in `out/`.


Command-line Usage
------------------

The main entry point is `python -m src.run_batch_pipeline`, which now auto-discovers PDFs. Useful flags:

- `--input <PDF ...>`: explicit list of PDFs (skips directory scan).
- `--input-dir DIR`: directory to search for PDFs (default `in/`). Created automatically if missing.
- `--output-dir DIR`: base directory for JSONL/CSV artefacts (default `OUT_DIR` env or `out/`).
- `--dpi`: render DPI (default 150).
- `--max_page_px`, `--max_crop_px`: image down-scaling limits before requests.
- `--poll_interval`: seconds between Batch status checks.
- `--kinds wind,precipitation,temperature`: subset of graphs to extract.

All paths accept absolute or relative values and are resolved before use. Duplicate PDF basenames are rejected early to avoid ambiguous batch IDs.


Supporting Tools
----------------

The standalone helpers still exist but now lean on the shared pipeline utilities:

- `src.build_batch`: build detection JSONL files (LLM-based region detection).
- `src.build_extract_batch_from_detection`: generate extraction JSONL from previously downloaded detection results.
- `src.build_extract_batch`: build extraction JSONL by running local heuristic detection.
- `src.batch_submit`: submit any JSONL to the Batch API and download its results.
- `src.parse_batch_results`: merge extraction results JSONL into a CSV (used automatically by the pipeline).

All helpers accept `--input` / `--input-dir` where relevant and write outputs under `out/` by default.


Testing
-------

Dependencies require native libraries (PyMuPDF, Pillow). The easiest way to run the test-suite is inside the container:

```
docker compose run --rm extractor pytest -q
```

`tests/test_local_detect_bulk.py` is marked `slow`; enable it with `-m slow` and adjust workload using `MAX_PDFS` / `TEST_DPI`.


Implementation Notes
--------------------

- Detection + extraction requests are validated via Pydantic-generated JSON Schema to keep Batch outputs consistent.
- Detection results are correlated back to PDFs using the filename embedded in each request `custom_id`; the runner refuses duplicates to keep this reliable.
- Intermediate artefacts are written with deterministic filenames so interrupted runs can be resumed or inspected.
- Local detection tooling remains available for audits and regression testing but the default workflow now uses the cloud model end-to-end.

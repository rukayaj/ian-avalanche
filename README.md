Overview

This repo contains a Dockerized pipeline to extract hourly forecast data from the three graphs on sportscotland/SAIS avalanche forecast PDFs using a multimodal LLM. It:
- Renders PDF pages to images (PyMuPDF)
- Default: LLM-based region detection (robust to layout shifts). Local detection tooling remains available for auditing only.
- Crops each graph and extracts structured series via JSON Schema outputs
- Flattens results into a CSV compatible with the provided sample schema

What it extracts

- Wind graph: hourly wind speed (mph), gust (mph), direction (compass)
- Precip graph: hourly rain (mm), snow (cm), precip type (text) with 0 when no bar
- Temperature graph: hourly air temp (degC, left axis), freezing level (m), wet bulb freezing level (m) from right axis

Detection and CV

- Region detection uses a vision LLM in default runs and Batch workflows. A local detection tool is included for audit/experiments only.
- Data extraction is LLM-only with strict structured outputs. No numeric digitization (OpenCV) is used.

Docker usage

1) Build
   docker build -t forecast-extractor .

2) Run (mount current directory for PDFs and outputs)
   docker run --rm \
     -e OPENAI_API_KEY=$OPENAI_API_KEY \
     -e MODEL_NAME=gpt-5 \
     -v "$PWD":/app \
     forecast-extractor \
     --input "Scottish Avalanche Information Service-5.pdf" \
     --out_csv out/extracted.csv

Docker Compose

1) Create a `.env.local` file with your key:
   OPENAI_API_KEY=sk-...

2) Build:
   docker compose build

3) Run (process 3 PDFs, override model if needed):
   docker compose run --rm -e MODEL_NAME=gpt-5 extractor \
     --input "Scottish Avalanche Information Service-5.pdf" \
            "sportscotland Avalanche Advice-1003.pdf" \
            "sportscotland Avalanche Advice-1010.pdf" \
     --out_csv out/extracted.csv

Batch API (LLM detection + extraction)

- Build a detection batch JSONL (LLM-based region detection):
  docker run --rm \
    -e MODEL_NAME=gpt-5 \
    -v "$PWD":/app \
    forecast-extractor \
    -m src.build_batch -- --input "Scottish Avalanche Information Service-5.pdf" "sportscotland Avalanche Advice-1003.pdf" --out_jsonl out/batch_detect.jsonl --dpi 200 --max_page_px 900

- Submit detection batch and download results via helper:
  docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -v "$PWD":/app forecast-extractor \
    -m src.batch_submit -- --jsonl out/batch_detect.jsonl --out_results out/batch_detect_results.jsonl

- Build extraction batch JSONL from detection results (per-graph requests using detected bboxes):
  docker run --rm \
    -e MODEL_NAME=gpt-5 \
    -v "$PWD":/app \
    forecast-extractor \
    -m src.build_extract_batch_from_detection -- --detect_results out/batch_detect_results.jsonl --out_jsonl out/batch_extract.jsonl --dpi 200 --max_crop_px 720 --kinds wind,precipitation,temperature

- Submit extraction batch and download results via helper:
  docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -v "$PWD":/app forecast-extractor \
    -m src.batch_submit -- --jsonl out/batch_extract.jsonl --out_results out/batch_extract_results.jsonl

- Parse and merge downloaded batch results JSONLs into one CSV:
  docker run --rm -v "$PWD":/app forecast-extractor -m src.parse_batch_results -- \
    --input_jsonl out/batch_extract_results.jsonl \
    --out_csv out/batch_results.csv

One-shot end-to-end run

- Single command to build detection JSONL, submit/poll, build extraction JSONL, submit/poll, and write CSV:
  docker compose run --rm --entrypoint python extractor -m src.run_batch_pipeline \
    --input "Scottish Avalanche Information Service-5.pdf" "sportscotland Avalanche Advice-1003.pdf" \
    --dpi 150 --max_page_px 900 --max_crop_px 720 --poll_interval 20 \
    --out_csv out/batch_results.csv

Note: You can still build per-graph batches directly from local detection if desired (faster pre-processing, but you requested LLM-based detection).

Testing

- Run unit tests in Docker:
  docker run --rm -v "$PWD":/app forecast-extractor python -m pytest -q

- Run thorough local-detection tests (slow, across all PDFs):
  docker compose run --rm --entrypoint bash extractor -lc "pip install -r requirements.txt >/dev/null && export PYTHONPATH=/app && pytest -q -m slow"
  - Tune with env vars: `MAX_PDFS=5 TEST_DPI=150 pytest -q -m slow`

- Audit detection visually and via CSV:
  docker run --rm -v "$PWD":/app forecast-extractor -m src.audit_local_detect -- --dpi 200 --out_dir out/local_detect_audit


Notes

- Requires OPENAI_API_KEY in the environment.
- Produces page crops for debug in out/<pdf>_page_<n>/crop_<kind>.png
- The pipeline uses LLM-driven region detection so it tolerates shifting layouts.
- If a series cannot be read precisely, the model estimates using axes; we always enforce 24 entries. You can re-run to refine estimates.
 - Station altitude is intentionally ignored here and can be parsed later from textual content if needed.

Error margin and QA

- Expect small rounding/reading error from chart digitization (usually within a few units). Gust ≥ speed is checked implicitly by the model; visually verify if needed using saved crops.
- We validate schema (24 entries, units) and will fail-fast if malformed; adjust prompts if a particular document varies.

CLI Options

- --input: list of PDF files (default: all PDFs in CWD)
- --out_csv: output CSV path (default: out/extracted.csv)
- --model: OpenAI model name (default from MODEL_NAME env)
- --out_dir: directory for debug crops (default: out)

Extending

- If certain graphs prove hard for the LLM, add a computer-vision fallback for that graph (e.g., line/bar digitization with OpenCV) and use the LLM only for axes/labels.

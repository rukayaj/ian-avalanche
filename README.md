Overview

This repo contains a Dockerized pipeline to extract hourly forecast data from the three graphs on sportscotland/SAIS avalanche forecast PDFs using a multimodal LLM. It:
- Renders PDF pages to images (PyMuPDF)
- Uses a vision LLM to detect graph regions with normalized bounding boxes (no fixed coordinates)
- Crops each graph and extracts structured series via JSON Schema outputs
- Flattens results into a CSV compatible with the provided sample schema

What it extracts

- Wind graph: hourly wind speed (mph), gust (mph), direction (compass)
- Precip graph: hourly rain (mm), snow (cm), precip type (text) with 0 when no bar
- Temperature graph: hourly air temp (degC, left axis), freezing level (m), wet bulb freezing level (m) from right axis

No CV fallback

- This build uses only the LLM for region detection and data extraction. There are no computer-vision fallbacks (no OpenCV digitization, no OCR). Validation rules trigger a re-ask to the LLM if inconsistencies are detected (e.g., gust < speed, non-constant station altitude).

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

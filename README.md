Overview
========

This project extracts hourly forecast series from sportscotland/SAIS avalanche PDFs and produces a long-format CSV. There are two pipelines:

1) **Local (default, no OpenAI)** — uses `pdftotext` and the PDF text layer to read numerics and labels, plus page-9 accumulations/date. Fast and offline.
2) **Cloud (optional)** — the original OpenAI Batch-based flow (`src.run_batch_pipeline`) if you still want LLM extraction.

Quick Start (local, no OpenAI)
------------------------------
1. Put PDFs in the `in/` directory (flat list; no recursion).
2. Run:
   ```
   python -m src.local_pdf_pipeline --input-dir in --out-csv out/local_results.csv
   ```
   or specify PDFs explicitly:
   ```
   python -m src.local_pdf_pipeline --input file1.pdf file2.pdf --out-csv out/local_results.csv
   ```
3. The CSV will be written to `out/local_results.csv` (same long format as before, including page 9 totals and forecast start date).

Docker (local pipeline)
-----------------------
Build and run the container (entrypoint is the local pipeline):
```
docker build -t sais-local .
docker run --rm -v "$(pwd)/in:/app/in" -v "$(pwd)/out:/app/out" sais-local --input-dir /app/in --out-csv /app/out/local_results.csv
```

Optional: OpenAI Batch Pipeline
-------------------------------
If you still need the cloud flow:
1. Create `.env.local` with your key:
   ```
   OPENAI_API_KEY=sk-...
   MODEL_NAME=gpt-5
   ```
2. Run:
   ```
   python -m src.run_batch_pipeline --input-dir in --output-dir out
   ```
   This submits Batch jobs and writes `out/batch_results.csv` plus JSONL artefacts in `out/`.

Notes
-----
- The local pipeline reads numerics via `pdftotext -layout`, directions/precip types from the PDF text layer, and page-9 accumulations/date from the table. It does not require OpenAI.
- Dockerfile now installs `poppler-utils` for `pdftotext` and defaults to the local pipeline entrypoint. Use `python -m src.run_batch_pipeline` inside the container if you need the cloud path.
- `in/` and `out/` are created automatically if missing; `out/` is in `.gitignore`—delete or ignore large outputs before committing.

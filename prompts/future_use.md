You have avalanche forecast PDFs from sportscotland/SAIS. These PDFs already contain the numbers and labels as real text/objects (not just pixels). Avoid using vision/LLM extraction if the PDF text layer can provide the data directly.

Task:
- Extract hourly graph data (pages 2–8): wind speed/gust (mph), wind direction labels, rain (mm), snow (cm), precip type, air temp (°C), freezing level (m), wet-bulb freezing level (m).
- Extract page 9 table: cumulative snow (cm) and rain (mm) per site, plus the forecast start date (the “1800 on [DATE1] until 1800 [DATE2]” line — capture DATE1).
- Output long-format CSV with the same columns as our previous runs (SourceFile, Page, Location, Section, Measurement, MeasurementType, Units, HourLabel, HourIndex, ValueNumeric, ValueText, etc.).

Requirements:
1) First, inspect the PDF text layer (e.g., PyMuPDF `page.get_text("words"/"blocks")` and/or `pdftotext -layout`). If the text layer contains the graph labels and numeric tables:
   - Use those values directly. Do NOT call OpenAI or run OCR.
   - Numeric series: parse from `pdftotext -layout` (24 values per series).
   - Directions/precip types: parse from the PDF text layer (sort left→right).
   - Page 9: parse the date line and the accumulation table via `pdftotext -layout`.
2) Only if the text layer is missing a needed field, then fall back to a minimal OCR/LLM pass for that field alone—otherwise stay local.
3) Provide a simple CLI script to run over a folder of PDFs, no network required.

Deliverables:
- A local pipeline script that produces the long-format CSV for all pages (including page 9).
- Dockerfile updated to include `poppler-utils` for `pdftotext` and default to the local pipeline entrypoint.
- A brief README update explaining the local flow and, optionally, how to run the old OpenAI pipeline if needed.

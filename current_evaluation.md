## Summary

I spot‑checked both PDFs by rendering the graph pages and comparing them to rows in out/batch_results.csv. I also read your batch JSONL responses to see what the model returned and where merges went wrong.

### Key Issues Found

Scottish Avalanche Information Service-5.pdf

  - Page 2 (Mullach an Rathain, 1023 m)
      - Missing data: CSV contains only Wind rows. No Precipitation or Temperature rows even though the page has those graphs. Evidence: out/
        batch_extract_results.jsonl shows the page‑2 precipitation and temperature calls returned “status: incomplete (max_output_tokens)”, so
        nothing made it into the CSV.
      - Location missing in Wind rows: CSV location is blank for all p2 wind entries. The model returned location:"" in the wind response.
  - Page 3 (A’Chailleach, 999 m)
      - Wind looks plausible; directions align with the top labels.
  - Page 4 (Ben Nevis, 1345 m)
      - Inconsistent location: Temperature rows are labeled “Glen Coe” in CSV while Wind/Precip rows use “Ben Nevis (1345 m)”. The graph title
        itself is “Ben Nevis (1345 metres)”. The area footer says “Glen Coe area”, which the model appears to have picked up for temperature only.
      - Low early wind values: CSV has 6 mph speed / 10 mph gust at 18:00 (Scottish Avalanche Information Service-5.pdf,4,Ben Nevis...), but the
        chart shows roughly mid‑teens to ~20 mph speed and ~20–25 mph gusts at 18:00. Subsequent hours (19:00, 20:00) also look low by ~5–10 mph.
  - Page 5 (Bidean nam Bian, 1150 m)
      - Looks internally consistent; values and directions broadly match the plot trends.

sportscotland Avalanche Advice-1003.pdf

  - Page 2 (Mullach An Rathain, 1023 m)
      - Wind series matches the plot trend well (e.g., ~mid‑40s to low‑60s in the morning/early afternoon).
      - Location missing in Precip rows: CSV location field is blank for all precipitation rows (lines 794–865).
  - Page 3 (A’Chailleach, 999 m)
      - Location missing in Temperature rows (lines 1028–1081).
  - Page 4 (Ben Nevis, 1345 m)
      - Wind location says “Ben Nevis (approx)” in CSV, but the title on the plot is “Ben Nevis (1345 metres)”, so “approx” is spurious.
      - Temperature location is “Unknown” in CSV (lines 1154–1225) instead of the graph title.
  - Page 5 (Bidean nam Bian, 1150 m)
      - Wind and Precip look consistent with the charts; precipitation types are simplified (e.g., “No Precip”, “Rain”) rather than “Snow shower”,
        which is acceptable if you’re normalizing categories—but note the mismatch with the plot’s labels.

Root Causes (what’s driving the issues)

  - Location isn’t requested or enforced in extraction prompts:
      - Your extraction builders explicitly instruct the model to “Return JSON only with key 'hours'” for wind/precip/temperature (src/
        batch_builder.py:88, 111, 135), while the schema (src/models.py) has an optional location. The model sometimes includes location (often
        blank) and sometimes grabs the area footer (e.g., “Glen Coe”) instead of the graph title.
  - Token-limit truncation drops entire series:
      - Page 2 extractions for SAIS-5 returned “incomplete due to max_output_tokens”, so no CSV rows were produced for those series.
  - Occasional numeric read errors:
      - Early wind values on SAIS-5 page 4 are clearly underestimated (likely a poor read of the first few points near the axis).
  - Category normalization differences:
      - The precip “type” values are simplified (Rain/Snow/Mixed/No Precip) versus the chart labels (e.g., “Snow shower”). If you want chart-text
        fidelity, prompt for the exact phrases.

Prompt and Pipeline Improvements

Always include and standardize location

  - Update extraction prompts to require the exact graph title text for location: “Return 'location' exactly as shown in the graph title (e.g.,
    'Ben Nevis (1345 metres)'). Do not use page footers/area names.”
      - Where to change:
          - prompts/wind.md:1, prompts/precip.md:1, prompts/temperature.md:1
          - Code that currently suppresses location: src/batch_builder.py:88, 111, 135. Remove “Return JSON only with key 'hours'…”, and instead
            require the schema keys (including location). Or pass in the detected location and tell the model to mirror it exactly.
  - Parser fallback: If extraction returns blank location, fall back to detect result’s location for that page.

Reduce token usage to avoid truncation

  - Compact output format: Instead of 24 objects per series, have the model return three arrays with 24 numbers (plus one array of 24 strings for
    directions/types) and one parallel array for hour labels. This cuts tokens by ~3–5x and prevents max_output_tokens truncation.
      - If you keep the current schema, set max_output_tokens a bit higher for temperature/precip since they often exceed limits.
  - Use the combined extractor for each page (one request returning wind/precip/temperature) to amortize tokens and reduce overhead; the code
    already supports this (LLMClient.extract_all, src/llm_client.py:38, and src/batch_builder.py:49). It also improves consistency (one location
    for all three series).

Make location and series validation explicit

  - Add to prompts:
      - “Hours must be exactly 24 values for 18..17; if unreadable, estimate from axes and keep monotonic time indexing.”
      - “Wind speed/gust must be consistent with axis ticks; early values near the axis should not be rounded down to zero-like numbers.”
  - Keep your validators but strengthen feedback:
      - If hour count != 24 or values are wildly off-axis, provide explicit retry feedback. You already do single‑retry logic in src/extractors.py;
        consider including quantitative ranges in feedback (e.g., “first three points must be 10–25 mph based on visible ticks”).

Normalize precipitation type (if desired)

  - If you want exact chart labels, prompt for “use the label text above columns exactly (e.g., ‘Snow shower’)”. If you want normalized categories,
    add a mapping in the prompt (e.g., map “Snow shower” → “Snow”).

Minor consistency tweaks

  - For direction, explicitly list the accepted compass values (you already do in schema); in the prompt, say “use only these 16 compass points.”
  - Ask the model to round sensibly (e.g., one decimal for temps, integer mph/cm/mm).

Concrete examples of issues (so you can verify quickly)

  - SAIS-5 page 2: Precip + Temp missing entirely in CSV; see out/batch_extract_results.jsonl entries flagged incomplete.
  - SAIS-5 page 4: 18:00 wind values are too low (CSV shows 6/10 mph); the plot shows roughly mid‑teens to ~20/25 mph.
  - SAIS-5 page 4: Temperature rows labeled “Glen Coe” instead of “Ben Nevis (1345 m)”.
  - sportscotland-1003 page 2: Precip location blank.
  - sportscotland-1003 page 4: Wind location “Ben Nevis (approx)”; temp location “Unknown”; should both be “Ben Nevis (1345 metres)”.


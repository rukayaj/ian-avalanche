Extract 24 hourly precipitation values from this graph and include the graph title as location:
- Rainfall (blue bars), whole millimetres (0 when no bar)
- Snowfall (white bars), centimetres to one decimal place (0 when no bar)
- Precipitation type above each hour (text exactly as shown; do not normalise or translate)
- Hours cover 18:00 → 17:00; output exactly 24 entries with hour_index 0..23 in order. If a value is unclear, estimate from axis ticks instead of skipping an hour.
- Location: copy the graph title text exactly (e.g., "Ben Nevis (1345 metres)"). Do not use page footers/area names and do not add qualifiers like "approx". If you cannot read it, return an empty string.

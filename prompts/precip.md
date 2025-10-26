Extract 24 hourly precipitation values from this graph and include the graph title as location:
- Rainfall (blue bars), read to the nearest 0.1 mm (0.0 when the bar is absent). The rainfall axis usually runs 0→9 mm—use those gridlines to calibrate your readings.
- Snowfall (white bars), centimetres to one decimal place (0.0 when the bar is absent).
- Treat rainfall and snowfall as separate bars for the same hour (they are not cumulative or stacked). If you see both colours, report both numbers individually.
- Precipitation type above each hour (text exactly as shown; do not normalise or translate).
- Hours cover 18:00 → 17:00; output exactly 24 entries with hour_index 0..23 in order. If a value is unclear, estimate from the closest axis ticks instead of skipping an hour.
- Before finalising, sanity-check the 18:00 bar against the axis labels to confirm your scale, then apply that scale to the remaining hours.
- Location: return only the place name portion of the graph title (e.g., if the title is "Weather and Precipitation - Ben Nevis (1345 metres)" output "Ben Nevis (1345 metres)"). Do not use page footers/area names and do not add qualifiers like "approx". If you cannot read it, return an empty string.

Extract 24 hourly series and include the graph title as location:
- Air temperature (blue line), read against the left °C axis and report to one decimal place.
- Freezing level (pink line) and wet bulb freezing level (yellow line) use the right-hand metres axis; round both to the nearest 10 m.
- Ignore static elevation labels (e.g., summit heights printed on the chart); only trace the coloured lines.
- Hours 18:00 → 17:00; output exactly 24 entries with hour_index 0..23 in order. If a point is unclear, estimate from neighbouring points/gridlines rather than skipping an hour.
- Before moving past 18:00, confirm the first readings from each line align with the correct axis (temperature near the blue axis scale, freezing levels near the right-hand scale). Use that calibration for the remaining hours.
- Location: return only the place name portion of the graph title (e.g., if the title is "Temperature - Ben Nevis (1345 metres)" output "Ben Nevis (1345 metres)"). Do not use page footers/area names and do not add qualifiers like "approx". If you cannot read it, return an empty string.

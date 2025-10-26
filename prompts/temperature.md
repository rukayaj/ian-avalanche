Extract 24 hourly series and include the graph title as location:
- Air temperature (blue line), left axis °C, reported to one decimal place
- Freezing level (pink line), right axis metres, rounded to the nearest 10 m
- Wet bulb freezing level (yellow line), right axis metres, rounded to the nearest 10 m
- Hours 18:00 → 17:00; output exactly 24 entries with hour_index 0..23 in order. If a point is unclear, estimate from neighbouring points/gridlines rather than skipping an hour.
- Location: return only the place name portion of the graph title (e.g., if the title is "Temperature - Ben Nevis (1345 metres)" output "Ben Nevis (1345 metres)"). Do not use page footers/area names and do not add qualifiers like "approx". If you cannot read it, return an empty string.

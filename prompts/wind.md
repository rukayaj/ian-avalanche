Extract 24 hourly wind values from this graph and include the graph title as location:
- Wind speed (blue line), integer mph aligned with the axis ticks
- Wind gust (pink line), integer mph aligned with the axis ticks
- Wind direction above each hour (text; use only 16‑point compass: N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW)
- Hours cover 18:00 → 17:00; output exactly 24 entries with hour_index 0..23 in order. Estimate values from neighbouring points/gridlines rather than dropping an hour.
- Location: return only the place name portion of the graph title (e.g., if the title is "Wind - Ben Nevis (1345 metres)" output "Ben Nevis (1345 metres)"). Do not use page footers/area names and do not add qualifiers like "approx". If you truly cannot read it, return an empty string.

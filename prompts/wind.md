Extract 24 hourly wind values from this graph and include the graph title as location:
- Wind speed (blue line), mph
- Wind gust (pink line), mph
- Wind direction above each hour (text; use only 16‑point compass: N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW)
- Location: copy the graph title text exactly (e.g., "Ben Nevis (1345 metres)"). Do not use page footers/area names and do not add qualifiers like "approx". If you truly cannot read it, return an empty string.
Hours run from 18:00 to 17:00 next day. Return exactly 24 entries and ensure hour_index 0..23 aligns with 18..17.

Extract 24 hourly precipitation values from this graph and include the graph title as location:
- Rainfall (blue bars), mm
- Snowfall (white bars), cm
- Precipitation type above each hour (text). Normalize to one of: No Precip, Rain, Snow, Mixed, or Rain/Snow.
- Location: copy the graph title text exactly (e.g., "Ben Nevis (1345 metres)"). Do not use page footers/area names and do not add qualifiers like "approx". If you cannot read it, return an empty string.
If there is no bar, set the numeric value to 0. Hours run 18:00 to 17:00 next day. Return exactly 24 entries.

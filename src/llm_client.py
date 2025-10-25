import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from .models import RegionDetection, WindSeries, PrecipSeries, TempSeries, CombinedSeries


class LLMClient:
    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("MODEL_NAME", "gpt-5")
        self.client = OpenAI()

    def _call_schema(
        self,
        prompt: str,
        images_data_urls: List[str],
        text_format,
        *,
        max_output_tokens: int = 8_000,
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for url in images_data_urls:
            content.append({"type": "input_image", "image_url": url})

        # Use parse to get structured output validated to the given schema
        resp = self.client.responses.parse(
            model=self.model,
            max_output_tokens=max_output_tokens,
            input=[{"role": "user", "content": content}],
            text_format=text_format,
            reasoning={"effort": "low"},
            service_tier=os.getenv("SERVICE_TIER", "priority"),
        )
        # Parsed Pydantic model instance
        out = resp.output_parsed
        return out.model_dump() if out is not None else {}

    def detect_regions(self, page_data_url: str, text_format=RegionDetection, feedback: str = "") -> Dict[str, Any]:
        prompt = (
            "You are locating three graphs on a forecast page. "
            "Return normalized bounding boxes [x,y,w,h] in 0..1 for each graph and classify each as one of: "
            "wind, precipitation, temperature. Also provide the page_type as 'area_graphs' if you see three graphs "
            "with hourly 24h forecast from 18→17; otherwise 'other'. Provide 'location' as the exact graph title text (e.g., 'Ben Nevis (1345 metres)'); do not use page footers/area names. "
            "Ensure bboxes are tight around the plotted area and legends/top titles.\n\n"
            + (f"Constraints/Corrections: {feedback}" if feedback else "")
        )
        return self._call_schema(prompt, [page_data_url], text_format)

    def extract_wind(self, crop_url: str, text_format=WindSeries, feedback: str = "") -> Dict[str, Any]:
        prompt = (
            "Extract 24 hourly wind series for this graph: speed (blue line, mph), gust (pink line, mph), "
            "and wind direction text above each hour. Hours run from 18:00 to 17:00 next day; ensure all 24 hours appear with hour_index 0..23. "
            "Return exactly 24 entries containing hour_label, hour_index, wind_speed_mph, wind_gust_mph, and wind_direction (use only these compass points: N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW). "
            "Report wind_speed_mph and wind_gust_mph as integer mph values aligned with the axis ticks—never round them down to near-zero when the plotted line is clearly above the baseline. "
            "If a value cannot be read, estimate it from neighbouring points and the axis gridlines so the series remains smooth. "
            "Include 'location' exactly as the graph title text; do not use page footers or add qualifiers. If the title is unreadable, return an empty string.\n\n"
            + (f"Constraints/Corrections: {feedback}" if feedback else "")
        )
        return self._call_schema(prompt, [crop_url], text_format, max_output_tokens=16_000)

    def extract_precip(self, crop_url: str, text_format=PrecipSeries, feedback: str = "") -> Dict[str, Any]:
        prompt = (
            "Extract 24 hourly precipitation series for this graph: rain (blue bars, mm), snow (white bars, cm), "
            "and precipitation type text above each hour. Hours run 18→17, so output must contain 24 ordered entries with hour_index 0..23. "
            "If no bar is present for an hour, set the numeric value to 0. "
            "Report rain_mm as whole millimetres and snow_cm with one decimal place, keeping values non-negative and consistent with the axis ticks. "
            "Include the precipitation type text exactly as printed above each hour (do not normalise or translate it). "
            "Include 'location' exactly as the graph title text; do not use page footers or add qualifiers. If the title is unreadable, return an empty string.\n\n"
            + (f"Constraints/Corrections: {feedback}" if feedback else "")
        )
        return self._call_schema(prompt, [crop_url], text_format, max_output_tokens=16_000)

    def extract_temperature(self, crop_url: str, text_format=TempSeries, feedback: str = "") -> Dict[str, Any]:
        prompt = (
            "Extract 24 hourly series: air temperature (blue, read left axis in °C), "
            "freezing level (pink, metres), and wet bulb freezing level (yellow, metres) from the right axis. "
            "Hours cover 18→17, so provide exactly 24 entries with hour_index 0..23 in order. "
            "Report temperatures to one decimal place and altitude values to the nearest 10 metres, matching the axis scale. "
            "If a value is unclear, estimate from neighbouring points and axis ticks instead of omitting it. "
            "Include 'location' exactly as the graph title text; do not use page footers or add qualifiers. If unreadable, return an empty string.\n\n"
            + (f"Constraints/Corrections: {feedback}" if feedback else "")
        )
        return self._call_schema(prompt, [crop_url], text_format, max_output_tokens=16_000)

    def extract_all(self, crop_urls: Dict[str, str], text_format=CombinedSeries, feedback: str = "") -> Dict[str, Any]:
        order = ["wind", "precipitation", "temperature"]
        images = [crop_urls[k] for k in order]
        prompt = (
            "Three images follow in order: (1) wind graph, (2) precipitation graph, (3) temperature graph.\n"
            "Extract 24 hourly series for each and return a single object with keys 'wind', 'precipitation', 'temperature'. Use the same 'location' copied exactly from the shared graph title (no page footers or qualifiers; return an empty string if unreadable).\n"
            "- Wind: wind_speed_mph and wind_gust_mph as integer mph values aligned with the axis ticks, wind_direction using only the 16-point compass list provided.\n"
            "- Precipitation: rain_mm as whole millimetres, snow_cm with one decimal place, precip_type exactly as printed above each hour (no normalisation). Values must be non-negative and 0 when no bar is present.\n"
            "- Temperature: air_temp_c to one decimal place, freezing_level_m and wet_bulb_freezing_level_m to the nearest 10 metres, following the right-hand axis scale.\n"
            "All three series must contain exactly 24 entries covering hour_index 0..23 (hours 18→17). If data is unclear, estimate from neighbouring points and axis gridlines rather than omitting the hour.\n\n"
            + (f"Constraints/Corrections: {feedback}" if feedback else "")
        )
        return self._call_schema(prompt, images, text_format, max_output_tokens=18_000)

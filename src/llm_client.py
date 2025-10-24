import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from .models import RegionDetection, WindSeries, PrecipSeries, TempSeries


class LLMClient:
    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("MODEL_NAME", "gpt-5")
        self.client = OpenAI()

    def _call_schema(self, prompt: str, images_data_urls: List[str], text_format) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for url in images_data_urls:
            content.append({"type": "input_image", "image_url": url})

        # Use parse to get structured output validated to the given schema
        resp = self.client.responses.parse(
            model=self.model,
            max_output_tokens=16_000,
            input=[{"role": "user", "content": content}],
            text_format=text_format,
        )
        # Parsed Pydantic model instance
        out = resp.output_parsed
        return out.model_dump() if out is not None else {}

    def detect_regions(self, page_data_url: str, text_format=RegionDetection, feedback: str = "") -> Dict[str, Any]:
        prompt = (
            "You are locating three graphs on a forecast page. "
            "Return normalized bounding boxes [x,y,w,h] in 0..1 for each graph and classify each as one of: "
            "wind, precipitation, temperature. Also provide the page_type as 'area_graphs' if you see three graphs "
            "with hourly 24h forecast from 18→17; otherwise 'other'. Provide 'location' if obvious from titles. "
            "Ensure bboxes are tight around the plotted area and legends/top titles.\n\n"
            + (f"Constraints/Corrections: {feedback}" if feedback else "")
        )
        return self._call_schema(prompt, [page_data_url], text_format)

    def extract_wind(self, crop_url: str, text_format=WindSeries, feedback: str = "") -> Dict[str, Any]:
        prompt = (
            "Extract 24 hourly wind series for this graph: speed (blue line, mph), gust (pink line, mph), "
            "and wind direction text above each hour. Hours run from 18:00 to 17:00 next day. "
            "Return exactly 24 entries: hour_label (like '18','19',...,'17'), hour_index (0..23), "
            "wind_speed_mph, wind_gust_mph, wind_direction (compass). "
            "If a value cannot be read, estimate from axes and be consistent across series.\n\n"
            + (f"Constraints/Corrections: {feedback}" if feedback else "")
        )
        return self._call_schema(prompt, [crop_url], text_format)

    def extract_precip(self, crop_url: str, text_format=PrecipSeries, feedback: str = "") -> Dict[str, Any]:
        prompt = (
            "Extract 24 hourly precipitation series for this graph: rain (blue bars, mm), snow (white bars, cm), "
            "and precipitation type text above each hour. Hours 18→17. If no bar present, value=0. "
            "Return exactly 24 entries with hour_label, hour_index, rain_mm, snow_cm, precip_type.\n\n"
            + (f"Constraints/Corrections: {feedback}" if feedback else "")
        )
        return self._call_schema(prompt, [crop_url], text_format)

    def extract_temperature(self, crop_url: str, text_format=TempSeries, feedback: str = "") -> Dict[str, Any]:
        prompt = (
            "Extract 24 hourly series: air temperature (blue, read left degC), "
            "freezing level (pink, metres), wet bulb freezing level (yellow, metres). "
            "Read the altitude-based lines from the right axis. Hours 18→17. "
            "Return exactly 24 entries with hour_label, hour_index, air_temp_c, freezing_level_m, "
            "wet_bulb_freezing_level_m.\n\n"
            + (f"Constraints/Corrections: {feedback}" if feedback else "")
        )
        return self._call_schema(prompt, [crop_url], text_format)

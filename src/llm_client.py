import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from .models import RegionDetection, WindSeries, PrecipSeries, TempSeries, CombinedSeries, DirectionStrip, PrecipTypeStrip
from .utils import pil_to_data_url


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
        # Some models do not support the reasoning.effort parameter; include it only when allowed.
        kwargs = {
            "model": self.model,
            "max_output_tokens": max_output_tokens,
            "input": [{"role": "user", "content": content}],
            "text_format": text_format,
            "service_tier": os.getenv("SERVICE_TIER", "priority"),
        }
        try:
            kwargs["reasoning"] = {"effort": "low"}
            resp = self.client.responses.parse(**kwargs)
        except Exception as exc:
            # Retry without reasoning when the model rejects that parameter.
            kwargs.pop("reasoning", None)
            resp = self.client.responses.parse(**kwargs)
        # Parsed Pydantic model instance
        out = resp.output_parsed
        return out.model_dump() if out is not None else {}

    def detect_regions(self, page_data_url: str, text_format=RegionDetection, feedback: str = "") -> Dict[str, Any]:
        prompt = (
            "You are locating three graphs on a forecast page. "
            "Return normalized bounding boxes [x,y,w,h] in 0..1 for each graph and classify each as one of: "
            "wind, precipitation, temperature. Also provide the page_type as 'area_graphs' if you see three graphs "
            "with hourly 24h forecast from 18→17; otherwise 'other'. When you provide 'location', strip any leading graph descriptor "
            "such as 'Wind - ' or 'Weather and Precipitation - ' and return only the place-name portion like 'Ben Nevis (1345 metres)'. "
            "Ignore area footer text at the bottom of the page. "
            "Ensure bboxes are tight around the plotted area and legends/top titles.\n\n"
            + (f"Constraints/Corrections: {feedback}" if feedback else "")
        )
        return self._call_schema(prompt, [page_data_url], text_format)

    def extract_wind(self, crop_url: str, text_format=WindSeries, feedback: str = "") -> Dict[str, Any]:
        prompt = (
            "Extract 24 hourly wind series for this graph: speed (blue line, mph), gust (pink line, mph), "
            "and wind direction text above each hour. The direction labels are printed at a 45-degree angle with the start of each label directly above the hour tick—read each one individually. "
            "Hours run from 18:00 to 17:00 next day; ensure all 24 hours appear with hour_index 0..23. "
            "Return exactly 24 entries containing hour_label, hour_index, wind_speed_mph, wind_gust_mph, and wind_direction (use only these compass points: N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW). "
            "Report wind_speed_mph and wind_gust_mph as integer mph values aligned with the mph axis ticks—read the gridlines to set the scale and avoid smoothing/rounding errors. "
            "Before filling the full series, read the 18:00 hour directly from the axis grid to confirm the scale and keep gusts greater than or equal to speeds. "
            "If a value cannot be read, estimate it from neighbouring points and the axis gridlines so the series remains smooth. "
            "Include 'location' exactly as the graph title text; do not use page footers or add qualifiers. If the title is unreadable, return an empty string.\n\n"
            + (f"Constraints/Corrections: {feedback}" if feedback else "")
        )
        return self._call_schema(prompt, [crop_url], text_format, max_output_tokens=16_000)

    def extract_direction_strip(self, crop_url: str, text_format=DirectionStrip, feedback: str = "") -> Dict[str, Any]:
        prompt = (
            "Read the wind-direction labels printed at a 45-degree angle above each hour tick. "
            "Return exactly 24 direction strings in order for hours 18, 19, 20, 21, 22, 23, 00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17. "
            "Use only these compass values: N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW. "
            "If any label is unreadable, use an empty string for that slot. Do not infer or repeat values unless they are printed."
            + (f"\nConstraints/Corrections: {feedback}" if feedback else "")
        )
        return self._call_schema(prompt, [crop_url], text_format, max_output_tokens=4_000)

    def extract_precip(self, crop_url: str, text_format=PrecipSeries, feedback: str = "") -> Dict[str, Any]:
        prompt = (
            "Extract 24 hourly precipitation series for this graph: rain (blue bars, mm), snow (white bars, cm), "
            "and precipitation type text above each hour. The precip-type labels are printed at a 45-degree angle with the start of each word directly above the matching hour tick—OCR each label there. "
            "Hours run 18→17, so output must contain 24 ordered entries with hour_index 0..23. "
            "If no bar is present for an hour, set the numeric value to 0.0. "
            "Report rain_mm to the nearest 0.1 mm (0.0 when the blue bar is absent) and snow_cm with one decimal place, keeping values non-negative and consistent with the axis ticks. "
            "Treat blue rainfall bars and white snowfall bars as separate values for the same hour—do not add them together. "
            "Before moving past 18:00, confirm the first hour aligns with the axis grid so subsequent readings stay calibrated. "
            "Precipitation type must be taken from the chart text and should match one of these when possible: Clear, Cloudy, Rain, Fog, Mist, Partly cloudy, Sunny, Snow, Snow showers, Sleet, Drizzle, Overcast. If unreadable, leave it empty rather than guessing. "
            "Include 'location' exactly as the graph title text; do not use page footers or add qualifiers. If the title is unreadable, return an empty string.\n\n"
            + (f"Constraints/Corrections: {feedback}" if feedback else "")
        )
        return self._call_schema(prompt, [crop_url], text_format, max_output_tokens=16_000)

    def extract_precip_type_strip(self, crop_url: str, text_format=PrecipTypeStrip, feedback: str = "") -> Dict[str, Any]:
        prompt = (
            "Read the precipitation-type labels printed at a 45-degree angle above each hour tick. "
            "Return exactly 24 precip types in order for hours 18, 19, 20, 21, 22, 23, 00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17. "
            "Use only these values: Clear, Cloudy, Rain, Fog, Mist, Partly cloudy, Sunny, Snow, Snow showers, Sleet, Drizzle, Overcast. "
            "If any label is unreadable, use an empty string for that slot. Do not infer or repeat values unless they are printed."
            + (f"\nConstraints/Corrections: {feedback}" if feedback else "")
        )
        return self._call_schema(prompt, [crop_url], text_format, max_output_tokens=4_000)

    def extract_temperature(self, crop_url: str, text_format=TempSeries, feedback: str = "") -> Dict[str, Any]:
        prompt = (
            "Extract 24 hourly series: air temperature (blue, read left axis in °C), "
            "freezing level (pink, metres), and wet bulb freezing level (yellow, metres) from the right axis. "
            "Hours cover 18→17, so provide exactly 24 entries with hour_index 0..23 in order. "
            "Report temperatures to one decimal place and altitude values to the nearest 10 metres, matching the axis scale. "
            "Ignore static elevation labels such as summit heights printed on the chart background. "
            "Before continuing past 18:00, confirm each coloured line matches its respective axis scale so later readings remain aligned. "
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
            "Before moving past the 18:00 hour, read each series directly from the grids to confirm the scale so later readings remain aligned.\n"
            "- Wind: wind_speed_mph and wind_gust_mph as integer mph values aligned with the axis ticks, wind_direction using only the 16-point compass list provided. Keep gusts ≥ speeds.\n"
            "- Precipitation: rain_mm to the nearest 0.1 mm (0.0 when the blue bar is absent), snow_cm with one decimal place, precip_type exactly as printed above each hour (no normalisation). Treat rainfall and snowfall bars as separate values for the same hour.\n"
            "- Temperature: air_temp_c to one decimal place, freezing_level_m and wet_bulb_freezing_level_m to the nearest 10 metres using the right-hand axis. Ignore static summit/elevation labels printed on the chart.\n"
            "All three series must contain exactly 24 entries covering hour_index 0..23 (hours 18→17). If data is unclear, estimate from neighbouring points and axis gridlines rather than omitting the hour.\n\n"
            + (f"Constraints/Corrections: {feedback}" if feedback else "")
        )
        return self._call_schema(prompt, images, text_format, max_output_tokens=18_000)

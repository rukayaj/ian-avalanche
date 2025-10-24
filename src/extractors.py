import os
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image

from .llm_client import LLMClient
from .utils import pil_to_data_url, crop_normalized_box, ensure_rgb
from .validate import validate_wind, validate_precip, validate_temperature


def detect_graphs_on_page(img: Image.Image, client: LLMClient, desired_kinds=None) -> Dict:
    data_url = pil_to_data_url(ensure_rgb(img))
    out = client.detect_regions(data_url)
    # If not all three kinds detected, retry with feedback
    kinds = {g.get("kind") for g in out.get("graphs", [])}
    expected = set(desired_kinds) if desired_kinds else {"wind", "precipitation", "temperature"}
    missing = expected - kinds
    if out.get("page_type") == "area_graphs" and missing:
        fb = f"Please return exactly three graphs including missing: {sorted(list(missing))}."
        out2 = client.detect_regions(data_url, feedback=fb)
        if out2.get("graphs"):
            return out2
    return out


def extract_graph_series(
    kind: str, crop: Image.Image, client: LLMClient
) -> Dict:
    crop_url = pil_to_data_url(ensure_rgb(crop))
    # First pass
    if kind == "wind":
        payload = client.extract_wind(crop_url)
        ok, fb = validate_wind(payload)
        if not ok:
            payload = client.extract_wind(crop_url, feedback=fb)
        return payload
    if kind == "precipitation":
        payload = client.extract_precip(crop_url)
        ok, fb = validate_precip(payload)
        if not ok:
            payload = client.extract_precip(crop_url, feedback=fb)
        return payload
    if kind == "temperature":
        payload = client.extract_temperature(crop_url)
        ok, fb = validate_temperature(payload)
        if not ok:
            payload = client.extract_temperature(crop_url, feedback=fb)
        return payload
    raise ValueError(f"Unknown graph kind: {kind}")


def series_to_rows(
    source_file: str,
    page_index: int,
    location: str,
    kind: str,
    payload: Dict,
) -> List[Dict]:
    rows: List[Dict] = []
    hours = payload.get("hours", [])

    if kind == "wind":
        for h in hours:
            rows.append(
                {
                    "SourceFile": os.path.basename(source_file),
                    "Page": page_index + 1,
                    "Location": location,
                    "Section": "Wind",
                    "Measurement": "Wind",
                    "MeasurementType": "Speed",
                    "Units": "mph",
                    "ForecastWindowStartLocal": "",
                    "HourLabel": h.get("hour_label"),
                    "HourIndex": h.get("hour_index"),
                    "TimestampLocal": "",
                    "ValueNumeric": h.get("wind_speed_mph"),
                    "ValueText": "",
                    "Notes": "",
                }
            )
            rows.append(
                {
                    "SourceFile": os.path.basename(source_file),
                    "Page": page_index + 1,
                    "Location": location,
                    "Section": "Wind",
                    "Measurement": "Wind",
                    "MeasurementType": "Gust",
                    "Units": "mph",
                    "ForecastWindowStartLocal": "",
                    "HourLabel": h.get("hour_label"),
                    "HourIndex": h.get("hour_index"),
                    "TimestampLocal": "",
                    "ValueNumeric": h.get("wind_gust_mph"),
                    "ValueText": "",
                    "Notes": "",
                }
            )
            rows.append(
                {
                    "SourceFile": os.path.basename(source_file),
                    "Page": page_index + 1,
                    "Location": location,
                    "Section": "Wind",
                    "Measurement": "Wind",
                    "MeasurementType": "Direction",
                    "Units": "",
                    "ForecastWindowStartLocal": "",
                    "HourLabel": h.get("hour_label"),
                    "HourIndex": h.get("hour_index"),
                    "TimestampLocal": "",
                    "ValueNumeric": "",
                    "ValueText": h.get("wind_direction"),
                    "Notes": "",
                }
            )

    elif kind == "precipitation":
        for h in hours:
            rows.append(
                {
                    "SourceFile": os.path.basename(source_file),
                    "Page": page_index + 1,
                    "Location": location,
                    "Section": "Precip",
                    "Measurement": "Precipitation",
                    "MeasurementType": "Rain",
                    "Units": "mm",
                    "ForecastWindowStartLocal": "",
                    "HourLabel": h.get("hour_label"),
                    "HourIndex": h.get("hour_index"),
                    "TimestampLocal": "",
                    "ValueNumeric": h.get("rain_mm"),
                    "ValueText": "",
                    "Notes": "",
                }
            )
            rows.append(
                {
                    "SourceFile": os.path.basename(source_file),
                    "Page": page_index + 1,
                    "Location": location,
                    "Section": "Precip",
                    "Measurement": "Precipitation",
                    "MeasurementType": "Snow",
                    "Units": "cm",
                    "ForecastWindowStartLocal": "",
                    "HourLabel": h.get("hour_label"),
                    "HourIndex": h.get("hour_index"),
                    "TimestampLocal": "",
                    "ValueNumeric": h.get("snow_cm"),
                    "ValueText": "",
                    "Notes": "",
                }
            )
            rows.append(
                {
                    "SourceFile": os.path.basename(source_file),
                    "Page": page_index + 1,
                    "Location": location,
                    "Section": "Precip",
                    "Measurement": "Precipitation",
                    "MeasurementType": "Type",
                    "Units": "",
                    "ForecastWindowStartLocal": "",
                    "HourLabel": h.get("hour_label"),
                    "HourIndex": h.get("hour_index"),
                    "TimestampLocal": "",
                    "ValueNumeric": "",
                    "ValueText": h.get("precip_type"),
                    "Notes": "",
                }
            )

    elif kind == "temperature":
        for h in hours:
            rows.append(
                {
                    "SourceFile": os.path.basename(source_file),
                    "Page": page_index + 1,
                    "Location": location,
                    "Section": "Temperature",
                    "Measurement": "Temperature",
                    "MeasurementType": "AirTemp_C",
                    "Units": "degC",
                    "ForecastWindowStartLocal": "",
                    "HourLabel": h.get("hour_label"),
                    "HourIndex": h.get("hour_index"),
                    "TimestampLocal": "",
                    "ValueNumeric": h.get("air_temp_c"),
                    "ValueText": "",
                    "Notes": "",
                }
            )
            rows.append(
                {
                    "SourceFile": os.path.basename(source_file),
                    "Page": page_index + 1,
                    "Location": location,
                    "Section": "Temperature",
                    "Measurement": "FreezingLevel",
                    "MeasurementType": "FreezingLevel_m",
                    "Units": "m",
                    "ForecastWindowStartLocal": "",
                    "HourLabel": h.get("hour_label"),
                    "HourIndex": h.get("hour_index"),
                    "TimestampLocal": "",
                    "ValueNumeric": h.get("freezing_level_m"),
                    "ValueText": "",
                    "Notes": "",
                }
            )
            rows.append(
                {
                    "SourceFile": os.path.basename(source_file),
                    "Page": page_index + 1,
                    "Location": location,
                    "Section": "Temperature",
                    "Measurement": "WetBulbFreezingLevel",
                    "MeasurementType": "WBFL_m",
                    "Units": "m",
                    "ForecastWindowStartLocal": "",
                    "HourLabel": h.get("hour_label"),
                    "HourIndex": h.get("hour_index"),
                    "TimestampLocal": "",
                    "ValueNumeric": h.get("wet_bulb_freezing_level_m"),
                    "ValueText": "",
                    "Notes": "",
                }
            )
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    return rows


def rows_to_dataframe(rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)

import os
from typing import Dict, List, Tuple, Optional

import pandas as pd
from PIL import Image

from .llm_client import LLMClient
from .utils import pil_to_data_url, crop_normalized_box, ensure_rgb, resize_max_side
from .metrics import time_block, log_event
from .validate import validate_wind, validate_precip, validate_temperature


def detect_graphs_on_page(img: Image.Image, client: LLMClient, desired_kinds=None, metrics_path: Optional[str] = None, page_num: Optional[int] = None, source_file: Optional[str] = None, max_page_px: Optional[int] = None) -> Dict:
    page_img = ensure_rgb(img)
    if max_page_px:
        page_img = resize_max_side(page_img, max_page_px)
    data_url = pil_to_data_url(page_img, format="JPEG", quality=70)
    with time_block("detect_regions", metrics_path, page=page_num, file=source_file):
        out = client.detect_regions(data_url)
    # If not all three kinds detected, retry with feedback
    kinds = {g.get("kind") for g in out.get("graphs", [])}
    expected = set(desired_kinds) if desired_kinds else {"wind", "precipitation", "temperature"}
    missing = expected - kinds
    if out.get("page_type") == "area_graphs" and missing:
        fb = f"Please return exactly three graphs including missing: {sorted(list(missing))}."
        with time_block("detect_regions_retry", metrics_path, page=page_num, file=source_file, missing=list(missing)):
            out2 = client.detect_regions(data_url, feedback=fb)
        if out2.get("graphs"):
            return out2
    return out


def extract_graph_series(
    kind: str, crop: Image.Image, client: LLMClient, metrics_path: Optional[str] = None, page_num: Optional[int] = None, source_file: Optional[str] = None, max_crop_px: Optional[int] = None
) -> Tuple[Dict, Dict]:
    crop_img = ensure_rgb(crop)
    if max_crop_px:
        crop_img = resize_max_side(crop_img, max_crop_px)
    crop_url = pil_to_data_url(crop_img, format="JPEG", quality=80)
    # First pass
    if kind == "wind":
        with time_block("extract_wind", metrics_path, page=page_num, file=source_file):
            payload = client.extract_wind(crop_url)
        ok, fb = validate_wind(payload)
        if not ok:
            with time_block("extract_wind_retry", metrics_path, page=page_num, file=source_file):
                payload = client.extract_wind(crop_url, feedback=fb)
        return payload, {"retry": not ok}
    if kind == "precipitation":
        with time_block("extract_precip", metrics_path, page=page_num, file=source_file):
            payload = client.extract_precip(crop_url)
        ok, fb = validate_precip(payload)
        if not ok:
            with time_block("extract_precip_retry", metrics_path, page=page_num, file=source_file):
                payload = client.extract_precip(crop_url, feedback=fb)
        return payload, {"retry": not ok}
    if kind == "temperature":
        with time_block("extract_temp", metrics_path, page=page_num, file=source_file):
            payload = client.extract_temperature(crop_url)
        ok, fb = validate_temperature(payload)
        if not ok:
            with time_block("extract_temp_retry", metrics_path, page=page_num, file=source_file):
                payload = client.extract_temperature(crop_url, feedback=fb)
        return payload, {"retry": not ok}
    raise ValueError(f"Unknown graph kind: {kind}")


def extract_all_series(
    crops_by_kind: Dict[str, Image.Image],
    client: LLMClient,
    metrics_path: Optional[str] = None,
    page_num: Optional[int] = None,
    source_file: Optional[str] = None,
    max_crop_px: Optional[int] = None,
) -> Tuple[Dict[str, Dict], Dict]:
    # Ensure we have all three kinds
    required = ["wind", "precipitation", "temperature"]
    if any(k not in crops_by_kind for k in required):
        raise ValueError("extract_all_series requires wind, precipitation, temperature crops")

    crop_urls: Dict[str, str] = {}
    for k, im in crops_by_kind.items():
        im_proc = ensure_rgb(im)
        if max_crop_px:
            im_proc = resize_max_side(im_proc, max_crop_px)
        crop_urls[k] = pil_to_data_url(im_proc, format="JPEG", quality=80)

    with time_block("extract_all", metrics_path, page=page_num, file=source_file):
        payload = client.extract_all(crop_urls)

    # Validate and optionally retry once with combined feedback
    fb_parts = []
    ok_w, fb_w = validate_wind(payload.get("wind", {}))
    if not ok_w:
        fb_parts.append(f"wind: {fb_w}")
    ok_p, fb_p = validate_precip(payload.get("precipitation", {}))
    if not ok_p:
        fb_parts.append(f"precipitation: {fb_p}")
    ok_t, fb_t = validate_temperature(payload.get("temperature", {}))
    if not ok_t:
        fb_parts.append(f"temperature: {fb_t}")

    retried = False
    if fb_parts:
        retried = True
        feedback = "; ".join(fb_parts)
        with time_block("extract_all_retry", metrics_path, page=page_num, file=source_file):
            payload = client.extract_all(crop_urls, feedback=feedback)

    return payload, {"retry": retried}


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

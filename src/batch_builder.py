import json
import os
import uuid
from typing import Dict, List

from pydantic import BaseModel

from .models import RegionDetection, CombinedSeries, WindSeries, PrecipSeries, TempSeries
from .utils import pil_to_data_url, ensure_rgb, resize_max_side


def _json_schema_response_format(name: str, model: type[BaseModel]) -> Dict:
    def _strictify(s: Dict) -> Dict:
        t = s.get("type")
        if t == "object":
            # Root or nested object: enforce additionalProperties: false
            s.setdefault("additionalProperties", False)
            # Recurse into properties
            props = s.get("properties", {})
            for k, v in list(props.items()):
                if isinstance(v, dict):
                    # remove nullable anyOf -> non-null type
                    if "anyOf" in v and isinstance(v["anyOf"], list):
                        non_null = [it for it in v["anyOf"] if not (isinstance(it, dict) and it.get("type") == "null")]
                        if non_null:
                            v = non_null[0]
                        v.pop("anyOf", None)
                        v.pop("default", None)
                    props[k] = _strictify(v)
            # Required must include all property keys
            keys = list(props.keys())
            if keys:
                s["required"] = keys
            # Recurse into items for arrays
            items = s.get("items")
            if isinstance(items, dict):
                s["items"] = _strictify(items)
            elif isinstance(items, list):
                s["items"] = [_strictify(it) if isinstance(it, dict) else it for it in items]
        elif t == "array":
            items = s.get("items")
            if isinstance(items, dict):
                s["items"] = _strictify(items)
            elif isinstance(items, list):
                s["items"] = [_strictify(it) if isinstance(it, dict) else it for it in items]
        return s

    base = model.model_json_schema()
    # Recurse into $defs
    defs = base.get("$defs") or base.get("definitions")
    if isinstance(defs, dict):
        for k, v in list(defs.items()):
            if isinstance(v, dict):
                defs[k] = _strictify(v)
    strict = _strictify(base)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": strict,
            "strict": True,
        },
    }


def build_detect_regions_request(model_name: str, page_img, prompt: str | None = None, max_side: int = 1200) -> Dict:
    prompt = prompt or (
        "You are locating three graphs on a forecast page. Return normalized bounding boxes [x,y,w,h] in 0..1 "
        "for each graph and classify each as wind, precipitation, or temperature. Mark page_type as 'area_graphs' "
        "only if you see three 24-hour graphs 18→17. Include location if obvious from titles."
    )
    page = ensure_rgb(page_img)
    page = resize_max_side(page, max_side)
    img_url = pil_to_data_url(page, format="JPEG", quality=70)

    body = {
        "model": model_name,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt + "\nReturn JSON only with keys: page_type, location, graphs (array of {kind,bbox})."},
                    {"type": "input_image", "image_url": img_url},
                ],
            }
        ],
        "max_output_tokens": 8000,
        "reasoning": {"effort": "low"},
        "response_format": _json_schema_response_format("graph_regions_schema", RegionDetection),
    }
    return {
        "custom_id": f"detect_regions_{uuid.uuid4().hex}",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def build_extract_all_request(model_name: str, crops_by_kind: Dict[str, any], prompt: str | None = None, max_side: int = 900) -> Dict:
    prompt = prompt or (
        "Three images follow in order: (1) wind graph, (2) precipitation graph, (3) temperature graph. "
        "Extract 24 hourly series for each graph and return a single object keyed by wind, precipitation, temperature."
    )
    order = ["wind", "precipitation", "temperature"]
    content = [{"type": "input_text", "text": prompt}]
    for k in order:
        im = ensure_rgb(crops_by_kind[k])
        im = resize_max_side(im, max_side)
        content.append({"type": "input_image", "image_url": pil_to_data_url(im, format="JPEG", quality=80)})

    content[0]["text"] += (
        "\nUse the same location string copied exactly from the shared graph title (empty string if unreadable). "
        "Ensure every series contains 24 entries for hour_index 0..23 (hours 18→17). "
        "For wind, output integer mph speeds/gusts aligned with the axis ticks and directions using only the 16-point compass list. "
        "For precipitation, report rain_mm as whole millimetres, snow_cm with one decimal, precip_type exactly as printed above each hour, and 0 values when no bar is present. "
        "For temperature, report air_temp_c to one decimal and altitude values to the nearest 10 metres following the right-hand axis."
    )
    body = {
        "model": model_name,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 18000,
        "reasoning": {"effort": "low"},
        "response_format": _json_schema_response_format("combined_series_schema", CombinedSeries),
    }
    return {
        "custom_id": f"extract_all_{uuid.uuid4().hex}",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def build_extract_wind_request(model_name: str, crop_img, prompt: str | None = None, max_side: int = 900) -> Dict:
    prompt = prompt or (
        "Extract 24 hourly wind series: speed (blue, mph), gust (pink, mph), direction (text). "
        "Hours span 18→17, so produce exactly 24 entries with hour_label and hour_index 0..23 in order. "
        "Use only these compass directions: N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW. "
        "Report wind_speed_mph and wind_gust_mph as integer mph values that line up with the axis ticks—never treat above-axis lines as near zero. "
        "If values are unclear, estimate from neighbouring points and the gridlines. "
        "Include 'location' exactly as the graph title text (empty string only if unreadable); do not use page footers or add qualifiers."
    )
    im = ensure_rgb(crop_img)
    im = resize_max_side(im, max_side)
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": pil_to_data_url(im, format="JPEG", quality=80)},
    ]
    content[0]["text"] += "\nReturn JSON only with keys: location, hours (24 items) and honour the schema's field names."
    body = {
        "model": model_name,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 16000,
        "reasoning": {"effort": "low"},
        "response_format": _json_schema_response_format("wind_series_schema", WindSeries),
    }
    return {
        "custom_id": f"extract_wind_{uuid.uuid4().hex}",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def build_extract_precip_request(model_name: str, crop_img, prompt: str | None = None, max_side: int = 900) -> Dict:
    prompt = prompt or (
        "Extract 24 hourly precipitation series: rain (blue bars, mm), snow (white bars, cm), precip_type (text). "
        "Hours cover 18→17, so output exactly 24 ordered entries with hour_index 0..23. "
        "If no bar appears, set rain_mm and snow_cm to 0 for that hour. "
        "Report rain_mm as whole millimetres and snow_cm with one decimal place, keeping values non-negative and aligned with the axis scale. "
        "Return precip_type exactly as printed above each hour without normalisation. "
        "Include 'location' exactly as the graph title text (empty string only if unreadable); avoid page footers and qualifiers."
    )
    im = ensure_rgb(crop_img)
    im = resize_max_side(im, max_side)
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": pil_to_data_url(im, format="JPEG", quality=80)},
    ]
    content[0]["text"] += "\nReturn JSON only with keys: location, hours (24 items) and honour the schema's field names."
    body = {
        "model": model_name,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 16000,
        "reasoning": {"effort": "low"},
        "response_format": _json_schema_response_format("precip_series_schema", PrecipSeries),
    }
    return {
        "custom_id": f"extract_precip_{uuid.uuid4().hex}",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def build_extract_temperature_request(model_name: str, crop_img, prompt: str | None = None, max_side: int = 900) -> Dict:
    prompt = prompt or (
        "Extract 24 hourly temperature/altitude series: air_temp_c (blue, left axis °C), freezing_level_m (pink, right axis m), "
        "wet_bulb_freezing_level_m (yellow, right axis m). Hours span 18→17, so return exactly 24 ordered entries with hour_index 0..23. "
        "Report air_temp_c to one decimal place, and altitude values to the nearest 10 metres following the axis markings. "
        "If a point is unclear, estimate from the surrounding curve and gridlines instead of dropping the hour. "
        "Include 'location' exactly as the graph title text (empty string only if unreadable); do not use page footers or add qualifiers."
    )
    im = ensure_rgb(crop_img)
    im = resize_max_side(im, max_side)
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": pil_to_data_url(im, format="JPEG", quality=80)},
    ]
    content[0]["text"] += "\nReturn JSON only with keys: location, hours (24 items) and honour the schema's field names."
    body = {
        "model": model_name,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 16000,
        "reasoning": {"effort": "low"},
        "response_format": _json_schema_response_format("temp_series_schema", TempSeries),
    }
    return {
        "custom_id": f"extract_temp_{uuid.uuid4().hex}",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def write_jsonl(items: List[Dict], path: str) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")

import uuid
from typing import Dict, List

from pydantic import BaseModel

from .models import RegionDetection, CombinedSeries, WindSeries, PrecipSeries, TempSeries
from .utils import pil_to_data_url, ensure_rgb, resize_max_side


def _text_format_json_schema(name: str, model: type[BaseModel]) -> Dict:
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
        "format": {
            "type": "json_schema",
            "name": name,
            "schema": strict,
            "strict": True,
        }
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
        "text": _text_format_json_schema("graph_regions_schema", RegionDetection),
    }
    return {
        "custom_id": f"detect_regions_{uuid.uuid4().hex}",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def build_extract_all_request(model_name: str, crops_by_kind: Dict[str, any], prompt: str | None = None, max_side: int = 900) -> Dict:
    prompt = prompt or (
        "Three images follow in order: (1) wind graph, (2) precipitation graph, (3) temperature graph. Extract 24 hourly series for each."
    )
    order = ["wind", "precipitation", "temperature"]
    content = [{"type": "input_text", "text": prompt}]
    for k in order:
        im = ensure_rgb(crops_by_kind[k])
        im = resize_max_side(im, max_side)
        content.append({"type": "input_image", "image_url": pil_to_data_url(im, format="JPEG", quality=80)})

    # Add strict JSON instruction
    content[0]["text"] += "\nReturn JSON only with keys: wind, precipitation, temperature."
    body = {
        "model": model_name,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 8000,
        "reasoning": {"effort": "low"},
        "text": _text_format_json_schema("combined_series_schema", CombinedSeries),
    }
    return {
        "custom_id": f"extract_all_{uuid.uuid4().hex}",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def build_extract_wind_request(model_name: str, crop_img, prompt: str | None = None, max_side: int = 900) -> Dict:
    prompt = prompt or (
        "Extract 24 hourly wind series: speed (blue, mph), gust (pink, mph), direction (text)."
        " Hours 18→17; return exactly 24 entries with hour_label, hour_index (0..23), wind_speed_mph, wind_gust_mph, wind_direction."
        " Include 'location' exactly as the graph title text; do not use page footers or add qualifiers."
    )
    im = ensure_rgb(crop_img)
    im = resize_max_side(im, max_side)
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": pil_to_data_url(im, format="JPEG", quality=80)},
    ]
    content[0]["text"] += "\nReturn JSON only with keys: location, hours (24 items)."
    body = {
        "model": model_name,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 8000,
        "reasoning": {"effort": "low"},
        "text": _text_format_json_schema("wind_series_schema", WindSeries),
    }
    return {
        "custom_id": f"extract_wind_{uuid.uuid4().hex}",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def build_extract_precip_request(model_name: str, crop_img, prompt: str | None = None, max_side: int = 900) -> Dict:
    prompt = prompt or (
        "Extract 24 hourly precipitation series: rain (blue bars, mm), snow (white bars, cm), precip_type (text)."
        " If no bar, numeric = 0. Hours 18→17; return exactly 24 entries with hour_label and hour_index (0..23)."
        " Include 'location' exactly as the graph title text; do not use page footers or add qualifiers."
    )
    im = ensure_rgb(crop_img)
    im = resize_max_side(im, max_side)
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": pil_to_data_url(im, format="JPEG", quality=80)},
    ]
    content[0]["text"] += "\nReturn JSON only with keys: location, hours (24 items)."
    body = {
        "model": model_name,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 8000,
        "reasoning": {"effort": "low"},
        "text": _text_format_json_schema("precip_series_schema", PrecipSeries),
    }
    return {
        "custom_id": f"extract_precip_{uuid.uuid4().hex}",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def build_extract_temperature_request(model_name: str, crop_img, prompt: str | None = None, max_side: int = 900) -> Dict:
    prompt = prompt or (
        "Extract 24 hourly temperature/altitude series: air_temp_c (blue, left degC), freezing_level_m (pink, right m),"
        " wet_bulb_freezing_level_m (yellow, right m). Hours 18→17; return exactly 24 entries with hour_label and hour_index (0..23)."
        " Include 'location' exactly as the graph title text; do not use page footers or add qualifiers."
    )
    im = ensure_rgb(crop_img)
    im = resize_max_side(im, max_side)
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": pil_to_data_url(im, format="JPEG", quality=80)},
    ]
    content[0]["text"] += "\nReturn JSON only with keys: location, hours (24 items)."
    body = {
        "model": model_name,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 8000,
        "reasoning": {"effort": "low"},
        "text": _text_format_json_schema("temp_series_schema", TempSeries),
    }
    return {
        "custom_id": f"extract_temp_{uuid.uuid4().hex}",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def write_jsonl(items: List[Dict], path: str) -> None:
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

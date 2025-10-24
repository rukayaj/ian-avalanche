from typing import Dict, Any


def region_detection_schema() -> Dict[str, Any]:
    return {
        "name": "graph_regions_schema",
        "schema": {
            "type": "object",
            "properties": {
                "page_type": {"type": "string", "enum": ["area_graphs", "other"]},
                "location": {"type": "string"},
                "graphs": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["wind", "precipitation", "temperature"],
                            },
                            "bbox": {
                                "type": "array",
                                "minItems": 4,
                                "maxItems": 4,
                                "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "description": "Normalized [x, y, w, h]",
                            },
                        },
                        "required": ["kind", "bbox"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["page_type", "graphs"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def wind_schema() -> Dict[str, Any]:
    return {
        "name": "wind_series_schema",
        "schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "hours": {
                    "type": "array",
                    "minItems": 24,
                    "maxItems": 24,
                    "items": {
                        "type": "object",
                        "properties": {
                            "hour_label": {"type": "string"},
                            "hour_index": {"type": "integer", "minimum": 0, "maximum": 23},
                            "wind_speed_mph": {"type": "number"},
                            "wind_gust_mph": {"type": "number"},
                            "wind_direction": {
                                "type": "string",
                                "enum": [
                                    "N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"
                                ],
                            },
                        },
                        "required": [
                            "hour_label","hour_index","wind_speed_mph","wind_gust_mph","wind_direction"
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["hours"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def precip_schema() -> Dict[str, Any]:
    return {
        "name": "precip_series_schema",
        "schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "hours": {
                    "type": "array",
                    "minItems": 24,
                    "maxItems": 24,
                    "items": {
                        "type": "object",
                        "properties": {
                            "hour_label": {"type": "string"},
                            "hour_index": {"type": "integer", "minimum": 0, "maximum": 23},
                            "rain_mm": {"type": "number"},
                            "snow_cm": {"type": "number"},
                            "precip_type": {"type": "string"},
                        },
                        "required": ["hour_label","hour_index","rain_mm","snow_cm","precip_type"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["hours"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def temperature_schema() -> Dict[str, Any]:
    return {
        "name": "temp_series_schema",
        "schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "hours": {
                    "type": "array",
                    "minItems": 24,
                    "maxItems": 24,
                    "items": {
                        "type": "object",
                        "properties": {
                            "hour_label": {"type": "string"},
                            "hour_index": {"type": "integer", "minimum": 0, "maximum": 23},
                            "air_temp_c": {"type": "number"},
                            "freezing_level_m": {"type": "number"},
                            "wet_bulb_freezing_level_m": {"type": "number"},
                        },
                        "required": [
                            "hour_label","hour_index","air_temp_c","freezing_level_m","wet_bulb_freezing_level_m"
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["hours"],
            "additionalProperties": False,
        },
        "strict": True,
    }

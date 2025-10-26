import math
import re
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

LOCATION_PREFIXES = (
    "Wind - ",
    "Precipitation - ",
    "Weather and Precipitation - ",
    "Weather & Precipitation - ",
    "Temperature - ",
)
LOCATION_PLACEHOLDER_VALUES = {"", "unknown", "n/a", "na", "none"}
PRECIP_TYPE_ALIASES = {
    "None": "No Precip",
    "none": "No Precip",
    "No precip": "No Precip",
    "no precip": "No Precip",
}
DEFAULT_NUMERIC_TOLERANCES = {
    ("Wind", "Speed"): 3.0,
    ("Wind", "Gust"): 5.0,
    ("Precip", "Rain"): 1.0,
    ("Precip", "Snow"): 0.6,
    ("Temperature", "AirTemp_C"): 0.7,
    ("Temperature", "FreezingLevel_m"): 80.0,
    ("Temperature", "WBFL_m"): 80.0,
}


def normalize_location_string(value: Optional[str]) -> str:
    if value is None:
        return ""
    stripped = value.strip()
    if stripped.lower() in LOCATION_PLACEHOLDER_VALUES:
        return ""
    for prefix in LOCATION_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    stripped = stripped.strip()
    match = re.search(r"\((\d+)\s*m\)$", stripped)
    if match:
        stripped = re.sub(r"\((\d+)\s*m\)$", lambda m: f"({m.group(1)} metres)", stripped)
    return stripped


def normalize_precip_type(value: Optional[str]) -> str:
    if value is None:
        return ""
    stripped = value.strip()
    if not stripped:
        return ""
    return PRECIP_TYPE_ALIASES.get(stripped, PRECIP_TYPE_ALIASES.get(stripped.lower(), stripped))


def normalize_text_value(section: str, measurement_type: str, value: Optional[str]) -> str:
    raw = (value or "").strip()
    if section == "Precip" and measurement_type == "Type":
        return normalize_precip_type(raw)
    return raw


def format_numeric_value(section: str, measurement_type: str, value: float) -> Any:
    if value is None:
        return ""
    if section == "Wind":
        return int(round(value))
    if section == "Precip":
        return round(value, 1)
    if section == "Temperature":
        if measurement_type == "AirTemp_C":
            return round(value, 1)
        return int(round(value, -1))
    return round(value, 2)


def harmonize_locations(final_rows: Dict[Tuple[Any, ...], Dict[str, Any]]) -> None:
    location_counts: Dict[Tuple[Any, ...], Counter] = {}
    for row in final_rows.values():
        graph_key = (row.get("SourceFile"), row.get("Page"), row.get("Section"))
        loc = normalize_location_string(row.get("Location") or "")
        if loc:
            location_counts.setdefault(graph_key, Counter()).update([loc])
    for row in final_rows.values():
        graph_key = (row.get("SourceFile"), row.get("Page"), row.get("Section"))
        counter = location_counts.get(graph_key)
        if counter:
            row["Location"] = counter.most_common(1)[0][0]
        else:
            row["Location"] = ""


def aggregate_runs(
    run_maps: List[Dict[Tuple[Any, ...], Dict[str, Any]]],
    tolerances: Dict[Tuple[str, str], float],
    default_numeric_tolerance: float = 1.0,
) -> Tuple[Dict[Tuple[Any, ...], Dict[str, Any]], List[Dict[str, Any]]]:
    final_rows: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    disagreements: List[Dict[str, Any]] = []
    if not run_maps:
        return final_rows, disagreements

    combined_keys = set()
    for run_map in run_maps:
        combined_keys.update(run_map.keys())

    for key in sorted(combined_keys):
        rows = [run_map.get(key) for run_map in run_maps]
        available_rows = [r for r in rows if r]
        if not available_rows:
            continue
        sample = dict(available_rows[0])
        section = sample.get("Section")
        measurement = sample.get("Measurement")
        measurement_type = sample.get("MeasurementType")
        graph_key = (sample.get("SourceFile"), sample.get("Page"), section)
        issues: List[Dict[str, Any]] = []

        missing_runs = [idx for idx, row in enumerate(rows) if row is None]
        if missing_runs:
            issues.append({"reason": "missing_runs", "runs": missing_runs})

        location_candidates = [
            normalize_location_string(row.get("Location") or "") for row in available_rows if (row.get("Location") or "").strip()
        ]
        final_location = ""
        if location_candidates:
            counts = Counter(location_candidates)
            final_location = counts.most_common(1)[0][0]
            if len(counts) > 1:
                issues.append({"reason": "location_mismatch", "locations": list(counts.keys())})

        numeric_values: List[float] = []
        numeric_runs: List[int] = []
        text_values: List[str] = []
        text_runs: List[int] = []
        for idx, row in enumerate(rows):
            if not row:
                continue
            val_text = row.get("ValueText")
            if isinstance(val_text, float) and math.isnan(val_text):
                val_text = ""
            val_text = normalize_text_value(section, measurement_type, val_text)
            if val_text:
                text_values.append(val_text)
                text_runs.append(idx)
            val_num = row.get("ValueNumeric")
            if val_num not in ("", None):
                try:
                    numeric_values.append(float(val_num))
                    numeric_runs.append(idx)
                except (TypeError, ValueError):
                    continue

        final_numeric = ""
        final_text = ""

        tolerance = tolerances.get((section, measurement_type))
        if tolerance is None:
            tolerance = tolerances.get((section, measurement), default_numeric_tolerance)

        if text_values:
            counts = Counter(text_values)
            final_text = counts.most_common(1)[0][0]
            if len(counts) > 1:
                issues.append({"reason": "text_mismatch", "values": list(counts.keys()), "runs": text_runs})
        elif numeric_values:
            median_val = statistics.median(numeric_values)
            final_numeric = format_numeric_value(section, measurement_type, median_val)
            if len(numeric_values) >= 2 and tolerance is not None:
                if max(numeric_values) - min(numeric_values) > tolerance + 1e-6:
                    issues.append(
                        {
                            "reason": "numeric_variation",
                            "values": [float(v) for v in numeric_values],
                            "runs": numeric_runs,
                            "tolerance": tolerance,
                        }
                    )
        else:
            final_numeric = sample.get("ValueNumeric") or ""
            final_text = normalize_text_value(section, measurement_type, sample.get("ValueText"))
            issues.append({"reason": "missing_values"})

        merged_row = dict(sample)
        merged_row["Location"] = final_location
        merged_row["ValueNumeric"] = final_numeric
        merged_row["ValueText"] = normalize_text_value(section, measurement_type, final_text)
        final_rows[key] = merged_row

        if issues:
            disagreements.append(
                {
                    "key": {
                        "SourceFile": sample.get("SourceFile"),
                        "Page": sample.get("Page"),
                        "Section": section,
                        "Measurement": measurement,
                        "MeasurementType": measurement_type,
                        "HourIndex": sample.get("HourIndex"),
                        "HourLabel": sample.get("HourLabel"),
                    },
                    "graph": {
                        "SourceFile": graph_key[0],
                        "Page": graph_key[1],
                        "Section": graph_key[2],
                    },
                    "issues": issues,
                }
            )

    harmonize_locations(final_rows)
    return final_rows, disagreements

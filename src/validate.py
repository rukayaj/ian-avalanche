from typing import Dict, Tuple


def validate_wind(payload: Dict) -> Tuple[bool, str]:
    hours = payload.get("hours", [])
    if len(hours) != 24:
        return False, f"Expected 24 hours, got {len(hours)}"
    violations = 0
    for h in hours:
        sp = h.get("wind_speed_mph")
        gu = h.get("wind_gust_mph")
        if sp is None or gu is None:
            return False, "Missing wind_speed_mph or wind_gust_mph"
        if gu + 1e-6 < sp:
            violations += 1
    if violations > 4:  # tolerate a few due to reading noise
        return False, (
            f"Gust < speed in {violations} hours; gusts must stay ≥ speeds and both should follow the plotted axis ticks "
            " (early hours typically sit around 10–25 mph)."
        )
    return True, ""


def validate_precip(payload: Dict) -> Tuple[bool, str]:
    hours = payload.get("hours", [])
    if len(hours) != 24:
        return False, f"Expected 24 hours, got {len(hours)}"
    for h in hours:
        rain = h.get("rain_mm")
        snow = h.get("snow_cm")
        if rain is None or snow is None:
            return False, "Missing rain_mm or snow_cm; every hour needs both values."
        if rain < 0 or snow < 0:
            return False, "Negative precipitation values; use 0 when no bar is present."
    return True, ""


def validate_temperature(payload: Dict) -> Tuple[bool, str]:
    hours = payload.get("hours", [])
    if len(hours) != 24:
        return False, f"Expected 24 hours, got {len(hours)}"
    # Ensure required numeric fields exist
    for h in hours:
        if (
            h.get("air_temp_c") is None
            or h.get("freezing_level_m") is None
            or h.get("wet_bulb_freezing_level_m") is None
        ):
            return False, (
                "Missing one of air_temp_c/freezing_level_m/wet_bulb_freezing_level_m; estimate from axis ticks instead of leaving blanks."
            )
    return True, ""

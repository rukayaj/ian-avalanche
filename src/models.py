from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class GraphRegion(BaseModel):
    kind: Literal["wind", "precipitation", "temperature"]
    bbox: List[float] = Field(..., min_length=4, max_length=4)


class RegionDetection(BaseModel):
    page_type: Literal["area_graphs", "other"]
    location: Optional[str] = None
    graphs: List[GraphRegion]


class WindHour(BaseModel):
    hour_label: str
    hour_index: int
    wind_speed_mph: float
    wind_gust_mph: float
    wind_direction: Literal[
        "N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"
    ]


class WindSeries(BaseModel):
    location: Optional[str] = None
    hours: List[WindHour]

    @field_validator("hours")
    @classmethod
    def _validate_24(cls, v: List[WindHour]):
        if len(v) != 24:
            raise ValueError("Expected 24 hourly entries for wind")
        return v


class PrecipHour(BaseModel):
    hour_label: str
    hour_index: int
    rain_mm: float
    snow_cm: float
    precip_type: str


class PrecipSeries(BaseModel):
    location: Optional[str] = None
    hours: List[PrecipHour]

    @field_validator("hours")
    @classmethod
    def _validate_24(cls, v: List[PrecipHour]):
        if len(v) != 24:
            raise ValueError("Expected 24 hourly entries for precipitation")
        return v


class TempHour(BaseModel):
    hour_label: str
    hour_index: int
    air_temp_c: float
    freezing_level_m: float
    wet_bulb_freezing_level_m: float


class TempSeries(BaseModel):
    location: Optional[str] = None
    hours: List[TempHour]

    @field_validator("hours")
    @classmethod
    def _validate_24(cls, v: List[TempHour]):
        if len(v) != 24:
            raise ValueError("Expected 24 hourly entries for temperature")
        return v


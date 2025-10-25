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
    hour_label: str = Field(pattern=r"^(18|19|2[0-3]|0[0-9]|1[0-7])$")
    hour_index: int = Field(ge=0, le=23)
    wind_speed_mph: float = Field(ge=0, le=150)
    wind_gust_mph: float = Field(ge=0, le=200)
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
    hour_label: str = Field(pattern=r"^(18|19|2[0-3]|0[0-9]|1[0-7])$")
    hour_index: int = Field(ge=0, le=23)
    rain_mm: float = Field(ge=0, le=100)
    snow_cm: float = Field(ge=0, le=100)
    precip_type: Literal[
        "None","No Precip","Dry","Rain","Snow","Sleet","Hail","Freezing Rain","Graupel","Mixed",
        "Rain/Snow","Snow Showers","Rain Showers","Drizzle","Showers","Thunderstorm"
    ]


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
    hour_label: str = Field(pattern=r"^(18|19|2[0-3]|0[0-9]|1[0-7])$")
    hour_index: int = Field(ge=0, le=23)
    air_temp_c: float = Field(ge=-50, le=20)
    freezing_level_m: float = Field(ge=0, le=6000)
    wet_bulb_freezing_level_m: float = Field(ge=0, le=6000)


class TempSeries(BaseModel):
    location: Optional[str] = None
    hours: List[TempHour]

    @field_validator("hours")
    @classmethod
    def _validate_24(cls, v: List[TempHour]):
        if len(v) != 24:
            raise ValueError("Expected 24 hourly entries for temperature")
        return v


class CombinedSeries(BaseModel):
    location: Optional[str] = None
    wind: WindSeries
    precipitation: PrecipSeries
    temperature: TempSeries

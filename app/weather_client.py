"""
Fetches historical daily weather for Sydney from the Open-Meteo Historical Weather API
and aggregates it to daily granularity, using the exact same logic as the experimentation
notebooks (00_data_acquisition.ipynb) to guarantee train/serve consistency.
"""
import datetime as dt

import pandas as pd
import requests

LATITUDE = -33.8688
LONGITUDE = 151.2093
TIMEZONE = "Australia/Sydney"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "precipitation",
    "snowfall",
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class WeatherFetchError(Exception):
    """Raised when the Open-Meteo API cannot supply the data required for a prediction."""


def fetch_daily_weather_window(end_date: dt.date, days_before: int) -> pd.DataFrame:
    """
    Fetch daily-aggregated Sydney weather covering [end_date - days_before, end_date],
    inclusive. Raises WeatherFetchError if the upstream API fails or returns no data
    (e.g. because end_date is too recent for the historical archive to have processed
    yet - Open-Meteo's reanalysis data typically lags a few days behind real-time).
    """
    start_date = end_date - dt.timedelta(days=days_before)

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }

    try:
        resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as e:
        raise WeatherFetchError(f"Failed to reach Open-Meteo API: {e}") from e

    if "hourly" not in payload:
        raise WeatherFetchError(
            "Open-Meteo API returned no hourly data for the requested window. "
            "This can happen if the date is too recent for historical data to be "
            "available yet (Open-Meteo's archive typically lags a few days behind "
            "real time), or if the date is invalid."
        )

    hourly_df = pd.DataFrame(payload["hourly"])
    hourly_df["time"] = pd.to_datetime(hourly_df["time"])

    hourly_df["date"] = hourly_df["time"].dt.date
    daily_df = hourly_df.groupby("date").agg(
        temperature_2m=("temperature_2m", "mean"),
        relative_humidity_2m=("relative_humidity_2m", "mean"),
        wind_speed_10m=("wind_speed_10m", "mean"),
        cloud_cover=("cloud_cover", "mean"),
        precipitation=("precipitation", "sum"),
        wind_gusts_10m=("wind_gusts_10m", "max"),
        snowfall=("snowfall", "sum"),
    ).reset_index()

    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    if daily_df.empty or daily_df["temperature_2m"].isnull().all():
        raise WeatherFetchError(
            "No usable weather data was returned for the requested date range."
        )

    return daily_df
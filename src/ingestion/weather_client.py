"""
Weather covariate fetcher using Open-Meteo (free, no API key).
Fetches hourly temperature, wind speed, cloud cover, and solar radiation
for a given lat/lon, aligned to the electricity load timestamps.
"""

import logging
import requests
import pandas as pd
from src.utils.config import DATA_RAW

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Representative coordinates for each region
REGION_COORDS = {
    "US48": (38.0, -97.0),   # geographic center of lower 48
    "CAL":  (36.7, -119.7),  # Fresno (central CA)
    "TEX":  (31.1, -97.9),   # Waco (central TX)
    "MIDA": (39.9, -75.2),   # Philadelphia
    "NE":   (42.4, -71.1),   # Boston
    "NW":   (47.6, -122.3),  # Seattle
    "SE":   (33.7, -84.4),   # Atlanta
    "SW":   (33.4, -112.1),  # Phoenix
    "TEN":  (36.2, -86.8),   # Nashville
    "CAR":  (35.2, -80.8),   # Charlotte
    "DE":   (52.5, 13.4),    # Berlin
    "FR":   (48.9, 2.3),     # Paris
    "GB":   (51.5, -0.1),    # London
}

WEATHER_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
    "relative_humidity_2m",
]


def fetch_weather(
    region: str = "US48",
    start: str = "2019-01-01",
    end: str = "2024-01-01",
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly weather covariates from Open-Meteo historical archive.

    Parameters
    ----------
    region : region code matching REGION_COORDS keys
    start  : ISO date string
    end    : ISO date string
    save   : save raw CSV to data/raw/

    Returns
    -------
    DataFrame with columns: [timestamp, temperature_2m, wind_speed_10m,
                              cloud_cover, shortwave_radiation,
                              relative_humidity_2m, region]
    """
    if region not in REGION_COORDS:
        raise ValueError(f"No coordinates for region '{region}'. Add to REGION_COORDS.")

    lat, lon = REGION_COORDS[region]
    logger.info(f"Fetching weather for {region} ({lat}, {lon}): {start} → {end}")

    # Open-Meteo allows up to ~1 year per request; split into annual chunks
    frames = []
    years = range(int(start[:4]), int(end[:4]) + 1)

    for year in years:
        y_start = f"{year}-01-01"
        y_end = f"{year}-12-31"

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": y_start,
            "end_date": y_end,
            "hourly": ",".join(WEATHER_VARS),
            "timezone": "UTC",
        }

        try:
            r = requests.get(OPEN_METEO_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            logger.error(f"Weather fetch failed for {region} {year}: {e}")
            continue

        hourly = data.get("hourly", {})
        if not hourly:
            logger.warning(f"No hourly data returned for {region} {year}")
            continue

        df_year = pd.DataFrame(hourly)
        df_year = df_year.rename(columns={"time": "timestamp"})
        df_year["timestamp"] = pd.to_datetime(df_year["timestamp"])
        frames.append(df_year)
        logger.info(f"  {year}: {len(df_year)} rows")

    if not frames:
        raise RuntimeError(f"No weather data retrieved for region={region}")

    df = pd.concat(frames, ignore_index=True)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    df["region"] = region
    df = df.sort_values("timestamp").reset_index(drop=True)

    if save:
        out = DATA_RAW / f"weather_{region}_{start[:4]}_{end[:4]}.csv"
        df.to_csv(out, index=False)
        logger.info(f"Saved weather data → {out}")

    return df
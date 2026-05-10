"""
EIA Open Data API v2 client.
Fetches hourly electricity demand by region.
API key: https://www.eia.gov/opendata/register.php
"""

import time
import logging
import requests
import pandas as pd
from pathlib import Path
from src.utils.config import EIA_API_KEY, DATA_RAW

logger = logging.getLogger(__name__)

EIA_BASE = "https://api.eia.gov/v2"

# EIA region codes → human-readable names
REGION_MAP = {
    "US48": "Lower 48 States",
    "CAL":  "California",
    "TEX":  "Texas",
    "MIDA": "Mid-Atlantic",
    "NE":   "New England",
    "NW":   "Northwest",
    "SE":   "Southeast",
    "SW":   "Southwest",
    "TEN":  "Tennessee",
    "CAR":  "Carolinas",
}


def _get(endpoint: str, params: dict, retries: int = 3) -> dict:
    """GET with retry + backoff."""
    params["api_key"] = EIA_API_KEY
    url = f"{EIA_BASE}/{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            wait = 2 ** attempt
            logger.warning(f"EIA request failed (attempt {attempt+1}): {e}. Retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"EIA API unreachable after {retries} attempts: {url}")


def fetch_hourly_demand(
    region: str = "US48",
    start: str = "2019-01-01",
    end: str = "2024-01-01",
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly electricity demand (MW) from EIA for a given region.

    Parameters
    ----------
    region : EIA region code, e.g. 'US48', 'CAL', 'TEX'
    start  : ISO date string, inclusive
    end    : ISO date string, exclusive
    save   : if True, saves raw CSV to data/raw/

    Returns
    -------
    DataFrame with columns: [timestamp, load_mw, region]
    """
    if not EIA_API_KEY:
        raise EnvironmentError(
            "EIA_API_KEY not set. Add it to your .env file.\n"
            "Register free at: https://www.eia.gov/opendata/register.php\n"
            "Or use fetch_opsd_demand() for European data without a key."
        )

    logger.info(f"Fetching EIA hourly demand: region={region}, {start} → {end}")

    all_rows = []
    offset = 0
    page_size = 5000

    while True:
        data = _get(
            "electricity/rto/region-data/data",
            params={
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": region,
                "facets[type][]": "D",      # D = demand
                "start": start,
                "end": end,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": page_size,
            },
        )

        rows = data.get("response", {}).get("data", [])
        if not rows:
            break

        all_rows.extend(rows)
        logger.info(f"  fetched {len(all_rows)} rows so far (offset={offset})")

        if len(rows) < page_size:
            break
        offset += page_size

    if not all_rows:
        raise ValueError(f"No data returned for region={region}, {start}→{end}")

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"period": "timestamp", "value": "load_mw"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
    df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
    df["region"] = region
    df = df[["timestamp", "load_mw", "region"]].sort_values("timestamp").reset_index(drop=True)

    if save:
        out = DATA_RAW / f"eia_{region}_{start[:4]}_{end[:4]}.csv"
        df.to_csv(out, index=False)
        logger.info(f"Saved raw EIA data → {out}")

    return df


def fetch_all_regions(start: str = "2019-01-01", end: str = "2024-01-01") -> pd.DataFrame:
    """Fetch and concatenate demand for all configured regions."""
    frames = []
    for region in REGION_MAP:
        try:
            frames.append(fetch_hourly_demand(region, start, end))
        except Exception as e:
            logger.error(f"Failed to fetch region {region}: {e}")
    return pd.concat(frames, ignore_index=True)
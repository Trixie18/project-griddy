"""
Open Power System Data (OPSD) client.
Downloads European hourly load + weather. No API key required.
Use this if you don't have an EIA key yet.
"""

import logging
import requests
import pandas as pd
from io import StringIO
from src.utils.config import DATA_RAW, OPSD_BASE_URL

logger = logging.getLogger(__name__)

# Direct CSV download URLs (stable OPSD releases)
OPSD_URLS = {
    "load_weather": (
        "https://data.open-power-system-data.org/weather_data/latest/"
        "weather_data.csv"
    ),
    "load": (
        "https://data.open-power-system-data.org/time_series/latest/"
        "time_series_60min_singleindex.csv"
    ),
}

# Which country columns to pull from OPSD
COUNTRY_LOAD_COLS = {
    "DE": "DE_load_actual_entsoe_transparency",
    "FR": "FR_load_actual_entsoe_transparency",
    "GB": "GB_GBN_load_actual_entsoe_transparency",
}


def fetch_opsd_load(
    country: str = "DE",
    start: str = "2019-01-01",
    end: str = "2024-01-01",
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly electricity load from OPSD for a European country.

    Parameters
    ----------
    country : 'DE' (Germany), 'FR' (France), or 'GB' (Great Britain)
    start   : ISO date string
    end     : ISO date string
    save    : save raw CSV to data/raw/

    Returns
    -------
    DataFrame with columns: [timestamp, load_mw, region]
    """
    if country not in COUNTRY_LOAD_COLS:
        raise ValueError(f"Country must be one of {list(COUNTRY_LOAD_COLS.keys())}")

    col = COUNTRY_LOAD_COLS[country]
    cache_path = DATA_RAW / "opsd_time_series_60min.csv"

    # Use cached file if available
    if cache_path.exists():
        logger.info(f"Loading OPSD from cache: {cache_path}")
        df_raw = pd.read_csv(cache_path, parse_dates=["utc_timestamp"], low_memory=False)
    else:
        logger.info("Downloading OPSD time series (this may take ~2 min, ~500MB)...")
        r = requests.get(OPSD_URLS["load"], timeout=120, stream=True)
        r.raise_for_status()

        chunks = []
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            chunks.append(chunk)
        content = b"".join(chunks).decode("utf-8")

        df_raw = pd.read_csv(StringIO(content), parse_dates=["utc_timestamp"], low_memory=False)
        df_raw.to_csv(cache_path, index=False)
        logger.info(f"Cached OPSD data → {cache_path}")

    if col not in df_raw.columns:
        raise KeyError(f"Column '{col}' not found. Available: {[c for c in df_raw.columns if country in c]}")

    df = df_raw[["utc_timestamp", col]].copy()
    df = df.rename(columns={"utc_timestamp": "timestamp", col: "load_mw"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
    df["region"] = country

    # Filter date range
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    df = df.sort_values("timestamp").reset_index(drop=True)

    if save:
        out = DATA_RAW / f"opsd_{country}_{start[:4]}_{end[:4]}.csv"
        df.to_csv(out, index=False)
        logger.info(f"Saved OPSD load → {out}")

    return df
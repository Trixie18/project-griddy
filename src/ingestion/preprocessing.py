"""
Preprocessing & feature engineering pipeline.
Merges load + weather, imputes gaps, engineers calendar/lag features,
and saves a clean parquet file for all downstream phases.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config import DATA_PROCESSED, TARGET_COL, TIME_COL

logger = logging.getLogger(__name__)


# ── Cleaning ──────────────────────────────────────────────────────────────

def impute_missing(df: pd.DataFrame, col: str = TARGET_COL, max_gap_hours: int = 6) -> pd.DataFrame:
    """
    Linear interpolation for gaps ≤ max_gap_hours, forward-fill for longer gaps.
    Clips extreme outliers (beyond 4 IQR from median).
    """
    df = df.copy()

    # Outlier clipping
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 4 * iqr, q3 + 4 * iqr
    n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
    if n_clipped > 0:
        logger.info(f"Clipping {n_clipped} outliers in '{col}' to [{lo:.0f}, {hi:.0f}]")
    df[col] = df[col].clip(lo, hi)

    # Interpolate short gaps, ffill long ones
    df[col] = (
        df[col]
        .interpolate(method="linear", limit=max_gap_hours)
        .ffill()
        .bfill()
    )
    return df


def ensure_hourly_index(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a complete hourly DatetimeIndex, inserting NaNs for missing hours."""
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.set_index(TIME_COL)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)
    df.index.name = TIME_COL
    return df.reset_index()


# ── Feature Engineering ───────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclically-encoded calendar features + binary flags."""
    df = df.copy()
    ts = pd.to_datetime(df[TIME_COL])

    # Raw calendar
    df["hour"]        = ts.dt.hour
    df["dayofweek"]   = ts.dt.dayofweek       # 0=Mon … 6=Sun
    df["dayofyear"]   = ts.dt.dayofyear
    df["month"]       = ts.dt.month
    df["weekofyear"]  = ts.dt.isocalendar().week.astype(int)
    df["quarter"]     = ts.dt.quarter
    df["year"]        = ts.dt.year

    # Cyclic encodings (sin/cos) to avoid boundary discontinuity
    df["hour_sin"]       = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]       = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]        = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]        = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofyear_sin"]  = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"]  = np.cos(2 * np.pi * df["dayofyear"] / 365)

    # Binary flags
    df["is_weekend"]  = (df["dayofweek"] >= 5).astype(int)
    df["is_night"]    = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)
    df["is_peak"]     = ((df["hour"] >= 8) & (df["hour"] <= 20) & (df["is_weekend"] == 0)).astype(int)

    return df


def add_holiday_features(df: pd.DataFrame, country: str = "US") -> pd.DataFrame:
    """
    Add binary holiday flag.
    Uses the 'holidays' library — install separately: pip install holidays
    """
    try:
        import holidays as hd
    except ImportError:
        logger.warning("'holidays' package not installed — skipping holiday features. pip install holidays")
        df["is_holiday"] = 0
        return df

    ts = pd.to_datetime(df[TIME_COL])
    years = ts.dt.year.unique().tolist()

    country_upper = country.upper()
    try:
        cal = hd.country_holidays(country_upper, years=years)
    except Exception:
        logger.warning(f"Could not load holidays for country='{country}'. Defaulting to 0.")
        df["is_holiday"] = 0
        return df

    holiday_dates = set(cal.keys())
    df["is_holiday"] = ts.dt.date.apply(lambda d: int(d in holiday_dates))
    return df


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] = [24, 48, 168],   # 1-day, 2-day, 1-week
    col: str = TARGET_COL,
) -> pd.DataFrame:
    """Add lagged target values as features."""
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = [24, 168],
    col: str = TARGET_COL,
) -> pd.DataFrame:
    """Add rolling mean and std features."""
    df = df.copy()
    for w in windows:
        df[f"{col}_roll_mean_{w}h"] = df[col].shift(1).rolling(w).mean()
        df[f"{col}_roll_std_{w}h"]  = df[col].shift(1).rolling(w).std()
    return df


# ── Master pipeline ───────────────────────────────────────────────────────

def build_features(
    load_df: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
    country: str = "US",
    save: bool = True,
    filename: str = "features.parquet",
) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Ensure hourly index
      2. Impute missing load values
      3. Merge weather covariates
      4. Add calendar, holiday, lag, rolling features
      5. Save to data/processed/

    Parameters
    ----------
    load_df    : DataFrame with [timestamp, load_mw, region]
    weather_df : optional DataFrame with [timestamp, temperature_2m, ...]
    country    : 'US' or 'DE' or 'FR' or 'GB' for holiday calendar
    save       : whether to persist the result
    filename   : output parquet filename

    Returns
    -------
    Feature-engineered DataFrame
    """
    logger.info("Starting preprocessing pipeline...")

    # 1. Complete hourly grid
    df = ensure_hourly_index(load_df)
    logger.info(f"  Hourly index: {len(df)} rows")

    # 2. Impute load
    df = impute_missing(df, col=TARGET_COL)

    # 3. Merge weather
    if weather_df is not None:
        weather_df = weather_df.copy()
        weather_df[TIME_COL] = pd.to_datetime(weather_df[TIME_COL])
        weather_cols = [c for c in weather_df.columns if c not in (TIME_COL, "region")]
        df = df.merge(weather_df[[TIME_COL] + weather_cols], on=TIME_COL, how="left")

        # Impute weather gaps too
        for col in weather_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear", limit=6).ffill().bfill()

        logger.info(f"  Merged weather columns: {weather_cols}")

    # 4. Calendar features
    df = add_calendar_features(df)
    df = add_holiday_features(df, country=country)

    # 5. Lag + rolling features
    df = add_lag_features(df)
    df = add_rolling_features(df)

    logger.info(f"  Final shape: {df.shape}")

    if save:
        out = DATA_PROCESSED / filename
        df.to_parquet(out, index=False)
        logger.info(f"Saved features → {out}")

    return df
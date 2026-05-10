"""
Persist and load forecast results across phases.
All forecasts saved to outputs/forecasts/ as parquet.
"""

import pandas as pd
from pathlib import Path
from src.utils.config import OUTPUTS_FORECASTS


def save_forecasts(df: pd.DataFrame, name: str) -> Path:
    """Save a forecast DataFrame to outputs/forecasts/<name>.parquet"""
    out = OUTPUTS_FORECASTS / f"{name}.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved forecasts → {out}")
    return out


def load_forecasts(name: str) -> pd.DataFrame:
    """Load a forecast DataFrame from outputs/forecasts/<name>.parquet"""
    path = OUTPUTS_FORECASTS / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No forecast file at {path}")
    return pd.read_parquet(path)


def load_all_forecasts() -> pd.DataFrame:
    """Load and concatenate all parquet files in outputs/forecasts/."""
    files = list(OUTPUTS_FORECASTS.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No forecast files found in outputs/forecasts/")
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)
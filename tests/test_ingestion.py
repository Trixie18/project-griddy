import pandas as pd
import pytest
from src.ingestion.preprocessing import (
    ensure_hourly_index,
    impute_missing,
    add_calendar_features,
    add_lag_features,
)


def _make_load_df(n=200):
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    import numpy as np
    return pd.DataFrame({
        "timestamp": idx,
        "load_mw": 30000 + 5000 * np.sin(2 * 3.14159 * idx.hour / 24),
        "region": "TEST",
    })


def test_ensure_hourly_index():
    df = _make_load_df(48)
    # Introduce a gap by removing row 10
    df = df.drop(index=10).reset_index(drop=True)
    result = ensure_hourly_index(df)
    assert len(result) == 48, "Missing hour should be reindexed"


def test_impute_missing():
    df = _make_load_df(100)
    df.loc[5:8, "load_mw"] = None
    result = impute_missing(df, col="load_mw")
    assert result["load_mw"].isna().sum() == 0


def test_calendar_features():
    df = _make_load_df(48)
    result = add_calendar_features(df)
    for col in ["hour_sin", "hour_cos", "is_weekend", "is_peak"]:
        assert col in result.columns, f"Missing column: {col}"


def test_lag_features():
    df = _make_load_df(200)
    result = add_lag_features(df, lags=[24])
    assert "load_mw_lag_24h" in result.columns
    # First 24 values should be NaN
    assert result["load_mw_lag_24h"].iloc[:24].isna().all()
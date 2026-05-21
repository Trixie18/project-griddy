import numpy as np
import pandas as pd
import pytest
from src.models.lgbm_model import (
    lgbm_point_fn, _add_time_features,
    _add_lag_features, get_feature_cols,
)


def _make_series(n=500):
    idx  = pd.date_range("2019-01-01", periods=n, freq="h")
    vals = 50000 + 5000 * np.sin(2 * np.pi * np.arange(n) / 24)
    return pd.Series(vals, index=idx)


def test_time_features():
    df = pd.DataFrame({"timestamp": pd.date_range("2019-01-01", periods=48, freq="h")})
    result = _add_time_features(df)
    assert "hour_sin" in result.columns
    assert "is_weekend" in result.columns
    assert result["hour_sin"].between(-1, 1).all()


def test_lag_features():
    s  = _make_series(300)
    df = s.reset_index()
    df.columns = ["timestamp", "load_mw"]
    result = _add_lag_features(df)
    assert "load_mw_lag_24h" in result.columns
    assert result["load_mw_lag_24h"].iloc[:24].isna().all()


def test_lgbm_point_fn_shape():
    s     = _make_series(500)
    preds = lgbm_point_fn(s, horizon=24)
    assert len(preds) == 24
    assert not np.any(np.isnan(preds))


def test_lgbm_point_fn_reasonable():
    s     = _make_series(500)
    preds = lgbm_point_fn(s, horizon=24)
    # Forecasts should be in a reasonable MW range
    assert preds.min() > 10000
    assert preds.max() < 200000
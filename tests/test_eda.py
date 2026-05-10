import numpy as np
import pandas as pd
import pytest
from src.utils.plot_helpers import set_style, save_fig


def test_set_style_runs():
    set_style()


def test_stl_decomposition():
    from statsmodels.tsa.seasonal import STL
    idx = pd.date_range("2022-01-01", periods=24 * 60, freq="h")
    np.random.seed(42)
    vals = (
        40000
        + 5000 * np.sin(2 * np.pi * np.arange(len(idx)) / 24)
        + np.random.normal(0, 500, len(idx))
    )
    s = pd.Series(vals, index=idx)
    result = STL(s, period=24, seasonal=25, robust=True).fit()
    assert len(result.trend) == len(s)
    assert len(result.seasonal) == len(s)


def test_acf_shape():
    from statsmodels.tsa.stattools import acf
    np.random.seed(0)
    s = pd.Series(np.random.randn(500))
    vals = acf(s, nlags=40, fft=True)
    assert len(vals) == 41
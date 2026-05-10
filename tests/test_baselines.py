import numpy as np
import pandas as pd
import pytest
from src.models.baselines import seasonal_naive_fn
from src.models.cross_validation import expanding_window_splits
from src.metrics.evaluation import smape, mase, rmse, coverage, winkler_score


def _make_series(n=500):
    idx  = pd.date_range("2019-01-01", periods=n, freq="h")
    vals = 50000 + 5000 * np.sin(2 * np.pi * np.arange(n) / 24)
    return pd.Series(vals, index=idx)


def test_seasonal_naive_shape():
    s     = _make_series(200)
    preds = seasonal_naive_fn(s, horizon=24)
    assert len(preds) == 24


def test_seasonal_naive_repeats_cycle():
    s     = _make_series(200)
    preds = seasonal_naive_fn(s, horizon=24, period=24)
    np.testing.assert_array_almost_equal(preds, s.values[-24:])


def test_expanding_window_splits():
    splits = list(expanding_window_splits(
        n=500, n_splits=3, horizon=24, min_train_size=24*7
    ))
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert len(test_idx) == 24
        assert len(train_idx) > 0
        assert train_idx[-1] < test_idx[0]


def test_smape_perfect():
    a = np.array([100.0, 200.0, 300.0])
    assert smape(a, a) == pytest.approx(0.0)


def test_smape_range():
    a = np.array([100.0, 200.0])
    f = np.array([200.0, 100.0])
    assert 0 <= smape(a, f) <= 200


def test_coverage():
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lower  = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    upper  = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    assert coverage(actual, lower, upper) == pytest.approx(1.0)


def test_winkler_no_penalty():
    actual = np.array([1.0, 2.0, 3.0])
    lower  = np.array([0.0, 1.0, 2.0])
    upper  = np.array([2.0, 3.0, 4.0])
    score  = winkler_score(actual, lower, upper, alpha=0.1)
    # All within interval — score should equal mean width = 2.0
    assert score == pytest.approx(2.0)
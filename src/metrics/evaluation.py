"""
Unified forecast evaluation metrics.
All functions accept numpy arrays or pandas Series.
"""

import numpy as np
import pandas as pd
from typing import Union

Array = Union[np.ndarray, pd.Series]


def smape(actual: Array, forecast: Array) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    Range: [0, 200]. Lower is better.
    """
    actual   = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask     = ~(np.isnan(actual) | np.isnan(forecast))
    actual, forecast = actual[mask], forecast[mask]
    if len(actual) == 0:
        return np.nan
    denom = (np.abs(actual) + np.abs(forecast)) / 2
    denom = np.where(denom == 0, 1e-8, denom)
    return float(np.mean(np.abs(actual - forecast) / denom) * 100)


def mase(actual: Array, forecast: Array, seasonal_period: int = 24) -> float:
    """
    Mean Absolute Scaled Error.
    Scaled by in-sample seasonal naive MAE.
    < 1 means better than seasonal naive.
    """
    actual   = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask     = ~(np.isnan(actual) | np.isnan(forecast))
    actual, forecast = actual[mask], forecast[mask]
    if len(actual) == 0:
        return np.nan
    mae_model  = np.mean(np.abs(actual - forecast))
    naive_diffs = np.abs(actual[seasonal_period:] - actual[:-seasonal_period])
    if len(naive_diffs) == 0 or np.mean(naive_diffs) == 0:
        return np.nan
    return float(mae_model / np.mean(naive_diffs))


def rmse(actual: Array, forecast: Array) -> float:
    actual   = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask     = ~(np.isnan(actual) | np.isnan(forecast))
    actual, forecast = actual[mask], forecast[mask]
    if len(actual) == 0:
        return np.nan
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mae(actual: Array, forecast: Array) -> float:
    actual   = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask     = ~(np.isnan(actual) | np.isnan(forecast))
    actual, forecast = actual[mask], forecast[mask]
    if len(actual) == 0:
        return np.nan
    return float(np.mean(np.abs(actual - forecast)))


def crps_gaussian(actual: Array, mu: Array, sigma: Array) -> float:
    """
    Continuous Ranked Probability Score assuming Gaussian predictive distribution.
    Lower is better.
    """
    from scipy import stats
    actual = np.asarray(actual, dtype=float)
    mu     = np.asarray(mu,     dtype=float)
    sigma  = np.asarray(sigma,  dtype=float)
    sigma  = np.maximum(sigma, 1e-6)
    z      = (actual - mu) / sigma
    crps_  = sigma * (
        z * (2 * stats.norm.cdf(z) - 1)
        + 2 * stats.norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )
    return float(np.mean(crps_))


def winkler_score(actual: Array, lower: Array, upper: Array,
                  alpha: float = 0.1) -> float:
    """
    Winkler score for a (1-alpha) prediction interval.
    Lower is better.
    """
    actual = np.asarray(actual, dtype=float)
    lower  = np.asarray(lower,  dtype=float)
    upper  = np.asarray(upper,  dtype=float)
    width  = upper - lower
    penalty_lo = 2 / alpha * np.maximum(lower - actual, 0)
    penalty_hi = 2 / alpha * np.maximum(actual - upper, 0)
    return float(np.mean(width + penalty_lo + penalty_hi))


def coverage(actual: Array, lower: Array, upper: Array) -> float:
    """Empirical coverage of a prediction interval."""
    actual = np.asarray(actual, dtype=float)
    lower  = np.asarray(lower,  dtype=float)
    upper  = np.asarray(upper,  dtype=float)
    return float(np.mean((actual >= lower) & (actual <= upper)))


def evaluate_point(actual: Array, forecast: Array,
                   label: str = "model",
                   seasonal_period: int = 24) -> dict:
    """Return a dict of all point forecast metrics."""
    return {
        "model":  label,
        "smape":  smape(actual, forecast),
        "mase":   mase(actual, forecast, seasonal_period),
        "rmse":   rmse(actual, forecast),
        "mae":    mae(actual, forecast),
        "n":      int(np.sum(~np.isnan(np.asarray(actual, float)))),
    }


def evaluate_interval(actual: Array, lower: Array, upper: Array,
                       mu: Array = None, sigma: Array = None,
                       label: str = "model",
                       alpha: float = 0.1) -> dict:
    """Return a dict of probabilistic metrics."""
    result = {
        "model":    label,
        "coverage": coverage(actual, lower, upper),
        "winkler":  winkler_score(actual, lower, upper, alpha),
    }
    if mu is not None and sigma is not None:
        result["crps"] = crps_gaussian(actual, mu, sigma)
    return result
"""
Time-series cross-validation (expanding window).
Returns fold indices and a unified scoring function.
"""

import numpy as np
import pandas as pd
from typing import Iterator


def expanding_window_splits(
    n: int,
    n_splits: int = 5,
    horizon: int = 24,
    min_train_size: int = 24 * 7 * 4,   # 4 weeks minimum
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yields (train_idx, test_idx) for expanding-window CV.

    Parameters
    ----------
    n             : total number of observations
    n_splits      : number of folds
    horizon       : forecast horizon (test window size)
    min_train_size: minimum training set size
    """
    total_test = n_splits * horizon
    if n - total_test < min_train_size:
        raise ValueError(
            f"Not enough data for {n_splits} folds with horizon={horizon}. "
            f"Need at least {min_train_size + total_test} rows, have {n}."
        )

    # Start of first test fold
    first_test_start = n - total_test

    for fold in range(n_splits):
        test_start = first_test_start + fold * horizon
        test_end   = test_start + horizon
        train_idx  = np.arange(0, test_start)
        test_idx   = np.arange(test_start, test_end)
        yield train_idx, test_idx


def run_cv(
    series: pd.Series,
    model_fn,
    n_splits: int = 5,
    horizon: int = 24,
    min_train_size: int = 24 * 7 * 4,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Run cross-validation for any model.

    Parameters
    ----------
    series     : full time series (pd.Series with DatetimeIndex)
    model_fn   : callable(train: pd.Series, horizon: int) -> np.ndarray
                 Must return point forecasts of length `horizon`
    n_splits   : number of CV folds
    horizon    : forecast horizon per fold
    model_name : label for results

    Returns
    -------
    DataFrame with columns: [fold, timestamp, actual, forecast, model]
    """
    from src.metrics.evaluation import smape, mase, rmse

    values  = series.values
    index   = series.index
    records = []

    for fold, (train_idx, test_idx) in enumerate(
        expanding_window_splits(len(values), n_splits, horizon, min_train_size)
    ):
        train = pd.Series(values[train_idx], index=index[train_idx])
        test  = pd.Series(values[test_idx],  index=index[test_idx])

        try:
            preds = model_fn(train, horizon)
            preds = np.asarray(preds, dtype=float)
        except Exception as e:
            print(f"  [fold {fold}] {model_name} failed: {e}")
            preds = np.full(horizon, np.nan)

        for i, (ts, actual, pred) in enumerate(
            zip(test.index, test.values, preds)
        ):
            records.append({
                "fold":      fold,
                "timestamp": ts,
                "actual":    float(actual),
                "forecast":  float(pred),
                "model":     model_name,
                "horizon_h": i + 1,
            })

        fold_smape = smape(test.values, preds)
        fold_mase  = mase(test.values, preds)
        print(f"  [fold {fold+1}/{n_splits}] sMAPE={fold_smape:.2f}%  MASE={fold_mase:.3f}")

    return pd.DataFrame(records)
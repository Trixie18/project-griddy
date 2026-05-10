"""
Classical baseline models.
Each model exposes a model_fn(train, horizon) -> np.ndarray interface
compatible with run_cv().
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── 1. Seasonal Naive ─────────────────────────────────────────────────────

def seasonal_naive_fn(train: pd.Series, horizon: int,
                      period: int = 24) -> np.ndarray:
    """
    Repeat the last full seasonal cycle.
    Default period=24 (daily seasonality for hourly data).
    """
    values = train.values
    if len(values) < period:
        return np.full(horizon, values[-1])
    last_cycle = values[-period:]
    reps = int(np.ceil(horizon / period))
    tiled = np.tile(last_cycle, reps)
    return tiled[:horizon]


# ── 2. Theta model ────────────────────────────────────────────────────────

def theta_fn(train: pd.Series, horizon: int) -> np.ndarray:
    """
    Theta method via statsforecast.
    Falls back to seasonal naive if unavailable.
    """
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import Theta

        df_sf = pd.DataFrame({
            "unique_id": "series",
            "ds":        train.index,
            "y":         train.values,
        })
        sf = StatsForecast(
            models=[Theta(season_length=24)],
            freq="h",
            n_jobs=1,
        )
        sf.fit(df_sf)
        preds = sf.predict(h=horizon)
        return preds["Theta"].values

    except Exception as e:
        print(f"    Theta failed ({e}), falling back to seasonal naive")
        return seasonal_naive_fn(train, horizon)


# ── 3. Prophet ────────────────────────────────────────────────────────────

def prophet_fn(train: pd.Series, horizon: int) -> np.ndarray:
    """
    Facebook Prophet with daily + weekly seasonality.
    """
    try:
        from prophet import Prophet

        df_p = pd.DataFrame({
            "ds": train.index,
            "y":  train.values,
        })

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode="multiplicative",
            interval_width=0.8,
            changepoint_prior_scale=0.05,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df_p)

        last_ts  = train.index[-1]
        future   = pd.DataFrame({
            "ds": pd.date_range(
                start=last_ts + pd.Timedelta(hours=1),
                periods=horizon,
                freq="h",
            )
        })
        forecast = m.predict(future)
        return forecast["yhat"].values

    except Exception as e:
        print(f"    Prophet failed ({e}), falling back to seasonal naive")
        return seasonal_naive_fn(train, horizon)


def prophet_fn_with_intervals(
    train: pd.Series, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prophet returning (point, lower_80, upper_80).
    Used for interval evaluation in Phase 5.
    """
    try:
        from prophet import Prophet

        df_p = pd.DataFrame({"ds": train.index, "y": train.values})
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode="multiplicative",
            interval_width=0.8,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df_p)

        future   = pd.DataFrame({
            "ds": pd.date_range(
                start=train.index[-1] + pd.Timedelta(hours=1),
                periods=horizon, freq="h",
            )
        })
        fc = m.predict(future)
        return (
            fc["yhat"].values,
            fc["yhat_lower"].values,
            fc["yhat_upper"].values,
        )
    except Exception as e:
        print(f"    Prophet interval failed: {e}")
        pt = seasonal_naive_fn(train, horizon)
        return pt, pt * 0.9, pt * 1.1


# ── 4. ARIMA / SARIMA ─────────────────────────────────────────────────────

def arima_fn(train: pd.Series, horizon: int) -> np.ndarray:
    """
    Auto ARIMA via pmdarima.
    Uses seasonal=True with m=24 for hourly data.
    Fits on daily aggregation to keep it fast, then scales back.
    """
    try:
        import pmdarima as pm

        # Fit on daily data for speed, forecast daily, interpolate hourly
        daily = train.resample("D").mean().dropna()

        horizon_days = int(np.ceil(horizon / 24))

        model = pm.auto_arima(
            daily.values,
            seasonal=True,
            m=7,               # weekly seasonality on daily data
            max_p=3, max_q=3,
            max_P=2, max_Q=2,
            d=None,            # auto-detect differencing
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion="aic",
            n_jobs=1,
        )

        daily_preds = model.predict(n_periods=horizon_days)

        # Repeat each daily forecast for 24 hours
        hourly_preds = np.repeat(daily_preds, 24)[:horizon]
        return hourly_preds

    except Exception as e:
        print(f"    ARIMA failed ({e}), falling back to seasonal naive")
        return seasonal_naive_fn(train, horizon)
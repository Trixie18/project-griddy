"""
XGBoost quantile regression model.
Stable on Apple M1 ARM64 — no OpenMP dependency.
"""

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from src.utils.config import TARGET_COL, TIME_COL
from src.models.forecast_store import save_forecasts

warnings.filterwarnings("ignore")


FEATURE_COLS = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin","month_cos",
    "dayofyear_sin", "dayofyear_cos",
    "is_weekend", "is_peak", "is_holiday",
    "load_mw_lag_24h",
    "load_mw_lag_48h",
    "load_mw_lag_168h",
    "load_mw_roll_mean_24h",
    "load_mw_roll_mean_168h",
    "load_mw_roll_std_24h",
    "load_mw_roll_std_168h",
]

XGB_PARAMS_BASE = {
    "n_estimators":     300,
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "n_jobs":           1,
    "random_state":     42,
    "tree_method":      "hist",
    "device":           "cpu",
}


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in FEATURE_COLS if c in df.columns]


def xgb_point_fn(train: pd.Series, horizon: int) -> np.ndarray:
    """
    CV-compatible interface for run_cv().
    Returns median (q50) point forecast.
    """
    train_df  = _series_to_features(train)
    future_df = _make_future_features(train, horizon)
    feat_cols = get_feature_cols(train_df)

    if not feat_cols:
        from src.models.baselines import seasonal_naive_fn
        return seasonal_naive_fn(train, horizon)

    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.5,
        **XGB_PARAMS_BASE,
    )
    model.fit(train_df[feat_cols], train_df[TARGET_COL])
    return model.predict(future_df[feat_cols])


def train_xgb_quantiles(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
) -> tuple:
    """
    Train three XGBoost models (q10, q50, q90).

    Returns
    -------
    (models_dict, val_preds_df, feature_importance_df)
    """
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    if "hour_sin" not in df.columns:
        df = _add_time_features(df)
    if "load_mw_lag_24h" not in df.columns:
        df = _add_lag_features(df)

    df = df.dropna(subset=[TARGET_COL])
    feat_cols = get_feature_cols(df)
    print(f"Feature columns : {feat_cols}")

    cutoff   = int(len(df) * (1 - val_fraction))
    train_df = df.iloc[:cutoff].dropna(subset=feat_cols + [TARGET_COL])
    val_df   = df.iloc[cutoff:].dropna(subset=feat_cols + [TARGET_COL])

    print(f"Train rows      : {len(train_df):,}")
    print(f"Val   rows      : {len(val_df):,}")

    X_train = train_df[feat_cols].values
    y_train = train_df[TARGET_COL].values
    X_val   = val_df[feat_cols].values
    y_val   = val_df[TARGET_COL].values

    models = {}
    preds  = {}

    for q in [0.1, 0.5, 0.9]:
        label = f"q{int(q*100)}"
        print(f"Training {label}...")
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            **XGB_PARAMS_BASE,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        models[label] = model
        preds[label]  = model.predict(X_val)
        print(f"  {label} done.")

    val_preds = val_df[[TIME_COL, TARGET_COL]].copy()
    val_preds = val_preds.rename(columns={TARGET_COL: "actual"})
    val_preds["q10"]   = preds["q10"]
    val_preds["q50"]   = preds["q50"]
    val_preds["q90"]   = preds["q90"]
    val_preds["model"] = "XGBoost"

    imp_df = pd.DataFrame({
        "feature":    feat_cols,
        "importance": models["q50"].feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    total = imp_df["importance"].sum()
    imp_df["importance_pct"] = imp_df["importance"] / total * 100

    return models, val_preds, imp_df


def compute_shap_values(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    n_samples: int = 500,
) -> tuple:
    feat_cols = get_feature_cols(df)
    sample    = df[feat_cols].dropna().head(n_samples)

    # Force numeric — XGBoost 3.x can return string arrays
    sample = sample.apply(pd.to_numeric, errors="coerce").fillna(0)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    return shap_values, feat_cols


# ── Feature engineering helpers ────────────────────────────────────────────

def _series_to_features(s: pd.Series) -> pd.DataFrame:
    df = s.reset_index()
    df.columns = [TIME_COL, TARGET_COL]
    df = _add_time_features(df)
    df = _add_lag_features(df)
    return df.dropna()


def _make_future_features(train: pd.Series, horizon: int) -> pd.DataFrame:
    last_ts    = train.index[-1]
    future_idx = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=horizon, freq="h",
    )
    future_df = pd.DataFrame({TIME_COL: future_idx})
    future_df = _add_time_features(future_df)

    history = train.values
    for lag, col in [(24,  "load_mw_lag_24h"),
                     (48,  "load_mw_lag_48h"),
                     (168, "load_mw_lag_168h")]:
        vals = []
        for i in range(horizon):
            idx = len(history) - lag + i
            vals.append(float(history[idx]) if 0 <= idx < len(history) else np.nan)
        future_df[col] = vals

    for w, col_mean, col_std in [
        (24,  "load_mw_roll_mean_24h",  "load_mw_roll_std_24h"),
        (168, "load_mw_roll_mean_168h", "load_mw_roll_std_168h"),
    ]:
        tail = history[-w:] if len(history) >= w else history
        future_df[col_mean] = float(np.mean(tail))
        future_df[col_std]  = float(np.std(tail))

    return future_df.ffill().fillna(0)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df[TIME_COL])
    df = df.copy()
    df["hour_sin"]      = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * ts.dt.hour / 24)
    df["dow_sin"]       = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df["dow_cos"]       = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    df["month_sin"]     = np.sin(2 * np.pi * ts.dt.month / 12)
    df["month_cos"]     = np.cos(2 * np.pi * ts.dt.month / 12)
    df["dayofyear_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365)
    df["is_weekend"]    = (ts.dt.dayofweek >= 5).astype(int)
    df["is_peak"]       = (
        (ts.dt.hour >= 8) & (ts.dt.hour <= 20) &
        (ts.dt.dayofweek < 5)
    ).astype(int)
    df["is_holiday"]    = 0
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for lag in [24, 48, 168]:
        df[f"load_mw_lag_{lag}h"] = df[TARGET_COL].shift(lag)
    for w in [24, 168]:
        df[f"load_mw_roll_mean_{w}h"] = df[TARGET_COL].shift(1).rolling(w).mean()
        df[f"load_mw_roll_std_{w}h"]  = df[TARGET_COL].shift(1).rolling(w).std()
    return df
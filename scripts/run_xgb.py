"""
Run XGBoost training outside Jupyter to avoid M1 kernel crash.
Execute with: python scripts/run_xgb.py
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import TARGET_COL, TIME_COL, OUTPUTS_FORECASTS
from src.utils.plot_helpers import set_style, save_fig, PALETTE
from src.models.xgb_model import train_xgb_quantiles, compute_shap_values, xgb_point_fn
from src.models.cross_validation import run_cv
from src.models.forecast_store import save_forecasts
from src.metrics.evaluation import smape, mase, coverage

set_style()

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet("data/processed/features_DE.parquet")
df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df = df.sort_values(TIME_COL).reset_index(drop=True)
print(f"Rows : {len(df):,}")
print(f"Range: {df[TIME_COL].min()} → {df[TIME_COL].max()}")

# ── Train quantile models ──────────────────────────────────────────────────
print("\nTraining XGBoost quantile models...")
models, val_preds, imp_df = train_xgb_quantiles(df, val_fraction=0.15)

clean = val_preds.dropna(subset=["actual", "q50"])
print(f"\nVal set scores:")
print(f"  sMAPE (q50)  : {smape(clean['actual'], clean['q50']):.2f}%")
print(f"  MASE  (q50)  : {mase(clean['actual'], clean['q50']):.3f}")
print(f"  Coverage 80% : {coverage(clean['actual'], clean['q10'], clean['q90'])*100:.1f}%")

print("\nFeature importance (q50):")
print(imp_df.to_string(index=False))

# ── CV run ─────────────────────────────────────────────────────────────────
print("\nRunning 5-fold CV...")
series = df.set_index(TIME_COL)[TARGET_COL].dropna()
cv_xgb = run_cv(
    series, xgb_point_fn,
    n_splits=5, horizon=24,
    min_train_size=24*7*8,
    model_name="XGBoost",
)
save_forecasts(cv_xgb, "cv_xgb")

# ── Save forecasts ─────────────────────────────────────────────────────────
print("\nSaving forecasts...")
save_forecasts(val_preds, "xgb_quantile_forecasts")

point = val_preds[[TIME_COL, "actual", "q50"]].copy()
point = point.rename(columns={"q50": "forecast"})
point["model"]     = "XGBoost"
point["fold"]      = 0
point["horizon_h"] = range(1, len(point) + 1)
save_forecasts(
    point[["fold", "timestamp", "actual", "forecast", "model", "horizon_h"]],
    "cv_xgb_point"
)

# ── Plots ──────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

plot_df = val_preds.dropna(subset=["actual"]).head(24*7).copy()
plot_df[TIME_COL] = pd.to_datetime(plot_df[TIME_COL])

fig, ax = plt.subplots(figsize=(16, 6))
ax.fill_between(
    plot_df[TIME_COL], plot_df["q10"], plot_df["q90"],
    alpha=0.25, color=PALETTE["primary"],
    label="80% interval (q10–q90)",
)
ax.plot(plot_df[TIME_COL], plot_df["q50"],
        color=PALETTE["primary"], linewidth=1.5, label="Median (q50)")
ax.plot(plot_df[TIME_COL], plot_df["actual"],
        color="black", linewidth=1.2, alpha=0.8, label="Actual")
ax.set_title("XGBoost quantile forecasts — first 7 days of validation")
ax.set_ylabel("Load (MW)")
ax.legend()
plt.tight_layout()
save_fig(fig, "14_xgb_quantile_forecast")

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.barh(imp_df["feature"], imp_df["importance_pct"],
         color=PALETTE["primary"], alpha=0.85)
ax2.set_xlabel("Importance (%)")
ax2.set_title("XGBoost feature importance (q50 model)")
ax2.invert_yaxis()
plt.tight_layout()
save_fig(fig2, "15_xgb_feature_importance")

# ── SHAP ───────────────────────────────────────────────────────────────────
print("\nComputing SHAP values...")
try:
    import shap
    from src.models.xgb_model import get_feature_cols
    import xgboost as xgb

    feat_cols = get_feature_cols(df)
    sample = (
        df[feat_cols]
        .dropna()
        .head(500)
        .astype(np.float32)
        .reset_index(drop=True)
    )

    # Use XGBoost's native SHAP — bypasses TreeExplainer bug in XGBoost 3.x
    dmat = xgb.DMatrix(sample, feature_names=feat_cols)
    shap_values = models["q50"].get_booster().predict(dmat, pred_contribs=True)
    shap_values = shap_values[:, :-1]  # drop bias column

    # Save raw SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feat_cols)
    save_forecasts(shap_df, "xgb_shap_values")

    # Mean absolute SHAP per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_imp  = pd.DataFrame({
        "feature":    feat_cols,
        "mean_shap":  mean_shap,
    }).sort_values("mean_shap", ascending=False).reset_index(drop=True)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.barh(shap_imp["feature"], shap_imp["mean_shap"],
             color=PALETTE["secondary"], alpha=0.85)
    ax3.set_xlabel("Mean |SHAP value| (MW)")
    ax3.set_title("SHAP feature importance — XGBoost q50")
    ax3.invert_yaxis()
    plt.tight_layout()
    save_fig(fig3, "16_shap_summary")
    print("SHAP plot saved.")

except Exception as e:
    print(f"SHAP failed (non-critical): {e}")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("PHASE 4 COMPLETE — XGBoost")
print("=" * 50)
print(f"sMAPE (q50)  : {smape(clean['actual'], clean['q50']):.2f}%")
print(f"MASE  (q50)  : {mase(clean['actual'], clean['q50']):.3f}")
print(f"Coverage 80% : {coverage(clean['actual'], clean['q10'], clean['q90'])*100:.1f}%")
print("\nFiles saved to outputs/forecasts/:")
for f in sorted(os.listdir("outputs/forecasts")):
    print(f"  {f}")
print("\nPlots saved to outputs/plots/:")
for f in sorted(f for f in os.listdir("outputs/plots") if f.startswith("1")):
    print(f"  {f}")
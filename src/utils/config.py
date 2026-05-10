from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS_FORECASTS = ROOT / "outputs" / "forecasts"
OUTPUTS_PLOTS = ROOT / "outputs" / "plots"
OUTPUTS_CALIBRATION = ROOT / "outputs" / "calibration"

for p in [DATA_RAW, DATA_PROCESSED, OUTPUTS_FORECASTS, OUTPUTS_PLOTS, OUTPUTS_CALIBRATION]:
    p.mkdir(parents=True, exist_ok=True)

# ── API keys ───────────────────────────────────────────────────────────────
EIA_API_KEY = os.getenv("EIA_API_KEY", "")
OPSD_BASE_URL = os.getenv("OPSD_BASE_URL", "https://data.open-power-system-data.org/time_series/latest")

# ── Modelling constants ────────────────────────────────────────────────────
TARGET_COL = "load_mw"
TIME_COL = "timestamp"
FREQ = "h"                          # hourly
FORECAST_HORIZONS = [24, 48, 168]   # 1-day, 2-day, 1-week ahead
QUANTILES = [0.1, 0.5, 0.9]

# Regions to model (matches EIA region codes)
REGIONS = ["US48", "CAL", "TEX", "MIDA", "NE", "NW", "SE", "SW", "TEN", "CAR"]

# ── TFT training defaults ──────────────────────────────────────────────────
TFT_MAX_ENCODER_LENGTH = 168        # 1 week of context
TFT_MAX_PREDICTION_LENGTH = 24      # 24-hour forecast
TFT_BATCH_SIZE = 64
TFT_MAX_EPOCHS = 50
TFT_LEARNING_RATE = 1e-3

# ── Device ─────────────────────────────────────────────────────────────────
import torch
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
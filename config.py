"""Single source of truth for paths, features, and demo locations.

Per CLAUDE.md: feature list, cache directory paths, and preset demo locations
live here. Do NOT redefine these in individual modules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / "cache"          # per-location NASA POWER caches (parquet)
ARTIFACTS_DIR = ROOT_DIR / "artifacts"  # saved models / metrics from train.py

# ---------------------------------------------------------------------------
# NASA POWER API
# ---------------------------------------------------------------------------
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
# ALLSKY_SFC_SW_DWN = all-sky surface shortwave downward irradiance (the target)
NASA_POWER_PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",   # target: solar irradiance (kWh/m^2/day)
    "T2M",                 # temperature at 2m
    "RH2M",                # relative humidity at 2m
    "WS2M",                # wind speed at 2m
    "PRECTOTCORR",         # precipitation
    # CLOUD_AMT and CLRSKY_SFC_SW_DWN deliberately excluded: both are valid
    # RE parameters but are CERES-derived and lag ~3 months behind the other
    # (near-real-time) parameters, which silently drops the most recent ~100
    # days of every location's history. Cloudiness is captured via the
    # clearness index instead — GHI / analytic TOA irradiance (src/solar.py),
    # which has no API dependency at all.
]
HISTORY_START = "20150101"  # ~10 years of daily history per location
TARGET_COL = "ALLSKY_SFC_SW_DWN"

# Cache staleness threshold, measured against the last OBSERVED day (last
# non-NaN target) in the cached parquet — not file mtime. NASA POWER's
# near-real-time feed lags ~5-7 days, so a fetch made TODAY already has a
# last observed day of ~today-6: this threshold must exceed the NRT lag by a
# real margin or every request would refetch in a loop. 10 = lag + ~3-5 days
# of acceptable drift; a location refetches at most ~once per 10 days.
CACHE_MAX_AGE_DAYS = 10

# ---------------------------------------------------------------------------
# Feature engineering (data.py builds exactly these; models consume FEATURES)
# ---------------------------------------------------------------------------
LAG_DAYS = [1, 2, 3, 7, 14]
ROLLING_WINDOWS = [3, 7, 30]

# Exogenous weather drivers, used as lag-1 features (data.py builds these)
EXOG_COLS = ["T2M", "RH2M", "WS2M", "PRECTOTCORR"]

FEATURES = (
    [f"{TARGET_COL}_lag{l}" for l in LAG_DAYS]
    + [f"{TARGET_COL}_roll{w}_mean" for w in ROLLING_WINDOWS]
    + [f"{TARGET_COL}_roll{w}_std" for w in ROLLING_WINDOWS]
    + ["clearness_index_lag1"]   # GHI / analytic TOA, lagged (no leakage)
    + [f"{c}_lag1" for c in EXOG_COLS]
    + [
        "doy_sin",               # seasonal encodings
        "doy_cos",
    ]
)

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
QUANTILES = [0.1, 0.5, 0.9]  # P10 / P50 / P90

# Tuned ONCE by src/tune.py (80-trial random search, val-split pinball on the
# demo locations, 2026-07-11) and reused fixed at inference — never retuned
# per location; the "fits in seconds" claim depends on this. Val pinball
# 0.2115 -> 0.1991 vs the previous hand-picked params; the strong
# regularization is what fixed the P10-P90 undercoverage (0.665 -> 0.785 val).
LGBM_PARAMS = {
    "num_leaves": 8,
    "min_child_samples": 47,
    "learning_rate": 0.01574834597749413,
    "n_estimators": 805,
    "colsample_bytree": 0.74442301806463,
    "reg_alpha": 9.011173123401376,
    "reg_lambda": 0.006592641307912177,
    "verbosity": -1,
}

# Phase-4 attention LSTM (src/model_seq.py). Fixed sizes, not tuned per
# location: the model is a comparison against LightGBM, trained offline for the
# demo locations only.
SEQ_PARAMS = {
    "window": 30,            # days of feature history per prediction
    "hidden": 32,
    "dropout": 0.1,
    "lr": 3e-3,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "max_epochs": 60,
    "patience": 8,           # early-stopping patience, in epochs
    "val_fraction": 0.15,    # chronological tail of the fit data
    "seed": 0,
}

# Chronological split fractions (train / val / test)
SPLIT_FRACTIONS = (0.7, 0.15, 0.15)

# Weather regimes (cluster.py maps cluster ids -> these labels)
REGIME_NAMES = ["Clear Sky", "Overcast", "Transitional", "Winter Low"]
N_REGIMES = len(REGIME_NAMES)
# Where irradiance has no real seasonal cycle (equatorial locations), the
# "Winter Low" label is replaced by this neutral one. The gate: relative
# amplitude of monthly-mean irradiance, measured 0.91-2.05 at the demo
# locations vs 0.37 at Nairobi — 0.6 splits with wide margin.
REGIME_LOW_SEASONALITY_NAME = "Low Irradiance"
SEASONAL_AMPLITUDE_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Demo locations — pre-cached so a live demo never depends on API latency
# ---------------------------------------------------------------------------
DEMO_LOCATIONS = {
    "Bristol, UK":        {"lat": 51.4545, "lon": -2.5879},   # temperate/mixed
    "Phoenix, AZ, USA":   {"lat": 33.4484, "lon": -112.0740}, # sunny/arid
    "Bergen, Norway":     {"lat": 60.3913, "lon": 5.3221},    # cloudy/wet
    "Kuala Lumpur, MY":   {"lat": 3.14,    "lon": 101.69},    # equatorial/humid
}

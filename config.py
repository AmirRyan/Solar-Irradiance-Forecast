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
    "CLRSKY_SFC_SW_DWN",   # clear-sky irradiance (for clearness index)
    "T2M",                 # temperature at 2m
    "RH2M",                # relative humidity at 2m
    "WS2M",                # wind speed at 2m
    "PRECTOTCORR",         # precipitation
    "CLOUD_AMT",           # cloud amount
]
HISTORY_START = "20150101"  # ~10 years of daily history per location
TARGET_COL = "ALLSKY_SFC_SW_DWN"

# ---------------------------------------------------------------------------
# Feature engineering (data.py builds exactly these; models consume FEATURES)
# ---------------------------------------------------------------------------
LAG_DAYS = [1, 2, 3, 7, 14]
ROLLING_WINDOWS = [3, 7, 30]

FEATURES = (
    [f"{TARGET_COL}_lag{l}" for l in LAG_DAYS]
    + [f"{TARGET_COL}_roll{w}_mean" for w in ROLLING_WINDOWS]
    + [f"{TARGET_COL}_roll{w}_std" for w in ROLLING_WINDOWS]
    + [
        "clearness_index_lag1",  # ALLSKY / CLRSKY, lagged (no leakage)
        "T2M_lag1",
        "RH2M_lag1",
        "WS2M_lag1",
        "PRECTOTCORR_lag1",
        "CLOUD_AMT_lag1",
        "doy_sin",               # seasonal encodings
        "doy_cos",
    ]
)

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
QUANTILES = [0.1, 0.5, 0.9]  # P10 / P50 / P90

# OPEN QUESTION (see CLAUDE.md): tuned once on demo locations in week 2, then
# reused fixed at inference. The "fits in seconds" claim depends on this.
LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "verbosity": -1,
}

# Chronological split fractions (train / val / test)
SPLIT_FRACTIONS = (0.7, 0.15, 0.15)

# Weather regimes (cluster.py maps cluster ids -> these labels)
REGIME_NAMES = ["Clear Sky", "Overcast", "Transitional", "Winter Low"]
N_REGIMES = len(REGIME_NAMES)

# ---------------------------------------------------------------------------
# Demo locations — pre-cached so a live demo never depends on API latency
# ---------------------------------------------------------------------------
DEMO_LOCATIONS = {
    "Bristol, UK":        {"lat": 51.4545, "lon": -2.5879},   # temperate/mixed
    "Phoenix, AZ, USA":   {"lat": 33.4484, "lon": -112.0740}, # sunny/arid
    "Bergen, Norway":     {"lat": 60.3913, "lon": 5.3221},    # cloudy/wet
}

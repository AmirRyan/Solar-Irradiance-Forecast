"""NASA POWER fetch (any lat/lon), per-location caching, features, split.

The fetch is the latency bottleneck (not the model fit), so every location's
raw history is cached to parquet after the first request.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import requests

import config
from src.solar import toa_irradiance


# ---------------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------------
def _cache_path(lat: float, lon: float):
    """Cache key rounded to 2dp — NASA POWER's grid is ~0.5°, so nearby
    coordinates share data anyway."""
    return config.CACHE_DIR / f"power_{lat:.2f}_{lon:.2f}.parquet"


def _parse_power_response(payload: dict) -> pd.DataFrame:
    """Turn a NASA POWER daily/point JSON payload into a clean frame:
    sorted DatetimeIndex, one column per parameter, -999 fill values
    converted to NaN, and rows without a target value (trailing
    not-yet-processed days) dropped."""
    raw = payload["properties"]["parameter"]

    df = pd.DataFrame({p: pd.Series(vals) for p, vals in raw.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.sort_index()
    # NASA POWER uses -999 for missing / not-yet-available days
    return df.replace(-999, np.nan).dropna(subset=[config.TARGET_COL])


def fetch_history(lat: float, lon: float, force_refresh: bool = False) -> pd.DataFrame:
    """Return the daily NASA POWER history for a location, cached after the
    first request. Index: DatetimeIndex. Columns: config.NASA_POWER_PARAMETERS.
    """
    path = _cache_path(lat, lon)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    resp = requests.get(
        config.NASA_POWER_BASE_URL,
        params={
            "parameters": ",".join(config.NASA_POWER_PARAMETERS),
            "community": "RE",
            "latitude": lat,
            "longitude": lon,
            "start": config.HISTORY_START,
            "end": date.today().strftime("%Y%m%d"),
            "format": "JSON",
        },
        timeout=120,
    )
    resp.raise_for_status()
    df = _parse_power_response(resp.json())

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


# ---------------------------------------------------------------------------
# Feature engineering — builds exactly config.FEATURES, nothing else
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame, lat: float) -> pd.DataFrame:
    """Add lag / rolling / seasonal features for next-day forecasting.

    `lat` is the location's latitude, needed for the analytic TOA irradiance
    that anchors the clearness index (GHI / TOA) — the API's
    CLRSKY_SFC_SW_DWN is CERES-derived and lags ~3 months, so it is no
    longer fetched.

    All features are lagged or rolling-on-past-values only: predicting day t
    uses information available at the end of day t-1 (no leakage).

    Sorts its input defensively — shift/rolling features are only correct on
    a chronologically ordered frame, so the contract is total rather than
    assuming callers pre-sort.
    """
    out = df.sort_index()
    y = out[config.TARGET_COL]

    for l in config.LAG_DAYS:
        out[f"{config.TARGET_COL}_lag{l}"] = y.shift(l)

    for w in config.ROLLING_WINDOWS:
        shifted = y.shift(1)  # window ends at t-1
        out[f"{config.TARGET_COL}_roll{w}_mean"] = shifted.rolling(w).mean()
        out[f"{config.TARGET_COL}_roll{w}_std"] = shifted.rolling(w).std()

    # Clearness index k_t = GHI / analytic TOA irradiance. TOA is pure solar
    # geometry (never missing, defined everywhere); where it is 0 (polar
    # night) k_t is undefined -> NaN, handled by the dropna below.
    h0 = toa_irradiance(lat, out.index.dayofyear)
    clearness = (out[config.TARGET_COL] / np.where(h0 > 0, h0, np.nan)).clip(0, 1.2)
    out["clearness_index_lag1"] = clearness.shift(1)

    for col in config.EXOG_COLS:
        out[f"{col}_lag1"] = out[col].shift(1)

    doy = out.index.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return out.dropna(subset=config.FEATURES)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split (never shuffle a time series)."""
    f_train, f_val, _ = config.SPLIT_FRACTIONS
    n = len(df)
    i1 = int(n * f_train)
    i2 = int(n * (f_train + f_val))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def load_location(lat: float, lon: float):
    """fetch -> features -> split, for one location. Convenience used by
    pipeline.py; keep any new preprocessing steps here or in the functions
    above so both callers stay in sync."""
    df = engineer_features(fetch_history(lat, lon), lat)
    return chronological_split(df)

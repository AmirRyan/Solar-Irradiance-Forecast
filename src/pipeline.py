"""THE single code path: fetch -> features -> fit -> predict for one location.

Both train.py (offline, demo locations) and dashboard/app.py (on-demand, live)
call run_location(). Per CLAUDE.md: do NOT duplicate any of this logic in
either caller — add new steps here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from src import data
from src.baseline import RidgeBaseline, persistence_forecast
from src.cluster import RegimeClusterer, per_regime_rmse
from src.model_gbm import QuantileGBM


# ---------------------------------------------------------------------------
# Metrics (kept here so every caller reports identically)
# ---------------------------------------------------------------------------
def rmse(y: pd.Series, yhat: pd.Series) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: pd.Series, yhat: pd.Series) -> float:
    return float(np.mean(np.abs(y - yhat)))


def pinball_loss(y: pd.Series, yhat: pd.Series, q: float) -> float:
    diff = y - yhat
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class LocationResult:
    lat: float
    lon: float
    test: pd.DataFrame                  # engineered test rows (incl. target)
    forecasts: pd.DataFrame             # p10/p50/p90 on the test set
    baselines: pd.DataFrame             # persistence + ridge on the test set
    regimes: pd.Series                  # regime label per test day
    metrics: dict = field(default_factory=dict)        # model + baseline metrics
    regime_breakdown: pd.DataFrame | None = None        # per-regime RMSE (P50)
    gbm: QuantileGBM | None = None       # fitted model (for explain.py / SHAP)


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------
def run_location(lat: float, lon: float) -> LocationResult:
    """Fetch (cached), engineer features, fit baselines + LightGBM, predict on
    the held-out test set, cluster regimes, compute metrics.

    Fit takes seconds; the NASA POWER fetch dominates latency on first request
    for a new location (the dashboard shows a loading state for this).
    """
    train, val, test = data.load_location(lat, lon)
    fit_df = pd.concat([train, val])  # val reserved for future tuning passes

    # --- models -----------------------------------------------------------
    gbm = QuantileGBM().fit(fit_df)
    ridge = RidgeBaseline().fit(fit_df)

    forecasts = gbm.predict(test)
    baselines = pd.concat(
        [persistence_forecast(test), ridge.predict(test)], axis=1
    )

    # --- regimes (analysis layer) ------------------------------------------
    # Refit per location by design (resolved phase 2 — see cluster.py header).
    regimes = RegimeClusterer().fit(fit_df).predict(test)

    # --- metrics: baselines alongside every headline number -----------------
    y = test[config.TARGET_COL]
    metrics = {
        "persistence_rmse": rmse(y, baselines["persistence"]),
        "ridge_rmse": rmse(y, baselines["ridge"]),
        "gbm_p50_rmse": rmse(y, forecasts["p50"]),
        "persistence_mae": mae(y, baselines["persistence"]),
        "ridge_mae": mae(y, baselines["ridge"]),
        "gbm_p50_mae": mae(y, forecasts["p50"]),
        **{
            f"gbm_pinball_p{int(q * 100)}": pinball_loss(
                y, forecasts[f"p{int(q * 100)}"], q
            )
            for q in config.QUANTILES
        },
        "test_days": int(len(test)),
    }

    return LocationResult(
        lat=lat,
        lon=lon,
        test=test,
        forecasts=forecasts,
        baselines=baselines,
        regimes=regimes,
        metrics=metrics,
        regime_breakdown=per_regime_rmse(y, forecasts["p50"], regimes),
        gbm=gbm,
    )

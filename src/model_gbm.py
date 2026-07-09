"""Headline model: LightGBM quantile regression (P10/P50/P90).

One LGBMRegressor per quantile, all on config.FEATURES. Hyperparameters come
from config.LGBM_PARAMS — tuned once on the demo locations and reused fixed
at inference (see the open-questions section of CLAUDE.md; the "fits in
seconds" claim depends on NOT retuning per location).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import config


class QuantileGBM:
    def __init__(self, quantiles: list[float] | None = None, **overrides):
        self.quantiles = quantiles or config.QUANTILES
        params = {**config.LGBM_PARAMS, **overrides}
        self.models = {
            q: LGBMRegressor(objective="quantile", alpha=q, **params)
            for q in self.quantiles
        }

    def fit(self, train: pd.DataFrame) -> "QuantileGBM":
        X, y = train[config.FEATURES], train[config.TARGET_COL]
        for m in self.models.values():
            m.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Columns: p10, p50, p90 (per config.QUANTILES).

        Quantile models are fit independently, so raw predictions can cross
        (p10 > p50 on some rows); we sort per-row to restore monotonicity —
        a standard post-hoc fix worth mentioning in the writeup.
        """
        X = df[config.FEATURES]
        preds = np.column_stack([self.models[q].predict(X) for q in self.quantiles])
        preds = np.sort(preds, axis=1)
        cols = [f"p{int(q * 100)}" for q in self.quantiles]
        return pd.DataFrame(preds, index=df.index, columns=cols)

    @property
    def median_model(self) -> LGBMRegressor:
        """The P50 model — the one explain.py runs SHAP on."""
        return self.models[0.5]

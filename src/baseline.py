"""Baselines: persistence (naive) + Ridge.

The floor every other model must beat. Per CLAUDE.md, report these alongside
every headline result — never show a bare number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


def persistence_forecast(df: pd.DataFrame) -> pd.Series:
    """Naive baseline: tomorrow = today. Uses the precomputed lag-1 feature so
    it is evaluated on exactly the same rows as every other model."""
    return df[f"{config.TARGET_COL}_lag1"].rename("persistence")


class RidgeBaseline:
    """Linear baseline on the shared feature set."""

    def __init__(self, alpha: float = 1.0):
        self.model = Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
        )

    def fit(self, train: pd.DataFrame) -> "RidgeBaseline":
        self.model.fit(train[config.FEATURES], train[config.TARGET_COL])
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        pred = self.model.predict(df[config.FEATURES])
        return pd.Series(np.asarray(pred), index=df.index, name="ridge")

"""SHAP analysis for the LightGBM (P50) model.

Two artifacts:
- global feature importance (mean |SHAP| per feature)
- per-prediction explanation for a chosen day (for the dashboard / writeup)
"""

from __future__ import annotations

import pandas as pd
import shap

import config
from src.model_gbm import QuantileGBM


def shap_values(model: QuantileGBM, df: pd.DataFrame) -> pd.DataFrame:
    """SHAP values for the P50 model on df. Returns a frame aligned with df
    (rows = days, cols = features)."""
    explainer = shap.TreeExplainer(model.median_model)
    vals = explainer.shap_values(df[config.FEATURES])
    return pd.DataFrame(vals, index=df.index, columns=config.FEATURES)


def global_importance(shap_df: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature, descending — the headline importance chart."""
    return shap_df.abs().mean().sort_values(ascending=False)


def explain_day(shap_df: pd.DataFrame, day) -> pd.Series:
    """Per-prediction explanation: signed SHAP contributions for one day,
    sorted by absolute magnitude."""
    row = shap_df.loc[day]
    return row.reindex(row.abs().sort_values(ascending=False).index)

"""SHAP analysis for the LightGBM (P50) model.

Two artifacts:
- global feature importance (mean |SHAP| per feature)
- per-prediction explanation for a chosen day (for the dashboard / writeup)

save_global_importance_artifacts() is the offline entry point used by
train.py. Whether SHAP also runs per-request in the dashboard is an open
phase-3 latency question (see CLAUDE.md) — do not wire it into pipeline.py
until that is resolved.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import shap

matplotlib.use("Agg")  # artifact rendering only, never an interactive window
import matplotlib.pyplot as plt

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


def save_global_importance_artifacts(
    model: QuantileGBM, df: pd.DataFrame, out_dir: Path, tag: str
) -> tuple[Path, Path]:
    """Write global importance (mean |SHAP| on df) as CSV + bar-plot PNG.
    Returns (csv_path, png_path)."""
    importance = global_importance(shap_values(model, df))

    csv_path = out_dir / f"shap_importance_{tag}.csv"
    importance.rename("mean_abs_shap").to_csv(csv_path, index_label="feature")

    ranked = importance.sort_values()  # barh plots bottom-up: biggest ends on top
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(ranked) + 1.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.barh(ranked.index, ranked.values, color="#2a78d6", height=0.62)
    for i, v in enumerate(ranked.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8, color="#52514e")
    ax.set_title(f"Mean |SHAP| — P50 model, {tag}", loc="left", color="#0b0b0b")
    ax.set_xlabel("mean |SHAP| (kWh/m²/day)", color="#52514e")
    ax.tick_params(colors="#52514e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()

    png_path = out_dir / f"shap_importance_{tag}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return csv_path, png_path

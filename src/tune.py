"""Offline hyperparameter search for the quantile LightGBM.

Run: python -m src.tune [--trials N]

OFFLINE-ONLY, like train.py — never called by pipeline.py or the
dashboard. Hyperparameters are tuned ONCE on the demo locations and the
winners are written into config.LGBM_PARAMS by hand (config stays the
single source of truth; this script prints the dict to paste). They are
then reused fixed at inference — the "fits in seconds" claim depends on
not retuning per location.

Method: seeded random search. No optuna dependency for a run-once script;
at 7 search dimensions and a <=100-trial budget, random search is
competitive with TPE and trivially reproducible. Objective: pinball loss
averaged over the three quantiles and the three demo locations, evaluated
on the VAL split (fit on train only). The test split stays untouched
until final evaluation in train.py.

P10-P90 empirical coverage on val is tracked per trial as a diagnostic
for the known undercoverage failure mode, but the objective is pinball —
coverage is handled separately (see model_gbm.py conformal step if
present, or the calibration notes in CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import config
from src import data
from src.model_gbm import QuantileGBM
from src.pipeline import pinball_loss

# Search space. sklearn-API names; the LightGBM-native aliases are
# feature_fraction (colsample_bytree), lambda_l1/l2 (reg_alpha/lambda).
def _sample_params(rng: np.random.Generator) -> dict:
    return {
        "num_leaves": int(rng.choice([8, 15, 31, 63])),
        "min_child_samples": int(np.exp(rng.uniform(np.log(10), np.log(150)))),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.1)))),
        "n_estimators": int(np.exp(rng.uniform(np.log(150), np.log(1200)))),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "reg_alpha": float(np.exp(rng.uniform(np.log(1e-3), np.log(10)))),
        "reg_lambda": float(np.exp(rng.uniform(np.log(1e-3), np.log(30)))),
    }


def _evaluate(overrides: dict, splits: list) -> tuple[float, float]:
    """Mean val pinball (the objective) and mean val P10-P90 coverage
    (diagnostic) across demo locations."""
    losses, coverages = [], []
    for train, val in splits:
        preds = QuantileGBM(**overrides).fit(train).predict(val)
        y = val[config.TARGET_COL]
        losses += [
            pinball_loss(y, preds[f"p{int(q * 100)}"], q) for q in config.QUANTILES
        ]
        coverages.append(float(((y >= preds["p10"]) & (y <= preds["p90"])).mean()))
    return float(np.mean(losses)), float(np.mean(coverages))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    splits = []
    for name, loc in config.DEMO_LOCATIONS.items():
        train, val, _test = data.load_location(loc["lat"], loc["lon"])
        splits.append((train, val))
        print(f"loaded {name}: train={len(train)} val={len(val)}")

    # Baseline: the params currently in config, for the before/after story.
    base_loss, base_cov = _evaluate({}, splits)
    print(f"\ncurrent config.LGBM_PARAMS: val pinball={base_loss:.4f} "
          f"coverage={base_cov:.3f}\n")

    rng = np.random.default_rng(args.seed)
    results = []
    best = {"loss": base_loss, "coverage": base_cov, "params": None}
    for i in range(args.trials):
        params = _sample_params(rng)
        loss, cov = _evaluate(params, splits)
        results.append({"trial": i, "loss": loss, "coverage": cov, "params": params})
        marker = ""
        if loss < best["loss"]:
            best = {"loss": loss, "coverage": cov, "params": params}
            marker = "  <-- best"
        print(f"trial {i:3d}  pinball={loss:.4f}  coverage={cov:.3f}{marker}")

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.ARTIFACTS_DIR / "tuning_results.json"
    out.write_text(json.dumps({"baseline": {"loss": base_loss, "coverage": base_cov},
                               "best": best, "trials": results}, indent=2))
    print(f"\nwrote {out}")

    if best["params"] is None:
        print("No trial beat the current config.LGBM_PARAMS — keep them.")
        return
    print(f"\nBest: val pinball {base_loss:.4f} -> {best['loss']:.4f} "
          f"({100 * (1 - best['loss'] / base_loss):.1f}% better), "
          f"val coverage {base_cov:.3f} -> {best['coverage']:.3f}")
    print("\nPaste into config.LGBM_PARAMS (keep verbosity: -1):")
    print(json.dumps(best["params"], indent=4))


if __name__ == "__main__":
    main()

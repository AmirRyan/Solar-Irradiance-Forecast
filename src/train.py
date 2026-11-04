"""Offline/batch runs for the pre-cached demo locations.

Run: python -m src.train

For each demo location in config.DEMO_LOCATIONS this warms the NASA POWER
cache (so a live demo never depends on API latency), runs the shared
pipeline, and writes metrics to artifacts/. All modeling logic lives in
src/pipeline.py — this file only orchestrates and reports.

If the optional torch extra is installed (requirements-seq.txt) it also fits
the phase-4 attention LSTM per preset and saves its forecasts for the
dashboard's second line.
"""

from __future__ import annotations

import importlib.util
import json

import config
from src.explain import save_global_importance_artifacts
from src.pipeline import run_location, seq_forecast_path, slugify


def main() -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, dict] = {}

    with_seq = importlib.util.find_spec("torch") is not None
    if not with_seq:
        print("torch not installed: skipping the LSTM (pip install -r requirements-seq.txt)")

    for name, loc in config.DEMO_LOCATIONS.items():
        print(f"\n=== {name} ({loc['lat']}, {loc['lon']}) ===")
        result = run_location(loc["lat"], loc["lon"], with_seq=with_seq)

        # Baselines alongside every headline number — never a bare metric.
        m = result.metrics
        print(f"  RMSE  persistence={m['persistence_rmse']:.3f}  "
              f"ridge={m['ridge_rmse']:.3f}  gbm_p50={m['gbm_p50_rmse']:.3f}")
        print(f"  MAE   persistence={m['persistence_mae']:.3f}  "
              f"ridge={m['ridge_mae']:.3f}  gbm_p50={m['gbm_p50_mae']:.3f}")
        if with_seq:
            print(f"  LSTM vs LightGBM on {m['seq_common_days']} common days:")
            for tag in ("gbm_common", "seq"):
                print(f"    {tag:10s} RMSE={m[f'{tag}_rmse']:.3f}  MAE={m[f'{tag}_mae']:.3f}  "
                      f"coverage={m[f'{tag}_coverage']:.3f}  pinball="
                      f"{m[f'{tag}_pinball_p10']:.3f}/{m[f'{tag}_pinball_p50']:.3f}/"
                      f"{m[f'{tag}_pinball_p90']:.3f}")
            result.seq_forecasts.to_parquet(seq_forecast_path(name))
        print("  Per-regime RMSE (P50):")
        print(result.regime_breakdown.to_string(index=False))

        # SHAP artifacts (offline only — dashboard use is an open question)
        csv_path, png_path = save_global_importance_artifacts(
            result.gbm, result.test, config.ARTIFACTS_DIR, slugify(name)
        )
        print(f"  SHAP: {csv_path.name}, {png_path.name}")

        all_metrics[name] = m

    out = config.ARTIFACTS_DIR / "demo_metrics.json"
    out.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

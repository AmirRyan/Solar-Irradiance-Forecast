"""Offline/batch runs for the pre-cached demo locations.

Run: python -m src.train

For each demo location in config.DEMO_LOCATIONS this warms the NASA POWER
cache (so a live demo never depends on API latency), runs the shared
pipeline, and writes metrics to artifacts/. All modeling logic lives in
src/pipeline.py — this file only orchestrates and reports.
"""

from __future__ import annotations

import json

import config
from src.pipeline import run_location


def main() -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, dict] = {}

    for name, loc in config.DEMO_LOCATIONS.items():
        print(f"\n=== {name} ({loc['lat']}, {loc['lon']}) ===")
        result = run_location(loc["lat"], loc["lon"])

        # Baselines alongside every headline number — never a bare metric.
        m = result.metrics
        print(f"  RMSE  persistence={m['persistence_rmse']:.3f}  "
              f"ridge={m['ridge_rmse']:.3f}  gbm_p50={m['gbm_p50_rmse']:.3f}")
        print(f"  MAE   persistence={m['persistence_mae']:.3f}  "
              f"ridge={m['ridge_mae']:.3f}  gbm_p50={m['gbm_p50_mae']:.3f}")
        print("  Per-regime RMSE (P50):")
        print(result.regime_breakdown.to_string(index=False))

        all_metrics[name] = m

    out = config.ARTIFACTS_DIR / "demo_metrics.json"
    out.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

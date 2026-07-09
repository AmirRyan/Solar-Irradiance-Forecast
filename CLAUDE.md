# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

No code exists in this repo yet — it is being built from scratch following the plan captured in this file. Update the Commands section once `requirements.txt`, `src/train.py`, and `dashboard/app.py` actually exist.

## What this project is

A solo portfolio project: probabilistic solar irradiance forecasting using NASA POWER data. Location is a runtime input (lat/lon via map click, preset dropdown, or manual entry) — the app fetches and fits per-location on demand. It is not a fixed-location dashboard.

**Core stack:**

1. **Baselines** — persistence (naive) and Ridge. The floor every other model must beat.
2. **Headline model** — LightGBM quantile regression predicting P10/P50/P90 irradiance, giving calibrated uncertainty. Fast to train, easy to iterate on.
3. **Stretch model (week 4, conditional)** — a compact attention LSTM or Temporal Fusion Transformer-lite for multi-horizon forecasting, built as a classical-vs-deep-learning comparison. It does not need to beat LightGBM to be a useful result.
4. **Weather regime clustering (KMeans/GMM)** — Clear Sky, Overcast, Transitional, Winter Low. An analysis layer for per-regime error breakdown, not a modeling input.
5. **Interpretability** — SHAP feature importance and per-prediction explanations for the LightGBM model.
6. **Web app** — Plotly Dash: location input, forecast vs. actual, uncertainty bands, regime coloring, residuals panel, loading state for new-location fetches.

## Commands (once implemented)

- `pip install -r requirements.txt` — install dependencies
- `python -m src.train` — offline/batch runs: fits baselines + LightGBM + clustering for the pre-cached demo locations
- `python dashboard/app.py` — launch the Plotly Dash web app

No test suite or linter is defined yet. Add this section once one exists — don't assume pytest/ruff/etc. without confirming.

## Repo structure

```
solar-forecast/
├── config.py              # shared paths, feature list, cache dir, preset demo locations
├── src/
│   ├── data.py            # NASA POWER fetch (any lat/lon), per-location caching, feature engineering, split
│   ├── baseline.py        # persistence + Ridge
│   ├── model_gbm.py       # LightGBM quantile regression
│   ├── model_seq.py       # attention LSTM / TFT-lite (week 4 stretch)
│   ├── cluster.py         # weather regime clustering + per-regime metrics
│   ├── explain.py         # SHAP analysis
│   ├── pipeline.py        # fetch → feature engineer → fit → predict, for one location on demand
│   └── train.py           # batch/offline runs for the pre-cached demo locations
├── dashboard/
│   └── app.py             # Plotly Dash app: location input, calls pipeline.py, renders results
├── notebooks/
│   └── eda.ipynb          # exploratory analysis, kept separate from production code
├── requirements.txt
└── README.md              # problem framing, results, how to run
```

## Architecture

### One pipeline, two callers

`src/pipeline.py` is the single function that does fetch → feature-engineer → fit → predict for one location. Both `train.py` (offline, pre-cached demo locations) and `dashboard/app.py` (on-demand, live) call it. This exists specifically to prevent drift between the training script and the live app — a problem in the v1 version of this project. Do not duplicate this logic in either caller; add new steps to `pipeline.py` itself, so there is exactly one code path for "given a location, produce a forecast."

### config.py is the single source of truth

Feature list, cache directory paths, and preset demo locations live in `config.py` from day one. Don't redefine these in individual modules.

### Model layering

- `baseline.py` — persistence (naive) + Ridge. Report alongside every headline result, never a bare number.
- `model_gbm.py` — LightGBM quantile regression (P10/P50/P90). The headline model.
- `model_seq.py` — attention LSTM / TFT-lite. Week-4 stretch, a classical-vs-deep-learning comparison; not required to beat LightGBM.
- `cluster.py` — KMeans/GMM weather-regime clustering (Clear Sky, Overcast, Transitional, Winter Low). Analysis layer only.
- `explain.py` — SHAP for the LightGBM model.

### Per-location fetch/fit design

`data.py` fetches NASA POWER history for a given lat/lon, caches it, then baseline + LightGBM are fit on the spot. The fit is fast (seconds, since it's tabular data); the NASA POWER fetch is the actual latency bottleneck, so:

- Repeat requests for a location hit the cache and are instant.
- The dashboard shows a loading state for first-time requests to a new location.

This design was chosen deliberately over one global multi-location model. A global model ("generalizes across geography, instant inference everywhere") requires assembling and cleaning multi-location training data and getting a model to condition well on lat/lon/climate — a meaningfully bigger, riskier undertaking. It is explicitly out of scope and noted as future work.

### Location input

Lat/lon via map click or numeric entry, plus a preset dropdown of 5–8 notable cities/climates — no free-text place names, to avoid a geocoding dependency. If free-text geocoding is later wanted, it's a small addition (a free geocoding API), not a rearchitecture.

### Demo locations

2–3 locations (Bristol plus one sunny and one cloudy climate) are kept pre-cached specifically so a live demo never depends on NASA POWER API latency.

## Build schedule (4 weeks)

**Phase 1 — Data + baselines.** `data.py` accepts arbitrary lat/lon and fetches/caches NASA POWER data. Feature engineering: lags, rolling stats, seasonal encodings. Train/val/test split. Persistence + Ridge baselines running end-to-end for the demo locations. Deliverable: a working, boring pipeline with a first RMSE number, provably location-agnostic (test it on two different locations).

**Phase 2 — Headline model.** LightGBM quantile regression (P10/P50/P90). Hyperparameter pass. SHAP analysis. Weather regime clustering + per-regime RMSE. `pipeline.py` wraps fetch → features → fit → predict as one callable function. Deliverable: calibrated forecasts beating baseline, with an interpretability story, for any location.

**Phase 3 — Web app.** Plotly Dash: location input (map click + preset dropdown + manual lat/lon), forecast vs. actual, uncertainty bands, regime coloring, residuals, loading state for new-location fetches. This is the demoable artifact — prioritize finishing it over the stretch model. Deliverable: a working web app that can be screen-shared and used live.

**Phase 4 — Stretch model + polish.** If on schedule: attention LSTM/TFT-lite as a comparison model, added to the dashboard as a second forecast line. If behind schedule: skip it and spend the week on README, error analysis writeup, and documentation. Decide which path *before* week 4 starts to avoid a last-minute scramble — both outcomes are acceptable.

## Evaluation conventions

- Metrics: RMSE, MAE, and pinball loss (for the quantile forecasts) on the held-out test set.
- Baselines reported alongside every model result — never show a number without its floor.
- Per-regime RMSE breakdown (via `cluster.py`) is the primary error-analysis lens — prefer it over aggregate metrics alone when writing up results.

## Open design questions (resolve before building the relevant piece)

- **Stretch model scope**: does `model_seq.py` ever run against on-demand/user-typed locations, or only against the pre-cached demo locations via `train.py`? Training an LSTM/TFT-lite per-request isn't feasible at interactive latency — this likely needs to be demo-locations-only, but confirm before wiring it into `dashboard/app.py`.
- **LightGBM hyperparameters**: tuned once (in phase 2, on demo locations) and reused fixed at inference, or retuned per new location? The "fits in seconds" claim only holds under the fixed-hyperparameter interpretation.
- **Regime clustering fit strategy**: are KMeans/GMM regimes fit once on demo-location data and applied (predict) to new locations, or refit per-location on whatever's cached? Refitting on a single new location's limited history risks unstable clusters.
- **Phase 4 go/no-go**: the stretch model is explicitly conditional — decide based on how phase 1–3 actually go, not now.

## Current Status: 


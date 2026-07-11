# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Phase 2 (headline model) is complete: `config.LGBM_PARAMS` holds hyperparameters tuned once by `src/tune.py` (val pinball −5.8%, P10–P90 test coverage 0.72–0.78 vs 0.62–0.68 before — no conformal step needed), regime clustering is resolved as refit-per-location, and `train.py` writes SHAP importance artifacts per demo location. Results tables live in README.md. Next up is phase 3 (the Dash web app). Note: `CLOUD_AMT` was deliberately dropped from the fetched parameters — it is CERES-derived and lags ~3 months behind the near-real-time parameters (see `config.py`); cloudiness is captured via the clearness index instead.

## What this project is

A solo portfolio project: probabilistic solar irradiance forecasting using NASA POWER data. Location is a runtime input (lat/lon via map click, preset dropdown, or manual entry) — the app fetches and fits per-location on demand. It is not a fixed-location dashboard.

**Core stack:**

1. **Baselines** — persistence (naive) and Ridge. The floor every other model must beat.
2. **Headline model** — LightGBM quantile regression predicting P10/P50/P90 irradiance, giving calibrated uncertainty. Fast to train, easy to iterate on.
3. **Stretch model (conditional)** — a compact attention LSTM or Temporal Fusion Transformer-lite for multi-horizon forecasting, built as a classical-vs-deep-learning comparison. It does not need to beat LightGBM to be a useful result.
4. **Weather regime clustering (KMeans/GMM)** — Clear Sky, Overcast, Transitional, Winter Low. An analysis layer for per-regime error breakdown, not a modeling input.
5. **Interpretability** — SHAP feature importance and per-prediction explanations for the LightGBM model.
6. **Web app** — Plotly Dash: location input, forecast vs. actual, uncertainty bands, regime coloring, residuals panel, loading state for new-location fetches.

## Commands

- `pip install -r requirements.txt` — install dependencies (on macOS, LightGBM also needs `brew install libomp`)
- `python -m src.train` — offline/batch runs: fits baselines + LightGBM + clustering for the pre-cached demo locations (hits the NASA POWER API on cache misses), writes metrics + SHAP artifacts to `artifacts/`
- `python -m src.tune` — offline hyperparameter search (~35 min for 80 trials); winners are pasted into `config.LGBM_PARAMS` by hand. Run rarely — params are tuned once, not per location
- `python -m pytest tests/` — run the test suite (synthetic data only, no network); run from the repo root
- `python dashboard/app.py` — launch the Plotly Dash web app (phase 3, not yet implemented)

No linter is configured yet.

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
├── tests/                 # network-free pytest suite (synthetic fixtures, see Testing conventions)
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

## Build schedule

**Phase 1 — Data + baselines.** `data.py` accepts arbitrary lat/lon and fetches/caches NASA POWER data. Feature engineering: lags, rolling stats, seasonal encodings. Train/val/test split. Persistence + Ridge baselines running end-to-end for the demo locations. Deliverable: a working, boring pipeline with a first RMSE number, provably location-agnostic (test it on two different locations).

**Phase 2 — Headline model.** LightGBM quantile regression (P10/P50/P90). Hyperparameter pass. SHAP analysis. Weather regime clustering + per-regime RMSE. `pipeline.py` wraps fetch → features → fit → predict as one callable function. Deliverable: calibrated forecasts beating baseline, with an interpretability story, for any location.

**Phase 3 — Web app.** Plotly Dash: location input (map click + preset dropdown + manual lat/lon), forecast vs. actual, uncertainty bands, regime coloring, residuals, loading state for new-location fetches. This is the demoable artifact — prioritize finishing it over the stretch model. Deliverable: a working web app that can be screen-shared and used live.

**Phase 4 — Stretch model + polish.** If on schedule: attention LSTM/TFT-lite as a comparison model, added to the dashboard as a second forecast line. If behind schedule: skip it and spend the week on README, error analysis writeup, and documentation. Decide which path *before* week 4 starts to avoid a last-minute scramble — both outcomes are acceptable.

## NASA POWER API facts (hard-won — verified against the live API)

- Daily/point endpoint response shape: values live at `properties.parameter.<PARAM>.<YYYYMMDD>`; the fill value for missing / not-yet-processed days is `-999.0`.
- Near-real-time parameters (irradiance, T2M, etc.) lag ~5–7 days behind today; those trailing days arrive as `-999` and are dropped via the target-column dropna.
- `CLOUD_AMT` is a valid RE-community parameter but is CERES-derived and lags **~3 months** — keeping it as a feature would silently drop the most recent ~100 days per location, which is why it was removed (cloudiness is captured by the clearness index).
- All parsing of this payload lives in `data._parse_power_response` (a pure function, extracted so it is testable without the network). Don't re-handle `-999` elsewhere.

## Testing conventions

- **No network in any test.** Everything runs on synthetic data; the NASA POWER payload handling is tested by feeding handcrafted payload dicts (matching the real JSON shape above) to `data._parse_power_response`.
- **Schemas derive from `config`, never hardcoded.** Fixtures and payload builders construct their columns from `config.NASA_POWER_PARAMETERS` and assert set-equality, so they fail loudly if the parameter list changes instead of passing against a stale schema.
- **The `synthetic_raw` fixture is deliberately learnable** (seasonal clear-sky sine × cloudiness multiplier, fixed seed, ~6.5 years). The GBM coverage assertion depends on both the learnability and the size — small fixtures make the quantile models memorize the training set and read artificially low coverage.
- **Leakage testing**: perturb every raw input column (parametrized) at a day t and assert all engineered feature rows ≤ t are unchanged — this catches un-lagged covariates and backward flow from future rows, and a failure names the leaking column.
- `engineer_features` sorts its input defensively (contract pinned by `test_engineer_features_sorts_unsorted_input`).
- Calibration (resolved phase 2): the original hand-picked `LGBM_PARAMS` undercovered (~0.62–0.68 P10–P90 coverage); the tuned params (strong regularization, see `config.py`) measure 0.72–0.78 on the real demo locations + Nairobi and ~0.73 on the synthetic fixture — no conformal post-hoc step was needed. The coverage test's lower bound is 0.65; if coverage regresses below that, suspect a params change before suspecting the test.

## Evaluation conventions

- Metrics: RMSE, MAE, and pinball loss (for the quantile forecasts) on the held-out test set.
- Baselines reported alongside every model result — never show a number without its floor.
- Per-regime RMSE breakdown (via `cluster.py`) is the primary error-analysis lens — prefer it over aggregate metrics alone when writing up results.

## Open design questions (resolve before building the relevant piece)

- **Stretch model scope**: does `model_seq.py` ever run against on-demand/user-typed locations, or only against the pre-cached demo locations via `train.py`? Training an LSTM/TFT-lite per-request isn't feasible at interactive latency — this likely needs to be demo-locations-only, but confirm before wiring it into `dashboard/app.py`.
- **SHAP in the dashboard**: `train.py` saves SHAP importance artifacts offline for the demo locations. Whether SHAP also runs per-request for on-demand locations in the dashboard is a phase-3 latency question (TreeExplainer on ~600 test rows takes a few seconds) — decide when building `dashboard/app.py`; do not wire `explain.py` into `pipeline.py` before then.
- **Phase 4 go/no-go**: the stretch model is explicitly conditional — decide based on how phase 1–3 actually go, not now.

## Resolved design decisions (phase 2)

- **LightGBM hyperparameters: tuned once, reused fixed.** `src/tune.py` (offline-only, like `train.py`) tuned on the demo locations' val splits; the winners live in `config.LGBM_PARAMS` and are reused unchanged for every location at inference. Do NOT add per-location tuning to `pipeline.py` — the "fits in seconds" latency claim depends on this.
- **Regime clustering: refit per location.** Decided empirically (demo locations + off-preset Nairobi): fit-once on pooled demo data produced degenerate regimes (<20 test days) at 3 of 4 probe locations (Phoenix Transitional 10d, Bergen Clear Sky 16d, Nairobi Overcast 14d) and transferred nonsensically to the equatorial climate (326 of Nairobi's 612 test days labeled "Winter Low" with the *highest* mean irradiance — the pooled clusters lean on day-of-year, which inverts meaning at the equator). Refit-per-location was non-degenerate everywhere (min 50 test days/regime); ~10 years of history per location makes refits stable. Known limitation: the cluster-*naming* heuristic is location-relative and can misfire outside temperate climates (equatorial locations still get a "Winter Low" label) — candidate phase-3 polish.


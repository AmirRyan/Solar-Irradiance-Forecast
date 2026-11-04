# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Phase 3 (web app) is complete: `dashboard/app.py` is a working Dash app — three verified input modes (preset / map click / manual lat/lon, resolved by which input fired so stale values never shadow the latest action), inline error handling for invalid input and NASA POWER failures, an in-process result memo (dashboard-layer LRU, keys match `data.py`'s 2dp cache rounding), honest P10–P90 band labeling with measured coverage in the caption, and a per-request SHAP panel (measured ~0.2s). Pure helpers live in `dashboard/helpers.py` (network-free tests in `tests/`). Phase 2 delivered: tuned `LGBM_PARAMS` via `src/tune.py` (val pinball −5.8%, P10–P90 test coverage 0.72–0.78 — no conformal step needed), refit-per-location clustering, SHAP artifacts from `train.py`. Results tables live in README.md. Phase 4 is built: an attention LSTM comparison (`src/model_seq.py`, optional torch extra) (presets read saved forecasts, custom locations fit live); it loses to LightGBM on RMSE/MAE/P50 pinball everywhere (see README) and that is an accepted result. Note: `CLOUD_AMT` and `CLRSKY_SFC_SW_DWN` were both deliberately dropped from the fetched parameters — they are CERES-derived and lag ~3 months behind the near-real-time parameters (see `config.py`); cloudiness is captured via the clearness index (GHI / analytic TOA, `src/solar.py`) instead. 2026-07-13 fix: the CLRSKY lag had been silently wiping most of the trailing ~3 months of test data at Bristol/Bergen — resolved (test days recovered, P10–P90 coverage re-measured at 0.73–0.77). 2026-07-14: caches now self-refresh via `CACHE_MAX_AGE_DAYS` (presets no longer freeze at their fetch date), and Kuala Lumpur joined as the fourth preset (equatorial-humid, pre-cached, gates to "Low Irradiance"); README metrics tables are from the 2026-07-15 run; the new LSTM-vs-LightGBM table is from a 2026-11-04 run (shifted test windows), so the two LightGBM sets differ slightly until the older tables are regenerated.

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
- `pip install -r requirements-seq.txt` — optional: PyTorch for the phase-4 LSTM. `python -m src.train` then also fits it per preset and saves `artifacts/seq_forecasts_<preset>.parquet`; without torch it skips the LSTM
- `python -m src.tune` — offline hyperparameter search (~35 min for 80 trials); winners are pasted into `config.LGBM_PARAMS` by hand. Run rarely — params are tuned once, not per location
- `python -m pytest tests/` — run the test suite (synthetic data only, no network); run from the repo root
- `python dashboard/app.py` — launch the Plotly Dash web app at http://127.0.0.1:8050 (pre-cache the demo locations with `python -m src.train` first so the presets respond instantly)

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

### Clearness index: GHI / analytic TOA irradiance

The clearness-index feature is `ALLSKY_SFC_SW_DWN / toa_irradiance(lat, doy)` — the Duffie & Beckman daily extraterrestrial-irradiance formula (`src/solar.py`, pure numpy: lat + day-of-year, no API dependency, defined at every location; polar night gives H0 = 0 → k_t = NaN). This replaced the API's `CLRSKY_SFC_SW_DWN` (see NASA POWER facts), which is why `engineer_features` takes `lat` explicitly. k_t values run lower than the old ALLSKY/CLRSKY ratio (TOA exceeds modeled clear-sky) — harmless for the model, and `cluster.py`'s naming heuristic only uses per-location relative order, verified post-switch (regimes non-degenerate at the presets, min 63 test days; Nairobi still gates to "Low Irradiance").

### Model layering

- `baseline.py` — persistence (naive) + Ridge. Report alongside every headline result, never a bare number.
- `model_gbm.py` — LightGBM quantile regression (P10/P50/P90). The headline model.
- `model_seq.py` — attention LSTM / TFT-lite. Week-4 stretch, a classical-vs-deep-learning comparison; not required to beat LightGBM.
- `cluster.py` — KMeans/GMM weather-regime clustering (Clear Sky, Overcast, Transitional, Winter Low). Analysis layer only.
- `explain.py` — SHAP for the LightGBM model.

### Per-location fetch/fit design

`data.py` fetches NASA POWER history for a given lat/lon, caches it, then baseline + LightGBM are fit on the spot. The fit is fast (seconds, since it's tabular data); the NASA POWER fetch is the actual latency bottleneck, so:

- Repeat requests for a location hit the cache and are instant.
- The dashboard shows a loading state for first-time requests to a new location (the `dcc.Loading` wraps the callback, so it also covers stale-cache refetches).
- Caches expire after `config.CACHE_MAX_AGE_DAYS` (10), measured against the last **observed** day in the parquet (last non-NaN target — never file mtime, and not the raw index max, since caches carry trailing all-NaN rows for NASA's unprocessed days). Stale caches refetch **in full** (no delta fetching — the fetch is seconds-scale and fires at most ~once per 10 days per location). If the refetch fails, the stale cache is served with a logged warning, so a live demo degrades to slightly-old data rather than an error; `force_refresh=True` always refetches and propagates errors. Staleness is checked on access only — the stray experiment caches (Nairobi, Marrakesh, Reykjavík, probe points, …) are deliberately left alone and self-refresh whenever next requested; there is no sweep or background refresh.

This design was chosen deliberately over one global multi-location model. A global model ("generalizes across geography, instant inference everywhere") requires assembling and cleaning multi-location training data and getting a model to condition well on lat/lon/climate — a meaningfully bigger, riskier undertaking. It is explicitly out of scope and noted as future work.

### Location input

Lat/lon via map click or numeric entry, plus a preset dropdown of 5–8 notable cities/climates — no free-text place names, to avoid a geocoding dependency. If free-text geocoding is later wanted, it's a small addition (a free geocoding API), not a rearchitecture.

### Demo locations

Four presets — Bristol (temperate/mixed), Phoenix (sunny/arid), Bergen (cloudy/wet), Kuala Lumpur (equatorial/humid) — are kept pre-cached specifically so a live demo never depends on NASA POWER API latency. KL also exercises the seasonal-amplitude gate in a shipped preset (measured relative monthly amplitude 0.21, far below the 0.6 threshold → its low cluster is named "Low Irradiance", not "Winter Low").

## Build schedule

**Phase 1 — Data + baselines.** `data.py` accepts arbitrary lat/lon and fetches/caches NASA POWER data. Feature engineering: lags, rolling stats, seasonal encodings. Train/val/test split. Persistence + Ridge baselines running end-to-end for the demo locations. Deliverable: a working, boring pipeline with a first RMSE number, provably location-agnostic (test it on two different locations).

**Phase 2 — Headline model.** LightGBM quantile regression (P10/P50/P90). Hyperparameter pass. SHAP analysis. Weather regime clustering + per-regime RMSE. `pipeline.py` wraps fetch → features → fit → predict as one callable function. Deliverable: calibrated forecasts beating baseline, with an interpretability story, for any location.

**Phase 3 — Web app.** Plotly Dash: location input (map click + preset dropdown + manual lat/lon), forecast vs. actual, uncertainty bands, regime coloring, residuals, loading state for new-location fetches. This is the demoable artifact — prioritize finishing it over the stretch model. Deliverable: a working web app that can be screen-shared and used live.

**Phase 4 — Stretch model + polish.** If on schedule: attention LSTM/TFT-lite as a comparison model, added to the dashboard as a second forecast line. If behind schedule: skip it and spend the week on README, error analysis writeup, and documentation. Decide which path *before* week 4 starts to avoid a last-minute scramble — both outcomes are acceptable.

## NASA POWER API facts (hard-won — verified against the live API)

- Daily/point endpoint response shape: values live at `properties.parameter.<PARAM>.<YYYYMMDD>`; the fill value for missing / not-yet-processed days is `-999.0`.
- Near-real-time parameters (irradiance, T2M, etc.) lag ~5–7 days behind today; those trailing days arrive as `-999` and are dropped by the target dropna at the end of feature engineering. **Invariant:** `config.CACHE_MAX_AGE_DAYS` must exceed this NRT lag by a real margin — a fetch made today already has a last observed day of ~today−6, so a threshold at or below the lag would make every request refetch in a loop. If either number is ever revised, re-check the other (pinned by `test_fresh_cache_never_triggers_refetch_loop`).
- `CLOUD_AMT` **and `CLRSKY_SFC_SW_DWN`** are valid RE-community parameters but are CERES-derived and lag **~3 months** (CLRSKY observed empirically: missing from 2026-03-31 onward in a 2026-07-09 fetch, at every location probed) — keeping either as a feature silently drops the most recent ~100 days per location, so neither is fetched. The clearness index uses analytic TOA irradiance as its denominator instead (see Architecture).
- All parsing of this payload lives in `data._parse_power_response` (a pure function, extracted so it is testable without the network). It returns a **continuous daily calendar**: `-999` becomes NaN, absent date keys come back as explicit all-NaN rows, and no rows are dropped in the parser. Don't re-handle `-999` elsewhere.

## Testing conventions

- **No network in any test.** Everything runs on synthetic data; the NASA POWER payload handling is tested by feeding handcrafted payload dicts (matching the real JSON shape above) to `data._parse_power_response`.
- **Schemas derive from `config`, never hardcoded.** Fixtures and payload builders construct their columns from `config.NASA_POWER_PARAMETERS` and assert set-equality, so they fail loudly if the parameter list changes instead of passing against a stale schema.
- **The `synthetic_raw` fixture is deliberately learnable** (seasonal clear-sky sine × cloudiness multiplier, fixed seed, ~6.5 years). The GBM coverage assertion depends on both the learnability and the size — small fixtures make the quantile models memorize the training set and read artificially low coverage.
- **Leakage testing**: perturb every raw input column (parametrized) at a day t and assert all engineered feature rows ≤ t are unchanged — this catches un-lagged covariates and backward flow from future rows, and a failure names the leaking column.
- `engineer_features` sorts its input defensively AND reindexes it to a continuous daily calendar (contracts pinned by `test_engineer_features_sorts_unsorted_input` and `test_lag_features_respect_calendar_gaps`): positional `shift`/`rolling` on a frame with a missing calendar day would silently read across the gap, so the day after a hole gets NaN lags (and drops), never the value from two days prior. Rolling windows tolerate one missing day per window (`min_periods=w-1`). `engineer_features` takes `lat` (for the TOA clearness denominator); the synthetic fixture "lives at" `conftest.SYNTHETIC_LAT` (45°N, matching its June-peaking seasonal cycle).
- Calibration (resolved phase 2): the original hand-picked `LGBM_PARAMS` undercovered (~0.62–0.68 P10–P90 coverage); the tuned params (strong regularization, see `config.py`) measure 0.72–0.78 on the real demo locations + Nairobi and ~0.73 on the synthetic fixture — no conformal post-hoc step was needed. The coverage test's lower bound is 0.65; if coverage regresses below that, suspect a params change before suspecting the test.

## Evaluation conventions

- Metrics: RMSE, MAE, and pinball loss (for the quantile forecasts) on the held-out test set.
- Baselines reported alongside every model result — never show a number without its floor.
- Per-regime RMSE breakdown (via `cluster.py`) is the primary error-analysis lens — prefer it over aggregate metrics alone when writing up results.

## Open design questions (resolve before building the relevant piece)

None open. The stretch-model scope and the phase-4 go/no-go are resolved (see the phase-4 decisions below).

## Resolved design decisions (phase 2)

- **LightGBM hyperparameters: tuned once, reused fixed.** `src/tune.py` (offline-only, like `train.py`) tuned on the demo locations' val splits; the winners live in `config.LGBM_PARAMS` and are reused unchanged for every location at inference. Do NOT add per-location tuning to `pipeline.py` — the "fits in seconds" latency claim depends on this.
- **Regime clustering: refit per location.** Decided empirically (demo locations + off-preset Nairobi): fit-once on pooled demo data produced degenerate regimes (<20 test days) at 3 of 4 probe locations (Phoenix Transitional 10d, Bergen Clear Sky 16d, Nairobi Overcast 14d) and transferred nonsensically to the equatorial climate (326 of Nairobi's 612 test days labeled "Winter Low" with the *highest* mean irradiance — the pooled clusters lean on day-of-year, which inverts meaning at the equator). Refit-per-location was non-degenerate everywhere (min 50 test days/regime); ~10 years of history per location makes refits stable. The "Winter Low" label is gated on the location having a real seasonal cycle (relative monthly irradiance amplitude ≥ `config.SEASONAL_AMPLITUDE_THRESHOLD`; measured 0.91–2.05 at Bristol/Phoenix/Bergen vs 0.37 at Nairobi and 0.21 at the Kuala Lumpur preset) — equatorial locations get the neutral "Low Irradiance" label instead. Residual quirk: names are still location-relative strata (e.g. Phoenix's clearest cluster is named "Transitional"), acceptable for the analysis-layer role.
- **SHAP in the dashboard: per-request, on the main callback path.** Measured TreeExplainer + SHAP on a full ~614-row test set with the tuned params: **0.2s** (the 8-leaf trees are tiny) — far under the 1.5s threshold, so the dashboard computes a top-8 mean-|SHAP| panel on every request. `explain.py` is imported by the dashboard layer only, never by `pipeline.py`.

## Resolved design decisions (phase 4)

- **Stretch model: single-layer LSTM + attention.** Presets: fit offline by `train.py` via `run_location(..., with_seq=True)`; the dashboard reads the saved `artifacts/seq_forecasts_<preset>.parquet` and reindexes to the current test window (caches refresh every ~10 days, so the window can shift and only overlapping days are drawn and scored). Custom locations: the dashboard calls `run_location(..., with_seq=True)` itself when torch is installed (measured ~5s extra on top of ~4.6s LightGBM, inside the existing lock and memo; later revisits are instant), without torch they show LightGBM only. Either way a dashed second line plus LSTM tiles. Hyperparameters are fixed in `config.SEQ_PARAMS`, never tuned per location.
- **torch and LightGBM share a process only with `torch.set_num_threads(1)`.** Each bundles its own OpenMP runtime: a multithreaded torch op after a LightGBM fit deadlocks on macOS (0% CPU, measured), and importing torch before lightgbm segfaults. `model_seq.py` pins one thread (the net is tiny) and `pipeline.py` imports it lazily after lightgbm. Do not raise the thread count or import torch first.
- **Windows spanning a calendar gap get NaN, not a misaligned guess.** LSTM metrics are scored only on days where it has a full 30-day window, against LightGBM on the same days (`pipeline.seq_comparison_metrics`).
- **Not built:** TFT-lite gated variable selection, kept as a possible ablation on top of the LSTM.

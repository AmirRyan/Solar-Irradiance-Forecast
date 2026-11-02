# Solar Irradiance Forecasting

Probabilistic solar irradiance forecasting for **any location on Earth**, using
[NASA POWER](https://power.larc.nasa.gov/) daily data. Enter a lat/lon (map
click, preset, or manual entry) and get a next-day P10/P50/P90 forecast fit
on that location's own history — with baselines, uncertainty bands, weather-
regime error analysis, and SHAP explanations.

![Dashboard: Bristol forecast with P10–P90 band and regime-colored actuals](docs/img/forecast-bristol.png)

## How it works

- **Per-location fetch-and-fit:** on request, the app fetches (and caches)
  NASA POWER daily history since 2015 (~11 years) for that lat/lon and fits
  baselines + LightGBM on the spot. The fit takes seconds; the API fetch is
  the latency bottleneck, so repeat requests are instant. Caches self-refresh:
  one older than 10 days (measured against its last observed day) is refetched
  in full on next access, and if that refetch fails the stale cache is served
  with a warning — a live demo degrades to slightly-old data, never an error.
- **Four pre-cached presets** — Bristol (temperate/mixed), Phoenix
  (sunny/arid), Bergen (cloudy/wet), Kuala Lumpur (equatorial/humid) — so a
  live demo never waits on the NASA POWER API.
- **Models:** persistence and Ridge baselines (the floor), LightGBM quantile
  regression (P10/P50/P90, the headline model), and an optional attention
  LSTM / TFT-lite comparison model (stretch).
- **Analysis:** KMeans weather-regime clustering (Clear Sky / Overcast /
  Transitional / Winter Low — locations without a real seasonal cycle, like
  Kuala Lumpur, get the neutral "Low Irradiance" instead of "Winter Low")
  drives a per-regime RMSE breakdown, plus SHAP feature importance and
  per-prediction explanations.
- **One code path:** `src/pipeline.py` is the single fetch → features → fit →
  predict function; both the offline training script and the live dashboard
  call it, so the two can never drift.

## Quickstart

```bash
pip install -r requirements.txt   # macOS: also `brew install libomp` for LightGBM

# Optional but recommended before launching (and before any live demo):
# warm the caches + fit models for the four presets, print metrics
python -m src.train

# Launch the web app at http://127.0.0.1:8050
python dashboard/app.py

# Run the test suite (synthetic data only, no network)
python -m pytest tests/
```

![Location input: preset dropdown, manual lat/lon, and clickable map](docs/img/location-input.png)

The dashboard labels its uncertainty band honestly: it is the model's
P10–P90 quantile forecast, whose **measured** coverage on held-out data is
0.73–0.77 (nominal 0.80) — shown in the caption and per-location coverage
tile rather than marketed as an "80% confidence interval".

## Results

Next-day forecasts on each location's held-out test set (kWh/m²/day),
regenerated 2026-07-15. Splits are chronological (70/15/15): each preset
trains on 2015 → early 2023, keeps a ~626-day validation slice between train
and test (used once, by the one-off hyperparameter tuning in `src/tune.py` —
the winning params are reused fixed everywhere), and is evaluated on the most
recent ~626 days:

| Location | Train window | Test window | Test days | Measured P10–P90 coverage |
|---|---|---|---|---|
| Bristol, UK (temperate/mixed) | 2015-01-30 → 2023-01-23 | 2024-10-10 → 2026-07-09 | 626 | 0.730 |
| Phoenix, AZ, USA (sunny/arid) | 2015-01-30 → 2023-02-01 | 2024-10-21 → 2026-07-09 | 627 | 0.756 |
| Bergen, Norway (cloudy/wet) | 2015-01-30 → 2023-01-23 | 2024-10-10 → 2026-07-09 | 626 | 0.765 |
| Kuala Lumpur, MY (equatorial/humid) | 2015-01-30 → 2023-01-27 | 2024-10-15 → 2026-07-08 | 626 | 0.740 |

Coverage is the measured fraction of test-set actuals falling inside the
P10–P90 band (nominal 0.80). Test windows end ~6 days before the fetch date
because NASA POWER's near-real-time feed lags ~5–7 days.

**Bristol, UK**

| Model | RMSE | MAE | Pinball (P10/P50/P90) |
|---|---|---|---|
| Persistence | 1.199 | 0.834 | – |
| Ridge | **0.978** | 0.735 | – |
| LightGBM (P50) | 0.985 | **0.725** | 0.164 / 0.362 / 0.132 |

**Phoenix, AZ, USA**

| Model | RMSE | MAE | Pinball (P10/P50/P90) |
|---|---|---|---|
| Persistence | 0.989 | 0.587 | – |
| Ridge | **0.809** | 0.534 | – |
| LightGBM (P50) | 0.836 | **0.486** | 0.180 / 0.243 / 0.072 |

**Bergen, Norway**

| Model | RMSE | MAE | Pinball (P10/P50/P90) |
|---|---|---|---|
| Persistence | 1.182 | 0.736 | – |
| Ridge | **0.955** | 0.643 | – |
| LightGBM (P50) | 0.958 | **0.613** | 0.135 / 0.307 / 0.126 |

**Kuala Lumpur, MY**

| Model | RMSE | MAE | Pinball (P10/P50/P90) |
|---|---|---|---|
| Persistence | 1.100 | 0.825 | – |
| Ridge | **0.846** | 0.638 | – |
| LightGBM (P50) | 0.854 | **0.636** | 0.179 / 0.318 / 0.119 |

The same fixed hyperparameters hold up off-preset — e.g. Nairobi, Kenya
(equatorial, not pre-cached, re-measured 2026-07-15): LightGBM RMSE 0.776 vs
Ridge 0.805 vs persistence 0.926, with measured band coverage 0.765 over 626
test days. LightGBM's P50 optimizes absolute error, which is why it wins MAE
at every location while Ridge stays marginally ahead on RMSE.

### Per-regime RMSE breakdown

The primary error-analysis lens (per CLAUDE.md's evaluation conventions):
LightGBM P50 error split by weather regime, where regimes are KMeans clusters
refit per location (see the error-analysis notes below for naming caveats).

| Location | Regime | Days | RMSE | MAE |
|---|---|---|---|---|
| Bristol | Overcast | 207 | 0.513 | 0.376 |
| Bristol | Winter Low | 163 | 0.760 | 0.564 |
| Bristol | Transitional | 103 | 1.240 | 1.012 |
| Bristol | Clear Sky | 153 | 1.402 | 1.175 |
| Phoenix | Winter Low | 220 | 0.661 | 0.402 |
| Phoenix | Clear Sky | 154 | 0.797 | 0.469 |
| Phoenix | Transitional | 188 | 0.979 | 0.493 |
| Phoenix | Overcast | 65 | 0.993 | 0.791 |
| Bergen | Overcast | 249 | 0.307 | 0.183 |
| Bergen | Winter Low | 115 | 0.773 | 0.505 |
| Bergen | Transitional | 152 | 1.238 | 0.996 |
| Bergen | Clear Sky | 110 | 1.506 | 1.172 |
| Kuala Lumpur | Low Irradiance | 127 | 0.730 | 0.572 |
| Kuala Lumpur | Transitional | 152 | 0.863 | 0.621 |
| Kuala Lumpur | Clear Sky | 219 | 0.863 | 0.626 |
| Kuala Lumpur | Overcast | 128 | 0.937 | 0.736 |

These tables and the SHAP feature-importance artifacts are regenerated into
`artifacts/` by `python -m src.train`. The dashboard also computes a SHAP
panel per request:

![SHAP panel: top-8 features by mean |SHAP| for the P50 model](docs/img/shap-phoenix.png)

## Error analysis

**Where the model earns its keep.** LightGBM's P50 beats both baselines on
MAE at all four presets (and off-preset Nairobi), while Ridge stays
marginally ahead on RMSE everywhere (by 0.003–0.027). That is expected, not
a bug: the P50 is trained on pinball loss, which optimizes absolute error,
so it wins MAE and cedes a sliver of RMSE to the least-squares baseline. The
headline value is not the point forecast anyway — it is the calibrated
P10–P90 band, which neither baseline provides at all.

**Where it struggles, by regime.** Absolute error tracks irradiance level
and volatility, not "cloudiness is hard". At Bristol and Bergen the easiest
stratum is Overcast (RMSE 0.51 / 0.31 — dim days leave little room to miss)
and the hardest is the high-irradiance Clear Sky stratum (1.40 / 1.51), with
Transitional close behind (~1.24): a bright day arriving after mixed weather
is exactly where lag-driven features whiff, and where the uncertainty band
matters most. Phoenix inverts the temperate pattern — its stable winter is
its easiest regime (0.66) and its *rare* Overcast days (65 of 627) are its
hardest (0.99): the regimes a location seldom sees are the regimes its model
is worst at. Kuala Lumpur has the narrowest spread (0.73–0.94); with no
seasonal cycle, day-to-day equatorial convection dominates and regime
membership carries less information.

**The honest-calibration story.** The original hand-picked LightGBM params
undercovered badly: the "P10–P90" band contained only ~0.62–0.68 of held-out
actuals. A one-off tuning pass (`src/tune.py`, objective = pinball loss,
coverage tracked as a diagnostic) landed on strongly regularized params that
measure 0.73–0.77 across the presets — close enough to the nominal 0.80 that
no conformal post-hoc step was added. That was a decision made by
measurement, not by default, and it is why the dashboard and this README
cite measured coverage instead of claiming "80% confidence".

**Known limitations, stated plainly:**

- **Regime names are location-relative strata, not absolute weather
  classes.** The naming heuristic ranks clusters within one location, so the
  same label means different weather in different climates — at Phoenix the
  cluster named "Transitional" actually has the *highest* mean irradiance
  (7.6 kWh/m²/day vs 6.4 for its "Clear Sky"). Read the labels as ordered
  strata for error analysis, not as meteorology.
- **"Today" is really today−~6.** NASA POWER's near-real-time feed lags
  ~5–7 days, so the most recent forecastable day trails the calendar by
  about a week, and test windows end ~6 days before their fetch date.
- **Per-location fetch-and-fit trades away cross-location learning.** Every
  model sees only one point's ~11-year history: first requests to a new
  location pay the fetch latency, and nothing transfers from data-rich to
  data-poor regions. This was chosen deliberately over a global model (see
  Future work) because the per-location fit runs in seconds and keeps the
  scope honest.

## Future work

- **Global multi-location model** : one model conditioned on lat/lon/climate
  with instant inference anywhere. Explicitly deferred: it needs a cleaned
  multi-location training corpus and a model that conditions well on
  geography, a meaningfully bigger and riskier undertaking than per-location
  fits that already run in seconds.
- **Free-text place-name input** : considered and deliberately not built, to
  avoid a geocoding dependency; the map click / preset / manual lat/lon
  inputs cover the demo. If wanted later it is a small addition (a free
  geocoding API), not a rearchitecture.
- **Sequence model (attention LSTM / TFT-lite)** : the phase-4 stretch
  comparison of classical vs. deep learning; its go/no-go is still open. It
  would not need to beat LightGBM to be a useful result, and would likely be
  demo-locations-only, since training a deep model per request is not
  feasible at interactive latency.

## Repo layout

See `CLAUDE.md` for the full architecture guide. In short: `config.py` is the
single source of truth for features/paths/demo locations; `src/` holds the
data layer, models, clustering, SHAP, and the shared pipeline; `dashboard/`
is the Plotly Dash app (pure helpers in `dashboard/helpers.py`); `tests/` is
a network-free pytest suite running entirely on synthetic data;
`notebooks/eda.ipynb` is exploratory only.

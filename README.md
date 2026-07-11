# Solar Irradiance Forecasting

Probabilistic solar irradiance forecasting for **any location on Earth**, using
[NASA POWER](https://power.larc.nasa.gov/) daily data. Enter a lat/lon (map
click, preset, or manual entry) and get a next-day P10/P50/P90 forecast fit
on that location's own history — with baselines, uncertainty bands, weather-
regime error analysis, and SHAP explanations.

## How it works

- **Per-location fetch-and-fit:** on request, the app fetches (and caches)
  ~10 years of NASA POWER history for that lat/lon and fits baselines +
  LightGBM on the spot. The fit takes seconds; the API fetch is the latency
  bottleneck, so repeat requests are instant.
- **Models:** persistence and Ridge baselines (the floor), LightGBM quantile
  regression (P10/P50/P90, the headline model), and an optional attention
  LSTM / TFT-lite comparison model (stretch).
- **Analysis:** KMeans weather-regime clustering (Clear Sky / Overcast /
  Transitional / Winter Low) drives a per-regime RMSE breakdown, plus SHAP
  feature importance and per-prediction explanations.
- **One code path:** `src/pipeline.py` is the single fetch → features → fit →
  predict function; both the offline training script and the live dashboard
  call it, so the two can never drift.

## Quickstart

```bash
pip install -r requirements.txt

# Warm the cache + fit models for the demo locations, print metrics
python -m src.train

# Launch the web app
python dashboard/app.py
```

## Results

Next-day forecasts on each location's held-out test set (~614–621 days,
kWh/m²/day). LightGBM hyperparameters were tuned once on the demo locations'
validation splits (`src/tune.py`) and reused fixed everywhere — baselines
alongside every number.

**Bristol, UK** (temperate/mixed)

| Model | RMSE | MAE | Pinball (P10/P50/P90) |
|---|---|---|---|
| Persistence | 1.139 | 0.798 | – |
| Ridge | 0.922 | 0.698 | – |
| LightGBM (P50) | 0.928 | **0.689** | 0.153 / 0.345 / 0.129 |

**Phoenix, AZ, USA** (sunny/arid)

| Model | RMSE | MAE | Pinball (P10/P50/P90) |
|---|---|---|---|
| Persistence | 0.864 | 0.512 | – |
| Ridge | **0.740** | 0.507 | – |
| LightGBM (P50) | 0.772 | **0.450** | 0.161 / 0.225 / 0.068 |

**Bergen, Norway** (cloudy/wet)

| Model | RMSE | MAE | Pinball (P10/P50/P90) |
|---|---|---|---|
| Persistence | 1.018 | 0.642 | – |
| Ridge | 0.842 | 0.574 | – |
| LightGBM (P50) | **0.835** | **0.535** | 0.117 / 0.268 / 0.116 |

The quantile model's P10–P90 band covers 72–78% of test-set actuals across
the demo locations (ideal: 80%), and the same fixed hyperparameters hold up
off-preset — e.g. Nairobi (equatorial): LightGBM RMSE 0.783 vs Ridge 0.796
vs persistence 0.930, with 77.5% band coverage. LightGBM's P50 optimizes
absolute error, which is why it consistently wins MAE while Ridge stays
competitive on RMSE. Per-regime RMSE breakdowns (the primary error-analysis
lens) and SHAP feature-importance artifacts are written to `artifacts/` by
`python -m src.train`.

## Repo layout

See `CLAUDE.md` for the full architecture guide. In short: `config.py` is the
single source of truth for features/paths/demo locations; `src/` holds the
data layer, models, clustering, SHAP, and the shared pipeline; `dashboard/`
is the Plotly Dash app; `notebooks/eda.ipynb` is exploratory only.

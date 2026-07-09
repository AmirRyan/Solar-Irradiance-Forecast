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

_TODO: fill in after week 2 — RMSE/MAE/pinball table with baselines alongside
the LightGBM numbers, and the per-regime breakdown._

| Model | RMSE | MAE | Pinball (P10/P50/P90) |
|---|---|---|---|
| Persistence | – | – | – |
| Ridge | – | – | – |
| LightGBM (P50) | – | – | – |

## Repo layout

See `CLAUDE.md` for the full architecture guide. In short: `config.py` is the
single source of truth for features/paths/demo locations; `src/` holds the
data layer, models, clustering, SHAP, and the shared pipeline; `dashboard/`
is the Plotly Dash app; `notebooks/eda.ipynb` is exploratory only.

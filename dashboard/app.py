"""Plotly Dash web app: location in, probabilistic forecast out.

Run: python dashboard/app.py

Location input three ways: map click, preset dropdown (demo locations),
manual lat/lon. All three funnel into the SAME callback, which resolves
whichever input FIRED (stale values never shadow the latest action) and
calls src.pipeline.run_location — the dashboard contains no modeling logic
of its own (per CLAUDE.md: one pipeline, two callers).

First request for a new location shows a loading state (the NASA POWER
fetch dominates latency, not the fit). Demo locations are pre-cached by
`python -m src.train` and respond instantly; revisited locations hit an
in-process memo (dashboard-layer only). SHAP importance is computed
per-request in the callback — measured ~0.2s with the tuned params.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Input, Output, State, ctx, dcc, html

import config
from dashboard.helpers import (
    ResultMemo, align_seq_forecasts, preset_name_for, resolve_location,
)
from src.explain import global_importance, shap_values
from src.pipeline import run_location, seq_comparison_metrics, seq_forecast_path

app = dash.Dash(__name__, title="Solar Irradiance Forecast")

# Regime identity colors (Winter Low / Low Irradiance are mutually exclusive
# alternatives of the same slot, so they share one).
REGIME_COLORS = {
    "Clear Sky": "#eda100",
    "Overcast": "#8a8f98",
    "Transitional": "#1baf7a",
    "Winter Low": "#4a3aa7",
    config.REGIME_LOW_SEASONALITY_NAME: "#4a3aa7",
}
P50_COLOR = "#2a78d6"
SEQ_COLOR = "#d6402a"
BAND_FILL = "rgba(42,120,214,0.18)"
INK, MUTED = "#0b0b0b", "#52514e"

_memo = ResultMemo(maxsize=20)
# Only probed, never imported here: torch loads lazily inside run_location.
HAS_TORCH = importlib.util.find_spec("torch") is not None
_pipeline_lock = threading.Lock()  # serialize concurrent fetch/fits


# ---------------------------------------------------------------------------
# Map: an invisible 2° grid of click targets — clickData only fires on trace
# points, so an empty Scattergeo would make map clicks dead.
# ---------------------------------------------------------------------------
_GRID_LATS, _GRID_LONS = np.meshgrid(np.arange(-88, 89, 2.0), np.arange(-180, 179, 2.0))


def make_map_figure(selected: tuple[float, float] | None = None) -> go.Figure:
    fig = go.Figure(
        go.Scattergeo(
            lat=_GRID_LATS.ravel(),
            lon=_GRID_LONS.ravel(),
            mode="markers",
            marker={"size": 10, "color": "rgba(0,0,0,0)"},
            hovertemplate="%{lat:.0f}°, %{lon:.0f}°<extra>click to forecast</extra>",
            showlegend=False,
        )
    )
    if selected is not None:
        fig.add_trace(
            go.Scattergeo(
                lat=[selected[0]], lon=[selected[1]], mode="markers",
                marker={"size": 12, "color": P50_COLOR, "symbol": "circle",
                        "line": {"width": 2, "color": "white"}},
                hovertemplate="%{lat:.2f}°, %{lon:.2f}°<extra>selected</extra>",
                showlegend=False,
            )
        )
    fig.update_layout(
        geo={"projection_type": "natural earth", "showcountries": True},
        height=300,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        uirevision="keep-zoom",  # don't reset the user's pan/zoom on update
    )
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "fontFamily": "sans-serif",
           "color": INK},
    children=[
        html.H2("Probabilistic Solar Irradiance Forecast"),
        html.P("Pick a location: click the map, choose a preset, or enter lat/lon."),

        html.Div(
            style={"display": "flex", "gap": "12px", "alignItems": "center"},
            children=[
                dcc.Dropdown(
                    id="preset",
                    options=[{"label": k, "value": k} for k in config.DEMO_LOCATIONS],
                    placeholder="Preset location…",
                    style={"width": "260px"},
                ),
                dcc.Input(id="lat", type="number", placeholder="lat", debounce=True),
                dcc.Input(id="lon", type="number", placeholder="lon", debounce=True),
                html.Button("Forecast", id="go", n_clicks=0),
            ],
        ),

        dcc.Graph(id="map", figure=make_map_figure(), config={"scrollZoom": True}),

        # Loading state: first fetch for a new location takes a while
        dcc.Loading(
            type="default",
            children=html.Div(
                [
                    html.Div(id="status", style={"padding": "8px 0", "color": "#b00020",
                                                 "fontWeight": "bold"}),
                    html.Div(id="metrics"),
                    dcc.Graph(id="forecast-plot"),
                    html.Div(
                        "Shaded band = model P10–P90 quantile forecasts. Measured "
                        "band coverage on held-out data is ~72–78% (nominal 80%).",
                        style={"fontSize": "12px", "color": MUTED,
                               "padding": "0 0 12px 4px"},
                    ),
                    dcc.Graph(id="residuals-plot"),
                    dcc.Graph(id="shap-plot"),
                ]
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _load_seq(loc, result):
    """LSTM forecasts for this request, aligned to the current test window.
    Custom locations carry theirs on the result (fit live); presets read the
    ones train.py saved. None when torch is absent or train.py has not run."""
    if result.seq_forecasts is not None:
        return result.seq_forecasts
    name = preset_name_for(*loc)
    if name is None or not seq_forecast_path(name).exists():
        return None
    return align_seq_forecasts(pd.read_parquet(seq_forecast_path(name)), result.test.index)


def _forecast_figure(result, seq=None) -> go.Figure:
    """Actual vs P50 with the P10–P90 band; one actuals trace per regime so
    the legend carries regime identity. `seq` adds the LSTM P50 as a dashed
    second line."""
    test, fc = result.test, result.forecasts
    y = test[config.TARGET_COL]
    # Reindex to daily so the line/band BREAK at missing days (recent rows
    # can be dropped when NASA's near-real-time exog values lag) instead of
    # drawing a continuous line across a gap that has no forecasts. The band
    # is drawn per contiguous run — a NaN breaks a line but plotly still
    # bridges `tonexty` fills across gaps otherwise.
    fc = fc.reindex(pd.date_range(fc.index.min(), fc.index.max(), freq="D"))
    fig = go.Figure()

    runs = (fc["p50"].isna() != fc["p50"].shift().isna()).cumsum()[fc["p50"].notna()]
    for i, (_, seg) in enumerate(fc.loc[runs.index].groupby(runs)):
        fig.add_trace(go.Scatter(x=seg.index, y=seg["p90"], mode="lines",
                                 line={"width": 0}, showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=seg.index, y=seg["p10"], mode="lines",
                                 fill="tonexty", fillcolor=BAND_FILL,
                                 line={"width": 0}, name="P10–P90",
                                 showlegend=(i == 0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["p50"], name="P50 forecast",
                             mode="lines", line={"color": P50_COLOR, "width": 2}))
    if seq is not None:
        fig.add_trace(go.Scatter(x=seq.index, y=seq["p50"], name="LSTM P50",
                                 mode="lines",
                                 line={"color": SEQ_COLOR, "width": 2, "dash": "dash"}))
    for regime in result.regimes.unique():
        mask = result.regimes == regime
        fig.add_trace(go.Scatter(
            x=y.index[mask], y=y[mask], name=regime, mode="markers",
            marker={"size": 5, "color": REGIME_COLORS.get(regime, "#333")},
        ))
    fig.update_layout(
        title="Forecast vs actual (held-out test set, colored by weather regime)",
        yaxis_title="Irradiance (kWh/m²/day)",
        height=420,
        plot_bgcolor="#fcfcfb",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0},
    )
    return fig


def _residuals_figure(result) -> go.Figure:
    resid = result.test[config.TARGET_COL] - result.forecasts["p50"]
    fig = go.Figure()
    for regime in result.regimes.unique():
        mask = result.regimes == regime
        fig.add_trace(go.Scatter(
            x=resid.index[mask], y=resid[mask], name=regime, mode="markers",
            marker={"size": 4, "color": REGIME_COLORS.get(regime, "#333")},
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED)
    fig.update_layout(title="P50 residuals by regime",
                      yaxis_title="Actual − P50", height=300,
                      plot_bgcolor="#fcfcfb",
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.0})
    return fig


def _shap_figure(result) -> go.Figure:
    """Top-8 global importance, computed per-request (~0.2s measured with the
    tuned params — see the resolved open question in CLAUDE.md)."""
    importance = global_importance(shap_values(result.gbm, result.test)).head(8)
    ranked = importance.sort_values()  # horizontal bars, biggest on top
    fig = go.Figure(go.Bar(
        x=ranked.values, y=ranked.index, orientation="h",
        marker={"color": P50_COLOR},
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="What drives this location's forecast (top 8, mean |SHAP|, P50 model)",
        xaxis_title="mean |SHAP| (kWh/m²/day)",
        height=320,
        plot_bgcolor="#fcfcfb",
        margin={"l": 220},
    )
    return fig


def _metrics_bar(result, seq=None) -> html.Div:
    m = result.metrics
    y = result.test[config.TARGET_COL]
    coverage = float(((y >= result.forecasts["p10"]) & (y <= result.forecasts["p90"])).mean())

    def tile(label, value):
        return html.Div([html.B(label), html.Div(value)])

    tiles = [
        tile("LightGBM P50 RMSE", f"{m['gbm_p50_rmse']:.3f}"),
        tile("Ridge RMSE", f"{m['ridge_rmse']:.3f}"),
        tile("Persistence RMSE", f"{m['persistence_rmse']:.3f}"),
        tile("LightGBM MAE", f"{m['gbm_p50_mae']:.3f}"),
        tile("Pinball P10/P50/P90",
             f"{m['gbm_pinball_p10']:.3f} / {m['gbm_pinball_p50']:.3f} / "
             f"{m['gbm_pinball_p90']:.3f}"),
        tile("P10–P90 coverage", f"{coverage:.0%}"),
        tile("Test days", str(m["test_days"])),
    ]
    note = None
    if seq is not None:
        c = seq_comparison_metrics(y, result.forecasts, seq)
        tiles[4:4] = [
            tile("LSTM P50 RMSE", f"{c['seq_rmse']:.3f}"),
            tile("LSTM MAE", f"{c['seq_mae']:.3f}"),
        ]
        note = html.Div(
            f"LSTM (dashed) is a comparison model, "
            f"{'fit just now for this location' if result.seq_forecasts is not None else 'trained offline by train.py'}; its "
            f"forecasts cover {c['seq_common_days']} of the days shown. On those same "
            f"days LightGBM RMSE is {c['gbm_common_rmse']:.3f}.",
            style={"fontSize": "12px", "color": MUTED},
        )
    # Baselines shown alongside the headline number, always.
    return html.Div([
        html.Div(tiles, style={"display": "flex", "gap": "24px",
                               "padding": "12px 0", "flexWrap": "wrap"}),
        note,
    ])


# ---------------------------------------------------------------------------
# The one callback — everything funnels into pipeline.run_location
# ---------------------------------------------------------------------------
@app.callback(
    Output("status", "children"),
    Output("metrics", "children"),
    Output("forecast-plot", "figure"),
    Output("residuals-plot", "figure"),
    Output("shap-plot", "figure"),
    Output("map", "figure"),
    Input("go", "n_clicks"),
    Input("preset", "value"),
    Input("map", "clickData"),
    State("lat", "value"),
    State("lon", "value"),
    running=[(Output("go", "disabled"), True, False),
             (Output("preset", "disabled"), True, False)],
    prevent_initial_call=True,
)
def update(_, preset, map_click, lat, lon):
    no5 = (dash.no_update,) * 5

    loc = resolve_location(ctx.triggered_id, preset, map_click, lat, lon)
    if loc is None:
        return "Choose a location to get a forecast.", *no5
    if isinstance(loc, str):  # validation error, already human-readable
        return loc, *no5

    try:
        # Memo first (revisits are instant); the lock serializes concurrent
        # fetch/fits from rapid repeated inputs.
        result = _memo.get(*loc)
        if result is None:
            with _pipeline_lock:
                result = _memo.get(*loc)  # a queued duplicate may have filled it
                if result is None:
                    # Presets read the LSTM train.py saved; custom locations fit
                    # it live (~5s) when torch is installed.
                    live_seq = HAS_TORCH and preset_name_for(*loc) is None
                    result = run_location(*loc, with_seq=live_seq)
                    _memo.put(*loc, result)
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        return (f"NASA POWER rejected this request (HTTP {code}) — "
                "try different coordinates.", *no5)
    except requests.RequestException:
        return ("Could not reach NASA POWER (network problem or timeout). "
                "Try again in a moment.", *no5)
    except ValueError as e:
        return str(e), *no5
    except Exception as e:  # last resort: never show a traceback in a demo
        return f"Unexpected error ({type(e).__name__}): {e}", *no5

    seq = _load_seq(loc, result)
    return (
        "",
        _metrics_bar(result, seq),
        _forecast_figure(result, seq),
        _residuals_figure(result),
        _shap_figure(result),
        make_map_figure(selected=loc),
    )


if __name__ == "__main__":
    app.run(debug=False)

"""Plotly Dash web app: location in, probabilistic forecast out.

Run: python dashboard/app.py

Location input three ways: map click, preset dropdown (demo locations),
manual lat/lon. All of them funnel into the SAME callback, which calls
src.pipeline.run_location — the dashboard contains no modeling logic of
its own (per CLAUDE.md: one pipeline, two callers).

First request for a new location shows a loading state (the NASA POWER
fetch dominates latency, not the fit). Demo locations are pre-cached by
`python -m src.train` and respond instantly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

import config
from src.pipeline import run_location

app = dash.Dash(__name__, title="Solar Irradiance Forecast")

REGIME_COLORS = {
    "Clear Sky": "#f5b942",
    "Overcast": "#8a8f98",
    "Transitional": "#5aa9e6",
    "Winter Low": "#4a5b8c",
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "fontFamily": "sans-serif"},
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

        # Clickable world map for lat/lon selection
        dcc.Graph(
            id="map",
            figure=go.Figure(
                go.Scattergeo()  # empty; clicks give lat/lon via clickData
            ).update_layout(
                geo={"projection_type": "natural earth"},
                height=300,
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
            ),
            config={"scrollZoom": True},
        ),

        # Loading state: first fetch for a new location takes a while
        dcc.Loading(
            type="default",
            children=html.Div(
                [
                    html.Div(id="metrics"),
                    dcc.Graph(id="forecast-plot"),
                    dcc.Graph(id="residuals-plot"),
                ]
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_location(preset, map_click, lat, lon):
    """Precedence: explicit lat/lon > map click > preset."""
    if lat is not None and lon is not None:
        return float(lat), float(lon)
    if map_click and map_click.get("points"):
        p = map_click["points"][0]
        if "lat" in p and "lon" in p:
            return float(p["lat"]), float(p["lon"])
    if preset:
        loc = config.DEMO_LOCATIONS[preset]
        return loc["lat"], loc["lon"]
    return None


def _forecast_figure(result) -> go.Figure:
    """Actual vs P50 with a P10–P90 band; markers colored by regime."""
    test, fc = result.test, result.forecasts
    y = test[config.TARGET_COL]
    fig = go.Figure()

    # Uncertainty band
    fig.add_trace(go.Scatter(x=fc.index, y=fc["p90"], line={"width": 0},
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["p10"], fill="tonexty",
                             fillcolor="rgba(90,169,230,0.25)", line={"width": 0},
                             name="P10–P90"))
    # Median forecast + actuals
    fig.add_trace(go.Scatter(x=fc.index, y=fc["p50"], name="P50 forecast",
                             line={"color": "#2b6cb0"}))
    fig.add_trace(go.Scatter(x=y.index, y=y, name="Actual", mode="markers",
                             marker={"size": 5,
                                     "color": [REGIME_COLORS.get(r, "#333")
                                               for r in result.regimes]}))
    fig.update_layout(
        title="Forecast vs actual (held-out test set, colored by weather regime)",
        yaxis_title="Irradiance (kWh/m²/day)",
        height=420,
    )
    return fig


def _residuals_figure(result) -> go.Figure:
    resid = result.test[config.TARGET_COL] - result.forecasts["p50"]
    fig = go.Figure(go.Scatter(x=resid.index, y=resid, mode="markers",
                               marker={"size": 4,
                                       "color": [REGIME_COLORS.get(r, "#333")
                                                 for r in result.regimes]}))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(title="P50 residuals by regime",
                      yaxis_title="Actual − P50", height=300)
    return fig


# ---------------------------------------------------------------------------
# The one callback — everything funnels into pipeline.run_location
# ---------------------------------------------------------------------------
@app.callback(
    Output("metrics", "children"),
    Output("forecast-plot", "figure"),
    Output("residuals-plot", "figure"),
    Input("go", "n_clicks"),
    Input("preset", "value"),
    Input("map", "clickData"),
    State("lat", "value"),
    State("lon", "value"),
    prevent_initial_call=True,
)
def update(_, preset, map_click, lat, lon):
    loc = _resolve_location(preset, map_click, lat, lon)
    if loc is None:
        return "Choose a location to get a forecast.", go.Figure(), go.Figure()

    result = run_location(*loc)
    m = result.metrics

    # Baselines shown alongside the headline number, always.
    metrics_bar = html.Div(
        style={"display": "flex", "gap": "24px", "padding": "12px 0"},
        children=[
            html.Div([html.B("LightGBM P50 RMSE"), html.Div(f"{m['gbm_p50_rmse']:.3f}")]),
            html.Div([html.B("Ridge RMSE"), html.Div(f"{m['ridge_rmse']:.3f}")]),
            html.Div([html.B("Persistence RMSE"), html.Div(f"{m['persistence_rmse']:.3f}")]),
            html.Div([html.B("Pinball P10/P50/P90"),
                      html.Div(f"{m['gbm_pinball_p10']:.3f} / "
                               f"{m['gbm_pinball_p50']:.3f} / "
                               f"{m['gbm_pinball_p90']:.3f}")]),
            html.Div([html.B("Test days"), html.Div(str(m["test_days"]))]),
        ],
    )
    return metrics_bar, _forecast_figure(result), _residuals_figure(result)


if __name__ == "__main__":
    app.run(debug=True)

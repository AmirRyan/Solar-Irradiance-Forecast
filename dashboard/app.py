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
    ResultMemo, align_seq_forecasts, plain_feature_name, preset_name_for,
    resolve_location, summary_text,
)
from src.explain import global_importance, shap_values
from src.pipeline import run_location, seq_comparison_metrics, seq_forecast_path

app = dash.Dash(__name__, title="Solar Sunlight Forecast")

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
BAND_FILL = "rgba(42,120,214,0.22)"
GRID = "rgba(128,128,128,0.22)"
# Legends sit under the plot so a wrapped legend never collides with the title.
_LEGEND_BELOW = {"orientation": "h", "yanchor": "top", "y": -0.16, "x": 0,
                 "itemsizing": "constant"}


def _empty(height: int) -> go.Figure:
    """Placeholder shown before the first result: no stray axes."""
    fig = _themed(go.Figure(), height=height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text="Select a location to see results", showarrow=False,
                       font={"size": 14, "color": "rgba(128,128,128,0.9)"})
    return fig


def _themed(fig: go.Figure, **layout) -> go.Figure:
    """Transparent backgrounds so the page theme (assets/style.css) shows
    through in light and dark; text color comes from the stylesheet."""
    title = layout.pop("title", None)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title={"text": title, "font": {"size": 15}, "x": 0.01},
                      margin={"l": 55, "r": 15, "t": 70, "b": 40}, **layout)
    fig.update_layout(hoverlabel={"bgcolor": "#1f2024", "bordercolor": "#1f2024",
                                  "font": {"color": "#f2f2f0", "size": 12}})
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig

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
        geo={"projection_type": "natural earth", "showcountries": True,
             "showland": True, "landcolor": "rgba(128,128,128,0.22)",
             "countrycolor": "rgba(128,128,128,0.55)", "coastlinecolor": "rgba(128,128,128,0.55)",
             "showocean": False, "showlakes": False, "bgcolor": "rgba(0,0,0,0)"},
        height=260,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        uirevision="keep-zoom",  # don't reset the user's pan/zoom on update
    )
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
GLOSSARY = [
    ("Sunlight, kWh/m²/day", "How much solar energy lands on one square metre in a day. "
     "Bigger means sunnier; a dull winter day is around 1, a bright summer day 5 or more."),
    ("Test days", "Recent days kept hidden while the model learned. Scoring on them shows "
     "how it does on days it has never seen."),
    ("Best guess (P50)", "The model's middle forecast: half the time reality should come "
     "in lower, half the time higher."),
    ("Shaded range (P10–P90)", "A low guess (P10) and a high guess (P90). Reality should "
     "land between them about 8 days out of 10."),
    ("Coverage", "The share of test days that really did land inside the shaded range."),
    ("Overall error (RMSE)", "How far off the forecast is, scored so that big misses "
     "count extra."),
    ("Average miss (MAE)", "The plain average of how far off the forecast is, in "
     "kWh/m² per day."),
    ("Range score (pinball loss)", "A single score for how good the low, middle and "
     "high guesses are together. Lower is better."),
    ("Weather types", "Each day is grouped by the kind of weather it had, such as Clear "
     "Sky or Overcast, so you can see where the model is strong or weak."),
    ("LightGBM", "The main model: many small decision trees combined."),
    ("LSTM", "A neural network that reads the last 30 days in order. Included as a comparison."),
    ("Ridge and repeat-yesterday", "Two simple baselines. A model is only worth using if it "
     "beats them."),
    ("SHAP", "A method that measures how much each clue pushes the forecast up or down."),
]

app.layout = html.Div(
    className="page",
    children=[
        html.H1("Solar Sunlight Forecast"),
        html.P("Predicts how much sunlight a place will get each day, using NASA satellite "
               "data. It also shows how sure it is, and how well it did on past days it "
               "had not seen.", className="subtitle"),

        html.H3("1. Choose a location", className="section-title"),
        html.P("Click the map, pick a preset city, or type coordinates, then press Get forecast.",
               className="section-sub"),
        html.Div(
            className="card controls",
            children=[
                dcc.Dropdown(
                    id="preset",
                    options=[{"label": k, "value": k} for k in config.DEMO_LOCATIONS],
                    placeholder="Preset city…",
                    className="preset",
                ),
                dcc.Input(id="lat", type="number", placeholder="Latitude", debounce=True),
                dcc.Input(id="lon", type="number", placeholder="Longitude", debounce=True),
                html.Button("Get forecast", id="go", n_clicks=0),
            ],
        ),

        html.Div(
            className="card map-card",
            children=dcc.Graph(id="map", figure=make_map_figure(),
                               config={"scrollZoom": True}),
        ),

        # Loading state: first fetch for a new location takes a while
        dcc.Loading(
            type="default",
            children=html.Div(
                [
                    html.Div(id="status", className="status"),
                    html.Div(id="metrics"),
                    html.H3("Forecast vs what actually happened", className="section-title"),
                    html.P("Each dot is a real day, coloured by the type of weather it had. "
                           "The solid line is the model's best guess and the shaded band is "
                           "where it expects reality to fall about 8 days out of 10. Use the "
                           "buttons to zoom in on the last month, season or year.",
                           className="section-sub"),
                    html.Div(className="card", children=dcc.Graph(id="forecast-plot", figure=_empty(460),
                                                     config={"displayModeBar": False})),
                    html.Div(
                        className="two-col",
                        children=[
                            html.Div([
                                html.H3("How big were the misses?", className="section-title"),
                                html.P("Each dot is one day's miss. A dot on the zero line was "
                                       "spot on, above it the day was sunnier than forecast, "
                                       "below it cloudier. Tighter clusters mean steadier forecasts.",
                                       className="section-sub"),
                                html.Div(className="card", children=dcc.Graph(id="residuals-plot", figure=_empty(340))),
                            ]),
                            html.Div([
                                html.H3("What does the model look at?", className="section-title"),
                                html.P("The clues that move the forecast most. A longer bar "
                                       "means a bigger influence on the answer.",
                                       className="section-sub"),
                                html.Div(className="card", children=dcc.Graph(id="shap-plot", figure=_empty(340))),
                            ]),
                        ],
                    ),
                ]
            ),
        ),

        html.Details(
            className="card glossary",
            children=[
                html.Summary("What do these terms mean?"),
                html.Dl([x for term, text in GLOSSARY
                         for x in (html.Dt(term), html.Dd(text))]),
            ],
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
                                 line={"width": 0}, name="LightGBM P10–P90",
                                 showlegend=(i == 0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["p50"], name="LightGBM P50",
                             mode="lines", line={"color": P50_COLOR, "width": 1.6},
                             hovertemplate="LightGBM P50 %{y:.2f}<extra></extra>"))
    if seq is not None:
        fig.add_trace(go.Scatter(x=seq.index, y=seq["p50"], name="LSTM P50",
                                 mode="lines",
                                 line={"color": SEQ_COLOR, "width": 1.6, "dash": "dash"},
                                 hovertemplate="LSTM P50 %{y:.2f}<extra></extra>"))
    for regime in result.regimes.unique():
        mask = result.regimes == regime
        fig.add_trace(go.Scatter(
            x=y.index[mask], y=y[mask], name=regime, mode="markers",
            marker={"size": 4, "opacity": 0.85, "color": REGIME_COLORS.get(regime, "#333")},
            hovertemplate=f"Actual %{{y:.2f}} ({regime})<extra></extra>",
        ))
    fig.update_xaxes(
        rangeselector={"buttons": [
            {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
            {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"},
            {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
            {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
            {"step": "all", "label": "All"},
        ], "bgcolor": "rgba(128,128,128,0.15)", "activecolor": "rgba(42,120,214,0.45)",
            "x": 1, "xanchor": "right", "y": 1.12},
        range=[fc.index.min(), fc.index.max()],
        showspikes=True, spikemode="across", spikethickness=1, spikecolor=GRID,
    )
    return _themed(
        fig,
        title="Daily sunlight: forecast vs actual",
        yaxis_title="Sunlight (kWh/m²/day)",
        height=460,
        hovermode="x unified",
        legend=_LEGEND_BELOW,
    ).update_layout(margin={"l": 55, "r": 15, "t": 60, "b": 110})


def _residuals_figure(result) -> go.Figure:
    resid = result.test[config.TARGET_COL] - result.forecasts["p50"]
    fig = go.Figure()
    for regime in result.regimes.unique():
        mask = result.regimes == regime
        fig.add_trace(go.Scatter(
            x=resid.index[mask], y=resid[mask], name=regime, mode="markers",
            marker={"size": 4, "color": REGIME_COLORS.get(regime, "#333")},
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(128,128,128,0.8)")
    _themed(fig, title="Forecast misses by weather type",
                   yaxis_title="Actual − forecast (kWh/m²/day)", height=340,
                   legend=_LEGEND_BELOW)
    fig.update_layout(margin={"l": 55, "r": 15, "t": 60, "b": 110})
    return fig


def _shap_figure(result) -> go.Figure:
    """Top-8 global importance, computed per-request (~0.2s measured with the
    tuned params — see the resolved open question in CLAUDE.md)."""
    importance = global_importance(shap_values(result.gbm, result.test)).head(8)
    ranked = importance.sort_values()  # horizontal bars, biggest on top
    fig = go.Figure(go.Bar(
        x=ranked.values, y=[plain_feature_name(c) for c in ranked.index],
        orientation="h", marker={"color": P50_COLOR},
        customdata=ranked.index,  # raw column name, shown on hover
        hovertemplate="%{y}: %{x:.3f}<br>(%{customdata})<extra></extra>",
    ))
    _themed(fig, title="Biggest influences on the forecast",
            xaxis_title="Influence on the forecast (kWh/m²/day)", height=340)
    fig.update_layout(margin={"l": 235, "r": 15, "t": 70, "b": 40})
    return fig


def _metrics_bar(result, seq=None) -> html.Div:
    m = result.metrics
    y = result.test[config.TARGET_COL]
    coverage = float(((y >= result.forecasts["p10"]) & (y <= result.forecasts["p90"])).mean())
    place = preset_name_for(result.lat, result.lon) or f"{result.lat:.2f}°, {result.lon:.2f}°"

    def tile(label, tech, value, hint, cls=""):
        # `tech` is the technical name, kept visible (small) and in the tooltip.
        return html.Div(
            [html.Div(label, className="label"),
             html.Div(value, className="value"),
             html.Div(hint, className="hint"),
             html.Div(tech, className="tech")],
            className=f"tile {cls}".strip(), title=f"{tech}: {hint}",
        )

    accuracy = [
        tile("Overall error", "LightGBM · RMSE", f"{m['gbm_p50_rmse']:.3f}",
             "Our main model; big misses count extra. Lower is better.", "headline"),
        tile("Average miss", "LightGBM · MAE", f"{m['gbm_p50_mae']:.3f}",
             "Plain average of how far off each day was. Lower is better."),
    ]
    note = None
    if seq is not None:
        c = seq_comparison_metrics(y, result.forecasts, seq)
        accuracy += [
            tile("LSTM overall error", "LSTM · RMSE", f"{c['seq_rmse']:.3f}",
                 "A second, neural-network model.", "seq"),
            tile("LSTM average miss", "LSTM · MAE", f"{c['seq_mae']:.3f}",
                 "Same measure, for the LSTM.", "seq"),
        ]
        how = ("fit just now for this location"
               if result.seq_forecasts is not None else "trained ahead of time")
        note = html.Div(
            f"The dashed line is the LSTM, a comparison model ({how}). It can only "
            f"forecast days that have a full 30-day history before them, so it covers "
            f"{c['seq_common_days']} of the days shown; on those same days the main "
            f"model's overall error is {c['gbm_common_rmse']:.3f}.",
            className="note",
        )
    accuracy += [
        tile("Basic model error", "Ridge · RMSE", f"{m['ridge_rmse']:.3f}",
             "A simple statistical model to beat."),
        tile("Repeat-yesterday error", "Persistence · RMSE", f"{m['persistence_rmse']:.3f}",
             "The simplest guess: tomorrow equals today."),
    ]
    reliability = [
        tile("Days inside the range", "P10–P90 coverage", f"{coverage:.0%}",
             "The shaded range aims for 80%."),
        tile("Range score", "Pinball loss P10 / P50 / P90",
             f"{m['gbm_pinball_p10']:.3f} / {m['gbm_pinball_p50']:.3f} / "
             f"{m['gbm_pinball_p90']:.3f}",
             "Scores the low, middle and high guesses. Lower is better."),
        tile("Days tested", "Test days", str(m["test_days"]),
             "Past days the model never saw while learning."),
    ]
    summary = summary_text(
        place, m["gbm_p50_mae"], m["persistence_mae"], coverage,
        float(y.mean()), m["test_days"],
    )
    return html.Div([
        html.Div([html.H3(f"Results for {place}"),
                  *[html.P(t) for t in summary]], className="card summary"),
        html.H3("How accurate is the forecast?", className="section-title"),
        html.P("Errors are in kWh/m² of sunlight per day. Smaller numbers are better. "
               "The last two tiles are simple guesses the models need to beat.",
               className="section-sub"),
        html.Div(accuracy, className="tiles"),
        note,
        html.H3("Can you trust the shaded range?", className="section-title"),
        html.P("Besides a best guess, the model gives a low and a high guess. "
               "These tiles check whether reality stayed between them.",
               className="section-sub"),
        html.Div(reliability, className="tiles"),
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

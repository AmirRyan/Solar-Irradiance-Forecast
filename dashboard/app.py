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
    ResultMemo, align_seq_forecasts, error_scale, plain_feature_name, preset_name_for,
    resolve_location, summary_text,
)
from src.explain import global_importance, shap_values
from src.pipeline import run_location, seq_comparison_metrics, seq_forecast_path

app = dash.Dash(__name__, title="Solar Irradiance Forecast")

# Regime identity colors (Winter Low / Low Irradiance are mutually exclusive
# alternatives of the same slot, so they share one).
REGIME_COLORS = {
    "Clear Sky": "#e9a93a",      # noon
    "Transitional": "#7fa6c4",   # haze
    "Overcast": "#8f99a3",       # slate
    "Winter Low": "#6a6fb5",     # twilight
    config.REGIME_LOW_SEASONALITY_NAME: "#6a6fb5",
}
NOON = "#e9a93a"                 # the one UI accent (stylesheet: --noon)
P50_COLOR = "#4f86c6"            # LightGBM series
SEQ_COLOR = "#d9604a"            # LSTM series
OTHER_MODEL_COLOR = "#8f99a3"    # baselines
BAND_FILL = "rgba(79,134,198,0.22)"
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
_GRID_LATS, _GRID_LONS = np.meshgrid(np.arange(-84, 85, 3.0), np.arange(-180, 180, 3.0))
_DEFAULT_VIEW = {"lon": 10, "lat": 30}  # Europe/Africa facing, where most presets sit


def make_map_figure(selected: tuple[float, float] | None = None) -> go.Figure:
    """Draggable globe. Clicks land on an invisible 3-degree grid (clickData only
    fires on trace points); ~6.8k points keeps rotation smooth. A new selection
    turns the globe to face it; otherwise the user's rotation is left alone."""
    fig = go.Figure(
        go.Scattergeo(
            lat=_GRID_LATS.ravel(),
            lon=_GRID_LONS.ravel(),
            mode="markers",
            marker={"size": 12, "color": "rgba(0,0,0,0)"},
            hovertemplate="%{lat:.0f}°, %{lon:.0f}°<extra>click to forecast</extra>",
            showlegend=False,
        )
    )
    if selected is not None:
        fig.add_trace(
            go.Scattergeo(
                lat=[selected[0]], lon=[selected[1]], mode="markers",
                marker={"size": 12, "color": NOON, "symbol": "circle",
                        "line": {"width": 2, "color": "white"}},
                hovertemplate="%{lat:.2f}°, %{lon:.2f}°<extra>selected</extra>",
                showlegend=False,
            )
        )
    view = {"lon": selected[1], "lat": selected[0]} if selected else _DEFAULT_VIEW
    edge = "rgba(128,128,128,0.55)"
    fig.update_layout(
        geo={"projection": {"type": "orthographic", "rotation": view},
             "showland": True, "landcolor": "rgba(128,128,128,0.38)",
             "showocean": True, "oceancolor": "rgba(128,128,128,0.10)",
             "showcountries": True, "countrycolor": edge, "coastlinecolor": edge,
             "showlakes": False, "showframe": True, "framecolor": edge,
             "lataxis": {"showgrid": True, "gridcolor": "rgba(128,128,128,0.22)", "dtick": 30},
             "lonaxis": {"showgrid": True, "gridcolor": "rgba(128,128,128,0.22)", "dtick": 30},
             "bgcolor": "rgba(0,0,0,0)"},
        height=300,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
        uirevision=f"globe-{selected}",  # new selection resets rotation; same one keeps it
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
        html.Div(className="header", children=[
            html.Div([
                html.H1("Solar Irradiance Forecast"),
                html.P("Predicts how much sunlight a place will get each day, using NASA "
                       "satellite data. It also shows how sure it is, and how well it did "
                       "on past days it had not seen.", className="subtitle"),
            ]),
            html.Button([html.Span("☾ Dark mode", className="to-dark"),
                         html.Span("☀ Light mode", className="to-light")],
                        id="theme-toggle", className="theme-toggle", n_clicks=0,
                        title="Switch between light and dark"),
        ]),

        html.Div(className="layout", children=[
            # Rail: the question. Stays put while you read the answer.
            html.Div(className="rail", children=[
                html.Div(className="card controls", children=[
                    html.H3("Choose a location", className="section-title"),
                    html.P("Click the map, pick a preset city, or type coordinates.",
                           className="section-sub"),
                    dcc.Dropdown(
                        id="preset",
                        options=[{"label": k, "value": k} for k in config.DEMO_LOCATIONS],
                        placeholder="Preset city…",
                        className="preset",
                    ),
                    html.Div(className="latlon", children=[
                        dcc.Input(id="lat", type="number", placeholder="Latitude", debounce=True),
                        dcc.Input(id="lon", type="number", placeholder="Longitude", debounce=True),
                    ]),
                    html.Button("Get forecast", id="go", n_clicks=0),
                ]),
                html.Div(className="card map-card",
                         children=dcc.Graph(id="map", figure=make_map_figure(),
                                            config={"scrollZoom": True,
                                                    "displayModeBar": False})),
            ]),

            # Main: the answer. Loading state covers first-time fetches.
            html.Div(className="main", children=[
                dcc.Loading(type="default", children=html.Div([
                    html.Div(id="status", className="status"),
                    html.Div(id="metrics"),
                    html.H3("Forecast vs what actually happened", className="section-title"),
                    html.P("Each dot is a real day, coloured by the type of weather it had. "
                           "The solid line is the model's best guess and the shaded band is "
                           "where it expects reality to fall about 8 days out of 10. Use the "
                           "buttons to zoom in on the last month, season or year.",
                           className="section-sub"),
                    html.Div(className="card", children=dcc.Graph(
                        id="forecast-plot", figure=_empty(460),
                        config={"displayModeBar": False})),
                    html.H3("Every test day, by weather type", className="section-title"),
                    html.P("One cell per day, in order. Long runs of one colour show the "
                           "seasons; mixed stretches are the days that are hardest to forecast.",
                           className="section-sub"),
                    html.Div(className="card", children=dcc.Graph(
                        id="ribbon-plot", figure=_empty(130),
                        config={"displayModeBar": False})),
                    html.Div(className="two-col", children=[
                        html.Div([
                            html.H3("How big were the misses?", className="section-title"),
                            html.P("Each dot is one day's miss. A dot on the zero line was "
                                   "spot on, above it the day was sunnier than forecast, "
                                   "below it cloudier. Tighter clusters mean steadier forecasts.",
                                   className="section-sub"),
                            html.Div(className="card", children=dcc.Graph(
                                id="residuals-plot", figure=_empty(340))),
                        ]),
                        html.Div([
                            html.H3("What does the model look at?", className="section-title"),
                            html.P("The clues that move the forecast most. A longer bar "
                                   "means a bigger influence on the answer.",
                                   className="section-sub"),
                            html.Div(className="card", children=dcc.Graph(
                                id="shap-plot", figure=_empty(340))),
                        ]),
                    ]),
                ])),
                html.Details(className="card glossary", children=[
                    html.Summary("What do these terms mean?"),
                    html.Dl([x for term, text in GLOSSARY
                             for x in (html.Dt(term), html.Dd(text))]),
                ]),
            ]),
        ]),
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


def _ribbon_figure(result) -> go.Figure:
    """One row, one cell per test day, coloured by weather type. Gaps stay
    blank. Legend entries are invisible markers so Plotly draws the key."""
    regimes = result.regimes.reindex(
        pd.date_range(result.regimes.index.min(), result.regimes.index.max(), freq="D"))
    names = list(result.regimes.unique())
    code = {n: i for i, n in enumerate(names)}
    n = len(names)
    z = [[code.get(r) if isinstance(r, str) else None for r in regimes]]
    colorscale = [stop for i, nm in enumerate(names)
                  for stop in ([i / n, REGIME_COLORS.get(nm, "#888")],
                               [(i + 1) / n, REGIME_COLORS.get(nm, "#888")])]
    fig = go.Figure(go.Heatmap(
        x=regimes.index, z=z, y=[""], zmin=-0.5, zmax=n - 0.5, colorscale=colorscale,
        showscale=False, xgap=0, hoverongaps=False,
        text=[[r if isinstance(r, str) else "" for r in regimes]],
        hovertemplate="%{x|%b %d, %Y}: %{text}<extra></extra>",
    ))
    for nm in names:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm,
                                 marker={"symbol": "square", "size": 10,
                                         "color": REGIME_COLORS.get(nm, "#888")}))
    _themed(fig, height=130, legend=_LEGEND_BELOW)
    fig.update_layout(margin={"l": 15, "r": 15, "t": 10, "b": 55}, hovermode="closest",
                      title=None)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(visible=False)
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
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(128,128,128,0.8)")
    _themed(fig, title="Forecast misses by weather type",
                   yaxis_title="Actual − forecast", height=340,
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
            xaxis_title="Influence (kWh/m²/day)", height=340)
    fig.update_layout(margin={"l": 235, "r": 15, "t": 70, "b": 40})
    return fig


def _scale(errors: dict[str, float], lead: str) -> html.Div:
    """One axis, one dot per model: the comparison you read at a glance."""
    sc = error_scale(errors)
    colors = {"LightGBM": P50_COLOR, "LSTM": SEQ_COLOR}
    kids = [html.Div(className="scale-axis")]
    for t in sc["ticks"]:
        kids += [html.Div(className="scale-tick", style={"left": f"{t['pct']:.2f}%"}),
                 html.Div(f"{t['value']:.2f}", className="scale-tick-label",
                          style={"left": f"{t['pct']:.2f}%"})]
    for pt in sc["points"]:
        size = 16 if pt["name"] == lead else 11
        kids += [
            html.Div(className="scale-dot", style={
                "left": f"{pt['pct']:.2f}%", "width": size, "height": size,
                "background": colors.get(pt["name"], OTHER_MODEL_COLOR)}),
            html.Div([pt["name"], html.B(f"{pt['value']:.3f}")],
                     className=f"scale-label {'above' if pt['above'] else 'below'}"
                               f"{' lead' if pt['name'] == lead else ''}",
                     style={"left": f"{pt['pct']:.2f}%"}),
        ]
    return html.Div(kids, className="scale")


def _metrics_bar(result, seq=None) -> html.Div:
    m = result.metrics
    y = result.test[config.TARGET_COL]
    coverage = float(((y >= result.forecasts["p10"]) & (y <= result.forecasts["p90"])).mean())
    place = preset_name_for(result.lat, result.lon) or f"{result.lat:.2f}°, {result.lon:.2f}°"

    def tile(label, tech, value, hint):
        # `tech` is the technical name, kept visible (small) and in the tooltip.
        return html.Div(
            [html.Div(label, className="label"),
             html.Div(value, className="value"),
             html.Div(hint, className="hint"),
             html.Div(tech, className="tech")],
            className="tile", title=f"{tech}: {hint}",
        )

    errors = {"LightGBM": m["gbm_p50_rmse"], "Ridge": m["ridge_rmse"],
              "Repeat yesterday": m["persistence_rmse"]}
    detail = []
    note = None
    if seq is not None:
        c = seq_comparison_metrics(y, result.forecasts, seq)
        errors["LSTM"] = c["seq_rmse"]
        detail.append(tile("LSTM average miss", "LSTM · MAE", f"{c['seq_mae']:.3f}",
                           "Same measure as the headline, for the LSTM."))
        how = ("fit just now for this location"
               if result.seq_forecasts is not None else "trained ahead of time")
        note = html.Div(
            f"The dashed line is the LSTM, a comparison model ({how}). It can only "
            f"forecast days that have a full 30-day history before them, so it covers "
            f"{c['seq_common_days']} of the days shown; on those same days the main "
            f"model's overall error is {c['gbm_common_rmse']:.3f}.",
            className="note",
        )
    detail += [
        tile("Days inside the range", "P10–P90 coverage", f"{coverage:.0%}",
             "The shaded range aims for 80%."),
        tile("Range score", "Pinball loss P10 / P50 / P90",
             f"{m['gbm_pinball_p10']:.3f} / {m['gbm_pinball_p50']:.3f} / "
             f"{m['gbm_pinball_p90']:.3f}",
             "Scores the low, middle and high guesses. Lower is better."),
        tile("Days tested", "Test days", str(m["test_days"]),
             "Past days the model never saw while learning."),
    ]
    mae, pmae = m["gbm_p50_mae"], m["persistence_mae"]
    better = (pmae - mae) / pmae if pmae > 0 else 0.0
    chip = (f"{abs(better):.0%} {'better' if better >= 0 else 'worse'} "
            f"than repeating yesterday")
    summary = summary_text(place, mae, pmae, coverage, float(y.mean()), m["test_days"])
    return html.Div([
        html.Div([
            html.Div(f"{place} · average miss", className="hero-place"),
            html.Div([
                html.Span([f"{mae:.2f}", html.Span("kWh/m² per day", className="hero-unit")],
                          className="hero-number", title="LightGBM · MAE"),
                html.Span(chip, className="chip"),
            ], className="hero-row"),
            *[html.P(t, className="hero-copy") for t in summary],
            html.Div("Overall error by model (RMSE), lower is better", className="scale-title"),
            _scale(errors, lead="LightGBM"),
            html.Div("Overall error counts big misses extra. Ridge and repeat-yesterday are "
                     "simple guesses the models need to beat.", className="note"),
        ], className="card hero"),
        html.H3("The numbers behind it", className="section-title"),
        html.P("Besides a best guess, the model gives a low and a high guess. These tiles "
               "check whether reality stayed between them and how many days were tested.",
               className="section-sub"),
        html.Div(detail, className="tiles"),
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
    Output("ribbon-plot", "figure"),
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
    no6 = (dash.no_update,) * 6

    loc = resolve_location(ctx.triggered_id, preset, map_click, lat, lon)
    if loc is None:
        return "Choose a location to get a forecast.", *no6
    if isinstance(loc, str):  # validation error, already human-readable
        return loc, *no6

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
                "try different coordinates.", *no6)
    except requests.RequestException:
        return ("Could not reach NASA POWER (network problem or timeout). "
                "Try again in a moment.", *no6)
    except ValueError as e:
        return str(e), *no6
    except Exception as e:  # last resort: never show a traceback in a demo
        return f"Unexpected error ({type(e).__name__}): {e}", *no6

    seq = _load_seq(loc, result)
    return (
        "",
        _metrics_bar(result, seq),
        _forecast_figure(result, seq),
        _residuals_figure(result),
        _shap_figure(result),
        make_map_figure(selected=loc),
        _ribbon_figure(result),
    )


if __name__ == "__main__":
    app.run(debug=False)

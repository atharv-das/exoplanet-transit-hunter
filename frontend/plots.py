"""Plotly figures for the UI.

Colors come from a validated dark-mode palette (categorical slots 1-3 pass colorblind checks on
the dark surface): blue = data, orange = model / highlight, aqua = third series. Class colors
follow the class, never its rank. Status colors (good / critical) are reserved for vetting
badges. Every chart has one y-axis, hover tooltips, and a legend when it has 2+ series.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

# Dark-mode palette (reference dataviz palette, dark steps).
SURFACE = "#1a1a19"
GRID = "#2c2c2a"
TEXT = "#f0efec"
TEXT_2 = "#c3c2b7"
MUTED = "#8a897f"
BLUE = "#3987e5"
ORANGE = "#d95926"
AQUA = "#199e70"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
DIVERGING_NEG = "#e66767"  # diverging pair: blue (+) <-> red (-); not the status red
CLASS_COLORS = {"PC": BLUE, "EB": ORANGE, "FP": AQUA}
CLASS_LABELS = {"PC": "Planet candidate", "EB": "Eclipsing binary", "FP": "Other false positive"}
# Single-hue sequential blue for saliency magnitude (dark surface: low = dark, high = light).
SALIENCY_SCALE = [[0.0, "#184f95"], [0.35, "#2a78d6"], [0.7, "#6da7ec"], [1.0, "#cde2fb"]]

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def _layout(fig: go.Figure, title: str, x_title: str, y_title: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": TEXT}, "x": 0.0, "xanchor": "left"},
        height=height,
        margin={"l": 60, "r": 20, "t": 40, "b": 95},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE,
        font={"color": TEXT_2, "size": 12},
        hovermode="closest",
        hoverlabel={"bgcolor": "#242422", "bordercolor": GRID, "font": {"color": TEXT}},
        # Legend under the plot so it never collides with the title on half-width charts.
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.28,
            "xanchor": "left",
            "x": 0.0,
            "font": {"color": TEXT_2},
        },
    )
    axis = {
        "gridcolor": GRID,
        "zerolinecolor": GRID,
        "linecolor": GRID,
        "tickfont": {"color": TEXT_2},
    }
    fig.update_xaxes(title={"text": x_title, "font": {"color": TEXT_2}}, **axis)
    fig.update_yaxes(title={"text": y_title, "font": {"color": TEXT_2}}, **axis)
    return fig


def _xy(series: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(series.get("x", []), dtype=float), np.asarray(
        series.get("y", []), dtype=float
    )


def _ppm(y: np.ndarray) -> np.ndarray:
    """Relative flux -> parts-per-million deviation (easier to read than 0.99987)."""
    return (y - 1.0) * 1e6


def _time_label(result: dict[str, Any]) -> str:
    return (
        "Time (BKJD = BJD − 2454833)"
        if result["lightcurve"]["mission"] == "Kepler"
        else "Time (BTJD = BJD − 2457000)"
    )


def raw_figure(result: dict[str, Any]) -> go.Figure:
    """Step 1: normalized flux with the fitted stellar trend on top."""
    t, f = _xy(result["plots"]["raw"])
    tt, tr = _xy(result["plots"]["trend"])
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=_ppm(f),
            mode="markers",
            name="Brightness",
            marker={"size": 3, "color": BLUE, "opacity": 0.6},
            hovertemplate="t = %{x:.3f}<br>%{y:.0f} ppm<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=tt,
            y=_ppm(tr),
            mode="lines",
            name="Stellar trend (removed next)",
            line={"width": 2, "color": ORANGE},
            hovertemplate="trend %{y:.0f} ppm<extra></extra>",
        )
    )
    return _layout(fig, "Raw light curve", _time_label(result), "Brightness change (ppm)")


def detrended_figure(result: dict[str, Any], cand: dict[str, Any] | None) -> go.Figure:
    """Step 2: flattened light curve, with the selected candidate's in-transit points highlighted."""
    t, f = _xy(result["plots"]["detrended"])
    fig = go.Figure()
    if cand is not None:
        ph = (t - cand["t0"] + 0.5 * cand["period"]) % cand["period"] - 0.5 * cand["period"]
        in_tr = np.abs(ph) < 0.5 * cand["duration"]
    else:
        in_tr = np.zeros(t.size, dtype=bool)
    fig.add_trace(
        go.Scattergl(
            x=t[~in_tr],
            y=_ppm(f[~in_tr]),
            mode="markers",
            name="Out of transit",
            marker={"size": 3, "color": BLUE, "opacity": 0.55},
            hovertemplate="t = %{x:.3f}<br>%{y:.0f} ppm<extra></extra>",
        )
    )
    if in_tr.any():
        fig.add_trace(
            go.Scattergl(
                x=t[in_tr],
                y=_ppm(f[in_tr]),
                mode="markers",
                name="During a detected dip",
                marker={"size": 5, "color": ORANGE},
                hovertemplate="t = %{x:.3f}<br>%{y:.0f} ppm<extra>in transit</extra>",
            )
        )
    return _layout(fig, "Detrended light curve", _time_label(result), "Brightness change (ppm)")


def periodogram_figure(cand: dict[str, Any]) -> go.Figure:
    """Step 3: BLS periodogram with the detected period (and its aliases) marked."""
    p, s = _xy(cand["plot_data"]["periodogram"])
    peak = float(cand["plot_data"]["periodogram_peak"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=p,
            y=s,
            mode="lines",
            name="Signal strength",
            line={"width": 1.5, "color": BLUE},
            hovertemplate="P = %{x:.4f} d<br>strength %{y:.1f}<extra></extra>",
        )
    )
    y_top = float(s.max()) * 1.08 if s.size else 1.0
    # Alias markers as traces (vline annotations misplace on log axes).
    ax, ay, at = [], [], []
    for mult, label in ((0.5, "P/2"), (2.0, "2P")):
        x = peak * mult
        if p.size and p.min() <= x <= p.max():
            ax += [x, x, None]
            ay += [0, y_top, None]
            at += ["", label, ""]
    if ax:
        fig.add_trace(
            go.Scatter(
                x=ax,
                y=ay,
                mode="lines+text",
                name="Aliases (P/2, 2P)",
                text=at,
                textposition="top center",
                textfont={"color": MUTED, "size": 11},
                line={"color": MUTED, "width": 1, "dash": "dot"},
                hoverinfo="skip",
            )
        )
    if p.size:
        y_peak = float(s[np.argmin(np.abs(p - peak))])
        fig.add_trace(
            go.Scatter(
                x=[peak],
                y=[y_peak],
                mode="markers+text",
                name="Detected period",
                marker={"size": 11, "color": ORANGE, "line": {"color": SURFACE, "width": 2}},
                text=[f"  P = {peak:.4f} d"],
                textposition="middle right",
                textfont={"color": TEXT},
                hovertemplate="P = %{x:.5f} d<extra>peak</extra>",
            )
        )
    fig = _layout(
        fig, "BLS periodogram", "Trial period (days, log scale)", "Detection strength (SNR)"
    )
    if p.size:
        fig.update_xaxes(type="log", range=[float(np.log10(p.min())), float(np.log10(p.max()))])
    fig.update_yaxes(range=[0, y_top * 1.08])
    return fig


def folded_figure(cand: dict[str, Any]) -> go.Figure:
    """Step 4: all transits stacked on one another, with binned data and the trapezoid fit."""
    fp = cand["plot_data"]["folded"]
    px, py = _xy(fp["points"])
    bx, by = _xy(fp["binned"])
    mx, my = _xy(fp["model"])
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=px,
            y=_ppm(py),
            mode="markers",
            name="Every observation",
            marker={"size": 3, "color": MUTED, "opacity": 0.45},
            hovertemplate="%{x:.2f} h<br>%{y:.0f} ppm<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bx,
            y=_ppm(by),
            mode="markers",
            name="Binned average",
            marker={"size": 8, "color": BLUE, "line": {"color": SURFACE, "width": 2}},
            hovertemplate="%{x:.2f} h<br>%{y:.0f} ppm<extra>binned</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mx,
            y=_ppm(my),
            mode="lines",
            name="Transit model fit",
            line={"width": 2.5, "color": ORANGE},
            hovertemplate="%{x:.2f} h<br>model %{y:.0f} ppm<extra></extra>",
        )
    )
    if py.size:
        lo, hi = np.percentile(_ppm(py), [0.5, 99.5])
        span = hi - lo
        fig.update_yaxes(range=[min(lo, float(_ppm(my).min())) - 0.1 * span, hi + 0.1 * span])
    return _layout(fig, "Folded transit", "Hours from mid-transit", "Brightness change (ppm)")


def probability_figure(cand: dict[str, Any]) -> go.Figure:
    """Horizontal bars, one per class, direct-labeled with the percentage."""
    probs = cand.get("probabilities") or {}
    order = ["FP", "EB", "PC"]  # bottom-to-top so the planet bar sits on top
    vals = [100 * float(probs.get(c.lower(), 0.0)) for c in order]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=[CLASS_LABELS[c] for c in order],
            orientation="h",
            marker={
                "color": [CLASS_COLORS[c] for c in order],
                "line": {"color": SURFACE, "width": 2},
            },
            text=[f"{v:.0f}%" for v in vals],
            textposition="outside",
            textfont={"color": TEXT},
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig = _layout(fig, "Model confidence", "", "", height=190)
    fig.update_xaxes(range=[0, 115], showticklabels=False, showgrid=False)
    fig.update_yaxes(tickfont={"color": TEXT, "size": 13})
    fig.update_layout(margin={"l": 10, "r": 10, "t": 35, "b": 10}, showlegend=False, bargap=0.35)
    return fig


def view_saliency_figure(
    view: dict[str, Any], saliency: list[float] | None, title: str, x_title: str
) -> go.Figure:
    """Model input view (line) with each bin colored by how much it drove the decision."""
    x, y = _xy(view)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="What the model sees",
            line={"width": 1.5, "color": MUTED},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    if saliency and len(saliency) == len(x):
        s = np.asarray(saliency, dtype=float)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                showlegend=False,
                marker={
                    "size": 8 + 10 * s,
                    "color": s,
                    "colorscale": SALIENCY_SCALE,
                    "cmin": 0,
                    "cmax": 1,
                    "line": {"color": SURFACE, "width": 1},
                    "colorbar": {
                        "title": {"text": "Influence", "font": {"color": TEXT_2}},
                        "tickvals": [0, 1],
                        "ticktext": ["low", "high"],
                        "tickfont": {"color": TEXT_2},
                        "thickness": 10,
                        "len": 0.8,
                    },
                },
                customdata=s,
                hovertemplate="%{x:.3f}<br>value %{y:.2f}<br>influence %{customdata:.2f}<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                showlegend=False,
                marker={"size": 6, "color": BLUE},
                hovertemplate="%{x:.3f}<br>%{y:.2f}<extra></extra>",
            )
        )
    fig = _layout(fig, title, x_title, "Normalized flux", height=300)
    fig.update_layout(margin={"l": 60, "r": 20, "t": 40, "b": 50})
    return fig


def feature_attribution_figure(features: dict[str, float], top_k: int = 8) -> go.Figure:
    """Signed attributions: blue pushes toward the predicted class, red pushes away."""
    items = sorted(features.items(), key=lambda kv: abs(kv[1]))[-top_k:]
    names = [k.replace("_", " ") for k, _ in items]
    vals = [v for _, v in items]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker={
                "color": [BLUE if v >= 0 else DIVERGING_NEG for v in vals],
                "line": {"color": SURFACE, "width": 2},
            },
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        )
    )
    fig = _layout(
        fig,
        "Which measurements mattered",
        "← pushed away · pushed toward →",
        "",
        height=60 + 32 * len(items),
    )
    fig.update_layout(showlegend=False, margin={"l": 10, "r": 20, "t": 35, "b": 45})
    fig.add_vline(x=0, line={"color": MUTED, "width": 1})
    return fig

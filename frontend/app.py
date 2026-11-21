"""Exoplanet Transit Hunter: Streamlit UI.

    streamlit run frontend/app.py          # or: make ui  /  make demo

The page tells one story, top to bottom:
  1 raw light curve -> 2 detrended -> 3 periodogram -> 4 folded transit -> 5 verdict -> 6 why.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from client import FRIENDLY_ERRORS, AnalyzeOptions, Backend, BackendError
from plots import (
    CLASS_COLORS,
    CLASS_LABELS,
    PLOT_CONFIG,
    detrended_figure,
    feature_attribution_figure,
    folded_figure,
    periodogram_figure,
    probability_figure,
    raw_figure,
    view_saliency_figure,
)

st.set_page_config(page_title="Exoplanet Transit Hunter", page_icon="🪐", layout="wide")

CATEGORY_ICONS = {
    "hot_jupiter": "🪐",
    "hot_jupiter_tess": "🛰️",
    "small_multi": "🌍",
    "small_tess": "🛰️",
    "eclipsing_binary": "⭐⭐",
    "false_positive": "📡",
}

# Panel captions: (technical, explain-like-I'm-12)
CAPTIONS: dict[str, tuple[str, str]] = {
    "raw": (
        "Normalized PDCSAP flux, each quarter/sector divided by its median. The orange line is the "
        "biweight trend that removes stellar variability and instrument drifts.",
        "This is how bright the star looked over time. Stars flicker and spin, and the telescope "
        "drifts a little. The orange line follows those slow wiggles so we can remove them.",
    ),
    "detrended": (
        "Flux divided by the trend. Orange points fall inside the selected candidate's transit windows.",
        "After removing the slow wiggles, the star looks steady. The orange dots are the moments "
        "when something seems to block a bit of its light.",
    ),
    "periodogram": (
        "Box Least Squares tries thousands of periods and durations and scores how well a periodic "
        "box-shaped dip fits. Dotted lines mark P/2 and 2P, where aliases appear.",
        "The computer tries every possible repeat time, from hours to months, and checks which one "
        "lines up the dips best. The tallest spike is the winner.",
    ),
    "folded": (
        "Every transit overlaid at the detected period and epoch. Blue: binned medians. Orange: "
        "least-squares trapezoid fit (depth, duration, ingress).",
        "If the dips really repeat, stacking them all on top of each other makes one clear dip. "
        "A flat bottom looks like a planet; a pointy V often means two stars.",
    ),
    "views": (
        "The exact inputs to the CNN: a 201-bin global view (whole orbit) and a 61-bin local view "
        "(the transit). Brighter, larger points had more influence on the prediction "
        "(Integrated Gradients).",
        "This is what the AI actually looks at. The bright, big dots show the parts that made "
        "it decide. For a planet, it should be staring at the dip.",
    ),
}


@st.cache_resource
def get_backend() -> Backend:
    return Backend()


def caption(key: str, eli12: bool) -> None:
    st.caption(CAPTIONS[key][1 if eli12 else 0])


def show_error(code: str, message: str) -> None:
    title, hint = FRIENDLY_ERRORS.get(code, FRIENDLY_ERRORS["internal_error"])
    st.error(f"**{title}**  \n{hint}", icon="🔭")
    with st.expander("Technical details"):
        st.code(f"{code}: {message}")


def badge(flag: dict[str, Any]) -> str:
    """Vetting badge: status color + icon + text, never color alone."""
    passed = flag["passed"]
    if passed is True:
        icon, color, word = "✓", "#0ca30c", "pass"
    elif passed is False:
        icon, color, word = "✗", "#d03b3b", "flag"
    else:
        icon, color, word = "–", "#8a897f", "n/a"
    name = flag["name"].replace("_", " ")
    return (
        f'<span title="{flag["message"]}" style="display:inline-block;margin:2px 4px 2px 0;padding:3px 9px;'
        f'border-radius:12px;border:1px solid {color};color:#f0efec;font-size:0.85rem;">'
        f'<span style="color:{color};font-weight:700">{icon}</span> {name} <span style="color:#8a897f">({word})</span></span>'
    )


def stat_grid(items: list[tuple[str, str, str]]) -> None:
    """Compact label/value grid (st.metric truncates long values in narrow columns)."""
    cells = "".join(
        f"<div title='{tip}' style='padding:6px 0'><div style='color:#8a897f;font-size:0.8rem'>{label}</div>"
        f"<div style='color:#f0efec;font-size:1.25rem;font-weight:600'>{value}</div></div>"
        for label, value, tip in items
    )
    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:4px 16px'>{cells}</div>",
        unsafe_allow_html=True,
    )


def run_analysis(target: str, opts: AnalyzeOptions) -> None:
    backend = get_backend()
    bar = st.progress(0.0, text=f"Starting analysis of {target}…")

    def progress(stage: str, frac: float) -> None:
        bar.progress(min(max(frac, 0.0), 1.0), text=f"{target}: {stage}…")

    try:
        result = backend.analyze(target, opts, progress)
    except BackendError as exc:
        bar.empty()
        st.session_state["error"] = (exc.code, exc.message)
        st.session_state.pop("result", None)
        return
    bar.empty()
    st.session_state["result"] = result
    st.session_state["error"] = None
    st.session_state["candidate_idx"] = 0


# ------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Settings")
    eli12 = st.toggle(
        "🧒 Explain like I'm 12", value=False, help="Plain-language captions and verdicts."
    )
    mission = st.selectbox("Mission", ["Auto", "Kepler", "TESS"], index=0)
    detrend = st.selectbox("Detrending", ["biweight (wotan)", "Savitzky–Golay"], index=0)
    max_cands = st.slider(
        "Max signals per star", 1, 5, 3, help="Iterative BLS: find, mask, search again."
    )
    max_segments = st.slider(
        "Max quarters / sectors", 1, 17, 8, help="Fewer = faster download and search."
    )
    force = st.checkbox("Ignore cached results", value=False)
    st.divider()
    backend = get_backend()
    health = backend.health()
    if health:
        model = health.get("model_kind") or "none"
        st.success(
            f"API online · model: **{model}**"
            + (" · offline mode" if health.get("offline") else ""),
            icon="🛰️",
        )
    else:
        st.warning(
            "API offline: running the pipeline inside the UI. Demo stars still work.", icon="⚠️"
        )
    st.caption(f"API: `{backend.api_url}`")

opts = AnalyzeOptions(
    mission=None if mission == "Auto" else mission,
    detrend_method="savgol" if detrend.startswith("Savitzky") else None,
    max_candidates=None if max_cands == 3 else max_cands,
    max_segments=max_segments,
    force=force,
)

# ------------------------------------------------------------------------------- header + search
st.markdown("# 🪐 Exoplanet Transit Hunter")
st.markdown(
    "Type a star. We download its brightness measurements from NASA's Kepler or TESS mission, "
    "look for dips that repeat like clockwork, and let a neural network decide: "
    "**planet**, **two stars eclipsing each other**, or **false alarm**."
)

with st.form("search", clear_on_submit=False, border=False):
    c1, c2 = st.columns([5, 1])
    query = c1.text_input(
        "Star",
        placeholder="Kepler-10, KIC 11904151, TIC 261136679, Pi Mensae…",
        label_visibility="collapsed",
    )
    submitted = c2.form_submit_button("Analyze", type="primary", use_container_width=True)
if submitted and query.strip():
    run_analysis(query.strip(), opts)

demos = get_backend().demo_targets()
if demos:
    st.markdown("**Try a demo star** (works offline):")
    cols = st.columns(min(len(demos), 6))
    for i, d in enumerate(demos[:6]):
        icon = CATEGORY_ICONS.get(d.get("category", ""), "✨")
        if cols[i].button(
            f"{icon} {d['title']}",
            key=f"demo_{d['slug']}",
            help=d.get("blurb", ""),
            use_container_width=True,
        ):
            # Demo buttons always use the cached result (default options).
            run_analysis(d["slug"], AnalyzeOptions())
else:
    st.info("No demo stars cached yet. Run `make warm-cache` (needs internet once).", icon="💡")

if st.session_state.get("error"):
    show_error(*st.session_state["error"])

result: dict[str, Any] | None = st.session_state.get("result")
if result is None:
    st.divider()
    a, b, c = st.columns(3)
    a.markdown(
        "#### 1 · Watch the star\nKepler and TESS measured the brightness of hundreds of thousands of stars every few minutes for months or years."
    )
    b.markdown(
        "#### 2 · Find the repeat\nA planet passing in front of its star causes the same small dip again and again. Box Least Squares finds that rhythm."
    )
    c.markdown(
        "#### 3 · Ask the AI\nA neural network trained on Kepler and TESS signals labeled by astronomers decides what caused the dip, and shows you why."
    )
    st.stop()

# ------------------------------------------------------------------------------- result header
lc = result["lightcurve"]
title = result.get("name") or result["target_id"]
st.divider()
meta = [result["target_id"], lc["mission"]] + (
    [lc["author"]] if lc["author"] != lc["mission"] else []
)
st.markdown(
    f"## {title}  <span style='color:#8a897f;font-size:1rem'>{' · '.join(meta)}</span>",
    unsafe_allow_html=True,
)
m = st.columns(5)
m[0].metric("Observations", f"{lc['n_points']:,}")
m[1].metric("Time span", f"{lc['baseline_days']:.0f} days")
m[2].metric("Quarters / sectors", len(lc["segments"]))
radius = result["stellar"].get("radius")
m[3].metric("Star radius", f"{radius:.2f} R☉" if radius else "unknown")
m[4].metric("Signals found", sum(c["significant"] for c in result["candidates"]))
src = {
    "demo_cache": "📦 demo cache",
    "cache": "⚡ cached result",
    "computed": f"⏱ computed in {result['timings_s'].get('total', 0):.1f}s",
}
st.caption(
    f"{src.get(result.get('source', 'computed'), '')} · pipeline {result['pipeline_version']}"
)
for w in result.get("warnings", []):
    st.warning(w, icon="⚠️")

cands = result["candidates"]
if len(cands) > 1:
    labels = [
        f"Signal {c['rank']}: P = {c['period']:.4f} d" + ("" if c["significant"] else " (weak)")
        for c in cands
    ]
    idx = st.radio(
        "Signal",
        range(len(cands)),
        format_func=lambda i: labels[i],
        horizontal=True,
        index=st.session_state.get("candidate_idx", 0),
        label_visibility="collapsed",
    )
    st.session_state["candidate_idx"] = idx
else:
    idx = 0
cand = cands[idx]

# ------------------------------------------------------------------------------- steps 1-4
st.markdown("### ① Raw light curve")
caption("raw", eli12)
st.plotly_chart(raw_figure(result), use_container_width=True, config=PLOT_CONFIG)

st.markdown("### ② Detrended light curve")
caption("detrended", eli12)
st.plotly_chart(
    detrended_figure(result, cand if cand["significant"] else None),
    use_container_width=True,
    config=PLOT_CONFIG,
)

left, right = st.columns(2)
with left:
    st.markdown("### ③ Search for a repeat")
    caption("periodogram", eli12)
    st.plotly_chart(periodogram_figure(cand), use_container_width=True, config=PLOT_CONFIG)
with right:
    st.markdown("### ④ Folded transit")
    caption("folded", eli12)
    st.plotly_chart(folded_figure(cand), use_container_width=True, config=PLOT_CONFIG)

# ------------------------------------------------------------------------------- step 5: verdict
st.markdown("### ⑤ Verdict")
v1, v2 = st.columns([3, 2])
with v1:
    pred = cand.get("predicted_class")
    if not cand["significant"]:
        st.markdown("#### No significant signal")
        st.caption(
            f"Best attempt: SNR {cand['snr']:.1f} (threshold 7.1). Treat any prediction below with caution."
        )
    if pred:
        color = CLASS_COLORS[pred]
        st.markdown(
            f"<div style='border-left:6px solid {color};padding:6px 14px;margin:4px 0 10px 0;'>"
            f"<div style='font-size:1.6rem;font-weight:700;color:#f0efec'>{CLASS_LABELS[pred]}</div>"
            f"<div style='color:#c3c2b7'>{cand['confidence']:.0%} confident</div></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(probability_figure(cand), use_container_width=True, config=PLOT_CONFIG)
    else:
        st.info("No trained model loaded. Showing search and vetting results only.", icon="🧪")
    st.markdown("**Vetting checks**", help="Hover a badge for details.")
    st.markdown("".join(badge(f) for f in cand["vetting_flags"]), unsafe_allow_html=True)
with v2:
    st.markdown("**What it means**")
    if eli12:
        st.markdown(cand["explanation"]["eli12"])
    else:
        for line in cand["explanation"]["technical"]:
            st.markdown(f"- {line}")
    phys = cand["physical"]
    st.markdown("**Signal**")
    rp = phys.get("planet_radius_rearth")
    rj = phys.get("planet_radius_rjup")
    stat_grid(
        [
            ("Period", f"{cand['period']:.4f} d", "time between dips"),
            ("Duration", f"{cand['duration_hours']:.2f} h", "how long each dip lasts"),
            (
                "Depth",
                f"{cand['depth_ppm']:,.0f} ppm",
                "parts per million of the star's light blocked",
            ),
            ("SNR", f"{cand['snr']:.1f}", "signal-to-noise ratio; 7.1 is the detection threshold"),
            ("Transits seen", f"{cand['n_transits']}", "individual dips with data"),
            (
                "Size if a planet",
                f"{rp:.1f} Earths" if rp else "—",
                f"{rj:.2f} Jupiter radii" if rj else "needs the star's radius",
            ),
        ]
    )
    st.caption(
        f"Mid-transit (BJD): {cand['t0_bjd']:.5f}. A candidate needs follow-up observations to be confirmed as a planet."
    )

# ------------------------------------------------------------------------------- step 6: why
st.markdown("### ⑥ Why the model decided this")
caption("views", eli12)
sal = cand.get("saliency") or {}
w1, w2 = st.columns(2)
with w1:
    st.plotly_chart(
        view_saliency_figure(
            cand["plot_data"]["local_view"],
            sal.get("local_view"),
            "Local view: the transit up close",
            "Hours from mid-transit",
        ),
        use_container_width=True,
        config=PLOT_CONFIG,
    )
with w2:
    st.plotly_chart(
        view_saliency_figure(
            cand["plot_data"]["global_view"],
            sal.get("global_view") or None,
            "Global view: the whole orbit",
            "Days from mid-transit",
        ),
        use_container_width=True,
        config=PLOT_CONFIG,
    )
if sal.get("features"):
    f1, f2 = st.columns([3, 2])
    with f1:
        st.plotly_chart(
            feature_attribution_figure(sal["features"]),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    with f2:
        st.caption(
            f"Method: {sal.get('method', '').replace('_', ' ')}, explaining the "
            f"“{CLASS_LABELS.get(sal.get('target_class', ''), '')}” score. "
            + (
                "Blue bars made the model more sure; red bars made it less sure."
                if eli12
                else "Signed attributions toward the predicted class (scalar inputs)."
            )
        )
        with st.expander("Table view"):
            rows = sorted(sal["features"].items(), key=lambda kv: -abs(kv[1]))
            st.dataframe(
                {"feature": [k for k, _ in rows], "attribution": [round(v, 4) for _, v in rows]},
                hide_index=True,
                use_container_width=True,
            )

st.divider()
st.download_button(
    "Download full result (JSON)",
    json.dumps(result),
    file_name=f"{result['target_id'].replace(' ', '_')}.json",
    mime="application/json",
)
st.caption(
    "Data: NASA MAST (Kepler, TESS) via lightkurve; labels: NASA Exoplanet Archive. "
    "This tool flags candidates for follow-up. It does not confirm planets."
)

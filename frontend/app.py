"""Exoplanet Transit Hunter: Streamlit UI.

    streamlit run frontend/app.py          # or: make ui  /  make demo

Landing: hero, search, demo stars, how it works.
Result:  header -> 3D model of the system + verdict -> key numbers -> tabs
         (light curve, finding the repeat, why the model decided this, details).
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
from system3d import render_system
from ui import CSS, badge, hero, how_it_works, section, stat_strip, title_block, verdict_card

st.set_page_config(
    page_title="Exoplanet Transit Hunter",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CSS, unsafe_allow_html=True)

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
    "system3d": (
        "Rebuilt from the measurement: star color from Teff, companion radius from sqrt(depth), "
        "orbit from the period and Kepler's third law, tilt from the transit duration. Drag to "
        "rotate, scroll to zoom.",
        "We rebuilt the star and the thing orbiting it from the dip we measured. Drag to spin it, "
        "and press 🔭 to see it the way the telescope does: the dip in the strip below happens "
        "every time the companion passes in front.",
    ),
}


@st.cache_resource
def get_backend() -> Backend:
    return Backend()


def caption(key: str, eli12: bool) -> None:
    st.caption(CAPTIONS[key][1 if eli12 else 0])


def html(block: str) -> None:
    st.markdown(block, unsafe_allow_html=True)


def show_error(code: str, message: str) -> None:
    title, hint = FRIENDLY_ERRORS.get(code, FRIENDLY_ERRORS["internal_error"])
    st.error(f"**{title}**  \n{hint}", icon="🔭")
    with st.expander("Technical details"):
        st.code(f"{code}: {message}")


def run_analysis(target: str, opts: AnalyzeOptions) -> None:
    backend = get_backend()
    bar = st.progress(0.0, text=f"Starting analysis of {target}…")

    def progress(stage: str, frac: float) -> None:
        bar.progress(min(max(frac, 0.0), 1.0), text=f"🔭 {target}: {stage}…")

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
    st.markdown("### ⚙️ Settings")
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
html(hero())

with st.form("search", clear_on_submit=False, border=False):
    c1, c2 = st.columns([6, 1])
    query = c1.text_input(
        "Star",
        placeholder="🔎  Search a star: Kepler-10, KIC 11904151, TIC 261136679, Pi Mensae…",
        label_visibility="collapsed",
    )
    submitted = c2.form_submit_button("Analyze 🚀", type="primary", use_container_width=True)
if submitted and query.strip():
    run_analysis(query.strip(), opts)

demos = get_backend().demo_targets()
if demos:
    st.caption("✨ **Try a demo star** · instant, works offline · settings are in the ☰ sidebar")
    cols = st.columns(min(len(demos), 6))
    for i, d in enumerate(demos[:6]):
        icon = CATEGORY_ICONS.get(d.get("category", ""), "✨")
        with cols[i]:
            if st.button(
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
    html(section("How it works", "From starlight to a verdict in four steps"))
    for col, card in zip(st.columns(4), how_it_works(), strict=True):
        with col:
            html(card)
    st.stop()

# ------------------------------------------------------------------------------- result header
lc = result["lightcurve"]
title = result.get("name") or result["target_id"]
src = {
    "demo_cache": "📦 demo cache",
    "cache": "⚡ cached result",
    "computed": f"⏱ computed in {result['timings_s'].get('total', 0):.1f}s",
}
chips = ([result["target_id"]] if result["target_id"] != title else []) + [f"🛰️ {lc['mission']}"]
if lc["author"] != lc["mission"]:
    chips.append(lc["author"])
chips += [
    f"{lc['n_points']:,} measurements",
    f"{lc['baseline_days']:.0f} days observed",
    f"{len(lc['segments'])} quarters/sectors",
    src.get(result.get("source", "computed"), "computed"),
]
st.divider()
html(title_block(title, chips))
for w in result.get("warnings", []):
    st.warning(w, icon="⚠️")

cands = result["candidates"]
if len(cands) > 1:
    labels = [
        f"Signal {c['rank']} · every {c['period']:.4f} d"
        + (f" · {CLASS_LABELS[c['predicted_class']]}" if c.get("predicted_class") else "")
        + ("" if c["significant"] else " (weak)")
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

# ------------------------------------------------------------------------------- 3D + verdict
left, right = st.columns([3, 2], gap="medium")
with left:
    html(section("The system", "Meet the system in 3D"))
    caption("system3d", eli12)
    render_system(result, cand if cand["significant"] else None, height=520)
with right:
    html(section("The answer", "What is it?"))
    html(verdict_card(cand))
    if cand.get("predicted_class"):
        st.plotly_chart(probability_figure(cand), use_container_width=True, config=PLOT_CONFIG)
    st.markdown("**Vetting checks**", help="Classic astronomer tests. Hover a badge for details.")
    html("".join(badge(f) for f in cand["vetting_flags"]))
    st.markdown("**What it means**")
    if eli12:
        st.markdown(cand["explanation"]["eli12"])
    else:
        for line in cand["explanation"]["technical"]:
            st.markdown(f"- {line}")

phys = cand["physical"]
rp = phys.get("planet_radius_rearth")
rj = phys.get("planet_radius_rjup")
radius = result["stellar"].get("radius")
html(
    stat_strip(
        [
            ("Repeats every", f"{cand['period']:.4f} d", "time between dips"),
            ("Dip lasts", f"{cand['duration_hours']:.2f} h", "transit duration"),
            ("Dip depth", f"{cand['depth_ppm']:,.0f} ppm", "parts per million of light blocked"),
            ("Signal strength", f"SNR {cand['snr']:.1f}", "7.1 is the detection threshold"),
            (
                "Size if a planet",
                f"{rp:.1f} × Earth" if rp else "—",
                f"{rj:.2f} Jupiter radii" if rj else "needs the star's radius",
            ),
            (
                "Host star",
                f"{radius:.2f} × Sun" if radius else "unknown",
                f"{cand['n_transits']} transits seen",
            ),
        ]
    )
)

# ------------------------------------------------------------------------------- tabs
t1, t2, t3, t4 = st.tabs(
    ["📈 Light curve", "🔁 Finding the repeat", "🧠 Why the AI decided", "📋 Details & download"]
)
with t1:
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
with t2:
    a, b = st.columns(2)
    with a:
        st.markdown("### ③ Search for a repeat")
        caption("periodogram", eli12)
        st.plotly_chart(periodogram_figure(cand), use_container_width=True, config=PLOT_CONFIG)
    with b:
        st.markdown("### ④ Folded transit")
        caption("folded", eli12)
        st.plotly_chart(folded_figure(cand), use_container_width=True, config=PLOT_CONFIG)
with t3:
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
                    {
                        "feature": [k for k, _ in rows],
                        "attribution": [round(v, 4) for _, v in rows],
                    },
                    hide_index=True,
                    use_container_width=True,
                )
    else:
        st.info("No explanation available for this signal.", icon="🧪")
with t4:
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Measurement**")
        st.markdown(
            f"- Mid-transit (BJD): `{cand['t0_bjd']:.5f}`\n"
            f"- Period: `{cand['period']:.6f}` d · duration `{cand['duration_hours']:.3f}` h\n"
            f"- Depth: `{cand['depth_ppm']:,.1f}` ppm · SNR `{cand['snr']:.1f}` · SDE `{cand['sde']:.1f}`\n"
            f"- Transits with data: `{cand['n_transits']}`"
        )
        if phys.get("fit_depth_ppm"):
            st.markdown(
                f"- Trapezoid fit: depth `{phys['fit_depth_ppm']:,.1f}` ppm, "
                f"duration `{phys.get('fit_duration_hours') or 0:.3f}` h, "
                f"ingress fraction `{phys.get('fit_ingress_fraction') or 0:.2f}`"
            )
    with d2:
        st.markdown("**Data & model**")
        st.markdown(
            f"- Mission: `{lc['mission']}` · pipeline `{result['pipeline_version']}`\n"
            f"- Measurements: `{lc['n_points']:,}` over `{lc['baseline_days']:.1f}` days\n"
            f"- Model: `{(result.get('model') or {}).get('kind') or 'none'}`"
        )
    st.download_button(
        "⬇️ Download full result (JSON)",
        json.dumps(result),
        file_name=f"{result['target_id'].replace(' ', '_')}.json",
        mime="application/json",
    )

st.divider()
st.caption(
    "Data: NASA MAST (Kepler, TESS) via lightkurve; labels: NASA Exoplanet Archive and the "
    "Prša et al. 2022 TESS eclipsing-binary catalog. This tool flags candidates for follow-up. "
    "It does not confirm planets."
)

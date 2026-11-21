"""Styling and small HTML building blocks for the Streamlit UI.

All text uses text tokens (primary / secondary / muted); class colors only mark identity
(a dot, a ring, a border) and every colored mark sits next to a text label.
"""

from __future__ import annotations

import html
from typing import Any

from plots import CLASS_COLORS, CLASS_LABELS, CRITICAL, GOOD, MUTED, TEXT, TEXT_2

CSS = """
<style>
:root { --th-surface:#1a1a19; --th-border:#2c2c2a; --th-text:#f0efec; --th-text2:#c3c2b7; --th-muted:#8a897f; --th-accent:#3987e5; }
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1360px; }
h1, h2, h3, h4 { letter-spacing: .2px; }

/* hero */
.th-hero { position: relative; overflow: hidden; border-radius: 20px; padding: 30px 34px 26px; margin: 0 0 18px;
  border: 1px solid var(--th-border);
  background:
    radial-gradient(1.5px 1.5px at 12% 22%, #fff9 50%, transparent 51%),
    radial-gradient(1px 1px at 28% 68%, #fff7 50%, transparent 51%),
    radial-gradient(1.5px 1.5px at 44% 34%, #fff8 50%, transparent 51%),
    radial-gradient(1px 1px at 63% 78%, #fff6 50%, transparent 51%),
    radial-gradient(1.5px 1.5px at 77% 18%, #fff9 50%, transparent 51%),
    radial-gradient(1px 1px at 88% 58%, #fff7 50%, transparent 51%),
    radial-gradient(1px 1px at 53% 12%, #fff6 50%, transparent 51%),
    radial-gradient(ellipse at 88% 120%, rgba(217,89,38,.35) 0%, transparent 55%),
    radial-gradient(ellipse at 0% 0%, rgba(57,135,229,.28) 0%, transparent 55%),
    linear-gradient(160deg, #0d1322 0%, #0a0c14 60%, #111110 100%); }
.th-hero .th-planet { position: absolute; right: 48px; top: 50%; transform: translateY(-50%); width: 132px; height: 132px; border-radius: 50%;
  background: radial-gradient(circle at 34% 30%, #ffe7b3 0%, #f3a64a 38%, #b4521c 72%, #5a240b 100%);
  box-shadow: 0 0 60px 12px rgba(243,166,74,.25); }
.th-hero .th-planet:after { content:""; position:absolute; left:-26px; top:58px; width:184px; height:16px; border-radius:50%;
  border: 2px solid rgba(240,239,236,.35); transform: rotate(-14deg); }
.th-hero h1 { margin: 0; font-size: 2.35rem; line-height: 1.15; color: var(--th-text); }
.th-hero p { margin: 8px 0 0; max-width: 760px; color: var(--th-text2); font-size: 1.04rem; }
.th-chips { margin-top: 14px; }
.th-chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px; margin: 0 6px 6px 0; border-radius: 999px;
  border: 1px solid var(--th-border); background: rgba(26,26,25,.75); color: var(--th-text2); font-size: .82rem; }
@media (max-width: 900px) { .th-hero .th-planet { display: none; } }

/* cards */
.th-card { background: var(--th-surface); border: 1px solid var(--th-border); border-radius: 16px; padding: 16px 18px; height: 100%; box-sizing: border-box; }
.th-steps .th-card { min-height: 178px; }
.th-card h4 { margin: 0 0 6px; font-size: 1.02rem; color: var(--th-text); }
.th-card p { margin: 0; color: var(--th-text2); font-size: .92rem; }
.th-step { color: var(--th-accent); font-weight: 700; font-size: .78rem; letter-spacing: .6px; text-transform: uppercase; }
.th-kicker { color: var(--th-muted); font-size: .76rem; font-weight: 700; letter-spacing: .8px; text-transform: uppercase; }
.th-section { margin: 26px 0 8px; }
.th-section .th-kicker { margin-bottom: 2px; }
.th-section h3 { margin: 0; padding: 0; }

/* result header */
.th-title { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin: 6px 0 2px; }
.th-title h2 { margin: 0; padding: 0; }

/* verdict */
.th-verdict { position: relative; }
.th-vrow { display: flex; align-items: center; gap: 18px; margin-top: 8px; }
.th-ring { --p: 50; --c: #3987e5; width: 96px; height: 96px; flex: 0 0 96px; border-radius: 50%; display: grid; place-items: center;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), #2c2c2a 0); }
.th-ring > div { width: 76px; height: 76px; border-radius: 50%; background: var(--th-surface); display: grid; place-items: center; text-align: center; }
.th-ring b { font-size: 1.35rem; color: var(--th-text); line-height: 1; }
.th-ring span { display: block; font-size: .68rem; color: var(--th-muted); margin-top: 2px; }
.th-vlabel { font-size: 1.55rem; font-weight: 700; color: var(--th-text); line-height: 1.2; }
.th-vsub { color: var(--th-text2); font-size: .95rem; margin-top: 3px; }
.th-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 7px; vertical-align: middle; }

/* stat strip */
.th-stats { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin: 6px 0 4px; }
.th-stat { background: var(--th-surface); border: 1px solid var(--th-border); border-radius: 14px; padding: 11px 14px; }
.th-stat .l { color: var(--th-muted); font-size: .78rem; }
.th-stat .v { color: var(--th-text); font-size: 1.28rem; font-weight: 650; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.th-stat .h { color: var(--th-muted); font-size: .74rem; margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
@media (max-width: 1100px) { .th-stats { grid-template-columns: repeat(3, minmax(0, 1fr)); } }

/* badges */
.th-badge { display: inline-block; margin: 3px 5px 3px 0; padding: 4px 10px; border-radius: 999px; border: 1px solid; color: var(--th-text);
  font-size: .84rem; background: rgba(255,255,255,.02); }

/* streamlit widgets */
div[data-testid="stForm"] { border: 1px solid var(--th-border); background: var(--th-surface); border-radius: 16px; padding: 10px 12px 4px; }
div[data-testid="stForm"] input { font-size: 1.05rem; }
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button { border-radius: 10px; font-weight: 600; }
.stButton > button { min-height: 46px; border: 1px solid var(--th-border); background: var(--th-surface); transition: all .15s ease; }
.stButton > button:hover { border-color: var(--th-accent); transform: translateY(-1px); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--th-border); }
.stTabs [data-baseweb="tab"] { padding: 8px 14px; border-radius: 10px 10px 0 0; font-weight: 600; }
.stTabs [aria-selected="true"] { background: var(--th-surface); }
div[data-testid="stPlotlyChart"] { border: 1px solid var(--th-border); border-radius: 14px; overflow: hidden; }
section[data-testid="stSidebar"] { border-right: 1px solid var(--th-border); }
footer { visibility: hidden; }
</style>
"""

VERDICT_LINES = {
    "PC": "The dip looks like a planet passing in front of its star.",
    "EB": "The dip looks like two stars eclipsing each other.",
    "FP": "Probably not a planet: noise, a blended neighbor, or an instrument effect.",
}


def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def hero() -> str:
    return (
        "<div class='th-hero'><div class='th-planet'></div>"
        "<h1>Exoplanet Transit Hunter</h1>"
        "<p>Type any star. We download its brightness from NASA's <b>Kepler</b> or <b>TESS</b> telescope, "
        "find dips that repeat like clockwork, and a neural network decides: a <b>planet</b>, "
        "<b>two stars eclipsing each other</b>, or a <b>false alarm</b>. Then we rebuild the system in 3D.</p>"
        "<div class='th-chips'>"
        "<span class='th-chip'>🛰️ Kepler + TESS data</span>"
        "<span class='th-chip'>🔁 Box Least Squares search</span>"
        "<span class='th-chip'>🧠 CNN + LightGBM classifier</span>"
        "<span class='th-chip'>🔍 Explains every decision</span>"
        "<span class='th-chip'>🪐 3D system model</span>"
        "</div></div>"
    )


def how_it_works() -> list[str]:
    steps = [
        (
            "Step 1",
            "Watch the star",
            "Kepler and TESS measured the brightness of hundreds of thousands of stars every few minutes for months or years.",
        ),
        (
            "Step 2",
            "Find the repeat",
            "A planet passing in front of its star causes the same small dip again and again. Box Least Squares finds that rhythm.",
        ),
        (
            "Step 3",
            "Ask the AI",
            "A neural network trained on signals labeled by astronomers decides what caused the dip, and shows you why.",
        ),
        (
            "Step 4",
            "See it in 3D",
            "From the dip's depth, length and timing we rebuild the star and its companion, and replay the transit.",
        ),
    ]
    return [
        f"<div class='th-steps'><div class='th-card'><div class='th-step'>{a}</div><h4>{esc(b)}</h4><p>{esc(c)}</p></div></div>"
        for a, b, c in steps
    ]


def section(kicker: str, title: str) -> str:
    return f"<div class='th-section'><div class='th-kicker'>{esc(kicker)}</div><h3>{esc(title)}</h3></div>"


def chip(text: str) -> str:
    return f"<span class='th-chip'>{esc(text)}</span>"


def title_block(title: str, chips: list[str]) -> str:
    return (
        f"<div class='th-title'><h2>{esc(title)}</h2></div>"
        f"<div class='th-chips' style='margin-top:4px'>{''.join(chip(c) for c in chips)}</div>"
    )


def verdict_card(cand: dict[str, Any]) -> str:
    """Big verdict with a confidence ring. Contains the word 'Verdict' as its heading."""
    pred = cand.get("predicted_class")
    if not pred:
        return (
            "<div class='th-card th-verdict'><div class='th-kicker'>Verdict</div>"
            "<div class='th-vlabel' style='margin-top:6px'>No trained model loaded</div>"
            "<div class='th-vsub'>Showing the search and vetting results only.</div></div>"
        )
    color = CLASS_COLORS[pred]
    pct = round(float(cand.get("confidence") or 0) * 100)
    weak = (
        ""
        if cand.get("significant")
        else f"<div class='th-vsub' style='color:{MUTED}'>⚠️ Weak signal (SNR {cand['snr']:.1f}, threshold 7.1). Treat with caution.</div>"
    )
    return (
        f"<div class='th-card th-verdict' style='border-top:4px solid {color}'>"
        "<div class='th-kicker'>Verdict</div>"
        "<div class='th-vrow'>"
        f"<div class='th-ring' style='--p:{pct};--c:{color}'><div><div><b>{pct}%</b><span>confident</span></div></div></div>"
        f"<div><div class='th-vlabel'><span class='th-dot' style='background:{color}'></span>{esc(CLASS_LABELS[pred])}</div>"
        f"<div class='th-vsub'>{esc(VERDICT_LINES[pred])}</div>{weak}</div>"
        "</div></div>"
    )


def stat_strip(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f"<div class='th-stat' title='{esc(tip)}'><div class='l'>{esc(label)}</div>"
        f"<div class='v'>{esc(value)}</div><div class='h'>{esc(tip)}</div></div>"
        for label, value, tip in items
    )
    return f"<div class='th-stats'>{cells}</div>"


def badge(flag: dict[str, Any]) -> str:
    """Vetting badge: status color + icon + text, never color alone."""
    passed = flag["passed"]
    if passed is True:
        icon, color, word = "✓", GOOD, "pass"
    elif passed is False:
        icon, color, word = "✗", CRITICAL, "flag"
    else:
        icon, color, word = "–", MUTED, "n/a"
    name = flag["name"].replace("_", " ")
    return (
        f"<span class='th-badge' title='{esc(flag['message'])}' style='border-color:{color}'>"
        f"<span style='color:{color};font-weight:700'>{icon}</span> {esc(name)} "
        f"<span style='color:{MUTED}'>({word})</span></span>"
    )


__all__ = [
    "CSS",
    "TEXT",
    "TEXT_2",
    "badge",
    "chip",
    "hero",
    "how_it_works",
    "section",
    "stat_strip",
    "title_block",
    "verdict_card",
]

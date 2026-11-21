"""Frontend: figure builders, the backend client's local fallback, and a headless app run."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
sys.path.insert(0, str(FRONTEND))

from client import FRIENDLY_ERRORS, AnalyzeOptions, Backend, BackendError  # noqa: E402
from plots import (  # noqa: E402
    detrended_figure,
    feature_attribution_figure,
    folded_figure,
    periodogram_figure,
    probability_figure,
    raw_figure,
    view_saliency_figure,
)

from conftest import write_fake_star  # noqa: E402
from transit_hunter.config import data_path  # noqa: E402
from transit_hunter.pipeline import analyze  # noqa: E402

Env = tuple[object, Path, Path]
DEAD_API = "http://127.0.0.1:9"  # nothing listens here -> forces the in-process fallback


@pytest.fixture()
def result(trained_settings: Env) -> dict[str, Any]:
    write_fake_star("KIC 800001", "PC", 3)
    res = analyze("KIC 800001", stellar_radius=1.0, force=True)
    out: dict[str, Any] = json.loads(res.model_dump_json())
    return out


def test_every_figure_builds(result: dict[str, Any]) -> None:
    cand = result["candidates"][0]
    figs = [
        raw_figure(result),
        detrended_figure(result, cand),
        detrended_figure(result, None),
        periodogram_figure(cand),
        folded_figure(cand),
        probability_figure(cand),
        view_saliency_figure(
            cand["plot_data"]["local_view"], cand["saliency"]["local_view"], "L", "h"
        ),
        view_saliency_figure(cand["plot_data"]["global_view"], None, "G", "d"),
        feature_attribution_figure(cand["saliency"]["features"]),
    ]
    for fig in figs:
        assert fig.data, "figure has no traces"
        fig.to_json()  # serializable for the browser
    # Periodogram x-range is pinned to the data (no runaway log axis).
    lo, hi = figs[3].layout.xaxis.range
    assert 10**lo <= cand["period"] <= 10**hi


def test_backend_local_fallback(result: dict[str, Any]) -> None:
    backend = Backend(api_url=DEAD_API)
    assert backend.health() is None and backend.mode == "local"
    stages: list[str] = []
    out = backend.analyze(
        "KIC 800001", AnalyzeOptions(max_candidates=1), lambda s, f: stages.append(s)
    )
    assert out["target_id"] == "KIC 800001"
    assert stages and stages[-1] == "done"
    with pytest.raises(BackendError) as err:
        backend.analyze("Kepler-10", AnalyzeOptions(), lambda s, f: None)
    assert err.value.code == "offline_not_cached"
    assert err.value.code in FRIENDLY_ERRORS


def test_backend_demo_targets_from_disk(result: dict[str, Any]) -> None:
    demo = data_path("demo_cache")
    (demo / "index.json").write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "slug": "s",
                        "title": "T",
                        "query": "KIC 800001",
                        "target_id": "KIC 800001",
                        "mission": "Kepler",
                        "category": "small_multi",
                        "blurb": "b",
                    }
                ]
            }
        )
    )
    try:
        assert Backend(api_url=DEAD_API).demo_targets()[0]["slug"] == "s"
    finally:
        (demo / "index.json").unlink()


def test_app_renders_landing_and_result(
    trained_settings: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("TH_API_URL", DEAD_API)
    write_fake_star("KIC 800002", "EB", 4)
    at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Exoplanet Transit Hunter" in m.value for m in at.markdown)
    at.text_input[0].set_value("KIC 800002")
    at.button[0].click()  # the form's Analyze button
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Verdict" in text and "Why the model decided this" in text
    # ELI12 toggle switches the verdict text to plain language.
    at.toggle[0].set_value(True)
    at.run()
    assert not at.exception


def test_system3d_params_match_known_planet() -> None:
    """Pi Mensae c: a/R* ~ 13, i ~ 87-88 deg, Teq ~ 1170 K (Huang et al. 2018)."""
    from system3d import scene_html, system_params, teff_to_hex

    res = {
        "target_id": "TIC 261136679",
        "name": "Pi Mensae",
        "lightcurve": {"mission": "TESS"},
        "stellar": {"radius": 1.10, "teff": 6037, "mass": 1.09},
    }
    cand = {
        "significant": True,
        "period": 6.268,
        "duration_hours": 2.9,
        "depth_ppm": 290,
        "predicted_class": "PC",
        "confidence": 0.92,
        "physical": {"planet_radius_rearth": 2.04},
    }
    p = system_params(res, cand)
    c = p["companion"]
    assert c["kind"] == "planet" and c["label"] == "sub-Neptune"
    assert 12 < c["a_over_r"] < 15
    assert 86 < c["inclination_deg"] < 89.5
    assert 1050 < c["teq_k"] < 1300
    assert p["star"]["spectral_type"] == "F" and not p["assumed"]
    # Hot stars are blue-white, cool stars orange-red.
    assert teff_to_hex(3200)[1:3] == "ff" and int(teff_to_hex(3200)[5:7], 16) < 0x90
    assert teff_to_hex(15000)[5:7] == "ff"
    page = scene_html(p)
    assert "WebGLRenderer" in page and '"Pi Mensae"' in page


def test_system3d_eb_and_missing_values() -> None:
    from system3d import system_params

    res = {"target_id": "KIC 1", "stellar": {"radius": None, "teff": None, "mass": None}}
    eb = {
        "significant": True,
        "period": 1.81,
        "duration_hours": 4.6,
        "depth_ppm": 655630,
        "predicted_class": "EB",
        "physical": {},
    }
    p = system_params(res, eb)
    assert p["companion"]["kind"] == "star"
    assert set(p["assumed"]) == {"radius", "teff", "mass"}
    assert system_params(res, dict(eb, significant=False))["companion"] is None
    assert system_params(res, None)["companion"] is None

"""End-to-end pipeline tests on cached synthetic stars (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from transit_hunter.config import data_path, get_settings
from transit_hunter.data.fetch import LightCurveData, cache_path
from transit_hunter.data.synthetic import make_time_grid, synthetic_star, trapezoid_transit
from transit_hunter.pipeline import (
    PipelineError,
    analyze,
    demo_lookup,
    downsample,
    fit_trapezoid,
    hunt,
    normalize_query,
    reset_predictor,
)

Env = tuple[object, Path, Path]


def _planet_star(star_id: str) -> None:
    t = make_time_grid(90.0, start=130.0)
    star, white = synthetic_star(t, noise_ppm=200, variability_ppm=1500, seed=99)
    model = trapezoid_transit(t, 3.3, 131.0, 0.13, 0.004)
    LightCurveData(
        star_id,
        "Kepler",
        t,
        star * model + white,
        np.full_like(t, 2e-4),
        "Kepler",
        1800.0,
        [1, 2, 3],
    ).save(cache_path(star_id, "Kepler"))


def _noise_star(star_id: str) -> None:
    t = make_time_grid(60.0, start=130.0)
    star, white = synthetic_star(t, noise_ppm=300, variability_ppm=1000, red_noise_ppm=0, seed=7)
    LightCurveData(
        star_id, "Kepler", t, star + white, np.full_like(t, 3e-4), "Kepler", 1800.0, [1]
    ).save(cache_path(star_id, "Kepler"))


def test_analyze_planet_full_result(trained_settings: Env) -> None:
    _planet_star("KIC 900001")
    res = analyze("KIC 900001", stellar_radius=1.0)
    assert res.status == "ok"
    assert res.target_id == "KIC 900001"
    assert res.source == "computed"
    c = res.candidates[0]
    assert c.rank == 1 and c.significant
    assert c.period == pytest.approx(3.3, rel=0.01)
    assert c.probabilities is not None
    p = c.probabilities
    assert p.pc + p.eb + p.fp == pytest.approx(1.0, abs=1e-5)
    assert c.predicted_class in ("PC", "EB", "FP")
    assert {f.name for f in c.vetting_flags} >= {"odd_even", "secondary_eclipse", "transit_shape"}
    assert c.saliency is not None and len(c.saliency.local_view) == 61
    assert c.physical.planet_radius_rearth == pytest.approx(np.sqrt(0.004) * 109.08, rel=0.25)
    assert c.physical.fit_depth_ppm == pytest.approx(4000, rel=0.2)
    assert len(c.plot_data.global_view.y) == 201
    assert len(c.plot_data.local_view.y) == 61
    assert len(c.plot_data.folded.model.x) == 400
    assert c.plot_data.periodogram.x and max(c.plot_data.periodogram.y) > 0
    assert (
        "dimmer" in c.explanation.eli12
        or "dims" in c.explanation.eli12
        or "dip" in c.explanation.eli12
    )
    assert c.explanation.technical
    assert len(res.plots.raw.x) <= 3000
    assert res.timings_s["total"] > 0
    assert len(res.model_dump_json()) < 1_500_000
    assert res.model.kind == "cnn"


def test_noise_only_star_reports_no_significant_signal(trained_settings: Env) -> None:
    _noise_star("KIC 900002")
    res = analyze("KIC 900002", stellar_radius=1.0)
    assert res.status == "no_significant_signal"
    assert not any(c.significant for c in res.candidates)
    assert any("No significant" in w for w in res.warnings)
    assert "too faint" in res.candidates[0].explanation.eli12


def test_offline_unknown_target_raises(trained_settings: Env) -> None:
    with pytest.raises(PipelineError) as err:
        analyze("Kepler-10")
    assert err.value.code == "offline_not_cached"


def test_analyze_without_model(
    trained_settings: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TH_MODELS_DIR", str(tmp_path / "empty_models"))
    get_settings.cache_clear()
    reset_predictor()
    _planet_star("KIC 900003")
    res = analyze("KIC 900003", stellar_radius=1.0)
    assert res.candidates[0].probabilities is None
    assert res.model.kind is None
    assert any("No trained model" in w for w in res.warnings)


def test_demo_cache_is_served(trained_settings: Env) -> None:
    _planet_star("KIC 900004")
    res = analyze("KIC 900004", stellar_radius=1.0)
    demo = data_path("demo_cache")
    (demo / "my-demo.json").write_text(res.model_dump_json())
    (demo / "index.json").write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "slug": "my-demo",
                        "title": "My Demo Star",
                        "query": "KIC 900004",
                        "target_id": "KIC 900004",
                        "mission": "Kepler",
                        "category": "x",
                        "blurb": "y",
                    }
                ]
            }
        )
    )
    assert demo_lookup("my demo star") is not None
    cached = analyze("My-Demo")
    assert cached.source == "demo_cache" and cached.cached
    assert cached.candidates[0].period == res.candidates[0].period
    fresh = analyze("KIC 900004", force=True, stellar_radius=1.0)
    assert fresh.source == "computed"
    (demo / "index.json").unlink()


def test_hunt_ranks_and_records_failures(trained_settings: Env) -> None:
    _planet_star("KIC 900005")
    _noise_star("KIC 900006")
    out = hunt(["KIC 900006", "KIC 900005", "Kepler-10"])
    statuses = [r.status for r in out.ranked]
    assert statuses[0] == "ok"  # the significant signal ranks first
    assert out.ranked[0].target_id == "KIC 900005"
    assert "offline_not_cached" in statuses


def test_fit_trapezoid_recovers_shape() -> None:
    x = np.linspace(-0.5, 0.5, 3000)
    y = trapezoid_transit(x, 1e6, 0.01, 0.2, 0.005, 0.2) + np.random.default_rng(0).normal(
        0, 1e-4, x.size
    )
    fit = fit_trapezoid(x, y, duration=0.15, depth=0.004)
    assert fit["depth"] == pytest.approx(0.005, rel=0.05)
    assert fit["duration"] == pytest.approx(0.2, rel=0.05)
    assert fit["t_offset"] == pytest.approx(0.01, abs=0.005)


def test_downsample_and_normalize_query() -> None:
    x = np.arange(10_000.0)
    s = downsample(x, x * 2, 1000)
    assert len(s.x) == 1000 and s.y[0] == pytest.approx(2 * s.x[0])
    assert normalize_query("  Kepler 10 ") == normalize_query("kepler-10") == "kepler10"


def test_tess_star_warns_when_model_is_kepler_only(trained_settings: Env) -> None:
    from conftest import write_fake_star

    write_fake_star("TIC 700001", "PC", 3, mission="TESS")
    res = analyze("TIC 700001", stellar_radius=1.0, max_candidates=1)
    assert res.lightcurve.mission == "TESS"
    assert any("outside its training domain" in w for w in res.warnings)
    assert res.candidates[0].features["is_tess"] == 1.0

"""Phase folding, view generation, vetting diagnostics and features."""

from __future__ import annotations

import numpy as np
import pytest

from transit_hunter.config import DataConfig
from transit_hunter.data.synthetic import SyntheticLightCurve, synthetic_eb_lightcurve
from transit_hunter.processing.clean import clean_lightcurve
from transit_hunter.processing.detrend import detrend
from transit_hunter.processing.features import SCALAR_FEATURES, build_example
from transit_hunter.processing.fold import (
    ViewError,
    bin_median,
    make_views,
    normalize_view,
    phase_fold,
)


def _flat(lc: SyntheticLightCurve, cfg: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    c = clean_lightcurve(lc.time, lc.flux, lc.flux_err, cfg.cleaning)
    d = detrend(c.time, c.flux, c.flux_err, cfg.detrend)
    return d.time, d.flux


def test_phase_fold_range_and_center() -> None:
    t = np.linspace(0, 100, 10001)
    ph = phase_fold(t, 7.0, 2.0)
    assert ph.min() >= -3.5
    assert ph.max() < 3.5
    assert phase_fold(np.array([2.0, 9.0, 16.0]), 7.0, 2.0) == pytest.approx([0, 0, 0], abs=1e-9)


def test_bin_median_counts_and_overlap() -> None:
    x = np.linspace(-1, 1, 2001)
    centers, vals, counts = bin_median(x, x, -1, 1, 11)
    assert centers.size == 11
    assert vals[5] == pytest.approx(0.0, abs=1e-6)
    assert counts.sum() == pytest.approx(x.size, abs=2)
    _, _, wide = bin_median(x, x, -1, 1, 11, bin_width=0.5)
    assert wide.sum() > counts.sum()  # overlapping bins share points


def test_normalize_view() -> None:
    v = normalize_view(np.array([1.0, 1.0, 0.98, 1.0, 1.0]))
    assert np.median(v) == pytest.approx(0)
    assert v.min() == pytest.approx(-1)
    assert np.all(normalize_view(np.ones(5)) == 0)


def test_views_shape_center_and_normalization(
    planet_lc: SyntheticLightCurve, data_cfg: DataConfig
) -> None:
    t, f = _flat(planet_lc, data_cfg)
    views = make_views(t, f, 3.5, 1.2, 0.12, data_cfg.views)
    assert views.global_view.shape == (201,)
    assert views.local_view.shape == (61,)
    assert views.global_view.dtype == np.float32
    for v in (views.global_view, views.local_view):
        assert np.isfinite(v).all()
        assert v.min() == pytest.approx(-1.0)
        assert abs(float(np.median(v))) < 0.05
    # Transit sits in the middle of both views.
    assert abs(int(np.argmin(views.global_view)) - 100) <= 2
    core = views.local_view[25:36]
    assert core.mean() < -0.7
    assert views.local_view[:8].mean() > -0.2 and views.local_view[-8:].mean() > -0.2
    assert views.local_coverage > 0.9


def test_wrong_epoch_raises_view_error(data_cfg: DataConfig) -> None:
    t = np.linspace(0, 10, 2000)
    t = t[np.abs(phase_fold(t, 2.0, 0.0)) > 0.3]  # no data within 0.3 d of any transit
    with pytest.raises(ViewError):
        make_views(t, np.ones_like(t), 2.0, 0.0, 0.1, data_cfg.views)


def test_vetting_planet_passes(planet_lc: SyntheticLightCurve, data_cfg: DataConfig) -> None:
    t, f = _flat(planet_lc, data_cfg)
    ex = build_example(t, f, 3.5, 1.2, 0.12, data_cfg.views, stellar_radius_rsun=1.0)
    flags = {fl.name: fl.passed for fl in ex.vetting.flags}
    assert flags["odd_even"] is True
    assert flags["secondary_eclipse"] is True
    assert flags["transit_shape"] is True
    assert flags["planet_radius"] is True
    assert ex.vetting.depth == pytest.approx(0.002, rel=0.15)
    assert ex.vetting.planet_radius_rjup == pytest.approx(np.sqrt(0.002) * 9.731, rel=0.1)


def test_vetting_eb_fails_secondary_and_shape(
    eb_lc: SyntheticLightCurve, data_cfg: DataConfig
) -> None:
    t, f = _flat(eb_lc, data_cfg)
    ex = build_example(t, f, 2.2, 0.7, 0.15, data_cfg.views)
    flags = {fl.name: fl.passed for fl in ex.vetting.flags}
    assert flags["secondary_eclipse"] is False
    assert flags["transit_shape"] is False
    assert ex.vetting.secondary_depth_ratio == pytest.approx(0.4, abs=0.1)
    assert flags["planet_radius"] is None  # no stellar radius given


def test_vetting_half_period_eb_fails_odd_even(data_cfg: DataConfig) -> None:
    lc = synthetic_eb_lightcurve(period=2.2, secondary_ratio=0.75, seed=4)
    t, f = _flat(lc, data_cfg)
    ex = build_example(t, f, 1.1, 0.7, 0.15, data_cfg.views)
    flags = {fl.name: fl.passed for fl in ex.vetting.flags}
    assert flags["odd_even"] is False
    assert ex.vetting.odd_even_sigma > 10


def test_features_complete_and_bounded(
    planet_lc: SyntheticLightCurve, data_cfg: DataConfig
) -> None:
    t, f = _flat(planet_lc, data_cfg)
    ex = build_example(t, f, 3.5, 1.2, 0.12, data_cfg.views, stellar_radius_rsun=1.0)
    assert set(ex.features) == set(SCALAR_FEATURES)
    vec = ex.feature_vector()
    assert vec.shape == (len(SCALAR_FEATURES),)
    assert np.isfinite(vec).all()
    assert ex.features["log_period"] == pytest.approx(np.log10(3.5))
    d = ex.vetting.to_dict()
    assert isinstance(d["flags"], list) and d["flags"][0]["name"] == "transit_count"

"""Cleaning, detrending and BLS search on synthetic light curves (no network)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from transit_hunter.config import DataConfig
from transit_hunter.data.synthetic import (
    SyntheticLightCurve,
    make_time_grid,
    synthetic_planet_lightcurve,
    synthetic_star,
    trapezoid_transit,
)
from transit_hunter.processing.clean import LightCurveTooShortError, clean_lightcurve
from transit_hunter.processing.detrend import detrend, recommended_window, split_segments
from transit_hunter.processing.search import (
    bin_lightcurve,
    bls_search,
    count_transits,
    iterative_search,
    period_grid,
    transit_mask,
)


def _flatten(lc: SyntheticLightCurve, cfg: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = clean_lightcurve(lc.time, lc.flux, lc.flux_err, cfg.cleaning)
    d = detrend(c.time, c.flux, c.flux_err, cfg.detrend)
    return d.time, d.flux, d.flux_err


# ----------------------------------------------------------------- cleaning


def test_clean_removes_nans_and_flares_but_keeps_transits(
    planet_lc: SyntheticLightCurve, data_cfg: DataConfig
) -> None:
    flux = planet_lc.flux.copy()
    flux[::97] = np.nan
    flares = np.arange(500, flux.size - 10, 503)
    flux[flares] += 0.05
    res = clean_lightcurve(planet_lc.time, flux, planet_lc.flux_err, data_cfg.cleaning)
    assert np.isfinite(res.flux).all()
    assert res.n_nan_removed == np.isnan(flux).sum()
    assert res.n_outliers_removed >= len(flares)
    assert abs(np.median(res.flux) - 1.0) < 1e-4
    # Every in-transit point survives (downward clipping is off).
    in_tr_before = transit_mask(planet_lc.time, 3.5, 1.2, 0.12, 0.4) & np.isfinite(flux)
    in_tr_before[flares] = False
    in_tr_after = transit_mask(res.time, 3.5, 1.2, 0.12, 0.4)
    assert in_tr_after.sum() == in_tr_before.sum()


def test_clean_rejects_short_lightcurve(data_cfg: DataConfig) -> None:
    t = np.linspace(0, 1, 50)
    with pytest.raises(LightCurveTooShortError):
        clean_lightcurve(t, np.ones_like(t), None, data_cfg.cleaning)


def test_clean_fills_missing_errors(planet_lc: SyntheticLightCurve, data_cfg: DataConfig) -> None:
    res = clean_lightcurve(planet_lc.time, planet_lc.flux, None, data_cfg.cleaning)
    assert np.all(res.flux_err > 0)


# ----------------------------------------------------------------- detrending


@pytest.mark.parametrize("method", ["biweight", "savgol"])
def test_detrend_flattens_variability_and_preserves_depth(
    planet_lc: SyntheticLightCurve, data_cfg: DataConfig, method: str
) -> None:
    c = clean_lightcurve(planet_lc.time, planet_lc.flux, planet_lc.flux_err, data_cfg.cleaning)
    d = detrend(c.time, c.flux, c.flux_err, data_cfg.detrend, method=method)
    oot = ~transit_mask(d.time, 3.5, 1.2, 0.12, 1.0)
    in_tr = transit_mask(d.time, 3.5, 1.2, 0.12, 0.3)
    # Raw variability is ~2000 ppm; the flattened baseline should be at the white-noise level.
    assert np.std(d.flux[oot]) < 2 * 300e-6
    depth = np.median(d.flux[oot]) - np.mean(d.flux[in_tr])
    tolerance = 0.15 if method == "biweight" else 0.35
    assert depth == pytest.approx(0.002, rel=tolerance)
    assert d.valid_mask.sum() == d.time.size


def test_recommended_window_scales_with_duration(data_cfg: DataConfig) -> None:
    assert recommended_window(data_cfg.detrend, None) == data_cfg.detrend.window_length_days
    assert recommended_window(data_cfg.detrend, [0.5]) == pytest.approx(1.5)


def test_split_segments() -> None:
    t = np.concatenate([np.arange(0, 10, 0.1), np.arange(20, 30, 0.1)])
    segs = split_segments(t, 0.5)
    assert len(segs) == 2
    assert segs[0].stop == 100


# ----------------------------------------------------------------- search


def test_period_grid_bounds_and_density(data_cfg: DataConfig) -> None:
    g = period_grid(90.0, data_cfg.search)
    assert g[0] == pytest.approx(data_cfg.search.period_min_days)
    assert g[-1] <= 90.0 / data_cfg.search.min_transits + 1e-9
    assert np.all(np.diff(g) > 0)
    # Short baselines (TESS sector) give small grids -> fast searches.
    assert period_grid(27.0, data_cfg.search).size < 20_000
    with pytest.raises(ValueError):
        period_grid(0.8, data_cfg.search)


def test_bin_lightcurve_reduces_points() -> None:
    t = np.arange(0, 10, 2 / 1440)
    f = np.ones_like(t)
    tb, fb, eb = bin_lightcurve(t, f, np.full_like(t, 1e-3), 30.0)
    assert tb.size == pytest.approx(t.size / 15, rel=0.01)
    assert np.allclose(fb, 1.0)
    assert np.all(eb < 1e-3)


@pytest.mark.parametrize("engine", ["numba", "astropy"])
def test_bls_recovers_period_within_1_percent(
    planet_lc: SyntheticLightCurve, data_cfg: DataConfig, engine: str
) -> None:
    t, f, e = _flatten(planet_lc, data_cfg)
    cfg = data_cfg.search.model_copy(update={"engine": engine})
    cand = bls_search(t, f, e, cfg)
    assert cand.period == pytest.approx(3.5, rel=0.01)
    assert cand.depth == pytest.approx(0.002, rel=0.3)
    assert 0.5 * 0.12 < cand.duration < 2.0 * 0.12
    phase_err = abs((cand.t0 - 1.2 + 0.5 * 3.5) % 3.5 - 0.5 * 3.5)
    assert phase_err < 0.05
    assert cand.significant
    assert cand.snr > 20
    assert cand.n_transits >= 20
    assert cand.periodogram is not None and len(cand.periodogram.period) > 100


def test_bls_engines_agree(planet_lc: SyntheticLightCurve, data_cfg: DataConfig) -> None:
    t, f, e = _flatten(planet_lc, data_cfg)
    a = bls_search(t, f, e, data_cfg.search.model_copy(update={"engine": "numba"}))
    b = bls_search(t, f, e, data_cfg.search.model_copy(update={"engine": "astropy"}))
    assert a.period == pytest.approx(b.period, rel=1e-3)
    assert a.depth == pytest.approx(b.depth, rel=0.15)


def test_bls_shallow_long_period_planet(data_cfg: DataConfig) -> None:
    lc = synthetic_planet_lightcurve(
        period=11.3, t0=4.0, depth=0.0006, duration=0.2, baseline_days=120, noise_ppm=250, seed=5
    )
    t, f, e = _flatten(lc, data_cfg)
    cand = bls_search(t, f, e, data_cfg.search)
    assert cand.period == pytest.approx(11.3, rel=0.01)
    assert cand.significant


def test_noise_only_is_not_significant(data_cfg: DataConfig) -> None:
    t = make_time_grid(60.0)
    star, white = synthetic_star(t, noise_ppm=300, variability_ppm=1500, red_noise_ppm=0, seed=11)
    lc = SyntheticLightCurve(t, star + white, np.full_like(t, 3e-4), star, {})
    tt, f, e = _flatten(lc, data_cfg)
    cand = bls_search(tt, f, e, data_cfg.search)
    assert cand.snr < 15
    assert cand.sde < 15


def test_iterative_search_finds_two_planets(data_cfg: DataConfig) -> None:
    t = make_time_grid(120.0)
    star, white = synthetic_star(t, noise_ppm=200, variability_ppm=1000, seed=3)
    model = trapezoid_transit(t, 2.7, 0.9, 0.1, 0.003) * trapezoid_transit(
        t, 8.9, 3.1, 0.17, 0.0015
    )
    lc = SyntheticLightCurve(t, star * model + white, np.full_like(t, 2e-4), star, {})
    tt, f, e = _flatten(lc, data_cfg)
    cands = iterative_search(tt, f, e, data_cfg.search, max_candidates=3)
    periods = sorted(c.period for c in cands if c.significant)
    assert len(periods) >= 2
    assert periods[0] == pytest.approx(2.7, rel=0.01)
    assert any(p == pytest.approx(8.9, rel=0.01) for p in periods)
    assert cands[0].rank == 1


def test_count_transits_with_gaps() -> None:
    t = make_time_grid(30.0, gap_days=5.0, segment_days=15.0)
    n = count_transits(t, 2.0, 0.5, 0.1)
    assert 10 <= n < 15


def test_tess_search_is_fast(data_cfg: DataConfig) -> None:
    lc = synthetic_planet_lightcurve(
        period=3.1, depth=0.003, duration=0.1, baseline_days=27, cadence_minutes=2, noise_ppm=800
    )
    t, f, e = _flatten(lc, data_cfg)
    bls_search(t[:2000], f[:2000], e[:2000], data_cfg.search)  # JIT warm-up
    start = time.perf_counter()
    cand = bls_search(t, f, e, data_cfg.search)
    assert time.perf_counter() - start < 2.0
    assert cand.period == pytest.approx(3.1, rel=0.01)

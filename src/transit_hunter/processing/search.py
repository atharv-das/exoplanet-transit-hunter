"""Transit search with Box Least Squares (Kovács, Zucker & Mazeh 2002).

Two engines:

* ``numba`` (default): a parallel BLS that bins each trial fold into phase bins and slides boxes
  over cumulative sums. Cost per period is O(N + n_bins * n_durations). Measured ~6x faster
  than astropy on 2 cores and it scales with core count, which makes 4-year Kepler searches
  practical.
* ``astropy``: ``astropy.timeseries.BoxLeastSquares``, kept as a reference implementation; the
  tests check that both engines agree.

The period grid follows the usual drift criterion (cf. Ofir 2014): neighboring trial periods differ
by less than ``duration / (baseline * oversample)``, with the duration floor scaled to the
shortest plausible transit at each period.
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from transit_hunter.config import SearchConfig
from transit_hunter.processing.clean import robust_std

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

#: Upper limit on phase bins per trial period (memory/time guard for very long periods).
_MAX_PHASE_BINS = 60_000
#: Largest transit duty cycle (duration / period) considered physical.
_MAX_DUTY_CYCLE = 0.25
#: Number of periodogram points returned for plotting.
_PERIODOGRAM_POINTS = 4000


@dataclass
class Periodogram:
    """Downsampled BLS periodogram for plotting (max-pooled so peaks survive)."""

    period: list[float]
    power: list[float]


@dataclass
class TransitCandidate:
    """One periodic transit-like signal found by the search."""

    period: float  # days
    t0: float  # mid-transit time of the first transit in the data (mission time system)
    duration: float  # days
    depth: float  # fractional
    snr: float  # depth / uncertainty, from the box fit
    sde: float  # signal detection efficiency of the peak in the detrended spectrum
    power: float
    n_transits: int  # transits with at least one in-transit data point
    significant: bool
    rank: int = 1
    search_seconds: float = 0.0
    periodogram: Periodogram | None = field(default=None, repr=False)

    def to_dict(self, include_periodogram: bool = True) -> dict[str, Any]:
        """JSON-serializable representation."""
        d = asdict(self)
        if not include_periodogram:
            d.pop("periodogram", None)
        return d


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def bin_lightcurve(
    time: FloatArray, flux: FloatArray, flux_err: FloatArray, bin_minutes: float | None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Average into fixed-width time bins; no-op if the data are already that coarse."""
    if bin_minutes is None or time.size < 2:
        return time, flux, flux_err
    width = bin_minutes / (24 * 60)
    cadence = float(np.median(np.diff(time)))
    if cadence >= 0.8 * width:
        return time, flux, flux_err
    idx = np.floor((time - time[0]) / width).astype(np.int64)
    uniq, inv, counts = np.unique(idx, return_inverse=True, return_counts=True)
    tb = np.bincount(inv, weights=time) / counts
    fb = np.bincount(inv, weights=flux) / counts
    eb = np.sqrt(np.bincount(inv, weights=flux_err**2)) / counts
    del uniq
    return tb, fb, eb


def plausible_min_duration(period: FloatArray | float, floor: float) -> FloatArray:
    """Shortest plausible transit duration at each period: half the central-transit duration of
    a solar-density star (13 h at 1 yr, scaling as P^(1/3)), never below ``floor``."""
    p = np.asarray(period, dtype=float)
    d_sun = 13.0 / 24.0 * (p / 365.25) ** (1.0 / 3.0)
    return np.maximum(0.5 * d_sun, floor)


def period_grid(baseline: float, cfg: SearchConfig) -> FloatArray:
    """Trial periods (days), log-spaced with density set by the transit-time drift criterion.

    Args:
        baseline: Time span of the data in days.
        cfg: Search settings (period bounds, oversampling, grid cap).

    Returns:
        Increasing array of trial periods.
    """
    p_min = cfg.period_min_days
    p_max = min(cfg.period_max_days, baseline / max(cfg.min_transits, 1))
    if p_max <= p_min:
        raise ValueError(
            f"baseline {baseline:.2f} d too short for min_transits={cfg.min_transits} "
            f"at period_min={p_min} d"
        )
    floor = min(cfg.durations_days)
    lnp = np.linspace(np.log(p_min), np.log(p_max), 20_000)
    density = baseline * cfg.period_oversample / plausible_min_duration(np.exp(lnp), floor)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(lnp))])
    n = int(np.clip(np.ceil(cum[-1]), 100, cfg.max_periods))
    if np.ceil(cum[-1]) > cfg.max_periods:
        logger.info("period grid capped at %d (ideal %d)", cfg.max_periods, int(cum[-1]))
    targets = np.linspace(0.0, cum[-1], n)
    return np.exp(np.interp(targets, cum, lnp))


def transit_mask(
    time: FloatArray, period: float, t0: float, duration: float, half_width_durations: float = 1.0
) -> NDArray[np.bool_]:
    """True for points within ``half_width_durations * duration`` of any transit center."""
    dist = np.abs((time - t0 + 0.5 * period) % period - 0.5 * period)
    return dist < half_width_durations * duration


def count_transits(time: FloatArray, period: float, t0: float, duration: float) -> int:
    """Number of individual transit epochs that contain at least one data point."""
    in_tr = transit_mask(time, period, t0, duration, 0.5)
    if not in_tr.any():
        return 0
    epochs = np.round((time[in_tr] - t0) / period).astype(np.int64)
    return int(np.unique(epochs).size)


def _maxpool(period: FloatArray, power: FloatArray, n: int) -> Periodogram:
    if period.size <= n:
        return Periodogram(period.tolist(), power.tolist())
    edges = np.linspace(0, period.size, n + 1).astype(np.int64)
    pp, pw = [], []
    for a, b in pairwise(edges):
        if b <= a:
            continue
        j = a + int(np.argmax(power[a:b]))
        pp.append(float(period[j]))
        pw.append(float(power[j]))
    return Periodogram(pp, pw)


def _running_median(x: FloatArray, window: int) -> FloatArray:
    from scipy.ndimage import median_filter

    window = max(3, window | 1)
    return np.asarray(median_filter(x, size=window, mode="nearest"), dtype=float)


# --------------------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------------------


@njit(parallel=True, cache=True, fastmath=True)
def _bls_numba(
    t: FloatArray,
    y: FloatArray,
    periods: FloatArray,
    durations: FloatArray,
    min_durations: FloatArray,
    bins_per_duration: float,
    max_bins: int,
    max_duty: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:  # pragma: no cover - compiled
    """Core BLS. ``y`` must be mean-subtracted; weights are uniform (1/N).

    ``min_durations[i]`` is the shortest box tried at ``periods[i]``; the phase-bin width is set
    from it, so long periods use fewer bins.

    Returns per-period (power, depth, duration, t_mid offset from t=0) where
    power = r^2 / (s (1 - s)) and depth = -r / (s (1 - s)).
    """
    n = t.size
    w = 1.0 / n
    n_p = periods.size
    out_pow = np.zeros(n_p)
    out_depth = np.zeros(n_p)
    out_dur = np.zeros(n_p)
    out_t0 = np.zeros(n_p)
    for ip in prange(n_p):
        per = periods[ip]
        dmin = min_durations[ip]
        nb = int(per / (dmin / bins_per_duration))
        nb = max(nb, 8)
        nb = min(nb, max_bins)
        bw = per / nb
        wb = np.zeros(nb)
        yb = np.zeros(nb)
        for i in range(n):
            ph = t[i] / per
            ph = ph - np.floor(ph)
            b = int(ph * nb)
            if b >= nb:
                b = nb - 1
            wb[b] += w
            yb[b] += w * y[i]
        cw = np.zeros(2 * nb + 1)
        cy = np.zeros(2 * nb + 1)
        for j in range(2 * nb):
            cw[j + 1] = cw[j] + wb[j % nb]
            cy[j + 1] = cy[j] + yb[j % nb]
        best = 0.0
        bdep = 0.0
        bdur = 0.0
        bt0 = 0.0
        for idur in range(durations.size):
            d = durations[idur]
            if d > max_duty * per or d < dmin:
                continue
            k = int(d / bw + 0.5)
            if k < 1:
                k = 1
            if k >= nb:
                continue
            for j in range(nb):
                s = cw[j + k] - cw[j]
                if s <= 0.0 or s >= 1.0:
                    continue
                r = cy[j + k] - cy[j]
                if r >= 0.0:
                    continue
                pw = r * r / (s * (1.0 - s))
                if pw > best:
                    best = pw
                    bdep = -r / (s * (1.0 - s))
                    bdur = k * bw
                    bt0 = (j + 0.5 * k) * bw
        out_pow[ip] = best
        out_depth[ip] = bdep
        out_dur[ip] = bdur
        out_t0[ip] = bt0
    return out_pow, out_depth, out_dur, out_t0


def _run_engine(
    t: FloatArray,
    f: FloatArray,
    e: FloatArray,
    periods: FloatArray,
    durations: FloatArray,
    cfg: SearchConfig,
    bins_per_duration: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Run the configured engine. Returns (snr, power, depth, duration, t0) per period."""
    sigma = robust_std(f)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(f)) or 1e-6
    if cfg.engine == "numba":
        tref = t[0]
        y = f - f.mean()
        # Shortest box per period: the largest configured duration not exceeding half the
        # plausible minimum (keeps at least one duration everywhere).
        plaus = 0.5 * plausible_min_duration(periods, float(durations.min()))
        idx = np.searchsorted(durations, plaus, side="right") - 1
        min_d = durations[np.clip(idx, 0, durations.size - 1)]
        pw, dep, dur, t0 = _bls_numba(
            t - tref,
            y,
            periods,
            durations,
            min_d.astype(np.float64),
            float(bins_per_duration),
            _MAX_PHASE_BINS,
            _MAX_DUTY_CYCLE,
        )
        snr = np.sqrt(np.maximum(pw, 0.0) * t.size) / sigma
        return snr, pw, dep, dur, t0 + tref

    from astropy.timeseries import BoxLeastSquares

    usable = durations[durations < periods.min()]
    bls = BoxLeastSquares(t, f, np.full_like(f, sigma))
    res = bls.power(periods, usable, objective=cfg.objective)
    snr = np.asarray(res.depth_snr, dtype=float)
    return (
        snr,
        np.asarray(res.power, dtype=float),
        np.asarray(res.depth, dtype=float),
        np.asarray(res.duration, dtype=float),
        np.asarray(res.transit_time, dtype=float),
    )


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def bls_search(
    time: FloatArray,
    flux: FloatArray,
    flux_err: FloatArray,
    cfg: SearchConfig,
    rank: int = 1,
    keep_periodogram: bool = True,
) -> TransitCandidate:
    """Find the strongest periodic box-shaped dip in a detrended light curve.

    Steps: bin (TESS 2-min -> 30-min), build the period grid, run BLS, pick the highest peak of the
    detrended SNR spectrum that has at least ``min_transits`` observed transits, then refine
    period/duration/t0 on a fine local grid.

    Args:
        time: Times in days.
        flux: Detrended relative flux (baseline ~ 1).
        flux_err: Uncertainties (used only for binning; BLS uses robust scatter).
        cfg: Search settings.
        rank: Index of this candidate in an iterative search (1 = strongest).
        keep_periodogram: Attach a downsampled periodogram for plotting.

    Returns:
        The best :class:`TransitCandidate`; ``significant`` says whether SNR passed the threshold.
    """
    start = _time.perf_counter()
    ok = np.isfinite(time) & np.isfinite(flux)
    t, f, e = bin_lightcurve(time[ok], flux[ok], flux_err[ok], cfg.bin_minutes)
    if t.size < 50:
        raise ValueError("too few points for a transit search")
    baseline = float(t.max() - t.min())
    periods = period_grid(baseline, cfg)
    durations = np.asarray(sorted(cfg.durations_days), dtype=float)

    snr, _, _, dur, t0 = _run_engine(t, f, e, periods, durations, cfg, cfg.phase_bins_per_duration)

    # Remove the slow rise of the noise floor with period before picking peaks.
    floor = _running_median(snr, max(51, periods.size // 100))
    resid = snr - floor
    spread = robust_std(resid)
    sde_spec = resid / spread if spread > 0 else resid

    order = np.argsort(sde_spec)[::-1]
    chosen = int(order[0])
    taken: list[float] = []
    for j in order[:200]:
        pj = periods[j]
        if any(abs(pj - q) / q < 0.01 for q in taken):
            continue
        taken.append(pj)
        if count_transits(t, pj, t0[j], max(dur[j], 1e-3)) >= cfg.min_transits:
            chosen = int(j)
            break
        if len(taken) >= 10:
            break

    # Local refinement: finer periods and durations around the chosen peak.
    p_best = periods[chosen]
    step = abs(periods[min(chosen + 1, periods.size - 1)] - periods[max(chosen - 1, 0)]) / 2 or 1e-4
    fine_p = np.linspace(max(p_best - 4 * step, 1e-3), p_best + 4 * step, 161)
    d_best = max(float(dur[chosen]), durations.min())
    fine_d = np.geomspace(max(0.5 * d_best, 0.01), min(2.0 * d_best, _MAX_DUTY_CYCLE * p_best), 17)
    rs, rp, rdep, rdur, rt0 = _run_engine(t, f, e, fine_p, fine_d, cfg, 8)
    k = int(np.argmax(rs))
    period, duration, depth_v = float(fine_p[k]), float(rdur[k]), float(rdep[k])
    epoch0 = float(rt0[k])
    # Reference epoch: first transit at or after the first data point.
    epoch0 = epoch0 + period * np.ceil((t.min() - epoch0) / period)
    if epoch0 - period >= t.min() - 0.5 * duration:
        epoch0 -= period
    snr_v = float(rs[k])
    n_tr = count_transits(t, period, epoch0, duration)
    sde_v = float(sde_spec[chosen])

    pgram = _maxpool(periods, snr, _PERIODOGRAM_POINTS) if keep_periodogram else None
    cand = TransitCandidate(
        period=period,
        t0=epoch0,
        duration=duration,
        depth=depth_v,
        snr=snr_v,
        sde=sde_v,
        power=float(rp[k]),
        n_transits=n_tr,
        significant=bool(snr_v >= cfg.snr_threshold and n_tr >= cfg.min_transits),
        rank=rank,
        search_seconds=_time.perf_counter() - start,
        periodogram=pgram,
    )
    logger.info(
        "BLS rank %d: P=%.5f d, D=%.2f h, depth=%.0f ppm, SNR=%.1f, SDE=%.1f (%d periods, %.1fs)",
        rank,
        period,
        duration * 24,
        depth_v * 1e6,
        snr_v,
        sde_v,
        periods.size,
        cand.search_seconds,
    )
    return cand


def iterative_search(
    time: FloatArray,
    flux: FloatArray,
    flux_err: FloatArray,
    cfg: SearchConfig,
    max_candidates: int | None = None,
) -> list[TransitCandidate]:
    """Repeated BLS: after each significant detection, mask its transits and search again.

    Always returns at least one candidate (possibly non-significant) so callers can report
    "no significant signal" with the best attempt's numbers.
    """
    n_max = max_candidates or cfg.max_candidates
    t, f, e = time.copy(), flux.copy(), flux_err.copy()
    found: list[TransitCandidate] = []
    for rank in range(1, n_max + 1):
        try:
            cand = bls_search(t, f, e, cfg, rank=rank)
        except ValueError as exc:
            logger.info("stopping iterative search: %s", exc)
            break
        # A repeat of an earlier period (or its 2x / 0.5x alias) is a leftover of that signal,
        # not a new planet: mask it and keep searching, but don't report it.
        repeat = any(
            abs(cand.period * m / prev.period - 1) < 0.01 for prev in found for m in (1, 2, 0.5)
        )
        if not repeat:
            found.append(cand)
        if not cand.significant:
            break
        keep = ~transit_mask(t, cand.period, cand.t0, cand.duration, 1.0)
        t, f, e = t[keep], f[keep], e[keep]
    return found

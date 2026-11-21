"""Detrending: remove stellar variability and instrumental drifts while preserving transits.

Default is ``wotan``'s time-windowed biweight (Hippke et al. 2019), which is robust to the transit
itself as long as the window is about 3x longer than the transit. Savitzky-Golay is available as
a faster, less transit-safe alternative.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from transit_hunter.config import DetrendConfig

FloatArray = NDArray[np.float64]

#: Window must be at least this many times the longest expected transit duration.
WINDOW_TO_DURATION = 3.0


@dataclass(frozen=True)
class DetrendResult:
    """Flattened light curve (relative flux ~ 1) and the trend that was divided out.

    ``time``, ``flux``, ``flux_err`` and ``trend`` contain only points where the trend is defined;
    ``valid_mask`` maps them back onto the input arrays.
    """

    time: FloatArray
    flux: FloatArray
    flux_err: FloatArray
    trend: FloatArray
    valid_mask: NDArray[np.bool_]
    method: str
    window_length_days: float


def recommended_window(cfg: DetrendConfig, durations_days: Sequence[float] | None = None) -> float:
    """Detrending window: the configured value, widened to 3x the longest known transit."""
    w = cfg.window_length_days
    if durations_days:
        finite = [d for d in durations_days if np.isfinite(d) and d > 0]
        if finite:
            w = max(w, WINDOW_TO_DURATION * max(finite))
    return float(w)


def split_segments(time: FloatArray, gap_days: float) -> list[slice]:
    """Contiguous index ranges separated by gaps longer than ``gap_days``."""
    if time.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(time) > gap_days) + 1
    edges = np.concatenate([[0], breaks, [time.size]])
    return [slice(int(a), int(b)) for a, b in pairwise(edges)]


def _savgol_trend(time: FloatArray, flux: FloatArray, cfg: DetrendConfig) -> FloatArray:
    trend = np.full_like(flux, np.nan)
    for seg in split_segments(time, cfg.break_tolerance_days):
        y = flux[seg]
        n = y.size
        if n < cfg.savgol_polyorder + 2:
            trend[seg] = np.median(y) if n else np.nan
            continue
        win = min(cfg.savgol_window_cadences, n if n % 2 == 1 else n - 1)
        if win <= cfg.savgol_polyorder:
            trend[seg] = np.median(y)
            continue
        # Two passes: the second ignores points far below the first trend (transits).
        first = savgol_filter(y, win, cfg.savgol_polyorder, mode="interp")
        resid = y - first
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-12
        y2 = np.where(resid < -3 * mad, first, y)
        trend[seg] = savgol_filter(y2, win, cfg.savgol_polyorder, mode="interp")
    return trend


def detrend(
    time: FloatArray,
    flux: FloatArray,
    flux_err: FloatArray,
    cfg: DetrendConfig,
    window_length_days: float | None = None,
    method: str | None = None,
) -> DetrendResult:
    """Divide out a smooth trend so transits sit on a flat baseline of 1.

    Args:
        time: Times in days (sorted).
        flux: Normalized flux.
        flux_err: Flux uncertainties.
        cfg: Detrending settings.
        window_length_days: Override for the biweight window (see :func:`recommended_window`).
        method: Override for ``cfg.method`` (``"biweight"`` or ``"savgol"``).

    Returns:
        A :class:`DetrendResult` with points where the trend is undefined (segment edges,
        sparse data) removed.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    e = np.asarray(flux_err, dtype=float)
    m = method or cfg.method
    window = float(window_length_days or cfg.window_length_days)

    if m == "biweight":
        from wotan import flatten

        _, trend = flatten(
            t,
            f,
            method="biweight",
            window_length=window,
            edge_cutoff=cfg.edge_cutoff_days,
            break_tolerance=cfg.break_tolerance_days,
            return_trend=True,
        )
        trend = np.asarray(trend, dtype=float)
    elif m == "savgol":
        trend = _savgol_trend(t, f, cfg)
    else:
        raise ValueError(f"unknown detrend method {m!r}")

    ok = np.isfinite(trend) & (trend > 0)
    flat = f[ok] / trend[ok]
    return DetrendResult(
        time=t[ok],
        flux=flat,
        flux_err=e[ok] / trend[ok],
        trend=trend[ok],
        valid_mask=ok,
        method=m,
        window_length_days=window,
    )

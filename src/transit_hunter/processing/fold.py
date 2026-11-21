"""Phase folding and AstroNet-style global/local views (Shallue & Vanderburg 2018).

* Global view: the full orbit (phase -P/2 .. P/2) median-binned into ``global_bins`` bins.
  Shows secondary eclipses, out-of-transit variability, and phase-dependent systematics.
* Local view: +/- ``local_half_width_durations`` transit durations around the transit,
  median-binned into ``local_bins`` overlapping bins. Shows the transit shape (U vs V).

Each view is normalized so its median is 0 and its minimum is -1. Absolute depth is therefore
carried by the scalar features, not the views.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from transit_hunter.config import ViewsConfig

FloatArray = NDArray[np.float64]


class ViewError(ValueError):
    """Raised when a view cannot be built (e.g. almost no data near the transit)."""


@dataclass(frozen=True)
class FoldedViews:
    """Normalized views plus their phase axes (in days) for plotting."""

    global_view: NDArray[np.float32]
    local_view: NDArray[np.float32]
    global_phase: FloatArray
    local_phase: FloatArray
    global_coverage: float  # fraction of non-empty bins before interpolation
    local_coverage: float


def phase_fold(time: FloatArray, period: float, t0: float) -> FloatArray:
    """Time from the nearest transit center in days, in [-P/2, P/2)."""
    return (np.asarray(time, dtype=float) - t0 + 0.5 * period) % period - 0.5 * period


def bin_median(
    x: FloatArray,
    y: FloatArray,
    xmin: float,
    xmax: float,
    n_bins: int,
    bin_width: float | None = None,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    """Median of ``y`` in ``n_bins`` evenly spaced (possibly overlapping) bins over [xmin, xmax].

    Args:
        x: Sample positions (need not be sorted).
        y: Sample values.
        xmin: Left edge of the first bin.
        xmax: Right edge of the last bin.
        n_bins: Number of bins.
        bin_width: Bin width; defaults to ``(xmax - xmin) / n_bins`` (non-overlapping).

    Returns:
        ``(centers, medians, counts)``; empty bins have median NaN and count 0.
    """
    if n_bins < 1 or xmax <= xmin:
        raise ValueError("need n_bins >= 1 and xmax > xmin")
    width = bin_width if bin_width is not None else (xmax - xmin) / n_bins
    spacing = (xmax - xmin - width) / (n_bins - 1) if n_bins > 1 else 0.0
    centers = xmin + 0.5 * width + spacing * np.arange(n_bins)
    order = np.argsort(x)
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    lo = np.searchsorted(xs, centers - 0.5 * width, side="left")
    hi = np.searchsorted(xs, centers + 0.5 * width, side="left")
    out = np.full(n_bins, np.nan)
    counts = (hi - lo).astype(np.int64)
    for i in np.flatnonzero(counts):
        out[i] = np.median(ys[lo[i] : hi[i]])
    return centers.astype(float), out, counts


def _fill_empty(values: FloatArray) -> FloatArray:
    """Linearly interpolate NaN bins from their neighbors (edges are held constant)."""
    good = np.isfinite(values)
    if good.sum() < 2:
        raise ViewError("fewer than 2 populated bins")
    idx = np.arange(values.size)
    return np.interp(idx, idx[good], values[good])


def normalize_view(view: FloatArray) -> FloatArray:
    """Shift to median 0 and scale so the minimum is -1 (AstroNet convention)."""
    med = np.median(view)
    v = view - med
    scale = abs(float(v.min()))
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(v)
    return v / scale


def global_view(
    time: FloatArray, flux: FloatArray, period: float, t0: float, n_bins: int
) -> tuple[FloatArray, FloatArray, float]:
    """Full-orbit view. Returns ``(phase_centers, normalized_view, coverage)``."""
    ph = phase_fold(time, period, t0)
    centers, vals, counts = bin_median(ph, flux, -0.5 * period, 0.5 * period, n_bins)
    coverage = float((counts > 0).mean())
    return centers, normalize_view(_fill_empty(vals)), coverage


def local_view(
    time: FloatArray,
    flux: FloatArray,
    period: float,
    t0: float,
    duration: float,
    n_bins: int,
    half_width_durations: float,
    bin_width_durations: float,
) -> tuple[FloatArray, FloatArray, float]:
    """Zoomed transit view. Returns ``(phase_centers, normalized_view, coverage)``."""
    ph = phase_fold(time, period, t0)
    half = min(half_width_durations * duration, 0.5 * period)
    width = max(bin_width_durations * duration, 2 * half / n_bins)
    centers, vals, counts = bin_median(ph, flux, -half, half, n_bins, bin_width=width)
    coverage = float((counts > 0).mean())
    center = slice(n_bins // 2 - n_bins // 8, n_bins // 2 + n_bins // 8 + 1)
    if not np.isfinite(vals[center]).any():
        raise ViewError("no data inside the transit window")
    return centers, normalize_view(_fill_empty(vals)), coverage


def make_views(
    time: FloatArray,
    flux: FloatArray,
    period: float,
    t0: float,
    duration: float,
    cfg: ViewsConfig,
) -> FoldedViews:
    """Build both views for one signal.

    Args:
        time: Times in days of the detrended light curve.
        flux: Detrended relative flux.
        period: Signal period in days.
        t0: Mid-transit epoch in days.
        duration: Transit duration in days.
        cfg: View settings (bin counts and local-window width).

    Returns:
        :class:`FoldedViews` with float32 views ready for the CNN.

    Raises:
        ViewError: if the transit window has no data.
    """
    if not (period > 0 and duration > 0):
        raise ViewError("period and duration must be positive")
    gph, gv, gcov = global_view(time, flux, period, t0, cfg.global_bins)
    lph, lv, lcov = local_view(
        time,
        flux,
        period,
        t0,
        duration,
        cfg.local_bins,
        cfg.local_half_width_durations,
        cfg.local_bin_width_durations,
    )
    return FoldedViews(
        global_view=gv.astype(np.float32),
        local_view=lv.astype(np.float32),
        global_phase=gph,
        local_phase=lph,
        global_coverage=gcov,
        local_coverage=lcov,
    )

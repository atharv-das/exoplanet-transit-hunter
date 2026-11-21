"""Light-curve cleaning: NaN removal, sorting, de-duplication, normalization, outlier clipping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter

from transit_hunter.config import CleaningConfig

FloatArray = NDArray[np.float64]

#: Median-filter length (cadences) for the local baseline used by outlier clipping.
_OUTLIER_FILTER_SIZE = 25


class LightCurveTooShortError(ValueError):
    """Raised when too few valid points remain after cleaning."""


@dataclass(frozen=True)
class CleanResult:
    """Cleaned light curve plus bookkeeping for the UI and the data card."""

    time: FloatArray
    flux: FloatArray
    flux_err: FloatArray
    n_input: int
    n_nan_removed: int
    n_outliers_removed: int


def robust_std(x: FloatArray) -> float:
    """Gaussian-equivalent standard deviation from the median absolute deviation."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def clean_lightcurve(
    time: FloatArray,
    flux: FloatArray,
    flux_err: FloatArray | None,
    cfg: CleaningConfig,
) -> CleanResult:
    """Remove bad points and normalize a light curve to median 1.

    Steps: drop non-finite points, sort by time, drop duplicate timestamps, normalize by the
    median, then sigma-clip *upward* outliers relative to a running median (flares, cosmic
    rays). Downward clipping is off by default because transits are downward outliers.

    Args:
        time: Times in days.
        flux: Flux (any units).
        flux_err: Flux uncertainties, or ``None`` to estimate them from the scatter.
        cfg: Cleaning settings.

    Returns:
        A :class:`CleanResult` whose flux is normalized to a median of 1.

    Raises:
        LightCurveTooShortError: if fewer than ``cfg.min_points`` points survive.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    e = np.full_like(f, np.nan) if flux_err is None else np.asarray(flux_err, dtype=float)
    n_input = t.size

    finite = np.isfinite(t) & np.isfinite(f)
    t, f, e = t[finite], f[finite], e[finite]
    n_nan = n_input - t.size

    order = np.argsort(t, kind="stable")
    t, f, e = t[order], f[order], e[order]
    if t.size:
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, f, e = t[keep], f[keep], e[keep]

    if t.size < max(cfg.min_points, _OUTLIER_FILTER_SIZE):
        raise LightCurveTooShortError(f"only {t.size} valid points (need {cfg.min_points})")

    med = np.median(f)
    if not np.isfinite(med) or med == 0:
        raise LightCurveTooShortError("median flux is zero or undefined")
    f = f / med
    e = e / abs(med)

    baseline = median_filter(f, size=_OUTLIER_FILTER_SIZE, mode="nearest")
    resid = f - baseline
    sigma = robust_std(resid)
    bad = np.zeros(f.size, dtype=bool)
    if np.isfinite(sigma) and sigma > 0:
        bad |= resid > cfg.sigma_upper * sigma
        if cfg.sigma_lower is not None:
            bad |= resid < -cfg.sigma_lower * sigma
    t, f, e = t[~bad], f[~bad], e[~bad]

    if t.size < cfg.min_points:
        raise LightCurveTooShortError(f"only {t.size} points after clipping")

    # Fill missing or non-positive uncertainties with the point-to-point scatter.
    bad_err = ~np.isfinite(e) | (e <= 0)
    if bad_err.any():
        p2p = robust_std(np.diff(f)) / np.sqrt(2)
        fill = p2p if np.isfinite(p2p) and p2p > 0 else robust_std(f)
        e = np.where(bad_err, fill, e)

    return CleanResult(t, f, e, n_input, int(n_nan), int(bad.sum()))

"""Scalar features and the single "signal -> model inputs" function used by training AND inference.

Keeping :func:`build_example` as the only path from (light curve, ephemeris) to model inputs is
what prevents train/serve skew: the dataset builder and the live pipeline both call it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from transit_hunter.config import ViewsConfig
from transit_hunter.processing.fold import FoldedViews, make_views
from transit_hunter.vetting.diagnostics import VettingResult, compute_vetting

FloatArray = NDArray[np.float64]

#: Every scalar feature, in a fixed order. The CNN uses the subset named in model_cnn.yaml;
#: the GBM uses all of them (NaN allowed, LightGBM handles missing values).
SCALAR_FEATURES: tuple[str, ...] = (
    "log_period",
    "log_duration",
    "log_depth",
    "snr",
    "odd_even_ratio",
    "odd_even_sigma",
    "secondary_depth_ratio",
    "secondary_snr",
    "secondary_max_snr",
    "shape_ratio",
    "duration_ratio",
    "log_n_transits",
    "log_rp_rjup",
    "is_tess",
)


@dataclass(frozen=True)
class Example:
    """Model inputs for one signal plus the diagnostics they were derived from."""

    views: FoldedViews
    features: dict[str, float]
    vetting: VettingResult

    def feature_vector(self, names: tuple[str, ...] | list[str] = SCALAR_FEATURES) -> FloatArray:
        """Features as an array in the requested order (missing -> NaN)."""
        return np.array([self.features.get(n, np.nan) for n in names], dtype=float)


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi)) if np.isfinite(x) else float("nan")


def compute_features(
    period: float, duration: float, vetting: VettingResult, mission: str = "Kepler"
) -> dict[str, float]:
    """Scalar features from the ephemeris and vetting statistics.

    Heavy-tailed quantities are log-scaled and clipped so a single extreme EB cannot dominate
    feature standardization.
    """
    depth_ppm = max(vetting.depth * 1e6, 1.0) if np.isfinite(vetting.depth) else float("nan")
    rp = vetting.planet_radius_rjup
    return {
        "log_period": float(np.log10(period)),
        "log_duration": float(np.log10(max(duration, 1e-4))),
        "log_depth": float(np.log10(depth_ppm)) if np.isfinite(depth_ppm) else float("nan"),
        "snr": _clip(vetting.snr, -10, 500),
        "odd_even_ratio": _clip(vetting.odd_even_ratio, -5, 1),
        "odd_even_sigma": _clip(vetting.odd_even_sigma, 0, 50),
        "secondary_depth_ratio": _clip(vetting.secondary_depth_ratio, -2, 2),
        "secondary_snr": _clip(vetting.secondary_snr, -20, 200),
        "secondary_max_snr": _clip(vetting.secondary_max_snr, -20, 200),
        "shape_ratio": _clip(vetting.shape_ratio, -1, 5),
        "duration_ratio": _clip(vetting.duration_ratio, 0, 20),
        "log_n_transits": float(np.log10(max(vetting.n_transits, 1))),
        "log_rp_rjup": float(np.log10(rp)) if rp is not None and rp > 0 else float("nan"),
        # Mission flag: lets one model learn Kepler- and TESS-specific noise and cadence.
        "is_tess": 1.0 if mission == "TESS" else 0.0,
    }


def build_example(
    time: FloatArray,
    flux: FloatArray,
    period: float,
    t0: float,
    duration: float,
    views_cfg: ViewsConfig,
    stellar_radius_rsun: float | None = None,
    mission: str = "Kepler",
) -> Example:
    """Turn a detrended light curve plus an ephemeris into model inputs.

    Args:
        time: Times in days of the detrended light curve.
        flux: Detrended relative flux.
        period: Period in days.
        t0: Mid-transit epoch in days (same time system as ``time``).
        duration: Duration in days.
        views_cfg: View binning settings.
        stellar_radius_rsun: Optional host radius for the radius feature.
        mission: "Kepler" or "TESS" (sets the ``is_tess`` feature).

    Returns:
        An :class:`Example` with views, scalar features and vetting results.

    Raises:
        transit_hunter.processing.fold.ViewError: if the views cannot be built.
    """
    views = make_views(time, flux, period, t0, duration, views_cfg)
    vetting = compute_vetting(time, flux, period, t0, duration, stellar_radius_rsun)
    return Example(views, compute_features(period, duration, vetting, mission), vetting)

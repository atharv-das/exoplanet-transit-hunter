"""Vetting diagnostics that separate planets from eclipsing binaries and systematics.

All tests work on a detrended light curve plus an ephemeris (period, t0, duration), so they apply
equally to catalog ephemerides (training) and BLS ephemerides (inference).

Tests:
    * Odd/even depth: an EB found at half its true period shows alternating depths.
    * Secondary eclipse: a dip at phase 0.5 (or anywhere off-transit for eccentric orbits)
      deeper than a planet's thermal emission could produce.
    * Shape: V-shaped dips (grazing EBs) vs U-shaped dips (planets) via inner/full depth ratio.
    * Duration vs period: compared with a central transit across a solar-density star.
    * Radius: Rp > 2 R_Jup is too large for a planet, so the companion is likely stellar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from transit_hunter.processing.clean import robust_std
from transit_hunter.processing.fold import phase_fold

FloatArray = NDArray[np.float64]

RSUN_PER_RJUP = 9.731

# Thresholds used for the green/red badges. Raw values are always kept as model features.
ODD_EVEN_SIGMA_MAX = 3.0
ODD_EVEN_REL_DIFF_MIN = 0.1  # also require >= 10 % relative difference (deep EBs have tiny sigmas)
SECONDARY_SNR_MAX = 3.0
SECONDARY_RATIO_MIN = (
    0.05  # secondaries shallower than 5 % of primary are plausible planet emission
)
SHAPE_V_THRESHOLD = 1.35
DURATION_RATIO_RANGE = (0.15, 3.0)
MAX_PLANET_RADIUS_RJUP = 2.0
MIN_TRANSITS = 2


@dataclass
class VettingFlag:
    """One badge in the UI. ``passed`` is None when the test could not be computed."""

    name: str
    passed: bool | None
    value: float | None
    message: str


@dataclass
class VettingResult:
    """Raw diagnostic numbers plus pass/fail flags."""

    depth: float
    depth_err: float
    snr: float
    n_transits: int
    depth_odd: float
    depth_even: float
    odd_even_sigma: float
    odd_even_ratio: float
    secondary_depth: float
    secondary_snr: float
    secondary_depth_ratio: float
    secondary_max_snr: float
    secondary_max_phase: float
    shape_ratio: float
    duration_ratio: float
    planet_radius_rjup: float | None
    flags: list[VettingFlag] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation (NaN -> None)."""

        def clean(v: Any) -> Any:
            if isinstance(v, float) and not np.isfinite(v):
                return None
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, list):
                return [clean(x) for x in v]
            return v

        result: dict[str, Any] = clean(asdict(self))
        return result


def expected_duration_solar(period: float) -> float:
    """Central-transit duration (days) of a small planet around a Sun-like star."""
    return 13.0 / 24.0 * (period / 365.25) ** (1.0 / 3.0)


def _box_depth(
    flux: FloatArray, in_mask: NDArray[np.bool_], baseline: float, noise: float
) -> tuple[float, float, int]:
    n = int(in_mask.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    depth = baseline - float(np.mean(flux[in_mask]))
    return depth, noise / np.sqrt(n), n


def compute_vetting(
    time: FloatArray,
    flux: FloatArray,
    period: float,
    t0: float,
    duration: float,
    stellar_radius_rsun: float | None = None,
) -> VettingResult:
    """Run every vetting test for one signal.

    Args:
        time: Times in days (detrended light curve).
        flux: Detrended relative flux (baseline ~ 1).
        period: Period in days.
        t0: Mid-transit epoch in days.
        duration: Transit duration in days.
        stellar_radius_rsun: Host radius in solar radii (TIC/KIC), enabling the radius test.

    Returns:
        A :class:`VettingResult`. Statistics that cannot be computed are NaN and their flag has
        ``passed=None``.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    ph = phase_fold(t, period, t0)
    aph = np.abs(ph)
    half = 0.5 * duration
    core = 0.4 * duration  # avoid ingress/egress for depth estimates

    sec_ph = phase_fold(t, period, t0 + 0.5 * period)
    oot = (aph > 1.5 * duration) & (np.abs(sec_ph) > 1.5 * duration)
    if oot.sum() < 10:
        oot = aph > 1.5 * duration
    baseline = float(np.median(f[oot])) if oot.any() else float(np.median(f))
    noise = robust_std(f[oot]) if oot.sum() > 5 else robust_std(f)

    in_core = aph < core
    depth, depth_err, _ = _box_depth(f, in_core, baseline, noise)
    snr = depth / depth_err if depth_err and np.isfinite(depth_err) and depth_err > 0 else np.nan

    # Individual transit epochs with data.
    epochs = np.round((t - t0) / period).astype(np.int64)
    in_full = aph < half
    n_transits = int(np.unique(epochs[in_full]).size) if in_full.any() else 0

    # Odd / even.
    odd = in_core & (epochs % 2 == 1)
    even = in_core & (epochs % 2 == 0)
    d_odd, e_odd, _ = _box_depth(f, odd, baseline, noise)
    d_even, e_even, _ = _box_depth(f, even, baseline, noise)
    if np.isfinite(d_odd) and np.isfinite(d_even):
        oe_sigma = abs(d_odd - d_even) / np.sqrt(e_odd**2 + e_even**2)
        denom = max(abs(d_odd), abs(d_even), 1e-12)
        oe_ratio = float(np.clip(min(d_odd, d_even) / denom, -5.0, 1.0))
    else:
        oe_sigma, oe_ratio = float("nan"), float("nan")

    # Secondary at phase 0.5.
    sec_in = np.abs(sec_ph) < core
    d_sec, e_sec, _ = _box_depth(f, sec_in, baseline, noise)
    sec_snr = d_sec / e_sec if np.isfinite(d_sec) and e_sec > 0 else np.nan
    sec_ratio = d_sec / depth if np.isfinite(d_sec) and depth > 0 else np.nan

    # Eccentric secondary: scan off-transit phases for the deepest box of the same duration.
    best_snr, best_phase = np.nan, np.nan
    phase_frac = (ph / period) % 1.0
    off_primary = aph > 2 * duration
    if off_primary.sum() > 20 and duration < 0.2 * period:
        grid = np.arange(0.15, 0.85, max(duration / period / 2, 0.005))
        half_frac = 0.4 * duration / period
        best = -np.inf
        for g in grid:
            m = off_primary & (np.abs(phase_frac - g) < half_frac)
            n = int(m.sum())
            if n < 3:
                continue
            s = (baseline - float(np.mean(f[m]))) / (noise / np.sqrt(n))
            if s > best:
                best, best_phase = s, float(g)
        best_snr = best if np.isfinite(best) else np.nan

    # Shape: inner-third depth over full-window depth (box ~1.0, triangle ~1.67).
    inner = aph < duration / 6.0
    d_inner, _, n_inner = _box_depth(f, inner, baseline, noise)
    d_full, _, n_full = _box_depth(f, in_full, baseline, noise)
    shape = d_inner / d_full if n_inner and n_full and d_full > 0 else np.nan

    duration_ratio = duration / expected_duration_solar(period)
    rp_rjup = None
    if stellar_radius_rsun and np.isfinite(stellar_radius_rsun) and depth > 0:
        rp_rjup = float(np.sqrt(depth) * stellar_radius_rsun * RSUN_PER_RJUP)

    flags = _make_flags(
        n_transits,
        oe_sigma,
        oe_ratio,
        float(sec_snr),
        float(sec_ratio),
        float(best_snr),
        float(shape),
        duration_ratio,
        rp_rjup,
    )
    return VettingResult(
        depth=float(depth),
        depth_err=float(depth_err),
        snr=float(snr),
        n_transits=n_transits,
        depth_odd=float(d_odd),
        depth_even=float(d_even),
        odd_even_sigma=float(oe_sigma),
        odd_even_ratio=float(oe_ratio),
        secondary_depth=float(d_sec),
        secondary_snr=float(sec_snr),
        secondary_depth_ratio=float(sec_ratio),
        secondary_max_snr=float(best_snr),
        secondary_max_phase=float(best_phase),
        shape_ratio=float(shape),
        duration_ratio=float(duration_ratio),
        planet_radius_rjup=rp_rjup,
        flags=flags,
    )


def _ok(x: float) -> bool:
    return bool(np.isfinite(x))


def _make_flags(
    n_transits: int,
    oe_sigma: float,
    oe_ratio: float,
    sec_snr: float,
    sec_ratio: float,
    sec_max_snr: float,
    shape: float,
    duration_ratio: float,
    rp_rjup: float | None,
) -> list[VettingFlag]:
    oe_bad = (
        _ok(oe_sigma)
        and oe_sigma >= ODD_EVEN_SIGMA_MAX
        and _ok(oe_ratio)
        and (1.0 - oe_ratio) >= ODD_EVEN_REL_DIFF_MIN
    )
    flags = [
        VettingFlag(
            "transit_count",
            bool(n_transits >= MIN_TRANSITS),
            float(n_transits),
            f"{n_transits} transits observed (need >= {MIN_TRANSITS})",
        ),
        VettingFlag(
            "odd_even",
            None if not _ok(oe_sigma) else not oe_bad,
            oe_sigma if _ok(oe_sigma) else None,
            "odd and even transits have consistent depths"
            if _ok(oe_sigma) and not oe_bad
            else (
                f"odd/even depths differ at {oe_sigma:.1f} sigma: possible EB at twice the period"
                if _ok(oe_sigma)
                else "not enough odd/even transits"
            ),
        ),
    ]
    sec_bad = _ok(sec_snr) and sec_snr > SECONDARY_SNR_MAX and _ok(sec_ratio)
    sec_bad = sec_bad and sec_ratio > SECONDARY_RATIO_MIN
    flags.append(
        VettingFlag(
            "secondary_eclipse",
            None if not _ok(sec_snr) else bool(not sec_bad),
            sec_snr if _ok(sec_snr) else None,
            f"secondary eclipse at phase 0.5 ({sec_snr:.1f} sigma, {sec_ratio:.0%} of primary)"
            if sec_bad
            else "no significant secondary eclipse at phase 0.5",
        )
    )
    ecc_bad = _ok(sec_max_snr) and sec_max_snr > 2 * SECONDARY_SNR_MAX + 1
    flags.append(
        VettingFlag(
            "off_phase_dip",
            None if not _ok(sec_max_snr) else bool(not ecc_bad),
            sec_max_snr if _ok(sec_max_snr) else None,
            f"another dip at a different phase ({sec_max_snr:.1f} sigma)"
            if ecc_bad
            else "no other significant dip in the orbit",
        )
    )
    flags.append(
        VettingFlag(
            "transit_shape",
            None if not _ok(shape) else bool(shape < SHAPE_V_THRESHOLD),
            shape if _ok(shape) else None,
            "U-shaped (flat-bottomed) like a planet"
            if _ok(shape) and shape < SHAPE_V_THRESHOLD
            else ("V-shaped: typical of a grazing eclipsing binary" if _ok(shape) else "unknown"),
        )
    )
    lo, hi = DURATION_RATIO_RANGE
    flags.append(
        VettingFlag(
            "duration_vs_period",
            bool(lo <= duration_ratio <= hi),
            float(duration_ratio),
            f"duration is {duration_ratio:.2f}x a central transit of a Sun-like star"
            + ("" if lo <= duration_ratio <= hi else ": unusual for a planet around a dwarf star"),
        )
    )
    if rp_rjup is None:
        flags.append(VettingFlag("planet_radius", None, None, "stellar radius unknown"))
    else:
        flags.append(
            VettingFlag(
                "planet_radius",
                bool(rp_rjup <= MAX_PLANET_RADIUS_RJUP),
                float(rp_rjup),
                f"implied radius {rp_rjup:.2f} R_Jup"
                + ("" if rp_rjup <= MAX_PLANET_RADIUS_RJUP else ": too large, likely a star"),
            )
        )
    return flags

"""Synthetic light curves and signal injection.

Used for three things:

1. Offline unit tests (a known transit must be recovered by BLS).
2. The hackathon fallback: augmenting a small real dataset with injected planets and binaries.
3. Injection-recovery tests (stretch goal).

Transit shapes use ``batman`` (Mandel & Agol 2002, quadratic limb darkening) when it is installed
(``make setup-full``), and otherwise a trapezoid model, which has no compiled dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: Solar radius / Jupiter radius.
RSUN_PER_RJUP: float = 9.731
#: a/R* for a solar-density star at P = 1 yr (AU / R_sun).
A_RSTAR_1YR: float = 215.03


def _phase_distance(time: FloatArray, period: float, t0: float) -> FloatArray:
    """Absolute time from the nearest transit center, in days."""
    return np.abs((time - t0 + 0.5 * period) % period - 0.5 * period)


def trapezoid_transit(
    time: FloatArray,
    period: float,
    t0: float,
    duration: float,
    depth: float,
    ingress_fraction: float = 0.15,
) -> FloatArray:
    """Relative flux (1 = no dip) of a periodic trapezoidal transit.

    Args:
        time: Times in days.
        period: Orbital period in days.
        t0: Mid-transit time of one transit, in days.
        duration: Total transit duration (first to fourth contact), in days.
        depth: Fractional depth at the flat bottom (e.g. 0.01 for 1 %).
        ingress_fraction: Ingress duration as a fraction of the total duration. 0.5 gives a
            V shape (typical of grazing eclipsing binaries); ~0.1 gives a U shape (small planet).

    Returns:
        Relative flux array with the same shape as ``time``.
    """
    if not 0 < ingress_fraction <= 0.5:
        raise ValueError("ingress_fraction must be in (0, 0.5]")
    x = _phase_distance(np.asarray(time, dtype=float), period, t0)
    half = 0.5 * duration
    ingress = ingress_fraction * duration
    flat_half = half - ingress
    dip = np.zeros_like(x)
    bottom = x <= flat_half
    ramp = (x > flat_half) & (x < half)
    dip[bottom] = depth
    dip[ramp] = depth * (half - x[ramp]) / ingress
    return 1.0 - dip


def solar_density_a_rstar(period: float, stellar_density_solar: float = 1.0) -> float:
    """Scaled semi-major axis a/R* from Kepler's third law for a star of given mean density."""
    return A_RSTAR_1YR * (stellar_density_solar ** (1.0 / 3.0)) * (period / 365.25) ** (2.0 / 3.0)


def batman_available() -> bool:
    """True if the optional ``batman-package`` is importable."""
    try:
        import batman  # noqa: F401
    except ImportError:
        return False
    return True


def physical_transit(
    time: FloatArray,
    period: float,
    t0: float,
    rp_rs: float,
    impact: float = 0.3,
    stellar_density_solar: float = 1.0,
    limb_darkening: tuple[float, float] = (0.4, 0.25),
) -> FloatArray:
    """Limb-darkened transit light curve (batman if available, else matched trapezoid).

    Args:
        time: Times in days.
        period: Orbital period in days.
        t0: Mid-transit time in days.
        rp_rs: Planet-to-star radius ratio.
        impact: Impact parameter b (0 = central transit).
        stellar_density_solar: Mean stellar density in solar units (sets a/R*).
        limb_darkening: Quadratic limb-darkening coefficients (u1, u2).

    Returns:
        Relative flux array.
    """
    a_rs = solar_density_a_rstar(period, stellar_density_solar)
    impact = float(np.clip(impact, 0.0, a_rs * 0.999))
    inc = float(np.degrees(np.arccos(impact / a_rs)))
    if batman_available():
        import batman

        params = batman.TransitParams()
        params.t0 = t0
        params.per = period
        params.rp = rp_rs
        params.a = a_rs
        params.inc = inc
        params.ecc = 0.0
        params.w = 90.0
        params.limb_dark = "quadratic"
        params.u = list(limb_darkening)
        model = batman.TransitModel(params, np.asarray(time, dtype=float))
        return np.asarray(model.light_curve(params), dtype=float)
    duration = transit_duration(period, rp_rs, impact, a_rs)
    if duration <= 0:
        return np.ones_like(np.asarray(time, dtype=float))
    # Trapezoid fallback: ingress fraction grows with impact parameter (grazing -> V-shaped).
    t_full = transit_duration(period, -rp_rs, impact, a_rs)
    ingress_fraction = (
        0.5 if t_full <= 0 else float(np.clip((duration - t_full) / (2 * duration), 0.02, 0.5))
    )
    return trapezoid_transit(time, period, t0, duration, rp_rs**2, ingress_fraction)


def transit_duration(period: float, rp_rs: float, impact: float, a_rs: float) -> float:
    """Total (T14) circular-orbit transit duration in days (Seager & Mallén-Ornelas 2003).

    Passing a negative ``rp_rs`` returns the full-transit duration T23 instead.
    """
    inc = np.arccos(impact / a_rs)
    arg = np.sqrt(max((1 + rp_rs) ** 2 - impact**2, 0.0)) / (a_rs * np.sin(inc))
    if arg >= 1:
        return period / 2
    return float(period / np.pi * np.arcsin(arg))


def eclipsing_binary(
    time: FloatArray,
    period: float,
    t0: float,
    duration: float,
    primary_depth: float,
    secondary_ratio: float,
    secondary_phase: float = 0.5,
    ingress_fraction: float = 0.45,
) -> FloatArray:
    """Relative flux of a detached/grazing eclipsing binary with primary and secondary eclipses.

    Args:
        time: Times in days.
        period: Binary period in days.
        t0: Mid-time of a primary eclipse.
        duration: Eclipse duration in days (used for both eclipses).
        primary_depth: Fractional depth of the primary eclipse.
        secondary_ratio: Secondary depth as a fraction of the primary depth (0..1).
        secondary_phase: Orbital phase of the secondary (0.5 for circular orbits).
        ingress_fraction: Shape parameter; EBs are usually V-shaped (close to 0.5).

    Returns:
        Relative flux array.
    """
    primary = trapezoid_transit(time, period, t0, duration, primary_depth, ingress_fraction)
    secondary = trapezoid_transit(
        time,
        period,
        t0 + secondary_phase * period,
        duration,
        primary_depth * secondary_ratio,
        ingress_fraction,
    )
    return primary * secondary


@dataclass(frozen=True)
class SyntheticLightCurve:
    """A synthetic light curve plus the truth used to generate it."""

    time: FloatArray
    flux: FloatArray
    flux_err: FloatArray
    trend: FloatArray
    truth: dict[str, float]


def make_time_grid(
    baseline_days: float = 90.0,
    cadence_minutes: float = 29.4,
    gap_days: float = 1.0,
    segment_days: float = 30.0,
    start: float = 0.0,
) -> FloatArray:
    """Evenly sampled times with a gap every ``segment_days`` (like Kepler quarters)."""
    dt = cadence_minutes / (24 * 60)
    t = np.arange(start, start + baseline_days, dt)
    if gap_days > 0 and segment_days > 0:
        in_gap = ((t - start) % segment_days) > (segment_days - gap_days)
        t = t[~in_gap]
    return t


def synthetic_star(
    time: FloatArray,
    noise_ppm: float = 200.0,
    variability_ppm: float = 2000.0,
    rotation_period: float = 12.0,
    red_noise_ppm: float = 50.0,
    seed: int | None = 0,
) -> tuple[FloatArray, FloatArray]:
    """Stellar flux without transits: rotation modulation + slow drift + red + white noise.

    Returns:
        ``(flux_without_white_noise, white_noise)``. Adding both gives the observed flux.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(time, dtype=float)
    span = max(t.max() - t.min(), 1e-6)
    phase = rng.uniform(0, 2 * np.pi)
    rot = (
        variability_ppm
        * 1e-6
        * (
            np.sin(2 * np.pi * t / rotation_period + phase)
            + 0.3 * np.sin(4 * np.pi * t / rotation_period + 2 * phase)
        )
    )
    drift = 0.5 * variability_ppm * 1e-6 * ((t - t.min()) / span - 0.5) ** 2
    # Red noise: smoothed random walk, ~0.5-day correlation length.
    steps = rng.normal(0, 1, t.size)
    kernel_n = max(int(0.5 / max(np.median(np.diff(t)), 1e-4)), 1)
    red = np.convolve(steps, np.ones(kernel_n) / kernel_n, mode="same")
    red = red / (red.std() + 1e-12) * red_noise_ppm * 1e-6
    white = rng.normal(0, noise_ppm * 1e-6, t.size)
    return 1.0 + rot + drift + red, white


def synthetic_planet_lightcurve(
    period: float = 3.5,
    t0: float = 1.2,
    depth: float = 0.002,
    duration: float = 0.12,
    baseline_days: float = 90.0,
    cadence_minutes: float = 29.4,
    noise_ppm: float = 200.0,
    variability_ppm: float = 2000.0,
    seed: int | None = 0,
) -> SyntheticLightCurve:
    """Synthetic star with one periodic planet transit (U-shaped trapezoid)."""
    t = make_time_grid(baseline_days, cadence_minutes)
    star, white = synthetic_star(t, noise_ppm, variability_ppm, seed=seed)
    model = trapezoid_transit(t, period, t0, duration, depth, ingress_fraction=0.12)
    flux = star * model + white
    err = np.full_like(t, noise_ppm * 1e-6)
    truth = {"period": period, "t0": t0, "depth": depth, "duration": duration}
    return SyntheticLightCurve(t, flux, err, star, truth)


def synthetic_eb_lightcurve(
    period: float = 2.2,
    t0: float = 0.7,
    primary_depth: float = 0.05,
    secondary_ratio: float = 0.4,
    duration: float = 0.15,
    baseline_days: float = 90.0,
    cadence_minutes: float = 29.4,
    noise_ppm: float = 300.0,
    seed: int | None = 1,
) -> SyntheticLightCurve:
    """Synthetic eclipsing binary: V-shaped primary and a secondary at phase 0.5."""
    t = make_time_grid(baseline_days, cadence_minutes)
    star, white = synthetic_star(t, noise_ppm, variability_ppm=1000.0, seed=seed)
    model = eclipsing_binary(t, period, t0, duration, primary_depth, secondary_ratio)
    flux = star * model + white
    err = np.full_like(t, noise_ppm * 1e-6)
    truth = {
        "period": period,
        "t0": t0,
        "depth": primary_depth,
        "duration": duration,
        "secondary_ratio": secondary_ratio,
    }
    return SyntheticLightCurve(t, flux, err, star, truth)


# --------------------------------------------------------------------------------------
# Injection into real light curves (training-data fallback)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Injection:
    """Parameters of one injected signal."""

    label: str  # "PC" or "EB"
    period: float
    t0: float
    duration: float
    depth: float
    secondary_ratio: float = 0.0


def measured_duration(time_model: FloatArray, flux_model: FloatArray, period: float) -> float:
    """Total duration of the dip in a noiseless, finely sampled one-period model, in days."""
    below = flux_model < 1.0 - 1e-7
    if not below.any():
        return 0.0
    dt = float(np.median(np.diff(time_model)))
    return float(min(below.sum() * dt, period / 2))


def random_planet_injection(
    time: FloatArray, rng: np.random.Generator, min_transits: int = 3
) -> tuple[Injection, FloatArray]:
    """Draw a random planet and return ``(injection, relative_flux_model)`` sampled at ``time``.

    Period is log-uniform between 0.5 d and baseline / min_transits (max 50 d); Rp/R* gives
    depths of roughly 100-20000 ppm; impact parameter is uniform in [0, 0.9].
    """
    baseline = float(time.max() - time.min())
    p_max = min(50.0, baseline / min_transits)
    period = float(np.exp(rng.uniform(np.log(0.5), np.log(max(p_max, 0.6)))))
    t0 = float(time.min() + rng.uniform(0, period))
    rp_rs = float(np.exp(rng.uniform(np.log(0.01), np.log(0.14))))
    impact = float(rng.uniform(0, 0.9))
    density = float(np.exp(rng.uniform(np.log(0.3), np.log(2.0))))
    model = physical_transit(time, period, t0, rp_rs, impact, density)
    fine_t = np.linspace(t0 - period / 2, t0 + period / 2, 20001)
    fine = physical_transit(fine_t, period, t0, rp_rs, impact, density)
    duration = measured_duration(fine_t, fine, period)
    depth = float(1.0 - fine.min())
    return Injection("PC", period, t0, duration, depth), model


def random_eb_injection(
    time: FloatArray, rng: np.random.Generator, min_transits: int = 3
) -> tuple[Injection, FloatArray]:
    """Draw a random eclipsing binary and return ``(injection, relative_flux_model)``.

    Two families, 50/50:
      * Clear EB: deep V-shaped primary (1-30 %), secondary at phase ~0.5.
      * "Half-period" EB: two near-equal eclipses; the catalog/BLS period is half the true
        period, so the signature is alternating odd/even depths at the reported period.
    """
    baseline = float(time.max() - time.min())
    p_max = min(30.0, baseline / min_transits)
    true_period = float(np.exp(rng.uniform(np.log(0.6), np.log(max(p_max, 0.7)))))
    t0 = float(time.min() + rng.uniform(0, true_period))
    depth = float(np.exp(rng.uniform(np.log(0.005), np.log(0.3))))
    duration = float(np.clip(0.06 * true_period ** (1 / 3) * rng.uniform(0.7, 1.6), 0.04, 0.4))
    if rng.uniform() < 0.5:
        ratio = float(rng.uniform(0.05, 0.9))
        sec_phase = 0.5 if rng.uniform() < 0.8 else float(rng.uniform(0.35, 0.65))
        model = eclipsing_binary(time, true_period, t0, duration, depth, ratio, sec_phase)
        return Injection("EB", true_period, t0, duration, depth, ratio), model
    ratio = float(rng.uniform(0.5, 0.95))
    model = eclipsing_binary(time, true_period, t0, duration, depth, ratio, 0.5)
    return Injection("EB", true_period / 2, t0, duration, depth, ratio), model

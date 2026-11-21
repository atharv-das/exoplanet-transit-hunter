"""Single entry point: target -> full analysis result.

    analyze("Kepler-10")
      1. fetch      light curve from MAST (or the local cache / demo cache)
      2. clean      NaNs, flares, normalization
      3. detrend    wotan biweight (window widened per candidate to 3x its duration)
      4. search     iterative BLS: find, mask, repeat
      5. per candidate:
           views + features + vetting  (same build_example as training -> no train/serve skew)
           classify (CNN or GBM), explain (Integrated Gradients / TreeSHAP)
           trapezoid fit, planet radius, plot data, plain-language explanation

Errors are raised as :class:`PipelineError` with a stable ``code`` that the API and UI map to
friendly messages.
"""

from __future__ import annotations

import json
import logging
import re
import time as _time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from scipy.optimize import least_squares

from transit_hunter import __version__
from transit_hunter.api.schemas import (
    AnalysisResult,
    Candidate,
    CandidatePlots,
    ErrorCode,
    Explanation,
    FoldedPlot,
    HuntResult,
    HuntRow,
    LightCurveInfo,
    LightCurvePlots,
    ModelInfo,
    PhysicalParams,
    Probabilities,
    Saliency,
    Series,
    VettingFlagOut,
)
from transit_hunter.config import (
    CLASS_NAMES,
    CLASSES,
    CONFIG_DIR,
    DataConfig,
    Mission,
    data_path,
    get_settings,
    load_data_config,
)
from transit_hunter.data.fetch import (
    LightCurveData,
    NoDataError,
    TargetNotFoundError,
    fetch_lightcurve,
    fetch_stellar_params,
)
from transit_hunter.data.synthetic import trapezoid_transit
from transit_hunter.processing.clean import LightCurveTooShortError, clean_lightcurve
from transit_hunter.processing.detrend import detrend, recommended_window
from transit_hunter.processing.features import SCALAR_FEATURES, Example, build_example
from transit_hunter.processing.fold import ViewError, phase_fold
from transit_hunter.processing.search import TransitCandidate, iterative_search

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
ProgressFn = Callable[[str, float], None]

PIPELINE_VERSION = f"{__version__}+p1"
RSUN_PER_REARTH = 109.08
RSUN_PER_RJUP = 9.731
MAX_PLOT_POINTS = 3000
MAX_FOLDED_POINTS = 2500
MAX_PERIODOGRAM_POINTS = 2000


class PipelineError(RuntimeError):
    """An expected failure with a stable machine-readable code."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code: ErrorCode = code
        self.message = message


# --------------------------------------------------------------------------------------
# Model loading (once per process)
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_predictor() -> Any | None:
    """The serving model, or ``None`` if nothing is trained yet (search + vetting still work)."""
    from transit_hunter.models.predict import load_predictor

    try:
        return load_predictor(get_settings().model_kind)
    except FileNotFoundError as exc:
        logger.warning("no trained model available: %s", exc)
        return None


def warmup() -> None:
    """Load the model and JIT-compile the BLS kernel so the first real request is fast."""
    from transit_hunter.data.synthetic import synthetic_planet_lightcurve
    from transit_hunter.processing.search import bls_search

    start = _time.perf_counter()
    get_predictor()
    cfg = load_data_config()
    lc = synthetic_planet_lightcurve(baseline_days=20.0, seed=0)
    try:
        bls_search(lc.time, lc.flux, lc.flux_err, cfg.search, keep_periodogram=False)
        detrend(lc.time, lc.flux, lc.flux_err, cfg.detrend)
    except ValueError:
        pass
    logger.info("warm-up finished in %.1fs", _time.perf_counter() - start)


def reset_predictor() -> None:
    """Forget the loaded model (after retraining, or in tests)."""
    get_predictor.cache_clear()


# --------------------------------------------------------------------------------------
# Demo cache
# --------------------------------------------------------------------------------------


def normalize_query(q: str) -> str:
    """Case/spacing-insensitive key: ``"  kepler 10 "`` -> ``"kepler10"``."""
    return re.sub(r"[^a-z0-9]", "", q.lower())


def demo_targets_config() -> list[dict[str, Any]]:
    """Entries of ``configs/demo_targets.yaml``."""
    path = CONFIG_DIR / "demo_targets.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    return list(data.get("targets", []))


def demo_index() -> dict[str, Any]:
    """``data/demo_cache/index.json`` (written by ``scripts/warm_demo_cache.py``)."""
    p = data_path("demo_cache") / "index.json"
    return json.loads(p.read_text()) if p.exists() else {"targets": []}


def demo_lookup(query: str) -> Path | None:
    """Path of a cached demo result matching a slug, title, query or target ID."""
    key = normalize_query(query)
    for t in demo_index().get("targets", []):
        aliases = {
            normalize_query(str(t.get(k, ""))) for k in ("slug", "title", "query", "target_id")
        }
        if key in aliases:
            p = data_path("demo_cache") / f"{t['slug']}.json"
            return p if p.exists() else None
    return None


def load_demo_result(path: Path) -> AnalysisResult:
    """Read one cached demo result."""
    res = AnalysisResult.model_validate_json(path.read_text())
    return res.model_copy(update={"cached": True, "source": "demo_cache"})


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def downsample(x: FloatArray, y: FloatArray, max_points: int = MAX_PLOT_POINTS) -> Series:
    """Bin to at most ``max_points`` equal-count bins (means), for plotting."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size <= max_points:
        return Series(x=x.tolist(), y=y.tolist())
    edges = np.linspace(0, x.size, max_points + 1).astype(int)
    sums_x = np.add.reduceat(x, edges[:-1])
    sums_y = np.add.reduceat(y, edges[:-1])
    n = np.diff(edges)
    return Series(x=(sums_x / n).tolist(), y=(sums_y / n).tolist())


def fit_trapezoid(
    phase_days: FloatArray, flux: FloatArray, duration: float, depth: float
) -> dict[str, float]:
    """Least-squares trapezoid fit to folded data near the transit.

    Returns ``depth``, ``duration`` (days), ``ingress_fraction`` and ``t_offset`` (days). Falls
    back to the BLS values if the fit fails.
    """
    window = np.abs(phase_days) < 2.5 * duration
    x, y = phase_days[window], flux[window]
    fallback = {"depth": depth, "duration": duration, "ingress_fraction": 0.15, "t_offset": 0.0}
    if x.size < 10 or depth <= 0:
        return fallback
    base = float(np.median(y[np.abs(x) > duration])) if (np.abs(x) > duration).any() else 1.0

    def model(p: FloatArray) -> FloatArray:
        d, dur, ingr, off = p
        return base * trapezoid_transit(x, 1e6, off, dur, d, ingr)

    p0 = np.array([depth, duration, 0.15, 0.0])
    lo = np.array([0.0, 0.3 * duration, 0.02, -0.5 * duration])
    hi = np.array([min(1.0, 5 * depth + 1e-3), 3.0 * duration, 0.5, 0.5 * duration])
    try:
        sol = least_squares(
            lambda p: model(p) - y, np.clip(p0, lo, hi), bounds=(lo, hi), loss="soft_l1"
        )
    except (ValueError, RuntimeError):
        return fallback
    if not sol.success:
        return fallback
    d, dur, ingr, off = (float(v) for v in sol.x)
    return {"depth": d, "duration": dur, "ingress_fraction": ingr, "t_offset": off}


def _finite_or_none(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _fmt_period(p: float) -> str:
    return f"{p * 24:.1f} hours" if p < 1 else f"{p:.2f} days"


def narrate(
    cand: TransitCandidate,
    example: Example,
    predicted: str | None,
    confidence: float | None,
    top_features: list[tuple[str, float]],
    mission: Mission,
    trained_missions: list[str] | None = None,
) -> Explanation:
    """Technical bullet points and an "explain like I'm 12" paragraph for one candidate."""
    v = example.vetting
    trained_missions = trained_missions or ["Kepler"]
    depth_pct = max(cand.depth, 0) * 100
    technical = [
        f"Periodic dip every {_fmt_period(cand.period)}, {cand.duration * 24:.1f} h long, "
        f"{cand.depth * 1e6:,.0f} ppm deep (SNR {cand.snr:.1f}, {cand.n_transits} transits)."
    ]
    for flag in v.flags:
        if flag.passed is False:
            technical.append(f"Red flag: {flag.message}.")
    passed = [f.name.replace("_", " ") for f in v.flags if f.passed]
    if passed:
        technical.append("Passed: " + ", ".join(passed) + ".")
    if top_features:
        names = ", ".join(n.replace("_", " ") for n, _ in top_features[:3])
        technical.append(f"Most influential inputs for this decision: {names}.")
    if not cand.significant:
        technical.insert(
            0, "This is the strongest dip found, but it is below the detection threshold."
        )
    if mission not in trained_missions:
        technical.append(
            f"The model was not trained on {mission} data; treat this prediction with care."
        )

    how_often = _fmt_period(cand.period)
    lasts = f"{cand.duration * 24:.1f} hours"
    if not cand.significant:
        eli12 = (
            "We looked for a repeating dip in this star's brightness, like the dimming you'd see "
            "if something kept passing in front of a light bulb. The best dip we found is too "
            "faint to "
            "tell apart from the normal flicker of the star and the telescope, so there's no "
            "convincing signal here."
        )
    elif predicted == "PC":
        eli12 = (
            f"Every {how_often}, this star gets about {depth_pct:.2g}% dimmer for {lasts}. "
            "That's what you'd see if a planet passed between us and the star, like a tiny moth "
            "flying in front of a streetlight. The dips are the same depth each time and flat at "
            "the bottom, which is how planets look. "
            f"The computer is {confidence or 0:.0%} sure it's a planet candidate, but astronomers "
            "would still need to check it with other telescopes before calling it a planet."
        )
    elif predicted == "EB":
        reasons = []
        if any(f.name == "secondary_eclipse" and f.passed is False for f in v.flags):
            reasons.append("there's a second, smaller dip halfway between the big ones")
        if any(f.name == "odd_even" and f.passed is False for f in v.flags):
            reasons.append("every other dip is a different depth")
        if any(f.name == "transit_shape" and f.passed is False for f in v.flags):
            reasons.append("the dip is pointy (V-shaped) instead of flat-bottomed")
        why = "; ".join(reasons) if reasons else "the dip is too deep and too pointy for a planet"
        eli12 = (
            f"This star dims every {how_often}, but it's probably not a planet. It looks like two "
            "stars circling each other and taking turns blocking each other's light. "
            f"Clues: {why}. "
            f"The computer is {confidence or 0:.0%} sure about that."
        )
    elif predicted == "FP":
        eli12 = (
            f"There's a small repeating dip every {how_often}, but it doesn't look like something "
            "orbiting the star. It's more likely a quirk of the telescope or noise from the star "
            f"itself. The computer is {confidence or 0:.0%} sure it's a false alarm."
        )
    else:
        eli12 = (
            f"This star gets a little dimmer every {how_often} for {lasts}. No trained model is "
            "loaded, so we can't say yet whether it's a planet."
        )
    return Explanation(technical=technical, eli12=eli12)


# --------------------------------------------------------------------------------------
# Per-candidate analysis
# --------------------------------------------------------------------------------------


def _candidate_result(
    cand: TransitCandidate,
    lc: LightCurveData,
    clean_t: FloatArray,
    clean_f: FloatArray,
    clean_e: FloatArray,
    cfg: DataConfig,
    detrend_method: str,
    stellar_radius: float | None,
    predictor: Any | None,
) -> Candidate:
    # Re-detrend with a window wide enough for this candidate's duration.
    window = recommended_window(cfg.detrend, [cand.duration])
    d = detrend(
        clean_t, clean_f, clean_e, cfg.detrend, window_length_days=window, method=detrend_method
    )

    # Refine the ephemeris with a trapezoid fit. BLS boxes underestimate the total (T14) duration,
    # especially for V-shaped eclipses; training used catalog T14 durations, so the model inputs
    # must use a comparable duration to avoid train/serve skew.
    fit0 = fit_trapezoid(
        phase_fold(d.time, cand.period, cand.t0), d.flux, cand.duration, cand.depth
    )
    if 0.5 * cand.duration <= fit0["duration"] <= 3.0 * cand.duration and fit0["depth"] > 0:
        cand = replace(cand, duration=fit0["duration"], t0=cand.t0 + fit0["t_offset"])
        if cand.duration * 3 > window:
            window = recommended_window(cfg.detrend, [cand.duration])
            d = detrend(
                clean_t,
                clean_f,
                clean_e,
                cfg.detrend,
                window_length_days=window,
                method=detrend_method,
            )
    example = build_example(
        d.time,
        d.flux,
        cand.period,
        cand.t0,
        cand.duration,
        cfg.views,
        stellar_radius,
        mission=lc.mission,
    )
    scalars = example.feature_vector(SCALAR_FEATURES)[None]
    gv, lv = example.views.global_view[None], example.views.local_view[None]

    probs = predicted = label = conf = saliency = None
    top: list[tuple[str, float]] = []
    if predictor is not None:
        from transit_hunter.models.explain import explain

        p = predictor.predict_proba(gv, lv, scalars)[0]
        k = int(np.argmax(p))
        probs = Probabilities(pc=float(p[0]), eb=float(p[1]), fp=float(p[2]))
        predicted, label, conf = CLASSES[k], CLASS_NAMES[CLASSES[k]], float(p[k])
        try:
            exp = explain(predictor, gv[0], lv[0], scalars[0], target=k)
            top = exp.top_features(5)
            saliency = Saliency(
                method=exp.method,
                target_class=exp.target_class,
                global_view=exp.global_magnitude,
                local_view=exp.local_magnitude,
                features=exp.feature_attributions,
            )
        except Exception:  # explanations are best-effort; never fail the analysis
            logger.warning("explanation failed", exc_info=True)

    # Folded light curve + trapezoid fit.
    ph = phase_fold(d.time, cand.period, cand.t0)
    fit = fit_trapezoid(ph, d.flux, cand.duration, cand.depth)
    near = np.abs(ph) < 3 * cand.duration
    idx = np.flatnonzero(near)
    if idx.size > MAX_FOLDED_POINTS:
        idx = np.sort(np.random.default_rng(0).choice(idx, MAX_FOLDED_POINTS, replace=False))
    order = np.argsort(ph[near])
    binned_x = ph[near][order]
    binned = downsample(binned_x * 24, d.flux[near][order], 120)
    grid = np.linspace(-3 * cand.duration, 3 * cand.duration, 400)
    model_y = trapezoid_transit(
        grid, 1e6, fit["t_offset"], fit["duration"], fit["depth"], fit["ingress_fraction"]
    )
    folded = FoldedPlot(
        points=Series(x=(ph[idx] * 24).tolist(), y=d.flux[idx].tolist()),
        binned=binned,
        model=Series(x=(grid * 24).tolist(), y=model_y.tolist()),
    )
    pg = cand.periodogram
    pg_series = Series(x=[], y=[])
    if pg is not None:
        pp, pw = np.asarray(pg.period), np.asarray(pg.power)
        if pp.size > MAX_PERIODOGRAM_POINTS:
            keep = np.linspace(0, pp.size - 1, MAX_PERIODOGRAM_POINTS).astype(int)
            # keep the peak itself even when thinning
            keep = np.unique(np.append(keep, int(np.argmax(pw))))
            pp, pw = pp[keep], pw[keep]
        pg_series = Series(x=pp.tolist(), y=pw.tolist())
    plots = CandidatePlots(
        periodogram=pg_series,
        periodogram_peak=cand.period,
        folded=folded,
        global_view=Series(
            x=example.views.global_phase.tolist(), y=example.views.global_view.tolist()
        ),
        local_view=Series(
            x=(example.views.local_phase * 24).tolist(), y=example.views.local_view.tolist()
        ),
    )

    rp_rsun = np.sqrt(max(fit["depth"], 0.0)) * stellar_radius if stellar_radius else None
    physical = PhysicalParams(
        stellar_radius_rsun=stellar_radius,
        planet_radius_rearth=float(rp_rsun * RSUN_PER_REARTH) if rp_rsun else None,
        planet_radius_rjup=float(rp_rsun * RSUN_PER_RJUP) if rp_rsun else None,
        fit_depth_ppm=fit["depth"] * 1e6,
        fit_duration_hours=fit["duration"] * 24,
        fit_ingress_fraction=fit["ingress_fraction"],
    )
    vet = example.vetting.to_dict()
    flags = [VettingFlagOut(**f) for f in vet.pop("flags")]
    return Candidate(
        rank=cand.rank,
        period=cand.period,
        t0=cand.t0,
        t0_bjd=cand.t0 + lc.time_offset,
        duration=cand.duration,
        duration_hours=cand.duration * 24,
        depth=cand.depth,
        depth_ppm=cand.depth * 1e6,
        snr=cand.snr,
        sde=cand.sde,
        n_transits=cand.n_transits,
        significant=cand.significant,
        probabilities=probs,
        predicted_class=predicted,
        predicted_label=label,
        confidence=conf,
        vetting_flags=flags,
        vetting=vet,
        features={k: _finite_or_none(v) for k, v in example.features.items()},
        physical=physical,
        plot_data=plots,
        saliency=saliency,
        explanation=narrate(
            cand,
            example,
            predicted,
            conf,
            top,
            lc.mission,
            getattr(predictor, "trained_missions", ["Kepler"]),
        ),
    )


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def analyze(
    target: str,
    mission: Mission | None = None,
    detrend_method: str | None = None,
    max_candidates: int | None = None,
    max_segments: int | None = None,
    force: bool = False,
    progress: ProgressFn | None = None,
    stellar_radius: float | None = None,
) -> AnalysisResult:
    """Run the full pipeline for one target.

    Args:
        target: Name or ID ("Kepler-10", "KIC 11904151", "TIC 261136679", a demo slug).
        mission: Restrict to "Kepler" or "TESS".
        detrend_method: "biweight" (default) or "savgol".
        max_candidates: Iterative-search depth (default from config).
        max_segments: Download at most this many quarters/sectors.
        force: Recompute even if a demo result exists.
        progress: Callback ``(stage, fraction)`` for job status updates.
        stellar_radius: Override the TIC stellar radius (R_sun).

    Returns:
        An :class:`~transit_hunter.api.schemas.AnalysisResult`.

    Raises:
        PipelineError: with codes ``target_not_found``, ``no_data``, ``offline_not_cached``,
            ``lightcurve_too_short`` or ``search_failed``.
    """
    report = progress or (lambda stage, frac: None)
    cfg = load_data_config()
    settings = get_settings()
    method = detrend_method or cfg.detrend.method

    if not force and detrend_method is None and max_candidates is None:
        demo = demo_lookup(target)
        if demo is not None:
            report("done", 1.0)
            return load_demo_result(demo)

    timings: dict[str, float] = {}
    t_start = _time.perf_counter()

    def mark(name: str, since: float) -> float:
        now = _time.perf_counter()
        timings[name] = round(now - since, 3)
        return now

    report("fetching light curve", 0.05)
    try:
        lc = fetch_lightcurve(target, mission=mission, max_segments=max_segments)
    except TargetNotFoundError as exc:
        code: ErrorCode = "offline_not_cached" if settings.offline else "target_not_found"
        raise PipelineError(code, str(exc)) from exc
    except NoDataError as exc:
        raise PipelineError("no_data", str(exc)) from exc
    t = mark("fetch", t_start)

    warnings: list[str] = []
    trained_on = getattr(get_predictor(), "trained_missions", ["Kepler"])
    if lc.mission not in trained_on:
        warnings.append(
            f"Model trained on {' + '.join(trained_on)} only; {lc.mission} results are outside "
            "its training domain."
        )
    if stellar_radius is None:
        report("looking up stellar parameters", 0.15)
        stellar = fetch_stellar_params(lc.target_id)
        stellar_radius = stellar.get("radius")
    else:
        stellar = {"radius": stellar_radius, "mass": None, "teff": None, "tmag": None}
    if stellar_radius is None:
        warnings.append("Stellar radius unknown: planet radius not estimated.")
    t = mark("stellar_params", t)

    report("cleaning and detrending", 0.2)
    try:
        c = clean_lightcurve(lc.time, lc.flux, lc.flux_err, cfg.cleaning)
    except LightCurveTooShortError as exc:
        raise PipelineError("lightcurve_too_short", str(exc)) from exc
    d = detrend(c.time, c.flux, c.flux_err, cfg.detrend, method=method)
    t = mark("clean_detrend", t)

    report("searching for transits (BLS)", 0.3)
    try:
        cands = iterative_search(d.time, d.flux, d.flux_err, cfg.search, max_candidates)
    except ValueError as exc:
        raise PipelineError("search_failed", str(exc)) from exc
    if not cands:
        raise PipelineError("search_failed", "the transit search returned no candidates")
    t = mark("search", t)

    predictor = get_predictor()
    if predictor is None:
        warnings.append("No trained model loaded: showing search and vetting only.")
    report("classifying candidates", 0.85)
    results: list[Candidate] = []
    for cand in cands:
        try:
            results.append(
                _candidate_result(
                    cand, lc, c.time, c.flux, c.flux_err, cfg, method, stellar_radius, predictor
                )
            )
        except ViewError as exc:
            warnings.append(f"Candidate {cand.rank} skipped: {exc}")
    if not results:
        raise PipelineError("search_failed", "no candidate could be folded into model views")
    t = mark("classify", t)

    any_sig = any(r.significant for r in results)
    if not any_sig:
        warnings.append("No significant periodic signal found (SNR below threshold).")
    plots = LightCurvePlots(
        raw=downsample(c.time, c.flux),
        trend=downsample(d.time, d.trend),
        detrended=downsample(d.time, d.flux),
    )
    timings["total"] = round(_time.perf_counter() - t_start, 3)
    report("done", 1.0)
    return AnalysisResult(
        query=target,
        target_id=lc.target_id,
        name=None if lc.query == lc.target_id else (lc.query or None),
        status="ok" if any_sig else "no_significant_signal",
        warnings=warnings,
        lightcurve=LightCurveInfo(
            mission=lc.mission,
            author=lc.author,
            exptime_s=lc.exptime_s,
            n_points=int(lc.time.size),
            n_points_clean=int(c.time.size),
            baseline_days=lc.baseline_days,
            segments=lc.segments,
            time_offset_bjd=lc.time_offset,
            detrend_method=method,
            detrend_window_days=d.window_length_days,
        ),
        plots=plots,
        candidates=results,
        stellar={k: _finite_or_none(v) for k, v in stellar.items()},
        model=ModelInfo(
            kind=getattr(predictor, "kind", None),
            checkpoint=str(getattr(predictor, "checkpoint", "")) or None,
        ),
        timings_s=timings,
        analyzed_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        pipeline_version=PIPELINE_VERSION,
    )


# --------------------------------------------------------------------------------------
# Hunt mode
# --------------------------------------------------------------------------------------


def hunt(
    targets: list[str],
    mission: Mission | None = None,
    max_segments: int | None = 4,
    progress: ProgressFn | None = None,
) -> HuntResult:
    """Analyze many targets (one candidate each) and rank them by planet probability.

    Failures are recorded per target and never abort the batch. Targets without a significant
    signal rank below every significant one.
    """
    report = progress or (lambda stage, frac: None)
    rows: list[HuntRow] = []
    for i, target in enumerate(targets):
        report(f"analyzing {target} ({i + 1}/{len(targets)})", i / max(len(targets), 1))
        try:
            res = analyze(target, mission=mission, max_candidates=1, max_segments=max_segments)
        except PipelineError as exc:
            rows.append(HuntRow(target_id=target, query=target, status=exc.code, error=exc.message))
            continue
        except Exception as exc:  # keep the batch going
            logger.exception("hunt: %s failed", target)
            rows.append(
                HuntRow(target_id=target, query=target, status="internal_error", error=str(exc))
            )
            continue
        c = res.candidates[0]
        rows.append(
            HuntRow(
                target_id=res.target_id,
                query=target,
                status=res.status,
                best_period=c.period,
                best_depth_ppm=c.depth_ppm,
                best_snr=c.snr,
                planet_probability=c.probabilities.pc if c.probabilities else None,
                predicted_class=c.predicted_class,
            )
        )

    def key(r: HuntRow) -> tuple[int, float]:
        sig = 1 if r.status == "ok" else 0
        return (-sig, -(r.planet_probability or 0.0))

    report("done", 1.0)
    return HuntResult(ranked=sorted(rows, key=key))

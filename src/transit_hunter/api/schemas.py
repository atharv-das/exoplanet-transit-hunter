"""Pydantic models for the API. The pipeline returns these same models, so there is one schema."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MissionName = Literal["Kepler", "TESS"]
ClassName = Literal["PC", "EB", "FP"]
JobState = Literal["queued", "running", "done", "failed"]
ErrorCode = Literal[
    "target_not_found",
    "no_data",
    "offline_not_cached",
    "lightcurve_too_short",
    "search_failed",
    "internal_error",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------- requests


class AnalyzeRequest(_Model):
    """Body of ``POST /analyze``."""

    target_id: str = Field(
        ..., min_length=1, examples=["Kepler-10", "TIC 261136679", "KIC 11904151"]
    )
    mission: MissionName | None = None
    detrend_method: Literal["biweight", "savgol"] | None = None
    search_method: Literal["bls"] = "bls"
    max_candidates: int | None = Field(None, ge=1, le=5)
    max_segments: int | None = Field(
        None, ge=1, le=40, description="limit quarters/sectors downloaded"
    )
    force: bool = Field(False, description="ignore cached results and recompute")


class HuntRequest(_Model):
    """Body of ``POST /hunt``."""

    target_ids: list[str] = Field(..., min_length=1, max_length=50)
    mission: MissionName | None = None
    max_segments: int | None = Field(4, ge=1, le=40)


# --------------------------------------------------------------------------------- results


def _round_sig(values: list[float], digits: int = 7) -> list[float]:
    out = []
    for v in values:
        if v == 0 or not math.isfinite(v):
            out.append(0.0 if not math.isfinite(v) else v)
        else:
            out.append(round(v, digits - 1 - math.floor(math.log10(abs(v)))))
    return out


class Series(_Model):
    """An x/y series for plotting (values rounded to 7 significant digits to keep JSON small)."""

    x: list[float]
    y: list[float]

    @field_validator("x", "y")
    @classmethod
    def _compact(cls, v: list[float]) -> list[float]:
        return _round_sig(v)


class VettingFlagOut(_Model):
    name: str
    passed: bool | None
    value: float | None
    message: str


class Probabilities(_Model):
    pc: float
    eb: float
    fp: float


class PhysicalParams(_Model):
    stellar_radius_rsun: float | None = None
    planet_radius_rearth: float | None = None
    planet_radius_rjup: float | None = None
    fit_depth_ppm: float | None = None
    fit_duration_hours: float | None = None
    fit_ingress_fraction: float | None = None


class FoldedPlot(_Model):
    """Phase-folded data around the transit (phase in hours from mid-transit)."""

    points: Series
    binned: Series
    model: Series


class CandidatePlots(_Model):
    periodogram: Series
    periodogram_peak: float
    folded: FoldedPlot
    global_view: Series  # phase in days
    local_view: Series  # phase in hours


class Saliency(_Model):
    method: str
    target_class: ClassName
    global_view: list[float]  # 0..1 magnitude per bin
    local_view: list[float]
    features: dict[str, float]

    @field_validator("global_view", "local_view")
    @classmethod
    def _compact(cls, v: list[float]) -> list[float]:
        return [round(x, 4) for x in v]


class Explanation(_Model):
    technical: list[str]
    eli12: str


class Candidate(_Model):
    rank: int
    period: float
    t0: float
    t0_bjd: float
    duration: float  # days
    duration_hours: float
    depth: float  # fractional
    depth_ppm: float
    snr: float
    sde: float
    n_transits: int
    significant: bool
    probabilities: Probabilities | None
    predicted_class: ClassName | None
    predicted_label: str | None
    confidence: float | None
    vetting_flags: list[VettingFlagOut]
    vetting: dict[str, Any]
    features: dict[str, float | None]
    physical: PhysicalParams
    plot_data: CandidatePlots
    saliency: Saliency | None
    explanation: Explanation


class LightCurveInfo(_Model):
    mission: MissionName
    author: str
    exptime_s: float
    n_points: int
    n_points_clean: int
    baseline_days: float
    segments: list[int]
    time_offset_bjd: float
    detrend_method: str
    detrend_window_days: float


class LightCurvePlots(_Model):
    raw: Series
    trend: Series
    detrended: Series


class ModelInfo(_Model):
    kind: str | None
    checkpoint: str | None
    trained_on: str = "Kepler DR25-era KOI cumulative table"


class AnalysisResult(_Model):
    """Everything the UI needs for one target."""

    query: str
    target_id: str
    name: str | None
    status: Literal["ok", "no_significant_signal"]
    warnings: list[str]
    lightcurve: LightCurveInfo
    plots: LightCurvePlots
    candidates: list[Candidate]
    stellar: dict[str, float | None]
    model: ModelInfo
    timings_s: dict[str, float]
    cached: bool = False
    source: Literal["computed", "cache", "demo_cache"] = "computed"
    analyzed_at_utc: str
    pipeline_version: str


class ErrorInfo(_Model):
    code: ErrorCode
    message: str


class JobStatus(_Model):
    """Body of ``GET /jobs/{job_id}``."""

    job_id: str
    kind: Literal["analyze", "hunt"] = "analyze"
    state: JobState
    stage: str | None = None
    progress: float = 0.0
    created_at_utc: str
    finished_at_utc: str | None = None
    result: AnalysisResult | None = None
    hunt_result: HuntResult | None = None
    error: ErrorInfo | None = None


class JobCreated(_Model):
    job_id: str
    state: JobState
    status_url: str


class TargetMatch(_Model):
    query: str
    target_id: str
    mission: MissionName
    n_products: int
    authors: list[str]
    source: Literal["mast", "cache", "demo_cache"]


class DemoTarget(_Model):
    slug: str
    title: str
    query: str
    target_id: str
    mission: MissionName
    category: str
    blurb: str
    predicted_class: ClassName | None = None
    confidence: float | None = None
    cached_at_utc: str | None = None


class HuntRow(_Model):
    target_id: str
    query: str
    status: str
    best_period: float | None = None
    best_depth_ppm: float | None = None
    best_snr: float | None = None
    planet_probability: float | None = None
    predicted_class: ClassName | None = None
    error: str | None = None


class HuntResult(_Model):
    ranked: list[HuntRow]


class Health(_Model):
    status: Literal["ok"]
    version: str
    model_kind: str | None
    offline: bool
    demo_targets: int
    jobs_running: int


JobStatus.model_rebuild()

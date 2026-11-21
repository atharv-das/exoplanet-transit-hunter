"""HTTP routes and the background job manager.

Slow work (MAST downloads, BLS) runs in a small thread pool. ``POST /analyze`` returns a job ID
right away; clients poll ``GET /jobs/{job_id}``. Finished analyses are stored in a ``diskcache``
keyed by the request, the pipeline version and the model checkpoint, so repeat requests are
instant and survive restarts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status

from transit_hunter import __version__
from transit_hunter.api.schemas import (
    AnalysisResult,
    AnalyzeRequest,
    DemoTarget,
    ErrorInfo,
    Health,
    HuntRequest,
    JobCreated,
    JobStatus,
    TargetMatch,
)
from transit_hunter.config import get_settings
from transit_hunter.data.fetch import (
    TargetNotFoundError,
    cache_path,
    parse_target,
    resolve_target,
)
from transit_hunter.pipeline import (
    PIPELINE_VERSION,
    PipelineError,
    analyze,
    demo_index,
    demo_lookup,
    get_predictor,
    hunt,
    load_demo_result,
    normalize_query,
)

logger = logging.getLogger(__name__)

router = APIRouter()
MAX_JOBS_KEPT = 500


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


# --------------------------------------------------------------------------------------
# Result cache
# --------------------------------------------------------------------------------------


class ResultCache:
    """Disk cache of finished analyses (JSON strings), 7-day expiry."""

    def __init__(self, directory: Path) -> None:
        import diskcache

        directory.mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(str(directory))

    @staticmethod
    def key(req: AnalyzeRequest) -> str:
        predictor = get_predictor()
        ckpt = str(getattr(predictor, "checkpoint", "none"))
        try:
            mtime = Path(ckpt).stat().st_mtime if predictor is not None else 0.0
        except OSError:
            mtime = 0.0
        payload = req.model_dump(exclude={"force"})
        payload["target_id"] = normalize_query(req.target_id)
        payload.update(version=PIPELINE_VERSION, ckpt=ckpt, mtime=mtime)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get(self, req: AnalyzeRequest) -> AnalysisResult | None:
        raw = self.cache.get(self.key(req))
        if raw is None:
            return None
        res = AnalysisResult.model_validate_json(raw)
        return res.model_copy(update={"cached": True, "source": "cache"})

    def put(self, req: AnalyzeRequest, res: AnalysisResult) -> None:
        self.cache.set(self.key(req), res.model_dump_json(), expire=7 * 24 * 3600)


# --------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------


class JobManager:
    """Thread-pool job runner with in-memory status (bounded history)."""

    def __init__(self, workers: int = 2, cache: ResultCache | None = None) -> None:
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        self.jobs: OrderedDict[str, JobStatus] = OrderedDict()
        self.lock = threading.Lock()
        self.cache = cache

    def _store(self, job: JobStatus) -> None:
        with self.lock:
            self.jobs[job.job_id] = job
            self.jobs.move_to_end(job.job_id)
            while len(self.jobs) > MAX_JOBS_KEPT:
                self.jobs.popitem(last=False)

    def _update(self, job_id: str, **fields: Any) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is not None:
                self.jobs[job_id] = job.model_copy(update=fields)

    def get(self, job_id: str) -> JobStatus | None:
        with self.lock:
            return self.jobs.get(job_id)

    @property
    def running(self) -> int:
        with self.lock:
            return sum(1 for j in self.jobs.values() if j.state in ("queued", "running"))

    def submit_analyze(self, req: AnalyzeRequest) -> JobStatus:
        job_id = uuid.uuid4().hex[:12]
        if not req.force:
            cached = self.cache.get(req) if self.cache else None
            if cached is None and req.detrend_method is None and req.max_candidates is None:
                demo = demo_lookup(req.target_id)
                cached = load_demo_result(demo) if demo else None
            if cached is not None:
                job = JobStatus(
                    job_id=job_id,
                    state="done",
                    stage="done",
                    progress=1.0,
                    created_at_utc=_now(),
                    finished_at_utc=_now(),
                    result=cached,
                )
                self._store(job)
                return job
        job = JobStatus(job_id=job_id, state="queued", created_at_utc=_now())
        self._store(job)
        self.pool.submit(self._run_analyze, job_id, req)
        return job

    def _run_analyze(self, job_id: str, req: AnalyzeRequest) -> None:
        self._update(job_id, state="running", stage="starting", progress=0.01)

        def progress(stage: str, frac: float) -> None:
            self._update(job_id, stage=stage, progress=round(frac, 3))

        try:
            res = analyze(
                req.target_id,
                mission=req.mission,
                detrend_method=req.detrend_method,
                max_candidates=req.max_candidates,
                max_segments=req.max_segments,
                force=req.force,
                progress=progress,
            )
            if self.cache is not None and res.source == "computed":
                self.cache.put(req, res)
            self._update(
                job_id, state="done", stage="done", progress=1.0, result=res, finished_at_utc=_now()
            )
        except PipelineError as exc:
            self._update(
                job_id,
                state="failed",
                error=ErrorInfo(code=exc.code, message=exc.message),
                finished_at_utc=_now(),
            )
        except Exception as exc:
            logger.exception("job %s failed", job_id)
            self._update(
                job_id,
                state="failed",
                error=ErrorInfo(code="internal_error", message=f"{type(exc).__name__}: {exc}"),
                finished_at_utc=_now(),
            )

    def submit_hunt(self, req: HuntRequest) -> JobStatus:
        job_id = uuid.uuid4().hex[:12]
        job = JobStatus(job_id=job_id, kind="hunt", state="queued", created_at_utc=_now())
        self._store(job)
        self.pool.submit(self._run_hunt, job_id, req)
        return job

    def _run_hunt(self, job_id: str, req: HuntRequest) -> None:
        self._update(job_id, state="running", stage="starting")
        try:
            res = hunt(
                req.target_ids,
                mission=req.mission,
                max_segments=req.max_segments,
                progress=lambda s, f: self._update(job_id, stage=s, progress=round(f, 3)),
            )
            self._update(
                job_id,
                state="done",
                stage="done",
                progress=1.0,
                hunt_result=res,
                finished_at_utc=_now(),
            )
        except Exception as exc:
            logger.exception("hunt %s failed", job_id)
            self._update(
                job_id,
                state="failed",
                error=ErrorInfo(code="internal_error", message=str(exc)),
                finished_at_utc=_now(),
            )

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def _jobs(request: Request) -> JobManager:
    jm: JobManager = request.app.state.jobs
    return jm


def _created(job: JobStatus) -> JobCreated:
    return JobCreated(job_id=job.job_id, state=job.state, status_url=f"/jobs/{job.job_id}")


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------


@router.get("/health", response_model=Health, tags=["meta"])
def health(request: Request) -> Health:
    """Liveness plus what is loaded."""
    predictor = get_predictor()
    return Health(
        status="ok",
        version=__version__,
        model_kind=getattr(predictor, "kind", None),
        offline=get_settings().offline,
        demo_targets=len(demo_index().get("targets", [])),
        jobs_running=_jobs(request).running,
    )


@router.get("/targets/search", response_model=list[TargetMatch], tags=["targets"])
def search_targets(
    q: str = Query(..., min_length=1, description="name or ID, e.g. Kepler-10, TIC 261136679"),
    mission: str | None = Query(None, pattern="^(Kepler|TESS)$"),
) -> list[TargetMatch]:
    """Resolve a name/ID to a canonical KIC/TIC target (demo cache, local cache, then MAST)."""
    key = normalize_query(q)
    for t in demo_index().get("targets", []):
        if key in {
            normalize_query(str(t.get(k, ""))) for k in ("slug", "title", "query", "target_id")
        }:
            return [
                TargetMatch(
                    query=q,
                    target_id=t["target_id"],
                    mission=t["mission"],
                    n_products=len(t.get("segments", [])),
                    authors=[t.get("author", "")],
                    source="demo_cache",
                )
            ]
    try:
        search_string, missions = parse_target(q, mission)  # type: ignore[arg-type]
    except TargetNotFoundError as exc:
        raise _error(404, "target_not_found", str(exc)) from exc
    for m in missions:
        if cache_path(search_string, m).exists():
            return [
                TargetMatch(
                    query=q,
                    target_id=search_string,
                    mission=m,
                    n_products=0,
                    authors=[],
                    source="cache",
                )
            ]
    if get_settings().offline:
        raise _error(404, "offline_not_cached", f"{q!r} is not cached and the server is offline")
    try:
        info = resolve_target(q, mission)  # type: ignore[arg-type]
    except TargetNotFoundError as exc:
        raise _error(404, "target_not_found", str(exc)) from exc
    except Exception as exc:
        raise _error(502, "internal_error", f"MAST lookup failed: {exc}") from exc
    return [
        TargetMatch(
            query=q,
            target_id=info.target_id,
            mission=info.mission,
            n_products=info.n_products,
            authors=info.authors,
            source="mast",
        )
    ]


@router.post(
    "/analyze", response_model=JobCreated, status_code=status.HTTP_202_ACCEPTED, tags=["analysis"]
)
def submit_analysis(req: AnalyzeRequest, request: Request) -> JobCreated:
    """Start an analysis. Poll ``status_url`` for the result."""
    return _created(_jobs(request).submit_analyze(req))


@router.get("/jobs/{job_id}", response_model=JobStatus, tags=["analysis"])
def job_status(job_id: str, request: Request) -> JobStatus:
    """Status, progress and (when done) the full result of a job."""
    job = _jobs(request).get(job_id)
    if job is None:
        raise _error(404, "job_not_found", f"unknown job {job_id}")
    return job


@router.get("/demo/targets", response_model=list[DemoTarget], tags=["demo"])
def demo_targets() -> list[DemoTarget]:
    """Pre-computed demo stars (available offline)."""
    out = []
    for t in demo_index().get("targets", []):
        out.append(DemoTarget(**{k: t.get(k) for k in DemoTarget.model_fields if k in t}))
    return out


@router.get("/demo/{slug}", response_model=AnalysisResult, tags=["demo"])
def demo_result(slug: str) -> AnalysisResult:
    """The cached full result for one demo star."""
    path = demo_lookup(slug)
    if path is None:
        raise _error(404, "target_not_found", f"no demo result for {slug!r}; run `make warm-cache`")
    return load_demo_result(path)


@router.post(
    "/hunt", response_model=JobCreated, status_code=status.HTTP_202_ACCEPTED, tags=["analysis"]
)
def submit_hunt(req: HuntRequest, request: Request) -> JobCreated:
    """Batch-scan targets and rank them by planet probability (poll ``status_url``)."""
    return _created(_jobs(request).submit_hunt(req))

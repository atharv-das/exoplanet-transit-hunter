"""Backend access for the UI: the HTTP API when it is up, the pipeline in-process when it is not.

The in-process fallback means the demo survives an API crash: demo stars are read straight from
``data/demo_cache`` and anything else runs through ``transit_hunter.pipeline.analyze``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

ProgressFn = Callable[[str, float], None]

#: User-facing text for every error code the backend can return.
FRIENDLY_ERRORS: dict[str, tuple[str, str]] = {
    "target_not_found": (
        "We couldn't find that star.",
        "Check the spelling, or try an ID like `KIC 11904151` or `TIC 261136679`.",
    ),
    "offline_not_cached": (
        "We're offline and that star isn't cached.",
        "Pick one of the demo stars above; they work without internet.",
    ),
    "no_data": (
        "This star has no usable Kepler or TESS light curve.",
        "Some stars were only observed in full-frame images we don't support yet. Try another target.",
    ),
    "lightcurve_too_short": (
        "The light curve is too short to search for repeating dips.",
        "Try a star with more observations, or allow more sectors in the sidebar.",
    ),
    "search_failed": (
        "The transit search couldn't run on this light curve.",
        "The observing window may be too short for two transits. Try more sectors.",
    ),
    "connection": (
        "The analysis service isn't reachable.",
        "Start it with `make api`, or keep going: demo stars still work from the local cache.",
    ),
    "internal_error": (
        "Something went wrong on our side.",
        "Try again, or pick a demo star. Details are in the API logs.",
    ),
}


class BackendError(RuntimeError):
    """An error with a code from :data:`FRIENDLY_ERRORS`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code if code in FRIENDLY_ERRORS else "internal_error"
        self.message = message


@dataclass
class AnalyzeOptions:
    mission: str | None = None
    detrend_method: str | None = None
    max_candidates: int | None = None
    max_segments: int | None = None
    force: bool = False

    def payload(self, target: str) -> dict[str, Any]:
        body: dict[str, Any] = {"target_id": target, "force": self.force}
        for k in ("mission", "detrend_method", "max_candidates", "max_segments"):
            v = getattr(self, k)
            if v is not None:
                body[k] = v
        return body


class Backend:
    """Talks to the API at ``TH_API_URL``; falls back to the local pipeline."""

    def __init__(self, api_url: str | None = None, timeout_s: float = 600.0) -> None:
        self.api_url = (api_url or os.environ.get("TH_API_URL", "http://localhost:8000")).rstrip(
            "/"
        )
        self.timeout_s = timeout_s
        self.http = httpx.Client(base_url=self.api_url, timeout=15.0)

    # ------------------------------------------------------------------ status
    def health(self) -> dict[str, Any] | None:
        """API health, or ``None`` if unreachable."""
        try:
            r = self.http.get("/health", timeout=2.0)
            r.raise_for_status()
            data: dict[str, Any] = r.json()
            return data
        except (httpx.HTTPError, ValueError):
            return None

    @property
    def mode(self) -> str:
        return "api" if self.health() is not None else "local"

    # ------------------------------------------------------------------ demo
    def demo_targets(self) -> list[dict[str, Any]]:
        """Demo stars from the API, or from the local index file."""
        try:
            r = self.http.get("/demo/targets", timeout=3.0)
            r.raise_for_status()
            targets: list[dict[str, Any]] = r.json()
            return targets
        except (httpx.HTTPError, ValueError):
            from transit_hunter.pipeline import demo_index

            return list(demo_index().get("targets", []))

    # ------------------------------------------------------------------ analysis
    def analyze(self, target: str, opts: AnalyzeOptions, progress: ProgressFn) -> dict[str, Any]:
        """Run an analysis and return the result dict (``AnalysisResult`` JSON).

        Raises:
            BackendError: with a code from :data:`FRIENDLY_ERRORS`.
        """
        if self.health() is not None:
            return self._analyze_api(target, opts, progress)
        return self._analyze_local(target, opts, progress)

    def _analyze_api(
        self, target: str, opts: AnalyzeOptions, progress: ProgressFn
    ) -> dict[str, Any]:
        try:
            r = self.http.post("/analyze", json=opts.payload(target))
        except httpx.HTTPError as exc:
            raise BackendError("connection", str(exc)) from exc
        if r.status_code == 422:
            raise BackendError("target_not_found", "invalid request")
        r.raise_for_status()
        job_id = r.json()["job_id"]
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            try:
                job = self.http.get(f"/jobs/{job_id}").json()
            except (httpx.HTTPError, ValueError) as exc:
                raise BackendError("connection", str(exc)) from exc
            if job["state"] == "done":
                progress("done", 1.0)
                result: dict[str, Any] = job["result"]
                return result
            if job["state"] == "failed":
                err = job.get("error") or {}
                raise BackendError(err.get("code", "internal_error"), err.get("message", "failed"))
            progress(job.get("stage") or "queued", float(job.get("progress") or 0.0))
            time.sleep(0.6)
        raise BackendError("internal_error", "analysis timed out")

    def _analyze_local(
        self, target: str, opts: AnalyzeOptions, progress: ProgressFn
    ) -> dict[str, Any]:
        from transit_hunter.pipeline import PipelineError, analyze

        try:
            res = analyze(
                target,
                mission=opts.mission,  # type: ignore[arg-type]
                detrend_method=opts.detrend_method,
                max_candidates=opts.max_candidates,
                max_segments=opts.max_segments,
                force=opts.force,
                progress=progress,
            )
        except PipelineError as exc:
            raise BackendError(exc.code, exc.message) from exc
        except Exception as exc:
            raise BackendError("internal_error", f"{type(exc).__name__}: {exc}") from exc
        out: dict[str, Any] = json.loads(res.model_dump_json())
        return out

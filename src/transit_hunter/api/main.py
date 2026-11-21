"""FastAPI application.

    uvicorn transit_hunter.api.main:app --reload       # dev
    transit-hunter-api                                  # console script (see pyproject)

Interactive docs at http://localhost:8000/docs.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from transit_hunter import __version__
from transit_hunter.api.routes import JobManager, ResultCache, router
from transit_hunter.config import get_settings, setup_logging
from transit_hunter.pipeline import warmup

logger = logging.getLogger(__name__)


def create_app(warm: bool = True, job_workers: int = 2) -> FastAPI:
    """Build the app.

    Args:
        warm: Load the model and JIT-compile BLS in a background thread at startup.
        job_workers: Concurrent analyses (each BLS already uses every core via numba).
    """
    setup_logging()
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cache = ResultCache(settings.cache_dir / "results")
        app.state.jobs = JobManager(workers=job_workers, cache=cache)
        if warm:
            threading.Thread(target=warmup, name="warmup", daemon=True).start()
        logger.info("API ready (offline=%s, data=%s)", settings.offline, settings.data_dir)
        yield
        app.state.jobs.shutdown()

    app = FastAPI(
        title="Exoplanet Transit Hunter API",
        version=__version__,
        description=(
            "Fetch Kepler/TESS light curves, search for periodic transits with BLS, and classify "
            "each signal as Planet Candidate, Eclipsing Binary or Other False Positive."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=2000)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    """Console-script entry point."""
    import uvicorn

    s = get_settings()
    uvicorn.run("transit_hunter.api.main:app", host=s.api_host, port=s.api_port)

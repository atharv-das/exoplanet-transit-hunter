"""Dataset build stages: labels -> download -> process -> (synth, tess) -> finalize.

Every stage is resumable:
    * download keeps an append-only manifest (``data/raw/manifest.jsonl``) and skips stars that
      already succeeded or have no data; lightkurve also caches the FITS files themselves.
    * process reads only cached light-curve arrays, so it never needs the network.

Training examples use *catalog* ephemerides (as AstroNet did), so label noise from a wrong BLS
period never enters the training set. At inference the same :func:`build_example` runs on BLS
ephemerides.
"""

from __future__ import annotations

import json
import logging
import time as _time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool

#: How often a crashed process pool is restarted before the remaining jobs are left for later.
_MAX_POOL_RESTARTS = 4
#: Seconds without any finished job before stuck jobs are skipped.
_STALL_TIMEOUT_S = 600
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from transit_hunter.config import CLASSES, DataConfig, Mission, data_path, load_data_config
from transit_hunter.data.dataset import finalize_dataset, save_part
from transit_hunter.data.fetch import (
    LightCurveData,
    NoDataError,
    TargetNotFoundError,
    cache_path,
    fetch_lightcurve,
    purge_fits,
)
from transit_hunter.data.labels import (
    fetch_koi_table,
    fetch_tess_eb_catalog,
    fetch_toi_table,
    label_summary,
    map_koi_labels,
    map_tess_labels,
    sample_stars,
)
from transit_hunter.data.synthetic import random_eb_injection, random_planet_injection
from transit_hunter.processing.clean import LightCurveTooShortError, clean_lightcurve
from transit_hunter.processing.detrend import detrend, recommended_window
from transit_hunter.processing.features import SCALAR_FEATURES, build_example
from transit_hunter.processing.fold import ViewError
from transit_hunter.processing.search import transit_mask

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------


def labels_path(mission: Mission = "Kepler") -> Path:
    """Mapped label table written by the labels stage."""
    return data_path("interim") / f"labels_{mission.lower()}.parquet"


def manifest_path() -> Path:
    """Append-only download log."""
    return data_path("raw") / "manifest.jsonl"


# --------------------------------------------------------------------------------------
# Stage: labels
# --------------------------------------------------------------------------------------


def stage_labels(refresh: bool = False, include_tess: bool = False) -> pd.DataFrame:
    """Fetch the KOI table (and optionally TOI), map labels, save to ``data/interim``."""
    koi = map_koi_labels(fetch_koi_table(refresh=refresh))
    koi.to_parquet(labels_path("Kepler"), index=False)
    logger.info("Kepler labels: %s", label_summary(koi))
    cfg = load_data_config()
    if include_tess or "TESS" in cfg.labels.train_missions:
        try:
            ebs = fetch_tess_eb_catalog(refresh=refresh)
        except Exception as exc:  # VizieR down / column change: degrade to PC/FP for TESS
            logger.warning("TESS EB catalog unavailable (%s); TESS gets PC/FP labels only", exc)
            ebs = None
        toi = map_tess_labels(
            fetch_toi_table(refresh=refresh),
            ebs,
            cfg.labels.tess_max_period_days,
            seed=cfg.split.seed,
        )
        toi.to_parquet(labels_path("TESS"), index=False)
        logger.info("TESS labels: %s", label_summary(toi))
    return koi


def load_labels(mission: Mission = "Kepler") -> pd.DataFrame:
    """Load mapped labels, running the labels stage if needed."""
    p = labels_path(mission)
    if not p.exists():
        stage_labels(include_tess=mission == "TESS")
    return pd.read_parquet(p)


# --------------------------------------------------------------------------------------
# Stage: download
# --------------------------------------------------------------------------------------


def read_manifest() -> dict[str, dict[str, Any]]:
    """Latest manifest record per star."""
    p = manifest_path()
    out: dict[str, dict[str, Any]] = {}
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a line truncated by a crash
        out[rec["star_id"]] = rec
    return out


def _append_manifest(rec: dict[str, Any]) -> None:
    with manifest_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def download_one(
    star_id: str, mission: Mission, purge: bool = False, max_segments: int | None = None
) -> dict[str, Any]:
    """Download one star's light curve into the cache. Never raises; returns a manifest record.

    With ``purge=True`` the FITS files are deleted once the compact ``.npz`` is written
    (~10x less disk; re-downloading is needed only if preprocessing of raw columns changes).
    """
    start = _time.perf_counter()
    rec: dict[str, Any] = {"star_id": star_id, "mission": mission}
    try:
        lc = fetch_lightcurve(star_id, mission=mission, use_cache=True, max_segments=max_segments)
        rec.update(status="ok", n_points=int(lc.time.size), segments=lc.segments)
        if purge:
            rec["fits_purged"] = purge_fits(lc.target_id, mission)
    except TargetNotFoundError as exc:
        rec.update(status="notfound", error=str(exc))
    except NoDataError as exc:
        rec.update(status="nodata", error=str(exc))
    except Exception as exc:  # network errors after retries, corrupt files, ...
        rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
    rec["seconds"] = round(_time.perf_counter() - start, 2)
    return rec


def _run_pool(
    fn: Callable[..., Any],
    jobs: list[tuple[Any, ...]],
    workers: int,
    desc: str,
    on_result: Callable[[Any], None],
) -> None:
    """Run jobs in worker processes; survive crashed workers and stalled jobs."""
    if workers <= 1:
        for job in tqdm(jobs, desc=desc):
            on_result(fn(*job))
        return
    remaining = list(jobs)
    bar = tqdm(total=len(remaining), desc=desc)
    restarts = 0
    while remaining and restarts <= _MAX_POOL_RESTARTS:
        finished: set[int] = set()
        hung: set[int] = set()
        stalled = crashed = False
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = {pool.submit(fn, *job): i for i, job in enumerate(remaining)}
        pending = set(futures)
        while pending and not (stalled or crashed):
            done, pending = wait(pending, timeout=_STALL_TIMEOUT_S, return_when=FIRST_COMPLETED)
            if not done:
                stalled = True
                hung = {futures[f] for f in pending if f.running()}
                logger.warning(
                    "no job finished for %ds; skipping stuck job(s) %s and restarting workers",
                    _STALL_TIMEOUT_S,
                    ", ".join(str(remaining[i][0]) for i in sorted(hung)) or "(none running)",
                )
                break
            for fut in done:
                i = futures[fut]
                try:
                    result = fut.result()
                except BrokenProcessPool:
                    crashed = True
                    continue
                except Exception:
                    logger.exception("job %s failed", remaining[i][0])
                    finished.add(i)
                    bar.update(1)
                    continue
                on_result(result)
                finished.add(i)
                bar.update(1)
        if not (stalled or crashed):
            pool.shutdown(wait=True)
            bar.close()
            return
        for proc in list((getattr(pool, "_processes", None) or {}).values()):
            proc.kill()
        pool.shutdown(wait=False, cancel_futures=True)
        bar.update(len(hung))
        remaining = [job for i, job in enumerate(remaining) if i not in finished and i not in hung]
        if crashed:
            restarts += 1
            workers = max(1, workers // 2)
            logger.warning(
                "a worker died (often out of memory); restart %d/%d with %d workers, %d jobs left",
                restarts, _MAX_POOL_RESTARTS, workers, len(remaining),
            )
    bar.close()
    if remaining:
        logger.error(
            "giving up on %d jobs after repeated worker crashes; re-run to retry", len(remaining)
        )


def stage_download(
    limit: int | None,
    workers: int,
    mission: Mission = "Kepler",
    retry_errors: bool = True,
    seed: int | None = None,
    purge: bool = False,
) -> dict[str, int]:
    """Download light curves for a class-balanced sample of labeled stars.

    Args:
        limit: Number of stars (``None`` = all labeled stars).
        workers: Parallel worker processes.
        mission: ``"Kepler"`` (training) or ``"TESS"`` (validation).
        retry_errors: Retry stars whose previous attempt hit a transient error.
        seed: Sampling seed (defaults to the split seed).
        purge: Delete FITS files after caching each star's arrays.

    Returns:
        Status counts for this run.
    """
    cfg = load_data_config()
    labels = load_labels(mission)
    stars = sample_stars(labels, limit, seed if seed is not None else cfg.split.seed)
    done = read_manifest()
    skip = {"ok", "nodata", "notfound"} | (set() if retry_errors else {"error"})
    todo = [s for s in stars if done.get(s, {}).get("status") not in skip]
    logger.info(
        "%d stars selected, %d already done, %d to download",
        len(stars),
        len(stars) - len(todo),
        len(todo),
    )
    counts: dict[str, int] = {}

    def record(rec: dict[str, Any]) -> None:
        _append_manifest(rec)
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1

    max_seg = cfg.download.tess_max_sectors if mission == "TESS" else None
    jobs = [(s, mission, purge, max_seg) for s in todo]
    _run_pool(download_one, jobs, workers, f"download {mission}", record)
    logger.info("download finished: %s", counts)
    return counts


# --------------------------------------------------------------------------------------
# Stage: process
# --------------------------------------------------------------------------------------


@dataclass
class StarResult:
    """Examples produced from one star (picklable, returned by worker processes)."""

    star_id: str
    rows: list[dict[str, Any]]
    global_views: list[np.ndarray]
    local_views: list[np.ndarray]
    errors: list[str]


def _example_row(sig: dict[str, Any], example: Any) -> dict[str, Any]:
    row = {
        k: sig.get(k)
        for k in (
            "sample_id",
            "star_id",
            "mission",
            "label",
            "label_idx",
            "label_reason",
            "period",
            "t0",
            "duration",
            "catalog_depth_ppm",
            "stellar_radius",
        )
    }
    row["is_synthetic"] = bool(sig.get("is_synthetic", False))
    row.update(example.features)
    row["measured_depth_ppm"] = example.vetting.depth * 1e6
    row["global_coverage"] = example.views.global_coverage
    row["local_coverage"] = example.views.local_coverage
    return row


def detrend_for_signals(
    lc: LightCurveData, durations: Iterable[float], cfg: DataConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean + detrend one light curve with a window wide enough for every listed transit."""
    c = clean_lightcurve(lc.time, lc.flux, lc.flux_err, cfg.cleaning)
    window = recommended_window(cfg.detrend, list(durations))
    d = detrend(c.time, c.flux, c.flux_err, cfg.detrend, window_length_days=window)
    return d.time, d.flux, d.flux_err


def process_star(star_id: str, mission: Mission, signals: list[dict[str, Any]]) -> StarResult:
    """Build model inputs for every labeled signal of one cached star."""
    cfg = load_data_config()
    res = StarResult(star_id, [], [], [], [])
    p = cache_path(star_id, mission)
    if not p.exists():
        res.errors.append("light curve not cached")
        return res
    try:
        lc = LightCurveData.load(p)
        t, f, _ = detrend_for_signals(lc, [s["duration"] for s in signals], cfg)
    except (LightCurveTooShortError, ValueError, OSError) as exc:
        res.errors.append(f"{type(exc).__name__}: {exc}")
        return res
    for sig in signals:
        try:
            ex = build_example(
                t,
                f,
                sig["period"],
                sig["t0"],
                sig["duration"],
                cfg.views,
                sig.get("stellar_radius"),
                mission=mission,
            )
        except (ViewError, ValueError) as exc:
            res.errors.append(f"{sig['sample_id']}: {exc}")
            continue
        res.rows.append(_example_row(sig, ex))
        res.global_views.append(ex.views.global_view)
        res.local_views.append(ex.views.local_view)
    return res


def _collect(results: list[StarResult], part: str) -> int:
    rows = [r for res in results for r in res.rows]
    if not rows:
        logger.warning("no examples produced for part %r", part)
        return 0
    gv = np.stack([g for res in results for g in res.global_views]).astype(np.float32)
    lv = np.stack([v for res in results for v in res.local_views]).astype(np.float32)
    df = pd.DataFrame(rows)
    for feat in SCALAR_FEATURES:
        if feat not in df:
            df[feat] = np.nan
    save_part(part, df, gv, lv)
    n_err = sum(len(res.errors) for res in results)
    if n_err:
        err_path = data_path("interim") / f"errors_{part}.txt"
        err_path.write_text("\n".join(f"{res.star_id}\t{e}" for res in results for e in res.errors))
        logger.info("%d signals skipped (details: %s)", n_err, err_path)
    return len(df)


def stage_process(workers: int, mission: Mission = "Kepler", part: str | None = None) -> int:
    """Turn every cached, labeled star into examples; writes ``parts/<part>.*``."""
    labels = load_labels(mission)
    cached = {s for s in labels["star_id"].unique() if cache_path(s, mission).exists()}
    if not cached:
        raise FileNotFoundError("no cached light curves; run the download stage first")
    groups = {
        sid: g.to_dict("records")
        for sid, g in labels[labels["star_id"].isin(cached)].groupby("star_id")
    }
    results: list[StarResult] = []
    _run_pool(
        process_star,
        [(sid, mission, sigs) for sid, sigs in groups.items()],
        workers,
        f"process {mission}",
        results.append,
    )
    return _collect(results, part or ("real" if mission == "Kepler" else "tess"))


# --------------------------------------------------------------------------------------
# Stage: synth (fallback / augmentation)
# --------------------------------------------------------------------------------------


def synth_star(star_id: str, known: list[dict[str, Any]], n: int, seed: int) -> StarResult:
    """Inject ``n`` synthetic signals (alternating PC / EB) into one real light curve.

    Known transits of the host are removed first so the injected signal is the only one.
    """
    cfg = load_data_config()
    res = StarResult(star_id, [], [], [], [])
    rng = np.random.default_rng(seed)
    try:
        lc = LightCurveData.load(cache_path(star_id, "Kepler"))
    except OSError as exc:
        res.errors.append(str(exc))
        return res
    keep = np.ones(lc.time.size, dtype=bool)
    for k in known:
        keep &= ~transit_mask(lc.time, k["period"], k["t0"], k["duration"], 1.5)
    base_t, base_f, base_e = lc.time[keep], lc.flux[keep], lc.flux_err[keep]
    if base_t.size < cfg.cleaning.min_points:
        res.errors.append("too few points after masking known transits")
        return res
    for i in range(n):
        kind = "PC" if (i + seed) % 2 == 0 else "EB"
        draw = random_planet_injection if kind == "PC" else random_eb_injection
        inj, model = draw(base_t, rng)
        injected = LightCurveData(
            star_id, "Kepler", base_t, base_f * model, base_e, lc.author, lc.exptime_s, lc.segments
        )
        sig: dict[str, Any] = {
            "sample_id": f"SYN-{kind}-{star_id.replace(' ', '')}-{seed}-{i}",
            "star_id": star_id,
            "mission": "Kepler",
            "label": kind,
            "label_idx": CLASSES.index(kind),
            "label_reason": "synthetic injection",
            "period": inj.period,
            "t0": inj.t0,
            "duration": max(inj.duration, 0.02),
            "catalog_depth_ppm": inj.depth * 1e6,
            "stellar_radius": known[0].get("stellar_radius") if known else None,
            "is_synthetic": True,
        }
        srad = sig["stellar_radius"]
        try:
            t, f, _ = detrend_for_signals(injected, [float(sig["duration"])], cfg)
            ex = build_example(
                t,
                f,
                float(sig["period"]),
                float(sig["t0"]),
                float(sig["duration"]),
                cfg.views,
                float(srad) if srad is not None else None,
                mission="Kepler",
            )
        except (ViewError, ValueError, LightCurveTooShortError) as exc:
            res.errors.append(f"{sig['sample_id']}: {exc}")
            continue
        res.rows.append(_example_row(sig, ex))
        res.global_views.append(ex.views.global_view)
        res.local_views.append(ex.views.local_view)
    return res


def stage_synth(count: int, workers: int, per_star: int = 4, seed: int = 0) -> int:
    """Create about ``count`` synthetic PC/EB examples from cached Kepler light curves."""
    labels = load_labels("Kepler")
    hosts = sorted(s for s in labels["star_id"].unique() if cache_path(s, "Kepler").exists())
    if not hosts:
        raise FileNotFoundError("no cached light curves to inject into")
    rng = np.random.default_rng(seed)
    n_hosts = min(len(hosts), max(1, count // per_star))
    chosen = rng.choice(hosts, size=n_hosts, replace=False)
    known = {sid: g.to_dict("records") for sid, g in labels.groupby("star_id")}
    jobs = [(sid, known.get(sid, []), per_star, seed + i) for i, sid in enumerate(chosen)]
    results: list[StarResult] = []
    _run_pool(synth_star, jobs, workers, "synthetic", results.append)
    return _collect(results, "synth")


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def stage_finalize() -> dict[str, object]:
    """Merge all parts into ``dataset.npz`` + ``index.parquet`` with star-level splits."""
    return finalize_dataset(load_data_config())


def run_all(
    limit: int | None,
    workers: int,
    refresh_labels: bool = False,
    tess_limit: int | None = None,
    purge: bool = False,
) -> dict[str, object]:
    """labels -> download -> process (every training mission) -> finalize."""
    cfg = load_data_config()
    stage_labels(refresh=refresh_labels)
    for mission in cfg.labels.train_missions:
        n = (
            limit
            if mission == "Kepler"
            else (tess_limit if tess_limit is not None else cfg.download.tess_limit)
        )
        stage_download(n, workers, mission=mission, purge=purge)
        stage_process(workers, mission=mission)
    return stage_finalize()

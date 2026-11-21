"""Light-curve download from MAST via ``lightkurve``, with retries and an on-disk array cache.

Resolution rules:
    * ``"KIC 11904151"``, ``"kplr011904151"`` -> Kepler; ``"TIC 307210830"`` -> TESS.
    * A bare number uses the ``mission`` argument; without one, < 13.2 million tries Kepler first.
    * A name (``"Kepler-10"``, ``"Pi Mensae"``) goes to MAST's name resolver. Kepler/KOI names
      try Kepler first, everything else tries TESS first.

TESS product preference (configurable): SPOC 2-min, then TESS-SPOC FFI, then QLP.

Each segment (Kepler quarter / TESS sector) is normalized by its own median before
concatenation, which removes the large flux offsets between segments.
"""

from __future__ import annotations

import json
import logging
import re
import time as _time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from transit_hunter.config import DataConfig, Mission, data_path, get_settings, load_data_config

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
T = TypeVar("T")

#: Time-system offsets: BKJD = BJD - 2454833, BTJD = BJD - 2457000.
TIME_OFFSETS: dict[str, float] = {"Kepler": 2454833.0, "TESS": 2457000.0}
#: KIC IDs stop at about 13.2 million; larger bare numbers are treated as TIC IDs.
_KIC_MAX = 13_200_000

_ID_RE = re.compile(r"^\s*(?P<prefix>KIC|KPLR|TIC)?[\s_-]*(?P<num>\d+)\s*$", re.IGNORECASE)


class TargetNotFoundError(LookupError):
    """No light-curve products exist for the query in any requested mission."""


class NoDataError(LookupError):
    """Products were found, but no usable flux could be downloaded."""


@dataclass
class TargetInfo:
    """A resolved target."""

    query: str
    target_id: str  # canonical, e.g. "KIC 11904151" or "TIC 307210830"
    mission: Mission
    n_products: int
    authors: list[str] = field(default_factory=list)


@dataclass
class LightCurveData:
    """A stitched light curve in the mission time system (BKJD for Kepler, BTJD for TESS)."""

    target_id: str
    mission: Mission
    time: FloatArray
    flux: FloatArray
    flux_err: FloatArray
    author: str
    exptime_s: float
    segments: list[int]
    query: str = ""
    fetched_at: float = 0.0

    @property
    def time_offset(self) -> float:
        """Add to ``time`` to get BJD_TDB."""
        return TIME_OFFSETS[self.mission]

    @property
    def baseline_days(self) -> float:
        """Time span covered by the data."""
        return float(self.time.max() - self.time.min()) if self.time.size else 0.0

    def save(self, path: Path) -> None:
        """Write arrays + metadata to a compressed ``.npz``."""
        meta = {k: v for k, v in asdict(self).items() if k not in ("time", "flux", "flux_err")}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            time=self.time,
            flux=self.flux,
            flux_err=self.flux_err,
            meta=np.array(json.dumps(meta)),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> LightCurveData:
        """Read a file written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            return cls(
                time=np.asarray(z["time"], dtype=float),
                flux=np.asarray(z["flux"], dtype=float),
                flux_err=np.asarray(z["flux_err"], dtype=float),
                **meta,
            )


# --------------------------------------------------------------------------------------
# Identifier handling
# --------------------------------------------------------------------------------------


def parse_target(query: str, mission: Mission | None = None) -> tuple[str, list[Mission]]:
    """Normalize a user query into ``(search_string, missions_to_try)``.

    Examples:
        >>> parse_target("kic 11904151")
        ('KIC 11904151', ['Kepler'])
        >>> parse_target("TIC 307210830")
        ('TIC 307210830', ['TESS'])
        >>> parse_target("Kepler-10")
        ('Kepler-10', ['Kepler', 'TESS'])
    """
    q = query.strip()
    if not q:
        raise TargetNotFoundError("empty target query")
    m = _ID_RE.match(q)
    if m:
        prefix = (m.group("prefix") or "").upper()
        num = int(m.group("num"))
        if prefix in ("KIC", "KPLR"):
            return f"KIC {num}", ["Kepler"]
        if prefix == "TIC":
            return f"TIC {num}", ["TESS"]
        if mission == "Kepler":
            return f"KIC {num}", ["Kepler"]
        if mission == "TESS":
            return f"TIC {num}", ["TESS"]
        return (f"KIC {num}", ["Kepler"]) if num < _KIC_MAX else (f"TIC {num}", ["TESS"])
    if mission is not None:
        return q, [mission]
    if re.match(r"^\s*(kepler|koi|kic)", q, re.IGNORECASE):
        return q, ["Kepler", "TESS"]
    return q, ["TESS", "Kepler"]


def canonical_id(target_name: str, mission: Mission) -> str:
    """Turn a MAST ``target_name`` (``kplr011904151``, ``307210830``) into ``KIC n`` / ``TIC n``."""
    digits = re.sub(r"\D", "", str(target_name))
    if not digits:
        return str(target_name)
    return f"{'KIC' if mission == 'Kepler' else 'TIC'} {int(digits)}"


def cache_path(target_id: str, mission: Mission) -> Path:
    """Path of the cached arrays for a canonical target ID."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", target_id).strip("_")
    return data_path("interim") / "lightcurves" / f"{mission}_{safe}.npz"


# --------------------------------------------------------------------------------------
# Network helpers
# --------------------------------------------------------------------------------------


def _configure_mast() -> None:
    settings = get_settings()
    try:
        from astroquery.mast import conf

        conf.timeout = settings.mast_timeout_s
    except Exception:  # pragma: no cover - astroquery internals vary by version
        logger.debug("could not set astroquery.mast timeout", exc_info=True)


def with_retries(fn: Callable[[], T], retries: int, backoff_s: float, what: str) -> T:
    """Call ``fn`` and retry transient failures with exponential backoff.

    :class:`TargetNotFoundError` and :class:`NoDataError` are never retried.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except (TargetNotFoundError, NoDataError):
            raise
        except Exception as exc:
            attempt += 1
            if attempt > retries:
                raise
            wait = backoff_s * 2 ** (attempt - 1)
            logger.warning(
                "%s failed (%s); retry %d/%d in %.0fs", what, exc, attempt, retries, wait
            )
            _time.sleep(wait)


def _filled(values: Any) -> FloatArray:
    """Convert (possibly masked) astropy/numpy values to float64 with NaN for masked entries."""
    v = getattr(values, "value", values)
    if hasattr(v, "filled"):
        v = v.filled(np.nan)
    return np.asarray(v, dtype=float)


def _restrict_to_target(sr: Any) -> Any:
    """Keep only rows for the single closest target (name searches can return neighbors)."""
    if len(sr) == 0:
        return sr
    table = sr.table
    names = np.asarray(table["target_name"]).astype(str)
    if "distance" in table.colnames:
        dist = np.asarray(table["distance"], dtype=float)
        best = names[int(np.nanargmin(dist))] if np.isfinite(dist).any() else names[0]
    else:
        best = names[0]
    return sr[names == best]


def _search(search_string: str, mission: Mission, cfg: DataConfig) -> tuple[Any, str]:
    """Return ``(search_result, author)`` for the best product family of one mission."""
    import lightkurve as lk

    if mission == "Kepler":
        src = cfg.sources.kepler
        sr = lk.search_lightcurve(
            search_string, mission="Kepler", author=src.author, exptime=src.exptime
        )
        return _restrict_to_target(sr), src.author

    tsrc = cfg.sources.tess
    sr = _restrict_to_target(lk.search_lightcurve(search_string, mission="TESS"))
    if len(sr) == 0:
        return sr, ""
    authors = np.asarray(sr.table["author"]).astype(str)
    exptimes = np.asarray(sr.table["exptime"], dtype=float)
    allowed = np.isin(np.round(exptimes).astype(int), tsrc.exptime_priority)
    for author in tsrc.author_priority:
        rows = (authors == author) & allowed
        if not rows.any():
            continue
        sub = sr[rows]
        # One product per sector: keep the shortest allowed cadence.
        sectors = np.asarray(sub.table["mission"]).astype(str)
        sub_exp = np.asarray(sub.table["exptime"], dtype=float)
        keep = np.zeros(len(sub), dtype=bool)
        for sec in np.unique(sectors):
            idx = np.flatnonzero(sectors == sec)
            keep[idx[np.argmin(sub_exp[idx])]] = True
        return sub[keep], author
    return sr[:0], ""


def resolve_target(query: str, mission: Mission | None = None) -> TargetInfo:
    """Resolve a name or ID to a canonical target with available light curves.

    Raises:
        TargetNotFoundError: if no mission has light curves for the query.
    """
    cfg = load_data_config()
    _configure_mast()
    search_string, missions = parse_target(query, mission)
    for m in missions:
        sr, author = with_retries(
            lambda m=m: _search(search_string, m, cfg),  # type: ignore[misc]
            cfg.download.retries,
            cfg.download.backoff_s,
            f"MAST search {search_string} ({m})",
        )
        if len(sr):
            tid = canonical_id(str(sr.table["target_name"][0]), m)
            return TargetInfo(query, tid, m, len(sr), [author])
    raise TargetNotFoundError(f"no Kepler/TESS light curves found for {query!r}")


def _segment_number(meta: dict[str, Any]) -> int:
    for key in ("QUARTER", "SECTOR", "CAMPAIGN"):
        if key in meta and meta[key] is not None:
            try:
                return int(meta[key])
            except (TypeError, ValueError):
                continue
    return -1


def _download(sr: Any, mission: Mission, author: str, cfg: DataConfig) -> tuple[list[Any], float]:
    settings = get_settings()
    src = cfg.sources.kepler if mission == "Kepler" else cfg.sources.tess
    kwargs: dict[str, Any] = {"quality_bitmask": src.quality_bitmask}
    if settings.lightkurve_cache_dir is not None:
        kwargs["download_dir"] = str(settings.lightkurve_cache_dir)
    # QLP files have no PDCSAP column; let lightkurve choose their default flux.
    if author in ("Kepler", "SPOC", "TESS-SPOC"):
        kwargs["flux_column"] = src.flux_column
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lcc = sr.download_all(**kwargs)
        except Exception as exc:
            # A download interrupted earlier leaves a truncated FITS file in lightkurve's cache,
            # which it then reuses forever. Delete it so the retry downloads it fresh.
            removed = remove_corrupt_fits(str(exc))
            if removed:
                logger.warning("deleted %d corrupt cached file(s); retrying download", removed)
            raise
    exptime = float(np.nanmin(np.asarray(sr.table["exptime"], dtype=float)))
    return ([] if lcc is None else list(lcc)), exptime


_FITS_PATH_RE = re.compile(r"(/[^\s'\"]+?\.fits)")


def remove_corrupt_fits(message: str) -> int:
    """Delete every cached .fits file named in an error message (mastDownload cache only)."""
    removed = 0
    for raw in set(_FITS_PATH_RE.findall(message)):
        p = Path(raw)
        if "mastDownload" in p.parts and p.suffix == ".fits" and p.is_file():
            p.unlink(missing_ok=True)
            removed += 1
    return removed


def _stitch(lcs: Iterable[Any]) -> tuple[FloatArray, FloatArray, FloatArray, list[int]]:
    times, fluxes, errs, segs = [], [], [], []
    for lc in lcs:
        t = _filled(lc.time.value)
        f = _filled(lc.flux)
        e = _filled(lc.flux_err)
        ok = np.isfinite(t) & np.isfinite(f)
        if ok.sum() < 10:
            continue
        med = np.nanmedian(f[ok])
        if not np.isfinite(med) or med <= 0:
            continue
        times.append(t[ok])
        fluxes.append(f[ok] / med)
        errs.append(e[ok] / med)
        segs.append(_segment_number(dict(lc.meta)))
    if not times:
        raise NoDataError("downloaded files contain no valid flux")
    t = np.concatenate(times)
    order = np.argsort(t, kind="stable")
    return t[order], np.concatenate(fluxes)[order], np.concatenate(errs)[order], sorted(segs)


def fetch_lightcurve(
    query: str,
    mission: Mission | None = None,
    use_cache: bool = True,
    max_segments: int | None = None,
) -> LightCurveData:
    """Download (or load from cache) the stitched light curve for a target.

    Args:
        query: Name or ID (see module docstring).
        mission: Restrict to one mission.
        use_cache: Read/write ``data/interim/lightcurves/*.npz``.
        max_segments: Download at most this many quarters/sectors (most recent last); ``None``
            downloads all.

    Returns:
        A :class:`LightCurveData` normalized per segment to median 1.

    Raises:
        TargetNotFoundError: target unknown or no light curves.
        NoDataError: products exist but contain no usable flux.
    """
    cfg = load_data_config()
    settings = get_settings()
    search_string, missions = parse_target(query, mission)

    # Fast path: an explicit ID already in the cache needs no network at all.
    if use_cache and _ID_RE.match(query.strip()):
        p = cache_path(search_string, missions[0])
        if p.exists():
            return LightCurveData.load(p)

    if settings.offline:
        raise TargetNotFoundError(f"offline mode: {query!r} is not in the local cache")

    _configure_mast()
    last_error: Exception | None = None
    for m in missions:
        sr, author = with_retries(
            lambda m=m: _search(search_string, m, cfg),  # type: ignore[misc]
            cfg.download.retries,
            cfg.download.backoff_s,
            f"MAST search {search_string} ({m})",
        )
        if len(sr) == 0:
            continue
        tid = canonical_id(str(sr.table["target_name"][0]), m)
        p = cache_path(tid, m)
        if use_cache and p.exists():
            return LightCurveData.load(p)
        if max_segments is not None and len(sr) > max_segments:
            sr = sr[len(sr) - max_segments :]
        try:
            lcs, exptime = with_retries(
                lambda sr=sr, m=m, author=author: _download(sr, m, author, cfg),  # type: ignore[misc]
                cfg.download.retries,
                cfg.download.backoff_s,
                f"MAST download {tid}",
            )
            t, f, e, segs = _stitch(lcs)
        except NoDataError as exc:
            last_error = exc
            continue
        data = LightCurveData(
            target_id=tid,
            mission=m,
            time=t,
            flux=f,
            flux_err=e,
            author=author,
            exptime_s=exptime,
            segments=segs,
            query=query,
            fetched_at=_time.time(),
        )
        if use_cache:
            data.save(p)
        logger.info("fetched %s (%s %s): %d points, %d segments", tid, m, author, t.size, len(segs))
        return data
    if last_error is not None:
        raise NoDataError(str(last_error))
    raise TargetNotFoundError(f"no Kepler/TESS light curves found for {query!r}")


def fits_cache_root() -> Path:
    """Directory where lightkurve stores downloaded FITS files."""
    settings = get_settings()
    if settings.lightkurve_cache_dir is not None:
        return Path(settings.lightkurve_cache_dir)
    from lightkurve.config import get_cache_dir

    return Path(get_cache_dir())


def purge_fits(target_id: str, mission: Mission) -> int:
    """Delete lightkurve's FITS files for one target (after its arrays are cached).

    Returns:
        Number of files removed.
    """
    import shutil

    digits = re.sub(r"\D", "", target_id)
    if not digits:
        return 0
    pattern = f"kplr{int(digits):09d}*" if mission == "Kepler" else f"*{int(digits):016d}*"
    root = fits_cache_root() / "mastDownload" / mission
    removed = 0
    if not root.exists():
        return 0
    for d in root.glob(pattern):
        if d.is_dir():
            removed += sum(1 for _ in d.rglob("*.fits"))
            shutil.rmtree(d, ignore_errors=True)
        elif d.suffix == ".fits":
            d.unlink(missing_ok=True)
            removed += 1
    return removed


def fetch_stellar_params(target_id: str) -> dict[str, float | None]:
    """Stellar radius/mass/Teff/magnitude from the TESS Input Catalog (covers Kepler stars too).

    Returns an empty-valued dict on failure; the pipeline treats missing values as unknown.

    Note:
        TIC column names (``rad``, ``mass``, ``Teff``, ``Tmag``, ``ID``) are from astroquery's
        ``Catalogs.query_object(..., catalog="TIC")``. Verify with
        ``Catalogs.query_object("TIC 307210830", catalog="TIC", radius=0.001).colnames``.
    """
    empty: dict[str, float | None] = {"radius": None, "mass": None, "teff": None, "tmag": None}
    if get_settings().offline:
        return empty
    try:
        from astroquery.mast import Catalogs

        _configure_mast()
        tab = Catalogs.query_object(target_id, catalog="TIC", radius=0.002)
        if len(tab) == 0:
            return empty
        row = tab[0]

        def val(col: str) -> float | None:
            if col not in tab.colnames:
                return None
            v = row[col]
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return None
            return fv if np.isfinite(fv) else None

        return {"radius": val("rad"), "mass": val("mass"), "teff": val("Teff"), "tmag": val("Tmag")}
    except Exception:
        logger.warning("TIC lookup failed for %s", target_id, exc_info=True)
        return empty

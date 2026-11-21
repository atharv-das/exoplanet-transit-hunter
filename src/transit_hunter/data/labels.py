"""Training labels from the NASA Exoplanet Archive (TAP API) and the 3-class mapping.

Kepler (training): the KOI ``cumulative`` table.
    CONFIRMED / CANDIDATE                         -> PC (Planet Candidate)
    FALSE POSITIVE with koi_fpflag_ss == 1        -> EB (stellar eclipse; includes background EBs
                                                     that also carry the centroid-offset flag)
    FALSE POSITIVE otherwise                      -> FP (other false positive / systematic)
    anything else (e.g. NOT DISPOSITIONED)        -> dropped

TESS (validation only): the ``toi`` table, TFOPWG dispositions.
    CP / KP / PC / APC -> PC;  FP / FA -> FP.  TFOPWG has no separate EB value, so TESS EBs are
    not labeled here (see docs/DATA_CARD.md).

Column names were checked against the Exoplanet Archive column definitions at the time of
writing. Verify quickly with::

    curl "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+top+3+*+from+cumulative&format=csv"
"""

from __future__ import annotations

import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from transit_hunter.config import CLASSES, data_path, get_settings, load_data_config

logger = logging.getLogger(__name__)

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

KOI_COLUMNS: tuple[str, ...] = (
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_pdisposition",
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_period",
    "koi_time0bk",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_model_snr",
    "koi_srad",
    "koi_steff",
    "koi_kepmag",
)

TOI_COLUMNS: tuple[str, ...] = (
    "toi",
    "tid",
    "tfopwg_disp",
    "pl_orbper",
    "pl_tranmid",
    "pl_trandurh",
    "pl_trandep",
    "st_rad",
    "st_tmag",
)

#: Columns every labeled table produced by this module has (used by the dataset builder).
LABEL_COLUMNS: tuple[str, ...] = (
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


def tap_query(query: str, timeout_s: float = 120.0) -> pd.DataFrame:
    """Run a synchronous ADQL query against the Exoplanet Archive and return a DataFrame."""
    logger.info("TAP query: %s", query)
    resp = httpx.get(
        TAP_URL,
        params={"query": query, "format": "csv"},
        timeout=timeout_s,
        follow_redirects=True,
    )
    resp.raise_for_status()
    text = resp.text
    if text.lstrip().startswith("<"):
        raise RuntimeError(f"TAP returned an error document: {text[:500]}")
    return pd.read_csv(io.StringIO(text))


def _labels_dir() -> Path:
    d = data_path("raw") / "labels"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cached_table(name: str, query: str, refresh: bool) -> pd.DataFrame:
    path = _labels_dir() / f"{name}.csv"
    meta_path = path.with_suffix(".meta.json")
    if path.exists() and not refresh:
        return pd.read_csv(path)
    if get_settings().offline:
        raise FileNotFoundError(f"{path} missing and TH_OFFLINE=true")
    df = tap_query(query, timeout_s=float(load_data_config().download.timeout_s))
    df.to_csv(path, index=False)
    meta = {
        "table": name,
        "query": query,
        "url": TAP_URL,
        "rows": len(df),
        "queried_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("saved %d rows to %s", len(df), path)
    return df


def fetch_koi_table(refresh: bool = False) -> pd.DataFrame:
    """Download (or load the cached copy of) the Kepler KOI cumulative table."""
    table = load_data_config().labels.kepler_koi_table
    q = f"select {','.join(KOI_COLUMNS)} from {table}"
    return _cached_table(f"koi_{table}", q, refresh)


def fetch_toi_table(refresh: bool = False) -> pd.DataFrame:
    """Download (or load the cached copy of) the TESS TOI table."""
    table = load_data_config().labels.tess_toi_table
    q = f"select {','.join(TOI_COLUMNS)} from {table}"
    return _cached_table(f"toi_{table}", q, refresh)


def map_koi_labels(koi: pd.DataFrame) -> pd.DataFrame:
    """Apply the 3-class mapping to a KOI table and standardize columns.

    Returns:
        DataFrame with :data:`LABEL_COLUMNS` plus ``koi_*`` provenance columns. Rows with no
        usable disposition or ephemeris are dropped.
    """
    df = koi.copy()
    disp = df["koi_disposition"].astype(str).str.upper().str.strip()
    ss = pd.to_numeric(df["koi_fpflag_ss"], errors="coerce").fillna(0).astype(int)

    label = pd.Series(pd.NA, index=df.index, dtype="object")
    reason = pd.Series("", index=df.index, dtype="object")
    is_pc = disp.isin(["CONFIRMED", "CANDIDATE"])
    is_fp = disp == "FALSE POSITIVE"
    label[is_pc] = "PC"
    reason[is_pc] = "koi_disposition=" + disp[is_pc]
    label[is_fp & (ss == 1)] = "EB"
    reason[is_fp & (ss == 1)] = "FALSE POSITIVE & koi_fpflag_ss=1"
    label[is_fp & (ss != 1)] = "FP"
    reason[is_fp & (ss != 1)] = "FALSE POSITIVE & koi_fpflag_ss=0"

    out = pd.DataFrame(
        {
            "sample_id": df["kepoi_name"].astype(str),
            "star_id": "KIC "
            + pd.to_numeric(df["kepid"], errors="coerce").astype("Int64").astype(str),
            "mission": "Kepler",
            "label": label,
            "label_reason": reason,
            "period": pd.to_numeric(df["koi_period"], errors="coerce"),
            "t0": pd.to_numeric(df["koi_time0bk"], errors="coerce"),  # BKJD, same as lightkurve
            "duration": pd.to_numeric(df["koi_duration"], errors="coerce") / 24.0,  # hours -> days
            "catalog_depth_ppm": pd.to_numeric(df["koi_depth"], errors="coerce"),
            "stellar_radius": pd.to_numeric(df["koi_srad"], errors="coerce"),
        }
    )
    for col in ("koi_disposition", "koi_pdisposition", "koi_score", "kepler_name"):
        if col in df:
            out[col] = df[col]
    for col in ("koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"):
        if col in df:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    valid = (
        out["label"].notna()
        & np.isfinite(out["period"])
        & (out["period"] > 0)
        & np.isfinite(out["t0"])
        & np.isfinite(out["duration"])
        & (out["duration"] > 0)
        & out["star_id"].ne("KIC <NA>")
    )
    dropped = int((~valid).sum())
    if dropped:
        logger.info("dropped %d KOIs without disposition/ephemeris", dropped)
    out = out[valid].reset_index(drop=True)
    out["label_idx"] = out["label"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    return out


def map_toi_labels(toi: pd.DataFrame) -> pd.DataFrame:
    """Map TFOPWG dispositions to PC/FP and standardize columns (TESS validation set).

    Times are converted from BJD to BTJD (BJD - 2457000) to match lightkurve TESS products.
    """
    df = toi.copy()
    disp = df["tfopwg_disp"].astype(str).str.upper().str.strip()
    label = pd.Series(pd.NA, index=df.index, dtype="object")
    label[disp.isin(["CP", "KP", "PC", "APC"])] = "PC"
    label[disp.isin(["FP", "FA"])] = "FP"
    out = pd.DataFrame(
        {
            "sample_id": "TOI-" + df["toi"].astype(str),
            "star_id": "TIC "
            + pd.to_numeric(df["tid"], errors="coerce").astype("Int64").astype(str),
            "mission": "TESS",
            "label": label,
            "label_reason": "tfopwg_disp=" + disp,
            "period": pd.to_numeric(df["pl_orbper"], errors="coerce"),
            "t0": pd.to_numeric(df["pl_tranmid"], errors="coerce") - 2457000.0,
            "duration": pd.to_numeric(df["pl_trandurh"], errors="coerce") / 24.0,
            "catalog_depth_ppm": pd.to_numeric(df["pl_trandep"], errors="coerce"),
            "stellar_radius": pd.to_numeric(df["st_rad"], errors="coerce"),
            "tfopwg_disp": disp,
        }
    )
    valid = (
        out["label"].notna()
        & np.isfinite(out["period"])
        & (out["period"] > 0)
        & np.isfinite(out["t0"])
        & np.isfinite(out["duration"])
        & (out["duration"] > 0)
    )
    out = out[valid].reset_index(drop=True)
    out["label_idx"] = out["label"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    return out


def star_primary_label(labels: pd.DataFrame) -> pd.Series:
    """One label per star for stratified sampling/splitting: EB > FP > PC by priority."""
    priority = {"EB": 0, "FP": 1, "PC": 2}
    ranked = labels.assign(_p=labels["label"].map(priority)).sort_values("_p")
    return ranked.groupby("star_id")["label"].first()


def sample_stars(labels: pd.DataFrame, limit: int | None, seed: int) -> list[str]:
    """Pick up to ``limit`` stars, balanced across the three primary-label classes.

    Each class gets ``limit // 3`` stars; unused quota from small classes is filled from the
    others. Returns all stars when ``limit`` is None.
    """
    primary = star_primary_label(labels)
    if limit is None or limit >= len(primary):
        return sorted(primary.index.tolist())
    rng = np.random.default_rng(seed)
    pools = {c: rng.permutation(primary.index[primary == c].to_numpy()).tolist() for c in CLASSES}
    quota = limit // len(CLASSES)
    chosen: list[str] = []
    for c in CLASSES:
        chosen += pools[c][:quota]
        pools[c] = pools[c][quota:]
    leftovers = rng.permutation([s for c in CLASSES for s in pools[c]]).tolist()
    chosen += leftovers[: limit - len(chosen)]
    return sorted(chosen)


def label_summary(labels: pd.DataFrame) -> dict[str, object]:
    """Counts per class (signals and stars) for logs and the data card."""
    return {
        "signals": {c: int((labels["label"] == c).sum()) for c in CLASSES},
        "stars": int(labels["star_id"].nunique()),
        "stars_by_primary_label": star_primary_label(labels).value_counts().to_dict(),
    }


# --------------------------------------------------------------------------------------
# TESS: TOI table + TESS Eclipsing Binary catalog -> PC / EB / FP
# --------------------------------------------------------------------------------------

#: Accepted spellings for each column of the TESS EB catalog. The paper's own catalog uses
#: ``tic``/``period``/``bjd0``/``morph_coeff``/``pwidth``; VizieR may relabel them. Verify with
#: ``Vizier(row_limit=5).get_catalogs("J/ApJS/258/16")[0].colnames``.
EB_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "tic": ("TIC", "tic", "TICID", "tic_id", "ID"),
    "period": ("Per", "period", "Period", "P"),
    "t0": ("BJD0", "bjd0", "T0", "Epoch", "BJD"),
    "pwidth": ("Wp", "pwidth", "Pwidth", "W1", "wp"),
    "morph": ("Morph", "morph_coeff", "morph", "Mcoeff", "c"),
}


def _pick(columns: list[str], options: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for o in options:
        if o in columns:
            return o
        if o.lower() in lower:
            return lower[o.lower()]
    return None


def normalize_eb_catalog(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a TESS EB catalog with any of the known column spellings to a standard frame.

    Returns columns ``tic`` (int), ``period``, ``t0`` (BTJD), ``duration`` (days, from the primary
    eclipse width in phase when available, else an estimate), ``morph`` (NaN if absent).

    Raises:
        KeyError: if TIC, period or epoch columns cannot be identified (lists what exists).
    """
    cols = list(raw.columns)
    found = {k: _pick(cols, v) for k, v in EB_COLUMN_CANDIDATES.items()}
    missing = [k for k in ("tic", "period", "t0") if found[k] is None]
    if missing:
        raise KeyError(f"EB catalog columns {missing} not found; available columns: {cols}")
    out = pd.DataFrame(
        {
            "tic": pd.to_numeric(raw[found["tic"]], errors="coerce"),
            "period": pd.to_numeric(raw[found["period"]], errors="coerce"),
            "t0": pd.to_numeric(raw[found["t0"]], errors="coerce"),
        }
    )
    # The catalog epoch is BJD - 2457000 (= BTJD); convert if a full BJD slipped through.
    out.loc[out["t0"] > 2_400_000, "t0"] -= 2_457_000.0
    if found["pwidth"] is not None:
        width = pd.to_numeric(raw[found["pwidth"]], errors="coerce")
        out["duration"] = (width * out["period"]).where((width > 0) & (width < 0.5))
    else:
        out["duration"] = np.nan
    # Fallback duration: typical detached-EB eclipse length (~6 % of P^(1/3) days, 1-10 h).
    est = (0.07 * out["period"] ** (1 / 3)).clip(1 / 24, 10 / 24)
    out["duration"] = out["duration"].fillna(est).clip(lower=0.5 / 24, upper=0.25 * out["period"])
    out["morph"] = pd.to_numeric(raw[found["morph"]], errors="coerce") if found["morph"] else np.nan
    out = out.dropna(subset=["tic", "period", "t0"])
    out = out[out["period"] > 0]
    out["tic"] = out["tic"].astype(np.int64)
    return out.reset_index(drop=True)


TAPVIZIER_URLS = (
    "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync",
    "http://tapvizier.u-strasbg.fr/TAPVizieR/tap/sync",
)
#: CDS archive (raw files + ReadMe). Independent of the VizieR query servers.
CDS_ARCHIVE_URLS = (
    "https://cdsarc.cds.unistra.fr/ftp",
    "https://cdsarc.u-strasbg.fr/ftp",
)
#: VizieR ASU-TSV endpoints: CDS plus mirrors (IUCAA Pune, CfA Harvard, Cambridge UK).
VIZIER_ASU_TSV_URLS = (
    "https://vizier.iucaa.in/viz-bin/asu-tsv",
    "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv",
    "https://vizier.ast.cam.ac.uk/viz-bin/asu-tsv",
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
)


def parse_vizier_tsv(text: str) -> pd.DataFrame:
    """Parse VizieR's ASU-TSV output: ``#`` comments, a header row, a units row, a dashes row,
    then tab-separated data."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise ValueError("empty VizieR TSV response")
    header = [h.strip() for h in lines[0].split("\t")]
    rows = []
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.split("\t")]
        if all(set(c) <= {"-", " "} for c in cells):  # the dashes separator row
            continue
        rows.append(cells)
    # Drop the units row if it sits directly above the dashes row (no numeric cells).
    if rows and not any(_is_number(c) for c in rows[0]):
        rows = rows[1:]
    width = len(header)
    rows = [(r + [""] * width)[:width] for r in rows]
    return pd.DataFrame(rows, columns=header).replace("", np.nan)


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def parse_cds_files(readme: str, data: bytes, table_file: str) -> pd.DataFrame:
    """Parse a CDS fixed-width data file using its ReadMe (astropy's CDS reader)."""
    import tempfile
    from pathlib import Path as _Path

    from astropy.io import ascii as astro_ascii

    with tempfile.TemporaryDirectory() as tmp:
        d = _Path(tmp)
        (d / "ReadMe").write_text(readme)
        (d / table_file).write_bytes(data)
        tab = astro_ascii.read(str(d / table_file), format="cds", readme=str(d / "ReadMe"))
    df: pd.DataFrame = tab.to_pandas()
    return df


def _http_get(url: str, timeout: float, **params: str) -> httpx.Response:
    resp = httpx.get(url, params=params or None, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp


def _eb_from_cds_archive(catalog: str, table: str, timeout: float) -> pd.DataFrame:
    """Raw ``<table>.dat`` (or ``.dat.gz``) plus ``ReadMe`` from the CDS archive."""
    import gzip

    errors = []
    for base in CDS_ARCHIVE_URLS:
        try:
            readme = _http_get(f"{base}/{catalog}/ReadMe", timeout).text
            if "Byte-by-byte" not in readme:
                raise ValueError("ReadMe has no byte-by-byte description")
            fname = f"{table}.dat"
            try:
                data = _http_get(f"{base}/{catalog}/{fname}", timeout).content
            except httpx.HTTPStatusError:
                data = gzip.decompress(_http_get(f"{base}/{catalog}/{fname}.gz", timeout).content)
            return parse_cds_files(readme, data, fname)
        except Exception as exc:
            errors.append(f"{base}: {type(exc).__name__}: {exc}"[:200])
    raise RuntimeError("; ".join(errors))


def _eb_from_tapvizier(catalog: str, table: str, timeout: float) -> pd.DataFrame:
    query = f'SELECT * FROM "{catalog}/{table}"'
    errors = []
    for url in TAPVIZIER_URLS:
        try:
            resp = _http_get(
                url, min(timeout, 60.0), REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=query
            )
            if resp.text.lstrip().startswith("<"):
                raise RuntimeError(f"error document: {resp.text[:200]}")
            return pd.read_csv(io.StringIO(resp.text))
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}"[:200])
    raise RuntimeError("; ".join(errors))


def _eb_from_asu_tsv(catalog: str, table: str, timeout: float) -> pd.DataFrame:
    errors = []
    for url in VIZIER_ASU_TSV_URLS:
        try:
            resp = _http_get(
                url,
                min(timeout, 60.0),
                **{"-source": f"{catalog}/{table}", "-out.max": "unlimited", "-out.all": ""},
            )
            df = parse_vizier_tsv(resp.text)
            if len(df) < 10:
                raise ValueError(f"only {len(df)} rows returned")
            return df
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}"[:200])
    raise RuntimeError("; ".join(errors))


def _eb_from_astroquery(catalog: str, table: str, timeout: float) -> pd.DataFrame:
    from astroquery.vizier import Vizier

    errors = []
    for server in ("vizier.cds.unistra.fr", "vizier.cfa.harvard.edu", "vizier.iucaa.in"):
        try:
            v = Vizier(columns=["**"], row_limit=-1, timeout=timeout, vizier_server=server)
            tables = v.get_catalogs(catalog)
            if len(tables) == 0:
                raise RuntimeError("no tables returned")
            raw: pd.DataFrame = tables[0].to_pandas()
            return raw
        except Exception as exc:
            errors.append(f"{server}: {type(exc).__name__}: {exc}"[:200])
    raise RuntimeError("; ".join(errors))


def fetch_tess_eb_catalog(refresh: bool = False) -> pd.DataFrame:
    """Download (or load the cached copy of) the TESS EB catalog from CDS VizieR, normalized.

    Tries four access routes in order and uses the first that works:

    1. the CDS archive: raw fixed-width file + ReadMe, parsed with astropy's CDS reader
       (independent of the VizieR query servers, which CDS has been migrating);
    2. TAPVizieR (ADQL over HTTP);
    3. the ASU-TSV export on CDS and its mirrors (IUCAA, CfA, Cambridge);
    4. astroquery against several VizieR servers.

    Raises:
        RuntimeError: if every route fails (the message lists each error).
    """
    cfg = load_data_config()
    catalog, table = cfg.labels.tess_eb_catalog, cfg.labels.tess_eb_table
    path = _labels_dir() / "tess_eb_catalog.csv"
    if path.exists() and not refresh:
        return normalize_eb_catalog(pd.read_csv(path))
    # Manual route: ReadMe + <table>.dat downloaded by hand (e.g. in a browser) into
    # data/raw/labels/tess_eb/. Used before any network access.
    manual = _labels_dir() / "tess_eb"
    if (manual / "ReadMe").exists() and (manual / f"{table}.dat").exists():
        raw = parse_cds_files(
            (manual / "ReadMe").read_text(), (manual / f"{table}.dat").read_bytes(), f"{table}.dat"
        )
        normalized = normalize_eb_catalog(raw)
        raw.to_csv(path, index=False)
        logger.info("loaded %d TESS EBs from manually downloaded files in %s", len(raw), manual)
        return normalized
    if get_settings().offline:
        raise FileNotFoundError(f"{path} missing and TH_OFFLINE=true")
    timeout = float(cfg.download.timeout_s)
    errors: list[str] = []
    for name, fetch in (
        ("cds-archive", _eb_from_cds_archive),
        ("tapvizier", _eb_from_tapvizier),
        ("asu-tsv", _eb_from_asu_tsv),
        ("astroquery", _eb_from_astroquery),
    ):
        try:
            raw = fetch(catalog, table, timeout)
            normalized = normalize_eb_catalog(raw)  # validates the columns before caching
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}"[:300])
            logger.warning("TESS EB catalog via %s failed: %s", name, errors[-1])
            continue
        raw.to_csv(path, index=False)
        path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "table": "tess_eb_catalog",
                    "vizier_catalog": f"{catalog}/{table}",
                    "route": name,
                    "rows": len(raw),
                    "columns": list(raw.columns),
                    "queried_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                },
                indent=2,
            )
        )
        logger.info("saved %d TESS EBs via %s (columns: %s)", len(raw), name, list(raw.columns))
        return normalized
    raise RuntimeError(
        "could not download the TESS EB catalog:\n  "
        + "\n  ".join(errors)
        + f"\nWorkaround: download ReadMe and {table}.dat (or .dat.gz, then gunzip) from "
        + f"https://cdsarc.cds.unistra.fr/ftp/{catalog}/ in a browser into {manual}/"
    )


def _period_match(p1: float, p2: float, tol: float = 0.02) -> bool:
    """True if periods agree within ``tol`` or at a factor of 2 (half/double-period aliases)."""
    return any(abs(p1 * m / p2 - 1) < tol for m in (1.0, 2.0, 0.5))


def map_tess_labels(
    toi: pd.DataFrame,
    eb_catalog: pd.DataFrame | None,
    max_period_days: float = 13.0,
    extra_eb_limit: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Three-class TESS labels from TOI dispositions plus the TESS EB catalog.

    Rules:
        * TOI CP / KP / PC / APC                          -> PC
        * TOI FP / FA whose TIC is in the EB catalog at a matching period (or 2x / 0.5x) -> EB
        * other TOI FP / FA                               -> FP
        * EB-catalog stars that are not TOIs              -> EB (catalog ephemeris), up to
          ``extra_eb_limit`` (default: enough to match the PC count)

    Only periods <= ``max_period_days`` are kept, so a few sectors contain >= 2 transits.
    """
    base = map_toi_labels(toi)
    base = base[base["period"] <= max_period_days].copy()
    if eb_catalog is None or eb_catalog.empty:
        logger.warning("no TESS EB catalog: TESS gets PC/FP labels only")
        return base.reset_index(drop=True)

    ebs = eb_catalog[eb_catalog["period"] <= 2 * max_period_days]
    by_tic: dict[int, list[float]] = {}
    for tic, per in zip(ebs["tic"], ebs["period"], strict=True):
        by_tic.setdefault(int(tic), []).append(float(per))

    tics = base["star_id"].str.replace("TIC ", "", regex=False).astype(np.int64)
    is_eb = [
        lab == "FP" and any(_period_match(p, q) for q in by_tic.get(int(t), []))
        for lab, t, p in zip(base["label"], tics, base["period"], strict=True)
    ]
    is_eb_s = pd.Series(is_eb, index=base.index)
    base.loc[is_eb_s, "label"] = "EB"
    base.loc[is_eb_s, "label_reason"] = base.loc[is_eb_s, "label_reason"] + " & in TESS EB catalog"

    toi_tics = set(int(t) for t in tics)
    extra = ebs[~ebs["tic"].isin(toi_tics) & (ebs["period"] <= max_period_days)].drop_duplicates(
        "tic"
    )
    n_pc = int((base["label"] == "PC").sum())
    limit = (
        extra_eb_limit
        if extra_eb_limit is not None
        else max(n_pc - int(base["label"].eq("EB").sum()), 0)
    )
    if limit and len(extra):
        extra = extra.sample(n=min(limit, len(extra)), random_state=seed)
        extra_rows = pd.DataFrame(
            {
                "sample_id": "TEB-" + extra["tic"].astype(str),
                "star_id": "TIC " + extra["tic"].astype(str),
                "mission": "TESS",
                "label": "EB",
                "label_reason": "TESS EB catalog (Prsa+ 2022)",
                "period": extra["period"].to_numpy(),
                "t0": extra["t0"].to_numpy(),
                "duration": extra["duration"].to_numpy(),
                "catalog_depth_ppm": np.nan,
                "stellar_radius": np.nan,
            }
        )
        base = pd.concat([base, extra_rows], ignore_index=True)
    base["label_idx"] = base["label"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    logger.info("TESS labels: %s", base["label"].value_counts().to_dict())
    return base.reset_index(drop=True)

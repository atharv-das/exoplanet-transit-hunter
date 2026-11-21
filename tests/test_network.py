"""Live tests against MAST and the NASA Exoplanet Archive.

Skipped by default. Run on a machine with internet access:

    make test-network        # or: uv run pytest -m network -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transit_hunter.config import get_settings, load_data_config

pytestmark = pytest.mark.network


@pytest.fixture()
def online_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TH_OFFLINE", "false")
    get_settings.cache_clear()
    return tmp_path / "data"


def test_koi_table_columns_and_mapping(online_data_dir: Path) -> None:
    from transit_hunter.data.labels import (
        KOI_COLUMNS,
        fetch_koi_table,
        label_summary,
        map_koi_labels,
    )

    koi = fetch_koi_table(refresh=True)
    assert set(KOI_COLUMNS) <= set(koi.columns)
    assert len(koi) > 9000
    mapped = map_koi_labels(koi)
    summary = label_summary(mapped)
    print(summary)
    assert all(n > 500 for n in summary["signals"].values())  # type: ignore[union-attr]


def test_toi_table_columns(online_data_dir: Path) -> None:
    from transit_hunter.data.labels import TOI_COLUMNS, fetch_toi_table, map_toi_labels

    toi = fetch_toi_table(refresh=True)
    assert set(TOI_COLUMNS) <= set(toi.columns)
    assert len(map_toi_labels(toi)) > 1000


def test_resolve_and_fetch_kepler10_and_recover_kepler10b(online_data_dir: Path) -> None:
    from transit_hunter.data.fetch import fetch_lightcurve, resolve_target
    from transit_hunter.processing.clean import clean_lightcurve
    from transit_hunter.processing.detrend import detrend
    from transit_hunter.processing.search import bls_search

    info = resolve_target("Kepler-10")
    assert info.target_id == "KIC 11904151"
    assert info.mission == "Kepler"
    lc = fetch_lightcurve("Kepler-10", max_segments=4)
    assert lc.time.size > 10_000
    cfg = load_data_config()
    c = clean_lightcurve(lc.time, lc.flux, lc.flux_err, cfg.cleaning)
    d = detrend(c.time, c.flux, c.flux_err, cfg.detrend)
    cand = bls_search(d.time, d.flux, d.flux_err, cfg.search)
    assert cand.period == pytest.approx(0.8375, rel=0.01)  # Kepler-10 b


def test_fetch_tess_target(online_data_dir: Path) -> None:
    from transit_hunter.data.fetch import fetch_lightcurve

    lc = fetch_lightcurve("TIC 261136679", max_segments=1)  # Pi Mensae
    assert lc.mission == "TESS"
    assert lc.author in ("SPOC", "TESS-SPOC", "QLP")
    assert lc.time.size > 1000


def test_tic_stellar_params(online_data_dir: Path) -> None:
    from transit_hunter.data.fetch import fetch_stellar_params

    p = fetch_stellar_params("TIC 261136679")
    assert p["radius"] is not None and 0.8 < p["radius"] < 1.5


def test_tess_eb_catalog_columns_and_labels(online_data_dir: Path) -> None:
    from transit_hunter.data.labels import (
        fetch_tess_eb_catalog,
        fetch_toi_table,
        label_summary,
        map_tess_labels,
    )

    ebs = fetch_tess_eb_catalog(refresh=True)  # raises KeyError listing columns if names changed
    assert len(ebs) > 4000
    assert ebs["period"].between(0.05, 1000).mean() > 0.95
    assert ebs["t0"].between(1000, 4000).mean() > 0.95  # BTJD
    labels = map_tess_labels(fetch_toi_table(refresh=True), ebs)
    summary = label_summary(labels)
    print(summary)
    assert all(n > 100 for n in summary["signals"].values())  # type: ignore[union-attr]

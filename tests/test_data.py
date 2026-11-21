"""Label mapping, target parsing, star sampling, splits and the offline dataset build."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import write_fake_star
from transit_hunter.config import CLASSES, DataConfig
from transit_hunter.data import build
from transit_hunter.data.dataset import assign_star_splits, load_dataset
from transit_hunter.data.fetch import (
    LightCurveData,
    TargetNotFoundError,
    cache_path,
    canonical_id,
    fetch_lightcurve,
    parse_target,
)
from transit_hunter.data.labels import map_koi_labels, map_toi_labels, sample_stars

# ----------------------------------------------------------------- targets


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("KIC 11904151", ("KIC 11904151", ["Kepler"])),
        ("kplr011904151", ("KIC 11904151", ["Kepler"])),
        ("tic 307210830", ("TIC 307210830", ["TESS"])),
        ("11904151", ("KIC 11904151", ["Kepler"])),
        ("307210830", ("TIC 307210830", ["TESS"])),
        ("Kepler-10", ("Kepler-10", ["Kepler", "TESS"])),
        ("Pi Mensae", ("Pi Mensae", ["TESS", "Kepler"])),
    ],
)
def test_parse_target(query: str, expected: tuple[str, list[str]]) -> None:
    assert parse_target(query) == expected


def test_parse_target_mission_override() -> None:
    assert parse_target("123", mission="TESS") == ("TIC 123", ["TESS"])
    with pytest.raises(TargetNotFoundError):
        parse_target("   ")


def test_canonical_id() -> None:
    assert canonical_id("kplr011904151", "Kepler") == "KIC 11904151"
    assert canonical_id("307210830", "TESS") == "TIC 307210830"


def test_lightcurve_cache_roundtrip_and_offline(data_dir: Path) -> None:
    lc = LightCurveData(
        "KIC 42",
        "Kepler",
        np.arange(10.0),
        np.ones(10),
        np.full(10, 1e-3),
        "Kepler",
        1800.0,
        [1, 2],
    )
    p = cache_path("KIC 42", "Kepler")
    lc.save(p)
    back = fetch_lightcurve("KIC 42")
    assert back.target_id == "KIC 42" and back.segments == [1, 2]
    assert np.array_equal(back.time, lc.time)
    with pytest.raises(TargetNotFoundError, match="offline"):
        fetch_lightcurve("KIC 43")


# ----------------------------------------------------------------- labels


def _fake_koi() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kepid": [1, 1, 2, 3, 4, 5],
            "kepoi_name": [
                "K00001.01",
                "K00001.02",
                "K00002.01",
                "K00003.01",
                "K00004.01",
                "K00005.01",
            ],
            "kepler_name": ["Kepler-1 b", None, None, None, None, None],
            "koi_disposition": [
                "CONFIRMED",
                "CANDIDATE",
                "FALSE POSITIVE",
                "FALSE POSITIVE",
                "NOT DISPOSITIONED",
                "CONFIRMED",
            ],
            "koi_pdisposition": ["CANDIDATE"] * 6,
            "koi_score": [1.0] * 6,
            "koi_fpflag_nt": [0, 0, 0, 1, 0, 0],
            "koi_fpflag_ss": [0, 0, 1, 0, 0, 0],
            "koi_fpflag_co": [0, 0, 1, 0, 0, 0],
            "koi_fpflag_ec": [0] * 6,
            "koi_period": [3.5, 10.0, 2.2, 0.9, 5.0, np.nan],
            "koi_time0bk": [131.5, 140.0, 133.0, 132.0, 135.0, 136.0],
            "koi_duration": [2.9, 4.0, 3.6, 1.0, 2.0, 2.0],
            "koi_depth": [2000, 300, 50000, 100, 1000, 800],
            "koi_srad": [1.0, 1.0, 1.2, 0.9, 1.0, 1.0],
        }
    )


def test_koi_mapping_rules() -> None:
    out = map_koi_labels(_fake_koi())
    by_id = out.set_index("sample_id")
    assert by_id.loc["K00001.01", "label"] == "PC"
    assert by_id.loc["K00001.02", "label"] == "PC"
    assert by_id.loc["K00002.01", "label"] == "EB"  # ss + co flags -> still EB
    assert by_id.loc["K00003.01", "label"] == "FP"
    assert "K00004.01" not in by_id.index  # not dispositioned
    assert "K00005.01" not in by_id.index  # no period
    assert by_id.loc["K00001.01", "duration"] == pytest.approx(2.9 / 24)
    assert by_id.loc["K00001.01", "star_id"] == "KIC 1"
    assert list(by_id["label_idx"]) == [CLASSES.index(x) for x in by_id["label"]]


def test_toi_mapping_converts_bjd_to_btjd() -> None:
    toi = pd.DataFrame(
        {
            "toi": [101.01, 102.01, 103.01],
            "tid": [111, 222, 333],
            "tfopwg_disp": ["KP", "FP", "PC"],
            "pl_orbper": [1.4, 2.0, 3.0],
            "pl_tranmid": [2458325.3, 2458326.0, 2458330.0],
            "pl_trandurh": [2.0, 3.0, 1.5],
            "pl_trandep": [10000, 500, 800],
            "st_rad": [1.1, 0.9, 1.0],
            "st_tmag": [9.0, 10.0, 11.0],
        }
    )
    out = map_toi_labels(toi)
    assert list(out["label"]) == ["PC", "FP", "PC"]
    assert out.loc[0, "t0"] == pytest.approx(1325.3)
    assert out.loc[0, "star_id"] == "TIC 111"


def _many_labels(n_per_class: int = 30) -> pd.DataFrame:
    rows = []
    for ci, c in enumerate(CLASSES):
        for i in range(n_per_class):
            rows.append({"star_id": f"KIC {ci * 1000 + i}", "label": c, "sample_id": f"{c}{i}"})
    rows.append({"star_id": "KIC 0", "label": "PC", "sample_id": "extra"})  # multi-signal star
    return pd.DataFrame(rows)


def test_sample_stars_balanced() -> None:
    labels = _many_labels()
    chosen = sample_stars(labels, 30, seed=0)
    assert len(chosen) == 30
    primary = labels.drop_duplicates("star_id").set_index("star_id")["label"]
    counts = primary.loc[chosen].value_counts()
    assert all(counts[c] == 10 for c in CLASSES)
    assert len(sample_stars(labels, None, seed=0)) == labels["star_id"].nunique()


def test_star_splits_no_leakage(data_cfg: DataConfig) -> None:
    labels = _many_labels()
    split = assign_star_splits(labels, data_cfg.split)
    per_star = labels.assign(split=split).groupby("star_id")["split"].nunique()
    assert (per_star == 1).all()
    assert set(split) == {"train", "val", "test"}
    frac_train = (split == "train").mean()
    assert 0.7 < frac_train < 0.9


# ----------------------------------------------------------------- offline build


def test_offline_build_process_synth_finalize(data_dir: Path) -> None:
    rows = [write_fake_star(f"KIC {i}", CLASSES[i % 3], i) for i in range(12)]
    labels = pd.DataFrame(rows)
    build.labels_path("Kepler").parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(build.labels_path("Kepler"), index=False)

    n_real = build.stage_process(workers=1)
    assert n_real == 12
    n_syn = build.stage_synth(count=8, workers=1, per_star=2, seed=0)
    assert n_syn >= 6
    meta = build.stage_finalize()
    assert meta["n_examples"] == n_real + n_syn
    ds = load_dataset()
    assert ds.global_views.shape == (len(ds), 201)
    assert ds.local_views.shape == (len(ds), 61)
    assert np.isfinite(ds.global_views).all() and np.isfinite(ds.local_views).all()
    # Synthetic examples share their host star's split.
    per_star = ds.index.groupby("star_id")["split"].nunique()
    assert (per_star == 1).all()
    assert ds.index["is_synthetic"].sum() == n_syn
    test_real = ds.subset("test", real_only=True)
    assert not test_real.index["is_synthetic"].any()
    assert ds.features().shape[0] == len(ds)
    assert (data_dir / "processed" / "dataset_meta.json").exists()


def test_download_manifest_is_resumable(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels = pd.DataFrame([write_fake_star(f"KIC {i}", CLASSES[i % 3], i) for i in range(6)])
    build.labels_path("Kepler").parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(build.labels_path("Kepler"), index=False)
    calls: list[str] = []

    def fake_download(
        star_id: str, mission: str, purge: bool = False, max_segments: int | None = None
    ) -> dict[str, object]:
        calls.append(star_id)
        return {"star_id": star_id, "mission": mission, "status": "ok", "seconds": 0.0}

    monkeypatch.setattr(build, "download_one", fake_download)
    build.stage_download(limit=None, workers=1)
    assert len(calls) == 6
    build.stage_download(limit=None, workers=1)
    assert len(calls) == 6  # nothing re-downloaded
    assert len(build.read_manifest()) == 6


# ----------------------------------------------------------------- TESS labels + mixed training


def test_normalize_eb_catalog_accepts_paper_and_vizier_names() -> None:
    from transit_hunter.data.labels import normalize_eb_catalog

    vizier = pd.DataFrame(
        {"TIC": [5, 6], "Per": [2.0, 3.0], "BJD0": [1400.5, 2459001.0], "Wp": [0.05, 0.9]}
    )
    out = normalize_eb_catalog(vizier)
    assert list(out["tic"]) == [5, 6]
    assert out.loc[0, "duration"] == pytest.approx(0.1)  # width 0.05 x period 2 d
    assert out.loc[1, "t0"] == pytest.approx(2001.0)  # full BJD converted to BTJD
    assert 0 < out.loc[1, "duration"] <= 0.75  # implausible width replaced by an estimate
    paper = pd.DataFrame({"tic": [7], "period": [1.5], "bjd0": [1500.0], "morph_coeff": [0.4]})
    assert normalize_eb_catalog(paper).loc[0, "morph"] == pytest.approx(0.4)
    with pytest.raises(KeyError, match="available columns"):
        normalize_eb_catalog(pd.DataFrame({"foo": [1]}))


def test_map_tess_labels_three_classes() -> None:
    from transit_hunter.data.labels import map_tess_labels, normalize_eb_catalog

    toi = pd.DataFrame(
        {
            "toi": [1.01, 2.01, 3.01, 4.01, 5.01],
            "tid": [11, 22, 33, 44, 55],
            "tfopwg_disp": ["CP", "FP", "FP", "PC", "KP"],
            "pl_orbper": [3.0, 2.0, 5.0, 40.0, 1.2],
            "pl_tranmid": [2458400.0] * 5,
            "pl_trandurh": [2.0, 3.0, 2.0, 4.0, 1.5],
            "pl_trandep": [1000, 5e4, 300, 500, 8000],
            "st_rad": [1.0] * 5,
            "st_tmag": [9.0] * 5,
        }
    )
    ebs = normalize_eb_catalog(
        pd.DataFrame(
            {
                "TIC": [22, 90, 91, 92],
                "Per": [4.0, 1.5, 2.5, 30.0],
                "BJD0": [1400.0, 1500.0, 1600.0, 1700.0],
                "Wp": [0.05, 0.06, 0.04, 0.01],
            }
        )
    )
    out = map_tess_labels(toi, ebs, max_period_days=13.0, seed=0).set_index("sample_id")
    assert out.loc["TOI-1.01", "label"] == "PC"
    assert out.loc["TOI-2.01", "label"] == "EB"  # FP + EB catalog at 2x the TOI period
    assert out.loc["TOI-3.01", "label"] == "FP"
    assert "TOI-4.01" not in out.index  # 40 d: too few transits in a few sectors
    extra = out[out.index.str.startswith("TEB-")]
    assert len(extra) == 1  # PCs (2) minus matched EBs (1)
    assert extra["star_id"].iloc[0] in ("TIC 90", "TIC 91")  # 30 d EB excluded
    assert set(out["mission"]) == {"TESS"}
    assert list(out["label_idx"]) == [CLASSES.index(x) for x in out["label"]]


def test_mixed_mission_build_splits_both_missions(data_dir: Path) -> None:
    kepler = [write_fake_star(f"KIC {i}", CLASSES[i % 3], i) for i in range(30)]
    tess = [write_fake_star(f"TIC {500 + i}", CLASSES[i % 3], i, mission="TESS") for i in range(30)]
    for mission, rows in (("Kepler", kepler), ("TESS", tess)):
        build.labels_path(mission).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(build.labels_path(mission), index=False)
    assert build.stage_process(workers=1, mission="Kepler") == 30
    assert build.stage_process(workers=1, mission="TESS") == 30
    meta = build.stage_finalize()
    ds = load_dataset()
    splits = ds.index.groupby("mission")["split"].agg(lambda s: set(s))
    assert splits["TESS"] == {"train", "val", "test"}  # TESS is trained on, not held out
    assert splits["Kepler"] == {"train", "val", "test"}
    assert meta["train_missions"] == ["Kepler", "TESS"]
    assert meta["counts_by_mission"]["TESS"] == {"PC": 10, "EB": 10, "FP": 10}  # type: ignore[index]
    assert set(ds.index.loc[ds.index["mission"] == "TESS", "is_tess"]) == {1.0}
    assert set(ds.index.loc[ds.index["mission"] == "Kepler", "is_tess"]) == {0.0}
    assert len(ds.mission("TESS")) == 30


def test_parse_vizier_tsv() -> None:
    from transit_hunter.data.labels import normalize_eb_catalog, parse_vizier_tsv

    text = (
        "#\n# VizieR Astronomical Server\n#Title: TESS EBs\n"
        "TIC\tPer\tBJD0\tMorph\n"
        "\td\td\t\n"
        "----------\t-------\t---------\t-----\n"
        "5\t2.0\t1400.5\t0.3\n"
        "6\t3.5\t1500.25\t\n"
    )
    df = parse_vizier_tsv(text)
    assert list(df.columns) == ["TIC", "Per", "BJD0", "Morph"]
    assert len(df) == 2
    out = normalize_eb_catalog(df)
    assert list(out["tic"]) == [5, 6] and out.loc[1, "period"] == pytest.approx(3.5)


def test_eb_catalog_falls_back_between_routes(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transit_hunter.config import get_settings
    from transit_hunter.data import labels

    monkeypatch.setenv("TH_OFFLINE", "false")
    get_settings.cache_clear()
    calls: list[str] = []

    def broken(catalog: str, table: str, timeout: float) -> pd.DataFrame:
        calls.append("broken")
        raise RuntimeError("404 Not Found")

    def working(catalog: str, table: str, timeout: float) -> pd.DataFrame:
        calls.append("working")
        return pd.DataFrame({"TIC": [1, 2], "Per": [1.0, 2.0], "BJD0": [1400.0, 1401.0]})

    monkeypatch.setattr(labels, "_eb_from_cds_archive", broken)
    monkeypatch.setattr(labels, "_eb_from_tapvizier", broken)
    monkeypatch.setattr(labels, "_eb_from_asu_tsv", working)
    monkeypatch.setattr(labels, "_eb_from_astroquery", broken)
    out = labels.fetch_tess_eb_catalog(refresh=True)
    assert calls == ["broken", "broken", "working"] and len(out) == 2
    assert labels.fetch_tess_eb_catalog().shape[0] == 2  # second call is served from the cache

    monkeypatch.setattr(labels, "_eb_from_asu_tsv", broken)
    with pytest.raises(RuntimeError, match="could not download"):
        labels.fetch_tess_eb_catalog(refresh=True)


CDS_README = """\
J/ApJS/258/16       TESS Eclipsing Binary stars. I.          (Prsa+, 2022)
================================================================================
TESS Eclipsing Binary stars. I. Short-cadence observations of 4584 eclipsing
binaries in sectors 1-26.
    Prsa A., Kochoska A., Conroy K.E.
   <Astrophys. J. Suppl. Ser., 258, 16 (2022)>
   =2022ApJS..258...16P
================================================================================

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
tess-ebs.dat      30        2   The Eclipsing Binary stars (TESS-EBs) catalog
--------------------------------------------------------------------------------

Byte-by-byte Description of file: tess-ebs.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 10  I10   ---     TIC       TIC identifier
  12- 20  F9.6  d       Per       Orbital period
  22- 30  F9.4  d       BJD0      Primary eclipse epoch (BJD-2457000)
--------------------------------------------------------------------------------
"""


def test_parse_cds_archive_files() -> None:
    from transit_hunter.data.labels import normalize_eb_catalog, parse_cds_files

    data = (
        f"{1234567:10d} {2.123456:9.6f} {1400.1234:9.4f}\n{7654321:10d} {10.5:9.6f} {1500.5:9.4f}\n"
    ).encode()
    df = parse_cds_files(CDS_README, data, "tess-ebs.dat")
    out = normalize_eb_catalog(df)
    assert list(out["tic"]) == [1234567, 7654321]
    assert out.loc[0, "period"] == pytest.approx(2.123456)
    assert out.loc[1, "t0"] == pytest.approx(1500.5)


def test_eb_catalog_manual_files(data_dir: Path) -> None:
    from transit_hunter.data import labels

    manual = data_dir / "raw" / "labels" / "tess_eb"
    manual.mkdir(parents=True)
    (manual / "ReadMe").write_text(CDS_README)
    (manual / "tess-ebs.dat").write_bytes(f"{42:10d} {3.0:9.6f} {1450.0:9.4f}\n".encode())
    out = labels.fetch_tess_eb_catalog()  # offline (TH_OFFLINE=true) but files are local
    assert list(out["tic"]) == [42]

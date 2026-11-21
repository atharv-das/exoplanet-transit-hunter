"""Shared fixtures: an isolated data directory and synthetic light curves."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transit_hunter.config import (
    CLASSES,
    DataConfig,
    get_settings,
    load_cnn_config,
    load_data_config,
    load_gbm_config,
)
from transit_hunter.data.fetch import LightCurveData, cache_path
from transit_hunter.data.synthetic import (
    SyntheticLightCurve,
    eclipsing_binary,
    make_time_grid,
    synthetic_eb_lightcurve,
    synthetic_planet_lightcurve,
    synthetic_star,
    trapezoid_transit,
)


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point TH_DATA_DIR at a temp folder so tests never touch the real data/ tree."""
    d = tmp_path / "data"
    monkeypatch.setenv("TH_DATA_DIR", str(d))
    monkeypatch.setenv("TH_OFFLINE", "true")
    get_settings.cache_clear()
    yield d
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def data_cfg() -> DataConfig:
    return load_data_config()


@pytest.fixture(scope="session")
def planet_lc() -> SyntheticLightCurve:
    return synthetic_planet_lightcurve(
        period=3.5, t0=1.2, depth=0.002, duration=0.12, noise_ppm=300, seed=0
    )


@pytest.fixture(scope="session")
def eb_lc() -> SyntheticLightCurve:
    return synthetic_eb_lightcurve(seed=1)


def write_fake_star(
    star_id: str, kind: str, seed: int, mission: str = "Kepler"
) -> dict[str, object]:
    """Write a cached synthetic light curve and return its label row (PC / EB / FP).

    ``mission="TESS"`` gives a 54-day, 10-min-cadence curve (two sectors) instead of Kepler's
    90 days at 29.4 min.
    """
    if mission == "TESS":
        t = make_time_grid(54.0, cadence_minutes=10.0, segment_days=27.0, start=1400.0)
    else:
        t = make_time_grid(90.0, start=130.0)
    star, white = synthetic_star(t, noise_ppm=250, variability_ppm=1500, seed=seed)
    period, t0, dur = 2.5 + 0.3 * (seed % 5), float(t[0]) + 1.0, 0.12
    if kind == "PC":
        model = trapezoid_transit(t, period, t0, dur, 0.003)
    elif kind == "EB":
        model = eclipsing_binary(t, period, t0, dur, 0.05, 0.5)
    else:  # systematic-like: shallow dip at random-ish ephemeris
        model = trapezoid_transit(t, period, t0, dur, 0.0003)
    LightCurveData(
        star_id,
        mission,  # type: ignore[arg-type]
        t,
        star * model + white,
        np.full_like(t, 2.5e-4),
        "SPOC" if mission == "TESS" else "Kepler",
        600.0 if mission == "TESS" else 1800.0,
        [1, 2] if mission == "TESS" else [1, 2, 3],
    ).save(cache_path(star_id, mission))  # type: ignore[arg-type]
    prefix = "T" if mission == "TESS" else "K"
    return {
        "sample_id": f"{prefix}{seed:05d}.01",
        "star_id": star_id,
        "mission": mission,
        "label": kind,
        "label_idx": CLASSES.index(kind),
        "label_reason": "test",
        "period": period,
        "t0": t0,
        "duration": dur,
        "catalog_depth_ppm": 1000.0,
        "stellar_radius": 1.0,
    }


N_FAKE_STARS = 45


@pytest.fixture(scope="session")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[object, Path, Path]]:
    """Session-wide offline environment: 45 cached synthetic stars, a processed dataset, and a
    3-epoch CNN + GBM. Yields ``(ProcessedDataset, models_dir, root)``.

    Env vars stay set for the whole session so the pipeline and API tests see the same data.
    """
    from transit_hunter.data import build
    from transit_hunter.data.dataset import load_dataset
    from transit_hunter.models.train import train_cnn, train_gbm
    from transit_hunter.pipeline import reset_predictor

    root = tmp_path_factory.mktemp("trained_env")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TH_DATA_DIR", str(root / "data"))
        mp.setenv("TH_MODELS_DIR", str(root / "models"))
        mp.setenv("TH_CACHE_DIR", str(root / "data" / "cache"))
        mp.setenv("TH_OFFLINE", "true")
        mp.setenv("TH_MODEL_KIND", "cnn")
        get_settings.cache_clear()
        reset_predictor()
        rows = [write_fake_star(f"KIC {i}", CLASSES[i % 3], i) for i in range(N_FAKE_STARS)]
        build.labels_path("Kepler").parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(build.labels_path("Kepler"), index=False)
        build.stage_process(workers=1)
        build.stage_finalize()
        data = load_dataset()
        models = root / "models"
        cnn_res = train_cnn(data, load_cnn_config(), models / "cnn", epochs=3)
        gbm_res = train_gbm(data, load_gbm_config(), models / "gbm")
        assert cnn_res.checkpoint.exists() and gbm_res.checkpoint.exists()
        assert 1 <= cnn_res.best_epoch <= 3
        yield data, models, root
        reset_predictor()
    get_settings.cache_clear()


@pytest.fixture()
def trained_settings(trained: tuple[object, Path, Path]) -> Iterator[tuple[object, Path, Path]]:
    """Re-assert the trained environment for one test (other fixtures may have changed env)."""
    from transit_hunter.pipeline import reset_predictor

    _, models, root = trained
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TH_DATA_DIR", str(root / "data"))
        mp.setenv("TH_MODELS_DIR", str(models))
        mp.setenv("TH_CACHE_DIR", str(root / "data" / "cache"))
        mp.setenv("TH_OFFLINE", "true")
        mp.setenv("TH_MODEL_KIND", "cnn")
        get_settings.cache_clear()
        reset_predictor()
        yield trained
        reset_predictor()
    get_settings.cache_clear()

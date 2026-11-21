"""Config loading and validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from transit_hunter.config import (
    CLASSES,
    CONFIG_DIR,
    CNNConfig,
    DataConfig,
    GBMConfig,
    Settings,
    load_cnn_config,
    load_config,
    load_data_config,
    load_gbm_config,
)


def test_shipped_configs_validate() -> None:
    data = load_data_config()
    cnn = load_cnn_config()
    gbm = load_gbm_config()
    assert isinstance(data, DataConfig)
    assert isinstance(cnn, CNNConfig)
    assert isinstance(gbm, GBMConfig)
    assert tuple(data.labels.classes) == CLASSES
    assert data.views.global_bins == 201
    assert data.views.local_bins == 61
    assert cnn.num_classes == gbm.num_classes == len(CLASSES)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load((CONFIG_DIR / "data.yaml").read_text())
    raw["views"]["glbal_bins"] = 201  # typo
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError):
        load_config(DataConfig, bad)


def test_bls_durations_must_be_shorter_than_min_period(tmp_path: Path) -> None:
    raw = yaml.safe_load((CONFIG_DIR / "data.yaml").read_text())
    raw["search"]["durations_days"] = [0.1, 0.6]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError, match="shorter than period_min_days"):
        load_config(DataConfig, bad)


def test_split_fractions_sum_to_one(tmp_path: Path) -> None:
    raw = yaml.safe_load((CONFIG_DIR / "data.yaml").read_text())
    raw["split"]["test_frac"] = 0.3
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        load_config(DataConfig, bad)


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TH_API_PORT", "9123")
    monkeypatch.setenv("TH_OFFLINE", "true")
    s = Settings()
    assert s.api_port == 9123
    assert s.offline is True

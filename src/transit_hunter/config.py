"""Typed configuration for Transit Hunter.

Two layers:

* :class:`Settings` — runtime/environment settings (paths, API URL, timeouts), read from
  environment variables prefixed ``TH_`` and an optional ``.env`` file.
* YAML experiment configs in ``configs/`` — :class:`DataConfig`, :class:`CNNConfig`,
  :class:`GBMConfig`. They are validated with pydantic so a typo fails loudly at load time
  instead of silently halfway through a training run.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root (…/exoplanet-transit-hunter). Resolved from this file's location.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = REPO_ROOT / "configs"

#: Output class order of every model. Index i of a probability vector is CLASSES[i].
CLASSES: tuple[str, str, str] = ("PC", "EB", "FP")
CLASS_NAMES: dict[str, str] = {
    "PC": "Planet Candidate",
    "EB": "Eclipsing Binary",
    "FP": "Other False Positive / Systematic",
}

Mission = Literal["Kepler", "TESS"]


class _Strict(BaseModel):
    """Base model that rejects unknown keys, so YAML typos raise errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------------------
# Runtime settings
# --------------------------------------------------------------------------------------


class Settings(BaseSettings):
    """Environment-driven runtime settings. Override any field with ``TH_<FIELD>``."""

    model_config = SettingsConfigDict(
        env_prefix="TH_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = REPO_ROOT / "data"
    models_dir: Path = REPO_ROOT / "models"
    cache_dir: Path = REPO_ROOT / "data" / "cache"
    lightkurve_cache_dir: Path | None = None  # None -> lightkurve default (~/.lightkurve)
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_url: str = "http://localhost:8000"
    mast_timeout_s: int = 120
    offline: bool = False  # True -> never hit MAST; serve demo cache only
    log_level: str = "INFO"
    model_kind: Literal["cnn", "gbm", "auto"] = "auto"  # which model the pipeline serves

    def resolve(self, path: Path | str) -> Path:
        """Resolve a repo-relative path to an absolute path."""
        p = Path(path)
        return p if p.is_absolute() else REPO_ROOT / p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    return Settings()


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, using ``Settings.log_level`` by default."""
    lvl = (level or get_settings().log_level).upper()
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------------------
# Data config
# --------------------------------------------------------------------------------------


class PathsConfig(_Strict):
    raw: Path
    interim: Path
    processed: Path
    demo_cache: Path


class KeplerSource(_Strict):
    mission: Literal["Kepler"] = "Kepler"
    author: str = "Kepler"
    exptime: int = 1800
    flux_column: str = "pdcsap_flux"
    quality_bitmask: str = "default"


class TessSource(_Strict):
    mission: Literal["TESS"] = "TESS"
    author_priority: list[str] = Field(default_factory=lambda: ["SPOC", "TESS-SPOC", "QLP"])
    exptime_priority: list[int] = Field(default_factory=lambda: [120, 200, 600, 1800])
    flux_column: str = "pdcsap_flux"
    quality_bitmask: str = "default"


class SourcesConfig(_Strict):
    kepler: KeplerSource
    tess: TessSource


class LabelsConfig(_Strict):
    kepler_koi_table: str = "cumulative"
    tess_toi_table: str = "toi"
    tess_eb_catalog: str = "J/ApJS/258/16"
    tess_eb_table: str = "tess-ebs"
    train_missions: list[Mission] = Field(default_factory=lambda: list[Mission](["Kepler"]))
    tess_max_period_days: float = Field(13.0, gt=0)
    classes: list[str] = Field(default_factory=lambda: list(CLASSES))

    @field_validator("classes")
    @classmethod
    def _classes_match(cls, v: list[str]) -> list[str]:
        if tuple(v) != CLASSES:
            raise ValueError(f"labels.classes must be {list(CLASSES)} (model output order)")
        return v


class CleaningConfig(_Strict):
    sigma_upper: float = 4.0
    sigma_lower: float | None = None
    min_points: int = 500


class DetrendConfig(_Strict):
    method: Literal["biweight", "savgol"] = "biweight"
    window_length_days: float = Field(1.0, gt=0)
    edge_cutoff_days: float = Field(0.0, ge=0)
    break_tolerance_days: float = Field(0.5, gt=0)
    savgol_window_cadences: int = 101
    savgol_polyorder: int = 2

    @field_validator("savgol_window_cadences")
    @classmethod
    def _odd(cls, v: int) -> int:
        if v % 2 == 0 or v < 3:
            raise ValueError("savgol_window_cadences must be an odd integer >= 3")
        return v


class SearchConfig(_Strict):
    method: Literal["bls", "tls"] = "bls"
    engine: Literal["numba", "astropy"] = "numba"
    period_min_days: float = Field(0.5, gt=0)
    period_max_days: float = Field(100.0, gt=0)
    min_transits: int = Field(2, ge=1)
    durations_days: list[float] = Field(default_factory=lambda: [0.04, 0.08, 0.12, 0.16, 0.25])
    period_oversample: float = Field(3.0, gt=0)
    max_periods: int = Field(250_000, ge=100)
    phase_bins_per_duration: int = Field(4, ge=1)
    bin_minutes: float | None = 30.0
    objective: Literal["likelihood", "snr"] = "likelihood"
    snr_threshold: float = 7.1
    max_candidates: int = Field(3, ge=1)

    @model_validator(mode="after")
    def _check_grid(self) -> SearchConfig:
        if self.period_max_days <= self.period_min_days:
            raise ValueError("period_max_days must exceed period_min_days")
        if max(self.durations_days) >= self.period_min_days:
            raise ValueError("every BLS duration must be shorter than period_min_days")
        return self


class ViewsConfig(_Strict):
    global_bins: int = Field(201, ge=11)
    local_bins: int = Field(61, ge=11)
    local_half_width_durations: float = Field(2.0, gt=0)
    local_bin_width_durations: float = Field(0.16, gt=0)


class SplitConfig(_Strict):
    seed: int = 42
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    group_by: Literal["star"] = "star"

    @model_validator(mode="after")
    def _sum_to_one(self) -> SplitConfig:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")
        return self


class DownloadConfig(_Strict):
    limit: int | None = 800
    tess_limit: int | None = 400
    tess_max_sectors: int | None = Field(4, ge=1)
    workers: int = Field(8, ge=1)
    timeout_s: int = 120
    retries: int = Field(3, ge=0)
    backoff_s: float = 5.0


class DataConfig(_Strict):
    """Top-level schema of ``configs/data.yaml``."""

    version: int = 1
    paths: PathsConfig
    sources: SourcesConfig
    labels: LabelsConfig
    cleaning: CleaningConfig
    detrend: DetrendConfig
    search: SearchConfig
    views: ViewsConfig
    split: SplitConfig
    download: DownloadConfig


# --------------------------------------------------------------------------------------
# Model configs
# --------------------------------------------------------------------------------------


class BranchConfig(_Strict):
    channels: list[int]
    kernel_size: int = 5
    convs_per_block: int = 2
    pool_size: int = 5
    pool_stride: int = 2


class HeadConfig(_Strict):
    hidden: list[int] = Field(default_factory=lambda: [256, 128])
    dropout: float = Field(0.3, ge=0, lt=1)


class AugmentConfig(_Strict):
    flip_prob: float = Field(0.5, ge=0, le=1)
    noise_std: float = Field(0.02, ge=0)
    max_shift_bins_global: int = Field(3, ge=0)
    max_shift_bins_local: int = Field(1, ge=0)


class CNNTrainConfig(_Strict):
    batch_size: int = 64
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss: Literal["weighted_ce", "focal"] = "weighted_ce"
    focal_gamma: float = 2.0
    early_stopping_metric: Literal["val_macro_pr_auc", "val_loss"] = "val_macro_pr_auc"
    patience: int = 8
    num_workers: int = 0
    device: str = "cpu"


class CNNConfig(_Strict):
    """Top-level schema of ``configs/model_cnn.yaml``."""

    seed: int = 42
    num_classes: int = 3
    global_branch: BranchConfig
    local_branch: BranchConfig
    scalar_features: list[str]
    head: HeadConfig
    augment: AugmentConfig
    train: CNNTrainConfig
    output_dir: Path = Path("models/cnn")


class GBMConfig(_Strict):
    """Top-level schema of ``configs/model_gbm.yaml``."""

    seed: int = 42
    num_classes: int = 3
    use_scalar_features: bool = True
    use_view_summaries: bool = True
    params: dict[str, Any]
    early_stopping_rounds: int = 50
    output_dir: Path = Path("models/gbm")


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------

_M = TypeVar("_M", bound=BaseModel)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_config(model: type[_M], path: Path | str) -> _M:
    """Load and validate a YAML file into the given pydantic model class.

    Args:
        model: Pydantic model class to validate against.
        path: YAML path; relative paths resolve against the repo root.

    Returns:
        The validated, frozen config instance.
    """
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return model.model_validate(_load_yaml(p))


def load_data_config(path: Path | str = CONFIG_DIR / "data.yaml") -> DataConfig:
    """Load ``configs/data.yaml`` (or another path) as a :class:`DataConfig`."""
    return load_config(DataConfig, path)


def load_cnn_config(path: Path | str = CONFIG_DIR / "model_cnn.yaml") -> CNNConfig:
    """Load ``configs/model_cnn.yaml`` (or another path) as a :class:`CNNConfig`."""
    return load_config(CNNConfig, path)


def load_gbm_config(path: Path | str = CONFIG_DIR / "model_gbm.yaml") -> GBMConfig:
    """Load ``configs/model_gbm.yaml`` (or another path) as a :class:`GBMConfig`."""
    return load_config(GBMConfig, path)


def data_path(kind: Literal["raw", "interim", "processed", "demo_cache"]) -> Path:
    """Absolute path of a data directory, honoring ``TH_DATA_DIR`` and creating it if needed.

    If ``TH_DATA_DIR`` is set, the ``data/`` prefix of the YAML path is replaced with it, so
    the whole dataset can live on an external drive.
    """
    cfg = load_data_config()
    settings = get_settings()
    rel = Path(getattr(cfg.paths, kind))
    default_root = REPO_ROOT / "data"
    if rel.is_absolute():
        out = rel
    elif rel.parts and rel.parts[0] == "data":
        out = settings.data_dir.joinpath(*rel.parts[1:])
    else:
        out = default_root / rel
    out.mkdir(parents=True, exist_ok=True)
    return out

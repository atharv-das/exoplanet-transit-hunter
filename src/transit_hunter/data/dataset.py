"""Processed-dataset storage and star-level train/val/test splits.

On-disk layout (``data/processed/``)::

    parts/<name>.npz        global_views, local_views (float32) for one build stage
    parts/<name>.parquet    one row per example: ids, label, ephemeris, every scalar feature
    dataset.npz             merged arrays, row-aligned with index.parquet
    index.parquet           merged table + ``split`` column
    dataset_meta.json       counts, config snapshot, label-query dates, build time

Missions listed in ``labels.train_missions`` are split per mission; any other mission gets its
own held-out split (e.g. ``"tess"``) for domain-shift evaluation only.

Splits are assigned per *star* (all signals of one star land in the same split), stratified by
each star's primary label. Synthetic examples carry their host star's ID, so they share its split.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from transit_hunter.config import CLASSES, AugmentConfig, DataConfig, SplitConfig, data_path
from transit_hunter.data.labels import star_primary_label
from transit_hunter.processing.features import SCALAR_FEATURES

logger = logging.getLogger(__name__)

SPLITS: tuple[str, ...] = ("train", "val", "test", "tess")


@dataclass
class ProcessedDataset:
    """In-memory view of ``data/processed``; arrays are row-aligned with ``index``."""

    index: pd.DataFrame
    global_views: NDArray[np.float32]
    local_views: NDArray[np.float32]

    def subset(self, split: str, real_only: bool = False) -> ProcessedDataset:
        """Rows of one split (optionally excluding synthetic examples)."""
        mask = (self.index["split"] == split).to_numpy()
        if real_only and "is_synthetic" in self.index:
            mask &= ~self.index["is_synthetic"].to_numpy(dtype=bool)
        return ProcessedDataset(
            self.index[mask].reset_index(drop=True),
            self.global_views[mask],
            self.local_views[mask],
        )

    def mission(self, mission: str) -> ProcessedDataset:
        """Rows of one mission ("Kepler" or "TESS")."""
        mask = (self.index["mission"] == mission).to_numpy()
        return ProcessedDataset(
            self.index[mask].reset_index(drop=True), self.global_views[mask], self.local_views[mask]
        )

    def features(self, names: tuple[str, ...] | list[str] = SCALAR_FEATURES) -> NDArray[np.float64]:
        """Scalar feature matrix (NaN where unavailable)."""
        cols = [
            self.index[n] if n in self.index else pd.Series(np.nan, index=self.index.index)
            for n in names
        ]
        return np.column_stack([c.to_numpy(dtype=float) for c in cols])

    @property
    def labels(self) -> NDArray[np.int64]:
        """Integer labels in :data:`transit_hunter.config.CLASSES` order."""
        return self.index["label_idx"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)


def parts_dir() -> Path:
    """Directory holding per-stage dataset parts."""
    d = data_path("processed") / "parts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_part(
    name: str,
    rows: pd.DataFrame,
    global_views: NDArray[np.float32],
    local_views: NDArray[np.float32],
) -> Path:
    """Write one stage's examples (e.g. ``real`` or ``synth``) to ``processed/parts``."""
    if not (len(rows) == len(global_views) == len(local_views)):
        raise ValueError("rows and view arrays must have the same length")
    d = parts_dir()
    np.savez_compressed(d / f"{name}.npz", global_views=global_views, local_views=local_views)
    rows.reset_index(drop=True).to_parquet(d / f"{name}.parquet", index=False)
    logger.info("saved part %r with %d examples", name, len(rows))
    return d / f"{name}.parquet"


def assign_star_splits(index: pd.DataFrame, cfg: SplitConfig) -> pd.Series:
    """Assign train/val/test per star, stratified by the star's primary label.

    Args:
        index: Table with ``star_id`` and ``label`` columns.
        cfg: Split fractions and seed.

    Returns:
        Series aligned with ``index`` holding ``"train"``, ``"val"`` or ``"test"``.
    """
    primary = star_primary_label(index)
    rng = np.random.default_rng(cfg.seed)
    star_split: dict[str, str] = {}
    for cls in CLASSES:
        stars = np.array(sorted(primary.index[primary == cls]))
        rng.shuffle(stars)
        n = len(stars)
        n_val = round(cfg.val_frac * n)
        n_test = round(cfg.test_frac * n)
        if n >= 3:
            n_val, n_test = max(n_val, 1), max(n_test, 1)
        for s in stars[:n_test]:
            star_split[s] = "test"
        for s in stars[n_test : n_test + n_val]:
            star_split[s] = "val"
        for s in stars[n_test + n_val :]:
            star_split[s] = "train"
    return index["star_id"].map(star_split).fillna("train")


def _config_hash(cfg: DataConfig) -> str:
    blob = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def finalize_dataset(cfg: DataConfig, parts: list[str] | None = None) -> dict[str, object]:
    """Merge parts, assign star-level splits, write ``dataset.npz`` + ``index.parquet``.

    Args:
        cfg: Data config (split settings + snapshot stored in the metadata).
        parts: Part names to merge; defaults to every part on disk.

    Returns:
        The metadata dictionary written to ``dataset_meta.json``.
    """
    d = parts_dir()
    names = parts or sorted(p.stem for p in d.glob("*.parquet"))
    if not names:
        raise FileNotFoundError(f"no dataset parts in {d}; run `make data` first")
    frames, gvs, lvs = [], [], []
    for name in names:
        frames.append(pd.read_parquet(d / f"{name}.parquet"))
        with np.load(d / f"{name}.npz") as z:
            gvs.append(z["global_views"])
            lvs.append(z["local_views"])
    index = pd.concat(frames, ignore_index=True)
    if "is_synthetic" not in index:
        index["is_synthetic"] = False
    index["is_synthetic"] = index["is_synthetic"].fillna(False).astype(bool)
    # Splits come from real signals only, so adding synthetic data never reshuffles the real
    # test set; synthetic rows then inherit their host star's split.
    # Each training mission is split separately (stratified by class), so val and test always
    # contain both Kepler and TESS stars. A mission not listed in labels.train_missions is a
    # held-out domain-shift check and gets its own split name (e.g. "tess").
    real = index[~index["is_synthetic"]]
    star_split: dict[str, str] = {}
    for mission, group in real.groupby("mission"):
        if mission in cfg.labels.train_missions:
            star_split.update(
                zip(group["star_id"], assign_star_splits(group, cfg.split), strict=True)
            )
        else:
            star_split.update(dict.fromkeys(group["star_id"], str(mission).lower()))
    index["split"] = index["star_id"].map(star_split).fillna("train")

    out_dir = data_path("processed")
    np.savez_compressed(
        out_dir / "dataset.npz",
        global_views=np.concatenate(gvs).astype(np.float32),
        local_views=np.concatenate(lvs).astype(np.float32),
    )
    index.to_parquet(out_dir / "index.parquet", index=False)

    label_meta: dict[str, object] = {}
    for meta_file in sorted((data_path("raw") / "labels").glob("*.meta.json")):
        label_meta[meta_file.stem] = json.loads(meta_file.read_text())

    by_mission = {
        str(m): {c: int(((index["mission"] == m) & (index["label"] == c)).sum()) for c in CLASSES}
        for m in sorted(index["mission"].unique())
    }
    counts = {
        split: {c: int(((index["split"] == split) & (index["label"] == c)).sum()) for c in CLASSES}
        for split in SPLITS
    }
    meta: dict[str, object] = {
        "built_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "parts": names,
        "n_examples": len(index),
        "n_synthetic": int(index["is_synthetic"].sum()),
        "n_stars": int(index["star_id"].nunique()),
        "counts": counts,
        "counts_by_mission": by_mission,
        "train_missions": [m for m in cfg.labels.train_missions if m in by_mission],
        "data_config_hash": _config_hash(cfg),
        "data_config": cfg.model_dump(mode="json"),
        "label_sources": label_meta,
        "scalar_features": list(SCALAR_FEATURES),
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    _check_no_star_leakage(index)
    logger.info("dataset: %d examples from %d stars; %s", len(index), meta["n_stars"], counts)
    return meta


def _check_no_star_leakage(index: pd.DataFrame) -> None:
    per_star = index.groupby("star_id")["split"].nunique()
    leaked = per_star[per_star > 1]
    if len(leaked):
        raise AssertionError(f"stars in multiple splits: {leaked.index[:5].tolist()}")


def load_dataset(processed_dir: Path | None = None) -> ProcessedDataset:
    """Load the merged dataset written by :func:`finalize_dataset`."""
    d = processed_dir or data_path("processed")
    index = pd.read_parquet(d / "index.parquet")
    with np.load(d / "dataset.npz") as z:
        gv = z["global_views"].astype(np.float32)
        lv = z["local_views"].astype(np.float32)
    if len(index) != len(gv):
        raise ValueError("index.parquet and dataset.npz are out of sync; re-run finalize")
    return ProcessedDataset(index, gv, lv)


# --------------------------------------------------------------------------------------
# PyTorch dataset
# --------------------------------------------------------------------------------------


def augment_views(
    global_view: NDArray[np.float32],
    local_view: NDArray[np.float32],
    rng: np.random.Generator,
    flip_prob: float,
    noise_std: float,
    max_shift_global: int,
    max_shift_local: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Random time reflection, phase shift and Gaussian noise (applied to both views).

    * Reflection reverses both views together (a transit is time-symmetric; systematics need
      not be, so this is a mild regularizer, as in AstroNet).
    * The global view is periodic, so it shifts with wrap-around. The local view shifts with
      edge padding, which models small epoch errors from the BLS.
    """
    g, loc = global_view.copy(), local_view.copy()
    if rng.uniform() < flip_prob:
        g, loc = g[::-1].copy(), loc[::-1].copy()
    if max_shift_global:
        g = np.roll(g, int(rng.integers(-max_shift_global, max_shift_global + 1)))
    if max_shift_local:
        s = int(rng.integers(-max_shift_local, max_shift_local + 1))
        if s > 0:
            loc = np.concatenate([np.full(s, loc[0]), loc[:-s]])
        elif s < 0:
            loc = np.concatenate([loc[-s:], np.full(-s, loc[-1])])
    if noise_std > 0:
        g = (g + rng.normal(0, noise_std, g.shape)).astype(np.float32)
        loc = (loc + rng.normal(0, noise_std, loc.shape)).astype(np.float32)
    return g.astype(np.float32), loc.astype(np.float32)


class TransitViewsDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """``(global_view, local_view, scalars, label)`` tensors for the CNN.

    Args:
        global_views: ``(n, global_bins)`` array.
        local_views: ``(n, local_bins)`` array.
        scalars: ``(n, k)`` *already standardized* scalar features.
        labels: ``(n,)`` integer labels.
        augment: Apply :func:`augment_views` on the fly (training split only).
        augment_cfg: Augmentation settings (required when ``augment`` is True).
        seed: Base seed for augmentation randomness.
    """

    def __init__(
        self,
        global_views: NDArray[np.float32],
        local_views: NDArray[np.float32],
        scalars: NDArray[np.float32],
        labels: NDArray[np.int64],
        augment: bool = False,
        augment_cfg: AugmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        if augment and augment_cfg is None:
            raise ValueError("augment_cfg is required when augment=True")
        self.g = np.asarray(global_views, dtype=np.float32)
        self.loc = np.asarray(local_views, dtype=np.float32)
        self.s = np.asarray(scalars, dtype=np.float32)
        self.y = np.asarray(labels, dtype=np.int64)
        self.augment = augment
        self.cfg = augment_cfg
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.y.size)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g, loc = self.g[i], self.loc[i]
        if self.augment and self.cfg is not None:
            g, loc = augment_views(
                g,
                loc,
                self.rng,
                self.cfg.flip_prob,
                self.cfg.noise_std,
                self.cfg.max_shift_bins_global,
                self.cfg.max_shift_bins_local,
            )
        return (
            torch.from_numpy(np.ascontiguousarray(g)),
            torch.from_numpy(np.ascontiguousarray(loc)),
            torch.from_numpy(self.s[i]),
            torch.tensor(self.y[i]),
        )

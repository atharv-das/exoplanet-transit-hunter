"""Training loops for the CNN and the GBM baseline.

CNN: AdamW, class-weighted cross-entropy (or focal loss), on-the-fly augmentation, early stopping
on validation macro PR-AUC, best checkpoint saved with its config, scaler and metrics.

GBM: LightGBM with balanced class weights and early stopping on validation log-loss.
"""

from __future__ import annotations

import json
import logging
import random
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from transit_hunter.config import CLASSES, REPO_ROOT, CNNConfig, GBMConfig
from transit_hunter.data.dataset import ProcessedDataset, TransitViewsDataset
from transit_hunter.models.cnn import DualBranchCNN, FeatureScaler, count_parameters
from transit_hunter.models.evaluate import evaluate_predictions, macro_pr_auc
from transit_hunter.models.gbm import GBMModel, feature_matrix

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
CNN_CHECKPOINT = "best.pt"
GBM_CHECKPOINT = "model.joblib"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds third-party code that uses the legacy global RNG
    torch.manual_seed(seed)


def class_weights(y: NDArray[np.int64], n_classes: int = len(CLASSES)) -> NDArray[np.float64]:
    """Inverse-frequency weights normalized to mean 1 (absent classes get weight 1)."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 1.0)
    return w / w.mean()


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017) with optional per-class weights."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1 - p_t) ** self.gamma) * logp_t
        weight = self.weight
        if isinstance(weight, torch.Tensor):
            loss = loss * weight[target]
        return loss.mean()


def resolve_dir(path: Path) -> Path:
    """Repo-relative output directory, created if missing."""
    p = path if path.is_absolute() else REPO_ROOT / path
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------------------
# CNN
# --------------------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Outcome of one training run."""

    kind: str
    checkpoint: Path
    best_epoch: int
    val_metrics: dict[str, Any]
    history: list[dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0


def _predict_cnn(
    model: DualBranchCNN, loader: DataLoader[Any]
) -> tuple[FloatArray, NDArray[np.int64], float]:
    model.eval()
    probs, ys, losses = [], [], []
    with torch.no_grad():
        for g, loc, s, y in loader:
            logits = model(g, loc, s)
            losses.append(float(F.cross_entropy(logits, y, reduction="sum")))
            probs.append(torch.softmax(logits, dim=1).numpy())
            ys.append(y.numpy())
    p = np.concatenate(probs) if probs else np.zeros((0, len(CLASSES)))
    yy = np.concatenate(ys) if ys else np.zeros(0, dtype=np.int64)
    return p.astype(float), yy.astype(np.int64), sum(losses) / max(len(yy), 1)


def train_cnn(
    data: ProcessedDataset,
    cfg: CNNConfig,
    output_dir: Path | None = None,
    epochs: int | None = None,
    data_meta: dict[str, Any] | None = None,
) -> TrainResult:
    """Train the dual-branch CNN and save the best checkpoint.

    Args:
        data: Full processed dataset (uses its ``train`` and ``val`` splits).
        cfg: CNN config.
        output_dir: Where to write ``best.pt`` + ``metrics.json`` (default ``cfg.output_dir``).
        epochs: Override ``cfg.train.epochs`` (handy for smoke tests).
        data_meta: Dataset metadata stored in the checkpoint for provenance.

    Returns:
        A :class:`TrainResult` with the best epoch's validation metrics.
    """
    start = _time.perf_counter()
    set_seed(cfg.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = resolve_dir(output_dir or cfg.output_dir)
    train, val = data.subset("train"), data.subset("val")
    if len(train) == 0:
        raise ValueError("empty training split")
    names = list(cfg.scalar_features)
    scaler = FeatureScaler.fit(train.features(names), names)

    def make_ds(split: ProcessedDataset, augment: bool) -> TransitViewsDataset:
        return TransitViewsDataset(
            split.global_views,
            split.local_views,
            scaler.transform(split.features(names)),
            split.labels,
            augment=augment,
            augment_cfg=cfg.augment,
            seed=cfg.seed,
        )

    tc = cfg.train
    gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        make_ds(train, True),
        batch_size=tc.batch_size,
        shuffle=True,
        generator=gen,
        num_workers=tc.num_workers,
    )
    eval_train_loader = DataLoader(make_ds(train, False), batch_size=256)
    val_loader = DataLoader(make_ds(val, False), batch_size=256) if len(val) else None

    model = DualBranchCNN(cfg, data.global_views.shape[1], data.local_views.shape[1], len(names))
    logger.info(
        "CNN with %d parameters; train=%d val=%d", count_parameters(model), len(train), len(val)
    )
    weights = torch.tensor(class_weights(train.labels), dtype=torch.float32)
    criterion: nn.Module = (
        FocalLoss(tc.focal_gamma, weights)
        if tc.loss == "focal"
        else nn.CrossEntropyLoss(weight=weights)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    n_epochs = epochs or tc.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_score, best_epoch, bad = -np.inf, 0, 0
    history: list[dict[str, float]] = []
    ckpt_path = out / CNN_CHECKPOINT
    best_val: dict[str, Any] = {}
    for epoch in range(1, n_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for g, loc, s, y in train_loader:
            opt.zero_grad()
            loss = criterion(model(g, loc, s), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        sched.step()
        row: dict[str, float] = {"epoch": epoch, "train_loss": total / max(n, 1)}
        if val_loader is not None:
            vp, vy, vloss = _predict_cnn(model, val_loader)
            row["val_loss"] = vloss
            row["val_macro_pr_auc"] = macro_pr_auc(vy, vp)
            score = (
                row["val_macro_pr_auc"]
                if tc.early_stopping_metric == "val_macro_pr_auc"
                else -vloss
            )
            if not np.isfinite(score):
                score = -vloss
        else:
            tp, ty, _ = _predict_cnn(model, eval_train_loader)
            score = macro_pr_auc(ty, tp)
        history.append(row)
        logger.info(
            "epoch %3d  %s",
            epoch,
            "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"),
        )
        if score > best_score + 1e-4:
            best_score, best_epoch, bad = score, epoch, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "cnn_config": cfg.model_dump(mode="json"),
                    "scaler": scaler.to_dict(),
                    "global_bins": int(data.global_views.shape[1]),
                    "local_bins": int(data.local_views.shape[1]),
                    "classes": list(CLASSES),
                    "epoch": epoch,
                    "data_meta": {
                        k: v
                        for k, v in (data_meta or {}).items()
                        if k
                        in (
                            "built_at_utc",
                            "data_config_hash",
                            "n_examples",
                            "counts",
                            "counts_by_mission",
                            "train_missions",
                        )
                    },
                },
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= tc.patience:
                logger.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
                break

    if val_loader is not None:
        best = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(best["state_dict"])
        vp, vy, _ = _predict_cnn(model, val_loader)
        best_val = evaluate_predictions(vy, vp)
    result = TrainResult(
        "cnn", ckpt_path, best_epoch, best_val, history, _time.perf_counter() - start
    )
    (out / "train_history.json").write_text(json.dumps(history, indent=2))
    return result


# --------------------------------------------------------------------------------------
# GBM
# --------------------------------------------------------------------------------------


def train_gbm(
    data: ProcessedDataset,
    cfg: GBMConfig,
    output_dir: Path | None = None,
    data_meta: dict[str, Any] | None = None,
) -> TrainResult:
    """Train the LightGBM baseline and save it.

    Args:
        data: Processed dataset (uses ``train`` and ``val`` splits).
        cfg: GBM config.
        output_dir: Output directory (default ``cfg.output_dir``).
        data_meta: Dataset metadata stored with the model (missions it was trained on).
    """
    start = _time.perf_counter()
    out = resolve_dir(output_dir or cfg.output_dir)
    train, val = data.subset("train"), data.subset("val")
    X, names = feature_matrix(train.features(), train.global_views, train.local_views, cfg)
    Xv, _ = (
        feature_matrix(val.features(), val.global_views, val.local_views, cfg)
        if len(val)
        else (None, names)
    )
    model = GBMModel(cfg).fit(X, train.labels, Xv, val.labels if len(val) else None, names)
    model.trained_missions = sorted(set(train.index["mission"])) or ["Kepler"]
    model.data_meta = {
        k: v
        for k, v in (data_meta or {}).items()
        if k in ("built_at_utc", "data_config_hash", "n_examples")
    }
    path = out / GBM_CHECKPOINT
    model.save(path)
    val_metrics = (
        evaluate_predictions(val.labels, model.predict_proba(Xv)) if Xv is not None else {}
    )
    importance = dict(sorted(model.feature_importance().items(), key=lambda kv: -kv[1]))
    (out / "feature_importance.json").write_text(json.dumps(importance, indent=2))
    return TrainResult(
        "gbm", path, int(model.best_iteration or 0), val_metrics, [], _time.perf_counter() - start
    )

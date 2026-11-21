"""Inference wrappers with one interface for both models.

Each predictor takes raw model inputs (views + the full scalar-feature matrix in
:data:`SCALAR_FEATURES` order) and returns ``(n, 3)`` probabilities in ``CLASSES`` order, so the
pipeline does not care which model is serving.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from transit_hunter.config import REPO_ROOT, CNNConfig, get_settings
from transit_hunter.models.cnn import DualBranchCNN, FeatureScaler
from transit_hunter.models.gbm import GBMModel, feature_matrix
from transit_hunter.processing.features import SCALAR_FEATURES

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


class Predictor(Protocol):
    """Common interface of :class:`CNNPredictor` and :class:`GBMPredictor`."""

    kind: str
    trained_missions: list[str]

    def predict_proba(
        self,
        global_views: NDArray[np.float32],
        local_views: NDArray[np.float32],
        scalars: FloatArray,
    ) -> FloatArray:
        """Probabilities ``(n, 3)``; ``scalars`` is ``(n, len(SCALAR_FEATURES))``."""
        ...


def _select(scalars: FloatArray, names: list[str]) -> FloatArray:
    idx = [SCALAR_FEATURES.index(n) for n in names]
    return np.asarray(scalars, dtype=float)[:, idx]


class CNNPredictor:
    """Loads ``best.pt`` and runs the CNN on CPU."""

    kind = "cnn"

    def __init__(self, checkpoint: Path) -> None:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.cfg = CNNConfig.model_validate(blob["cnn_config"])
        self.scaler = FeatureScaler.from_dict(blob["scaler"])
        self.model = DualBranchCNN(
            self.cfg, blob["global_bins"], blob["local_bins"], len(self.scaler.names)
        )
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        self.checkpoint = checkpoint
        meta = blob.get("data_meta", {}) or {}
        self.trained_missions: list[str] = list(meta.get("train_missions") or ["Kepler"])
        self.epoch = int(blob.get("epoch", 0))

    def tensors(
        self,
        global_views: NDArray[np.float32],
        local_views: NDArray[np.float32],
        scalars: FloatArray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Model-ready tensors (scalars selected and standardized)."""
        g = torch.as_tensor(np.asarray(global_views, dtype=np.float32))
        loc = torch.as_tensor(np.asarray(local_views, dtype=np.float32))
        s = torch.as_tensor(self.scaler.transform(_select(scalars, self.scaler.names)))
        return g, loc, s

    def predict_proba(
        self,
        global_views: NDArray[np.float32],
        local_views: NDArray[np.float32],
        scalars: FloatArray,
    ) -> FloatArray:
        g, loc, s = self.tensors(global_views, local_views, scalars)
        with torch.no_grad():
            p = torch.softmax(self.model(g, loc, s), dim=1).numpy()
        return p.astype(float)


class GBMPredictor:
    """Loads ``model.joblib`` and runs LightGBM."""

    kind = "gbm"

    def __init__(self, checkpoint: Path) -> None:
        self.model = GBMModel.load(checkpoint)
        self.checkpoint = checkpoint
        self.trained_missions = self.model.trained_missions

    def design(
        self,
        global_views: NDArray[np.float32],
        local_views: NDArray[np.float32],
        scalars: FloatArray,
    ) -> FloatArray:
        """GBM design matrix for these inputs."""
        X, names = feature_matrix(scalars, global_views, local_views, self.model.cfg)
        if names != self.model.feature_names:
            raise ValueError("feature layout differs from the trained model; retrain the GBM")
        return X

    def predict_proba(
        self,
        global_views: NDArray[np.float32],
        local_views: NDArray[np.float32],
        scalars: FloatArray,
    ) -> FloatArray:
        return self.model.predict_proba(self.design(global_views, local_views, scalars))


def model_paths(models_dir: Path | None = None) -> dict[str, Path]:
    """Checkpoint locations, preferring ``models/final`` (the committed demo model) if present."""
    root = models_dir or get_settings().models_dir
    root = root if root.is_absolute() else REPO_ROOT / root
    out = {}
    for kind, rel in (("cnn", "cnn/best.pt"), ("gbm", "gbm/model.joblib")):
        for base in (root / "final", root):
            if (base / rel).exists():
                out[kind] = base / rel
                break
    return out


def load_predictor(
    kind: Literal["cnn", "gbm", "auto"] | None = None, models_dir: Path | None = None
) -> Predictor:
    """Load a trained model.

    ``"auto"`` picks whichever model has the higher validation macro PR-AUC in
    ``models/metrics.json``; without metrics it prefers the CNN.

    Raises:
        FileNotFoundError: if the requested model has not been trained.
    """
    kind = kind or get_settings().model_kind
    paths = model_paths(models_dir)
    if not paths:
        raise FileNotFoundError("no trained models found; run `make train`")
    if kind == "auto":
        kind = "cnn" if "cnn" in paths else "gbm"
        metrics_file = next(
            (
                p.parent.parent / "metrics.json"
                for p in paths.values()
                if (p.parent.parent / "metrics.json").exists()
            ),
            None,
        )
        if metrics_file is not None and len(paths) == 2:
            m = json.loads(metrics_file.read_text())
            scores = {k: m.get(k, {}).get("val", {}).get("macro_pr_auc") for k in ("cnn", "gbm")}
            valid = {
                k: v for k, v in scores.items() if isinstance(v, (int, float)) and np.isfinite(v)
            }
            if valid:
                best_kind = max(valid, key=lambda k: valid[k])
                kind = "cnn" if best_kind == "cnn" else "gbm"
    if kind not in paths:
        raise FileNotFoundError(f"{kind} model not trained; run `make train-{kind}`")
    logger.info("serving %s model from %s", kind, paths[kind])
    return CNNPredictor(paths[kind]) if kind == "cnn" else GBMPredictor(paths[kind])

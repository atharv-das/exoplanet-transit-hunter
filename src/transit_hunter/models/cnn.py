"""Dual-branch 1D CNN (AstroNet-style; Shallue & Vanderburg 2018) with scalar side inputs.

    global view (1 x 201) -> deeper conv tower --\
    local view  (1 x 61)  -> shallower tower  ----+--> concat --> FC --> FC --> logits (3)
    scalar features (k)   -> standardized -------/

Softmax is applied outside the module (``predict_proba``) so training can use logits directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from transit_hunter.config import BranchConfig, CNNConfig

FloatArray = NDArray[np.float64]


@dataclass
class FeatureScaler:
    """Robust standardization (median / IQR), fitted on the training split only.

    Missing values become 0 after scaling (i.e. the training median), and values are clipped
    to +/- ``clip`` so outliers cannot saturate the network.
    """

    names: list[str]
    center: list[float]
    scale: list[float]
    clip: float = 10.0

    @classmethod
    def fit(cls, X: FloatArray, names: list[str], clip: float = 10.0) -> FeatureScaler:
        """Fit on a ``(n_samples, n_features)`` matrix that may contain NaN."""
        center, scale = [], []
        for j in range(X.shape[1]):
            col = X[:, j][np.isfinite(X[:, j])]
            if col.size == 0:
                center.append(0.0)
                scale.append(1.0)
                continue
            med = float(np.median(col))
            q75, q25 = np.percentile(col, [75, 25])
            s = float((q75 - q25) / 1.349)
            if not np.isfinite(s) or s < 1e-8:
                s = float(np.std(col)) or 1.0
            center.append(med)
            scale.append(s)
        return cls(list(names), center, scale, clip)

    def transform(self, X: FloatArray) -> NDArray[np.float32]:
        """Standardize; NaN -> 0; clip."""
        Z = (np.asarray(X, dtype=float) - np.asarray(self.center)) / np.asarray(self.scale)
        Z = np.where(np.isfinite(Z), Z, 0.0)
        return np.clip(Z, -self.clip, self.clip).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python representation (safe for ``torch.load(weights_only=True)``)."""
        return {"names": self.names, "center": self.center, "scale": self.scale, "clip": self.clip}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureScaler:
        """Inverse of :meth:`to_dict`."""
        return cls(list(d["names"]), list(d["center"]), list(d["scale"]), float(d["clip"]))


def _conv_tower(cfg: BranchConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_ch = 1
    for out_ch in cfg.channels:
        for _ in range(cfg.convs_per_block):
            layers += [nn.Conv1d(in_ch, out_ch, cfg.kernel_size, padding="same"), nn.ReLU()]
            in_ch = out_ch
        layers.append(nn.MaxPool1d(cfg.pool_size, cfg.pool_stride))
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)


class DualBranchCNN(nn.Module):
    """Global + local conv towers, concatenated with scalar features, followed by an MLP head.

    Args:
        cfg: CNN config (branch shapes, head sizes, number of classes).
        global_bins: Length of the global view.
        local_bins: Length of the local view.
        n_scalar: Number of scalar features (0 disables the scalar input).
    """

    def __init__(self, cfg: CNNConfig, global_bins: int, local_bins: int, n_scalar: int) -> None:
        super().__init__()
        self.global_bins = global_bins
        self.local_bins = local_bins
        self.n_scalar = n_scalar
        self.global_tower = _conv_tower(cfg.global_branch)
        self.local_tower = _conv_tower(cfg.local_branch)
        with torch.no_grad():
            g = self.global_tower(torch.zeros(1, 1, global_bins)).shape[1]
            loc = self.local_tower(torch.zeros(1, 1, local_bins)).shape[1]
        head: list[nn.Module] = []
        width = g + loc + n_scalar
        for h in cfg.head.hidden:
            head += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(cfg.head.dropout)]
            width = h
        head.append(nn.Linear(width, cfg.num_classes))
        self.head = nn.Sequential(*head)

    def forward(
        self, global_view: torch.Tensor, local_view: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        """Return logits of shape ``(batch, num_classes)``.

        Views may be ``(batch, bins)`` or ``(batch, 1, bins)``.
        """
        if global_view.dim() == 2:
            global_view = global_view.unsqueeze(1)
        if local_view.dim() == 2:
            local_view = local_view.unsqueeze(1)
        parts = [self.global_tower(global_view), self.local_tower(local_view)]
        if self.n_scalar:
            parts.append(scalars)
        out: torch.Tensor = self.head(torch.cat(parts, dim=1))
        return out


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

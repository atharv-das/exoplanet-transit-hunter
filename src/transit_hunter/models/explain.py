"""Explanations: which parts of the folded transit (and which features) drove a prediction.

* CNN: Integrated Gradients (Sundararajan et al. 2017, via Captum) on the global view, local
  view and standardized scalars. The baseline is all zeros, which for the normalized views means
  "flat light curve at the median level" and for the scalars "the training median".
* GBM: TreeSHAP contributions from LightGBM (``pred_contrib=True``), in log-odds space.

Saliency maps are returned both signed (positive = pushed toward the explained class) and as a
0..1 magnitude for overlay plots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from transit_hunter.config import CLASSES
from transit_hunter.models.predict import CNNPredictor, GBMPredictor, Predictor

FloatArray = NDArray[np.float64]


@dataclass
class Explanation:
    """Attributions for one example and one target class."""

    method: str
    target_class: str
    global_saliency: list[float] = field(default_factory=list)  # signed, per bin
    local_saliency: list[float] = field(default_factory=list)
    global_magnitude: list[float] = field(default_factory=list)  # |attr| scaled to 0..1
    local_magnitude: list[float] = field(default_factory=list)
    feature_attributions: dict[str, float] = field(default_factory=dict)
    convergence_delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return asdict(self)

    def top_features(self, k: int = 5) -> list[tuple[str, float]]:
        """Features with the largest absolute attribution."""
        items = sorted(self.feature_attributions.items(), key=lambda kv: -abs(kv[1]))
        return items[:k]


def _magnitude(a: FloatArray, scale: float) -> list[float]:
    return (np.abs(a) / scale).tolist() if scale > 0 else np.zeros_like(a).tolist()


def explain_cnn(
    predictor: CNNPredictor,
    global_view: NDArray[np.float32],
    local_view: NDArray[np.float32],
    scalars: FloatArray,
    target: int | None = None,
    n_steps: int = 32,
) -> Explanation:
    """Integrated Gradients for one example.

    Args:
        predictor: Loaded CNN.
        global_view: ``(global_bins,)`` view.
        local_view: ``(local_bins,)`` view.
        scalars: ``(len(SCALAR_FEATURES),)`` raw scalar features.
        target: Class index to explain; defaults to the predicted class.
        n_steps: Riemann steps for the path integral (32 is plenty for this model size).
    """
    from captum.attr import IntegratedGradients

    g, loc, s = predictor.tensors(global_view[None], local_view[None], np.asarray(scalars)[None])
    model = predictor.model
    model.eval()
    if target is None:
        with torch.no_grad():
            target = int(model(g, loc, s).argmax(dim=1).item())

    def forward(gv: torch.Tensor, lv: torch.Tensor, sv: torch.Tensor) -> torch.Tensor:
        return torch.softmax(model(gv, lv, sv), dim=1)

    ig = IntegratedGradients(forward)
    baselines = (torch.zeros_like(g), torch.zeros_like(loc), torch.zeros_like(s))
    attrs, delta = ig.attribute(
        (g, loc, s),
        baselines=baselines,
        target=target,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    ga = attrs[0].detach().numpy().reshape(-1).astype(float)
    la = attrs[1].detach().numpy().reshape(-1).astype(float)
    sa = attrs[2].detach().numpy().reshape(-1).astype(float)
    scale = float(max(np.abs(ga).max(initial=0), np.abs(la).max(initial=0)))
    return Explanation(
        method="integrated_gradients",
        target_class=CLASSES[target],
        global_saliency=ga.tolist(),
        local_saliency=la.tolist(),
        global_magnitude=_magnitude(ga, scale),
        local_magnitude=_magnitude(la, scale),
        feature_attributions={n: float(v) for n, v in zip(predictor.scaler.names, sa, strict=True)},
        convergence_delta=float(delta.abs().max()),
    )


def explain_gbm(
    predictor: GBMPredictor,
    global_view: NDArray[np.float32],
    local_view: NDArray[np.float32],
    scalars: FloatArray,
    target: int | None = None,
) -> Explanation:
    """TreeSHAP feature contributions for one example (no per-bin saliency for the GBM)."""
    X = predictor.design(global_view[None], local_view[None], np.asarray(scalars)[None])
    if target is None:
        target = int(predictor.model.predict_proba(X).argmax(axis=1)[0])
    contrib = predictor.model.contributions(X)[0, target, :-1]
    names = predictor.model.feature_names
    # Map local-view summary bins back onto the 61-bin axis so the UI can still overlay them.
    local_bins = [i for i, n in enumerate(names) if n.startswith("local_b")]
    local_sal: list[float] = []
    if local_bins:
        per_chunk = np.array([contrib[i] for i in local_bins])
        edges = np.linspace(0, local_view.shape[0], len(local_bins) + 1).astype(int)
        arr = np.zeros(local_view.shape[0])
        for k, (a, b) in enumerate(pairwise(edges)):
            arr[a:b] = per_chunk[k]
        local_sal = arr.tolist()
    scale = float(np.abs(local_sal).max(initial=0)) if local_sal else 0.0
    return Explanation(
        method="tree_shap",
        target_class=CLASSES[target],
        local_saliency=local_sal,
        local_magnitude=_magnitude(np.asarray(local_sal), scale) if local_sal else [],
        feature_attributions={n: float(v) for n, v in zip(names, contrib, strict=True)},
    )


def explain(
    predictor: Predictor,
    global_view: NDArray[np.float32],
    local_view: NDArray[np.float32],
    scalars: FloatArray,
    target: int | None = None,
) -> Explanation:
    """Dispatch to the right explainer for the serving model."""
    if isinstance(predictor, CNNPredictor):
        return explain_cnn(predictor, global_view, local_view, scalars, target)
    if isinstance(predictor, GBMPredictor):
        return explain_gbm(predictor, global_view, local_view, scalars, target)
    raise TypeError(f"no explainer for {type(predictor).__name__}")

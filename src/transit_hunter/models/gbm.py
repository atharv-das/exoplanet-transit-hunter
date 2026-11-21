"""LightGBM baseline on hand-engineered features.

Inputs are every scalar feature (vetting statistics; NaN is fine, LightGBM routes missing values
natively) plus coarse summaries of the two views, so the baseline sees the transit shape and the
secondary-eclipse region without a CNN.
"""

from __future__ import annotations

import inspect
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from transit_hunter.config import CLASSES, GBMConfig
from transit_hunter.processing.features import SCALAR_FEATURES

FloatArray = NDArray[np.float64]

#: The local view is averaged down to this many bins for the GBM.
LOCAL_SUMMARY_BINS = 13
_CENTER_HALF_WIDTH = 5  # global-view bins treated as "the transit"


def view_summary_names() -> list[str]:
    """Names of the view-summary features, in the order of :func:`view_summaries`."""
    return [f"local_b{i}" for i in range(LOCAL_SUMMARY_BINS)] + [
        "global_min_off_center",
        "global_half_phase",
        "global_oot_std",
        "global_n_below_half",
        "local_asymmetry",
    ]


def view_summaries(
    global_views: NDArray[np.float32], local_views: NDArray[np.float32]
) -> FloatArray:
    """Hand-crafted summaries of the views, shape ``(n, len(view_summary_names()))``."""
    g = np.asarray(global_views, dtype=float)
    loc = np.asarray(local_views, dtype=float)
    n, gb = g.shape
    # Local view averaged into LOCAL_SUMMARY_BINS chunks (shape of the transit).
    edges = np.linspace(0, loc.shape[1], LOCAL_SUMMARY_BINS + 1).astype(int)
    local_bins = np.column_stack([loc[:, a:b].mean(axis=1) for a, b in pairwise(edges)])
    c = gb // 2
    off = np.ones(gb, dtype=bool)
    off[c - _CENTER_HALF_WIDTH : c + _CENTER_HALF_WIDTH + 1] = False
    min_off = g[:, off].min(axis=1)  # deepest dip away from the transit: secondary eclipse?
    half_phase = np.concatenate([g[:, :3], g[:, -3:]], axis=1).mean(axis=1)  # phase +/- 0.5
    oot_std = g[:, off].std(axis=1)
    n_below = (g < -0.5).sum(axis=1).astype(float)
    mid = loc.shape[1] // 2
    asym = np.abs(loc[:, :mid] - loc[:, -mid:][:, ::-1]).mean(axis=1)
    out = np.column_stack([local_bins, min_off, half_phase, oot_std, n_below, asym])
    assert out.shape == (n, len(view_summary_names()))
    return out


def feature_matrix(
    scalars: FloatArray,
    global_views: NDArray[np.float32],
    local_views: NDArray[np.float32],
    cfg: GBMConfig,
) -> tuple[FloatArray, list[str]]:
    """Assemble the GBM design matrix and its column names.

    Args:
        scalars: ``(n, len(SCALAR_FEATURES))`` raw scalar features (NaN allowed).
        global_views: Global views.
        local_views: Local views.
        cfg: Selects which feature groups to include.
    """
    blocks, names = [], []
    if cfg.use_scalar_features:
        blocks.append(np.asarray(scalars, dtype=float))
        names += list(SCALAR_FEATURES)
    if cfg.use_view_summaries:
        blocks.append(view_summaries(global_views, local_views))
        names += view_summary_names()
    if not blocks:
        raise ValueError("GBM config disables every feature group")
    return np.column_stack(blocks), names


class GBMModel:
    """Thin wrapper around ``lightgbm.LGBMClassifier`` with save/load and SHAP explanations."""

    kind = "gbm"

    def __init__(self, cfg: GBMConfig, model: lgb.LGBMClassifier | None = None) -> None:
        self.cfg = cfg
        self.model = model
        self.feature_names: list[str] = []
        self.trained_missions: list[str] = ["Kepler"]
        self.data_meta: dict[str, Any] = {}
        self.best_iteration: int | None = None

    def _frame(self, X: FloatArray) -> pd.DataFrame:
        return pd.DataFrame(np.asarray(X, dtype=float), columns=self.feature_names)

    def fit(
        self,
        X: FloatArray,
        y: NDArray[np.int64],
        X_val: FloatArray | None,
        y_val: NDArray[np.int64] | None,
        feature_names: list[str],
    ) -> GBMModel:
        """Train with early stopping on the validation multi-class log-loss."""
        params: dict[str, Any] = dict(self.cfg.params)
        params.setdefault("random_state", self.cfg.seed)
        params["num_class"] = len(CLASSES)
        self.model = lgb.LGBMClassifier(**params)
        self.feature_names = list(feature_names)
        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None and len(np.unique(y_val)) > 1:
            # LightGBM >= 4.7 renamed eval_set -> eval_X / eval_y; support both.
            if "eval_X" in inspect.signature(lgb.LGBMClassifier.fit).parameters:
                fit_kwargs["eval_X"], fit_kwargs["eval_y"] = (self._frame(X_val),), (y_val,)
            else:
                fit_kwargs["eval_set"] = [(self._frame(X_val), y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ]
        self.model.fit(self._frame(X), y, **fit_kwargs)
        self.best_iteration = getattr(self.model, "best_iteration_", None) or None
        return self

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Class probabilities, shape ``(n, 3)``."""
        if self.model is None:
            raise RuntimeError("model is not trained")
        p = self.model.predict_proba(self._frame(X), num_iteration=self.best_iteration)
        return np.asarray(p, dtype=float)

    def contributions(self, X: FloatArray) -> FloatArray:
        """Per-feature SHAP contributions (TreeSHAP), shape ``(n, n_classes, n_features + 1)``.

        The last column is the bias term. Values are in log-odds (raw score) space.
        """
        if self.model is None:
            raise RuntimeError("model is not trained")
        raw = self.model.predict(
            self._frame(X), pred_contrib=True, num_iteration=self.best_iteration
        )
        raw = np.asarray(raw, dtype=float)
        return raw.reshape(X.shape[0], len(CLASSES), len(self.feature_names) + 1)

    def feature_importance(self) -> dict[str, float]:
        """Gain-based importance per feature."""
        if self.model is None:
            return {}
        gains = self.model.booster_.feature_importance(importance_type="gain")
        return {n: float(v) for n, v in zip(self.feature_names, gains, strict=True)}

    def save(self, path: Path) -> None:
        """Pickle the model and metadata with joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "best_iteration": self.best_iteration,
                "cfg": self.cfg.model_dump(mode="json"),
                "trained_missions": self.trained_missions,
                "data_meta": self.data_meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> GBMModel:
        """Inverse of :meth:`save`."""
        blob = joblib.load(path)
        obj = cls(GBMConfig.model_validate(blob["cfg"]), blob["model"])
        obj.feature_names = list(blob["feature_names"])
        obj.best_iteration = blob["best_iteration"]
        obj.trained_missions = list(blob.get("trained_missions", ["Kepler"]))
        obj.data_meta = dict(blob.get("data_meta", {}))
        return obj

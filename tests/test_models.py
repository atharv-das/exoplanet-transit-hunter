"""Models: shapes, forward pass, training smoke runs, predictors, explanations, metrics."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from transit_hunter.config import CLASSES, load_cnn_config, load_gbm_config
from transit_hunter.data.dataset import (
    ProcessedDataset,
    TransitViewsDataset,
    augment_views,
)
from transit_hunter.models.cnn import DualBranchCNN, FeatureScaler, count_parameters
from transit_hunter.models.evaluate import (
    calibration,
    comparison_table,
    evaluate_predictions,
    macro_pr_auc,
    recall_at_precision,
)
from transit_hunter.models.explain import explain
from transit_hunter.models.gbm import feature_matrix, view_summaries, view_summary_names
from transit_hunter.models.predict import CNNPredictor, GBMPredictor, load_predictor
from transit_hunter.models.train import FocalLoss, class_weights
from transit_hunter.processing.features import SCALAR_FEATURES

# ----------------------------------------------------------------- CNN building blocks


def test_cnn_forward_shapes() -> None:
    cfg = load_cnn_config()
    model = DualBranchCNN(cfg, 201, 61, len(cfg.scalar_features))
    g, loc, s = torch.randn(4, 201), torch.randn(4, 61), torch.randn(4, len(cfg.scalar_features))
    assert model(g, loc, s).shape == (4, 3)
    assert model(g[:, None], loc[:, None], s).shape == (4, 3)  # (batch, 1, bins) also accepted
    assert 50_000 < count_parameters(model) < 2_000_000


def test_cnn_without_scalars() -> None:
    cfg = load_cnn_config()
    model = DualBranchCNN(cfg, 201, 61, 0)
    assert model(torch.randn(2, 201), torch.randn(2, 61), torch.zeros(2, 0)).shape == (2, 3)


def test_feature_scaler_handles_nan_and_roundtrips() -> None:
    X = np.array([[1.0, np.nan], [2.0, 5.0], [3.0, 7.0], [100.0, 6.0]])
    sc = FeatureScaler.fit(X, ["a", "b"])
    Z = sc.transform(X)
    assert Z.dtype == np.float32
    assert np.isfinite(Z).all()
    assert Z[0, 1] == 0.0  # NaN -> median
    assert Z.max() <= sc.clip
    again = FeatureScaler.from_dict(sc.to_dict())
    assert np.allclose(again.transform(X), Z)


def test_augment_views() -> None:
    rng = np.random.default_rng(0)
    g = np.linspace(-1, 1, 201).astype(np.float32)
    loc = np.linspace(-1, 1, 61).astype(np.float32)
    g2, l2 = augment_views(
        g, loc, rng, flip_prob=1.0, noise_std=0.0, max_shift_global=0, max_shift_local=0
    )
    assert np.allclose(g2, g[::-1]) and np.allclose(l2, loc[::-1])
    g3, l3 = augment_views(g, loc, rng, 0.0, 0.0, 3, 1)
    assert g3.shape == g.shape and l3.shape == loc.shape
    g4, _ = augment_views(g, loc, rng, 0.0, 0.05, 0, 0)
    assert 0.02 < np.std(g4 - g) < 0.08


def test_views_dataset_items() -> None:
    cfg = load_cnn_config()
    ds = TransitViewsDataset(
        np.zeros((5, 201), np.float32),
        np.zeros((5, 61), np.float32),
        np.zeros((5, 6), np.float32),
        np.arange(5) % 3,
        augment=True,
        augment_cfg=cfg.augment,
    )
    g, loc, s, y = ds[2]
    assert g.shape == (201,) and loc.shape == (61,) and s.shape == (6,)
    assert y.dtype == torch.int64 and int(y) == 2
    with pytest.raises(ValueError):
        TransitViewsDataset(ds.g, ds.loc, ds.s, ds.y, augment=True)


def test_focal_loss_reduces_to_ce_at_gamma_zero() -> None:
    logits, y = torch.randn(8, 3), torch.randint(0, 3, (8,))
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert torch.allclose(FocalLoss(0.0)(logits, y), ce, atol=1e-6)
    assert FocalLoss(2.0)(logits, y) < ce


def test_class_weights_inverse_frequency() -> None:
    w = class_weights(np.array([0, 0, 0, 1, 2, 2]))
    assert w[1] > w[2] > w[0]
    assert w.mean() == pytest.approx(1.0)


def test_view_summaries_shape() -> None:
    rng = np.random.default_rng(0)
    g = rng.normal(size=(3, 201)).astype(np.float32)
    loc = rng.normal(size=(3, 61)).astype(np.float32)
    out = view_summaries(g, loc)
    assert out.shape == (3, len(view_summary_names()))
    X, names = feature_matrix(np.zeros((3, len(SCALAR_FEATURES))), g, loc, load_gbm_config())
    assert X.shape[1] == len(names) == len(SCALAR_FEATURES) + len(view_summary_names())


# ----------------------------------------------------------------- metrics


def test_metrics_perfect_and_random() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    perfect = np.eye(3)[y]
    m = evaluate_predictions(y, perfect)
    assert m["accuracy"] == 1.0 and m["macro_pr_auc"] == pytest.approx(1.0)
    assert m["pc_recall_at_95_precision"] == 1.0
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    rng = np.random.default_rng(0)
    yr = rng.integers(0, 3, 3000)
    pr = rng.dirichlet(np.ones(3), 3000)
    assert macro_pr_auc(yr, pr) == pytest.approx(1 / 3, abs=0.05)
    table = comparison_table({"cnn": {"test": m}})
    assert "| cnn |" in table


def test_recall_at_precision_and_calibration() -> None:
    y = np.array([1, 1, 1, 0, 0, 1, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.65, 0.6, 0.4, 0.3, 0.1])
    rec, thr = recall_at_precision(y, score, 0.95)
    assert rec == pytest.approx(0.75) and thr == pytest.approx(0.7)
    cal = calibration(np.array([0, 1] * 50), np.full(100, 0.5))
    assert cal["ece"] == pytest.approx(0.0)
    assert sum(cal["counts"]) == 100


# ----------------------------------------------------------------- training smoke tests


def test_predictors_output_probabilities(trained: tuple[ProcessedDataset, Path, Path]) -> None:
    data, models, _ = trained
    test = data.subset("test")
    for pred in (
        CNNPredictor(models / "cnn" / "best.pt"),
        GBMPredictor(models / "gbm" / "model.joblib"),
    ):
        p = pred.predict_proba(test.global_views, test.local_views, test.features())
        assert p.shape == (len(test), 3)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
        assert (p >= 0).all()


def test_inference_under_two_seconds_on_cpu(trained: tuple[ProcessedDataset, Path, Path]) -> None:
    data, models, _ = trained
    pred = CNNPredictor(models / "cnn" / "best.pt")
    g, loc, s = data.global_views[:1], data.local_views[:1], data.features()[:1]
    start = time.perf_counter()
    pred.predict_proba(g, loc, s)
    explain(pred, g[0], loc[0], s[0])
    assert time.perf_counter() - start < 2.0


def test_load_predictor_kinds(trained: tuple[ProcessedDataset, Path, Path]) -> None:
    _, models, _ = trained
    assert load_predictor("cnn", models).kind == "cnn"
    assert load_predictor("gbm", models).kind == "gbm"
    assert load_predictor("auto", models).kind in ("cnn", "gbm")
    with pytest.raises(FileNotFoundError):
        load_predictor("cnn", models / "nope")


def test_explanations(trained: tuple[ProcessedDataset, Path, Path]) -> None:
    data, models, _ = trained
    i = 0
    args = (data.global_views[i], data.local_views[i], data.features()[i])
    e = explain(CNNPredictor(models / "cnn" / "best.pt"), *args)
    assert e.method == "integrated_gradients"
    assert len(e.global_saliency) == 201 and len(e.local_saliency) == 61
    assert max(max(e.global_magnitude), max(e.local_magnitude)) == pytest.approx(1.0)
    assert set(e.feature_attributions) == set(load_cnn_config().scalar_features)
    assert e.target_class in CLASSES
    eg = explain(GBMPredictor(models / "gbm" / "model.joblib"), *args)
    assert eg.method == "tree_shap"
    assert len(eg.local_saliency) == 61
    assert len(eg.top_features(3)) == 3

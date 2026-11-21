"""Evaluation metrics and the CNN-vs-GBM comparison table.

Reported for every model and split:
    * per-class precision / recall / F1 and support
    * macro PR-AUC (one-vs-rest average precision, averaged over classes present)
    * confusion matrix (rows = true, columns = predicted, order PC/EB/FP)
    * recall of the Planet class at >= 95 % precision (how many planets survive a strict cut)
    * reliability curve + expected calibration error (ECE) for P(planet)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from transit_hunter.config import CLASSES

FloatArray = NDArray[np.float64]
PC = CLASSES.index("PC")


def macro_pr_auc(y: NDArray[np.int64], proba: FloatArray) -> float:
    """Mean one-vs-rest average precision over classes that occur in ``y`` (NaN if none)."""
    scores = []
    for k in range(proba.shape[1]):
        pos = (y == k).astype(int)
        if 0 < pos.sum() < len(pos):
            scores.append(average_precision_score(pos, proba[:, k]))
    return float(np.mean(scores)) if scores else float("nan")


def recall_at_precision(
    y_true_binary: NDArray[np.int64], score: FloatArray, min_precision: float = 0.95
) -> tuple[float, float]:
    """Highest recall reachable with precision >= ``min_precision``, and its score threshold.

    Returns ``(0.0, nan)`` if the precision target is never reached.
    """
    if y_true_binary.sum() == 0:
        return float("nan"), float("nan")
    precision, recall, thresholds = precision_recall_curve(y_true_binary, score)
    ok = precision[:-1] >= min_precision
    if not ok.any():
        return 0.0, float("nan")
    best = int(np.argmax(np.where(ok, recall[:-1], -1)))
    return float(recall[best]), float(thresholds[best])


def calibration(
    y_true_binary: NDArray[np.int64], score: FloatArray, n_bins: int = 10
) -> dict[str, Any]:
    """Reliability curve (equal-width bins) and expected calibration error."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(score, edges[1:-1]), 0, n_bins - 1)
    mean_pred: list[float | None] = []
    frac_pos: list[float | None] = []
    counts: list[int] = []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        counts.append(n)
        if n == 0:
            mean_pred.append(None)
            frac_pos.append(None)
            continue
        mp, fp = float(score[m].mean()), float(y_true_binary[m].mean())
        mean_pred.append(mp)
        frac_pos.append(fp)
        ece += n / len(score) * abs(mp - fp)
    return {
        "bin_edges": edges.tolist(),
        "mean_predicted": mean_pred,
        "fraction_positive": frac_pos,
        "counts": counts,
        "ece": float(ece),
    }


def evaluate_predictions(y: NDArray[np.int64], proba: FloatArray) -> dict[str, Any]:
    """Every metric for one model on one split.

    Args:
        y: True labels, shape ``(n,)``.
        proba: Predicted probabilities, shape ``(n, 3)``.
    """
    y = np.asarray(y, dtype=np.int64)
    proba = np.asarray(proba, dtype=float)
    pred = proba.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = {
        c: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i, c in enumerate(CLASSES)
    }
    is_pc = (y == PC).astype(np.int64)
    rec95, thr95 = recall_at_precision(is_pc, proba[:, PC], 0.95)
    return {
        "n": len(y),
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
        "macro_f1": float(np.mean(f)),
        "macro_pr_auc": macro_pr_auc(y, proba),
        "pc_recall_at_95_precision": rec95,
        "pc_threshold_at_95_precision": thr95,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "calibration_pc": calibration(is_pc, proba[:, PC]),
    }


def _fmt(x: Any) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def comparison_table(results: dict[str, dict[str, Any]], split: str = "test") -> str:
    """Markdown table comparing models on one split.

    Args:
        results: ``{model_name: {split_name: metrics_dict}}``.
        split: Which split to tabulate.
    """
    header = (
        "| Model | Macro PR-AUC | Macro F1 | Accuracy | PC recall @95% prec. | "
        "PC F1 | EB F1 | FP F1 | ECE (PC) | n |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for name, by_split in results.items():
        m = by_split.get(split)
        if not m:
            continue
        pc = m["per_class"]
        cells = [
            name,
            _fmt(m["macro_pr_auc"]),
            _fmt(m["macro_f1"]),
            _fmt(m["accuracy"]),
            _fmt(m["pc_recall_at_95_precision"]),
            _fmt(pc["PC"]["f1"]),
            _fmt(pc["EB"]["f1"]),
            _fmt(pc["FP"]["f1"]),
            _fmt(m["calibration_pc"]["ece"]),
            str(m["n"]),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return header + "\n".join(rows) + "\n"


def confusion_markdown(cm: list[list[int]]) -> str:
    """Confusion matrix as a small markdown table."""
    head = "| true \\ pred | " + " | ".join(CLASSES) + " |\n|---|" + "---|" * len(CLASSES) + "\n"
    body = "\n".join(
        f"| {CLASSES[i]} | " + " | ".join(str(v) for v in row) + " |" for i, row in enumerate(cm)
    )
    return head + body + "\n"

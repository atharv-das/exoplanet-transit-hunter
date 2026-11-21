#!/usr/bin/env python
"""Train and evaluate the models.

Examples:
    python scripts/train.py --model gbm
    python scripts/train.py --model cnn
    python scripts/train.py --model both --export-final     # also copy best weights to models/final
    python scripts/train.py --evaluate-only                  # re-evaluate saved checkpoints

Outputs:
    models/cnn/best.pt, models/gbm/model.joblib     checkpoints
    models/metrics.json                              every metric, every split, both models
    models/results.md                                comparison table + confusion matrices
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from transit_hunter.config import (
    REPO_ROOT,
    get_settings,
    load_cnn_config,
    load_gbm_config,
    setup_logging,
)
from transit_hunter.data.dataset import ProcessedDataset, load_dataset
from transit_hunter.models.evaluate import (
    comparison_table,
    confusion_markdown,
    evaluate_predictions,
)
from transit_hunter.models.predict import CNNPredictor, GBMPredictor, Predictor
from transit_hunter.models.train import CNN_CHECKPOINT, GBM_CHECKPOINT, train_cnn, train_gbm

logger = logging.getLogger("train")


def _models_dir() -> Path:
    d = get_settings().models_dir
    return d if d.is_absolute() else REPO_ROOT / d


def evaluate_splits(predictor: Predictor, data: ProcessedDataset) -> dict[str, Any]:
    """Metrics on val, test (real examples only), per mission when there are several (e.g.
    ``test_kepler``, ``test_tess``), and the held-out ``tess`` split if present; plus latency."""
    out: dict[str, Any] = {}
    for split, real_only in (("val", False), ("test", True), ("tess", True)):
        sub = data.subset(split, real_only=real_only)
        if len(sub) == 0:
            continue
        proba = predictor.predict_proba(sub.global_views, sub.local_views, sub.features())
        out[split] = evaluate_predictions(sub.labels, proba)
        missions = sorted(sub.index["mission"].unique())
        if split == "test" and len(missions) > 1:
            for m in missions:
                part = sub.mission(m)
                p = predictor.predict_proba(part.global_views, part.local_views, part.features())
                out[f"test_{m.lower()}"] = evaluate_predictions(part.labels, p)
    # Single-example latency (what the API pays per candidate).
    one = data.subset("test") if len(data.subset("test")) else data
    if len(one):
        g, loc, s = one.global_views[:1], one.local_views[:1], one.features()[:1]
        predictor.predict_proba(g, loc, s)  # warm-up
        t0 = time.perf_counter()
        for _ in range(20):
            predictor.predict_proba(g, loc, s)
        out["latency_ms_per_example"] = (time.perf_counter() - t0) / 20 * 1000
    return out


def write_reports(metrics: dict[str, Any], out_dir: Path) -> None:
    """Write metrics.json and results.md."""
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    models = {k: v for k, v in metrics.items() if k in ("cnn", "gbm")}
    lines = ["# Results\n", f"_Generated {metrics.get('generated_at_utc', '')}_\n"]
    for split, title in (
        ("test", "Test split (real signals, unseen stars)"),
        ("test_kepler", "Test split: Kepler stars only"),
        ("test_tess", "Test split: TESS stars only"),
        ("val", "Validation split"),
        ("tess", "TESS held-out (domain shift: TESS not trained on)"),
    ):
        if any(split in v for v in models.values()):
            lines += [f"\n## {title}\n", comparison_table(models, split)]
    for name, v in models.items():
        if "test" in v:
            lines += [
                f"\n### Confusion matrix: {name} (test)\n",
                confusion_markdown(v["test"]["confusion_matrix"]),
            ]
        if "latency_ms_per_example" in v:
            lines.append(
                f"\n{name} CPU latency: {v['latency_ms_per_example']:.2f} ms per candidate\n"
            )
    ds = metrics.get("dataset", {})
    if ds:
        lines.append(
            f"\nDataset: {ds.get('n_examples')} examples from {ds.get('n_stars')} stars "
            f"({ds.get('n_synthetic', 0)} synthetic), built {ds.get('built_at_utc')}.\n"
        )
    (out_dir / "results.md").write_text("\n".join(lines))


def export_final(out_dir: Path) -> None:
    """Copy checkpoints + metrics into models/final (small enough to commit for the demo)."""
    final = out_dir / "final"
    for sub, name in (("cnn", CNN_CHECKPOINT), ("gbm", GBM_CHECKPOINT)):
        src = out_dir / sub / name
        if src.exists():
            (final / sub).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, final / sub / name)
    for f in ("metrics.json", "results.md"):
        if (out_dir / f).exists():
            shutil.copy2(out_dir / f, final / f)
    logger.info("exported final models to %s", final)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", choices=["cnn", "gbm", "both"], default="both")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="override CNN epochs")
    ap.add_argument("--export-final", action="store_true")
    ap.add_argument("--processed-dir", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    setup_logging(args.log_level)

    data = load_dataset(args.processed_dir)
    meta_path = (
        args.processed_dir or (get_settings().data_dir / "processed")
    ) / "dataset_meta.json"
    data_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    out_dir = _models_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics: dict[str, Any] = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    cnn_cfg, gbm_cfg = load_cnn_config(), load_gbm_config()
    kinds = ["gbm", "cnn"] if args.model == "both" else [args.model]
    for kind in kinds:
        if not args.evaluate_only:
            if kind == "gbm":
                res = train_gbm(data, gbm_cfg, out_dir / "gbm", data_meta=data_meta)
            else:
                res = train_cnn(
                    data, cnn_cfg, out_dir / "cnn", epochs=args.epochs, data_meta=data_meta
                )
            logger.info(
                "%s trained in %.1fs (best epoch/iter %d)", kind, res.seconds, res.best_epoch
            )
        ckpt = out_dir / kind / (GBM_CHECKPOINT if kind == "gbm" else CNN_CHECKPOINT)
        if not ckpt.exists():
            logger.warning("no %s checkpoint at %s; skipping evaluation", kind, ckpt)
            continue
        predictor: Predictor = GBMPredictor(ckpt) if kind == "gbm" else CNNPredictor(ckpt)
        m = evaluate_splits(predictor, data)
        m["checkpoint"] = (
            str(ckpt.relative_to(REPO_ROOT)) if ckpt.is_relative_to(REPO_ROOT) else str(ckpt)
        )
        m["trained_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        metrics[kind] = m
        test = m.get("test", {})
        logger.info(
            "%s test: macro PR-AUC=%.3f  macro F1=%.3f  PC recall@95%%prec=%.3f",
            kind,
            test.get("macro_pr_auc", np.nan),
            test.get("macro_f1", np.nan),
            test.get("pc_recall_at_95_precision", np.nan),
        )
    metrics["generated_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    metrics["dataset"] = {
        k: data_meta.get(k)
        for k in (
            "built_at_utc",
            "n_examples",
            "n_stars",
            "n_synthetic",
            "counts",
            "data_config_hash",
        )
    }
    write_reports(metrics, out_dir)
    print((out_dir / "results.md").read_text())
    if args.export_final:
        export_final(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

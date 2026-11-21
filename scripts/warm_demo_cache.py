#!/usr/bin/env python
"""Pre-compute the demo stars so the demo works with no internet.

Reads ``configs/demo_targets.yaml``, analyzes each target with the current model, and writes
``data/demo_cache/<slug>.json`` plus ``data/demo_cache/index.json``.

Re-run after retraining (results embed the model's predictions).

Examples:
    python scripts/warm_demo_cache.py
    python scripts/warm_demo_cache.py --only kepler-10 wasp-18
    python scripts/warm_demo_cache.py --skip-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from transit_hunter.config import data_path, setup_logging
from transit_hunter.pipeline import PipelineError, analyze, demo_targets_config

logger = logging.getLogger("warm_demo_cache")


def pick_auto_target(rule: dict[str, Any]) -> str:
    """Choose a KIC ID from the KOI labels matching an ``auto:`` rule (deterministic)."""
    from transit_hunter.data.build import load_labels

    labels = load_labels("Kepler")
    df = labels[labels["label"] == rule["label"]].copy()
    lo, hi = rule.get("period_range", [0, 1e9])
    df = df[(df["period"] >= lo) & (df["period"] <= hi)]
    if "min_catalog_depth_ppm" in rule:
        df = df[df["catalog_depth_ppm"] >= rule["min_catalog_depth_ppm"]]
    flag = rule.get("require_flag")
    if flag and flag in df:
        df = df[df[flag] == 1]
    if rule["label"] == "FP":
        df = df[df["catalog_depth_ppm"].between(300, 5000)]
    # One signal per star keeps the demo story simple.
    df = df[df.groupby("star_id")["sample_id"].transform("count") == 1]
    if df.empty:
        raise LookupError(f"no KOI matches auto rule {rule}")
    df = df.sort_values(["catalog_depth_ppm", "sample_id"], ascending=[False, True])
    choice = df.iloc[len(df) // 4 if rule["label"] == "FP" else 0]
    logger.info(
        "auto %s -> %s (%s, P=%.3f d)",
        rule["label"],
        choice["star_id"],
        choice["sample_id"],
        choice["period"],
    )
    return str(choice["star_id"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="*", help="slugs to (re)compute")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    setup_logging(args.log_level)

    out_dir = data_path("demo_cache")
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"targets": []}
    by_slug = {t["slug"]: t for t in index["targets"]}
    failures = 0

    for entry in demo_targets_config():
        slug = entry["slug"]
        if args.only and slug not in args.only:
            continue
        if args.skip_existing and (out_dir / f"{slug}.json").exists() and slug in by_slug:
            logger.info("skip %s (exists)", slug)
            continue
        try:
            query = entry.get("query") or pick_auto_target(entry["auto"])
            logger.info("analyzing %s (%s)", slug, query)
            res = analyze(
                query,
                mission=entry.get("mission"),
                max_segments=entry.get("max_segments"),
                force=True,
            )
        except (PipelineError, LookupError, FileNotFoundError) as exc:
            failures += 1
            logger.error("FAILED %s: %s", slug, exc)
            continue
        res = res.model_copy(update={"name": entry.get("title")})
        (out_dir / f"{slug}.json").write_text(res.model_dump_json())
        top = next((c for c in res.candidates if c.significant), res.candidates[0])
        by_slug[slug] = {
            "slug": slug,
            "title": entry.get("title", slug),
            "query": query,
            "target_id": res.target_id,
            "mission": res.lightcurve.mission,
            "category": entry.get("category", ""),
            "blurb": entry.get("blurb", ""),
            "predicted_class": top.predicted_class,
            "confidence": top.confidence,
            "author": res.lightcurve.author,
            "segments": res.lightcurve.segments,
            "cached_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        # Save after every star so Ctrl-C never loses finished results.
        index["targets"] = list(by_slug.values())
        index_path.write_text(json.dumps(index, indent=2))
        logger.info(
            "%s: %s P=%.4f d SNR=%.1f -> %s (%.0f%%), %.1fs",
            slug,
            res.target_id,
            top.period,
            top.snr,
            top.predicted_class,
            100 * (top.confidence or 0),
            res.timings_s.get("total", 0),
        )

    order = [e["slug"] for e in demo_targets_config()]
    index["targets"] = sorted(
        by_slug.values(), key=lambda t: order.index(t["slug"]) if t["slug"] in order else 99
    )
    index_path.write_text(json.dumps(index, indent=2))
    print(
        pd.DataFrame(index["targets"])[
            ["slug", "target_id", "mission", "predicted_class", "confidence"]
        ].to_string(index=False)
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

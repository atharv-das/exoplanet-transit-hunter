#!/usr/bin/env python
"""Build the training dataset.

Examples:
    # The night before: labels + light curves (resumable; Ctrl-C and re-run any time)
    python scripts/build_dataset.py --stage download --limit 800 --workers 8

    # Turn cached light curves into views/features and write data/processed/
    python scripts/build_dataset.py --stage process
    python scripts/build_dataset.py --stage finalize

    # Everything in one go
    python scripts/build_dataset.py --stage all --limit 800

    # Fallback / augmentation: synthetic planets + EBs injected into real light curves
    python scripts/build_dataset.py --stage synth --synth-count 600

    # Optional TESS domain-shift validation set (never used for training)
    python scripts/build_dataset.py --stage tess --tess-limit 150

    # Training missions (Kepler and/or TESS) come from labels.train_missions in configs/data.yaml;
    # --limit sets the Kepler star count and --tess-limit the TESS star count.

    # Progress of a running download
    python scripts/build_dataset.py --stage status
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter

from transit_hunter.config import load_data_config, setup_logging
from transit_hunter.data import build

logger = logging.getLogger("build_dataset")


def _status() -> None:
    manifest = build.read_manifest()
    counts = Counter(r["status"] for r in manifest.values())
    secs = [r.get("seconds", 0.0) for r in manifest.values() if r.get("status") == "ok"]
    print(f"stars in manifest: {len(manifest)}  {dict(counts)}")
    if secs:
        print(f"mean download time per OK star: {sum(secs) / len(secs):.1f}s")
    errors = [r for r in manifest.values() if r.get("status") == "error"][:5]
    for r in errors:
        print(f"  error {r['star_id']}: {r.get('error', '')[:120]}")


def main(argv: list[str] | None = None) -> int:
    cfg = load_data_config()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stage",
        choices=["labels", "download", "process", "synth", "tess", "finalize", "all", "status"],
        default="all",
    )
    ap.add_argument("--limit", type=int, default=cfg.download.limit, help="number of Kepler stars")
    ap.add_argument("--workers", type=int, default=cfg.download.workers)
    ap.add_argument("--refresh-labels", action="store_true", help="re-query the Exoplanet Archive")
    ap.add_argument("--synth-count", type=int, default=600)
    ap.add_argument(
        "--tess-limit", type=int, default=cfg.download.tess_limit, help="number of TESS stars"
    )
    ap.add_argument(
        "--purge-fits",
        action="store_true",
        help="delete lightkurve FITS files once each star's arrays are cached (saves disk)",
    )
    ap.add_argument("--no-finalize", action="store_true", help="skip the merge after synth/tess")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    setup_logging(args.log_level)
    limit = None if args.limit is not None and args.limit <= 0 else args.limit

    if args.stage == "status":
        _status()
        return 0
    if args.stage == "labels":
        build.stage_labels(refresh=args.refresh_labels, include_tess=True)
    elif args.stage == "download":
        build.stage_labels(refresh=args.refresh_labels)
        for mission in cfg.labels.train_missions:
            n_stars = limit if mission == "Kepler" else args.tess_limit
            logger.info("downloading %s stars for %s", n_stars, mission)
            build.stage_download(n_stars, args.workers, mission=mission, purge=args.purge_fits)
        _status()
    elif args.stage == "process":
        for mission in cfg.labels.train_missions:
            try:
                n = build.stage_process(args.workers, mission=mission)
                logger.info("processed %d real %s examples", n, mission)
            except FileNotFoundError as exc:
                logger.warning("skipping %s: %s", mission, exc)
        build.stage_finalize()
    elif args.stage == "synth":
        n = build.stage_synth(args.synth_count, args.workers)
        logger.info("created %d synthetic examples", n)
        if not args.no_finalize:
            build.stage_finalize()
    elif args.stage == "tess":
        build.stage_labels(refresh=args.refresh_labels, include_tess=True)
        build.stage_download(args.tess_limit, args.workers, mission="TESS", purge=args.purge_fits)
        build.stage_process(args.workers, mission="TESS")
        if not args.no_finalize:
            build.stage_finalize()
    elif args.stage == "finalize":
        meta = build.stage_finalize()
        print(json.dumps(meta["counts"], indent=2))
    elif args.stage == "all":
        meta = build.run_all(
            limit,
            args.workers,
            refresh_labels=args.refresh_labels,
            tess_limit=args.tess_limit,
            purge=args.purge_fits,
        )
        print(json.dumps(meta["counts"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

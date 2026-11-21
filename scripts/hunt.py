#!/usr/bin/env python
"""Hunt mode: batch-scan targets and rank them by planet probability.

Examples:
    python scripts/hunt.py "TIC 261136679" "TIC 100100827" Kepler-10
    python scripts/hunt.py --file targets.txt --mission TESS --out hunt.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from transit_hunter.config import setup_logging
from transit_hunter.pipeline import hunt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("targets", nargs="*", help="names or IDs")
    ap.add_argument("--file", type=Path, help="text file with one target per line")
    ap.add_argument("--mission", choices=["Kepler", "TESS"])
    ap.add_argument("--max-segments", type=int, default=4)
    ap.add_argument("--out", type=Path, help="write the ranking to CSV")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)
    setup_logging(args.log_level)

    targets = list(args.targets)
    if args.file:
        targets += [
            ln.strip()
            for ln in args.file.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    if not targets:
        ap.error("give targets on the command line or with --file")

    def progress(stage: str, frac: float) -> None:
        print(f"[{frac:5.0%}] {stage}", file=sys.stderr)

    result = hunt(targets, mission=args.mission, max_segments=args.max_segments, progress=progress)
    df = pd.DataFrame([r.model_dump() for r in result.ranked])
    cols = [
        "target_id",
        "status",
        "predicted_class",
        "planet_probability",
        "best_period",
        "best_depth_ppm",
        "best_snr",
        "error",
    ]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

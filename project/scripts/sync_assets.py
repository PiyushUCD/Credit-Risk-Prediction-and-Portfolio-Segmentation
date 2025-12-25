"""Copy latest generated plots from results/ into assets/plots for README.

Usage:
    python scripts/sync_assets.py --results-dir project/results --assets-dir project/assets/plots
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_PLOTS = [
    "01_model_performance_comparison.png",
    "02_roc_curves.png",
    "03_confusion_matrix.png",
    "04_feature_importance.png",
    "05_portfolio_analysis.png",
    "06_probability_distribution.png",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--assets-dir", type=str, default=os.path.join("assets", "plots"))
    p.add_argument("--all", action="store_true", help="Copy all PNGs from results/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        raise SystemExit(f"Results dir not found: {results_dir}")

    if args.all:
        to_copy = sorted(results_dir.glob("*.png"))
    else:
        to_copy = [results_dir / p for p in DEFAULT_PLOTS]

    copied = 0
    for src in to_copy:
        if src.exists():
            dst = assets_dir / src.name
            shutil.copy2(src, dst)
            copied += 1

    print(f"✅ Copied {copied} plot(s) into {assets_dir}")


if __name__ == "__main__":
    main()

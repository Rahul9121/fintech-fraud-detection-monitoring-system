from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fraud_monitoring.pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fraud model training + monitoring pipeline.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=12_000,
        help="Number of transactions to sample from the public dataset.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the source dataset even if a local copy exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_training_pipeline(
        sample_size=args.sample_size,
        force_download=args.force_download,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json

from .config import ProjectConfig
from .pipeline import prepare_dataset, train_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare data and train a probabilistic house-pricing model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare-data", help="Build the engineered dataset and feature list.")
    train_parser = subparsers.add_parser(
        "train",
        help="Train quantile models and export portfolio artifacts.",
    )
    train_parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional per-split sample size for a fast smoke test run.",
    )

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="Prepare the data and train the full pipeline.",
    )
    run_all_parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional per-split sample size for a fast smoke test run.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = ProjectConfig()

    if args.command == "prepare-data":
        dataset, feature_columns = prepare_dataset(config)
        print(f"Prepared dataset with {len(dataset):,} rows.")
        print(f"Saved engineered features to {config.paths.processed_dataset}.")
        print(f"Feature count: {len(feature_columns)}")
        return

    if args.command == "train":
        metrics = train_pipeline(config, sample_rows=args.sample_rows)
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "run-all":
        prepare_dataset(config)
        metrics = train_pipeline(config, sample_rows=args.sample_rows)
        print(json.dumps(metrics, indent=2))
        return

    parser.error(f"Unsupported command: {args.command}")

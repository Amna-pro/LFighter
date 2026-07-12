#!/usr/bin/env python3
"""Prepare a leakage-controlled CIC IoT-DIAD 2024 subset for baseline training."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from iot_diad_dataset import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_CLASS_CAPS,
    sample_dataset,
    save_prepared_data,
    set_global_seed,
    stratified_split_and_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CIC IoT-DIAD 2024 for a clean eight-class centralized baseline."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "cic_iot_diad_2024",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--major-class-cap", type=int, default=50_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    set_global_seed(args.seed)

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not dataset_root.is_dir():
        print(f"ERROR: Dataset folder not found: {dataset_root}", file=sys.stderr)
        return 2
    if args.chunksize < 1 or args.major_class_cap < 1:
        print("ERROR: chunksize and major-class-cap must be positive", file=sys.stderr)
        return 2

    caps = dict(DEFAULT_CLASS_CAPS)
    for class_name in CLASS_NAMES:
        if class_name not in {"BruteForce", "Web-Based"}:
            caps[class_name] = args.major_class_cap

    print("CIC IoT-DIAD 2024 preparation")
    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {output_dir}")
    print(f"Class caps: {caps}")
    print("Metadata identifiers are excluded from model features.")
    print()

    dataframe, sampling_report = sample_dataset(
        dataset_root=dataset_root,
        class_caps=caps,
        seed=args.seed,
        chunksize=args.chunksize,
    )

    arrays, preprocessors, split_metadata = stratified_split_and_transform(
        dataframe=dataframe,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    metadata = {
        "dataset": "CIC IoT-DIAD 2024, flow-based features",
        "class_names": list(CLASS_NAMES),
        "class_to_id": {name: idx for idx, name in enumerate(CLASS_NAMES)},
        "seed": args.seed,
        "class_caps": caps,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "sampling": sampling_report,
        "preprocessing": split_metadata,
        "leakage_control": {
            "dropped_columns": ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"],
            "retained_for_first_baseline": ["Src Port", "Dst Port", "Protocol"],
        },
        "elapsed_seconds": round(time.time() - start, 2),
    }

    paths = save_prepared_data(
        output_dir=output_dir,
        arrays=arrays,
        preprocessors=preprocessors,
        metadata=metadata,
    )

    print()
    print("Preparation complete")
    print(f"Train: {arrays['X_train'].shape}")
    print(f"Validation: {arrays['X_val'].shape}")
    print(f"Test: {arrays['X_test'].shape}")
    print(f"Selected features: {split_metadata['selected_feature_count']}")
    print(f"Saved to: {paths.root}")
    print(f"Metadata: {paths.metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

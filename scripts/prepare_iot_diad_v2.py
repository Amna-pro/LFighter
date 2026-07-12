#!/usr/bin/env python3
"""Prepare CIC IoT-DIAD 2024 using a capture-aware, leakage-audited protocol."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from iot_diad_protocol_v2 import (  # noqa: E402
    CLASS_NAMES,
    assign_capture_splits,
    build_capture_inventory,
    class_distribution_table,
    clean_and_deduplicate_samples,
    collect_protocol_samples,
    fit_and_save_feature_configurations,
    make_protocol_paths,
    missingness_table,
    numeric_feature_columns,
    raw_split_counts,
    save_json,
    save_sample_parquets,
    set_global_seed,
)
from research_plotting import (  # noqa: E402
    plot_capture_counts,
    plot_cleaning_summary,
    plot_correlation,
    plot_missingness,
    plot_pca,
    plot_raw_class_distribution,
    plot_split_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the CIC IoT-DIAD 2024 Q1-grade data protocol V2."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "cic_iot_diad_2024_v2",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-cap-per-class", type=int, default=80_000)
    parser.add_argument("--val-cap-per-class", type=int, default=20_000)
    parser.add_argument("--diagnostic-test-cap-per-class", type=int, default=20_000)
    parser.add_argument("--natural-test-total", type=int, default=200_000)
    parser.add_argument("--compact-k", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    set_global_seed(args.seed)
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        print(f"ERROR: Dataset folder not found: {dataset_root}", file=sys.stderr)
        return 2
    if min(
        args.chunksize,
        args.train_cap_per_class,
        args.val_cap_per_class,
        args.diagnostic_test_cap_per_class,
        args.natural_test_total,
        args.compact_k,
    ) < 1:
        print("ERROR: numeric arguments must be positive", file=sys.stderr)
        return 2

    paths = make_protocol_paths(args.output_dir)
    print("CIC IoT-DIAD 2024, Data Protocol V2")
    print(f"Dataset root: {dataset_root}")
    print(f"Output root: {paths.root}")
    print("Protocol: capture-held-out, chronological fallback, exact overlap audit")
    print()

    inventory, canonical_columns = build_capture_inventory(dataset_root)
    manifest = assign_capture_splits(
        inventory,
        seed=args.seed,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    manifest.to_csv(paths.manifest_csv, index=False)
    raw_counts = raw_split_counts(manifest)
    raw_counts.to_csv(paths.tables_dir / "raw_split_class_counts.csv", index=False)
    raw_class = raw_counts.groupby("class_name", as_index=False)["rows"].sum()
    raw_class.to_csv(paths.tables_dir / "raw_class_distribution.csv", index=False)

    datasets, sampling_report = collect_protocol_samples(
        dataset_root=dataset_root,
        manifest=manifest,
        canonical_columns=canonical_columns,
        raw_counts=raw_counts,
        train_cap_per_class=args.train_cap_per_class,
        val_cap_per_class=args.val_cap_per_class,
        diagnostic_test_cap_per_class=args.diagnostic_test_cap_per_class,
        natural_test_total=args.natural_test_total,
        seed=args.seed,
        chunksize=args.chunksize,
    )

    cleaned, cleaning_summary, overlap = clean_and_deduplicate_samples(
        datasets,
        canonical_columns=canonical_columns,
        seed=args.seed,
    )
    cleaning_summary.to_csv(paths.tables_dir / "cleaning_summary.csv", index=False)
    overlap.to_csv(paths.tables_dir / "row_hash_overlap_audit.csv", index=False)
    disallowed_overlap = overlap.loc[~overlap["allowed_overlap"], "overlap_rows"]
    max_disallowed_overlap = int(disallowed_overlap.max()) if len(disallowed_overlap) else 0
    if max_disallowed_overlap != 0:
        raise RuntimeError("Cross-split row hash overlap remains after cleaning")

    parquet_paths = save_sample_parquets(cleaned, paths)
    prepared_distribution = class_distribution_table(cleaned)
    prepared_distribution.to_csv(paths.tables_dir / "prepared_class_distribution.csv", index=False)

    full_features = numeric_feature_columns(canonical_columns, strict_behavioral=False)
    missingness = missingness_table(cleaned["train"], full_features)
    missingness.to_csv(paths.tables_dir / "training_feature_missingness.csv", index=False)

    feature_configs = fit_and_save_feature_configurations(
        cleaned,
        canonical_columns=canonical_columns,
        paths=paths,
        compact_k=args.compact_k,
    )

    plot_raw_class_distribution(raw_class, paths.figures_dir / "raw_class_distribution_log")
    plot_split_distribution(raw_counts, paths.figures_dir / "raw_capture_aware_split_distribution", "Raw Capture-Aware Split Distribution")
    plot_split_distribution(prepared_distribution, paths.figures_dir / "prepared_split_distribution", "Prepared Dataset Distribution")
    plot_capture_counts(manifest, paths.figures_dir / "capture_split_allocation")
    plot_missingness(missingness, paths.figures_dir / "training_missingness_top20")
    plot_cleaning_summary(cleaning_summary, paths.figures_dir / "duplicate_removal_by_split")

    compact_key = next(key for key in feature_configs if key.startswith("compact_k"))
    compact_path = Path(str(feature_configs[compact_key]["arrays"]))
    with np.load(compact_path) as arrays:
        plot_pca(
            arrays["X_train"],
            arrays["y_train"],
            CLASS_NAMES,
            paths.figures_dir / "pca_compact_training",
            seed=args.seed,
        )

    correlation = plot_correlation(
        cleaned["train"],
        paths.figures_dir / "behavioral_feature_correlation",
        seed=args.seed,
    )
    correlation.to_csv(paths.tables_dir / "behavioral_feature_correlation.csv")

    metadata = {
        "dataset": "CIC IoT-DIAD 2024, flow-based features",
        "protocol_version": "2.0",
        "classes": list(CLASS_NAMES),
        "seed": args.seed,
        "split_strategy": {
            "primary": "capture-held-out group split",
            "fallback": "chronological row blocks for classes with fewer than three capture files",
            "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        },
        "sampling_caps": {
            "train_per_class": args.train_cap_per_class,
            "validation_per_class": args.val_cap_per_class,
            "diagnostic_test_per_class": args.diagnostic_test_cap_per_class,
            "natural_test_total": args.natural_test_total,
        },
        "canonical_column_count": len(canonical_columns),
        "canonical_columns": canonical_columns,
        "capture_files": len(manifest),
        "split_modes": manifest["split_mode"].value_counts().to_dict(),
        "sampling_report": sampling_report,
        "cleaning_summary": cleaning_summary.to_dict(orient="records"),
        "row_hash_overlap": overlap.to_dict(orient="records"),
        "maximum_disallowed_row_hash_overlap": max_disallowed_overlap,
        "sample_parquets": parquet_paths,
        "feature_configurations": feature_configs,
        "outputs": {
            "tables": str(paths.tables_dir),
            "figures_png_pdf": str(paths.figures_dir),
            "arrays": str(paths.arrays_dir),
            "preprocessors": str(paths.preprocessors_dir),
        },
        "elapsed_seconds": round(time.time() - start, 2),
        "scientific_scope": (
            "This protocol prevents exact prepared-row overlap and holds out entire captures "
            "where possible. It does not claim that IP addresses are verified physical devices."
        ),
    }
    save_json(paths.metadata_json, metadata)

    print()
    print("Protocol V2 preparation complete")
    for split, frame in cleaned.items():
        print(f"{split}: {frame.shape}")
    print(f"Feature configurations: {', '.join(feature_configs)}")
    print(f"Maximum disallowed cross-split row-hash overlap: {max_disallowed_overlap}")
    print(f"CSV tables: {paths.tables_dir}")
    print(f"PNG and PDF figures: {paths.figures_dir}")
    print(f"Metadata: {paths.metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

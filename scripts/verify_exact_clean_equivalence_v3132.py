#!/usr/bin/env python3
"""Verify exact clean training equivalence for a V3.13.2 source pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


GENERIC_VALIDATION_METRICS = [
    "val_accuracy",
    "val_balanced_accuracy",
    "val_macro_f1",
    "val_weighted_f1",
    "val_mcc",
    "val_log_loss",
    "val_ece_15bin",
]
TRAINING_METRICS = [
    "mean_local_train_loss",
    "mean_local_train_accuracy",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--original-clean-dir", required=True, type=Path)
    p.add_argument("--current-clean-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--source-class", required=True)
    p.add_argument("--target-class", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    out = a.output_dir.expanduser().resolve()
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    original = pd.read_csv(
        a.original_clean_dir / "tables" / "continuation_round_metrics.csv"
    ).sort_values("monitoring_round").reset_index(drop=True)
    current = pd.read_csv(
        a.current_clean_dir / "tables" / "continuation_round_metrics.csv"
    ).sort_values("monitoring_round").reset_index(drop=True)

    metrics = [
        name
        for name in GENERIC_VALIDATION_METRICS + TRAINING_METRICS
        if name in original.columns and name in current.columns
    ]
    if len(metrics) != len(GENERIC_VALIDATION_METRICS + TRAINING_METRICS):
        missing = sorted(
            set(GENERIC_VALIDATION_METRICS + TRAINING_METRICS) - set(metrics)
        )
        raise RuntimeError(f"Missing generic equivalence metrics: {missing}")

    metric_rows = []
    for metric in metrics:
        difference = float(
            np.max(
                np.abs(
                    original[metric].to_numpy(dtype=float)
                    - current[metric].to_numpy(dtype=float)
                )
            )
        )
        metric_rows.append({
            "metric": metric,
            "maximum_absolute_difference": difference,
        })
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(
        tables / "v3132_clean_equivalence_metric_differences.csv",
        index=False,
    )
    maximum_metric_difference = float(
        metric_table["maximum_absolute_difference"].max()
    )

    original_state = torch.load(
        a.original_clean_dir
        / "checkpoints" / "continuation_last_round_model.pt",
        map_location="cpu",
        weights_only=False,
    )["model_state_dict"]
    current_state = torch.load(
        a.current_clean_dir
        / "checkpoints" / "continuation_last_round_model.pt",
        map_location="cpu",
        weights_only=False,
    )["model_state_dict"]

    maximum_state_difference = max(
        float(
            torch.max(
                torch.abs(
                    original_state[key].double()
                    - current_state[key].double()
                )
            )
        )
        for key in original_state
        if torch.is_floating_point(original_state[key])
    )

    passed = bool(
        maximum_metric_difference <= 1e-8
        and maximum_state_difference <= 1e-8
    )
    summary = pd.DataFrame([{
        "seed": a.seed,
        "source_class": a.source_class,
        "target_class": a.target_class,
        "maximum_generic_metric_difference": maximum_metric_difference,
        "maximum_model_state_difference": maximum_state_difference,
        "passes_exact_clean_training_equivalence_1e8": passed,
        "pair_specific_metrics_excluded_from_equivalence_check": True,
        "test_sets_accessed": False,
    }])
    summary.to_csv(
        tables / "v3132_clean_equivalence_verification.csv",
        index=False,
    )
    with (out / "v3132_clean_equivalence_verification.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary.iloc[0].to_dict(), handle, indent=2)

    print("V3.13.2 exact clean-training equivalence")
    print(summary.to_string(index=False))
    if not passed:
        raise RuntimeError(
            "Exact V3.10 plain FedAvg adapter failed clean equivalence"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

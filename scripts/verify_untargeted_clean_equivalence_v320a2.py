#!/usr/bin/env python3
"""Verify exact clean chronology for the V3.20A.2 qualification audit."""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--original-clean-dir", required=True, type=Path)
    p.add_argument("--audit-clean-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", required=True, type=int)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    original_dir = a.original_clean_dir.expanduser().resolve()
    audit_dir = a.audit_clean_dir.expanduser().resolve()
    output = a.output_dir.expanduser().resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    original = pd.read_csv(
        original_dir / "tables" / "continuation_round_metrics.csv"
    ).sort_values("monitoring_round").reset_index(drop=True)
    audit = pd.read_csv(
        audit_dir / "tables" / "reconstruction_round_metrics.csv"
    ).sort_values("monitoring_round").reset_index(drop=True)

    if len(original) != 4 or len(audit) != 4:
        raise RuntimeError(
            f"Expected four clean rounds, found original={len(original)}, "
            f"audit={len(audit)}"
        )

    required = GENERIC_VALIDATION_METRICS + TRAINING_METRICS
    missing = [
        metric
        for metric in required
        if metric not in original.columns or metric not in audit.columns
    ]
    if missing:
        raise RuntimeError(
            "Missing clean-equivalence metrics: " + ", ".join(missing)
        )

    metric_rows = []
    for metric in required:
        difference = float(
            np.max(
                np.abs(
                    original[metric].to_numpy(dtype=float)
                    - audit[metric].to_numpy(dtype=float)
                )
            )
        )
        metric_rows.append({
            "metric": metric,
            "maximum_absolute_difference": difference,
        })
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(
        tables / "v320a2_clean_equivalence_metric_differences.csv",
        index=False,
    )
    maximum_metric_difference = float(
        metric_table["maximum_absolute_difference"].max()
    )

    original_state = torch.load(
        original_dir / "checkpoints" / "continuation_last_round_model.pt",
        map_location="cpu",
        weights_only=False,
    )["model_state_dict"]
    audit_state = torch.load(
        audit_dir / "checkpoints" / "reconstruction_last_round_model.pt",
        map_location="cpu",
        weights_only=False,
    )["model_state_dict"]

    if set(original_state) != set(audit_state):
        raise RuntimeError("Clean checkpoint parameter keys differ")

    floating_differences = []
    exact_nonfloating = True
    for key in original_state:
        left = original_state[key]
        right = audit_state[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"Checkpoint tensor mismatch for {key}")
        if torch.is_floating_point(left):
            floating_differences.append(
                float(
                    torch.max(
                        torch.abs(left.double() - right.double())
                    )
                )
            )
        elif not torch.equal(left, right):
            exact_nonfloating = False

    maximum_state_difference = (
        max(floating_differences) if floating_differences else 0.0
    )
    passed = bool(
        maximum_metric_difference <= 1e-8
        and maximum_state_difference <= 1e-8
        and exact_nonfloating
    )

    result = {
        "experiment_version": "3.20A.2",
        "seed": int(a.seed),
        "maximum_generic_metric_difference":
            maximum_metric_difference,
        "maximum_model_state_difference":
            maximum_state_difference,
        "nonfloating_state_exact": exact_nonfloating,
        "passes_exact_clean_training_equivalence_1e8": passed,
        "test_sets_accessed": False,
    }
    pd.DataFrame([result]).to_csv(
        tables / "v320a2_clean_equivalence_verification.csv",
        index=False,
    )
    with (output / "v320a2_clean_equivalence_verification.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)

    print("V3.20A.2 exact clean-training equivalence")
    print(pd.DataFrame([result]).to_string(index=False))
    if not passed:
        raise RuntimeError(
            "V3.20A.2 clean audit did not exactly reproduce the frozen "
            "clean continuation"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

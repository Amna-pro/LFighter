#!/usr/bin/env python3
"""Verify V3.20B adapter against the exact frozen clean continuation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


GENERIC_METRICS = [
    "val_accuracy",
    "val_balanced_accuracy",
    "val_macro_f1",
    "val_weighted_f1",
    "val_mcc",
    "val_log_loss",
    "val_ece_15bin",
    "mean_local_train_loss",
    "mean_local_train_accuracy",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--original-clean-dir", required=True, type=Path)
    p.add_argument("--adapter-clean-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", required=True, type=int)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    original = a.original_clean_dir.expanduser().resolve()
    adapter = a.adapter_clean_dir.expanduser().resolve()
    output = a.output_dir.expanduser().resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    left = pd.read_csv(
        original / "tables" / "continuation_round_metrics.csv"
    ).sort_values("monitoring_round").reset_index(drop=True)
    right = pd.read_csv(
        adapter / "tables" / "reconstruction_round_metrics.csv"
    ).sort_values("monitoring_round").reset_index(drop=True)

    if len(left) != 4 or len(right) != 4:
        raise RuntimeError("Expected four clean continuation rounds")

    metric_rows = []
    for metric in GENERIC_METRICS:
        if metric not in left.columns or metric not in right.columns:
            raise RuntimeError(f"Missing equivalence metric: {metric}")
        difference = float(
            np.max(
                np.abs(
                    left[metric].to_numpy(dtype=float)
                    - right[metric].to_numpy(dtype=float)
                )
            )
        )
        metric_rows.append({
            "metric": metric,
            "maximum_absolute_difference": difference,
        })
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(
        tables / "v320b_clean_equivalence_metric_differences.csv",
        index=False,
    )

    left_state = torch.load(
        original / "checkpoints" / "continuation_last_round_model.pt",
        map_location="cpu",
        weights_only=False,
    )["model_state_dict"]
    right_state = torch.load(
        adapter / "checkpoints" / "reconstruction_last_round_model.pt",
        map_location="cpu",
        weights_only=False,
    )["model_state_dict"]

    if set(left_state) != set(right_state):
        raise RuntimeError("Clean checkpoint parameter keys differ")

    maximum_state_difference = 0.0
    exact_nonfloating = True
    for key in left_state:
        left_tensor = left_state[key]
        right_tensor = right_state[key]
        if left_tensor.shape != right_tensor.shape:
            raise RuntimeError(f"Shape mismatch for {key}")
        if torch.is_floating_point(left_tensor):
            maximum_state_difference = max(
                maximum_state_difference,
                float(
                    torch.max(
                        torch.abs(
                            left_tensor.double()
                            - right_tensor.double()
                        )
                    )
                ),
            )
        elif not torch.equal(left_tensor, right_tensor):
            exact_nonfloating = False

    maximum_metric_difference = float(
        metric_table["maximum_absolute_difference"].max()
    )
    passed = bool(
        maximum_metric_difference <= 1e-8
        and maximum_state_difference <= 1e-8
        and exact_nonfloating
    )
    result = {
        "experiment_version": "3.20B",
        "seed": int(a.seed),
        "maximum_generic_metric_difference":
            maximum_metric_difference,
        "maximum_model_state_difference":
            maximum_state_difference,
        "nonfloating_state_exact": exact_nonfloating,
        "passes_exact_clean_training_equivalence_1e8":
            passed,
        "test_sets_accessed": False,
    }
    pd.DataFrame([result]).to_csv(
        tables / "v320b_clean_equivalence_verification.csv",
        index=False,
    )
    with (output / "v320b_clean_equivalence_verification.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)

    print("V3.20B exact clean-training equivalence")
    print(pd.DataFrame([result]).to_string(index=False))
    if not passed:
        raise RuntimeError(
            "V3.20B adapter failed exact clean equivalence"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

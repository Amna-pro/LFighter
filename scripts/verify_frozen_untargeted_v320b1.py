#!/usr/bin/env python3
"""Verify exact clean or exact attacked pairing for V3.20B.1."""
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
    p.add_argument("--reference-dir", required=True, type=Path)
    p.add_argument("--adapter-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--verification-kind", choices=["clean", "attack"], required=True)
    p.add_argument("--attack-type", default="")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    reference = a.reference_dir.expanduser().resolve()
    adapter = a.adapter_dir.expanduser().resolve()
    output = a.output_dir.expanduser().resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    left = pd.read_csv(
        reference / "tables" / "continuation_round_metrics.csv"
    ).sort_values("monitoring_round").reset_index(drop=True)
    right = pd.read_csv(
        adapter / "tables" / "continuation_round_metrics.csv"
    ).sort_values("monitoring_round").reset_index(drop=True)
    if len(left) != 4 or len(right) != 4:
        raise RuntimeError("Expected four paired rounds")

    rows = []
    for metric in GENERIC_METRICS:
        if metric not in left.columns or metric not in right.columns:
            raise RuntimeError(f"Missing metric: {metric}")
        difference = float(
            np.max(
                np.abs(
                    left[metric].to_numpy(dtype=float)
                    - right[metric].to_numpy(dtype=float)
                )
            )
        )
        rows.append({
            "metric": metric,
            "maximum_absolute_difference": difference,
        })
    metric_table = pd.DataFrame(rows)
    metric_table.to_csv(
        tables / "v320b1_equivalence_metric_differences.csv",
        index=False,
    )

    left_state = torch.load(
        reference / "checkpoints" / "continuation_last_round_model.pt",
        map_location="cpu",
        weights_only=False,
    )["model_state_dict"]
    right_state = torch.load(
        adapter / "checkpoints" / "v320b1_last_round_model.pt",
        map_location="cpu",
        weights_only=False,
    )["model_state_dict"]
    if set(left_state) != set(right_state):
        raise RuntimeError("Checkpoint parameter keys differ")

    maximum_state_difference = 0.0
    nonfloating_exact = True
    for key in left_state:
        x = left_state[key]
        y = right_state[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise RuntimeError(f"Tensor mismatch for {key}")
        if torch.is_floating_point(x):
            maximum_state_difference = max(
                maximum_state_difference,
                float(
                    torch.max(
                        torch.abs(x.double() - y.double())
                    )
                ),
            )
        elif not torch.equal(x, y):
            nonfloating_exact = False

    maximum_metric_difference = float(
        metric_table["maximum_absolute_difference"].max()
    )
    passed = bool(
        maximum_metric_difference <= 1e-8
        and maximum_state_difference <= 1e-8
        and nonfloating_exact
    )
    result = {
        "experiment_version": "3.20B.1",
        "verification_kind": a.verification_kind,
        "attack_type": a.attack_type,
        "seed": int(a.seed),
        "maximum_generic_metric_difference": maximum_metric_difference,
        "maximum_model_state_difference": maximum_state_difference,
        "nonfloating_state_exact": nonfloating_exact,
        "passes_exact_equivalence_1e8": passed,
        "test_sets_accessed": False,
    }
    pd.DataFrame([result]).to_csv(
        tables / "v320b1_equivalence_verification.csv",
        index=False,
    )
    with (output / "v320b1_equivalence_verification.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)

    print("V3.20B.1 exact equivalence verification")
    print(pd.DataFrame([result]).to_string(index=False))
    if not passed:
        raise RuntimeError("V3.20B.1 exact equivalence failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

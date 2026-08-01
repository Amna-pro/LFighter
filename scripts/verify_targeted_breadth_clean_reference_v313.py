#!/usr/bin/env python3
"""Verify that the V3.13 clean reference exactly reproduces V3.10.1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--original-clean-dir", required=True, type=Path)
    p.add_argument("--v313-clean-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", required=True, type=int)
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
        a.v313_clean_dir / "tables" / "reconstruction_round_metrics.csv"
    ).sort_values("monitoring_round").reset_index(drop=True)

    common = sorted(
        set(column for column in original.columns if column.startswith("val_"))
        & set(current.columns)
    )
    metric_rows = []
    for column in common:
        difference = float(
            np.max(
                np.abs(
                    original[column].to_numpy(dtype=float)
                    - current[column].to_numpy(dtype=float)
                )
            )
        )
        metric_rows.append({
            "metric": column,
            "maximum_absolute_difference": difference,
        })
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(
        tables / "v313_clean_reference_metric_differences.csv",
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
        a.v313_clean_dir
        / "checkpoints" / "reconstruction_last_round_model.pt",
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
        "maximum_metric_difference": maximum_metric_difference,
        "maximum_model_state_difference": maximum_state_difference,
        "passes_exact_clean_reference_1e8": passed,
        "test_sets_accessed": False,
    }])
    summary.to_csv(
        tables / "v313_clean_reference_verification.csv",
        index=False,
    )
    with (out / "v313_clean_reference_verification.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary.iloc[0].to_dict(), handle, indent=2)

    print("V3.13 clean-reference verification")
    print(summary.to_string(index=False))
    if not passed:
        raise RuntimeError(
            "V3.13 clean reference does not reproduce V3.10.1 exactly"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

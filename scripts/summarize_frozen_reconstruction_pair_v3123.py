#!/usr/bin/env python3
"""Summarize one held-out V3.12.3 paired chronology."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plain-clean-dir", required=True, type=Path)
    p.add_argument("--plain-attack-dir", required=True, type=Path)
    p.add_argument("--reconstruction-clean-dir", required=True, type=Path)
    p.add_argument("--reconstruction-attack-dir", required=True, type=Path)
    p.add_argument("--oracle-clean-replacement-attack-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", required=True, type=int)
    return p.parse_args()


def load(path: Path, filename: str) -> pd.DataFrame:
    full = path / "tables" / filename
    if not full.exists():
        raise FileNotFoundError(full)
    table = pd.read_csv(full).sort_values("monitoring_round").reset_index(drop=True)
    if len(table) != 4:
        raise ValueError(f"Expected four rounds in {full}")
    return table


def excess_removed(method_attack, method_clean, plain_attack, plain_clean):
    denominator = plain_attack - plain_clean
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(1.0 - (method_attack - method_clean) / denominator)


def main():
    a = parse_args()
    out = a.output_dir.expanduser().resolve()
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    pc = load(a.plain_clean_dir, "continuation_round_metrics.csv")
    pa = load(a.plain_attack_dir, "continuation_round_metrics.csv")
    rc = load(a.reconstruction_clean_dir, "reconstruction_round_metrics.csv")
    ra = load(a.reconstruction_attack_dir, "reconstruction_round_metrics.csv")
    oa = load(a.oracle_clean_replacement_attack_dir, "reconstruction_round_metrics.csv")

    pc_rate = float(pc["val_source_to_target_rate"].mean())
    pa_rate = float(pa["val_source_to_target_rate"].mean())
    rc_rate = float(rc["val_source_to_target_rate"].mean())
    ra_rate = float(ra["val_source_to_target_rate"].mean())

    common_metrics = sorted(
        set(column for column in pc.columns if column.startswith("val_"))
        & set(oa.columns)
    )
    oracle_max_difference = float(
        max(
            np.max(
                np.abs(
                    pc[column].to_numpy(dtype=float)
                    - oa[column].to_numpy(dtype=float)
                )
            )
            for column in common_metrics
        )
    )

    row = {
        "seed": a.seed,
        "plain_clean_mean_macro_f1": float(pc["val_macro_f1"].mean()),
        "reconstruction_clean_mean_macro_f1": float(rc["val_macro_f1"].mean()),
        "reconstruction_clean_macro_f1_delta": float(
            rc["val_macro_f1"].mean() - pc["val_macro_f1"].mean()
        ),
        "plain_clean_mean_source_to_target_rate": pc_rate,
        "plain_attack_mean_source_to_target_rate": pa_rate,
        "reconstruction_clean_mean_source_to_target_rate": rc_rate,
        "reconstruction_attack_mean_source_to_target_rate": ra_rate,
        "reconstruction_attack_absolute_reduction": float(pa_rate - ra_rate),
        "reconstruction_attack_excess_removed": excess_removed(
            ra_rate, rc_rate, pa_rate, pc_rate
        ),
        "rounds_improved_vs_plain_attack": int(
            np.sum(
                ra["val_source_to_target_rate"].to_numpy()
                < pa["val_source_to_target_rate"].to_numpy()
            )
        ),
        "mean_malicious_recall": float(ra["malicious_recall"].mean()),
        "minimum_malicious_recall": float(ra["malicious_recall"].min()),
        "mean_benign_fpr": float(ra["benign_false_positive_rate"].mean()),
        "maximum_benign_fpr": float(ra["benign_false_positive_rate"].max()),
        "oracle_maximum_metric_difference_from_clean": oracle_max_difference,
        "passes_oracle_exact_1e8": bool(oracle_max_difference <= 1e-8),
        "passes_clean_macro_penalty_002": bool(
            rc["val_macro_f1"].mean() - pc["val_macro_f1"].mean() >= -0.02
        ),
        "passes_positive_attack_reduction": bool(ra_rate < pa_rate),
        "passes_attack_excess_removed_040": bool(
            excess_removed(ra_rate, rc_rate, pa_rate, pc_rate) >= 0.40
        ),
        "test_sets_accessed": False,
    }
    pd.DataFrame([row]).to_csv(
        tables / "v3123_seed_summary.csv", index=False
    )

    round_table = pd.DataFrame({
        "seed": a.seed,
        "monitoring_round": pc["monitoring_round"],
        "plain_clean_source_to_target_rate": pc["val_source_to_target_rate"],
        "plain_attack_source_to_target_rate": pa["val_source_to_target_rate"],
        "reconstruction_clean_source_to_target_rate": rc["val_source_to_target_rate"],
        "reconstruction_attack_source_to_target_rate": ra["val_source_to_target_rate"],
        "oracle_clean_replacement_source_to_target_rate": oa["val_source_to_target_rate"],
    })
    round_table["reconstruction_improved_vs_plain_attack"] = (
        round_table["reconstruction_attack_source_to_target_rate"]
        < round_table["plain_attack_source_to_target_rate"]
    )
    round_table.to_csv(tables / "v3123_seed_rounds.csv", index=False)

    with (out / "v3123_seed_summary_metadata.json").open("w", encoding="utf-8") as h:
        json.dump({
            "experiment_version": "3.12.3",
            "seed": a.seed,
            "frozen_policy": "center_plus_residual",
            "rng_isolation_patch": "3.12.2",
            "oracle_exact": bool(oracle_max_difference <= 1e-8),
            "test_sets_accessed": False,
        }, h, indent=2)

    print("V3.12.3 seed summary complete")
    print(pd.DataFrame([row]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize one exact-baseline V3.13.2 source/seed experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-dir", required=True, type=Path)
    p.add_argument("--plain-dir", required=True, type=Path)
    p.add_argument("--defended-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--source-class", required=True)
    p.add_argument("--target-class", required=True)
    p.add_argument("--coalition-manifest", required=True, type=Path)
    return p.parse_args()


def load(path: Path, filename: str) -> pd.DataFrame:
    full = path / "tables" / filename
    if not full.exists():
        raise FileNotFoundError(full)
    table = pd.read_csv(full).sort_values(
        "monitoring_round"
    ).reset_index(drop=True)
    if len(table) != 4:
        raise ValueError(f"Expected four rounds in {full}")
    return table


def main():
    a = parse_args()
    out = a.output_dir.expanduser().resolve()
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    clean = load(a.clean_dir, "continuation_round_metrics.csv")
    plain = load(a.plain_dir, "continuation_round_metrics.csv")
    defended = load(
        a.defended_dir, "reconstruction_round_metrics.csv"
    )

    coalition = pd.read_csv(a.coalition_manifest)
    coalition = coalition[
        coalition["source_class"].eq(a.source_class)
        & coalition["target_class"].eq(a.target_class)
    ]
    if len(coalition) != 1:
        raise RuntimeError("Expected one matching coalition")
    coalition = coalition.iloc[0]

    clean_rate = float(clean["val_source_to_target_rate"].mean())
    plain_rate = float(plain["val_source_to_target_rate"].mean())
    defended_rate = float(
        defended["val_source_to_target_rate"].mean()
    )
    attack_excess = plain_rate - clean_rate
    residual_excess = defended_rate - clean_rate
    excess_removed = (
        float(1.0 - residual_excess / attack_excess)
        if attack_excess > 1e-12
        else float("nan")
    )

    round_table = pd.DataFrame({
        "seed": a.seed,
        "source_class": a.source_class,
        "target_class": a.target_class,
        "monitoring_round": clean["monitoring_round"],
        "clean_macro_f1": clean["val_macro_f1"],
        "plain_attack_macro_f1": plain["val_macro_f1"],
        "defended_attack_macro_f1": defended["val_macro_f1"],
        "clean_source_to_target_rate": clean[
            "val_source_to_target_rate"
        ],
        "plain_attack_source_to_target_rate": plain[
            "val_source_to_target_rate"
        ],
        "defended_attack_source_to_target_rate": defended[
            "val_source_to_target_rate"
        ],
        "malicious_recall": defended["malicious_recall"],
        "benign_false_positive_rate": defended[
            "benign_false_positive_rate"
        ],
    })
    round_table["attack_excess"] = (
        round_table["plain_attack_source_to_target_rate"]
        - round_table["clean_source_to_target_rate"]
    )
    round_table["defended_residual_excess"] = (
        round_table["defended_attack_source_to_target_rate"]
        - round_table["clean_source_to_target_rate"]
    )
    round_table["defense_improved_vs_plain"] = (
        round_table["defended_attack_source_to_target_rate"]
        < round_table["plain_attack_source_to_target_rate"]
    )
    round_table.to_csv(
        tables / "v313_pair_seed_rounds.csv", index=False
    )

    row = {
        "seed": a.seed,
        "source_class": a.source_class,
        "target_class": a.target_class,
        "pair_name": f"{a.source_class}_to_{a.target_class}",
        "is_development_pair": bool(
            a.source_class == "DDoS"
            and a.target_class == "Benign"
        ),
        "selected_clients": coalition["selected_clients"],
        "coalition_hash_sha256": coalition["coalition_hash_sha256"],
        "coalition_selection_rule": coalition["selection_rule"],
        "global_source_rows": int(coalition["global_source_rows"]),
        "selected_source_rows": int(
            coalition["selected_source_rows"]
        ),
        "global_source_exposure_fraction": float(
            coalition["global_source_exposure_fraction"]
        ),
        "clean_mean_macro_f1": float(clean["val_macro_f1"].mean()),
        "plain_attack_mean_macro_f1": float(
            plain["val_macro_f1"].mean()
        ),
        "defended_attack_mean_macro_f1": float(
            defended["val_macro_f1"].mean()
        ),
        "clean_mean_source_to_target_rate": clean_rate,
        "plain_attack_mean_source_to_target_rate": plain_rate,
        "defended_attack_mean_source_to_target_rate": defended_rate,
        "plain_attack_excess_over_clean": attack_excess,
        "defended_residual_excess_over_clean": residual_excess,
        "absolute_rate_reduction": plain_rate - defended_rate,
        "attack_excess_removed": excess_removed,
        "rounds_improved_vs_plain": int(
            round_table["defense_improved_vs_plain"].sum()
        ),
        "mean_malicious_recall": float(
            defended["malicious_recall"].mean()
        ),
        "minimum_malicious_recall": float(
            defended["malicious_recall"].min()
        ),
        "mean_benign_fpr": float(
            defended["benign_false_positive_rate"].mean()
        ),
        "maximum_benign_fpr": float(
            defended["benign_false_positive_rate"].max()
        ),
        "plain_attack_qualified_absolute_002": bool(
            attack_excess >= 0.02
        ),
        "defense_positive_reduction": bool(
            defended_rate < plain_rate
        ),
        "exact_v310_plain_baseline_used": True,
        "test_sets_accessed": False,
    }
    pd.DataFrame([row]).to_csv(
        tables / "v313_pair_seed_summary.csv", index=False
    )
    with (out / "v313_pair_seed_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "experiment_version": "3.13.2",
            "seed": a.seed,
            "source_class": a.source_class,
            "target_class": a.target_class,
            "exact_v310_plain_baseline_used": True,
            "frozen_detector": True,
            "frozen_reconstruction_policy": "center_plus_residual",
            "test_sets_accessed": False,
        }, handle, indent=2)

    print("V3.13.2 pair/seed summary complete")
    print(pd.DataFrame([row]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

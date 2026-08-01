#!/usr/bin/env python3
"""Aggregate held-out V3.12.3 reconstruction results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seeds", default="7,99,123,2026")
    return p.parse_args()


def save(fig, base):
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    a = parse_args()
    seeds = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]
    out = a.output_dir.expanduser().resolve()
    tables = out / "tables"
    figures = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    seed_tables = []
    round_tables = []
    for seed in seeds:
        base = a.input_root / f"seed_{seed}" / "summary" / "tables"
        seed_tables.append(pd.read_csv(base / "v3123_seed_summary.csv"))
        round_tables.append(pd.read_csv(base / "v3123_seed_rounds.csv"))
    seed_table = pd.concat(seed_tables, ignore_index=True).sort_values("seed")
    round_table = pd.concat(round_tables, ignore_index=True).sort_values(
        ["seed", "monitoring_round"]
    )
    seed_table.to_csv(tables / "v3123_heldout_seed_summary.csv", index=False)
    round_table.to_csv(tables / "v3123_heldout_round_summary.csv", index=False)

    aggregate = pd.DataFrame([{
        "seed_count": len(seed_table),
        "mean_clean_macro_f1_delta": seed_table[
            "reconstruction_clean_macro_f1_delta"
        ].mean(),
        "minimum_clean_macro_f1_delta": seed_table[
            "reconstruction_clean_macro_f1_delta"
        ].min(),
        "mean_plain_attack_source_to_target_rate": seed_table[
            "plain_attack_mean_source_to_target_rate"
        ].mean(),
        "mean_reconstruction_attack_source_to_target_rate": seed_table[
            "reconstruction_attack_mean_source_to_target_rate"
        ].mean(),
        "mean_attack_excess_removed": seed_table[
            "reconstruction_attack_excess_removed"
        ].mean(),
        "median_attack_excess_removed": seed_table[
            "reconstruction_attack_excess_removed"
        ].median(),
        "minimum_attack_excess_removed": seed_table[
            "reconstruction_attack_excess_removed"
        ].min(),
        "positive_reduction_seed_count": int(
            seed_table["passes_positive_attack_reduction"].sum()
        ),
        "excess_removed_040_seed_count": int(
            seed_table["passes_attack_excess_removed_040"].sum()
        ),
        "total_improved_rounds": int(
            round_table["reconstruction_improved_vs_plain_attack"].sum()
        ),
        "total_rounds": int(len(round_table)),
        "mean_malicious_recall": seed_table["mean_malicious_recall"].mean(),
        "minimum_malicious_recall": seed_table[
            "minimum_malicious_recall"
        ].min(),
        "mean_benign_fpr": seed_table["mean_benign_fpr"].mean(),
        "maximum_benign_fpr": seed_table["maximum_benign_fpr"].max(),
        "maximum_oracle_metric_difference": seed_table[
            "oracle_maximum_metric_difference_from_clean"
        ].max(),
        "passes_oracle_exact_all": bool(
            seed_table["passes_oracle_exact_1e8"].all()
        ),
        "passes_clean_penalty_all": bool(
            seed_table["passes_clean_macro_penalty_002"].all()
        ),
        "passes_mean_excess_removed_040": bool(
            seed_table["reconstruction_attack_excess_removed"].mean() >= 0.40
        ),
        "passes_positive_reduction_3_of_4_seeds": bool(
            seed_table["passes_positive_attack_reduction"].sum() >= 3
        ),
        "passes_improved_10_of_16_rounds": bool(
            round_table["reconstruction_improved_vs_plain_attack"].sum() >= 10
        ),
        "frozen_policy": "center_plus_residual",
        "policy_reopened": False,
        "test_sets_accessed": False,
    }])
    aggregate.to_csv(tables / "v3123_heldout_aggregate_decision.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(seed_table))
    width = 0.36
    ax.bar(
        x - width / 2,
        seed_table["plain_attack_mean_source_to_target_rate"],
        width,
        label="Plain attack",
    )
    ax.bar(
        x + width / 2,
        seed_table["reconstruction_attack_mean_source_to_target_rate"],
        width,
        label="Trusted reconstruction",
    )
    ax.set_xticks(x, seed_table["seed"].astype(str))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Held-out development seed")
    ax.set_ylabel("Mean DDoS to Benign rate")
    ax.set_title("V3.12.3 held-out closed-loop reconstruction")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save(fig, figures / "heldout_attack_rate")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        seed_table["seed"].astype(str),
        seed_table["reconstruction_attack_excess_removed"],
    )
    ax.axhline(0.40, linestyle="--", label="Predefined aggregate target")
    ax.set_ylim(-0.25, 1.0)
    ax.set_xlabel("Held-out development seed")
    ax.set_ylabel("Attack-induced excess removed")
    ax.set_title("V3.12.3 reconstruction effectiveness")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save(fig, figures / "heldout_attack_excess_removed")

    with (out / "v3123_aggregate_metadata.json").open("w", encoding="utf-8") as h:
        json.dump({
            "experiment_version": "3.12.3",
            "heldout_development_seeds": seeds,
            "frozen_policy": "center_plus_residual",
            "acceptance_criteria_frozen_before_heldout_review": {
                "oracle_exact_all": True,
                "clean_macro_f1_penalty_each_seed": -0.02,
                "mean_attack_excess_removed": 0.40,
                "positive_reduction_seed_count": 3,
                "improved_round_count": 10,
                "total_round_count": 16,
            },
            "test_sets_accessed": False,
        }, h, indent=2)

    print("V3.12.3 held-out aggregate complete")
    print(aggregate.to_string(index=False))
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Aggregate V3.13 targeted label-flip breadth results."""
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
    p.add_argument(
        "--source-classes",
        default="BruteForce,DDoS,DoS,Mirai,Recon,Spoofing,Web-Based",
    )
    p.add_argument("--target-class", default="Benign")
    return p.parse_args()


def save(fig, base):
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    a = parse_args()
    seeds = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]
    sources = [
        x.strip() for x in a.source_classes.split(",") if x.strip()
    ]
    out = a.output_dir.expanduser().resolve()
    tables = out / "tables"
    figures = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    round_rows = []
    for source in sources:
        pair_slug = f"{source.lower().replace('-', '_')}_to_{a.target_class.lower()}"
        for seed in seeds:
            base = (
                a.input_root.expanduser().resolve()
                / "runs" / pair_slug / f"seed_{seed}"
                / "summary" / "tables"
            )
            seed_rows.append(
                pd.read_csv(base / "v313_pair_seed_summary.csv")
            )
            round_rows.append(
                pd.read_csv(base / "v313_pair_seed_rounds.csv")
            )

    seed_table = pd.concat(seed_rows, ignore_index=True).sort_values(
        ["source_class", "seed"]
    )
    round_table = pd.concat(round_rows, ignore_index=True).sort_values(
        ["source_class", "seed", "monitoring_round"]
    )
    seed_table.to_csv(
        tables / "v313_pair_seed_summary.csv", index=False
    )
    round_table.to_csv(
        tables / "v313_round_summary.csv", index=False
    )

    pair_rows = []
    for source, group in seed_table.groupby("source_class", sort=False):
        qualified_seed_count = int(
            group["plain_attack_qualified_absolute_002"].sum()
        )
        positive_seed_count = int(
            group["defense_positive_reduction"].sum()
        )
        finite_removed = group["attack_excess_removed"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        pair_rows.append({
            "source_class": source,
            "target_class": a.target_class,
            "pair_name": f"{source}_to_{a.target_class}",
            "is_development_pair": bool(source == "DDoS"),
            "seed_count": len(group),
            "selected_clients": group["selected_clients"].iloc[0],
            "global_source_exposure_fraction": float(
                group["global_source_exposure_fraction"].iloc[0]
            ),
            "mean_clean_source_to_target_rate": float(
                group["clean_mean_source_to_target_rate"].mean()
            ),
            "mean_plain_attack_source_to_target_rate": float(
                group["plain_attack_mean_source_to_target_rate"].mean()
            ),
            "mean_defended_attack_source_to_target_rate": float(
                group["defended_attack_mean_source_to_target_rate"].mean()
            ),
            "mean_plain_attack_excess_over_clean": float(
                group["plain_attack_excess_over_clean"].mean()
            ),
            "mean_absolute_rate_reduction": float(
                group["absolute_rate_reduction"].mean()
            ),
            "mean_attack_excess_removed": (
                float(finite_removed.mean())
                if len(finite_removed)
                else float("nan")
            ),
            "minimum_seed_attack_excess_removed": (
                float(finite_removed.min())
                if len(finite_removed)
                else float("nan")
            ),
            "qualified_attack_seed_count": qualified_seed_count,
            "pair_attack_qualified": bool(
                qualified_seed_count >= 3
                and group["plain_attack_excess_over_clean"].mean() >= 0.02
            ),
            "positive_reduction_seed_count": positive_seed_count,
            "total_rounds_improved": int(
                group["rounds_improved_vs_plain"].sum()
            ),
            "mean_malicious_recall": float(
                group["mean_malicious_recall"].mean()
            ),
            "minimum_malicious_recall": float(
                group["minimum_malicious_recall"].min()
            ),
            "mean_benign_fpr": float(
                group["mean_benign_fpr"].mean()
            ),
            "maximum_benign_fpr": float(
                group["maximum_benign_fpr"].max()
            ),
            "defense_positive_on_3_of_4_seeds": bool(
                positive_seed_count >= 3
            ),
        })

    pair_table = pd.DataFrame(pair_rows)
    pair_table.to_csv(
        tables / "v313_pair_aggregate_summary.csv", index=False
    )

    unseen = pair_table[~pair_table["is_development_pair"]].copy()
    qualified_unseen = unseen[unseen["pair_attack_qualified"]].copy()
    decision = pd.DataFrame([{
        "total_pair_count": len(pair_table),
        "unseen_pair_count": len(unseen),
        "qualified_unseen_pair_count": len(qualified_unseen),
        "positive_unseen_pair_count": int(
            unseen["defense_positive_on_3_of_4_seeds"].sum()
        ),
        "mean_unseen_plain_attack_source_to_target_rate": float(
            unseen["mean_plain_attack_source_to_target_rate"].mean()
        ),
        "mean_unseen_defended_attack_source_to_target_rate": float(
            unseen["mean_defended_attack_source_to_target_rate"].mean()
        ),
        "mean_qualified_unseen_attack_excess_removed": (
            float(
                qualified_unseen["mean_attack_excess_removed"].mean()
            )
            if len(qualified_unseen)
            else float("nan")
        ),
        "minimum_qualified_unseen_pair_excess_removed": (
            float(
                qualified_unseen[
                    "mean_attack_excess_removed"
                ].min()
            )
            if len(qualified_unseen)
            else float("nan")
        ),
        "total_improved_rounds": int(
            round_table["defense_improved_vs_plain"].sum()
        ),
        "total_rounds": int(len(round_table)),
        "mean_malicious_recall": float(
            seed_table["mean_malicious_recall"].mean()
        ),
        "minimum_malicious_recall": float(
            seed_table["minimum_malicious_recall"].min()
        ),
        "mean_benign_fpr": float(
            seed_table["mean_benign_fpr"].mean()
        ),
        "maximum_benign_fpr": float(
            seed_table["maximum_benign_fpr"].max()
        ),
        "frozen_detector": True,
        "frozen_reconstruction_policy": "center_plus_residual",
        "method_reopened": False,
        "test_sets_accessed": False,
    }])
    decision.to_csv(
        tables / "v313_breadth_decision.csv", index=False
    )

    x = np.arange(len(pair_table))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width,
        pair_table["mean_clean_source_to_target_rate"],
        width,
        label="Clean",
    )
    ax.bar(
        x,
        pair_table["mean_plain_attack_source_to_target_rate"],
        width,
        label="Plain attacked FedAvg",
    )
    ax.bar(
        x + width,
        pair_table["mean_defended_attack_source_to_target_rate"],
        width,
        label="Trusted reconstruction",
    )
    ax.set_xticks(x, pair_table["source_class"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Source class redirected to Benign")
    ax.set_ylabel("Mean source-to-Benign rate")
    ax.set_title("V3.13 frozen targeted label-flip breadth")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save(fig, figures / "v313_source_to_benign_rates")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        pair_table["source_class"],
        pair_table["mean_attack_excess_removed"],
    )
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Source class redirected to Benign")
    ax.set_ylabel("Mean attack-induced excess removed")
    ax.set_title("V3.13 reconstruction effectiveness by attack pair")
    ax.grid(axis="y", alpha=0.25)
    save(fig, figures / "v313_attack_excess_removed")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        pair_table["mean_malicious_recall"],
        width,
        label="Malicious recall",
    )
    ax.bar(
        x + width / 2,
        pair_table["mean_benign_fpr"],
        width,
        label="Benign FPR",
    )
    ax.set_xticks(x, pair_table["source_class"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Source class redirected to Benign")
    ax.set_ylabel("Rate")
    ax.set_title("V3.13 detector behavior by attack pair")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save(fig, figures / "v313_detection_rates")

    with (out / "v313_aggregate_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "experiment_version": "3.13.2",
            "seeds": seeds,
            "source_classes": sources,
            "target_class": a.target_class,
            "primary_breadth_analysis_excludes_development_pair": True,
            "attack_qualification_rule": {
                "mean_absolute_attack_excess_minimum": 0.02,
                "qualified_seed_count_minimum": 3,
            },
            "frozen_detector": True,
            "frozen_reconstruction_policy": "center_plus_residual",
            "test_sets_accessed": False,
        }, handle, indent=2)

    print("V3.13 targeted attack breadth aggregate complete")
    print(pair_table.to_string(index=False))
    print("\nBREADTH DECISION")
    print(decision.to_string(index=False))
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

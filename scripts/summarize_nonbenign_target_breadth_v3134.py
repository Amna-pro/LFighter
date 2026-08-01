#!/usr/bin/env python3
"""Aggregate V3.13.4 non-Benign target breadth and close Roadmap Task 39."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = [7, 99, 123, 2026]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True, type=Path)
    p.add_argument("--pair-manifest", required=True, type=Path)
    p.add_argument("--task39a-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seeds", default="7,99,123,2026")
    return p.parse_args()


def slug(source: str, target: str) -> str:
    return (
        source.lower().replace("-", "_")
        + "_to_"
        + target.lower().replace("-", "_")
    )


def save(fig, base):
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def classify(row: pd.Series) -> str:
    if bool(row["passes_strict_frozen_criteria"]):
        return "STRICT_PASS"
    if bool(row["passes_mitigation_only_criteria"]):
        return "MITIGATION_PASS_DETECTOR_LIMITED"
    if bool(row["defense_positive_on_3_of_4_seeds"]):
        return "CONSISTENT_PARTIAL_MITIGATION"
    return "NO_CONSISTENT_MITIGATION"


def main() -> int:
    a = parse_args()
    seeds = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]
    if seeds != SEEDS:
        raise ValueError(f"V3.13.4 is frozen to seeds {SEEDS}")

    pairs = pd.read_csv(a.pair_manifest.expanduser().resolve()).sort_values(
        "pair_id"
    )
    if len(pairs) != 7:
        raise RuntimeError("Expected seven non-Benign selected pairs")

    out = a.output_dir.expanduser().resolve()
    tables = out / "tables"
    figures = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    round_rows = []
    for _, pair in pairs.iterrows():
        source = str(pair["source_class"])
        target = str(pair["target_class"])
        pair_slug = slug(source, target)
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
        ["source_class", "target_class", "seed"]
    )
    round_table = pd.concat(round_rows, ignore_index=True).sort_values(
        ["source_class", "target_class", "seed", "monitoring_round"]
    )
    if len(seed_table) != 28 or len(round_table) != 112:
        raise RuntimeError(
            f"Expected 28 seed and 112 round rows, got "
            f"{len(seed_table)} and {len(round_table)}"
        )
    if seed_table["test_sets_accessed"].astype(bool).any():
        raise RuntimeError("A V3.13.4 branch reports test-set access")
    if not seed_table["exact_v310_plain_baseline_used"].astype(bool).all():
        raise RuntimeError("A V3.13.4 branch lacks exact baseline")

    seed_table.to_csv(
        tables / "v3134_pair_seed_summary.csv", index=False
    )
    round_table.to_csv(
        tables / "v3134_round_summary.csv", index=False
    )

    pair_rows: List[Dict[str, object]] = []
    for (source, target), group in seed_table.groupby(
        ["source_class", "target_class"], sort=False
    ):
        finite = group["attack_excess_removed"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        clean_macro = float(group["clean_mean_macro_f1"].mean())
        plain_macro = float(group["plain_attack_mean_macro_f1"].mean())
        defended_macro = float(
            group["defended_attack_mean_macro_f1"].mean()
        )
        attack_loss = clean_macro - plain_macro
        macro_recovery = defended_macro - plain_macro
        qualified_seed_count = int(
            group["plain_attack_qualified_absolute_002"].sum()
        )
        positive_seed_count = int(
            group["defense_positive_reduction"].sum()
        )
        row: Dict[str, object] = {
            "source_class": source,
            "target_class": target,
            "pair_name": f"{source}_to_{target}",
            "seed_count": 4,
            "selected_clients": str(group["selected_clients"].iloc[0]),
            "global_source_exposure_fraction": float(
                group["global_source_exposure_fraction"].iloc[0]
            ),
            "mean_clean_macro_f1": clean_macro,
            "mean_plain_attack_macro_f1": plain_macro,
            "mean_defended_attack_macro_f1": defended_macro,
            "mean_attack_macro_f1_recovery_vs_plain": macro_recovery,
            "macro_f1_loss_recovery_fraction": (
                float(macro_recovery / attack_loss)
                if attack_loss > 1e-12 else float("nan")
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
                float(finite.mean()) if len(finite) else float("nan")
            ),
            "minimum_seed_attack_excess_removed": (
                float(finite.min()) if len(finite) else float("nan")
            ),
            "qualified_attack_seed_count": qualified_seed_count,
            "pair_attack_qualified": bool(
                qualified_seed_count >= 3
                and group["plain_attack_excess_over_clean"].mean() >= 0.02
            ),
            "positive_reduction_seed_count": positive_seed_count,
            "defense_positive_on_3_of_4_seeds": bool(
                positive_seed_count >= 3
            ),
            "total_rounds_improved": int(
                group["rounds_improved_vs_plain"].sum()
            ),
            "total_rounds": 16,
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
            "test_sets_accessed": False,
        }
        row["passes_mitigation_only_criteria"] = bool(
            row["pair_attack_qualified"]
            and positive_seed_count >= 3
            and row["mean_attack_excess_removed"] >= 0.40
            and row["total_rounds_improved"] >= 10
        )
        row["passes_detector_criteria"] = bool(
            row["mean_malicious_recall"] >= 0.90
            and row["minimum_malicious_recall"] >= 0.75
            and row["mean_benign_fpr"] <= 0.05
            and row["maximum_benign_fpr"] <= 0.10
        )
        row["passes_strict_frozen_criteria"] = bool(
            row["passes_mitigation_only_criteria"]
            and row["passes_detector_criteria"]
        )
        pair_rows.append(row)

    pair_table = pd.DataFrame(pair_rows)
    pair_table["evidence_label"] = pair_table.apply(classify, axis=1)
    pair_table.to_csv(
        tables / "v3134_nonbenign_pair_summary.csv", index=False
    )

    task39a = pd.read_csv(
        a.task39a_root.expanduser().resolve()
        / "tables" / "v3133_targeted_breadth_pair_summary.csv"
    )
    common_columns = sorted(
        set(task39a.columns).intersection(pair_table.columns)
    )
    combined = pd.concat(
        [
            task39a[common_columns].assign(
                breadth_group="source_to_benign"
            ),
            pair_table[common_columns].assign(
                breadth_group="selected_nonbenign_target"
            ),
        ],
        ignore_index=True,
    )
    combined.to_csv(
        tables / "v3134_complete_task39_targeted_breadth.csv",
        index=False,
    )

    qualified = pair_table[pair_table["pair_attack_qualified"]].copy()
    decision = {
        "experiment_version": "3.13.4",
        "stage": "task39b_frozen_nonbenign_target_pair_breadth",
        "selected_nonbenign_pair_count": int(len(pair_table)),
        "qualified_nonbenign_pair_count": int(len(qualified)),
        "positive_reduction_pair_count": int(
            pair_table["defense_positive_on_3_of_4_seeds"].sum()
        ),
        "strict_pass_pair_count": int(
            pair_table["passes_strict_frozen_criteria"].sum()
        ),
        "mean_qualified_attack_excess_removed": (
            float(qualified["mean_attack_excess_removed"].mean())
            if len(qualified) else float("nan")
        ),
        "minimum_qualified_attack_excess_removed": (
            float(qualified["mean_attack_excess_removed"].min())
            if len(qualified) else float("nan")
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
        "task39a_status": "COMPLETED",
        "task39b_status": "COMPLETED",
        "task39_targeted_label_flip_breadth_status": "COMPLETED",
        "method_reopened": False,
        "attack_specific_retuning": False,
        "federated_training_rerun_for_task39b": True,
        "natural_test_accessed": False,
        "diagnostic_test_accessed": False,
        "next_stage": "Task 40 frozen untargeted label poisoning",
    }
    pd.DataFrame([decision]).to_csv(
        tables / "v3134_task39_decision.csv", index=False
    )
    with (out / "v3134_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(decision, handle, indent=2)

    x = np.arange(len(pair_table))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6.5))
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
        label="Frozen trusted reconstruction",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        pair_table["pair_name"], rotation=30, ha="right"
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean source-to-target rate")
    ax.set_title("Selected non-Benign targeted label-flip breadth")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save(fig, figures / "v3134_nonbenign_source_target_rates")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(
        pair_table["pair_name"],
        pair_table["mean_attack_excess_removed"],
    )
    ax.axhline(0.40, linestyle="--", linewidth=1)
    ax.axhline(0, linestyle=":", linewidth=1)
    ax.set_xticklabels(
        pair_table["pair_name"], rotation=30, ha="right"
    )
    ax.set_ylabel("Mean attack-induced excess removed")
    ax.set_title("Frozen mitigation on non-Benign targets")
    ax.grid(axis="y", alpha=0.25)
    save(fig, figures / "v3134_nonbenign_excess_removed")

    print("V3.13.4 non-Benign target breadth complete")
    print()
    print("PAIR SUMMARY")
    print(pair_table[
        [
            "pair_name",
            "mean_plain_attack_excess_over_clean",
            "mean_attack_excess_removed",
            "minimum_seed_attack_excess_removed",
            "mean_attack_macro_f1_recovery_vs_plain",
            "total_rounds_improved",
            "mean_malicious_recall",
            "minimum_malicious_recall",
            "mean_benign_fpr",
            "maximum_benign_fpr",
            "passes_mitigation_only_criteria",
            "passes_detector_criteria",
            "passes_strict_frozen_criteria",
            "evidence_label",
        ]
    ].to_string(index=False))
    print()
    print("TASK 39 DECISION")
    print(pd.DataFrame([decision]).to_string(index=False))
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

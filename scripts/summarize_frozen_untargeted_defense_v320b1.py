#!/usr/bin/env python3
"""Summarize exact-paired V3.20B.1 frozen untargeted defense."""
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


ATTACK_TYPES = [
    "all_to_one_benign",
    "cyclic_shift",
    "multiclass_partial_cycle",
    "pairwise_swap",
    "random_flip",
]
SEEDS = [7, 99, 123, 2026]

MITIGATION_REMOVED_THRESHOLD = 0.40
MITIGATION_POSITIVE_SEEDS = 3
MITIGATION_IMPROVED_ROUNDS = 10
DETECTOR_MEAN_RECALL = 0.90
DETECTOR_MIN_RECALL = 0.75
DETECTOR_MEAN_FPR = 0.05
DETECTOR_MAX_FPR = 0.10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--v320a3-root", required=True, type=Path)
    p.add_argument("--v320b1-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def load_rounds(path: Path) -> pd.DataFrame:
    file = path / "tables" / "continuation_round_metrics.csv"
    if not file.exists():
        raise FileNotFoundError(file)
    table = pd.read_csv(file).sort_values(
        "monitoring_round"
    ).reset_index(drop=True)
    if len(table) != 4:
        raise RuntimeError(f"Expected four rounds in {file}")
    return table


def load_classes(path: Path) -> pd.DataFrame:
    file = path / "tables" / "validation_class_metrics_long.csv"
    if not file.exists():
        raise FileNotFoundError(file)
    table = pd.read_csv(file)
    if len(table) != 32:
        raise RuntimeError(f"Expected 32 class rows in {file}")
    return table


def save_figure(fig, base: Path):
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def label(row: pd.Series) -> str:
    if row["passes_strict_frozen_criteria"]:
        return "STRICT_PASS"
    if row["passes_mitigation_only_criteria"]:
        return "MITIGATION_PASS_DETECTOR_LIMITED"
    if row["positive_recovery_seed_count"] >= 3:
        return "CONSISTENT_PARTIAL_MITIGATION"
    return "NO_CONSISTENT_MITIGATION"


def main() -> int:
    a = parse_args()
    a3 = a.v320a3_root.expanduser().resolve()
    b1 = a.v320b1_root.expanduser().resolve()
    output = a.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    qualification = pd.read_csv(
        a3 / "summary" / "tables"
        / "v320a3_attack_qualification_summary.csv"
    )
    qualification = qualification[
        qualification["qualified_for_v320b"].astype(bool)
    ]
    if set(qualification["attack_type"]) != set(ATTACK_TYPES):
        raise RuntimeError("Expected all five qualified attacks")

    verification_files = list(
        (b1 / "verification").glob(
            "**/tables/v320b1_equivalence_verification.csv"
        )
    )
    verification = pd.concat(
        [pd.read_csv(path) for path in verification_files],
        ignore_index=True,
    )
    if len(verification) != 9:
        raise RuntimeError(
            f"Expected 9 equivalence checks, found {len(verification)}"
        )
    if not verification[
        "passes_exact_equivalence_1e8"
    ].astype(bool).all():
        raise RuntimeError("At least one equivalence check failed")
    verification.to_csv(
        tables / "v320b1_all_equivalence_checks.csv",
        index=False,
    )

    seed_rows: List[Dict[str, object]] = []
    round_rows: List[pd.DataFrame] = []
    class_rows: List[pd.DataFrame] = []

    for attack_type in ATTACK_TYPES:
        q = qualification[
            qualification["attack_type"].eq(attack_type)
        ].iloc[0]
        global_primary = bool(
            q["passes_global_macro_f1_qualification"]
        )
        primary_class = str(q["most_consistently_harmed_class"])
        primary_metric = (
            "macro_f1"
            if global_primary
            else f"{primary_class}_recall"
        )

        for seed in SEEDS:
            clean_dir = a3 / "clean_exact" / f"seed_{seed}"
            plain_dir = (
                a3 / "runs" / attack_type
                / f"seed_{seed}" / "plain_fedavg"
            )
            defended_dir = (
                b1 / "runs" / attack_type
                / f"seed_{seed}" / "trusted_reconstruction"
            )
            clean = load_rounds(clean_dir)
            plain = load_rounds(plain_dir)
            defended = load_rounds(defended_dir)

            with (
                defended_dir
                / "frozen_untargeted_defense_v320b1_metadata.json"
            ).open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata["attack_type"] != attack_type:
                raise RuntimeError("Defended attack metadata mismatch")
            if int(metadata["model_seed"]) != seed:
                raise RuntimeError("Defended seed metadata mismatch")
            if not bool(metadata["exact_v320a3_poison_plan_reused"]):
                raise RuntimeError("Exact poison-plan reuse not recorded")
            if bool(metadata["test_sets_accessed"]):
                raise RuntimeError("A defended branch reports test access")

            clean_class_table = load_classes(clean_dir)
            plain_class_table = load_classes(plain_dir)
            defended_class_table = load_classes(defended_dir)

            clean_class = (
                clean_class_table.groupby("class_name")["recall"].mean()
            )
            plain_class = (
                plain_class_table.groupby("class_name")["recall"].mean()
            )
            defended_class = (
                defended_class_table.groupby("class_name")["recall"].mean()
            )
            class_table = pd.DataFrame({
                "class_name": clean_class.index,
                "clean_mean_recall": clean_class.values,
                "plain_attack_mean_recall":
                    plain_class.reindex(clean_class.index).values,
                "defended_attack_mean_recall":
                    defended_class.reindex(clean_class.index).values,
            })
            class_table.insert(0, "attack_type", attack_type)
            class_table.insert(0, "seed", seed)
            class_table["plain_recall_loss"] = (
                class_table["clean_mean_recall"]
                - class_table["plain_attack_mean_recall"]
            )
            class_table["defended_recall_recovery"] = (
                class_table["defended_attack_mean_recall"]
                - class_table["plain_attack_mean_recall"]
            )
            class_rows.append(class_table)

            if global_primary:
                clean_values = clean["val_macro_f1"].to_numpy(dtype=float)
                plain_values = plain["val_macro_f1"].to_numpy(dtype=float)
                defended_values = defended[
                    "val_macro_f1"
                ].to_numpy(dtype=float)
            else:
                clean_values = (
                    clean_class_table.query(
                        "class_name == @primary_class"
                    )
                    .sort_values("monitoring_round")["recall"]
                    .to_numpy(dtype=float)
                )
                plain_values = (
                    plain_class_table.query(
                        "class_name == @primary_class"
                    )
                    .sort_values("monitoring_round")["recall"]
                    .to_numpy(dtype=float)
                )
                defended_values = (
                    defended_class_table.query(
                        "class_name == @primary_class"
                    )
                    .sort_values("monitoring_round")["recall"]
                    .to_numpy(dtype=float)
                )

            damage_values = clean_values - plain_values
            recovery_values = defended_values - plain_values
            positive_damage = damage_values > 1e-12
            clean_primary = float(clean_values.mean())
            plain_primary = float(plain_values.mean())
            defended_primary = float(defended_values.mean())
            mean_damage = clean_primary - plain_primary
            mean_recovery = defended_primary - plain_primary
            removed = (
                float(mean_recovery / mean_damage)
                if mean_damage > 1e-12
                else float("nan")
            )

            current_rounds = pd.DataFrame({
                "attack_type": attack_type,
                "seed": seed,
                "monitoring_round":
                    clean["monitoring_round"].astype(int),
                "primary_metric": primary_metric,
                "clean_primary": clean_values,
                "plain_primary": plain_values,
                "defended_primary": defended_values,
                "plain_damage": damage_values,
                "defense_recovery": recovery_values,
                "defense_improved":
                    defended_values > plain_values,
            })
            round_rows.append(current_rounds)

            seed_rows.append({
                "attack_type": attack_type,
                "seed": seed,
                "primary_metric": primary_metric,
                "primary_class":
                    "" if global_primary else primary_class,
                "clean_primary": clean_primary,
                "plain_attack_primary": plain_primary,
                "defended_attack_primary": defended_primary,
                "plain_damage": mean_damage,
                "defense_recovery": mean_recovery,
                "damage_removed_fraction": removed,
                "defense_positive_recovery": bool(
                    mean_recovery > 0
                ),
                "rounds_improved": int(
                    (defended_values > plain_values).sum()
                ),
                "mean_macro_f1_recovery": float(
                    defended["val_macro_f1"].mean()
                    - plain["val_macro_f1"].mean()
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
                "mean_replaced_clients": float(
                    defended["replaced_clients"].mean()
                ),
                "exact_poison_plan_reused": True,
                "test_sets_accessed": False,
            })

    seed_table = pd.DataFrame(seed_rows).sort_values(
        ["attack_type", "seed"]
    )
    round_table = pd.concat(
        round_rows, ignore_index=True
    ).sort_values(
        ["attack_type", "seed", "monitoring_round"]
    )
    class_table = pd.concat(
        class_rows, ignore_index=True
    ).sort_values(
        ["attack_type", "seed", "class_name"]
    )
    seed_table.to_csv(
        tables / "v320b1_seed_attack_summary.csv",
        index=False,
    )
    round_table.to_csv(
        tables / "v320b1_round_primary_recovery.csv",
        index=False,
    )
    class_table.to_csv(
        tables / "v320b1_class_recall_recovery.csv",
        index=False,
    )

    attack_rows = []
    for attack_type, group in seed_table.groupby(
        "attack_type", sort=False
    ):
        valid_removed = group[
            "damage_removed_fraction"
        ].replace([np.inf, -np.inf], np.nan)
        row = {
            "attack_type": attack_type,
            "primary_metric": group["primary_metric"].iloc[0],
            "mean_plain_damage": float(group["plain_damage"].mean()),
            "minimum_plain_damage": float(group["plain_damage"].min()),
            "mean_defense_recovery": float(
                group["defense_recovery"].mean()
            ),
            "mean_damage_removed_fraction": float(
                valid_removed.mean()
            ),
            "minimum_seed_damage_removed_fraction": float(
                valid_removed.min()
            ),
            "positive_recovery_seed_count": int(
                group["defense_positive_recovery"].sum()
            ),
            "total_rounds_improved": int(
                group["rounds_improved"].sum()
            ),
            "total_rounds": 16,
            "mean_macro_f1_recovery": float(
                group["mean_macro_f1_recovery"].mean()
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
            "mean_replaced_clients": float(
                group["mean_replaced_clients"].mean()
            ),
            "exact_poison_plan_all_seeds": bool(
                group["exact_poison_plan_reused"].all()
            ),
        }
        row["passes_mitigation_only_criteria"] = bool(
            row["positive_recovery_seed_count"]
            >= MITIGATION_POSITIVE_SEEDS
            and row["mean_damage_removed_fraction"]
            >= MITIGATION_REMOVED_THRESHOLD
            and row["total_rounds_improved"]
            >= MITIGATION_IMPROVED_ROUNDS
        )
        row["passes_detector_criteria"] = bool(
            row["mean_malicious_recall"]
            >= DETECTOR_MEAN_RECALL
            and row["minimum_malicious_recall"]
            >= DETECTOR_MIN_RECALL
            and row["mean_benign_fpr"]
            <= DETECTOR_MEAN_FPR
            and row["maximum_benign_fpr"]
            <= DETECTOR_MAX_FPR
        )
        row["passes_strict_frozen_criteria"] = bool(
            row["passes_mitigation_only_criteria"]
            and row["passes_detector_criteria"]
        )
        attack_rows.append(row)

    attack_table = pd.DataFrame(attack_rows).sort_values(
        "attack_type"
    )
    attack_table["evidence_label"] = attack_table.apply(
        label, axis=1
    )
    attack_table.to_csv(
        tables / "v320b1_attack_summary.csv",
        index=False,
    )

    decision = {
        "experiment_version": "3.20B.1",
        "stage":
            "task40_exact_paired_frozen_untargeted_defense",
        "qualified_attack_count": 5,
        "completed_attack_count": int(len(attack_table)),
        "positive_recovery_attack_count": int(
            (attack_table["positive_recovery_seed_count"] >= 3).sum()
        ),
        "mitigation_pass_attack_count": int(
            attack_table[
                "passes_mitigation_only_criteria"
            ].sum()
        ),
        "strict_pass_attack_count": int(
            attack_table[
                "passes_strict_frozen_criteria"
            ].sum()
        ),
        "total_improved_rounds": int(
            round_table["defense_improved"].sum()
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
        "exact_equivalence_check_count": int(
            len(verification)
        ),
        "exact_equivalence_pass_count": int(
            verification[
                "passes_exact_equivalence_1e8"
            ].sum()
        ),
        "exact_v320a3_poison_plan_reused": True,
        "prior_v320b_status":
            "INVALIDATED_BY_CLEAN_EQUIVALENCE_GATE",
        "task40_status": "COMPLETED",
        "method_reopened": False,
        "attack_specific_retuning": False,
        "natural_test_accessed": False,
        "diagnostic_test_accessed": False,
        "next_stage": "Task 41 frozen backdoor and trigger attacks",
    }
    pd.DataFrame([decision]).to_csv(
        tables / "v320b1_task40_decision.csv",
        index=False,
    )
    with (output / "v320b1_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(decision, handle, indent=2)

    x = np.arange(len(attack_table))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(
        x,
        attack_table["mean_damage_removed_fraction"],
    )
    ax.axhline(
        MITIGATION_REMOVED_THRESHOLD,
        linestyle="--",
        linewidth=1,
        label="Mitigation threshold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        attack_table["attack_type"],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Mean primary damage removed")
    ax.set_title("V3.20B.1 frozen untargeted defense")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "v320b1_damage_removed")

    print("V3.20B.1 frozen untargeted defense complete")
    print()
    print("ATTACK SUMMARY")
    columns = [
        "attack_type",
        "primary_metric",
        "mean_plain_damage",
        "mean_damage_removed_fraction",
        "minimum_seed_damage_removed_fraction",
        "positive_recovery_seed_count",
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
    print(attack_table[columns].to_string(index=False))
    print()
    print("TASK 40 DECISION")
    print(pd.DataFrame([decision]).to_string(index=False))
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

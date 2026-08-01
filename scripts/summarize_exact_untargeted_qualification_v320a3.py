#!/usr/bin/env python3
"""Summarize exact plain-only untargeted qualification for V3.20A.3."""
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
    "random_flip",
    "cyclic_shift",
    "pairwise_swap",
    "all_to_one_benign",
    "multiclass_partial_cycle",
]
ALL_SEEDS = [7, 99, 123, 2026]
CONFIRMATORY_SEEDS = [99, 123, 2026]

MIN_MEAN_MACRO_F1_LOSS = 0.02
MIN_MEAN_BALANCED_ACCURACY_LOSS = 0.02
MIN_MEAN_WORST_CLASS_RECALL_LOSS = 0.10
MIN_POSITIVE_CONFIRMATORY_SEEDS = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seeds", default="7,99,123,2026")
    p.add_argument("--confirmatory-seeds", default="99,123,2026")
    return p.parse_args()


def load_rounds(path: Path) -> pd.DataFrame:
    file = path / "tables" / "continuation_round_metrics.csv"
    if not file.exists():
        raise FileNotFoundError(file)
    table = pd.read_csv(file).sort_values(
        "monitoring_round"
    ).reset_index(drop=True)
    if len(table) != 4:
        raise RuntimeError(
            f"Expected four rounds in {file}, found {len(table)}"
        )
    return table


def load_classes(path: Path) -> pd.DataFrame:
    file = path / "tables" / "validation_class_metrics_long.csv"
    if not file.exists():
        raise FileNotFoundError(file)
    table = pd.read_csv(file)
    if len(table) != 32:
        raise RuntimeError(
            f"Expected 32 class rows in {file}, found {len(table)}"
        )
    return table


def load_confusion(path: Path) -> pd.DataFrame:
    file = (
        path / "tables"
        / "validation_confusion_matrix_long.csv"
    )
    if not file.exists():
        raise FileNotFoundError(file)
    table = pd.read_csv(file)
    if len(table) != 256:
        raise RuntimeError(
            f"Expected 256 confusion rows in {file}, found {len(table)}"
        )
    return table


def normalize_confusion(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    totals = result.groupby(
        ["monitoring_round", "true_id"]
    )["count"].transform("sum")
    result["row_rate"] = np.where(
        totals > 0,
        result["count"].to_numpy(dtype=float)
        / totals.to_numpy(dtype=float),
        0.0,
    )
    return result


def save_figure(fig, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def label(row: pd.Series) -> str:
    global_pass = bool(row["passes_global_macro_f1_qualification"])
    bacc_pass = bool(row["passes_balanced_accuracy_qualification"])
    class_pass = bool(row["passes_class_recall_qualification"])
    if global_pass and (bacc_pass or class_pass):
        return "GLOBAL_AND_CLASS_QUALIFIED"
    if global_pass:
        return "GLOBAL_QUALIFIED"
    if bacc_pass or class_pass:
        return "CLASS_SPECIFIC_QUALIFIED"
    return "NOT_QUALIFIED"


def main() -> int:
    a = parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    confirmatory = [
        int(x) for x in a.confirmatory_seeds.split(",")
    ]
    if seeds != ALL_SEEDS:
        raise ValueError(f"Frozen seeds are {ALL_SEEDS}")
    if confirmatory != CONFIRMATORY_SEEDS:
        raise ValueError(
            f"Frozen confirmatory seeds are {CONFIRMATORY_SEEDS}"
        )

    root = a.input_root.expanduser().resolve()
    output = a.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    verification_rows = []
    for seed in seeds:
        file = (
            root / "clean_verification" / f"seed_{seed}"
            / "tables"
            / "v320a3_clean_equivalence_verification.csv"
        )
        if not file.exists():
            raise FileNotFoundError(file)
        verification_rows.append(pd.read_csv(file))
    verification = pd.concat(
        verification_rows,
        ignore_index=True,
    )
    if not verification[
        "passes_exact_clean_training_equivalence_1e8"
    ].astype(bool).all():
        raise RuntimeError(
            "At least one exact clean verification failed"
        )
    verification.to_csv(
        tables / "v320a3_clean_equivalence_all_seeds.csv",
        index=False,
    )

    seed_rows: List[Dict[str, object]] = []
    round_tables: List[pd.DataFrame] = []
    class_tables: List[pd.DataFrame] = []
    confusion_rows: List[Dict[str, object]] = []

    for seed in seeds:
        clean_dir = root / "clean_exact" / f"seed_{seed}"
        clean_rounds = load_rounds(clean_dir)
        clean_classes = load_classes(clean_dir)
        clean_conf = normalize_confusion(
            load_confusion(clean_dir)
        )

        for attack_type in ATTACK_TYPES:
            attack_dir = (
                root / "runs" / attack_type
                / f"seed_{seed}" / "plain_fedavg"
            )
            attack_rounds = load_rounds(attack_dir)
            attack_classes = load_classes(attack_dir)
            attack_conf = normalize_confusion(
                load_confusion(attack_dir)
            )

            metadata_file = (
                attack_dir
                / "exact_untargeted_plain_v320a3_metadata.json"
            )
            with metadata_file.open(
                "r",
                encoding="utf-8",
            ) as handle:
                metadata = json.load(handle)

            if metadata["attack_type"] != attack_type:
                raise RuntimeError(
                    f"Attack metadata mismatch for {attack_type}, "
                    f"seed {seed}"
                )
            if int(metadata["model_seed"]) != seed:
                raise RuntimeError(
                    f"Seed metadata mismatch for {attack_type}, "
                    f"seed {seed}"
                )
            if bool(metadata["test_sets_accessed"]):
                raise RuntimeError(
                    f"Test access reported by {attack_dir}"
                )

            manifest = pd.read_csv(
                attack_dir / "attack_manifest"
                / "malicious_client_poison_manifest.csv"
            )
            malicious = manifest[
                manifest["is_malicious"].astype(bool)
            ]
            mechanics_valid = bool(
                len(malicious) == 8
                and int(malicious["poisoned_rows"].sum()) > 0
                and (
                    malicious["poisoned_rows"].astype(int)
                    == malicious[
                        "changed_rows_verified"
                    ].astype(int)
                ).all()
            )
            if not mechanics_valid:
                raise RuntimeError(
                    f"Invalid mechanics for {attack_type}, "
                    f"seed {seed}"
                )

            current = pd.DataFrame({
                "seed": seed,
                "seed_role": (
                    "exploratory_smoke"
                    if seed == 7
                    else "confirmatory_development"
                ),
                "attack_type": attack_type,
                "monitoring_round":
                    clean_rounds["monitoring_round"].astype(int),
                "clean_macro_f1":
                    clean_rounds["val_macro_f1"],
                "plain_attack_macro_f1":
                    attack_rounds["val_macro_f1"],
                "macro_f1_loss":
                    clean_rounds["val_macro_f1"]
                    - attack_rounds["val_macro_f1"],
                "clean_balanced_accuracy":
                    clean_rounds["val_balanced_accuracy"],
                "plain_attack_balanced_accuracy":
                    attack_rounds["val_balanced_accuracy"],
                "balanced_accuracy_loss":
                    clean_rounds["val_balanced_accuracy"]
                    - attack_rounds["val_balanced_accuracy"],
            })
            round_tables.append(current)

            clean_class = (
                clean_classes.groupby(
                    ["class_id", "class_name"],
                    as_index=False,
                )["recall"].mean()
                .rename(
                    columns={
                        "recall": "clean_mean_recall"
                    }
                )
            )
            attack_class = (
                attack_classes.groupby(
                    ["class_id", "class_name"],
                    as_index=False,
                )["recall"].mean()
                .rename(
                    columns={
                        "recall":
                            "plain_attack_mean_recall"
                    }
                )
            )
            class_loss = clean_class.merge(
                attack_class,
                on=["class_id", "class_name"],
                validate="one_to_one",
            )
            class_loss["recall_loss"] = (
                class_loss["clean_mean_recall"]
                - class_loss["plain_attack_mean_recall"]
            )
            class_loss.insert(0, "attack_type", attack_type)
            class_loss.insert(0, "seed", seed)
            class_tables.append(class_loss)
            worst = class_loss.sort_values(
                ["recall_loss", "class_id"],
                ascending=[False, True],
            ).iloc[0]

            merged_conf = clean_conf.merge(
                attack_conf,
                on=[
                    "monitoring_round",
                    "true_id",
                    "true_class",
                    "predicted_id",
                    "predicted_class",
                ],
                suffixes=("_clean", "_attack"),
                validate="one_to_one",
            )
            merged_conf["absolute_shift"] = np.abs(
                merged_conf["row_rate_clean"]
                - merged_conf["row_rate_attack"]
            )
            per_round_tv = (
                merged_conf.groupby("monitoring_round")[
                    "absolute_shift"
                ].sum()
                / 2.0
            )
            for round_id, shift in per_round_tv.items():
                confusion_rows.append({
                    "seed": seed,
                    "attack_type": attack_type,
                    "monitoring_round": int(round_id),
                    "confusion_total_variation_shift":
                        float(shift),
                })

            seed_rows.append({
                "seed": seed,
                "seed_role": (
                    "exploratory_smoke"
                    if seed == 7
                    else "confirmatory_development"
                ),
                "attack_type": attack_type,
                "changed_malicious_rows": int(
                    malicious[
                        "changed_rows_verified"
                    ].sum()
                ),
                "attack_mechanics_valid": mechanics_valid,
                "clean_mean_macro_f1": float(
                    clean_rounds["val_macro_f1"].mean()
                ),
                "plain_attack_mean_macro_f1": float(
                    attack_rounds["val_macro_f1"].mean()
                ),
                "macro_f1_loss": float(
                    current["macro_f1_loss"].mean()
                ),
                "positive_macro_f1_loss": bool(
                    current["macro_f1_loss"].mean() > 0
                ),
                "clean_mean_balanced_accuracy": float(
                    clean_rounds[
                        "val_balanced_accuracy"
                    ].mean()
                ),
                "plain_attack_mean_balanced_accuracy":
                    float(
                        attack_rounds[
                            "val_balanced_accuracy"
                        ].mean()
                    ),
                "balanced_accuracy_loss": float(
                    current[
                        "balanced_accuracy_loss"
                    ].mean()
                ),
                "positive_balanced_accuracy_loss": bool(
                    current[
                        "balanced_accuracy_loss"
                    ].mean() > 0
                ),
                "worst_class": str(worst["class_name"]),
                "worst_class_recall_loss": float(
                    worst["recall_loss"]
                ),
                "positive_worst_class_recall_loss": bool(
                    float(worst["recall_loss"]) > 0
                ),
                "mean_confusion_total_variation_shift":
                    float(per_round_tv.mean()),
                "test_sets_accessed": False,
            })

    seed_table = pd.DataFrame(seed_rows).sort_values(
        ["attack_type", "seed"]
    )
    round_table = pd.concat(
        round_tables,
        ignore_index=True,
    ).sort_values(
        ["attack_type", "seed", "monitoring_round"]
    )
    class_table = pd.concat(
        class_tables,
        ignore_index=True,
    ).sort_values(["attack_type", "seed", "class_id"])
    confusion_table = pd.DataFrame(
        confusion_rows
    ).sort_values(
        ["attack_type", "seed", "monitoring_round"]
    )

    if len(seed_table) != 20:
        raise RuntimeError(
            f"Expected 20 seed-attack rows, got {len(seed_table)}"
        )
    if len(round_table) != 80:
        raise RuntimeError(
            f"Expected 80 round rows, got {len(round_table)}"
        )

    seed_table.to_csv(
        tables / "v320a3_seed_attack_qualification.csv",
        index=False,
    )
    round_table.to_csv(
        tables / "v320a3_round_utility_losses.csv",
        index=False,
    )
    class_table.to_csv(
        tables / "v320a3_class_recall_losses.csv",
        index=False,
    )
    confusion_table.to_csv(
        tables / "v320a3_confusion_shifts.csv",
        index=False,
    )

    confirm = seed_table[
        seed_table["seed"].isin(CONFIRMATORY_SEEDS)
    ]
    qualification_rows = []
    for attack_type, group in confirm.groupby(
        "attack_type",
        sort=False,
    ):
        mean_macro = float(group["macro_f1_loss"].mean())
        positive_macro = int(
            group["positive_macro_f1_loss"].sum()
        )
        mean_bacc = float(
            group["balanced_accuracy_loss"].mean()
        )
        positive_bacc = int(
            group["positive_balanced_accuracy_loss"].sum()
        )
        mean_worst = float(
            group["worst_class_recall_loss"].mean()
        )
        positive_worst = int(
            group[
                "positive_worst_class_recall_loss"
            ].sum()
        )

        global_pass = bool(
            mean_macro >= MIN_MEAN_MACRO_F1_LOSS
            and positive_macro
            >= MIN_POSITIVE_CONFIRMATORY_SEEDS
        )
        bacc_pass = bool(
            mean_bacc
            >= MIN_MEAN_BALANCED_ACCURACY_LOSS
            and positive_bacc
            >= MIN_POSITIVE_CONFIRMATORY_SEEDS
        )
        class_pass = bool(
            mean_worst
            >= MIN_MEAN_WORST_CLASS_RECALL_LOSS
            and positive_worst
            >= MIN_POSITIVE_CONFIRMATORY_SEEDS
        )

        per_class = class_table[
            class_table["attack_type"].eq(attack_type)
            & class_table["seed"].isin(CONFIRMATORY_SEEDS)
        ]
        consistent = (
            per_class.groupby(
                ["class_id", "class_name"],
                as_index=False,
            )
            .agg(
                mean_confirmatory_recall_loss=(
                    "recall_loss",
                    "mean",
                ),
                positive_seed_count=(
                    "recall_loss",
                    lambda values:
                        int((values > 0).sum()),
                ),
            )
            .sort_values(
                [
                    "mean_confirmatory_recall_loss",
                    "class_id",
                ],
                ascending=[False, True],
            )
            .iloc[0]
        )

        row = {
            "attack_type": attack_type,
            "mean_confirmatory_macro_f1_loss":
                mean_macro,
            "minimum_confirmatory_macro_f1_loss":
                float(group["macro_f1_loss"].min()),
            "positive_macro_f1_loss_seed_count":
                positive_macro,
            "mean_confirmatory_balanced_accuracy_loss":
                mean_bacc,
            "minimum_confirmatory_balanced_accuracy_loss":
                float(
                    group[
                        "balanced_accuracy_loss"
                    ].min()
                ),
            "positive_balanced_accuracy_loss_seed_count":
                positive_bacc,
            "mean_confirmatory_worst_class_recall_loss":
                mean_worst,
            "minimum_confirmatory_worst_class_recall_loss":
                float(
                    group[
                        "worst_class_recall_loss"
                    ].min()
                ),
            "positive_worst_class_recall_loss_seed_count":
                positive_worst,
            "most_consistently_harmed_class":
                str(consistent["class_name"]),
            "most_consistently_harmed_class_mean_recall_loss":
                float(
                    consistent[
                        "mean_confirmatory_recall_loss"
                    ]
                ),
            "most_consistently_harmed_class_positive_seed_count":
                int(consistent["positive_seed_count"]),
            "mean_confirmatory_confusion_shift":
                float(
                    group[
                        "mean_confusion_total_variation_shift"
                    ].mean()
                ),
            "passes_global_macro_f1_qualification":
                global_pass,
            "passes_balanced_accuracy_qualification":
                bacc_pass,
            "passes_class_recall_qualification":
                class_pass,
            "qualified_for_v320b": bool(
                global_pass or bacc_pass or class_pass
            ),
            "exploratory_seed7_macro_f1_loss":
                float(
                    seed_table.loc[
                        seed_table["attack_type"].eq(
                            attack_type
                        )
                        & seed_table["seed"].eq(7),
                        "macro_f1_loss",
                    ].iloc[0]
                ),
            "attack_definitions_changed_after_seed7":
                False,
            "attack_specific_retuning": False,
            "test_sets_accessed": False,
        }
        qualification_rows.append(row)

    qualification = pd.DataFrame(
        qualification_rows
    ).sort_values("attack_type")
    qualification["qualification_label"] = (
        qualification.apply(label, axis=1)
    )
    qualification.to_csv(
        tables / "v320a3_attack_qualification_summary.csv",
        index=False,
    )

    qualified = qualification.loc[
        qualification["qualified_for_v320b"],
        "attack_type",
    ].tolist()
    decision = {
        "experiment_version": "3.20A.3",
        "stage":
            "exact_plain_cross_seed_untargeted_qualification",
        "exploratory_seed": 7,
        "confirmatory_seeds": "99|123|2026",
        "attack_count": 5,
        "exact_clean_equivalence_pass_count": int(
            verification[
                "passes_exact_clean_training_equivalence_1e8"
            ].sum()
        ),
        "mechanics_valid_seed_attack_count": int(
            seed_table[
                "attack_mechanics_valid"
            ].sum()
        ),
        "total_seed_attack_count": int(len(seed_table)),
        "qualified_attack_count": int(len(qualified)),
        "qualified_attacks": "|".join(qualified),
        "defense_branches_run": False,
        "prior_v320a_utility_results_status":
            "SUPERSEDED_BY_EXACT_PLAIN_BASELINE",
        "prior_v320a_mechanics_status":
            "RETAINED_AS_VALID",
        "method_reopened": False,
        "attack_specific_retuning": False,
        "natural_test_accessed": False,
        "diagnostic_test_accessed": False,
        "proceed_to_v320b": bool(qualified),
        "next_stage": (
            "V3.20B frozen defended multiseed on qualified attacks"
            if qualified
            else
            "Close Task 40 qualification as no consistent harmful attacks"
        ),
    }
    pd.DataFrame([decision]).to_csv(
        tables / "v320a3_qualification_decision.csv",
        index=False,
    )
    with (output / "v320a3_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(decision, handle, indent=2)

    x = np.arange(len(qualification))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(
        x,
        qualification[
            "mean_confirmatory_macro_f1_loss"
        ],
    )
    ax.axhline(
        MIN_MEAN_MACRO_F1_LOSS,
        linestyle="--",
        linewidth=1,
        label="Qualification threshold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        qualification["attack_type"],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Mean confirmatory macro-F1 loss")
    ax.set_title(
        "V3.20A.3 exact untargeted attack qualification"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(
        fig,
        figures / "v320a3_confirmatory_macro_f1_losses",
    )

    print("V3.20A.3 exact qualification complete")
    print()
    print("ATTACK QUALIFICATION")
    columns = [
        "attack_type",
        "mean_confirmatory_macro_f1_loss",
        "positive_macro_f1_loss_seed_count",
        "mean_confirmatory_balanced_accuracy_loss",
        "positive_balanced_accuracy_loss_seed_count",
        "mean_confirmatory_worst_class_recall_loss",
        "most_consistently_harmed_class",
        "passes_global_macro_f1_qualification",
        "passes_balanced_accuracy_qualification",
        "passes_class_recall_qualification",
        "qualified_for_v320b",
        "qualification_label",
    ]
    print(qualification[columns].to_string(index=False))
    print()
    print("QUALIFICATION DECISION")
    print(pd.DataFrame([decision]).to_string(index=False))
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

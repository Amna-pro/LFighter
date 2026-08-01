#!/usr/bin/env python3
"""Summarize the V3.20A.2 plain-only cross-seed qualification audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

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
    p.add_argument("--seed7-smoke-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seeds", default="7,99,123,2026")
    p.add_argument("--confirmatory-seeds", default="99,123,2026")
    return p.parse_args()


def semantic_poison_plan_hash(branch_dir: Path) -> str:
    digest = hashlib.sha256()
    attack_dir = branch_dir / "attack_manifest"
    for filename in ("poisoned_indices.npz", "poisoned_labels.npz"):
        path = attack_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        digest.update(filename.encode("utf-8"))
        with np.load(path, allow_pickle=False) as archive:
            for key in sorted(archive.files):
                array = np.ascontiguousarray(archive[key])
                digest.update(key.encode("utf-8"))
                digest.update(str(array.dtype).encode("utf-8"))
                digest.update(
                    np.asarray(array.shape, dtype=np.int64).tobytes()
                )
                digest.update(array.tobytes())
    return digest.hexdigest()


def load_metadata(branch_dir: Path) -> Dict[str, object]:
    path = branch_dir / "untargeted_label_poisoning_v320a_metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_rounds(branch_dir: Path) -> pd.DataFrame:
    path = branch_dir / "tables" / "reconstruction_round_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path).sort_values(
        "monitoring_round"
    ).reset_index(drop=True)
    if len(table) != 4:
        raise RuntimeError(f"Expected four rounds in {path}, found {len(table)}")
    return table


def load_class_metrics(branch_dir: Path) -> pd.DataFrame:
    path = branch_dir / "tables" / "validation_class_metrics_long.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    expected = 4 * 8
    if len(table) != expected:
        raise RuntimeError(
            f"Expected {expected} class-metric rows in {path}, "
            f"found {len(table)}"
        )
    return table


def load_confusion(branch_dir: Path) -> pd.DataFrame:
    path = branch_dir / "tables" / "validation_confusion_matrix_long.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    expected = 4 * 8 * 8
    if len(table) != expected:
        raise RuntimeError(
            f"Expected {expected} confusion rows in {path}, found {len(table)}"
        )
    return table


def normalize_confusion(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    totals = result.groupby(
        ["monitoring_round", "true_id"]
    )["count"].transform("sum")
    result["row_normalized_rate"] = np.where(
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


def attack_branch(
    input_root: Path,
    seed7_root: Path,
    attack_type: str,
    seed: int,
) -> Path:
    if seed == 7:
        return seed7_root / "runs" / attack_type / "plain_fedavg"
    return (
        input_root / "runs" / attack_type
        / f"seed_{seed}" / "plain_fedavg"
    )


def qualification_label(row: pd.Series) -> str:
    global_pass = bool(row["passes_global_macro_f1_qualification"])
    balanced_pass = bool(
        row["passes_balanced_accuracy_qualification"]
    )
    class_pass = bool(row["passes_class_recall_qualification"])
    if global_pass and (balanced_pass or class_pass):
        return "GLOBAL_AND_CLASS_QUALIFIED"
    if global_pass:
        return "GLOBAL_QUALIFIED"
    if balanced_pass or class_pass:
        return "CLASS_SPECIFIC_QUALIFIED"
    return "NOT_QUALIFIED"


def main() -> int:
    a = parse_args()
    seeds = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]
    confirmatory = [
        int(x.strip())
        for x in a.confirmatory_seeds.split(",")
        if x.strip()
    ]
    if seeds != ALL_SEEDS:
        raise ValueError(
            f"V3.20A.2 is frozen to all seeds {ALL_SEEDS}, got {seeds}"
        )
    if confirmatory != CONFIRMATORY_SEEDS:
        raise ValueError(
            "V3.20A.2 confirmatory seeds are frozen to "
            f"{CONFIRMATORY_SEEDS}, got {confirmatory}"
        )

    input_root = a.input_root.expanduser().resolve()
    seed7_root = a.seed7_smoke_root.expanduser().resolve()
    output = a.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    verification_rows = []
    for seed in seeds:
        verification_path = (
            input_root / "clean_verification" / f"seed_{seed}"
            / "tables" / "v320a2_clean_equivalence_verification.csv"
        )
        if not verification_path.exists():
            raise FileNotFoundError(verification_path)
        verification_rows.append(pd.read_csv(verification_path))
    verification = pd.concat(verification_rows, ignore_index=True)
    if len(verification) != 4:
        raise RuntimeError("Expected four clean-equivalence rows")
    if not verification[
        "passes_exact_clean_training_equivalence_1e8"
    ].astype(bool).all():
        raise RuntimeError("At least one clean audit failed exact equivalence")
    if verification["test_sets_accessed"].astype(bool).any():
        raise RuntimeError("A clean verification reports test-set access")
    verification.to_csv(
        tables / "v320a2_clean_equivalence_all_seeds.csv",
        index=False,
    )

    seed_attack_rows: List[Dict[str, object]] = []
    round_rows: List[pd.DataFrame] = []
    class_loss_rows: List[pd.DataFrame] = []
    confusion_shift_rows: List[Dict[str, object]] = []

    for seed in seeds:
        clean_dir = input_root / "clean_audit" / f"seed_{seed}"
        clean_rounds = load_rounds(clean_dir)
        clean_classes = load_class_metrics(clean_dir)
        clean_confusion = normalize_confusion(load_confusion(clean_dir))

        for attack_type in ATTACK_TYPES:
            plain_dir = attack_branch(
                input_root, seed7_root, attack_type, seed
            )
            plain_rounds = load_rounds(plain_dir)
            plain_classes = load_class_metrics(plain_dir)
            plain_confusion = normalize_confusion(
                load_confusion(plain_dir)
            )
            metadata = load_metadata(plain_dir)

            if metadata.get("attack_type") != attack_type:
                raise RuntimeError(
                    f"Attack metadata mismatch for {attack_type}, seed {seed}"
                )
            if int(metadata.get("model_seed")) != seed:
                raise RuntimeError(
                    f"Seed metadata mismatch for {attack_type}, seed {seed}"
                )
            if metadata.get("mode") != "strong_attack":
                raise RuntimeError(
                    f"Branch is not strong_attack: {plain_dir}"
                )
            if metadata.get("replacement_policy") != "plain_fedavg":
                raise RuntimeError(
                    f"Branch is not plain_fedavg: {plain_dir}"
                )
            if bool(metadata.get("test_sets_accessed")):
                raise RuntimeError(f"Branch reports test access: {plain_dir}")
            if list(metadata.get("malicious_clients", [])) != [
                1, 7, 8, 10, 14, 15, 17, 18
            ]:
                raise RuntimeError(
                    f"Unexpected coalition in {plain_dir}"
                )
            if abs(float(metadata.get("poison_fraction")) - 1.0) > 1e-12:
                raise RuntimeError(
                    f"Unexpected poison fraction in {plain_dir}"
                )

            manifest = pd.read_csv(
                plain_dir / "attack_manifest"
                / "malicious_client_poison_manifest.csv"
            )
            malicious_manifest = manifest[
                manifest["is_malicious"].astype(bool)
            ].copy()
            changed_rows = int(
                malicious_manifest["changed_rows_verified"].sum()
            )
            mechanics_valid = bool(
                len(malicious_manifest) == 8
                and changed_rows > 0
                and (
                    malicious_manifest["poisoned_rows"].astype(int)
                    == malicious_manifest[
                        "changed_rows_verified"
                    ].astype(int)
                ).all()
            )
            if not mechanics_valid:
                raise RuntimeError(
                    f"Attack mechanics invalid for {attack_type}, seed {seed}"
                )

            current_rounds = pd.DataFrame({
                "seed": seed,
                "seed_role": (
                    "exploratory_smoke"
                    if seed == 7
                    else "confirmatory_development"
                ),
                "attack_type": attack_type,
                "monitoring_round":
                    clean_rounds["monitoring_round"].astype(int),
                "clean_macro_f1": clean_rounds["val_macro_f1"],
                "plain_attack_macro_f1":
                    plain_rounds["val_macro_f1"],
                "macro_f1_loss":
                    clean_rounds["val_macro_f1"]
                    - plain_rounds["val_macro_f1"],
                "clean_balanced_accuracy":
                    clean_rounds["val_balanced_accuracy"],
                "plain_attack_balanced_accuracy":
                    plain_rounds["val_balanced_accuracy"],
                "balanced_accuracy_loss":
                    clean_rounds["val_balanced_accuracy"]
                    - plain_rounds["val_balanced_accuracy"],
                "clean_accuracy": clean_rounds["val_accuracy"],
                "plain_attack_accuracy": plain_rounds["val_accuracy"],
                "accuracy_loss":
                    clean_rounds["val_accuracy"]
                    - plain_rounds["val_accuracy"],
                "clean_weighted_f1":
                    clean_rounds["val_weighted_f1"],
                "plain_attack_weighted_f1":
                    plain_rounds["val_weighted_f1"],
                "weighted_f1_loss":
                    clean_rounds["val_weighted_f1"]
                    - plain_rounds["val_weighted_f1"],
                "clean_mcc": clean_rounds["val_mcc"],
                "plain_attack_mcc": plain_rounds["val_mcc"],
                "mcc_loss":
                    clean_rounds["val_mcc"]
                    - plain_rounds["val_mcc"],
            })
            round_rows.append(current_rounds)

            clean_class_mean = (
                clean_classes.groupby(
                    ["class_id", "class_name"], as_index=False
                )["recall"].mean()
                .rename(columns={"recall": "clean_mean_recall"})
            )
            plain_class_mean = (
                plain_classes.groupby(
                    ["class_id", "class_name"], as_index=False
                )["recall"].mean()
                .rename(columns={"recall": "plain_attack_mean_recall"})
            )
            class_loss = clean_class_mean.merge(
                plain_class_mean,
                on=["class_id", "class_name"],
                validate="one_to_one",
            )
            class_loss["recall_loss"] = (
                class_loss["clean_mean_recall"]
                - class_loss["plain_attack_mean_recall"]
            )
            class_loss.insert(0, "attack_type", attack_type)
            class_loss.insert(0, "seed", seed)
            class_loss.insert(
                1,
                "seed_role",
                "exploratory_smoke"
                if seed == 7
                else "confirmatory_development",
            )
            class_loss_rows.append(class_loss)

            worst = class_loss.sort_values(
                ["recall_loss", "class_id"],
                ascending=[False, True],
            ).iloc[0]

            confusion = clean_confusion.merge(
                plain_confusion,
                on=[
                    "monitoring_round",
                    "true_id",
                    "true_class",
                    "predicted_id",
                    "predicted_class",
                ],
                suffixes=("_clean", "_plain"),
                validate="one_to_one",
            )
            confusion["absolute_rate_shift"] = np.abs(
                confusion["row_normalized_rate_clean"]
                - confusion["row_normalized_rate_plain"]
            )
            per_round_shift = (
                confusion.groupby("monitoring_round")[
                    "absolute_rate_shift"
                ].sum()
                / 2.0
            )
            for monitoring_round, shift in per_round_shift.items():
                confusion_shift_rows.append({
                    "seed": seed,
                    "attack_type": attack_type,
                    "monitoring_round": int(monitoring_round),
                    "confusion_total_variation_shift": float(shift),
                })

            seed_attack_rows.append({
                "seed": seed,
                "seed_role": (
                    "exploratory_smoke"
                    if seed == 7
                    else "confirmatory_development"
                ),
                "attack_type": attack_type,
                "changed_malicious_rows": changed_rows,
                "poison_plan_semantic_hash":
                    semantic_poison_plan_hash(plain_dir),
                "attack_mechanics_valid": mechanics_valid,
                "clean_mean_macro_f1": float(
                    clean_rounds["val_macro_f1"].mean()
                ),
                "plain_attack_mean_macro_f1": float(
                    plain_rounds["val_macro_f1"].mean()
                ),
                "macro_f1_loss": float(
                    current_rounds["macro_f1_loss"].mean()
                ),
                "positive_macro_f1_loss": bool(
                    current_rounds["macro_f1_loss"].mean() > 0
                ),
                "clean_mean_balanced_accuracy": float(
                    clean_rounds["val_balanced_accuracy"].mean()
                ),
                "plain_attack_mean_balanced_accuracy": float(
                    plain_rounds["val_balanced_accuracy"].mean()
                ),
                "balanced_accuracy_loss": float(
                    current_rounds["balanced_accuracy_loss"].mean()
                ),
                "positive_balanced_accuracy_loss": bool(
                    current_rounds[
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
                "classes_with_recall_loss_010": int(
                    (class_loss["recall_loss"] >= 0.10).sum()
                ),
                "mean_confusion_total_variation_shift": float(
                    per_round_shift.mean()
                ),
                "test_sets_accessed": False,
            })

    seed_attack = pd.DataFrame(seed_attack_rows).sort_values(
        ["attack_type", "seed"]
    )
    round_table = pd.concat(round_rows, ignore_index=True).sort_values(
        ["attack_type", "seed", "monitoring_round"]
    )
    class_loss_table = pd.concat(
        class_loss_rows, ignore_index=True
    ).sort_values(["attack_type", "seed", "class_id"])
    confusion_shift_table = pd.DataFrame(
        confusion_shift_rows
    ).sort_values(["attack_type", "seed", "monitoring_round"])

    if len(seed_attack) != 20:
        raise RuntimeError(
            f"Expected 20 seed-attack rows, found {len(seed_attack)}"
        )
    if len(round_table) != 80:
        raise RuntimeError(
            f"Expected 80 round rows, found {len(round_table)}"
        )
    if len(class_loss_table) != 160:
        raise RuntimeError(
            f"Expected 160 class-loss rows, found {len(class_loss_table)}"
        )

    seed_attack.to_csv(
        tables / "v320a2_seed_attack_qualification.csv",
        index=False,
    )
    round_table.to_csv(
        tables / "v320a2_round_utility_losses.csv",
        index=False,
    )
    class_loss_table.to_csv(
        tables / "v320a2_class_recall_losses.csv",
        index=False,
    )
    confusion_shift_table.to_csv(
        tables / "v320a2_confusion_shifts.csv",
        index=False,
    )

    confirm = seed_attack[
        seed_attack["seed"].isin(CONFIRMATORY_SEEDS)
    ].copy()
    qualification_rows: List[Dict[str, object]] = []
    for attack_type, group in confirm.groupby(
        "attack_type", sort=False
    ):
        if len(group) != 3:
            raise RuntimeError(
                f"Expected three confirmatory seeds for {attack_type}"
            )

        mean_macro_loss = float(group["macro_f1_loss"].mean())
        positive_macro_count = int(
            group["positive_macro_f1_loss"].sum()
        )
        mean_bacc_loss = float(
            group["balanced_accuracy_loss"].mean()
        )
        positive_bacc_count = int(
            group["positive_balanced_accuracy_loss"].sum()
        )
        mean_worst_recall_loss = float(
            group["worst_class_recall_loss"].mean()
        )
        positive_worst_count = int(
            group["positive_worst_class_recall_loss"].sum()
        )

        global_pass = bool(
            mean_macro_loss >= MIN_MEAN_MACRO_F1_LOSS
            and positive_macro_count >= MIN_POSITIVE_CONFIRMATORY_SEEDS
        )
        balanced_pass = bool(
            mean_bacc_loss >= MIN_MEAN_BALANCED_ACCURACY_LOSS
            and positive_bacc_count >= MIN_POSITIVE_CONFIRMATORY_SEEDS
        )
        class_pass = bool(
            mean_worst_recall_loss
            >= MIN_MEAN_WORST_CLASS_RECALL_LOSS
            and positive_worst_count >= MIN_POSITIVE_CONFIRMATORY_SEEDS
        )

        all_four = seed_attack[
            seed_attack["attack_type"].eq(attack_type)
        ]
        per_class_confirm = class_loss_table[
            class_loss_table["attack_type"].eq(attack_type)
            & class_loss_table["seed"].isin(CONFIRMATORY_SEEDS)
        ]
        consistent_class = (
            per_class_confirm.groupby(
                ["class_id", "class_name"], as_index=False
            )
            .agg(
                mean_confirmatory_recall_loss=(
                    "recall_loss", "mean"
                ),
                positive_seed_count=(
                    "recall_loss",
                    lambda values: int((values > 0).sum()),
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
            "confirmatory_seed_count": 3,
            "mean_confirmatory_macro_f1_loss": mean_macro_loss,
            "minimum_confirmatory_macro_f1_loss": float(
                group["macro_f1_loss"].min()
            ),
            "positive_macro_f1_loss_seed_count":
                positive_macro_count,
            "mean_confirmatory_balanced_accuracy_loss":
                mean_bacc_loss,
            "minimum_confirmatory_balanced_accuracy_loss": float(
                group["balanced_accuracy_loss"].min()
            ),
            "positive_balanced_accuracy_loss_seed_count":
                positive_bacc_count,
            "mean_confirmatory_worst_class_recall_loss":
                mean_worst_recall_loss,
            "minimum_confirmatory_worst_class_recall_loss": float(
                group["worst_class_recall_loss"].min()
            ),
            "positive_worst_class_recall_loss_seed_count":
                positive_worst_count,
            "most_consistently_harmed_class": str(
                consistent_class["class_name"]
            ),
            "most_consistently_harmed_class_mean_recall_loss":
                float(
                    consistent_class[
                        "mean_confirmatory_recall_loss"
                    ]
                ),
            "most_consistently_harmed_class_positive_seed_count":
                int(consistent_class["positive_seed_count"]),
            "mean_confirmatory_confusion_shift": float(
                group[
                    "mean_confusion_total_variation_shift"
                ].mean()
            ),
            "passes_global_macro_f1_qualification": global_pass,
            "passes_balanced_accuracy_qualification":
                balanced_pass,
            "passes_class_recall_qualification": class_pass,
            "qualified_for_v320b": bool(
                global_pass or balanced_pass or class_pass
            ),
            "all_four_seed_mean_macro_f1_loss": float(
                all_four["macro_f1_loss"].mean()
            ),
            "exploratory_seed7_macro_f1_loss": float(
                all_four.loc[
                    all_four["seed"].eq(7),
                    "macro_f1_loss",
                ].iloc[0]
            ),
            "attack_definitions_changed_after_seed7": False,
            "attack_specific_retuning": False,
            "test_sets_accessed": False,
        }
        qualification_rows.append(row)

    qualification = pd.DataFrame(qualification_rows)
    qualification["qualification_label"] = qualification.apply(
        qualification_label, axis=1
    )
    qualification = qualification.sort_values("attack_type")
    qualification.to_csv(
        tables / "v320a2_attack_qualification_summary.csv",
        index=False,
    )

    qualified_attacks = qualification.loc[
        qualification["qualified_for_v320b"],
        "attack_type",
    ].tolist()
    decision = {
        "experiment_version": "3.20A.2",
        "stage":
            "plain_only_cross_seed_untargeted_attack_qualification",
        "exploratory_seed": 7,
        "confirmatory_seeds": "99|123|2026",
        "attack_count": 5,
        "clean_equivalence_pass_count": int(
            verification[
                "passes_exact_clean_training_equivalence_1e8"
            ].sum()
        ),
        "mechanics_valid_seed_attack_count": int(
            seed_attack["attack_mechanics_valid"].sum()
        ),
        "total_seed_attack_count": int(len(seed_attack)),
        "qualified_attack_count": int(len(qualified_attacks)),
        "qualified_attacks": "|".join(qualified_attacks),
        "global_qualified_attack_count": int(
            qualification[
                "passes_global_macro_f1_qualification"
            ].sum()
        ),
        "class_specific_qualified_attack_count": int(
            (
                qualification[
                    "passes_balanced_accuracy_qualification"
                ]
                | qualification[
                    "passes_class_recall_qualification"
                ]
            ).sum()
        ),
        "qualification_threshold_mean_macro_f1_loss":
            MIN_MEAN_MACRO_F1_LOSS,
        "qualification_threshold_mean_balanced_accuracy_loss":
            MIN_MEAN_BALANCED_ACCURACY_LOSS,
        "qualification_threshold_mean_worst_class_recall_loss":
            MIN_MEAN_WORST_CLASS_RECALL_LOSS,
        "minimum_positive_confirmatory_seeds":
            MIN_POSITIVE_CONFIRMATORY_SEEDS,
        "defense_branches_run_in_v320a2": False,
        "method_reopened": False,
        "attack_specific_retuning": False,
        "natural_test_accessed": False,
        "diagnostic_test_accessed": False,
        "proceed_to_v320b": bool(len(qualified_attacks) > 0),
        "next_stage": (
            "V3.20B frozen defended multiseed on qualified attacks only"
            if qualified_attacks
            else "Close Task 40 as attacks non-effective under protocol"
        ),
    }
    pd.DataFrame([decision]).to_csv(
        tables / "v320a2_qualification_decision.csv",
        index=False,
    )
    with (output / "v320a2_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(decision, handle, indent=2)

    order = qualification["attack_type"].tolist()
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(
        x,
        qualification["mean_confirmatory_macro_f1_loss"],
    )
    ax.axhline(
        MIN_MEAN_MACRO_F1_LOSS,
        linestyle="--",
        linewidth=1,
        label="Global qualification threshold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("Mean confirmatory macro-F1 loss")
    ax.set_title("V3.20A.2 untargeted attack qualification")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(
        fig, figures / "v320a2_confirmatory_macro_f1_losses"
    )

    fig, ax = plt.subplots(figsize=(12, 6.5))
    width = 0.38
    ax.bar(
        x - width / 2,
        qualification[
            "mean_confirmatory_balanced_accuracy_loss"
        ],
        width,
        label="Balanced-accuracy loss",
    )
    ax.bar(
        x + width / 2,
        qualification[
            "mean_confirmatory_worst_class_recall_loss"
        ],
        width,
        label="Worst-class recall loss",
    )
    ax.axhline(
        MIN_MEAN_BALANCED_ACCURACY_LOSS,
        linestyle="--",
        linewidth=1,
    )
    ax.axhline(
        MIN_MEAN_WORST_CLASS_RECALL_LOSS,
        linestyle=":",
        linewidth=1,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("Mean confirmatory loss")
    ax.set_title("Class-sensitive untargeted attack qualification")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(
        fig, figures / "v320a2_class_sensitive_losses"
    )

    print("V3.20A.2 untargeted qualification audit complete")
    print()
    print("ATTACK QUALIFICATION")
    display_columns = [
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
    print(qualification[display_columns].to_string(index=False))
    print()
    print("QUALIFICATION DECISION")
    print(pd.DataFrame([decision]).to_string(index=False))
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

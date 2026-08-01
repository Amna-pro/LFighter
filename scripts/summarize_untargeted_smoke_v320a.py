#!/usr/bin/env python3
"""Summarize the five frozen V3.20A untargeted poisoning smoke attacks."""
from __future__ import annotations

import argparse
import hashlib
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
MIN_MACRO_F1_LOSS = 0.02


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True, type=Path)
    p.add_argument("--clean-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def load_rounds(path: Path, filename: str) -> pd.DataFrame:
    full = path / "tables" / filename
    if not full.exists():
        raise FileNotFoundError(full)
    table = pd.read_csv(full).sort_values(
        "monitoring_round"
    ).reset_index(drop=True)
    if len(table) != 4:
        raise RuntimeError(f"Expected four rounds in {full}")
    return table


def load_metadata(path: Path) -> Dict[str, object]:
    full = path / "untargeted_label_poisoning_v320a_metadata.json"
    if not full.exists():
        raise FileNotFoundError(full)
    with full.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def semantic_poison_plan_hash(branch_dir: Path) -> str:
    """Hash poisoned positions and replacement labels by array contents.

    NPZ file bytes are not hashed directly because ZIP metadata can differ
    despite identical arrays. The semantic hash is stable across branches.
    """
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


def save_figure(fig, base: Path):
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    a = parse_args()
    if int(a.seed) != 7:
        raise ValueError("V3.20A smoke is frozen to seed 7")

    input_root = a.input_root.expanduser().resolve()
    clean_dir = a.clean_dir.expanduser().resolve()
    output = a.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    clean = load_rounds(clean_dir, "continuation_round_metrics.csv")
    round_rows: List[pd.DataFrame] = []
    attack_rows: List[Dict[str, object]] = []

    for attack_type in ATTACK_TYPES:
        attack_root = input_root / "runs" / attack_type
        plain_dir = attack_root / "plain_fedavg"
        defended_dir = attack_root / "trusted_reconstruction"

        plain = load_rounds(
            plain_dir, "reconstruction_round_metrics.csv"
        )
        defended = load_rounds(
            defended_dir, "reconstruction_round_metrics.csv"
        )
        plain_meta = load_metadata(plain_dir)
        defended_meta = load_metadata(defended_dir)

        if plain_meta["attack_type"] != attack_type:
            raise RuntimeError(f"Plain metadata attack mismatch: {attack_type}")
        if defended_meta["attack_type"] != attack_type:
            raise RuntimeError(
                f"Defended metadata attack mismatch: {attack_type}"
            )
        plain_plan_hash = semantic_poison_plan_hash(plain_dir)
        defended_plan_hash = semantic_poison_plan_hash(defended_dir)
        if plain_plan_hash != defended_plan_hash:
            raise RuntimeError(
                f"Plain/defended poison plan mismatch for {attack_type}"
            )
        if plain_meta["malicious_clients"] != defended_meta["malicious_clients"]:
            raise RuntimeError(
                f"Plain/defended coalition mismatch for {attack_type}"
            )
        if bool(plain_meta["test_sets_accessed"]) or bool(
            defended_meta["test_sets_accessed"]
        ):
            raise RuntimeError("A V3.20A branch reports test-set access")

        plain_manifest = pd.read_csv(
            plain_dir / "attack_manifest"
            / "malicious_client_poison_manifest.csv"
        )
        defended_manifest = pd.read_csv(
            defended_dir / "attack_manifest"
            / "malicious_client_poison_manifest.csv"
        )
        manifest_columns = [
            "client_id",
            "is_malicious",
            "eligible_rows",
            "poisoned_rows",
            "changed_rows_verified",
            "attack_type",
        ]
        if not plain_manifest[manifest_columns].equals(
            defended_manifest[manifest_columns]
        ):
            raise RuntimeError(
                f"Plain/defended poison manifests differ for {attack_type}"
            )
        malicious_manifest = plain_manifest[
            plain_manifest["is_malicious"].astype(bool)
        ]
        changed_rows = int(
            malicious_manifest["changed_rows_verified"].sum()
        )
        mechanics_valid = bool(
            len(malicious_manifest) == 8
            and changed_rows > 0
            and (
                malicious_manifest["poisoned_rows"].astype(int)
                == malicious_manifest["changed_rows_verified"].astype(int)
            ).all()
        )

        current = pd.DataFrame({
            "seed": int(a.seed),
            "attack_type": attack_type,
            "monitoring_round": clean["monitoring_round"].astype(int),
            "clean_macro_f1": clean["val_macro_f1"],
            "plain_attack_macro_f1": plain["val_macro_f1"],
            "defended_attack_macro_f1": defended["val_macro_f1"],
            "clean_balanced_accuracy": clean["val_balanced_accuracy"],
            "plain_attack_balanced_accuracy":
                plain["val_balanced_accuracy"],
            "defended_attack_balanced_accuracy":
                defended["val_balanced_accuracy"],
            "malicious_recall": defended["malicious_recall"],
            "benign_fpr": defended["benign_false_positive_rate"],
            "flagged_clients": defended["flagged_clients"],
            "replaced_clients": defended["replaced_clients"],
        })
        current["macro_f1_attack_loss"] = (
            current["clean_macro_f1"]
            - current["plain_attack_macro_f1"]
        )
        current["macro_f1_recovery"] = (
            current["defended_attack_macro_f1"]
            - current["plain_attack_macro_f1"]
        )
        current["defense_improved_macro_f1"] = (
            current["defended_attack_macro_f1"]
            > current["plain_attack_macro_f1"]
        )
        round_rows.append(current)

        clean_macro = float(current["clean_macro_f1"].mean())
        plain_macro = float(current["plain_attack_macro_f1"].mean())
        defended_macro = float(
            current["defended_attack_macro_f1"].mean()
        )
        attack_loss = clean_macro - plain_macro
        recovery = defended_macro - plain_macro
        recovery_fraction = (
            float(recovery / attack_loss)
            if attack_loss > 1e-12 else float("nan")
        )
        clean_bacc = float(
            current["clean_balanced_accuracy"].mean()
        )
        plain_bacc = float(
            current["plain_attack_balanced_accuracy"].mean()
        )
        defended_bacc = float(
            current["defended_attack_balanced_accuracy"].mean()
        )

        defended_class = pd.read_csv(
            defended_dir / "tables"
            / "validation_class_metrics_long.csv"
        )
        mean_class_recall = (
            defended_class.groupby("class_name")["recall"]
            .mean()
            .reset_index()
        )
        worst_row = mean_class_recall.sort_values(
            ["recall", "class_name"]
        ).iloc[0]

        attack_rows.append({
            "seed": int(a.seed),
            "attack_type": attack_type,
            "malicious_clients": "|".join(
                map(str, plain_meta["malicious_clients"])
            ),
            "poison_fraction": float(plain_meta["poison_fraction"]),
            "poison_plan_semantic_hash": plain_plan_hash,
            "changed_malicious_rows": changed_rows,
            "attack_mechanics_valid": mechanics_valid,
            "clean_mean_macro_f1": clean_macro,
            "plain_attack_mean_macro_f1": plain_macro,
            "defended_attack_mean_macro_f1": defended_macro,
            "plain_attack_macro_f1_loss": attack_loss,
            "defended_macro_f1_recovery": recovery,
            "utility_recovery_fraction": recovery_fraction,
            "attack_qualified_macro_f1_loss_002": bool(
                attack_loss >= MIN_MACRO_F1_LOSS
            ),
            "defense_positive_macro_f1_recovery": bool(
                recovery > 0
            ),
            "rounds_improved_macro_f1": int(
                current["defense_improved_macro_f1"].sum()
            ),
            "clean_mean_balanced_accuracy": clean_bacc,
            "plain_attack_mean_balanced_accuracy": plain_bacc,
            "defended_attack_mean_balanced_accuracy": defended_bacc,
            "balanced_accuracy_recovery": (
                defended_bacc - plain_bacc
            ),
            "mean_malicious_recall": float(
                current["malicious_recall"].mean()
            ),
            "minimum_malicious_recall": float(
                current["malicious_recall"].min()
            ),
            "mean_benign_fpr": float(
                current["benign_fpr"].mean()
            ),
            "maximum_benign_fpr": float(
                current["benign_fpr"].max()
            ),
            "mean_replaced_clients": float(
                current["replaced_clients"].mean()
            ),
            "worst_defended_class": str(worst_row["class_name"]),
            "worst_defended_class_mean_recall": float(
                worst_row["recall"]
            ),
            "same_plain_defended_poison_plan": True,
            "frozen_detector": True,
            "frozen_reconstruction_policy":
                "center_plus_residual",
            "attack_specific_retuning": False,
            "test_sets_accessed": False,
        })

    round_table = pd.concat(round_rows, ignore_index=True)
    attack_table = pd.DataFrame(attack_rows)
    round_table.to_csv(
        tables / "v320a_untargeted_smoke_rounds.csv",
        index=False,
    )
    attack_table.to_csv(
        tables / "v320a_untargeted_smoke_attack_summary.csv",
        index=False,
    )

    mechanics_pass = bool(
        attack_table["attack_mechanics_valid"].all()
        and attack_table["same_plain_defended_poison_plan"].all()
        and not attack_table["test_sets_accessed"].any()
    )
    qualified_count = int(
        attack_table["attack_qualified_macro_f1_loss_002"].sum()
    )
    positive_count = int(
        attack_table["defense_positive_macro_f1_recovery"].sum()
    )
    proceed = bool(mechanics_pass and qualified_count >= 3)
    decision = {
        "experiment_version": "3.20A",
        "stage": "frozen_untargeted_label_poisoning_smoke_seed7",
        "attack_count": int(len(attack_table)),
        "mechanics_valid_attack_count": int(
            attack_table["attack_mechanics_valid"].sum()
        ),
        "qualified_attack_count_macro_f1_loss_002":
            qualified_count,
        "positive_recovery_attack_count": positive_count,
        "total_improved_rounds": int(
            round_table["defense_improved_macro_f1"].sum()
        ),
        "total_rounds": int(len(round_table)),
        "mean_qualified_utility_recovery_fraction": (
            float(
                attack_table.loc[
                    attack_table[
                        "attack_qualified_macro_f1_loss_002"
                    ],
                    "utility_recovery_fraction",
                ].mean()
            )
            if qualified_count else float("nan")
        ),
        "mean_malicious_recall": float(
            attack_table["mean_malicious_recall"].mean()
        ),
        "minimum_malicious_recall": float(
            attack_table["minimum_malicious_recall"].min()
        ),
        "mean_benign_fpr": float(
            attack_table["mean_benign_fpr"].mean()
        ),
        "maximum_benign_fpr": float(
            attack_table["maximum_benign_fpr"].max()
        ),
        "smoke_mechanics_pass": mechanics_pass,
        "method_reopened": False,
        "attack_specific_retuning": False,
        "natural_test_accessed": False,
        "diagnostic_test_accessed": False,
        "proceed_to_v320b_multiseed": proceed,
        "next_stage": (
            "V3.20B frozen untargeted label-poisoning multiseed"
            if proceed
            else "Audit untargeted attack qualification before multiseed"
        ),
    }
    pd.DataFrame([decision]).to_csv(
        tables / "v320a_smoke_decision.csv",
        index=False,
    )
    with (output / "v320a_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(decision, handle, indent=2)

    x = np.arange(len(attack_table))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(
        x - width,
        attack_table["clean_mean_macro_f1"],
        width,
        label="Clean FedAvg",
    )
    ax.bar(
        x,
        attack_table["plain_attack_mean_macro_f1"],
        width,
        label="Plain attacked FedAvg",
    )
    ax.bar(
        x + width,
        attack_table["defended_attack_mean_macro_f1"],
        width,
        label="Frozen reconstruction",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        attack_table["attack_type"], rotation=25, ha="right"
    )
    ax.set_ylabel("Mean validation macro-F1")
    ax.set_title("V3.20A untargeted label-poisoning smoke")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "v320a_macro_f1_comparison")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(
        attack_table["attack_type"],
        attack_table["utility_recovery_fraction"],
    )
    ax.axhline(0, linestyle=":", linewidth=1)
    ax.set_xticklabels(
        attack_table["attack_type"], rotation=25, ha="right"
    )
    ax.set_ylabel("Utility recovery fraction")
    ax.set_title("Frozen-defense utility recovery")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures / "v320a_utility_recovery")

    print("V3.20A untargeted label-poisoning smoke complete")
    print()
    print("ATTACK SUMMARY")
    columns = [
        "attack_type",
        "changed_malicious_rows",
        "clean_mean_macro_f1",
        "plain_attack_mean_macro_f1",
        "defended_attack_mean_macro_f1",
        "plain_attack_macro_f1_loss",
        "utility_recovery_fraction",
        "attack_qualified_macro_f1_loss_002",
        "defense_positive_macro_f1_recovery",
        "rounds_improved_macro_f1",
        "mean_malicious_recall",
        "minimum_malicious_recall",
        "mean_benign_fpr",
        "maximum_benign_fpr",
        "attack_mechanics_valid",
    ]
    print(attack_table[columns].to_string(index=False))
    print()
    print("SMOKE DECISION")
    print(pd.DataFrame([decision]).to_string(index=False))
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

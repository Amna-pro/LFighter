#!/usr/bin/env python3
"""Task 44 C2 selection: build the real strength curve, evaluate H1-H3.

Reads every branch's continuation_round_metrics.csv (real schema
confirmed from actual output: malicious_recall, benign_false_positive_rate,
val_macro_f1, among others -- no attack_type/poison_fraction column exists
in the CSV itself, so these are parsed from each branch's own directory
path, matching this project's established convention).

Combines the three newly-run fractions (0.25/0.50/0.75) with the existing
frozen poison_fraction=1.0 anchor (from Task 40, reused via
task44c1_strength_curve_anchor_1p0.csv) into one full 4-point curve per
attack type, then evaluates H1 (recall monotonicity), H2 (FPR relief at
weaker strength), and H3 (damage floor) exactly as preregistered.

Read-only: reprocesses already-completed branch CSVs, trains nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ATTACK_TYPES = (
    "all_to_one_benign",
    "cyclic_shift",
    "multiclass_partial_cycle",
    "pairwise_swap",
    "random_flip",
)
NEW_FRACTIONS = (0.25, 0.50, 0.75)
ANCHOR_FRACTION = 1.00
SEEDS = (7, 99, 123, 2026)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 44 C2 strength-curve selection.")
    parser.add_argument("--c2-output-root", type=Path, required=True)
    parser.add_argument(
        "--anchor-csv",
        type=Path,
        required=True,
        help="task44c1_strength_curve_anchor_1p0.csv (frozen Task 40, fraction=1.0)",
    )
    return parser.parse_args()


def fraction_tag(fraction: float) -> str:
    return f"{fraction:.2f}".replace(".", "p")


def load_branch_round_means(
    output_root: Path, fraction: float, attack_type: str, seed: int, arm: str
) -> Dict[str, float]:
    path = (
        output_root
        / f"fraction_{fraction_tag(fraction)}"
        / attack_type
        / f"seed_{seed}"
        / arm
        / "tables"
        / "continuation_round_metrics.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return {
        "mean_malicious_recall": float(df["malicious_recall"].mean()),
        "mean_benign_fpr": float(df["benign_false_positive_rate"].mean()),
        "mean_val_macro_f1": float(df["val_macro_f1"].mean()),
    }


def load_clean_seed_macro_f1(output_root: Path, seed: int) -> float:
    path = (
        output_root
        / "clean_adapter_verification"
        / f"seed_{seed}"
        / "tables"
        / "continuation_round_metrics.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return float(df["val_macro_f1"].mean())


def main() -> int:
    args = parse_args()
    output_root = args.c2_output_root.expanduser().resolve()
    summary_dir = output_root / "summary"
    tables_dir = summary_dir / "tables"
    figures_dir = summary_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    clean_macro_f1_by_seed = {seed: load_clean_seed_macro_f1(output_root, seed) for seed in SEEDS}

    per_seed_rows: List[Dict[str, Any]] = []
    for attack_type in ATTACK_TYPES:
        for fraction in NEW_FRACTIONS:
            for seed in SEEDS:
                plain = load_branch_round_means(output_root, fraction, attack_type, seed, "plain_fedavg")
                defended = load_branch_round_means(
                    output_root, fraction, attack_type, seed, "trusted_reconstruction"
                )
                clean_f1 = clean_macro_f1_by_seed[seed]
                damage = clean_f1 - plain["mean_val_macro_f1"]
                recovery = defended["mean_val_macro_f1"] - plain["mean_val_macro_f1"]
                removed_fraction = recovery / damage if abs(damage) > 1e-9 else float("nan")
                per_seed_rows.append(
                    {
                        "attack_type": attack_type,
                        "poison_fraction": fraction,
                        "model_seed": seed,
                        "plain_macro_f1": plain["mean_val_macro_f1"],
                        "defended_macro_f1": defended["mean_val_macro_f1"],
                        "clean_macro_f1": clean_f1,
                        "damage": damage,
                        "recovery": recovery,
                        "damage_removed_fraction": removed_fraction,
                        "malicious_recall": defended["mean_malicious_recall"],
                        "benign_fpr": defended["mean_benign_fpr"],
                    }
                )
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(tables_dir / "task44c2_per_seed_new_fractions.csv", index=False)

    aggregated_new = (
        per_seed.groupby(["attack_type", "poison_fraction"], as_index=False)
        .agg(
            mean_malicious_recall=("malicious_recall", "mean"),
            mean_benign_fpr=("benign_fpr", "mean"),
            mean_damage=("damage", "mean"),
            mean_damage_removed_fraction=("damage_removed_fraction", "mean"),
            min_damage_removed_fraction=("damage_removed_fraction", "min"),
        )
    )

    anchor = pd.read_csv(args.anchor_csv)
    anchor_renamed = anchor.copy()
    anchor_renamed["poison_fraction"] = ANCHOR_FRACTION
    anchor_renamed["mean_damage"] = np.nan
    anchor_renamed["min_damage_removed_fraction"] = np.nan
    anchor_cols = [
        "attack_type",
        "poison_fraction",
        "mean_malicious_recall",
        "mean_benign_fpr",
        "mean_damage",
        "mean_damage_removed_fraction",
        "min_damage_removed_fraction",
    ]

    full_curve = pd.concat(
        [aggregated_new[anchor_cols], anchor_renamed[anchor_cols]], ignore_index=True
    )
    full_curve = full_curve.sort_values(["attack_type", "poison_fraction"]).reset_index(drop=True)
    full_curve.to_csv(tables_dir / "task44c2_full_strength_curve.csv", index=False)

    h1_rows: List[Dict[str, Any]] = []
    for attack_type in ATTACK_TYPES:
        subset = full_curve[full_curve["attack_type"] == attack_type].sort_values("poison_fraction")
        recalls = subset["mean_malicious_recall"].to_numpy()
        fractions = subset["poison_fraction"].to_numpy()
        monotonic = bool(np.all(np.diff(recalls) >= -1e-9))
        h1_rows.append(
            {
                "attack_type": attack_type,
                "fractions": list(fractions),
                "recalls": list(recalls),
                "monotonic_non_decreasing": monotonic,
            }
        )
    h1_table = pd.DataFrame(h1_rows)
    h1_pass_count = int(h1_table["monotonic_non_decreasing"].sum())

    h2_rows: List[Dict[str, Any]] = []
    for attack_type in ATTACK_TYPES:
        subset = full_curve[full_curve["attack_type"] == attack_type]
        fpr_025 = float(subset[subset["poison_fraction"] == 0.25]["mean_benign_fpr"].iloc[0])
        fpr_100 = float(subset[subset["poison_fraction"] == 1.00]["mean_benign_fpr"].iloc[0])
        h2_rows.append(
            {
                "attack_type": attack_type,
                "fpr_at_0p25": fpr_025,
                "fpr_at_1p00": fpr_100,
                "relief": bool(fpr_025 < fpr_100),
            }
        )
    h2_table = pd.DataFrame(h2_rows)
    h2_pass_count = int(h2_table["relief"].sum())

    h3_rows: List[Dict[str, Any]] = []
    for attack_type in ATTACK_TYPES:
        seed_rows = per_seed[
            (per_seed["attack_type"] == attack_type) & (per_seed["poison_fraction"] == 0.25)
        ]
        mean_damage = float(seed_rows["damage"].mean())
        h3_rows.append(
            {
                "attack_type": attack_type,
                "mean_damage_at_0p25": mean_damage,
                "damage_still_meaningful": bool(mean_damage > 0.05),
            }
        )
    h3_table = pd.DataFrame(h3_rows)
    h3_pass_count = int(h3_table["damage_still_meaningful"].sum())

    h1_table.to_csv(tables_dir / "task44c2_h1_monotonicity.csv", index=False)
    h2_table.to_csv(tables_dir / "task44c2_h2_fpr_relief.csv", index=False)
    h3_table.to_csv(tables_dir / "task44c2_h3_damage_floor.csv", index=False)

    decision = {
        "experiment_version": "4.15.C2",
        "stage": "task44_c2_strength_curve_selection",
        "h1_monotonicity_pass_count": h1_pass_count,
        "h1_total_attacks": len(ATTACK_TYPES),
        "h2_fpr_relief_pass_count": h2_pass_count,
        "h2_required": 3,
        "h2_pass": h2_pass_count >= 3,
        "h3_damage_floor_pass_count": h3_pass_count,
        "h3_required": 3,
        "h3_pass": h3_pass_count >= 3,
    }
    (summary_dir / "task44c2_strength_curve_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    for attack_type in ATTACK_TYPES:
        subset = full_curve[full_curve["attack_type"] == attack_type].sort_values("poison_fraction")
        figure, axis = plt.subplots(figsize=(8.0, 5.2))
        axis.plot(subset["poison_fraction"], subset["mean_malicious_recall"], marker="o", label="Malicious recall")
        axis.plot(subset["poison_fraction"], subset["mean_benign_fpr"], marker="s", label="Benign FPR")
        axis.axhline(0.05, color="#C44E52", linestyle="--", linewidth=1, label="0.05 FPR ceiling")
        axis.set_xlabel("Poison fraction")
        axis.set_ylabel("Rate")
        axis.set_title(f"Task 44: D0 strength response, {attack_type}")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(figures_dir / f"task44c2_curve_{attack_type}.png", dpi=300, bbox_inches="tight")
        figure.savefig(figures_dir / f"task44c2_curve_{attack_type}.pdf", bbox_inches="tight")
        plt.close(figure)

    print("=" * 100)
    print("TASK 44 C2: STRENGTH CURVE")
    print("=" * 100)
    print(full_curve.to_string(index=False))
    print()
    print(f"H1 (monotonicity): {h1_pass_count}/{len(ATTACK_TYPES)} attacks monotonic")
    print(f"H2 (FPR relief at 0.25): {h2_pass_count}/5, pass={decision['h2_pass']}")
    print(f"H3 (damage floor at 0.25): {h3_pass_count}/5, pass={decision['h3_pass']}")
    print()
    print("Tables:", tables_dir)
    print("Figures:", figures_dir)
    print("Decision file:", summary_dir / "task44c2_strength_curve_decision.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

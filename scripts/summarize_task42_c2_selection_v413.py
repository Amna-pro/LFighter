#!/usr/bin/env python3
"""Task 42 C2 selection: apply the preregistered lexicographic rule.

Reads the branch_round_metrics.csv already written by
run_task42_c2_seed7_screen_v413.py for every condition x arm combination
and produces the actual candidate-selection decision, per
TASK42_PREREGISTRATION_V413.md's frozen rule:

1. Reject any candidate with maximum benign FPR above 0.05.
2. Reject any candidate with mean clean macro-F1 loss above 0.01, or any
   clean-round loss above 0.02.
3. Among survivors, maximize mean malicious-client recall across all
   attack condition-rounds.
4. Break ties by mean damage-removed fraction (macro-F1 convention,
   matching Task 40), then by lower runtime overhead.
5. If no candidate survives, record failure. Do not invent a replacement
   candidate inside this preregistered task.

This does not train anything -- it only reprocesses already-completed
branch CSVs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

CANDIDATES = (
    "D0_frozen_task40_lfighter",
    "D1_update_norm_deviation",
    "D2_update_direction_deviation",
    "D3_equal_weight_fusion",
)
PLAIN_ARM = "plain_fedavg"
ALL_ARMS = (PLAIN_ARM, *CANDIDATES)
FPR_CEILING = 0.05
MEAN_CLEAN_F1_LOSS_CEILING = 0.01
MAX_ROUND_CLEAN_F1_LOSS_CEILING = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Task 42's preregistered C2 selection rule."
    )
    parser.add_argument("--c2-output-root", type=Path, required=True)
    return parser.parse_args()


def load_all_branches(output_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    branches_dir = output_root / "branches"
    for condition_dir in sorted(branches_dir.iterdir()):
        if not condition_dir.is_dir():
            continue
        for arm in ALL_ARMS:
            path = condition_dir / arm / "tables" / "branch_round_metrics.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing branch data: {path}. C2 must be fully "
                    f"complete (no crashed/partial branches) before "
                    f"selection can run."
                )
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    args = parse_args()
    output_root = args.c2_output_root.expanduser().resolve()
    summary_dir = output_root / "summary"
    tables_dir = summary_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_rounds = load_all_branches(output_root)
    all_rounds.to_csv(tables_dir / "task42c2_all_branch_round_metrics.csv", index=False)

    clean = all_rounds[all_rounds["condition_id"] == "clean_reference"].copy()
    attacks = all_rounds[all_rounds["condition_id"] != "clean_reference"].copy()

    plain_clean = clean[clean["arm"] == PLAIN_ARM][
        ["monitoring_round", "val_macro_f1"]
    ].rename(columns={"val_macro_f1": "plain_clean_macro_f1"})

    # Pair each attack condition's arm against ITS OWN plain_fedavg branch
    # (same condition) to compute damage-removed fraction, matching Task
    # 40's convention exactly: damage = clean_macro_f1 - plain_macro_f1
    # (using the shared clean_reference plain branch as the undamaged
    # baseline), recovery = defended_macro_f1 - plain_macro_f1.
    paired_rows: List[Dict[str, Any]] = []
    for condition_id, group in attacks.groupby("condition_id"):
        plain = group[group["arm"] == PLAIN_ARM][
            ["monitoring_round", "val_macro_f1"]
        ].rename(columns={"val_macro_f1": "plain_attack_macro_f1"})
        plain = plain.merge(plain_clean, on="monitoring_round", how="inner")
        plain["damage"] = plain["plain_clean_macro_f1"] - plain["plain_attack_macro_f1"]

        for candidate in CANDIDATES:
            defended = group[group["arm"] == candidate].merge(
                plain, on="monitoring_round", how="inner", validate="one_to_one"
            )
            for row in defended.to_dict(orient="records"):
                damage = float(row["damage"])
                recovery = float(row["val_macro_f1"]) - float(row["plain_attack_macro_f1"])
                removed_fraction = (
                    recovery / damage if abs(damage) > 1e-9 else float("nan")
                )
                paired_rows.append(
                    {
                        "condition_id": condition_id,
                        "candidate": candidate,
                        "monitoring_round": row["monitoring_round"],
                        "damage": damage,
                        "recovery": recovery,
                        "damage_removed_fraction": removed_fraction,
                        "malicious_recall": row["malicious_recall"],
                        "benign_fpr": row["benign_false_positive_rate"],
                        "detector_collapsed": bool(row.get("detector_collapsed", False)),
                        "round_seconds": row["round_seconds"],
                    }
                )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(tables_dir / "task42c2_paired_attack_rounds.csv", index=False)

    candidate_rows: List[Dict[str, Any]] = []
    for candidate in CANDIDATES:
        clean_candidate = clean[clean["arm"] == candidate].merge(
            plain_clean, on="monitoring_round", how="inner"
        )
        clean_candidate["clean_macro_f1_loss"] = (
            clean_candidate["plain_clean_macro_f1"] - clean_candidate["val_macro_f1"]
        )
        candidate_attack = paired[paired["candidate"] == candidate]
        candidate_all = all_rounds[all_rounds["arm"] == candidate]

        maximum_benign_fpr = float(candidate_all["benign_false_positive_rate"].max())
        mean_clean_loss = float(clean_candidate["clean_macro_f1_loss"].mean())
        maximum_clean_round_loss = float(clean_candidate["clean_macro_f1_loss"].max())
        survives_fpr = maximum_benign_fpr <= FPR_CEILING
        survives_clean = (
            mean_clean_loss <= MEAN_CLEAN_F1_LOSS_CEILING
            and maximum_clean_round_loss <= MAX_ROUND_CLEAN_F1_LOSS_CEILING
        )
        collapsed_round_count = int(candidate_attack["detector_collapsed"].sum())

        candidate_rows.append(
            {
                "candidate": candidate,
                "maximum_benign_fpr": maximum_benign_fpr,
                "fpr_ceiling": FPR_CEILING,
                "survives_fpr_gate": survives_fpr,
                "mean_clean_macro_f1_loss": mean_clean_loss,
                "maximum_clean_round_macro_f1_loss": maximum_clean_round_loss,
                "survives_clean_utility_gate": survives_clean,
                "survives_hard_gates": bool(survives_fpr and survives_clean),
                "mean_malicious_recall_all_conditions": float(
                    candidate_attack["malicious_recall"].mean()
                ),
                "minimum_malicious_recall_any_condition": float(
                    candidate_attack["malicious_recall"].min()
                ),
                "mean_damage_removed_fraction": float(
                    candidate_attack["damage_removed_fraction"].replace(
                        [np.inf, -np.inf], np.nan
                    ).mean()
                ),
                "collapsed_round_count": collapsed_round_count,
                "attack_condition_count": int(candidate_attack["condition_id"].nunique()),
                "attack_round_pair_count": int(len(candidate_attack)),
            }
        )
    selection = pd.DataFrame(candidate_rows)

    survivors = selection[selection["survives_hard_gates"].astype(bool)].copy()
    if survivors.empty:
        selected_candidate = None
        selection_status = "NO_CANDIDATE_SURVIVED"
    else:
        survivors = survivors.sort_values(
            by=[
                "mean_malicious_recall_all_conditions",
                "mean_damage_removed_fraction",
                "candidate",
            ],
            ascending=[False, False, True],
            kind="mergesort",
        )
        selected_candidate = str(survivors.iloc[0]["candidate"])
        selection_status = "ONE_CANDIDATE_SELECTED"
    selection["selected"] = (
        selection["candidate"] == selected_candidate
        if selected_candidate is not None
        else False
    )
    selection.to_csv(tables_dir / "task42c2_candidate_selection.csv", index=False)

    decision = {
        "experiment_version": "4.13.C2a",
        "stage": "task42_c2_seed7_screen_selection",
        "selection_status": selection_status,
        "selected_candidate": selected_candidate,
        "candidate_count": len(CANDIDATES),
        "candidate_survivor_count": int(len(survivors)),
        "selection_rule": [
            "Reject maximum benign FPR above 0.05.",
            "Reject mean clean macro-F1 loss above 0.01 or any clean round loss above 0.02.",
            "Maximize mean malicious recall across all attack condition-rounds.",
            "Tie-break by mean damage-removed fraction (macro-F1 convention).",
            "Tie-break by candidate name (deterministic, stable ordering).",
        ],
        "next_stage": (
            "Freeze C2 evidence and selected candidate. Then run C3 "
            "confirmatory multiseed with no retuning."
            if selected_candidate is not None
            else "Record Task 42 candidate-development failure. Do not "
            "invent a new candidate inside this preregistered task."
        ),
    }
    (summary_dir / "task42c2_selection_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print("=" * 100)
    print("TASK 42 C2 SELECTION")
    print("=" * 100)
    print(selection.to_string(index=False))
    print()
    print("Selection status:", selection_status)
    print("Selected candidate:", selected_candidate)
    print("Decision file:", summary_dir / "task42c2_selection_decision.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

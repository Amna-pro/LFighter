#!/usr/bin/env python3
"""Aggregate frozen V3.10 true-warmup chronology results across seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", required=True)
    return parser.parse_args()


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    seed_rows: List[dict] = []
    round_frames: List[pd.DataFrame] = []

    for seed in seeds:
        seed_root = input_root / f"seed_{seed}"
        clean_path = seed_root / "clean_continuation" / "tables" / "continuation_round_metrics.csv"
        attack_path = seed_root / "attack_continuation" / "tables" / "continuation_round_metrics.csv"
        paired_path = seed_root / "paired_summary" / "tables" / "paired_aggregate_summary.csv"
        for path in (clean_path, attack_path, paired_path):
            if not path.exists():
                raise FileNotFoundError(path)

        clean = pd.read_csv(clean_path)
        attack = pd.read_csv(attack_path)
        paired = pd.read_csv(paired_path).iloc[0]

        merged = clean.merge(
            attack,
            on=["monitoring_round", "global_round"],
            suffixes=("_clean", "_attack"),
        )
        merged.insert(0, "seed", seed)
        round_frames.append(merged)

        seed_rows.append({
            "seed": seed,
            **paired.to_dict(),
            "clean_maximum_benign_fpr": float(clean["benign_false_positive_rate"].max()),
            "clean_final_round_benign_fpr": float(clean.iloc[-1]["benign_false_positive_rate"]),
            "attack_first_round_malicious_recall": float(attack.iloc[0]["malicious_recall"]),
            "attack_final_round_malicious_recall": float(attack.iloc[-1]["malicious_recall"]),
            "attack_final_round_benign_fpr": float(attack.iloc[-1]["benign_false_positive_rate"]),
        })

    seed_table = pd.DataFrame(seed_rows).sort_values("seed")
    round_table = pd.concat(round_frames, ignore_index=True).sort_values(
        ["seed", "monitoring_round"]
    )
    seed_table.to_csv(tables / "multiseed_seed_level_summary.csv", index=False)
    round_table.to_csv(tables / "multiseed_round_level_summary.csv", index=False)

    aggregate = pd.DataFrame([{
        "seed_count": int(len(seed_table)),
        "mean_clean_macro_f1": float(seed_table["clean_mean_macro_f1"].mean()),
        "mean_attack_macro_f1": float(seed_table["attack_mean_macro_f1"].mean()),
        "mean_attack_minus_clean_macro_f1": float(seed_table["attack_minus_clean_macro_f1"].mean()),
        "mean_clean_source_to_target_rate": float(seed_table["clean_mean_source_to_target_rate"].mean()),
        "mean_attack_source_to_target_rate": float(seed_table["attack_mean_source_to_target_rate"].mean()),
        "mean_attack_minus_clean_source_to_target_rate": float(seed_table["attack_minus_clean_source_to_target_rate"].mean()),
        "mean_clean_benign_fpr": float(seed_table["clean_mean_benign_fpr"].mean()),
        "maximum_clean_benign_fpr_any_seed": float(seed_table["clean_maximum_benign_fpr"].max()),
        "mean_attack_benign_fpr": float(seed_table["attack_mean_benign_fpr"].mean()),
        "maximum_attack_benign_fpr_any_seed": float(seed_table["attack_maximum_benign_fpr"].max()),
        "mean_attack_malicious_recall": float(seed_table["attack_mean_malicious_recall"].mean()),
        "minimum_attack_malicious_recall_any_seed_round": float(seed_table["attack_minimum_malicious_recall"].min()),
        "mean_first_round_malicious_recall": float(seed_table["attack_first_round_malicious_recall"].mean()),
        "minimum_first_round_malicious_recall": float(seed_table["attack_first_round_malicious_recall"].min()),
        "minimum_final_round_malicious_recall": float(seed_table["attack_final_round_malicious_recall"].min()),
        "passes_mean_recall_090": bool(seed_table["attack_mean_malicious_recall"].mean() >= 0.90),
        "passes_minimum_round_recall_075": bool(seed_table["attack_minimum_malicious_recall"].min() >= 0.75),
        "passes_mean_clean_fpr_005": bool(seed_table["clean_mean_benign_fpr"].mean() <= 0.05),
        "passes_maximum_clean_fpr_010": bool(seed_table["clean_maximum_benign_fpr"].max() <= 0.10),
        "passes_mean_attack_fpr_005": bool(seed_table["attack_mean_benign_fpr"].mean() <= 0.05),
        "passes_maximum_attack_fpr_010": bool(seed_table["attack_maximum_benign_fpr"].max() <= 0.10),
        "common_warmup_checkpoint_verified_all": bool(seed_table["common_warmup_checkpoint_verified"].all()),
        "defense_weights_applied": False,
        "test_sets_accessed": False,
    }])
    aggregate.to_csv(tables / "multiseed_aggregate_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(seed_table))
    width = 0.36
    ax.bar(x - width / 2, seed_table["attack_mean_malicious_recall"], width, label="Mean malicious recall")
    ax.bar(x + width / 2, seed_table["attack_first_round_malicious_recall"], width, label="First-round recall")
    ax.set_xticks(x, seed_table["seed"].astype(str))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Recall")
    ax.set_title("V3.10.1 true-chronology detection recall")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "multiseed_detection_recall")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(seed_table["seed"].astype(str), seed_table["clean_mean_benign_fpr"], marker="o", label="Clean mean FPR")
    ax.plot(seed_table["seed"].astype(str), seed_table["attack_mean_benign_fpr"], marker="s", label="Attack mean FPR")
    ax.axhline(0.05, linestyle="--", label="5% target")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Benign false-positive rate")
    ax.set_title("V3.10.1 benign false-positive rates")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "multiseed_benign_fpr")

    with (output_dir / "v3101_multiseed_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_version": "3.10.1",
                "seeds": seeds,
                "protocol_frozen": True,
                "warmup_rounds": 4,
                "continuation_rounds": 4,
                "ema_decay": 0.65,
                "monitoring_ema_reset": True,
                "defense_weights_applied": False,
                "test_sets_accessed": False,
            },
            handle,
            indent=2,
        )

    print("V3.10.1 multiseed aggregate complete")
    print(aggregate.to_string(index=False))
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

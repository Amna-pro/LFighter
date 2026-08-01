#!/usr/bin/env python3
"""Summarize the clean and attacked V3.10 continuations from one warmup checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", required=True, type=Path)
    parser.add_argument("--attack-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    clean_meta = json.load(open(args.clean_dir / "post_warmup_capture_v310_metadata.json", encoding="utf-8"))
    attack_meta = json.load(open(args.attack_dir / "post_warmup_capture_v310_metadata.json", encoding="utf-8"))
    for key in ("partition_hash_sha256", "warmup_profile_sha256", "common_branch_checkpoint", "model_seed"):
        if clean_meta[key] != attack_meta[key]:
            raise RuntimeError(f"Paired branch mismatch for {key}")

    clean = pd.read_csv(args.clean_dir / "tables" / "continuation_round_metrics.csv")
    attack = pd.read_csv(args.attack_dir / "tables" / "continuation_round_metrics.csv")
    paired = clean.merge(attack, on=["monitoring_round", "global_round"], suffixes=("_clean", "_attack"))
    paired["macro_f1_attack_minus_clean"] = paired["val_macro_f1_attack"] - paired["val_macro_f1_clean"]
    paired["source_to_target_attack_minus_clean"] = paired["val_source_to_target_rate_attack"] - paired["val_source_to_target_rate_clean"]
    paired.to_csv(tables_dir / "paired_round_comparison.csv", index=False)

    summary = pd.DataFrame([
        {
            "clean_mean_macro_f1": clean["val_macro_f1"].mean(),
            "attack_mean_macro_f1": attack["val_macro_f1"].mean(),
            "attack_minus_clean_macro_f1": attack["val_macro_f1"].mean() - clean["val_macro_f1"].mean(),
            "clean_mean_source_to_target_rate": clean["val_source_to_target_rate"].mean(),
            "attack_mean_source_to_target_rate": attack["val_source_to_target_rate"].mean(),
            "attack_minus_clean_source_to_target_rate": attack["val_source_to_target_rate"].mean() - clean["val_source_to_target_rate"].mean(),
            "clean_mean_benign_fpr": clean["benign_false_positive_rate"].mean(),
            "attack_mean_benign_fpr": attack["benign_false_positive_rate"].mean(),
            "attack_mean_malicious_recall": attack["malicious_recall"].mean(),
            "attack_minimum_malicious_recall": attack["malicious_recall"].min(),
            "attack_maximum_benign_fpr": attack["benign_false_positive_rate"].max(),
            "common_warmup_checkpoint_verified": True,
            "defense_weights_applied": False,
            "test_sets_accessed": False,
        }
    ])
    summary.to_csv(tables_dir / "paired_aggregate_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(clean["monitoring_round"], clean["val_macro_f1"], marker="o", label="Clean continuation")
    ax.plot(attack["monitoring_round"], attack["val_macro_f1"], marker="s", label="Attacked continuation")
    ax.set_xlabel("Post-warmup monitoring round")
    ax.set_ylabel("Validation macro F1")
    ax.set_title("V3.10 paired continuation utility")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "paired_macro_f1.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "paired_macro_f1.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Paired V3.10 summary complete")
    print(summary.to_string(index=False))
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

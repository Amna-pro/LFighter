#!/usr/bin/env python3
"""Compare V3.11 defended branches with the paired V3.10 plain-FedAvg baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-clean-dir", required=True, type=Path)
    p.add_argument("--baseline-attack-dir", required=True, type=Path)
    p.add_argument("--defended-clean-dir", required=True, type=Path)
    p.add_argument("--defended-attack-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def load_rounds(path: Path, defended: bool) -> pd.DataFrame:
    filename = (
        "integrated_round_metrics.csv"
        if defended
        else "continuation_round_metrics.csv"
    )
    full = path / "tables" / filename
    if not full.exists():
        raise FileNotFoundError(full)
    table = pd.read_csv(full)
    if len(table) != 4:
        raise ValueError(f"Expected four rounds in {full}, found {len(table)}")
    return table.sort_values("monitoring_round").reset_index(drop=True)


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def relative_reduction(defended: float, baseline: float) -> float:
    if baseline <= 1e-12:
        return 0.0
    return float(1.0 - defended / baseline)


def main() -> int:
    a = parse_args()
    out = a.output_dir.expanduser().resolve()
    tables = out / "tables"
    figures = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    bc = load_rounds(a.baseline_clean_dir, defended=False)
    ba = load_rounds(a.baseline_attack_dir, defended=False)
    dc = load_rounds(a.defended_clean_dir, defended=True)
    da = load_rounds(a.defended_attack_dir, defended=True)

    key = ["monitoring_round", "global_round"]
    merged = (
        bc[key + ["val_macro_f1", "val_source_to_target_rate"]]
        .rename(columns={
            "val_macro_f1": "baseline_clean_macro_f1",
            "val_source_to_target_rate": "baseline_clean_source_to_target_rate",
        })
        .merge(
            ba[key + ["val_macro_f1", "val_source_to_target_rate"]],
            on=key,
        )
        .rename(columns={
            "val_macro_f1": "baseline_attack_macro_f1",
            "val_source_to_target_rate": "baseline_attack_source_to_target_rate",
        })
        .merge(
            dc[key + [
                "val_macro_f1",
                "val_source_to_target_rate",
                "mean_benign_trust",
                "benign_low_trust_rate",
            ]],
            on=key,
        )
        .rename(columns={
            "val_macro_f1": "defended_clean_macro_f1",
            "val_source_to_target_rate": "defended_clean_source_to_target_rate",
            "mean_benign_trust": "defended_clean_mean_benign_trust",
            "benign_low_trust_rate": "defended_clean_benign_low_trust_rate",
        })
        .merge(
            da[key + [
                "val_macro_f1",
                "val_source_to_target_rate",
                "malicious_recall",
                "benign_false_positive_rate",
                "mean_benign_trust",
                "mean_malicious_trust",
                "malicious_influence_reduction",
                "malicious_harm_reduction",
            ]],
            on=key,
        )
        .rename(columns={
            "val_macro_f1": "defended_attack_macro_f1",
            "val_source_to_target_rate": "defended_attack_source_to_target_rate",
            "malicious_recall": "defended_attack_malicious_recall",
            "benign_false_positive_rate": "defended_attack_benign_fpr",
            "mean_benign_trust": "defended_attack_mean_benign_trust",
            "mean_malicious_trust": "defended_attack_mean_malicious_trust",
        })
    )

    merged["attack_rate_absolute_reduction"] = (
        merged["baseline_attack_source_to_target_rate"]
        - merged["defended_attack_source_to_target_rate"]
    )
    merged["attack_rate_relative_reduction"] = [
        relative_reduction(defended, baseline)
        for defended, baseline in zip(
            merged["defended_attack_source_to_target_rate"],
            merged["baseline_attack_source_to_target_rate"],
        )
    ]
    merged["clean_macro_f1_delta"] = (
        merged["defended_clean_macro_f1"]
        - merged["baseline_clean_macro_f1"]
    )
    merged["attack_macro_f1_delta"] = (
        merged["defended_attack_macro_f1"]
        - merged["baseline_attack_macro_f1"]
    )
    merged.to_csv(tables / "v311_round_level_comparison.csv", index=False)

    aggregate = pd.DataFrame([{
        "round_count": 4,
        "baseline_clean_mean_macro_f1": bc["val_macro_f1"].mean(),
        "defended_clean_mean_macro_f1": dc["val_macro_f1"].mean(),
        "defended_minus_baseline_clean_macro_f1": (
            dc["val_macro_f1"].mean() - bc["val_macro_f1"].mean()
        ),
        "baseline_attack_mean_macro_f1": ba["val_macro_f1"].mean(),
        "defended_attack_mean_macro_f1": da["val_macro_f1"].mean(),
        "defended_minus_baseline_attack_macro_f1": (
            da["val_macro_f1"].mean() - ba["val_macro_f1"].mean()
        ),
        "baseline_attack_mean_source_to_target_rate": (
            ba["val_source_to_target_rate"].mean()
        ),
        "defended_attack_mean_source_to_target_rate": (
            da["val_source_to_target_rate"].mean()
        ),
        "mean_absolute_attack_rate_reduction": (
            ba["val_source_to_target_rate"].mean()
            - da["val_source_to_target_rate"].mean()
        ),
        "relative_attack_rate_reduction": relative_reduction(
            da["val_source_to_target_rate"].mean(),
            ba["val_source_to_target_rate"].mean(),
        ),
        "maximum_defended_attack_source_to_target_rate": (
            da["val_source_to_target_rate"].max()
        ),
        "defended_clean_mean_benign_trust": dc["mean_benign_trust"].mean(),
        "minimum_defended_clean_round_benign_trust": (
            dc["mean_benign_trust"].min()
        ),
        "defended_clean_maximum_low_trust_rate": (
            dc["benign_low_trust_rate"].max()
        ),
        "defended_attack_mean_benign_trust": da["mean_benign_trust"].mean(),
        "defended_attack_mean_malicious_trust": da["mean_malicious_trust"].mean(),
        "defended_attack_mean_malicious_recall": da["malicious_recall"].mean(),
        "defended_attack_minimum_malicious_recall": da["malicious_recall"].min(),
        "defended_attack_mean_benign_fpr": (
            da["benign_false_positive_rate"].mean()
        ),
        "defended_attack_maximum_benign_fpr": (
            da["benign_false_positive_rate"].max()
        ),
        "defended_attack_mean_malicious_influence_reduction": (
            da["malicious_influence_reduction"].mean()
        ),
        "defended_attack_minimum_malicious_influence_reduction": (
            da["malicious_influence_reduction"].min()
        ),
        "defended_attack_mean_malicious_harm_reduction": (
            da["malicious_harm_reduction"].mean()
        ),
        "defended_attack_minimum_malicious_harm_reduction": (
            da["malicious_harm_reduction"].min()
        ),
        "passes_clean_macro_f1_penalty_002": bool(
            dc["val_macro_f1"].mean() - bc["val_macro_f1"].mean() >= -0.02
        ),
        "passes_relative_attack_reduction_025": bool(
            relative_reduction(
                da["val_source_to_target_rate"].mean(),
                ba["val_source_to_target_rate"].mean(),
            ) >= 0.25
        ),
        "passes_mean_benign_trust_095": bool(
            da["mean_benign_trust"].mean() >= 0.95
        ),
        "passes_minimum_influence_reduction_030": bool(
            da["malicious_influence_reduction"].min() >= 0.30
        ),
        "passes_minimum_harm_reduction_030": bool(
            da["malicious_harm_reduction"].min() >= 0.30
        ),
        "common_warmup_checkpoint_required": True,
        "count_cap_used": False,
        "defense_weights_applied": True,
        "test_sets_accessed": False,
    }])
    aggregate.to_csv(tables / "v311_aggregate_decision.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        merged["monitoring_round"],
        merged["baseline_attack_source_to_target_rate"],
        marker="o",
        label="Plain FedAvg attack",
    )
    ax.plot(
        merged["monitoring_round"],
        merged["defended_attack_source_to_target_rate"],
        marker="s",
        label="V3.11 defended attack",
    )
    ax.set_xlabel("Post-warmup monitoring round")
    ax.set_ylabel("DDoS to Benign rate")
    ax.set_ylim(0, 1)
    ax.set_title("V3.11 closed-loop attack suppression")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "attack_rate_comparison")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        merged["monitoring_round"],
        merged["defended_attack_mean_benign_trust"],
        marker="o",
        label="Benign trust",
    )
    ax.plot(
        merged["monitoring_round"],
        merged["defended_attack_mean_malicious_trust"],
        marker="s",
        label="Malicious trust",
    )
    ax.set_xlabel("Post-warmup monitoring round")
    ax.set_ylabel("Mean trust")
    ax.set_ylim(0, 1)
    ax.set_title("V3.11 closed-loop trust separation")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "trust_separation")

    with (out / "v311_summary_metadata.json").open("w", encoding="utf-8") as h:
        json.dump({
            "experiment_version": "3.11",
            "development_seed": 42,
            "paired_common_warmup_checkpoint": True,
            "baseline_aggregation": "raw_sample_count_fedavg",
            "defended_aggregation": "raw_sample_count_times_frozen_soft_trust",
            "count_cap_used": False,
            "test_sets_accessed": False,
        }, h, indent=2)

    print("V3.11 paired closed-loop summary complete")
    print(aggregate.to_string(index=False))
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

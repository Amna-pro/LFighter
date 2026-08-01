#!/usr/bin/env python3
"""Summarize V3.12 trusted reconstruction against prior aggregation branches."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain-clean-dir", required=True, type=Path)
    parser.add_argument("--plain-attack-dir", required=True, type=Path)
    parser.add_argument("--soft-attack-dir", required=True, type=Path)
    parser.add_argument("--hard-attack-dir", required=True, type=Path)
    parser.add_argument("--oracle-drop-attack-dir", required=True, type=Path)
    parser.add_argument("--reconstruction-clean-dir", required=True, type=Path)
    parser.add_argument("--reconstruction-attack-dir", required=True, type=Path)
    parser.add_argument(
        "--oracle-clean-replacement-attack-dir",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load(path: Path, filename: str) -> pd.DataFrame:
    full = path / "tables" / filename
    if not full.exists():
        raise FileNotFoundError(full)
    table = pd.read_csv(full).sort_values("monitoring_round").reset_index(
        drop=True
    )
    if len(table) != 4:
        raise ValueError(f"Expected four rows in {full}")
    return table


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def excess_removed(
    method_attack: float,
    method_clean: float,
    plain_attack: float,
    plain_clean: float,
) -> float:
    denominator = plain_attack - plain_clean
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(
        1.0 - (method_attack - method_clean) / denominator
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plain_clean = load(
        args.plain_clean_dir,
        "continuation_round_metrics.csv",
    )
    plain_attack = load(
        args.plain_attack_dir,
        "continuation_round_metrics.csv",
    )
    soft_attack = load(
        args.soft_attack_dir,
        "integrated_round_metrics.csv",
    )
    hard_attack = load(
        args.hard_attack_dir,
        "headroom_round_metrics.csv",
    )
    oracle_drop_attack = load(
        args.oracle_drop_attack_dir,
        "headroom_round_metrics.csv",
    )
    reconstruction_clean = load(
        args.reconstruction_clean_dir,
        "reconstruction_round_metrics.csv",
    )
    reconstruction_attack = load(
        args.reconstruction_attack_dir,
        "reconstruction_round_metrics.csv",
    )
    oracle_clean_replacement = load(
        args.oracle_clean_replacement_attack_dir,
        "reconstruction_round_metrics.csv",
    )

    plain_clean_rate = float(
        plain_clean["val_source_to_target_rate"].mean()
    )
    plain_attack_rate = float(
        plain_attack["val_source_to_target_rate"].mean()
    )
    reconstruction_clean_rate = float(
        reconstruction_clean["val_source_to_target_rate"].mean()
    )

    oracle_clean_max_metric_difference = float(
        max(
            np.max(
                np.abs(
                    oracle_clean_replacement["val_macro_f1"].to_numpy()
                    - plain_clean["val_macro_f1"].to_numpy()
                )
            ),
            np.max(
                np.abs(
                    oracle_clean_replacement[
                        "val_source_to_target_rate"
                    ].to_numpy()
                    - plain_clean[
                        "val_source_to_target_rate"
                    ].to_numpy()
                )
            ),
        )
    )

    methods = [
        ("plain_attack", plain_attack, plain_clean_rate),
        ("soft_trust", soft_attack, plain_clean_rate),
        ("hard_gate", hard_attack, plain_clean_rate),
        ("oracle_drop", oracle_drop_attack, plain_clean_rate),
        (
            "trusted_reconstruction",
            reconstruction_attack,
            reconstruction_clean_rate,
        ),
        (
            "oracle_clean_replacement",
            oracle_clean_replacement,
            plain_clean_rate,
        ),
    ]

    rows = []
    for name, table, clean_reference_rate in methods:
        attack_rate = float(table["val_source_to_target_rate"].mean())
        rows.append(
            {
                "method": name,
                "mean_macro_f1": float(table["val_macro_f1"].mean()),
                "mean_source_to_target_rate": attack_rate,
                "absolute_rate_reduction_vs_plain_attack": float(
                    plain_attack_rate - attack_rate
                ),
                "attack_excess_removed_relative_to_clean": excess_removed(
                    attack_rate,
                    clean_reference_rate,
                    plain_attack_rate,
                    plain_clean_rate,
                ),
                "maximum_source_to_target_rate": float(
                    table["val_source_to_target_rate"].max()
                ),
                "rounds_improved_vs_plain_attack": int(
                    np.sum(
                        table["val_source_to_target_rate"].to_numpy()
                        < plain_attack[
                            "val_source_to_target_rate"
                        ].to_numpy()
                    )
                ),
            }
        )

    method_table = pd.DataFrame(rows)
    method_table.to_csv(
        tables_dir / "v312_method_comparison.csv",
        index=False,
    )

    reconstruction_attack_rate = float(
        reconstruction_attack["val_source_to_target_rate"].mean()
    )
    reconstruction_excess_removed = excess_removed(
        reconstruction_attack_rate,
        reconstruction_clean_rate,
        plain_attack_rate,
        plain_clean_rate,
    )

    decision = pd.DataFrame(
        [
            {
                "plain_clean_mean_macro_f1": float(
                    plain_clean["val_macro_f1"].mean()
                ),
                "reconstruction_clean_mean_macro_f1": float(
                    reconstruction_clean["val_macro_f1"].mean()
                ),
                "reconstruction_clean_macro_f1_delta": float(
                    reconstruction_clean["val_macro_f1"].mean()
                    - plain_clean["val_macro_f1"].mean()
                ),
                "plain_clean_mean_source_to_target_rate": (
                    plain_clean_rate
                ),
                "plain_attack_mean_source_to_target_rate": (
                    plain_attack_rate
                ),
                "trusted_reconstruction_attack_mean_source_to_target_rate": (
                    reconstruction_attack_rate
                ),
                "trusted_reconstruction_attack_excess_removed": (
                    reconstruction_excess_removed
                ),
                "trusted_reconstruction_rounds_improved": int(
                    np.sum(
                        reconstruction_attack[
                            "val_source_to_target_rate"
                        ].to_numpy()
                        < plain_attack[
                            "val_source_to_target_rate"
                        ].to_numpy()
                    )
                ),
                "soft_trust_attack_mean_source_to_target_rate": float(
                    soft_attack["val_source_to_target_rate"].mean()
                ),
                "hard_gate_attack_mean_source_to_target_rate": float(
                    hard_attack["val_source_to_target_rate"].mean()
                ),
                "oracle_drop_attack_mean_source_to_target_rate": float(
                    oracle_drop_attack[
                        "val_source_to_target_rate"
                    ].mean()
                ),
                "oracle_clean_replacement_mean_source_to_target_rate": float(
                    oracle_clean_replacement[
                        "val_source_to_target_rate"
                    ].mean()
                ),
                "oracle_clean_replacement_max_metric_difference_from_clean": (
                    oracle_clean_max_metric_difference
                ),
                "passes_oracle_clean_replacement_verification": bool(
                    oracle_clean_max_metric_difference <= 1e-8
                ),
                "passes_clean_macro_f1_penalty_002": bool(
                    reconstruction_clean["val_macro_f1"].mean()
                    - plain_clean["val_macro_f1"].mean()
                    >= -0.02
                ),
                "passes_attack_excess_removed_050": bool(
                    reconstruction_excess_removed >= 0.50
                ),
                "passes_better_than_soft_trust": bool(
                    reconstruction_attack_rate
                    < soft_attack["val_source_to_target_rate"].mean()
                ),
                "passes_three_of_four_rounds_improved": bool(
                    np.sum(
                        reconstruction_attack[
                            "val_source_to_target_rate"
                        ].to_numpy()
                        < plain_attack[
                            "val_source_to_target_rate"
                        ].to_numpy()
                    )
                    >= 3
                ),
                "count_cap_used": False,
                "test_sets_accessed": False,
            }
        ]
    )
    decision.to_csv(
        tables_dir / "v312_aggregate_decision.csv",
        index=False,
    )

    round_table = plain_clean[
        ["monitoring_round", "global_round"]
    ].copy()
    for name, table in [
        ("plain_clean", plain_clean),
        ("plain_attack", plain_attack),
        ("soft_trust", soft_attack),
        ("hard_gate", hard_attack),
        ("oracle_drop", oracle_drop_attack),
        ("trusted_reconstruction", reconstruction_attack),
        ("oracle_clean_replacement", oracle_clean_replacement),
    ]:
        round_table[f"{name}_source_to_target_rate"] = table[
            "val_source_to_target_rate"
        ].to_numpy()
        round_table[f"{name}_macro_f1"] = table[
            "val_macro_f1"
        ].to_numpy()
    round_table.to_csv(
        tables_dir / "v312_round_level_comparison.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for name in [
        "plain_clean",
        "plain_attack",
        "soft_trust",
        "hard_gate",
        "oracle_drop",
        "trusted_reconstruction",
        "oracle_clean_replacement",
    ]:
        ax.plot(
            round_table["monitoring_round"],
            round_table[f"{name}_source_to_target_rate"],
            marker="o",
            label=name,
        )
    ax.set_xlabel("Post-warmup monitoring round")
    ax.set_ylabel("DDoS to Benign rate")
    ax.set_ylim(0, 1)
    ax.set_title("V3.12 trusted update-reconstruction comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "source_to_target_comparison")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_table = method_table[
        method_table["method"].ne("plain_attack")
    ].copy()
    positions = np.arange(len(plot_table))
    ax.bar(
        positions,
        plot_table["attack_excess_removed_relative_to_clean"],
    )
    ax.set_xticks(
        positions,
        plot_table["method"],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Fraction of attack-induced excess removed")
    ax.set_title("V3.12 causal recovery effectiveness")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "attack_excess_removed")

    with (output_dir / "v312_summary_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "experiment_version": "3.12",
                "purpose": (
                    "test whether trusted update reconstruction preserves "
                    "non-IID client information better than exclusion"
                ),
                "development_seed": 42,
                "oracle_clean_replacement_is_non_deployable": True,
                "test_sets_accessed": False,
            },
            handle,
            indent=2,
        )

    print("Trusted Update Reconstruction Summary V3.12 complete")
    print(decision.to_string(index=False))
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

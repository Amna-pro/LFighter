#!/usr/bin/env python3
"""Summarize V3.11.2 aggregation headroom and residual-weight causality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plain-clean-dir", required=True, type=Path)
    p.add_argument("--plain-attack-dir", required=True, type=Path)
    p.add_argument("--soft-clean-dir", required=True, type=Path)
    p.add_argument("--soft-attack-dir", required=True, type=Path)
    p.add_argument("--hard-clean-dir", required=True, type=Path)
    p.add_argument("--hard-attack-dir", required=True, type=Path)
    p.add_argument("--oracle-attack-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def load(path: Path, kind: str) -> pd.DataFrame:
    filename = {
        "plain": "continuation_round_metrics.csv",
        "soft": "integrated_round_metrics.csv",
        "headroom": "headroom_round_metrics.csv",
    }[kind]
    full = path / "tables" / filename
    if not full.exists():
        raise FileNotFoundError(full)
    table = pd.read_csv(full).sort_values("monitoring_round").reset_index(drop=True)
    if len(table) != 4:
        raise ValueError(f"Expected four rows in {full}, found {len(table)}")
    return table


def safe_relative_reduction(method: float, baseline: float) -> float:
    if baseline <= 1e-12:
        return 0.0
    return float(1.0 - method / baseline)


def attack_excess_removed(method_attack: float, method_clean: float, plain_attack: float, plain_clean: float) -> float:
    baseline_excess = plain_attack - plain_clean
    method_excess = method_attack - method_clean
    if abs(baseline_excess) <= 1e-12:
        return 0.0
    return float(1.0 - method_excess / baseline_excess)


def save(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    a = parse_args()
    out = a.output_dir.expanduser().resolve()
    tables = out / "tables"
    figures = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    pc = load(a.plain_clean_dir, "plain")
    pa = load(a.plain_attack_dir, "plain")
    sc = load(a.soft_clean_dir, "soft")
    sa = load(a.soft_attack_dir, "soft")
    hc = load(a.hard_clean_dir, "headroom")
    ha = load(a.hard_attack_dir, "headroom")
    oa = load(a.oracle_attack_dir, "headroom")

    branches: Dict[str, pd.DataFrame] = {
        "plain_clean": pc,
        "plain_attack": pa,
        "soft_clean": sc,
        "soft_attack": sa,
        "hard_clean": hc,
        "hard_attack": ha,
        "oracle_attack": oa,
    }

    round_table = pc[["monitoring_round", "global_round"]].copy()
    for name, table in branches.items():
        round_table[f"{name}_macro_f1"] = table["val_macro_f1"].to_numpy()
        round_table[f"{name}_source_to_target_rate"] = table[
            "val_source_to_target_rate"
        ].to_numpy()
    for name, table in {
        "soft_attack": sa,
        "hard_attack": ha,
        "oracle_attack": oa,
    }.items():
        round_table[f"{name}_malicious_influence_reduction"] = table[
            "malicious_influence_reduction"
        ].to_numpy()
        round_table[f"{name}_malicious_harm_reduction"] = table[
            "malicious_harm_reduction"
        ].to_numpy()

    round_table["soft_attack_rate_reduction_vs_plain"] = (
        round_table["plain_attack_source_to_target_rate"]
        - round_table["soft_attack_source_to_target_rate"]
    )
    round_table["hard_attack_rate_reduction_vs_plain"] = (
        round_table["plain_attack_source_to_target_rate"]
        - round_table["hard_attack_source_to_target_rate"]
    )
    round_table["oracle_attack_rate_reduction_vs_plain"] = (
        round_table["plain_attack_source_to_target_rate"]
        - round_table["oracle_attack_source_to_target_rate"]
    )
    round_table.to_csv(tables / "v3112_round_level_headroom.csv", index=False)

    plain_clean_rate = float(pc["val_source_to_target_rate"].mean())
    plain_attack_rate = float(pa["val_source_to_target_rate"].mean())
    soft_clean_rate = float(sc["val_source_to_target_rate"].mean())
    soft_attack_rate = float(sa["val_source_to_target_rate"].mean())
    hard_clean_rate = float(hc["val_source_to_target_rate"].mean())
    hard_attack_rate = float(ha["val_source_to_target_rate"].mean())
    oracle_attack_rate = float(oa["val_source_to_target_rate"].mean())

    rows = []
    for method, clean_table, attack_table, clean_rate, attack_rate in [
        ("soft_trust", sc, sa, soft_clean_rate, soft_attack_rate),
        ("hard_gate", hc, ha, hard_clean_rate, hard_attack_rate),
        ("oracle_drop", pc, oa, plain_clean_rate, oracle_attack_rate),
    ]:
        rows.append({
            "method": method,
            "clean_mean_macro_f1": float(clean_table["val_macro_f1"].mean()),
            "clean_macro_f1_delta_vs_plain": float(
                clean_table["val_macro_f1"].mean() - pc["val_macro_f1"].mean()
            ),
            "attack_mean_macro_f1": float(attack_table["val_macro_f1"].mean()),
            "attack_macro_f1_delta_vs_plain": float(
                attack_table["val_macro_f1"].mean() - pa["val_macro_f1"].mean()
            ),
            "clean_mean_source_to_target_rate": clean_rate,
            "attack_mean_source_to_target_rate": attack_rate,
            "absolute_attack_rate_reduction_vs_plain": float(
                plain_attack_rate - attack_rate
            ),
            "relative_attack_rate_reduction_vs_plain": safe_relative_reduction(
                attack_rate, plain_attack_rate
            ),
            "attack_excess_removed_relative_to_clean": attack_excess_removed(
                attack_rate,
                clean_rate,
                plain_attack_rate,
                plain_clean_rate,
            ),
            "maximum_attack_source_to_target_rate": float(
                attack_table["val_source_to_target_rate"].max()
            ),
            "mean_malicious_influence_reduction": float(
                attack_table["malicious_influence_reduction"].mean()
            ),
            "minimum_malicious_influence_reduction": float(
                attack_table["malicious_influence_reduction"].min()
            ),
            "mean_malicious_harm_reduction": float(
                attack_table["malicious_harm_reduction"].mean()
            ),
            "minimum_malicious_harm_reduction": float(
                attack_table["malicious_harm_reduction"].min()
            ),
        })
    method_table = pd.DataFrame(rows)
    method_table.to_csv(tables / "v3112_method_comparison.csv", index=False)

    oracle_gain_beyond_soft = float(soft_attack_rate - oracle_attack_rate)
    hard_gain_beyond_soft = float(soft_attack_rate - hard_attack_rate)
    residual_weight_explanation_supported = bool(
        hard_attack_rate < soft_attack_rate
        and oracle_attack_rate <= hard_attack_rate
        and hard_gain_beyond_soft >= 0.01
    )
    limited_aggregation_headroom_supported = bool(
        abs(oracle_attack_rate - soft_attack_rate) < 0.01
    )

    decision = pd.DataFrame([{
        "plain_clean_mean_source_to_target_rate": plain_clean_rate,
        "plain_attack_mean_source_to_target_rate": plain_attack_rate,
        "soft_attack_mean_source_to_target_rate": soft_attack_rate,
        "hard_attack_mean_source_to_target_rate": hard_attack_rate,
        "oracle_attack_mean_source_to_target_rate": oracle_attack_rate,
        "hard_gain_beyond_soft": hard_gain_beyond_soft,
        "oracle_gain_beyond_soft": oracle_gain_beyond_soft,
        "soft_attack_excess_removed": float(
            method_table.loc[
                method_table["method"].eq("soft_trust"),
                "attack_excess_removed_relative_to_clean",
            ].iloc[0]
        ),
        "hard_attack_excess_removed": float(
            method_table.loc[
                method_table["method"].eq("hard_gate"),
                "attack_excess_removed_relative_to_clean",
            ].iloc[0]
        ),
        "oracle_attack_excess_removed": float(
            method_table.loc[
                method_table["method"].eq("oracle_drop"),
                "attack_excess_removed_relative_to_clean",
            ].iloc[0]
        ),
        "residual_weight_explanation_supported": residual_weight_explanation_supported,
        "limited_aggregation_headroom_supported": limited_aggregation_headroom_supported,
        "oracle_is_non_deployable_upper_bound": True,
        "hard_gate_is_development_ablation": True,
        "test_sets_accessed": False,
    }])
    decision.to_csv(tables / "v3112_causal_decision.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = round_table["monitoring_round"]
    ax.plot(x, round_table["plain_attack_source_to_target_rate"], marker="o", label="Plain attack")
    ax.plot(x, round_table["soft_attack_source_to_target_rate"], marker="s", label="Soft trust")
    ax.plot(x, round_table["hard_attack_source_to_target_rate"], marker="^", label="Hard gate")
    ax.plot(x, round_table["oracle_attack_source_to_target_rate"], marker="D", label="Oracle drop")
    ax.plot(x, round_table["plain_clean_source_to_target_rate"], marker="x", label="Plain clean")
    ax.set_xlabel("Post-warmup monitoring round")
    ax.set_ylabel("DDoS to Benign rate")
    ax.set_ylim(0, 1)
    ax.set_title("V3.11.2 aggregation headroom")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, figures / "aggregation_headroom")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(method_table))
    ax.bar(x, method_table["attack_excess_removed_relative_to_clean"])
    ax.set_xticks(x, method_table["method"])
    ax.set_ylabel("Fraction of attack-induced excess removed")
    ax.set_title("V3.11.2 causal suppression headroom")
    ax.grid(axis="y", alpha=0.25)
    save(fig, figures / "attack_excess_removed")

    with (out / "v3112_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "experiment_version": "3.11.2",
            "purpose": "separate residual malicious weight from limited aggregation headroom",
            "development_seed": 42,
            "hard_gate_uses_frozen_detector": True,
            "oracle_drop_uses_ground_truth_malicious_labels": True,
            "oracle_drop_is_deployable": False,
            "count_cap_used": False,
            "test_sets_accessed": False,
        }, handle, indent=2)

    print("Aggregation Headroom Summary V3.11.2 complete")
    print(decision.to_string(index=False))
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

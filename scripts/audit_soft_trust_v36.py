#!/usr/bin/env python3
"""Offline signal and soft-trust audit for CIC IoT-DIAD V3.6 development.

This script uses completed V3.5.3 held-out results. It does not retrain models.

Goals
-----
1. Quantify which personalized signals actually separate malicious and benign clients.
2. Attribute false positives from the frozen hard-rejection rule.
3. Compare a small, predeclared family of clean-calibrated soft-trust policies.
4. Estimate malicious aggregation-influence reduction while preserving benign trust.

All thresholds are derived from the clean run of the same seed. Attack labels are
used only for development evaluation and exploratory ranking. The recommended policy
must still be integrated into federated aggregation and tested on new final seeds and
attack conditions before supporting a paper claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SIGNALS = {
    "update_l2": "personalized_abs_z_update_l2",
    "update_direction": "personalized_abs_z_update_direction_distance",
    "prediction_drift": "personalized_abs_z_max_offdiag_probability_drift",
    "explanation_drift": "personalized_abs_z_explanation_cosine_distance",
    "four_signal_fusion": "personalized_instantaneous_score",
}

POLICIES = [
    {
        "name": "prediction_soft_q85",
        "primary": "prediction_drift",
        "soft_quantile": 0.85,
        "ema_decay": 0.65,
        "gamma": 1.20,
        "minimum_trust": 0.15,
        "confirmation": None,
    },
    {
        "name": "prediction_soft_q90",
        "primary": "prediction_drift",
        "soft_quantile": 0.90,
        "ema_decay": 0.65,
        "gamma": 1.20,
        "minimum_trust": 0.15,
        "confirmation": None,
    },
    {
        "name": "prediction_q85_explanation_confirm",
        "primary": "prediction_drift",
        "soft_quantile": 0.85,
        "ema_decay": 0.65,
        "gamma": 1.20,
        "minimum_trust": 0.15,
        "confirmation": {
            "auxiliary": "explanation_drift",
            "primary_severe_quantile": 0.95,
            "auxiliary_quantile": 0.90,
            "multiplier": 0.35,
        },
    },
    {
        "name": "prediction_q85_direction_confirm",
        "primary": "prediction_drift",
        "soft_quantile": 0.85,
        "ema_decay": 0.65,
        "gamma": 1.20,
        "minimum_trust": 0.15,
        "confirmation": {
            "auxiliary": "update_direction",
            "primary_severe_quantile": 0.95,
            "auxiliary_quantile": 0.90,
            "multiplier": 0.35,
        },
    },
    {
        "name": "four_signal_fusion_soft_q85",
        "primary": "four_signal_fusion",
        "soft_quantile": 0.85,
        "ema_decay": 0.65,
        "gamma": 1.20,
        "minimum_trust": 0.15,
        "confirmation": None,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit personalized signals and clean-calibrated soft-trust policies."
    )
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", default="7,99,123,2026")
    return parser.parse_args()


def parse_seeds(text: str) -> List[int]:
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds are not allowed")
    return seeds


def quantile_higher(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    try:
        return float(np.quantile(array, quantile, method="higher"))
    except TypeError:
        return float(np.quantile(array, quantile, interpolation="higher"))


def add_client_ema(table: pd.DataFrame, source_column: str, output_column: str, decay: float) -> pd.DataFrame:
    result = table.copy().sort_values(["round", "client_id"]).reset_index(drop=True)
    memory: Dict[int, float] = {}
    values = []
    for _, row in result.iterrows():
        client_id = int(row["client_id"])
        current = float(row[source_column])
        previous = float(memory.get(client_id, 0.0))
        ema = float(decay) * previous + (1.0 - float(decay)) * current
        memory[client_id] = ema
        values.append(ema)
    result[output_column] = values
    return result


def safe_auc(labels: pd.Series, values: pd.Series) -> float:
    labels_array = labels.astype(int).to_numpy()
    if len(np.unique(labels_array)) < 2:
        return math.nan
    return float(roc_auc_score(labels_array, values.to_numpy(dtype=float)))


def signal_auc_rows(seed: int, attack: pd.DataFrame) -> List[Dict[str, float]]:
    rows = []
    for signal_name, column in SIGNALS.items():
        grouped = attack.groupby("client_id", as_index=False).agg(
            actual_malicious=("actual_malicious", "max"),
            client_mean=(column, "mean"),
            client_max=(column, "max"),
        )
        rows.append(
            {
                "seed": int(seed),
                "signal": signal_name,
                "row_roc_auc": safe_auc(attack["actual_malicious"], attack[column]),
                "client_mean_roc_auc": safe_auc(
                    grouped["actual_malicious"], grouped["client_mean"]
                ),
                "client_max_roc_auc": safe_auc(
                    grouped["actual_malicious"], grouped["client_max"]
                ),
            }
        )
    return rows


def false_positive_rows(seed: int, mode: str, table: pd.DataFrame) -> pd.DataFrame:
    rejected = table["frozen_rule_rejected"].astype(bool)
    malicious = table["actual_malicious"].astype(bool)
    false_positive = rejected if mode == "clean" else rejected & ~malicious
    selected = table.loc[false_positive].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "seed",
                "mode",
                "round",
                "client_id",
                "actual_malicious",
                "dominant_personalized_signal",
                "personalized_instantaneous_score",
                "frozen_rule_ema",
            ]
        )
    selected.insert(0, "seed", int(seed))
    selected.insert(1, "mode", mode)
    keep = [
        "seed",
        "mode",
        "round",
        "client_id",
        "actual_malicious",
        "dominant_personalized_signal",
        "personalized_instantaneous_score",
        "frozen_rule_ema",
    ]
    return selected[keep]


def apply_policy(
    clean: pd.DataFrame,
    attack: pd.DataFrame,
    policy: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    primary_name = str(policy["primary"])
    primary_column = SIGNALS[primary_name]
    decay = float(policy["ema_decay"])
    primary_ema_column = "policy_primary_ema"

    clean_scored = add_client_ema(clean, primary_column, primary_ema_column, decay)
    attack_scored = add_client_ema(attack, primary_column, primary_ema_column, decay)

    soft_threshold = quantile_higher(
        clean_scored[primary_ema_column], float(policy["soft_quantile"])
    )
    risk_excess = np.maximum(
        attack_scored[primary_ema_column].to_numpy(dtype=float)
        / max(soft_threshold, 1e-12)
        - 1.0,
        0.0,
    )
    clean_risk_excess = np.maximum(
        clean_scored[primary_ema_column].to_numpy(dtype=float)
        / max(soft_threshold, 1e-12)
        - 1.0,
        0.0,
    )

    gamma = float(policy["gamma"])
    attack_trust = np.exp(-gamma * risk_excess)
    clean_trust = np.exp(-gamma * clean_risk_excess)

    confirmation = policy.get("confirmation")
    severe_threshold = math.nan
    auxiliary_threshold = math.nan
    if confirmation:
        auxiliary_name = str(confirmation["auxiliary"])
        auxiliary_column = SIGNALS[auxiliary_name]
        auxiliary_ema_column = "policy_auxiliary_ema"
        clean_scored = add_client_ema(
            clean_scored, auxiliary_column, auxiliary_ema_column, decay
        )
        attack_scored = add_client_ema(
            attack_scored, auxiliary_column, auxiliary_ema_column, decay
        )
        severe_threshold = quantile_higher(
            clean_scored[primary_ema_column],
            float(confirmation["primary_severe_quantile"]),
        )
        auxiliary_threshold = quantile_higher(
            clean_scored[auxiliary_ema_column],
            float(confirmation["auxiliary_quantile"]),
        )
        multiplier = float(confirmation["multiplier"])
        attack_confirmed = (
            attack_scored[primary_ema_column].to_numpy(dtype=float) >= severe_threshold
        ) & (
            attack_scored[auxiliary_ema_column].to_numpy(dtype=float)
            >= auxiliary_threshold
        )
        clean_confirmed = (
            clean_scored[primary_ema_column].to_numpy(dtype=float) >= severe_threshold
        ) & (
            clean_scored[auxiliary_ema_column].to_numpy(dtype=float)
            >= auxiliary_threshold
        )
        attack_trust[attack_confirmed] *= multiplier
        clean_trust[clean_confirmed] *= multiplier

    minimum_trust = float(policy["minimum_trust"])
    attack_scored["policy_trust"] = np.clip(attack_trust, minimum_trust, 1.0)
    clean_scored["policy_trust"] = np.clip(clean_trust, minimum_trust, 1.0)

    round_rows = []
    for round_id, group in attack_scored.groupby("round", sort=True):
        counts = group["client_samples"].to_numpy(dtype=float)
        count_cap = float(np.median(counts) * 3.0)
        bounded_counts = np.minimum(counts, max(count_cap, 1.0))
        base_weights = bounded_counts / max(float(bounded_counts.sum()), 1e-12)
        adjusted = bounded_counts * group["policy_trust"].to_numpy(dtype=float)
        adjusted_weights = adjusted / max(float(adjusted.sum()), 1e-12)
        malicious = group["actual_malicious"].astype(bool).to_numpy()
        base_malicious_share = float(base_weights[malicious].sum())
        adjusted_malicious_share = float(adjusted_weights[malicious].sum())
        round_rows.append(
            {
                "round": int(round_id),
                "base_malicious_weight_share": base_malicious_share,
                "adjusted_malicious_weight_share": adjusted_malicious_share,
                "malicious_influence_reduction": float(
                    1.0
                    - adjusted_malicious_share
                    / max(base_malicious_share, 1e-12)
                ),
                "mean_malicious_trust": float(
                    group.loc[
                        group["actual_malicious"].astype(bool), "policy_trust"
                    ].mean()
                ),
                "mean_benign_trust": float(
                    group.loc[
                        ~group["actual_malicious"].astype(bool), "policy_trust"
                    ].mean()
                ),
            }
        )

    round_table = pd.DataFrame(round_rows)
    malicious_mask = attack_scored["actual_malicious"].astype(bool)
    benign_mask = ~malicious_mask
    metrics = {
        "soft_threshold": soft_threshold,
        "severe_threshold": severe_threshold,
        "auxiliary_threshold": auxiliary_threshold,
        "clean_mean_trust": float(clean_scored["policy_trust"].mean()),
        "clean_minimum_trust": float(clean_scored["policy_trust"].min()),
        "clean_low_trust_row_rate": float(
            (clean_scored["policy_trust"] < 0.50).mean()
        ),
        "attack_malicious_mean_trust": float(
            attack_scored.loc[malicious_mask, "policy_trust"].mean()
        ),
        "attack_benign_mean_trust": float(
            attack_scored.loc[benign_mask, "policy_trust"].mean()
        ),
        "attack_malicious_low_trust_row_rate": float(
            (attack_scored.loc[malicious_mask, "policy_trust"] < 0.50).mean()
        ),
        "attack_benign_low_trust_row_rate": float(
            (attack_scored.loc[benign_mask, "policy_trust"] < 0.50).mean()
        ),
        "mean_malicious_influence_reduction": float(
            round_table["malicious_influence_reduction"].mean()
        ),
        "final_malicious_influence_reduction": float(
            round_table.iloc[-1]["malicious_influence_reduction"]
        ),
        "mean_base_malicious_weight_share": float(
            round_table["base_malicious_weight_share"].mean()
        ),
        "mean_adjusted_malicious_weight_share": float(
            round_table["adjusted_malicious_weight_share"].mean()
        ),
    }
    return clean_scored, attack_scored, metrics


def aggregate_policy_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "clean_mean_trust",
        "clean_minimum_trust",
        "clean_low_trust_row_rate",
        "attack_malicious_mean_trust",
        "attack_benign_mean_trust",
        "attack_malicious_low_trust_row_rate",
        "attack_benign_low_trust_row_rate",
        "mean_malicious_influence_reduction",
        "final_malicious_influence_reduction",
        "mean_base_malicious_weight_share",
        "mean_adjusted_malicious_weight_share",
    ]
    rows = []
    for policy_name, group in per_seed.groupby("policy", sort=False):
        row = {"policy": policy_name, "n_seeds": int(len(group))}
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sample_std"] = float(values.std(ddof=1))
            row[f"{metric}_minimum"] = float(values.min())
            row[f"{metric}_maximum"] = float(values.max())
        rows.append(row)
    result = pd.DataFrame(rows)

    result["clean_feasible"] = (
        result["clean_mean_trust_mean"].ge(0.94)
        & result["clean_low_trust_row_rate_mean"].le(0.05)
        & result["attack_benign_mean_trust_mean"].ge(0.94)
    )
    result = result.sort_values(
        [
            "clean_feasible",
            "mean_malicious_influence_reduction_mean",
            "attack_benign_mean_trust_mean",
            "clean_mean_trust_mean",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    result.insert(0, "development_rank", np.arange(1, len(result) + 1))
    return result


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_signal_auc(signal_table: pd.DataFrame, base: Path) -> None:
    pivot = signal_table.pivot(
        index="signal", columns="seed", values="client_mean_roc_auc"
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Personalized signal")
    ax.set_ylabel("Client-mean ROC AUC")
    ax.set_title("Held-out signal separation by seed")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Seed")
    save_figure(fig, base)


def plot_policy_tradeoff(aggregate: pd.DataFrame, base: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        aggregate["attack_benign_mean_trust_mean"],
        aggregate["mean_malicious_influence_reduction_mean"],
        s=80,
    )
    for row in aggregate.itertuples():
        ax.annotate(
            row.policy,
            (
                row.attack_benign_mean_trust_mean,
                row.mean_malicious_influence_reduction_mean,
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Mean benign trust during attack")
    ax.set_ylabel("Mean malicious influence reduction")
    ax.set_title("Soft-trust policy tradeoff")
    ax.grid(alpha=0.25)
    save_figure(fig, base)


def plot_policy_trust(aggregate: pd.DataFrame, base: Path) -> None:
    ordered = aggregate.sort_values(
        "mean_malicious_influence_reduction_mean", ascending=False
    )
    x = np.arange(len(ordered))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width,
        ordered["clean_mean_trust_mean"],
        width,
        label="Clean mean trust",
    )
    ax.bar(
        x,
        ordered["attack_benign_mean_trust_mean"],
        width,
        label="Attack benign mean trust",
    )
    ax.bar(
        x + width,
        ordered["attack_malicious_mean_trust_mean"],
        width,
        label="Attack malicious mean trust",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["policy"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean trust")
    ax.set_title("Clean, benign, and malicious trust by policy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, base)


def plot_recommended_per_seed(per_seed: pd.DataFrame, recommended: str, base: Path) -> None:
    selected = per_seed[per_seed["policy"].eq(recommended)].sort_values("seed")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        [str(int(seed)) for seed in selected["seed"]],
        selected["mean_malicious_influence_reduction"],
    )
    ax.set_xlabel("Validation seed")
    ax.set_ylabel("Mean malicious influence reduction")
    ax.set_title(f"Per-seed influence reduction, {recommended}")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, base)


def main() -> int:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    decision_dir = output_dir / "policy_decisions"
    for directory in [tables_dir, figures_dir, decision_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    signal_rows: List[Dict[str, float]] = []
    policy_rows: List[Dict[str, float]] = []
    false_positive_tables = []

    for seed in seeds:
        seed_root = results_root / "seed_runs" / f"seed_{seed}"
        personalized_tables = seed_root / "personalized_scores" / "tables"
        validation_tables = seed_root / "frozen_rule_validation"

        clean_path = personalized_tables / "clean_leave_one_out_scores.csv"
        attack_path = personalized_tables / "attack_personalized_scores.csv"
        frozen_clean_path = validation_tables / "clean_frozen_rule_decisions.csv"
        frozen_attack_path = validation_tables / "attack_frozen_rule_decisions.csv"

        for path in [clean_path, attack_path, frozen_clean_path, frozen_attack_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required result file not found: {path}")

        clean = pd.read_csv(clean_path)
        attack = pd.read_csv(attack_path)
        frozen_clean = pd.read_csv(frozen_clean_path)
        frozen_attack = pd.read_csv(frozen_attack_path)

        signal_rows.extend(signal_auc_rows(seed, attack))
        false_positive_tables.append(
            false_positive_rows(seed, "clean", frozen_clean)
        )
        false_positive_tables.append(
            false_positive_rows(seed, "attack", frozen_attack)
        )

        for policy in POLICIES:
            clean_decisions, attack_decisions, metrics = apply_policy(
                clean, attack, policy
            )
            policy_name = str(policy["name"])
            policy_rows.append(
                {
                    "seed": int(seed),
                    "policy": policy_name,
                    "primary_signal": str(policy["primary"]),
                    "soft_quantile": float(policy["soft_quantile"]),
                    "ema_decay": float(policy["ema_decay"]),
                    "gamma": float(policy["gamma"]),
                    "minimum_trust": float(policy["minimum_trust"]),
                    "confirmation_signal": (
                        str(policy["confirmation"]["auxiliary"])
                        if policy.get("confirmation")
                        else ""
                    ),
                    **metrics,
                }
            )
            seed_policy_dir = decision_dir / f"seed_{seed}"
            seed_policy_dir.mkdir(parents=True, exist_ok=True)
            clean_decisions.to_csv(
                seed_policy_dir / f"{policy_name}_clean_scores_and_trust.csv",
                index=False,
            )
            attack_decisions.to_csv(
                seed_policy_dir / f"{policy_name}_attack_scores_and_trust.csv",
                index=False,
            )

    signal_table = pd.DataFrame(signal_rows)
    false_positive_table = pd.concat(false_positive_tables, ignore_index=True)
    policy_per_seed = pd.DataFrame(policy_rows)
    policy_aggregate = aggregate_policy_metrics(policy_per_seed)

    false_positive_summary = (
        false_positive_table.groupby(
            ["mode", "dominant_personalized_signal"], dropna=False
        )
        .size()
        .rename("false_positive_rows")
        .reset_index()
        .sort_values(["mode", "false_positive_rows"], ascending=[True, False])
    )

    signal_table.to_csv(tables_dir / "heldout_signal_auc_by_seed.csv", index=False)
    (
        signal_table.groupby("signal", as_index=False)
        .agg(
            row_roc_auc_mean=("row_roc_auc", "mean"),
            row_roc_auc_sample_std=("row_roc_auc", "std"),
            client_mean_roc_auc_mean=("client_mean_roc_auc", "mean"),
            client_mean_roc_auc_sample_std=("client_mean_roc_auc", "std"),
            client_max_roc_auc_mean=("client_max_roc_auc", "mean"),
            client_max_roc_auc_sample_std=("client_max_roc_auc", "std"),
        )
        .sort_values("client_mean_roc_auc_mean", ascending=False)
        .to_csv(tables_dir / "heldout_signal_auc_aggregate.csv", index=False)
    )
    false_positive_table.to_csv(
        tables_dir / "hard_rule_false_positive_rows.csv", index=False
    )
    false_positive_summary.to_csv(
        tables_dir / "hard_rule_false_positive_signal_summary.csv", index=False
    )
    policy_per_seed.to_csv(
        tables_dir / "soft_trust_policy_metrics_per_seed.csv", index=False
    )
    policy_aggregate.to_csv(
        tables_dir / "soft_trust_policy_metrics_aggregate.csv", index=False
    )

    feasible = policy_aggregate[policy_aggregate["clean_feasible"]]
    recommended_row = (
        feasible.iloc[0] if not feasible.empty else policy_aggregate.iloc[0]
    )
    recommended_policy = str(recommended_row["policy"])

    plot_signal_auc(signal_table, figures_dir / "heldout_signal_client_auc")
    plot_policy_tradeoff(
        policy_aggregate, figures_dir / "soft_trust_policy_tradeoff"
    )
    plot_policy_trust(
        policy_aggregate, figures_dir / "soft_trust_policy_trust_comparison"
    )
    plot_recommended_per_seed(
        policy_per_seed,
        recommended_policy,
        figures_dir / "recommended_policy_influence_reduction_by_seed",
    )

    metadata = {
        "experiment": "cic_iot_diad_soft_trust_audit_v36",
        "status": "development_audit_not_final_paper_result",
        "results_root": str(results_root),
        "seeds": seeds,
        "policies": POLICIES,
        "recommended_development_policy": recommended_policy,
        "recommendation_uses_attack_labels": True,
        "threshold_calibration_uses_attack_labels": False,
        "threshold_calibration_source": "clean_personalized_scores_only",
        "base_aggregation_for_influence_audit": "client_sample_count_capped_at_three_times_round_median",
        "hard_rejection_used": False,
        "final_paper_claim_allowed": False,
    }
    with (output_dir / "soft_trust_audit_v36_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print("CIC IoT-DIAD Soft Trust Audit V3.6 complete")
    print("Seeds:", seeds)
    print("Policies evaluated:", len(POLICIES))
    print("")
    print("Signal ranking by mean client-level ROC AUC")
    signal_aggregate = pd.read_csv(
        tables_dir / "heldout_signal_auc_aggregate.csv"
    )
    print(
        signal_aggregate[
            [
                "signal",
                "client_mean_roc_auc_mean",
                "client_max_roc_auc_mean",
                "row_roc_auc_mean",
            ]
        ].to_string(index=False)
    )
    print("")
    print("Exploratory recommended soft-trust policy:", recommended_policy)
    display = [
        "development_rank",
        "policy",
        "clean_feasible",
        "clean_mean_trust_mean",
        "clean_low_trust_row_rate_mean",
        "attack_benign_mean_trust_mean",
        "attack_malicious_mean_trust_mean",
        "mean_malicious_influence_reduction_mean",
        "final_malicious_influence_reduction_mean",
    ]
    print(policy_aggregate[display].to_string(index=False))
    print("")
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("This is an offline development audit, not a final defense result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit soft-trust mappings for the frozen V3.10.2 temporal policy.

Frozen signal and temporal decision:
- candidate_max_anchor_absolute
- EMA decay 0.65
- EMA reset at monitoring start
- clean EMA threshold quantile 0.99
- clean instantaneous threshold quantile 0.95
- policy evidence = max(EMA ratio, instantaneous ratio)

This audit does not retrain models or alter aggregation. It replays trust over the
completed true-chronology V3.10/V3.10.1 client-score captures. Seed 42 selects a
mapping under frozen benign-safety constraints. Seeds 7, 99, 123, and 2026 are
held out until the mapping is frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CANDIDATE = "candidate_max_anchor_absolute"
EMA_COLUMN = f"{CANDIDATE}_ema"
EMA_QUANTILE = 0.99
INSTANT_QUANTILE = 0.95
LOW_TRUST_CUTOFF = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-warmup-dir", required=True, type=Path)
    parser.add_argument("--development-clean-dir", required=True, type=Path)
    parser.add_argument("--development-attack-dir", required=True, type=Path)
    parser.add_argument("--heldout-root", required=True, type=Path)
    parser.add_argument("--heldout-seeds", default="7,99,123,2026")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--development-seed", type=int, default=42)
    parser.add_argument("--ema-decay", type=float, default=0.65)
    parser.add_argument("--min-clean-mean-trust", type=float, default=0.97)
    parser.add_argument("--max-clean-low-trust-rate", type=float, default=0.05)
    parser.add_argument("--min-attack-benign-mean-trust", type=float, default=0.95)
    parser.add_argument("--max-attack-benign-low-trust-rate", type=float, default=0.05)
    parser.add_argument("--min-final-benign-mean-trust", type=float, default=0.90)
    parser.add_argument("--max-final-benign-low-trust-rate", type=float, default=0.10)
    return parser.parse_args()


def quantile_higher(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    try:
        return float(np.quantile(array, q, method="higher"))
    except TypeError:
        return float(np.quantile(array, q, interpolation="higher"))


def load_seed(
    warmup_dir: Path,
    clean_dir: Path,
    attack_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    warmup_path = warmup_dir / "calibration" / "clean_leave_one_round_out_scores.csv"
    clean_path = clean_dir / "tables" / "continuation_client_anchor_scores.csv"
    attack_path = attack_dir / "tables" / "continuation_client_anchor_scores.csv"
    for path in (warmup_path, clean_path, attack_path):
        if not path.exists():
            raise FileNotFoundError(path)

    warmup = pd.read_csv(warmup_path)
    clean = pd.read_csv(clean_path)
    attack = pd.read_csv(attack_path)

    required_warmup = {"client_id", CANDIDATE, EMA_COLUMN}
    required_continuation = {
        "monitoring_round",
        "client_id",
        "actual_malicious",
        "client_samples",
        "aggregation_weight",
        "source_target_growth_from_profile",
        CANDIDATE,
        EMA_COLUMN,
    }
    missing = required_warmup.difference(warmup.columns)
    if missing:
        raise ValueError(f"Warmup table missing columns: {sorted(missing)}")
    for name, table in (("clean", clean), ("attack", attack)):
        missing = required_continuation.difference(table.columns)
        if missing:
            raise ValueError(f"{name} table missing columns: {sorted(missing)}")

    clean = clean.copy()
    clean["actual_malicious"] = False
    return warmup, clean, attack


def recompute_reset_ema(table: pd.DataFrame, ema_decay: float) -> pd.Series:
    output = pd.Series(index=table.index, dtype=float)
    ordered = table.sort_values(["client_id", "monitoring_round"])
    for _, group in ordered.groupby("client_id", sort=False):
        previous = 0.0
        for row_index in group.index:
            previous = float(ema_decay) * previous + (
                1.0 - float(ema_decay)
            ) * float(table.at[row_index, CANDIDATE])
            output.at[row_index] = previous
    return output


def add_frozen_policy_evidence(
    warmup: pd.DataFrame,
    continuation: pd.DataFrame,
    ema_decay: float,
) -> Tuple[pd.DataFrame, float, float]:
    ema_threshold = quantile_higher(warmup[EMA_COLUMN], EMA_QUANTILE)
    instant_threshold = quantile_higher(warmup[CANDIDATE], INSTANT_QUANTILE)

    result = continuation.copy()
    result["policy_ema"] = recompute_reset_ema(result, ema_decay)
    result["ema_threshold"] = float(ema_threshold)
    result["instant_threshold"] = float(instant_threshold)
    result["ema_ratio"] = (
        result["policy_ema"].to_numpy(dtype=np.float64)
        / max(float(ema_threshold), 1e-12)
    )
    result["instant_ratio"] = (
        result[CANDIDATE].to_numpy(dtype=np.float64)
        / max(float(instant_threshold), 1e-12)
    )
    result["policy_ratio"] = np.maximum(
        result["ema_ratio"].to_numpy(dtype=np.float64),
        result["instant_ratio"].to_numpy(dtype=np.float64),
    )
    result["policy_exceedance"] = np.maximum(
        result["policy_ratio"].to_numpy(dtype=np.float64) - 1.0,
        0.0,
    )
    result["frozen_policy_flagged"] = result["policy_ratio"] > 1.0
    return result, float(ema_threshold), float(instant_threshold)


def apply_trust(
    table: pd.DataFrame,
    gamma: float,
    minimum_trust: float,
) -> pd.DataFrame:
    result = table.copy()
    trust = np.exp(
        -float(gamma) * result["policy_exceedance"].to_numpy(dtype=np.float64)
    )
    result["trust_factor"] = np.clip(
        trust,
        float(minimum_trust),
        1.0,
    )

    adjusted_parts = []
    for monitoring_round, group in result.groupby("monitoring_round", sort=True):
        current = group.copy()
        raw_weight = current["aggregation_weight"].to_numpy(dtype=np.float64)
        if not np.isclose(raw_weight.sum(), 1.0, atol=1e-8):
            raw_weight = raw_weight / max(raw_weight.sum(), 1e-12)
        unnormalized = raw_weight * current["trust_factor"].to_numpy(dtype=np.float64)
        adjusted_weight = unnormalized / max(unnormalized.sum(), 1e-12)
        current["raw_aggregation_weight"] = raw_weight
        current["trust_adjusted_weight"] = adjusted_weight
        current["raw_harm_weight"] = (
            raw_weight
            * current["source_target_growth_from_profile"].to_numpy(dtype=np.float64)
        )
        current["trust_adjusted_harm_weight"] = (
            adjusted_weight
            * current["source_target_growth_from_profile"].to_numpy(dtype=np.float64)
        )
        adjusted_parts.append(current)
    return pd.concat(adjusted_parts, ignore_index=True)


def safe_reduction(after: float, before: float) -> float:
    if before <= 1e-12:
        return 0.0
    return float(1.0 - after / before)


def summarize_condition(
    table: pd.DataFrame,
    condition: str,
) -> Dict[str, float]:
    labels = table["actual_malicious"].astype(bool)
    benign = ~labels
    final_round = int(table["monitoring_round"].max())
    final = table["monitoring_round"].eq(final_round)

    summary: Dict[str, float] = {
        f"{condition}_mean_trust": float(table["trust_factor"].mean()),
        f"{condition}_low_trust_rate": float(
            np.mean(table["trust_factor"].to_numpy(dtype=np.float64) < LOW_TRUST_CUTOFF)
        ),
    }

    if benign.any():
        summary[f"{condition}_benign_mean_trust"] = float(
            table.loc[benign, "trust_factor"].mean()
        )
        summary[f"{condition}_benign_low_trust_rate"] = float(
            np.mean(
                table.loc[benign, "trust_factor"].to_numpy(dtype=np.float64)
                < LOW_TRUST_CUTOFF
            )
        )
        summary[f"{condition}_final_benign_mean_trust"] = float(
            table.loc[benign & final, "trust_factor"].mean()
        )
        summary[f"{condition}_final_benign_low_trust_rate"] = float(
            np.mean(
                table.loc[benign & final, "trust_factor"].to_numpy(dtype=np.float64)
                < LOW_TRUST_CUTOFF
            )
        )

    if labels.any():
        summary[f"{condition}_malicious_mean_trust"] = float(
            table.loc[labels, "trust_factor"].mean()
        )
        summary[f"{condition}_malicious_low_trust_recall"] = float(
            np.mean(
                table.loc[labels, "trust_factor"].to_numpy(dtype=np.float64)
                < LOW_TRUST_CUTOFF
            )
        )
        summary[f"{condition}_first_malicious_mean_trust"] = float(
            table.loc[
                labels & table["monitoring_round"].eq(1),
                "trust_factor",
            ].mean()
        )
        summary[f"{condition}_final_malicious_mean_trust"] = float(
            table.loc[labels & final, "trust_factor"].mean()
        )

        round_metrics = []
        for monitoring_round, group in table.groupby("monitoring_round", sort=True):
            round_labels = group["actual_malicious"].astype(bool)
            raw_malicious_share = float(
                group.loc[round_labels, "raw_aggregation_weight"].sum()
            )
            adjusted_malicious_share = float(
                group.loc[round_labels, "trust_adjusted_weight"].sum()
            )
            raw_malicious_harm = float(
                group.loc[round_labels, "raw_harm_weight"].sum()
            )
            adjusted_malicious_harm = float(
                group.loc[round_labels, "trust_adjusted_harm_weight"].sum()
            )
            round_metrics.append(
                {
                    "monitoring_round": int(monitoring_round),
                    "malicious_influence_reduction": safe_reduction(
                        adjusted_malicious_share,
                        raw_malicious_share,
                    ),
                    "malicious_harm_reduction": safe_reduction(
                        adjusted_malicious_harm,
                        raw_malicious_harm,
                    ),
                }
            )
        round_table = pd.DataFrame(round_metrics)
        summary[f"{condition}_mean_malicious_influence_reduction"] = float(
            round_table["malicious_influence_reduction"].mean()
        )
        summary[f"{condition}_minimum_malicious_influence_reduction"] = float(
            round_table["malicious_influence_reduction"].min()
        )
        summary[f"{condition}_final_malicious_influence_reduction"] = float(
            round_table.iloc[-1]["malicious_influence_reduction"]
        )
        summary[f"{condition}_mean_malicious_harm_reduction"] = float(
            round_table["malicious_harm_reduction"].mean()
        )
        summary[f"{condition}_minimum_malicious_harm_reduction"] = float(
            round_table["malicious_harm_reduction"].min()
        )
        summary[f"{condition}_final_malicious_harm_reduction"] = float(
            round_table.iloc[-1]["malicious_harm_reduction"]
        )

    return summary


def evaluate_mapping(
    warmup: pd.DataFrame,
    clean: pd.DataFrame,
    attack: pd.DataFrame,
    ema_decay: float,
    gamma: float,
    minimum_trust: float,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    clean_evidence, ema_threshold, instant_threshold = add_frozen_policy_evidence(
        warmup, clean, ema_decay
    )
    attack_evidence, _, _ = add_frozen_policy_evidence(
        warmup, attack, ema_decay
    )
    clean_trust = apply_trust(clean_evidence, gamma, minimum_trust)
    attack_trust = apply_trust(attack_evidence, gamma, minimum_trust)

    row: Dict[str, object] = {
        "gamma": float(gamma),
        "minimum_trust": float(minimum_trust),
        "ema_quantile": EMA_QUANTILE,
        "instant_quantile": INSTANT_QUANTILE,
        "ema_threshold": ema_threshold,
        "instant_threshold": instant_threshold,
        "low_trust_cutoff": LOW_TRUST_CUTOFF,
        **summarize_condition(clean_trust, "clean"),
        **summarize_condition(attack_trust, "attack"),
    }
    return row, clean_trust, attack_trust


def mapping_grid() -> List[Tuple[float, float]]:
    return [
        (gamma, minimum_trust)
        for gamma in (0.50, 0.75, 1.00, 1.50, 2.00, 3.00)
        for minimum_trust in (0.05, 0.10, 0.15, 0.25)
    ]


def add_safety(
    table: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    result = table.copy()
    result["passes_clean_mean_trust"] = (
        result["clean_mean_trust"] >= args.min_clean_mean_trust
    )
    result["passes_clean_low_trust_rate"] = (
        result["clean_low_trust_rate"] <= args.max_clean_low_trust_rate
    )
    result["passes_attack_benign_mean_trust"] = (
        result["attack_benign_mean_trust"]
        >= args.min_attack_benign_mean_trust
    )
    result["passes_attack_benign_low_trust_rate"] = (
        result["attack_benign_low_trust_rate"]
        <= args.max_attack_benign_low_trust_rate
    )
    result["passes_final_benign_mean_trust"] = (
        result["attack_final_benign_mean_trust"]
        >= args.min_final_benign_mean_trust
    )
    result["passes_final_benign_low_trust_rate"] = (
        result["attack_final_benign_low_trust_rate"]
        <= args.max_final_benign_low_trust_rate
    )
    result["passes_all_safety_constraints"] = result[
        [
            "passes_clean_mean_trust",
            "passes_clean_low_trust_rate",
            "passes_attack_benign_mean_trust",
            "passes_attack_benign_low_trust_rate",
            "passes_final_benign_mean_trust",
            "passes_final_benign_low_trust_rate",
        ]
    ].all(axis=1)
    return result


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dev_warmup, dev_clean, dev_attack = load_seed(
        args.development_warmup_dir,
        args.development_clean_dir,
        args.development_attack_dir,
    )

    development_rows = []
    development_cache: Dict[Tuple[float, float], Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for gamma, minimum_trust in mapping_grid():
        row, clean_trust, attack_trust = evaluate_mapping(
            dev_warmup,
            dev_clean,
            dev_attack,
            args.ema_decay,
            gamma,
            minimum_trust,
        )
        development_rows.append(row)
        development_cache[(gamma, minimum_trust)] = (clean_trust, attack_trust)

    development = add_safety(pd.DataFrame(development_rows), args)
    eligible = development[development["passes_all_safety_constraints"]].copy()
    if eligible.empty:
        development.to_csv(
            tables_dir / "development_mapping_ranking.csv",
            index=False,
        )
        raise RuntimeError(
            "No soft-trust mapping satisfies the frozen benign-safety constraints"
        )

    eligible = eligible.sort_values(
        [
            "attack_mean_malicious_harm_reduction",
            "attack_minimum_malicious_harm_reduction",
            "attack_mean_malicious_influence_reduction",
            "attack_first_malicious_mean_trust",
            "attack_malicious_mean_trust",
            "attack_benign_mean_trust",
        ],
        ascending=[False, False, False, True, True, False],
    )
    selected = eligible.iloc[0]
    development = development.sort_values(
        [
            "passes_all_safety_constraints",
            "attack_mean_malicious_harm_reduction",
            "attack_minimum_malicious_harm_reduction",
            "attack_mean_malicious_influence_reduction",
        ],
        ascending=[False, False, False, False],
    )
    development.to_csv(
        tables_dir / "development_mapping_ranking.csv",
        index=False,
    )
    eligible.to_csv(
        tables_dir / "development_safety_eligible_mappings.csv",
        index=False,
    )

    selected_key = (
        float(selected["gamma"]),
        float(selected["minimum_trust"]),
    )
    selected_dev_clean, selected_dev_attack = development_cache[selected_key]
    selected_dev_clean.to_csv(
        tables_dir / "selected_mapping_development_clean_client_rows.csv",
        index=False,
    )
    selected_dev_attack.to_csv(
        tables_dir / "selected_mapping_development_attack_client_rows.csv",
        index=False,
    )

    heldout_rows = []
    heldout_client_tables = []
    heldout_seeds = [
        int(value.strip())
        for value in args.heldout_seeds.split(",")
        if value.strip()
    ]
    for seed in heldout_seeds:
        seed_root = args.heldout_root / f"seed_{seed}"
        warmup, clean, attack = load_seed(
            seed_root / "warmup",
            seed_root / "clean_continuation",
            seed_root / "attack_continuation",
        )
        row, clean_trust, attack_trust = evaluate_mapping(
            warmup,
            clean,
            attack,
            args.ema_decay,
            selected_key[0],
            selected_key[1],
        )
        row["seed"] = int(seed)
        heldout_rows.append(row)

        clean_trust.insert(0, "seed", seed)
        clean_trust.insert(1, "condition", "clean")
        attack_trust.insert(0, "seed", seed)
        attack_trust.insert(1, "condition", "strong_attack")
        heldout_client_tables.extend([clean_trust, attack_trust])

    heldout = pd.DataFrame(heldout_rows).sort_values("seed")
    heldout.to_csv(
        tables_dir / "selected_mapping_heldout_metrics.csv",
        index=False,
    )
    pd.concat(heldout_client_tables, ignore_index=True).to_csv(
        tables_dir / "selected_mapping_heldout_client_rows.csv",
        index=False,
    )

    aggregate = pd.DataFrame(
        [
            {
                "selected_gamma": selected_key[0],
                "selected_minimum_trust": selected_key[1],
                "frozen_ema_quantile": EMA_QUANTILE,
                "frozen_instant_quantile": INSTANT_QUANTILE,
                "development_seed": int(args.development_seed),
                "heldout_seed_count": int(len(heldout)),
                "heldout_mean_clean_mean_trust": float(
                    heldout["clean_mean_trust"].mean()
                ),
                "heldout_minimum_clean_mean_trust": float(
                    heldout["clean_mean_trust"].min()
                ),
                "heldout_mean_attack_benign_mean_trust": float(
                    heldout["attack_benign_mean_trust"].mean()
                ),
                "heldout_minimum_attack_benign_mean_trust": float(
                    heldout["attack_benign_mean_trust"].min()
                ),
                "heldout_mean_attack_malicious_mean_trust": float(
                    heldout["attack_malicious_mean_trust"].mean()
                ),
                "heldout_mean_first_malicious_mean_trust": float(
                    heldout["attack_first_malicious_mean_trust"].mean()
                ),
                "heldout_mean_malicious_influence_reduction": float(
                    heldout[
                        "attack_mean_malicious_influence_reduction"
                    ].mean()
                ),
                "heldout_minimum_malicious_influence_reduction": float(
                    heldout[
                        "attack_minimum_malicious_influence_reduction"
                    ].min()
                ),
                "heldout_mean_malicious_harm_reduction": float(
                    heldout["attack_mean_malicious_harm_reduction"].mean()
                ),
                "heldout_minimum_malicious_harm_reduction": float(
                    heldout[
                        "attack_minimum_malicious_harm_reduction"
                    ].min()
                ),
                "selection_frozen_before_heldout_review": True,
                "temporal_policy_reopened": False,
                "candidate_selection_reopened": False,
                "count_cap_used": False,
                "defense_weights_applied": False,
                "test_sets_accessed": False,
            }
        ]
    )
    aggregate.to_csv(
        tables_dir / "selected_mapping_aggregate_summary.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(heldout))
    width = 0.36
    ax.bar(
        positions - width / 2,
        heldout["attack_benign_mean_trust"],
        width,
        label="Benign mean trust",
    )
    ax.bar(
        positions + width / 2,
        heldout["attack_malicious_mean_trust"],
        width,
        label="Malicious mean trust",
    )
    ax.set_xticks(positions, heldout["seed"].astype(str))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Held-out development seed")
    ax.set_ylabel("Mean trust")
    ax.set_title("V3.10.3 frozen soft-trust mapping")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "heldout_benign_vs_malicious_trust")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        heldout["seed"].astype(str),
        heldout["attack_mean_malicious_harm_reduction"],
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Held-out development seed")
    ax.set_ylabel("Mean malicious harm reduction")
    ax.set_title("Offline harm-weight reduction under frozen trust")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "heldout_malicious_harm_reduction")

    metadata = {
        "experiment_version": "3.10.3",
        "experiment_type": "frozen_soft_trust_mapping_audit",
        "frozen_candidate": CANDIDATE,
        "frozen_temporal_policy": {
            "ema_initialization": "reset",
            "ema_quantile": EMA_QUANTILE,
            "instant_quantile": INSTANT_QUANTILE,
            "policy_ratio": "max(ema/ema_threshold, instant/instant_threshold)",
        },
        "selected_gamma": selected_key[0],
        "selected_minimum_trust": selected_key[1],
        "development_seed": int(args.development_seed),
        "heldout_seeds": heldout_seeds,
        "selection_frozen_before_heldout_review": True,
        "candidate_selection_reopened": False,
        "temporal_policy_reopened": False,
        "count_cap_used": False,
        "defense_weights_applied": False,
        "test_sets_accessed": False,
        "malicious_labels_used_for_aggregation": False,
        "malicious_labels_used_for_development_mapping_selection": True,
    }
    with (
        output_dir / "frozen_soft_trust_mapping_audit_v3103_metadata.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Frozen Soft-Trust Mapping Audit V3.10.3 complete")
    print("Selected gamma:", f"{selected_key[0]:.2f}")
    print("Selected minimum trust:", f"{selected_key[1]:.2f}")
    print("Frozen EMA quantile:", EMA_QUANTILE)
    print("Frozen instant quantile:", INSTANT_QUANTILE)
    print("Count cap used: False")
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

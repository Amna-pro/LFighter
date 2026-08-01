#!/usr/bin/env python3
"""Analyze V3.7 transition signatures with safety-constrained development selection.

The analysis evaluates fixed, interpretable candidate scores. Seed 42 is treated
as development-only when present. Other seeds are reported as held-out
development validation. Malicious labels are used only for diagnostic metrics,
never for calibration thresholds or aggregation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURES = [
    "source_target_growth",
    "source_diagonal_confidence_loss",
    "max_offdiag_growth",
    "mean_diagonal_confidence_loss",
    "max_diagonal_confidence_loss",
    "frobenius_drift",
    "mean_row_js_reference",
    "max_row_js_reference",
    "frobenius_to_consensus",
    "mean_row_js_consensus",
    "max_row_js_consensus",
    "source_target_consensus_residual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze V3.7 full transition signatures."
    )
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", default="42,7,99,123,2026")
    parser.add_argument("--development-seed", type=int, default=42)
    parser.add_argument("--ema-decay", type=float, default=0.65)
    parser.add_argument("--clean-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--max-development-clean-fpr", type=float, default=0.05)
    parser.add_argument("--max-development-attack-benign-fpr", type=float, default=0.05)
    parser.add_argument("--max-development-final-benign-fpr", type=float, default=0.10)
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def robust_center_scale(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    center = float(np.median(array))
    mad = float(np.median(np.abs(array - center)))
    scale = 1.4826 * mad
    if scale < 1e-9:
        std = float(np.std(array))
        scale = std if std >= 1e-9 else 1.0
    return center, scale


def positive_z(values: np.ndarray, center: float, scale: float) -> np.ndarray:
    return np.clip(np.maximum((values - center) / max(scale, 1e-12), 0.0), 0.0, 20.0)


def quantile_higher(values: Sequence[float], q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    try:
        return float(np.quantile(array, q, method="higher"))
    except TypeError:
        return float(np.quantile(array, q, interpolation="higher"))


def add_temporal_ema(
    table: pd.DataFrame, score_columns: Iterable[str], decay: float
) -> pd.DataFrame:
    result = table.sort_values(["client_id", "round"]).copy()
    for score in score_columns:
        output = np.zeros(len(result), dtype=np.float64)
        for _, indices in result.groupby("client_id", sort=False).groups.items():
            previous = 0.0
            for index in indices:
                previous = float(decay) * previous + (1.0 - float(decay)) * float(
                    result.at[index, score]
                )
                output[result.index.get_loc(index)] = previous
        result[f"{score}_ema"] = output
    return result.sort_values(["round", "client_id"]).reset_index(drop=True)


def historical_scores(
    clean: pd.DataFrame, attack: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_scored = clean.copy()
    attack_scored = attack.copy()
    profile_rows: List[Dict[str, object]] = []

    for client_id in sorted(clean["client_id"].unique()):
        clean_client = clean[clean["client_id"].eq(client_id)].sort_values("round")
        attack_mask = attack_scored["client_id"].eq(client_id)
        clean_mask = clean_scored["client_id"].eq(client_id)
        for feature in FEATURES:
            values = clean_client[feature].to_numpy(dtype=float)
            center, scale = robust_center_scale(values)
            profile_rows.append(
                {
                    "client_id": int(client_id),
                    "feature": feature,
                    "clean_median": center,
                    "clean_scale": scale,
                    "clean_rows": int(len(values)),
                }
            )
            attack_scored.loc[attack_mask, f"hist_z_{feature}"] = positive_z(
                attack_scored.loc[attack_mask, feature].to_numpy(dtype=float),
                center,
                scale,
            )

            clean_indices = clean_scored.index[clean_mask].tolist()
            for row_index in clean_indices:
                other = clean_scored.loc[
                    clean_mask & clean_scored.index.to_series().ne(row_index),
                    feature,
                ].to_numpy(dtype=float)
                loo_center, loo_scale = robust_center_scale(other)
                clean_scored.at[row_index, f"hist_z_{feature}"] = float(
                    positive_z(
                        np.asarray([clean_scored.at[row_index, feature]], dtype=float),
                        loo_center,
                        loo_scale,
                    )[0]
                )

    return clean_scored, attack_scored, pd.DataFrame(profile_rows)


def add_round_relative_scores(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    for feature in FEATURES:
        column = f"round_z_{feature}"
        result[column] = 0.0
        for _, group in result.groupby("round"):
            center, scale = robust_center_scale(group[feature].to_numpy(dtype=float))
            result.loc[group.index, column] = positive_z(
                group[feature].to_numpy(dtype=float),
                center,
                scale,
            )
    return result


def add_coalition_support(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["generic_pair_coalition_support"] = 0.0
    for _, group in result.groupby("round"):
        pair_counts = group["max_offdiag_pair"].value_counts()
        denominator = max(len(group) - 1, 1)
        support = [
            max(int(pair_counts[pair]) - 1, 0) / denominator
            for pair in group["max_offdiag_pair"]
        ]
        result.loc[group.index, "generic_pair_coalition_support"] = support
    return result


def add_candidate_scores(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    historical_transition = result[
        [
            "hist_z_source_target_growth",
            "hist_z_source_diagonal_confidence_loss",
            "hist_z_mean_row_js_reference",
            "hist_z_frobenius_drift",
        ]
    ].mean(axis=1)
    historical_generic = result[
        [
            "hist_z_max_offdiag_growth",
            "hist_z_mean_diagonal_confidence_loss",
            "hist_z_mean_row_js_reference",
            "hist_z_frobenius_drift",
        ]
    ].mean(axis=1)
    round_relative = result[
        [
            "round_z_source_target_consensus_residual",
            "round_z_mean_row_js_consensus",
            "round_z_frobenius_to_consensus",
        ]
    ].mean(axis=1)
    round_generic = result[
        [
            "round_z_max_offdiag_growth",
            "round_z_mean_row_js_consensus",
            "round_z_frobenius_to_consensus",
        ]
    ].mean(axis=1)

    result["candidate_source_target_historical"] = result[
        "hist_z_source_target_growth"
    ]
    result["candidate_historical_transition"] = historical_transition
    result["candidate_dual_reference_transition"] = (
        0.60 * historical_transition + 0.40 * round_relative
    )
    result["candidate_generic_dual_reference"] = (
        0.60 * historical_generic + 0.40 * round_generic
    )
    result["candidate_generic_dual_coalition"] = (
        0.55 * historical_generic
        + 0.30 * round_generic
        + 0.15 * result["generic_pair_coalition_support"] * 5.0
    )
    return result


def diagnostic_metrics(
    seed: int,
    candidate: str,
    clean: pd.DataFrame,
    attack: pd.DataFrame,
    threshold_quantile: float,
) -> Dict[str, object]:
    score_col = f"{candidate}_ema"
    threshold = quantile_higher(clean[score_col], threshold_quantile)
    labels = attack["actual_malicious"].astype(bool).to_numpy()
    scores = attack[score_col].to_numpy(dtype=float)
    predicted = scores > threshold
    benign = ~labels
    final = attack["round"].eq(attack["round"].max()).to_numpy()

    roc_auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else math.nan
    pr_auc = float(average_precision_score(labels, scores)) if labels.any() else math.nan
    tp = int(np.sum(labels & predicted))
    fp = int(np.sum(benign & predicted))
    fn = int(np.sum(labels & ~predicted))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        "seed": int(seed),
        "candidate": candidate,
        "clean_threshold_quantile": float(threshold_quantile),
        "clean_threshold": float(threshold),
        "clean_false_positive_rate": float(
            np.mean(clean[score_col].to_numpy(dtype=float) > threshold)
        ),
        "attack_row_roc_auc": roc_auc,
        "attack_row_pr_auc": pr_auc,
        "attack_row_malicious_recall": float(recall),
        "attack_row_precision": float(precision),
        "attack_row_benign_false_positive_rate": float(
            np.mean(predicted[benign])
        ),
        "final_round_malicious_recall": float(
            np.mean(predicted[labels & final])
        ),
        "final_round_benign_false_positive_rate": float(
            np.mean(predicted[benign & final])
        ),
        "attack_rows": int(len(attack)),
        "malicious_rows": int(labels.sum()),
    }


def load_seed(runs_root: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_path = (
        runs_root
        / "clean"
        / f"seed_{seed}"
        / "tables"
        / "client_signature_summary.csv"
    )
    attack_path = (
        runs_root
        / "strong_attack"
        / f"seed_{seed}"
        / "tables"
        / "client_signature_summary.csv"
    )
    if not clean_path.exists() or not attack_path.exists():
        raise FileNotFoundError(
            f"Missing clean or attack V3.7 table for seed {seed}"
        )
    return pd.read_csv(clean_path), pd.read_csv(attack_path)


def plot_candidate_auc(table: pd.DataFrame, out: Path) -> None:
    summary = (
        table.groupby("candidate", as_index=False)["attack_row_pr_auc"]
        .mean()
        .sort_values("attack_row_pr_auc")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(summary["candidate"], summary["attack_row_pr_auc"])
    ax.set_xlabel("Mean attack-row PR-AUC")
    ax.set_title("V3.7 fixed transition-signature candidates")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, out)


def plot_selected_distribution(
    selected: str,
    scored: pd.DataFrame,
    out: Path,
) -> None:
    column = f"{selected}_ema"
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group in scored.groupby("actual_malicious"):
        name = "Malicious attack rows" if bool(label) else "Benign attack rows"
        ax.hist(group[column], bins=20, alpha=0.55, label=name)
    ax.set_xlabel("Temporal candidate score")
    ax.set_ylabel("Rows")
    ax.set_title(f"Selected V3.7 candidate: {selected}")
    ax.legend()
    save_figure(fig, out)


def main() -> int:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, object]] = []
    all_profiles: List[pd.DataFrame] = []
    all_scored_attack: List[pd.DataFrame] = []
    candidate_columns: List[str] = []

    for seed in seeds:
        clean, attack = load_seed(args.runs_root.expanduser().resolve(), seed)
        clean_scored, attack_scored, profiles = historical_scores(clean, attack)
        clean_scored = add_round_relative_scores(clean_scored)
        attack_scored = add_round_relative_scores(attack_scored)
        clean_scored = add_coalition_support(clean_scored)
        attack_scored = add_coalition_support(attack_scored)
        clean_scored = add_candidate_scores(clean_scored)
        attack_scored = add_candidate_scores(attack_scored)
        candidate_columns = [
            column
            for column in clean_scored.columns
            if column.startswith("candidate_")
        ]
        clean_scored = add_temporal_ema(
            clean_scored, candidate_columns, args.ema_decay
        )
        attack_scored = add_temporal_ema(
            attack_scored, candidate_columns, args.ema_decay
        )
        clean_scored.insert(0, "seed", seed)
        attack_scored.insert(0, "seed", seed)
        profiles.insert(0, "seed", seed)
        all_profiles.append(profiles)
        all_scored_attack.append(attack_scored)

        clean_scored.to_csv(
            tables_dir / f"seed_{seed}_clean_scored_rows.csv", index=False
        )
        attack_scored.to_csv(
            tables_dir / f"seed_{seed}_attack_scored_rows.csv", index=False
        )
        for candidate in candidate_columns:
            all_metrics.append(
                diagnostic_metrics(
                    seed=seed,
                    candidate=candidate,
                    clean=clean_scored,
                    attack=attack_scored,
                    threshold_quantile=args.clean_threshold_quantile,
                )
            )

    metrics = pd.DataFrame(all_metrics)
    profiles = pd.concat(all_profiles, ignore_index=True)
    scored_attack = pd.concat(all_scored_attack, ignore_index=True)
    metrics.to_csv(tables_dir / "candidate_metrics_all_seeds.csv", index=False)
    profiles.to_csv(tables_dir / "personalized_clean_feature_profiles.csv", index=False)
    scored_attack.to_csv(tables_dir / "attack_scored_rows_all_seeds.csv", index=False)

    development = metrics[metrics["seed"].eq(args.development_seed)].copy()
    if development.empty:
        raise ValueError("Development seed not found in completed runs")

    development["passes_clean_fpr_constraint"] = (
        development["clean_false_positive_rate"]
        <= float(args.max_development_clean_fpr)
    )
    development["passes_attack_benign_fpr_constraint"] = (
        development["attack_row_benign_false_positive_rate"]
        <= float(args.max_development_attack_benign_fpr)
    )
    development["passes_final_benign_fpr_constraint"] = (
        development["final_round_benign_false_positive_rate"]
        <= float(args.max_development_final_benign_fpr)
    )
    development["passes_all_safety_constraints"] = (
        development["passes_clean_fpr_constraint"]
        & development["passes_attack_benign_fpr_constraint"]
        & development["passes_final_benign_fpr_constraint"]
    )

    eligible = development[
        development["passes_all_safety_constraints"]
    ].copy()
    if eligible.empty:
        raise RuntimeError(
            "No candidate satisfies the frozen development safety constraints. "
            "Do not select from held-out seeds. Revise the candidate family using "
            "development evidence only."
        )

    eligible = eligible.sort_values(
        [
            "attack_row_pr_auc",
            "attack_row_roc_auc",
            "attack_row_malicious_recall",
            "attack_row_benign_false_positive_rate",
        ],
        ascending=[False, False, False, True],
    )
    selected = str(eligible.iloc[0]["candidate"])

    development = development.sort_values(
        [
            "passes_all_safety_constraints",
            "attack_row_pr_auc",
            "attack_row_roc_auc",
            "attack_row_benign_false_positive_rate",
        ],
        ascending=[False, False, False, True],
    )
    development.to_csv(
        tables_dir / "development_candidate_ranking.csv", index=False
    )
    eligible.to_csv(
        tables_dir / "development_safety_eligible_candidates.csv", index=False
    )

    heldout = metrics[
        metrics["candidate"].eq(selected)
        & ~metrics["seed"].eq(args.development_seed)
    ].copy()
    heldout.to_csv(
        tables_dir / "selected_candidate_heldout_metrics.csv", index=False
    )
    aggregate = pd.DataFrame(
        [
            {
                "selected_candidate": selected,
        "selection_frozen_before_heldout_review": True,
        "development_safety_constraints": {
            "max_clean_false_positive_rate": float(args.max_development_clean_fpr),
            "max_attack_benign_false_positive_rate": float(args.max_development_attack_benign_fpr),
            "max_final_benign_false_positive_rate": float(args.max_development_final_benign_fpr),
        },
                "development_seed": int(args.development_seed),
                "heldout_seed_count": int(len(heldout)),
                "heldout_mean_roc_auc": float(heldout["attack_row_roc_auc"].mean())
                if len(heldout)
                else math.nan,
                "heldout_mean_pr_auc": float(heldout["attack_row_pr_auc"].mean())
                if len(heldout)
                else math.nan,
                "heldout_mean_malicious_recall": float(
                    heldout["attack_row_malicious_recall"].mean()
                )
                if len(heldout)
                else math.nan,
                "heldout_mean_benign_false_positive_rate": float(
                    heldout["attack_row_benign_false_positive_rate"].mean()
                )
                if len(heldout)
                else math.nan,
                "heldout_mean_final_malicious_recall": float(
                    heldout["final_round_malicious_recall"].mean()
                )
                if len(heldout)
                else math.nan,
                "heldout_mean_final_benign_false_positive_rate": float(
                    heldout["final_round_benign_false_positive_rate"].mean()
                )
                if len(heldout)
                else math.nan,
                "selection_rule": "highest development-seed PR-AUC among candidates satisfying frozen safety constraints",
                "threshold_rule": "per-seed clean-only temporal-score quantile",
                "development_max_clean_fpr": float(args.max_development_clean_fpr),
                "development_max_attack_benign_fpr": float(args.max_development_attack_benign_fpr),
                "development_max_final_benign_fpr": float(args.max_development_final_benign_fpr),
                "malicious_labels_used_for_weights": False,
                "test_sets_accessed": False,
            }
        ]
    )
    aggregate.to_csv(
        tables_dir / "selected_candidate_aggregate_summary.csv", index=False
    )

    plot_candidate_auc(metrics, figures_dir / "candidate_pr_auc_across_seeds")
    plot_selected_distribution(
        selected,
        scored_attack[scored_attack["seed"].eq(args.development_seed)],
        figures_dir / "selected_candidate_development_distribution",
    )

    metadata = {
        "experiment_version": "3.7.1-analysis",
        "status": "development_signal_audit_not_final_paper_result",
        "seeds": seeds,
        "development_seed": int(args.development_seed),
        "heldout_development_seeds": [
            seed for seed in seeds if seed != args.development_seed
        ],
        "selected_candidate": selected,
        "selection_frozen_before_heldout_review": True,
        "development_safety_constraints": {
            "max_clean_false_positive_rate": float(args.max_development_clean_fpr),
            "max_attack_benign_false_positive_rate": float(args.max_development_attack_benign_fpr),
            "max_final_benign_false_positive_rate": float(args.max_development_final_benign_fpr),
        },
        "ema_decay": float(args.ema_decay),
        "clean_threshold_quantile": float(args.clean_threshold_quantile),
        "candidate_count": int(len(candidate_columns)),
        "selection_uses_test_sets": False,
        "calibration_uses_attack_labels": False,
        "diagnostic_metrics_use_attack_labels": True,
    }
    with (output_dir / "transition_signature_analysis_v37_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print("Transition Signature Audit V3.7.1 safety-constrained analysis complete")
    print("Selected development candidate:", selected)
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

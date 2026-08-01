#!/usr/bin/env python3
"""V3.14 cross-attack detector failure audit.

This stage is analysis-only. It does not retrain the model, modify the frozen
detector, choose a new policy, or access the natural/diagnostic test sets.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


DEFAULT_SOURCES = [
    "DoS",
    "BruteForce",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web-Based",
]
DEFAULT_SEEDS = [7, 99, 123, 2026]
SOURCE_TO_SLUG = {
    "DoS": "dos_to_benign",
    "BruteForce": "bruteforce_to_benign",
    "Mirai": "mirai_to_benign",
    "Recon": "recon_to_benign",
    "Spoofing": "spoofing_to_benign",
    "Web-Based": "web_based_to_benign",
}
PREDEFINED_CANDIDATES = [
    "source_row_profile_max_abs",
    "source_row_profile_mean_abs",
    "source_row_consensus_max_abs",
    "source_row_consensus_mean_abs",
    "source_target_profile_positive_growth",
    "source_target_consensus_positive_growth",
    "global_profile_max_abs",
]
PREDEFINED_QUANTILES = [0.95, 0.975, 0.99]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dos-root", required=True, type=Path)
    p.add_argument("--remaining-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated attack source classes.",
    )
    p.add_argument(
        "--seeds",
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma-separated model seeds.",
    )
    p.add_argument("--target-class", default="Benign")
    return p.parse_args()


def q_higher(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    if len(y) == 0 or len(np.unique(y)) < 2 or roc_auc_score is None:
        return float("nan")
    return float(roc_auc_score(y.astype(int), score))


def weighted_recall(
    labels: np.ndarray,
    flags: np.ndarray,
    weights: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=bool)
    flags = np.asarray(flags, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    denom = float(weights[labels].sum())
    if denom <= 0:
        return float("nan")
    return float(weights[labels & flags].sum() / denom)


def find_root(source: str, dos_root: Path, remaining_root: Path) -> Path:
    return dos_root if source == "DoS" else remaining_root


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_pair_seed_summary(root: Path, source: str, seed: int) -> pd.DataFrame:
    slug = SOURCE_TO_SLUG[source]
    path = (
        root / "runs" / slug / f"seed_{seed}" / "summary"
        / "tables" / "v313_pair_seed_summary.csv"
    )
    d = pd.read_csv(require(path))
    if len(d) != 1:
        raise RuntimeError(f"Expected one pair-seed summary row: {path}")
    return d


def load_client_rows(root: Path, source: str, seed: int) -> pd.DataFrame:
    slug = SOURCE_TO_SLUG[source]
    path = (
        root / "runs" / slug / f"seed_{seed}" / "trusted_reconstruction"
        / "tables" / "reconstruction_client_rows.csv"
    )
    d = pd.read_csv(require(path))
    expected = {"monitoring_round", "client_id", "actual_malicious", "flagged"}
    missing = sorted(expected - set(d.columns))
    if missing:
        raise RuntimeError(f"Missing client columns {missing}: {path}")
    return d


def load_signature(
    root: Path,
    source: str,
    seed: int,
    branch: str,
) -> pd.DataFrame:
    slug = SOURCE_TO_SLUG[source]
    if branch == "attacked":
        path = (
            root / "runs" / slug / f"seed_{seed}" / "trusted_reconstruction"
            / "tables" / "reconstruction_transition_signature_long.csv"
        )
    elif branch == "clean":
        path = (
            root / "runs" / slug / f"seed_{seed}" / "exact_clean"
            / "tables" / "continuation_transition_signature_long.csv"
        )
    else:
        raise ValueError(branch)
    d = pd.read_csv(require(path))
    expected = {
        "monitoring_round", "client_id", "actual_malicious",
        "source_name", "target_name", "local_probability",
        "round_consensus_probability", "frozen_profile_probability",
    }
    missing = sorted(expected - set(d.columns))
    if missing:
        raise RuntimeError(f"Missing signature columns {missing}: {path}")
    return d


def signature_features(
    signature: pd.DataFrame,
    attacked_source: str,
    target_class: str,
) -> pd.DataFrame:
    d = signature.copy()
    d["profile_delta"] = (
        d["local_probability"].astype(float)
        - d["frozen_profile_probability"].astype(float)
    )
    d["consensus_delta"] = (
        d["local_probability"].astype(float)
        - d["round_consensus_probability"].astype(float)
    )
    d["profile_abs"] = d["profile_delta"].abs()
    d["consensus_abs"] = d["consensus_delta"].abs()
    key = ["monitoring_round", "client_id", "actual_malicious"]

    global_agg = d.groupby(key, as_index=False).agg(
        global_profile_max_abs=("profile_abs", "max"),
        global_profile_mean_abs=("profile_abs", "mean"),
        global_consensus_max_abs=("consensus_abs", "max"),
        global_consensus_mean_abs=("consensus_abs", "mean"),
    )

    source_row = d[d["source_name"].eq(attacked_source)].copy()
    source_agg = source_row.groupby(key, as_index=False).agg(
        source_row_profile_max_abs=("profile_abs", "max"),
        source_row_profile_mean_abs=("profile_abs", "mean"),
        source_row_consensus_max_abs=("consensus_abs", "max"),
        source_row_consensus_mean_abs=("consensus_abs", "mean"),
    )

    cell = source_row[source_row["target_name"].eq(target_class)].copy()
    cell = cell[key + [
        "profile_delta", "consensus_delta", "profile_abs", "consensus_abs"
    ]].rename(columns={
        "profile_delta": "source_target_profile_growth",
        "consensus_delta": "source_target_consensus_growth",
        "profile_abs": "source_target_profile_abs",
        "consensus_abs": "source_target_consensus_abs",
    })
    cell["source_target_profile_positive_growth"] = np.maximum(
        cell["source_target_profile_growth"].to_numpy(dtype=float), 0.0
    )
    cell["source_target_consensus_positive_growth"] = np.maximum(
        cell["source_target_consensus_growth"].to_numpy(dtype=float), 0.0
    )

    out = global_agg.merge(source_agg, on=key, how="inner", validate="one_to_one")
    out = out.merge(cell, on=key, how="inner", validate="one_to_one")
    return out


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    a = parse_args()
    dos_root = a.dos_root.expanduser().resolve()
    remaining_root = a.remaining_root.expanduser().resolve()
    out = a.output_dir.expanduser().resolve()
    tables = out / "tables"
    figures = out / "figures"
    for p in (tables, figures):
        p.mkdir(parents=True, exist_ok=True)

    sources = [x.strip() for x in a.sources.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]
    unknown = sorted(set(sources) - set(SOURCE_TO_SLUG))
    if unknown:
        raise ValueError(f"Unknown sources: {unknown}")

    pair_seed_parts: List[pd.DataFrame] = []
    attacked_client_parts: List[pd.DataFrame] = []
    attacked_feature_parts: List[pd.DataFrame] = []
    clean_feature_parts: List[pd.DataFrame] = []

    for source in sources:
        root = find_root(source, dos_root, remaining_root)
        for seed in seeds:
            ps = load_pair_seed_summary(root, source, seed)
            pair_seed_parts.append(ps)

            clients = load_client_rows(root, source, seed)
            clients["source_class"] = source
            clients["seed"] = seed
            attacked_client_parts.append(clients)

            attack_sig = load_signature(root, source, seed, "attacked")
            attack_features = signature_features(
                attack_sig, source, a.target_class
            )
            attack_features["source_class"] = source
            attack_features["seed"] = seed
            attacked_feature_parts.append(attack_features)

            clean_sig = load_signature(root, source, seed, "clean")
            clean_features = signature_features(
                clean_sig, source, a.target_class
            )
            clean_features["source_class"] = source
            clean_features["seed"] = seed
            clean_feature_parts.append(clean_features)

    pair_seed = pd.concat(pair_seed_parts, ignore_index=True)
    clients = pd.concat(attacked_client_parts, ignore_index=True)
    attack_features = pd.concat(attacked_feature_parts, ignore_index=True)
    clean_features = pd.concat(clean_feature_parts, ignore_index=True)

    merge_key = [
        "source_class", "seed", "monitoring_round", "client_id",
        "actual_malicious",
    ]
    client_feature = clients.merge(
        attack_features,
        on=merge_key,
        how="left",
        validate="one_to_one",
    )
    client_feature.to_csv(
        tables / "v314_client_round_audit.csv", index=False
    )

    # Current frozen detector, count-aware and weight-aware audit.
    detection_rows = []
    for source, g in client_feature.groupby("source_class", sort=False):
        labels = g["actual_malicious"].astype(bool).to_numpy()
        flags = g["flagged"].astype(bool).to_numpy()
        benign = ~labels
        poisoned = (
            g["poisoned_rows"].to_numpy(dtype=float)
            if "poisoned_rows" in g.columns
            else labels.astype(float)
        )
        agg_w = (
            g["aggregation_weight"].to_numpy(dtype=float)
            if "aggregation_weight" in g.columns
            else np.ones(len(g), dtype=float)
        )
        detection_rows.append({
            "source_class": source,
            "client_round_count": int(len(g)),
            "malicious_client_round_count": int(labels.sum()),
            "benign_client_round_count": int(benign.sum()),
            "client_recall": float(flags[labels].mean()),
            "poisoned_row_weighted_recall": weighted_recall(
                labels, flags, poisoned
            ),
            "aggregation_weighted_recall": weighted_recall(
                labels, flags, agg_w
            ),
            "benign_fpr": float(flags[benign].mean()),
            "flagged_malicious_client_rounds": int((labels & flags).sum()),
            "missed_malicious_client_rounds": int((labels & ~flags).sum()),
            "false_positive_client_rounds": int((benign & flags).sum()),
            "total_poisoned_rows": float(poisoned[labels].sum()),
            "flagged_poisoned_rows": float(
                poisoned[labels & flags].sum()
            ),
            "total_malicious_aggregation_weight": float(
                agg_w[labels].sum()
            ),
            "flagged_malicious_aggregation_weight": float(
                agg_w[labels & flags].sum()
            ),
        })
    detection = pd.DataFrame(detection_rows)
    detection.to_csv(
        tables / "v314_count_weighted_detection_summary.csv",
        index=False,
    )

    missed = client_feature[
        client_feature["actual_malicious"].astype(bool)
        & ~client_feature["flagged"].astype(bool)
    ].copy()
    missed.to_csv(
        tables / "v314_missed_malicious_client_rounds.csv",
        index=False,
    )
    false_pos = client_feature[
        ~client_feature["actual_malicious"].astype(bool)
        & client_feature["flagged"].astype(bool)
    ].copy()
    false_pos.to_csv(
        tables / "v314_false_positive_client_rounds.csv",
        index=False,
    )

    fp_repeat = (
        false_pos.groupby(
            ["source_class", "seed", "client_id"], as_index=False
        )
        .agg(
            false_positive_rounds=("monitoring_round", "nunique"),
            mean_policy_ratio=("policy_ratio", "mean")
            if "policy_ratio" in false_pos.columns
            else ("monitoring_round", "size"),
            maximum_policy_ratio=("policy_ratio", "max")
            if "policy_ratio" in false_pos.columns
            else ("monitoring_round", "size"),
        )
        .sort_values(
            ["false_positive_rounds", "source_class", "seed"],
            ascending=[False, True, True],
        )
    )
    fp_repeat.to_csv(
        tables / "v314_repeated_false_positive_clients.csv",
        index=False,
    )

    # Feature discrimination audit, descriptive only.
    feature_rows = []
    for source, g in client_feature.groupby("source_class", sort=False):
        y = g["actual_malicious"].astype(bool).to_numpy()
        for feature in PREDEFINED_CANDIDATES:
            if feature not in g.columns:
                continue
            scores = g[feature].to_numpy(dtype=float)
            feature_rows.append({
                "source_class": source,
                "feature": feature,
                "roc_auc": safe_auc(y, scores),
                "malicious_mean": float(np.nanmean(scores[y])),
                "benign_mean": float(np.nanmean(scores[~y])),
                "malicious_median": float(np.nanmedian(scores[y])),
                "benign_median": float(np.nanmedian(scores[~y])),
            })
    feature_discrimination = pd.DataFrame(feature_rows)
    feature_discrimination.to_csv(
        tables / "v314_feature_discrimination.csv",
        index=False,
    )

    # Clean-only calibration of predefined class-conditional diagnostics.
    threshold_rows = []
    candidate_eval_rows = []
    for source in sources:
        for seed in seeds:
            clean_g = clean_features[
                clean_features["source_class"].eq(source)
                & clean_features["seed"].eq(seed)
            ]
            attack_g = client_feature[
                client_feature["source_class"].eq(source)
                & client_feature["seed"].eq(seed)
            ]
            labels = attack_g["actual_malicious"].astype(bool).to_numpy()
            for feature in PREDEFINED_CANDIDATES:
                if feature not in clean_g.columns or feature not in attack_g.columns:
                    continue
                for quantile in PREDEFINED_QUANTILES:
                    threshold = q_higher(
                        clean_g[feature].to_numpy(dtype=float), quantile
                    )
                    scores = attack_g[feature].to_numpy(dtype=float)
                    flags = scores > threshold
                    benign = ~labels
                    threshold_rows.append({
                        "source_class": source,
                        "seed": seed,
                        "feature": feature,
                        "clean_quantile": quantile,
                        "clean_threshold": threshold,
                        "clean_row_count": int(len(clean_g)),
                    })
                    candidate_eval_rows.append({
                        "source_class": source,
                        "seed": seed,
                        "feature": feature,
                        "clean_quantile": quantile,
                        "malicious_recall": float(flags[labels].mean()),
                        "benign_fpr": float(flags[benign].mean()),
                        "flagged_clients_mean_per_round": float(
                            pd.Series(flags).groupby(
                                attack_g["monitoring_round"].to_numpy()
                            ).sum().mean()
                        ),
                        "attack_outcomes_used_for_threshold": False,
                    })
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(
        tables / "v314_clean_calibrated_candidate_thresholds.csv",
        index=False,
    )
    candidate_eval = pd.DataFrame(candidate_eval_rows)
    candidate_eval.to_csv(
        tables / "v314_clean_calibrated_candidate_evaluation.csv",
        index=False,
    )
    candidate_aggregate = (
        candidate_eval.groupby(
            ["feature", "clean_quantile"], as_index=False
        )
        .agg(
            mean_malicious_recall=("malicious_recall", "mean"),
            minimum_source_seed_recall=("malicious_recall", "min"),
            mean_benign_fpr=("benign_fpr", "mean"),
            maximum_source_seed_fpr=("benign_fpr", "max"),
            source_seed_count=("source_class", "size"),
        )
    )
    candidate_aggregate["predefined_candidate_only"] = True
    candidate_aggregate["selected_as_new_detector"] = False
    candidate_aggregate.to_csv(
        tables / "v314_candidate_aggregate_summary.csv",
        index=False,
    )

    # Link round-level detection coverage to mitigation.
    pair_round_parts = []
    for source in sources:
        root = find_root(source, dos_root, remaining_root)
        slug = SOURCE_TO_SLUG[source]
        for seed in seeds:
            path = (
                root / "runs" / slug / f"seed_{seed}" / "summary"
                / "tables" / "v313_pair_seed_rounds.csv"
            )
            d = pd.read_csv(require(path))
            d["source_class"] = source
            d["seed"] = seed
            pair_round_parts.append(d)
    pair_round = pd.concat(pair_round_parts, ignore_index=True)

    coverage_rows = []
    for keys, g in client_feature.groupby(
        ["source_class", "seed", "monitoring_round"], sort=False
    ):
        labels = g["actual_malicious"].astype(bool).to_numpy()
        flags = g["flagged"].astype(bool).to_numpy()
        poisoned = (
            g["poisoned_rows"].to_numpy(dtype=float)
            if "poisoned_rows" in g.columns
            else labels.astype(float)
        )
        agg_w = (
            g["aggregation_weight"].to_numpy(dtype=float)
            if "aggregation_weight" in g.columns
            else np.ones(len(g), dtype=float)
        )
        coverage_rows.append({
            "source_class": keys[0],
            "seed": keys[1],
            "monitoring_round": keys[2],
            "client_recall": float(flags[labels].mean()),
            "poisoned_row_weighted_recall": weighted_recall(
                labels, flags, poisoned
            ),
            "aggregation_weighted_recall": weighted_recall(
                labels, flags, agg_w
            ),
            "benign_fpr": float(flags[~labels].mean()),
        })
    coverage = pd.DataFrame(coverage_rows)
    round_link = pair_round.merge(
        coverage,
        on=["source_class", "seed", "monitoring_round"],
        how="left",
        validate="one_to_one",
    )
    round_link["absolute_target_rate_reduction"] = (
        round_link["plain_attack_source_to_target_rate"]
        - round_link["defended_attack_source_to_target_rate"]
    )
    round_link["macro_f1_recovery"] = (
        round_link["defended_attack_macro_f1"]
        - round_link["plain_attack_macro_f1"]
    )
    round_link.to_csv(
        tables / "v314_round_detection_mitigation_link.csv",
        index=False,
    )

    corr_rows = []
    for source, g in round_link.groupby("source_class", sort=False):
        for coverage_col in [
            "client_recall",
            "poisoned_row_weighted_recall",
            "aggregation_weighted_recall",
        ]:
            for outcome in [
                "absolute_target_rate_reduction",
                "macro_f1_recovery",
            ]:
                corr_rows.append({
                    "source_class": source,
                    "coverage_measure": coverage_col,
                    "outcome": outcome,
                    "pearson_correlation": float(
                        g[[coverage_col, outcome]].corr().iloc[0, 1]
                    ),
                    "round_count": int(len(g)),
                })
    pd.DataFrame(corr_rows).to_csv(
        tables / "v314_detection_mitigation_correlations.csv",
        index=False,
    )

    # Combined pair-level scientific status.
    pair_aggregate = (
        pair_seed.groupby("source_class", as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            mean_attack_excess_removed=("attack_excess_removed", "mean"),
            positive_reduction_seed_count=(
                "defense_positive_reduction", "sum"
            ),
            total_rounds_improved=("rounds_improved_vs_plain", "sum"),
            qualified_attack_seed_count=(
                "plain_attack_qualified_absolute_002", "sum"
            ),
            mean_clean_macro_f1=("clean_mean_macro_f1", "mean"),
            mean_plain_attack_macro_f1=(
                "plain_attack_mean_macro_f1", "mean"
            ),
            mean_defended_attack_macro_f1=(
                "defended_attack_mean_macro_f1", "mean"
            ),
        )
        .merge(detection, on="source_class", how="left")
    )
    pair_aggregate["reconstruction_mitigation_positive"] = (
        pair_aggregate["positive_reduction_seed_count"] >= 3
    )
    pair_aggregate["frozen_detector_meets_recall_criteria"] = (
        (pair_aggregate["client_recall"] >= 0.90)
    )
    pair_aggregate["frozen_detector_meets_fpr_criteria"] = (
        pair_aggregate["benign_fpr"] <= 0.05
    )
    pair_aggregate.to_csv(
        tables / "v314_pair_detection_mitigation_summary.csv",
        index=False,
    )

    # Figures.
    x = np.arange(len(detection))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(x - width, detection["client_recall"], width, label="Client recall")
    ax.bar(
        x,
        detection["poisoned_row_weighted_recall"],
        width,
        label="Poisoned-row weighted recall",
    )
    ax.bar(
        x + width,
        detection["aggregation_weighted_recall"],
        width,
        label="Aggregation-weighted recall",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(detection["source_class"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("V3.14 frozen detector coverage by attack family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "detector_coverage_by_attack")

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.scatter(
        round_link["aggregation_weighted_recall"],
        round_link["absolute_target_rate_reduction"],
        alpha=0.75,
    )
    ax.set_xlabel("Aggregation-weighted malicious recall")
    ax.set_ylabel("Plain minus defended source-to-target rate")
    ax.set_title("Detection coverage versus targeted mitigation")
    ax.grid(alpha=0.25)
    save_figure(fig, figures / "coverage_vs_targeted_mitigation")

    best_display = candidate_aggregate.sort_values(
        ["clean_quantile", "mean_malicious_recall"],
        ascending=[True, False],
    )
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for q, g in best_display.groupby("clean_quantile"):
        ax.scatter(
            g["mean_benign_fpr"],
            g["mean_malicious_recall"],
            label=f"Clean q={q:g}",
            s=55,
        )
        for _, row in g.iterrows():
            ax.annotate(
                row["feature"].replace("_", " "),
                (row["mean_benign_fpr"], row["mean_malicious_recall"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Mean benign FPR")
    ax.set_ylabel("Mean malicious recall")
    ax.set_title(
        "Exploratory clean-calibrated class-conditional diagnostics"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "candidate_recall_fpr_frontier")

    decision = {
        "experiment_version": "3.14",
        "stage": "cross_attack_detector_failure_audit",
        "source_classes": sources,
        "seeds": seeds,
        "target_class": a.target_class,
        "frozen_detector_modified": False,
        "reconstruction_policy_modified": False,
        "new_detector_selected": False,
        "attack_outcomes_used_for_candidate_thresholds": False,
        "candidate_features_predefined": PREDEFINED_CANDIDATES,
        "candidate_clean_quantiles_predefined": PREDEFINED_QUANTILES,
        "all_pairs_positive_mitigation": bool(
            pair_aggregate["reconstruction_mitigation_positive"].all()
        ),
        "all_pairs_meet_client_recall_090": bool(
            pair_aggregate["frozen_detector_meets_recall_criteria"].all()
        ),
        "all_pairs_meet_benign_fpr_005": bool(
            pair_aggregate["frozen_detector_meets_fpr_criteria"].all()
        ),
        "test_sets_accessed": False,
        "recommended_next_stage": (
            "freeze a class-conditional detector candidate only after "
            "reviewing this audit; do not tune on final untouched seeds"
        ),
    }
    with (out / "v314_audit_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    pd.DataFrame([{
        "all_pairs_positive_mitigation":
            decision["all_pairs_positive_mitigation"],
        "all_pairs_meet_client_recall_090":
            decision["all_pairs_meet_client_recall_090"],
        "all_pairs_meet_benign_fpr_005":
            decision["all_pairs_meet_benign_fpr_005"],
        "new_detector_selected": False,
        "method_reopened": False,
        "test_sets_accessed": False,
    }]).to_csv(tables / "v314_audit_decision.csv", index=False)

    print("V3.14 cross-attack detector failure audit complete")
    print()
    print("COUNT-WEIGHTED DETECTION")
    print(detection.to_string(index=False))
    print()
    print("PAIR DETECTION AND MITIGATION")
    print(pair_aggregate.to_string(index=False))
    print()
    print("AUDIT DECISION")
    print(pd.read_csv(
        tables / "v314_audit_decision.csv"
    ).to_string(index=False))
    print()
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

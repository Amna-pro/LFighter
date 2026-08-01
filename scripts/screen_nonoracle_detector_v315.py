#!/usr/bin/env python3
"""V3.15 non-oracle detector candidate screen.

Analysis-only stage using completed V3.13.2 validation evidence. It does not
retrain models, alter reconstruction, access test sets, or claim a deployable
detector. Candidate scores never use the true attacked source class.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SOURCES = [
    "DoS", "BruteForce", "Mirai", "Recon", "Spoofing", "Web-Based"
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
QUANTILES = [0.95, 0.975, 0.99]

# target_aware=True means the candidate assumes the attacker redirects into
# Benign. It is reported as a family-specific diagnostic but is ineligible for
# selection as an attack-agnostic detector.
CANDIDATES = {
    "raw_max_positive_offdiag": False,
    "raw_max_abs_offdiag": False,
    "robust_z_max_positive_offdiag": False,
    "robust_z_max_abs_offdiag": False,
    "max_row_mean_abs_profile_delta": False,
    "raw_max_positive_to_benign": True,
    "robust_z_max_positive_to_benign": True,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dos-root", required=True, type=Path)
    p.add_argument("--remaining-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    p.add_argument(
        "--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS)
    )
    p.add_argument("--target-class", default="Benign")
    return p.parse_args()


def q_higher(values: np.ndarray, q: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    try:
        return float(np.quantile(x, q, method="higher"))
    except TypeError:
        return float(np.quantile(x, q, interpolation="higher"))


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def root_for(source: str, dos_root: Path, remaining_root: Path) -> Path:
    return dos_root if source == "DoS" else remaining_root


def signature_path(
    root: Path, source: str, seed: int, branch: str
) -> Path:
    slug = SOURCE_TO_SLUG[source]
    if branch == "attacked":
        return (
            root / "runs" / slug / f"seed_{seed}"
            / "trusted_reconstruction" / "tables"
            / "reconstruction_transition_signature_long.csv"
        )
    if branch == "clean":
        return (
            root / "runs" / slug / f"seed_{seed}"
            / "exact_clean" / "tables"
            / "continuation_transition_signature_long.csv"
        )
    raise ValueError(branch)


def client_path(root: Path, source: str, seed: int) -> Path:
    slug = SOURCE_TO_SLUG[source]
    return (
        root / "runs" / slug / f"seed_{seed}"
        / "trusted_reconstruction" / "tables"
        / "reconstruction_client_rows.csv"
    )


def load_signature(path: Path) -> pd.DataFrame:
    d = pd.read_csv(require(path))
    needed = {
        "monitoring_round", "client_id", "actual_malicious",
        "source_name", "target_name", "local_probability",
        "frozen_profile_probability",
    }
    missing = sorted(needed - set(d.columns))
    if missing:
        raise RuntimeError(f"Missing columns {missing}: {path}")
    return d


def canonical_clean_signature(
    dos_root: Path, remaining_root: Path, seed: int
) -> pd.DataFrame:
    # Exact clean trajectories are model-identical across source-pair runs.
    # Prefer DoS, then any available remaining source. This avoids duplicating
    # the same 80 clean client-rounds six times during calibration.
    candidates = [
        signature_path(dos_root, "DoS", seed, "clean")
    ]
    for source in DEFAULT_SOURCES:
        if source == "DoS":
            continue
        candidates.append(
            signature_path(remaining_root, source, seed, "clean")
        )
    for path in candidates:
        if path.exists():
            return load_signature(path)
    raise FileNotFoundError(
        f"No exact clean transition signature found for seed {seed}"
    )


def robust_cell_stats(clean_long: pd.DataFrame) -> pd.DataFrame:
    d = clean_long.copy()
    d["profile_delta"] = (
        d["local_probability"].astype(float)
        - d["frozen_profile_probability"].astype(float)
    )
    rows = []
    for (s, t), g in d.groupby(["source_name", "target_name"]):
        x = g["profile_delta"].to_numpy(dtype=float)
        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))
        q25, q75 = np.quantile(x, [0.25, 0.75])
        iqr_scale = float((q75 - q25) / 1.349)
        robust_scale = max(1.4826 * mad, iqr_scale, 1e-6)
        rows.append({
            "source_name": s,
            "target_name": t,
            "clean_median_delta": median,
            "clean_robust_scale": robust_scale,
            "clean_mad": mad,
            "clean_iqr_scale": iqr_scale,
            "clean_cell_rows": int(len(x)),
        })
    return pd.DataFrame(rows)


def score_signature(
    long_df: pd.DataFrame,
    stats: pd.DataFrame,
    target_class: str,
) -> pd.DataFrame:
    d = long_df.copy()
    d["profile_delta"] = (
        d["local_probability"].astype(float)
        - d["frozen_profile_probability"].astype(float)
    )
    d = d.merge(
        stats,
        on=["source_name", "target_name"],
        how="left",
        validate="many_to_one",
    )
    d["robust_z"] = (
        d["profile_delta"] - d["clean_median_delta"]
    ) / d["clean_robust_scale"]
    d["positive_delta"] = np.maximum(
        d["profile_delta"].to_numpy(dtype=float), 0.0
    )
    d["positive_z"] = np.maximum(
        d["robust_z"].to_numpy(dtype=float), 0.0
    )
    d["abs_delta"] = d["profile_delta"].abs()
    d["abs_z"] = d["robust_z"].abs()
    d["offdiag"] = ~d["source_name"].eq(d["target_name"])
    d["to_benign"] = (
        d["target_name"].eq(target_class)
        & ~d["source_name"].eq(target_class)
    )

    key = ["monitoring_round", "client_id", "actual_malicious"]
    base = d[key].drop_duplicates().copy()

    off = d[d["offdiag"]]
    off_agg = off.groupby(key, as_index=False).agg(
        raw_max_positive_offdiag=("positive_delta", "max"),
        raw_max_abs_offdiag=("abs_delta", "max"),
        robust_z_max_positive_offdiag=("positive_z", "max"),
        robust_z_max_abs_offdiag=("abs_z", "max"),
    )

    row_means = (
        d.groupby(key + ["source_name"], as_index=False)
        .agg(row_mean_abs=("abs_delta", "mean"))
        .groupby(key, as_index=False)
        .agg(max_row_mean_abs_profile_delta=("row_mean_abs", "max"))
    )

    to_b = d[d["to_benign"]].groupby(key, as_index=False).agg(
        raw_max_positive_to_benign=("positive_delta", "max"),
        robust_z_max_positive_to_benign=("positive_z", "max"),
    )

    out = base.merge(
        off_agg, on=key, how="left", validate="one_to_one"
    )
    out = out.merge(
        row_means, on=key, how="left", validate="one_to_one"
    )
    out = out.merge(
        to_b, on=key, how="left", validate="one_to_one"
    )
    return out


def crossfit_clean_flags(
    clean_scores: pd.DataFrame, feature: str, quantile: float
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    # Four-fold cross-fitting by monitoring round. Each round is scored using
    # a threshold from the other three exact clean rounds.
    flags = np.zeros(len(clean_scores), dtype=bool)
    details: List[Dict[str, float]] = []
    rounds = sorted(clean_scores["monitoring_round"].unique().tolist())
    for heldout in rounds:
        train = clean_scores[
            ~clean_scores["monitoring_round"].eq(heldout)
        ]
        valid = clean_scores[
            clean_scores["monitoring_round"].eq(heldout)
        ]
        threshold = q_higher(
            train[feature].to_numpy(dtype=float), quantile
        )
        idx = valid.index.to_numpy()
        flags[idx] = valid[feature].to_numpy(dtype=float) > threshold
        details.append({
            "heldout_round": int(heldout),
            "threshold": threshold,
            "calibration_rows": int(len(train)),
            "validation_rows": int(len(valid)),
        })
    return flags, details


def safe_mean_bool(x: pd.Series) -> float:
    return float(x.astype(bool).mean()) if len(x) else float("nan")


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
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    sources = [x.strip() for x in a.sources.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]
    unknown = sorted(set(sources) - set(SOURCE_TO_SLUG))
    if unknown:
        raise ValueError(f"Unknown sources: {unknown}")

    clean_score_parts = []
    cell_stat_parts = []
    for seed in seeds:
        clean_long = canonical_clean_signature(
            dos_root, remaining_root, seed
        )
        stats = robust_cell_stats(clean_long)
        stats["seed"] = seed
        cell_stat_parts.append(stats)
        clean_scores = score_signature(
            clean_long, stats.drop(columns=["seed"]), a.target_class
        )
        clean_scores["seed"] = seed
        clean_score_parts.append(clean_scores)

    cell_stats = pd.concat(cell_stat_parts, ignore_index=True)
    cell_stats.to_csv(
        tables / "v315_clean_cell_robust_statistics.csv", index=False
    )
    clean_scores_all = pd.concat(clean_score_parts, ignore_index=True)
    clean_scores_all.to_csv(
        tables / "v315_clean_candidate_scores.csv", index=False
    )

    attacked_parts = []
    for source in sources:
        root = root_for(source, dos_root, remaining_root)
        for seed in seeds:
            stats = cell_stats[
                cell_stats["seed"].eq(seed)
            ].drop(columns=["seed"])
            sig = load_signature(
                signature_path(root, source, seed, "attacked")
            )
            scores = score_signature(sig, stats, a.target_class)
            clients = pd.read_csv(
                require(client_path(root, source, seed))
            )
            key = [
                "monitoring_round", "client_id", "actual_malicious"
            ]
            keep = key + ["flagged"]
            missing = sorted(set(keep) - set(clients.columns))
            if missing:
                raise RuntimeError(
                    f"Missing client columns {missing} for {source}, {seed}"
                )
            clients = clients[keep].copy()
            scores = scores.merge(
                clients,
                on=key,
                how="left",
                validate="one_to_one",
            )
            scores["source_class"] = source
            scores["seed"] = seed
            attacked_parts.append(scores)
    attacked = pd.concat(attacked_parts, ignore_index=True)
    attacked.to_csv(
        tables / "v315_attacked_candidate_scores.csv", index=False
    )

    threshold_rows = []
    evaluation_rows = []
    source_rows = []

    for seed in seeds:
        clean_seed = (
            clean_scores_all[clean_scores_all["seed"].eq(seed)]
            .reset_index(drop=True)
        )
        for feature, target_aware in CANDIDATES.items():
            for quantile in QUANTILES:
                full_threshold = q_higher(
                    clean_seed[feature].to_numpy(dtype=float), quantile
                )
                cross_flags, details = crossfit_clean_flags(
                    clean_seed, feature, quantile
                )
                crossfit_clean_fpr = float(cross_flags.mean())
                for detail in details:
                    threshold_rows.append({
                        "seed": seed,
                        "feature": feature,
                        "target_aware": target_aware,
                        "clean_quantile": quantile,
                        "full_clean_threshold": full_threshold,
                        "crossfit_clean_fpr": crossfit_clean_fpr,
                        **detail,
                    })

                for source in sources:
                    g = attacked[
                        attacked["seed"].eq(seed)
                        & attacked["source_class"].eq(source)
                    ].copy()
                    labels = g["actual_malicious"].astype(bool)
                    current = g["flagged"].astype(bool)
                    candidate = g[feature].astype(float) > full_threshold
                    for fusion, flags in [
                        ("candidate_only", candidate),
                        ("or_current_detector", candidate | current),
                    ]:
                        malicious_recall = safe_mean_bool(flags[labels])
                        benign_fpr = safe_mean_bool(flags[~labels])
                        evaluation_rows.append({
                            "source_class": source,
                            "seed": seed,
                            "feature": feature,
                            "fusion": fusion,
                            "target_aware": target_aware,
                            "clean_quantile": quantile,
                            "threshold": full_threshold,
                            "crossfit_clean_fpr": crossfit_clean_fpr,
                            "malicious_recall": malicious_recall,
                            "benign_fpr": benign_fpr,
                            "flagged_malicious_client_rounds": int(
                                (flags & labels).sum()
                            ),
                            "missed_malicious_client_rounds": int(
                                ((~flags) & labels).sum()
                            ),
                            "false_positive_client_rounds": int(
                                (flags & (~labels)).sum()
                            ),
                            "attack_labels_used_for_threshold": False,
                        })

    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(
        tables / "v315_crossfit_clean_thresholds.csv", index=False
    )
    evaluation = pd.DataFrame(evaluation_rows)
    evaluation.to_csv(
        tables / "v315_candidate_source_seed_evaluation.csv",
        index=False,
    )

    source_summary = (
        evaluation.groupby(
            [
                "source_class", "feature", "fusion",
                "target_aware", "clean_quantile"
            ],
            as_index=False,
        )
        .agg(
            mean_malicious_recall=("malicious_recall", "mean"),
            minimum_seed_recall=("malicious_recall", "min"),
            mean_benign_fpr=("benign_fpr", "mean"),
            maximum_seed_fpr=("benign_fpr", "max"),
            mean_crossfit_clean_fpr=("crossfit_clean_fpr", "mean"),
            seed_count=("seed", "nunique"),
        )
    )
    source_summary.to_csv(
        tables / "v315_candidate_source_summary.csv", index=False
    )

    aggregate = (
        evaluation.groupby(
            ["feature", "fusion", "target_aware", "clean_quantile"],
            as_index=False,
        )
        .agg(
            mean_malicious_recall=("malicious_recall", "mean"),
            minimum_source_seed_recall=("malicious_recall", "min"),
            mean_attack_benign_fpr=("benign_fpr", "mean"),
            maximum_source_seed_fpr=("benign_fpr", "max"),
            mean_crossfit_clean_fpr=("crossfit_clean_fpr", "mean"),
            source_seed_count=("source_class", "size"),
        )
    )

    wide_recall = source_summary.pivot_table(
        index=["feature", "fusion", "target_aware", "clean_quantile"],
        columns="source_class",
        values="mean_malicious_recall",
    ).reset_index()
    aggregate = aggregate.merge(
        wide_recall,
        on=["feature", "fusion", "target_aware", "clean_quantile"],
        how="left",
        validate="one_to_one",
    )

    # Predefined eligibility and selection criteria. Target-aware candidates
    # are never eligible for the universal detector.
    aggregate["attack_agnostic_eligible"] = ~aggregate["target_aware"]
    aggregate["passes_mean_recall_090"] = (
        aggregate["mean_malicious_recall"] >= 0.90
    )
    aggregate["passes_min_source_seed_recall_050"] = (
        aggregate["minimum_source_seed_recall"] >= 0.50
    )
    aggregate["passes_mean_attack_fpr_005"] = (
        aggregate["mean_attack_benign_fpr"] <= 0.05
    )
    aggregate["passes_max_attack_fpr_010"] = (
        aggregate["maximum_source_seed_fpr"] <= 0.10
    )
    aggregate["passes_crossfit_clean_fpr_005"] = (
        aggregate["mean_crossfit_clean_fpr"] <= 0.05
    )
    aggregate["passes_bruteforce_recall_075"] = (
        aggregate.get("BruteForce", np.nan) >= 0.75
    )
    aggregate["passes_web_recall_075"] = (
        aggregate.get("Web-Based", np.nan) >= 0.75
    )
    aggregate["passes_dos_recall_095"] = (
        aggregate.get("DoS", np.nan) >= 0.95
    )
    criteria = [
        "attack_agnostic_eligible",
        "passes_mean_recall_090",
        "passes_min_source_seed_recall_050",
        "passes_mean_attack_fpr_005",
        "passes_max_attack_fpr_010",
        "passes_crossfit_clean_fpr_005",
        "passes_bruteforce_recall_075",
        "passes_web_recall_075",
        "passes_dos_recall_095",
    ]
    aggregate["passes_all_predefined_criteria"] = aggregate[
        criteria
    ].all(axis=1)
    aggregate["selected_as_v315_candidate"] = False

    passing = aggregate[
        aggregate["passes_all_predefined_criteria"]
    ].copy()
    selected = None
    if len(passing):
        # Deterministic tie-break: lowest attack FPR, highest minimum recall,
        # highest mean recall, candidate-only before OR, then feature name.
        passing["fusion_order"] = (
            passing["fusion"].eq("or_current_detector").astype(int)
        )
        passing = passing.sort_values(
            [
                "mean_attack_benign_fpr",
                "minimum_source_seed_recall",
                "mean_malicious_recall",
                "fusion_order",
                "feature",
                "clean_quantile",
            ],
            ascending=[True, False, False, True, True, False],
        )
        selected = passing.iloc[0]
        mask = (
            aggregate["feature"].eq(selected["feature"])
            & aggregate["fusion"].eq(selected["fusion"])
            & aggregate["clean_quantile"].eq(
                selected["clean_quantile"]
            )
        )
        aggregate.loc[mask, "selected_as_v315_candidate"] = True

    aggregate.to_csv(
        tables / "v315_candidate_aggregate_summary.csv", index=False
    )

    decision = {
        "experiment_version": "3.15",
        "stage": "non_oracle_detector_candidate_screen",
        "sources": sources,
        "seeds": seeds,
        "target_class": a.target_class,
        "training_rerun": False,
        "reconstruction_modified": False,
        "current_detector_modified": False,
        "test_sets_accessed": False,
        "attack_outcomes_used_for_threshold_calibration": False,
        "attack_outcomes_used_for_development_candidate_selection": True,
        "final_untouched_seeds_required_after_freeze": True,
        "candidate_selected": selected is not None,
        "selected_feature": (
            None if selected is None else str(selected["feature"])
        ),
        "selected_fusion": (
            None if selected is None else str(selected["fusion"])
        ),
        "selected_clean_quantile": (
            None if selected is None
            else float(selected["clean_quantile"])
        ),
        "selected_target_aware": (
            None if selected is None
            else bool(selected["target_aware"])
        ),
        "recommended_next_stage": (
            "implement and freeze selected non-oracle detector, then validate "
            "on new untouched seeds and broader attacks"
            if selected is not None
            else
            "no candidate satisfies the frozen criteria; perform a focused "
            "support-normalized or conformal detector design stage before "
            "integration"
        ),
    }
    with (out / "v315_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    pd.DataFrame([{
        "candidate_selected": decision["candidate_selected"],
        "selected_feature": decision["selected_feature"],
        "selected_fusion": decision["selected_fusion"],
        "selected_clean_quantile":
            decision["selected_clean_quantile"],
        "selected_target_aware": decision["selected_target_aware"],
        "current_detector_modified": False,
        "method_reopened": False,
        "test_sets_accessed": False,
    }]).to_csv(tables / "v315_decision.csv", index=False)

    # Publication-ready diagnostic figures.
    plot = aggregate[
        aggregate["attack_agnostic_eligible"]
        & aggregate["fusion"].eq("or_current_detector")
    ].copy()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for q, g in plot.groupby("clean_quantile"):
        ax.scatter(
            g["mean_attack_benign_fpr"],
            g["mean_malicious_recall"],
            s=65,
            label=f"Clean q={q:g}",
        )
        for _, row in g.iterrows():
            ax.annotate(
                row["feature"].replace("_", " "),
                (
                    row["mean_attack_benign_fpr"],
                    row["mean_malicious_recall"],
                ),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.axhline(0.90, linestyle="--", linewidth=1)
    ax.axvline(0.05, linestyle="--", linewidth=1)
    ax.set_xlabel("Mean benign FPR during attack")
    ax.set_ylabel("Mean malicious recall")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.set_title("V3.15 non-oracle OR-fusion candidate frontier")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "non_oracle_or_fusion_frontier")

    best_by_source = source_summary[
        (~source_summary["target_aware"])
        & source_summary["fusion"].eq("or_current_detector")
    ].copy()
    best_idx = (
        best_by_source.sort_values(
            [
                "source_class", "mean_malicious_recall",
                "mean_benign_fpr"
            ],
            ascending=[True, False, True],
        )
        .groupby("source_class", as_index=False)
        .head(1)
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    x = np.arange(len(best_idx))
    ax.bar(x, best_idx["mean_malicious_recall"])
    ax.set_xticks(x)
    ax.set_xticklabels(
        best_idx["source_class"], rotation=25, ha="right"
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean malicious recall")
    ax.set_title(
        "Best exploratory non-oracle OR-fusion recall by attack family"
    )
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures / "best_non_oracle_recall_by_attack")

    print("V3.15 non-oracle detector candidate screen complete")
    print()
    print("TOP ATTACK-AGNOSTIC CANDIDATES")
    display = aggregate[
        aggregate["attack_agnostic_eligible"]
    ].sort_values(
        [
            "passes_all_predefined_criteria",
            "mean_malicious_recall",
            "mean_attack_benign_fpr",
        ],
        ascending=[False, False, True],
    )
    cols = [
        "feature", "fusion", "clean_quantile",
        "mean_malicious_recall", "minimum_source_seed_recall",
        "mean_attack_benign_fpr", "maximum_source_seed_fpr",
        "mean_crossfit_clean_fpr", "BruteForce", "Web-Based", "DoS",
        "passes_all_predefined_criteria",
        "selected_as_v315_candidate",
    ]
    print(display[cols].head(20).to_string(index=False))
    print()
    print("V3.15 DECISION")
    print(pd.read_csv(
        tables / "v315_decision.csv"
    ).to_string(index=False))
    print()
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

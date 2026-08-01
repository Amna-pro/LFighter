#!/usr/bin/env python3
"""V3.16 impact-aware temporal conformal detector development screen.

This is a development-stage, analysis-only experiment over completed V3.13.2
validation runs. It does not retrain models, alter trusted reconstruction, or
access natural/diagnostic test sets.

All detector candidates are non-oracle:
- they scan every off-diagonal transition cell,
- they do not use the true attacked source or target,
- they use only clean-calibrated cellwise empirical tail probabilities,
  aggregation weights, stable client IDs, and the existing frozen detector.

Leave-one-seed-out clean calibration is used throughout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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

DIRECTIONS = ["positive", "absolute"]
TOP_K_VALUES = [1, 2, 4, 8]
TEMPORAL_DECAYS = [0.0, 0.65, 0.85]
IMPACT_BETAS = [0.0, 0.5, 1.0]
CLEAN_QUANTILES = [0.95, 0.975, 0.99]
FUSIONS = ["candidate_only", "or_current_detector"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dos-root", required=True, type=Path)
    p.add_argument("--remaining-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    p.add_argument(
        "--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS)
    )
    return p.parse_args()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def q_higher(values: np.ndarray, q: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    try:
        return float(np.quantile(x, q, method="higher"))
    except TypeError:
        return float(np.quantile(x, q, interpolation="higher"))


def root_for(source: str, dos_root: Path, remaining_root: Path) -> Path:
    return dos_root if source == "DoS" else remaining_root


def slug_for(source: str) -> str:
    return SOURCE_TO_SLUG[source]


def clean_signature_path(dos_root: Path, seed: int) -> Path:
    return (
        dos_root / "runs" / "dos_to_benign" / f"seed_{seed}"
        / "exact_clean" / "tables"
        / "continuation_transition_signature_long.csv"
    )


def clean_anchor_path(dos_root: Path, seed: int) -> Path:
    return (
        dos_root / "runs" / "dos_to_benign" / f"seed_{seed}"
        / "exact_clean" / "tables"
        / "continuation_client_anchor_scores.csv"
    )


def attack_signature_path(
    root: Path, source: str, seed: int
) -> Path:
    return (
        root / "runs" / slug_for(source) / f"seed_{seed}"
        / "trusted_reconstruction" / "tables"
        / "reconstruction_transition_signature_long.csv"
    )


def attack_client_path(root: Path, source: str, seed: int) -> Path:
    return (
        root / "runs" / slug_for(source) / f"seed_{seed}"
        / "trusted_reconstruction" / "tables"
        / "reconstruction_client_rows.csv"
    )


def read_signature(path: Path) -> pd.DataFrame:
    d = pd.read_csv(require(path))
    needed = {
        "monitoring_round", "client_id", "actual_malicious",
        "source_name", "target_name", "local_probability",
        "frozen_profile_probability",
    }
    missing = sorted(needed - set(d.columns))
    if missing:
        raise RuntimeError(f"Missing signature columns {missing}: {path}")
    d = d.copy()
    d["profile_delta"] = (
        d["local_probability"].astype(float)
        - d["frozen_profile_probability"].astype(float)
    )
    d["offdiag"] = ~d["source_name"].eq(d["target_name"])
    return d[d["offdiag"]].copy()


def read_clean_clients(path: Path) -> pd.DataFrame:
    d = pd.read_csv(require(path))
    needed = {
        "monitoring_round", "client_id", "actual_malicious",
        "aggregation_weight", "flagged",
    }
    missing = sorted(needed - set(d.columns))
    if missing:
        raise RuntimeError(f"Missing clean client columns {missing}: {path}")
    return d[list(needed)].copy()


def read_attack_clients(path: Path) -> pd.DataFrame:
    d = pd.read_csv(require(path))
    needed = {
        "monitoring_round", "client_id", "actual_malicious",
        "aggregation_weight", "flagged",
    }
    missing = sorted(needed - set(d.columns))
    if missing:
        raise RuntimeError(f"Missing attacked client columns {missing}: {path}")
    return d[list(needed)].copy()


def build_tail_reference(
    train_long: pd.DataFrame,
    direction: str,
) -> Dict[Tuple[str, str], np.ndarray]:
    ref: Dict[Tuple[str, str], np.ndarray] = {}
    for (source, target), g in train_long.groupby(
        ["source_name", "target_name"], sort=False
    ):
        x = g["profile_delta"].to_numpy(dtype=float)
        if direction == "positive":
            values = x
        elif direction == "absolute":
            values = np.abs(x)
        else:
            raise ValueError(direction)
        ref[(str(source), str(target))] = np.sort(values)
    return ref


def upper_tail_p(sorted_reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.asarray(sorted_reference, dtype=float)
    x = np.asarray(values, dtype=float)
    n = len(ref)
    # Number of calibration values >= x, with finite-sample +1 correction.
    left = np.searchsorted(ref, x, side="left")
    ge = n - left
    return (ge.astype(float) + 1.0) / float(n + 1)


def score_client_rounds(
    long_df: pd.DataFrame,
    clients: pd.DataFrame,
    reference: Dict[Tuple[str, str], np.ndarray],
    direction: str,
    top_k: int,
) -> pd.DataFrame:
    d = long_df.copy().reset_index(drop=True)
    if direction == "positive":
        d["tail_value"] = d["profile_delta"].astype(float)
    elif direction == "absolute":
        d["tail_value"] = d["profile_delta"].abs().astype(float)
    else:
        raise ValueError(direction)

    p_values = np.empty(len(d), dtype=float)
    for (source, target), idx in d.groupby(
        ["source_name", "target_name"], sort=False
    ).groups.items():
        key = (str(source), str(target))
        if key not in reference:
            raise RuntimeError(f"Missing clean cell reference: {key}")
        positions = np.asarray(list(idx), dtype=int)
        p_values[positions] = upper_tail_p(
            reference[key],
            d.loc[positions, "tail_value"].to_numpy(dtype=float),
        )
    d["tail_p"] = p_values
    d["surprisal"] = -np.log10(np.maximum(d["tail_p"], 1e-12))

    key_cols = ["monitoring_round", "client_id", "actual_malicious"]
    rows = []
    for key, g in d.groupby(key_cols, sort=False):
        values = np.sort(g["surprisal"].to_numpy(dtype=float))[::-1]
        k = min(int(top_k), len(values))
        rows.append({
            "monitoring_round": int(key[0]),
            "client_id": int(key[1]),
            "actual_malicious": bool(key[2]),
            "cell_surprisal_topk_mean": float(values[:k].mean()),
            "cell_surprisal_max": float(values[0]),
            "cells_p_le_005": int((g["tail_p"] <= 0.05).sum()),
            "cells_p_le_001": int((g["tail_p"] <= 0.01).sum()),
        })
    scored = pd.DataFrame(rows)
    scored = scored.merge(
        clients,
        on=key_cols,
        how="left",
        validate="one_to_one",
        suffixes=("", "_client"),
    )
    return scored


def add_impact_and_temporal_score(
    scored: pd.DataFrame,
    beta: float,
    decay: float,
    reference_weight: float,
) -> pd.DataFrame:
    d = scored.copy()
    weight_ratio = (
        d["aggregation_weight"].astype(float)
        / max(float(reference_weight), 1e-12)
    )
    d["impact_multiplier"] = np.power(
        np.maximum(weight_ratio.to_numpy(dtype=float), 1e-6),
        float(beta),
    )
    d["impact_score"] = (
        d["cell_surprisal_topk_mean"].astype(float)
        * d["impact_multiplier"].astype(float)
    )
    d = d.sort_values(
        ["client_id", "monitoring_round"]
    ).reset_index(drop=True)
    temporal = np.empty(len(d), dtype=float)
    for _, idx in d.groupby("client_id", sort=False).groups.items():
        state = 0.0
        for position in list(idx):
            state = float(decay) * state + float(
                d.at[position, "impact_score"]
            )
            temporal[position] = state
    d["temporal_conformal_score"] = temporal
    return d


def candidate_id(
    direction: str,
    top_k: int,
    decay: float,
    beta: float,
    quantile: float,
    fusion: str,
) -> str:
    return (
        f"{direction}_top{top_k}_decay{decay:g}_"
        f"beta{beta:g}_q{quantile:g}_{fusion}"
    )


def safe_rate(flags: pd.Series) -> float:
    return float(flags.astype(bool).mean()) if len(flags) else float("nan")


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
        raise ValueError(f"Unknown source classes: {unknown}")
    if len(seeds) < 3:
        raise ValueError("V3.16 requires at least three development seeds")

    clean_long_by_seed: Dict[int, pd.DataFrame] = {}
    clean_clients_by_seed: Dict[int, pd.DataFrame] = {}
    attack_long: Dict[Tuple[str, int], pd.DataFrame] = {}
    attack_clients: Dict[Tuple[str, int], pd.DataFrame] = {}

    for seed in seeds:
        clean_long_by_seed[seed] = read_signature(
            clean_signature_path(dos_root, seed)
        )
        clean_clients_by_seed[seed] = read_clean_clients(
            clean_anchor_path(dos_root, seed)
        )
    for source in sources:
        root = root_for(source, dos_root, remaining_root)
        for seed in seeds:
            attack_long[(source, seed)] = read_signature(
                attack_signature_path(root, source, seed)
            )
            attack_clients[(source, seed)] = read_attack_clients(
                attack_client_path(root, source, seed)
            )

    evaluation_rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []
    compact_score_rows: List[pd.DataFrame] = []

    for heldout_seed in seeds:
        train_seeds = [s for s in seeds if s != heldout_seed]
        train_long = pd.concat(
            [clean_long_by_seed[s] for s in train_seeds],
            ignore_index=True,
        )
        train_clients = pd.concat(
            [clean_clients_by_seed[s] for s in train_seeds],
            ignore_index=True,
        )
        reference_weight = float(
            train_clients["aggregation_weight"].median()
        )

        for direction in DIRECTIONS:
            reference = build_tail_reference(train_long, direction)

            # Base scores depend on direction and top-k only.
            train_base_by_k: Dict[int, pd.DataFrame] = {}
            heldout_clean_base_by_k: Dict[int, pd.DataFrame] = {}
            attack_base_by_source_k: Dict[
                Tuple[str, int], pd.DataFrame
            ] = {}

            for top_k in TOP_K_VALUES:
                train_parts = []
                for seed in train_seeds:
                    train_parts.append(
                        score_client_rounds(
                            clean_long_by_seed[seed],
                            clean_clients_by_seed[seed],
                            reference,
                            direction,
                            top_k,
                        ).assign(calibration_seed=seed)
                    )
                train_base_by_k[top_k] = pd.concat(
                    train_parts, ignore_index=True
                )
                heldout_clean_base_by_k[top_k] = score_client_rounds(
                    clean_long_by_seed[heldout_seed],
                    clean_clients_by_seed[heldout_seed],
                    reference,
                    direction,
                    top_k,
                )
                for source in sources:
                    attack_base_by_source_k[(source, top_k)] = (
                        score_client_rounds(
                            attack_long[(source, heldout_seed)],
                            attack_clients[(source, heldout_seed)],
                            reference,
                            direction,
                            top_k,
                        )
                    )

            for top_k in TOP_K_VALUES:
                for beta in IMPACT_BETAS:
                    for decay in TEMPORAL_DECAYS:
                        train_scored_parts = []
                        for seed, g in train_base_by_k[top_k].groupby(
                            "calibration_seed", sort=False
                        ):
                            train_scored_parts.append(
                                add_impact_and_temporal_score(
                                    g.drop(columns=["calibration_seed"]),
                                    beta,
                                    decay,
                                    reference_weight,
                                ).assign(calibration_seed=int(seed))
                            )
                        train_scored = pd.concat(
                            train_scored_parts, ignore_index=True
                        )
                        heldout_clean_scored = (
                            add_impact_and_temporal_score(
                                heldout_clean_base_by_k[top_k],
                                beta,
                                decay,
                                reference_weight,
                            )
                        )

                        attack_scored_by_source = {
                            source: add_impact_and_temporal_score(
                                attack_base_by_source_k[(source, top_k)],
                                beta,
                                decay,
                                reference_weight,
                            )
                            for source in sources
                        }

                        for quantile in CLEAN_QUANTILES:
                            threshold = q_higher(
                                train_scored[
                                    "temporal_conformal_score"
                                ].to_numpy(dtype=float),
                                quantile,
                            )
                            clean_candidate = (
                                heldout_clean_scored[
                                    "temporal_conformal_score"
                                ].astype(float) > threshold
                            )
                            clean_current = heldout_clean_scored[
                                "flagged"
                            ].astype(bool)

                            for fusion in FUSIONS:
                                if fusion == "candidate_only":
                                    clean_flags = clean_candidate
                                elif fusion == "or_current_detector":
                                    clean_flags = (
                                        clean_candidate | clean_current
                                    )
                                else:
                                    raise ValueError(fusion)

                                threshold_rows.append({
                                    "heldout_seed": heldout_seed,
                                    "calibration_seeds": "|".join(
                                        str(x) for x in train_seeds
                                    ),
                                    "direction": direction,
                                    "top_k": top_k,
                                    "temporal_decay": decay,
                                    "impact_beta": beta,
                                    "clean_quantile": quantile,
                                    "fusion": fusion,
                                    "threshold": threshold,
                                    "reference_weight_median":
                                        reference_weight,
                                    "heldout_clean_fpr":
                                        safe_rate(clean_flags),
                                    "attack_labels_used_for_threshold":
                                        False,
                                })

                                for source in sources:
                                    attacked = attack_scored_by_source[
                                        source
                                    ]
                                    labels = attacked[
                                        "actual_malicious"
                                    ].astype(bool)
                                    candidate = (
                                        attacked[
                                            "temporal_conformal_score"
                                        ].astype(float) > threshold
                                    )
                                    current = attacked[
                                        "flagged"
                                    ].astype(bool)
                                    if fusion == "candidate_only":
                                        flags = candidate
                                    else:
                                        flags = candidate | current
                                    benign = ~labels
                                    evaluation_rows.append({
                                        "candidate_id": candidate_id(
                                            direction,
                                            top_k,
                                            decay,
                                            beta,
                                            quantile,
                                            fusion,
                                        ),
                                        "heldout_seed": heldout_seed,
                                        "source_class": source,
                                        "direction": direction,
                                        "top_k": top_k,
                                        "temporal_decay": decay,
                                        "impact_beta": beta,
                                        "clean_quantile": quantile,
                                        "fusion": fusion,
                                        "threshold": threshold,
                                        "heldout_clean_fpr":
                                            safe_rate(clean_flags),
                                        "malicious_recall":
                                            safe_rate(flags[labels]),
                                        "benign_fpr":
                                            safe_rate(flags[benign]),
                                        "flagged_malicious_client_rounds":
                                            int((flags & labels).sum()),
                                        "missed_malicious_client_rounds":
                                            int(((~flags) & labels).sum()),
                                        "false_positive_client_rounds":
                                            int((flags & benign).sum()),
                                        "attack_labels_used_for_threshold":
                                            False,
                                    })

                            # Save only one compact diagnostic configuration
                            # per heldout seed/direction to limit output size.
                            if (
                                top_k == 4
                                and beta == 0.5
                                and decay == 0.65
                                and quantile == 0.975
                            ):
                                compact = heldout_clean_scored[[
                                    "monitoring_round", "client_id",
                                    "actual_malicious",
                                    "aggregation_weight",
                                    "flagged",
                                    "cell_surprisal_topk_mean",
                                    "impact_score",
                                    "temporal_conformal_score",
                                ]].copy()
                                compact["heldout_seed"] = heldout_seed
                                compact["dataset"] = "clean"
                                compact["source_class"] = "Clean"
                                compact["direction"] = direction
                                compact_score_rows.append(compact)
                                for source in sources:
                                    ag = attack_scored_by_source[source][[
                                        "monitoring_round", "client_id",
                                        "actual_malicious",
                                        "aggregation_weight",
                                        "flagged",
                                        "cell_surprisal_topk_mean",
                                        "impact_score",
                                        "temporal_conformal_score",
                                    ]].copy()
                                    ag["heldout_seed"] = heldout_seed
                                    ag["dataset"] = "attacked"
                                    ag["source_class"] = source
                                    ag["direction"] = direction
                                    compact_score_rows.append(ag)

    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(
        tables / "v316_leave_one_seed_out_thresholds.csv",
        index=False,
    )
    evaluation = pd.DataFrame(evaluation_rows)
    evaluation.to_csv(
        tables / "v316_source_seed_candidate_evaluation.csv",
        index=False,
    )
    if compact_score_rows:
        pd.concat(compact_score_rows, ignore_index=True).to_csv(
            tables / "v316_compact_score_diagnostics.csv",
            index=False,
        )

    source_summary = (
        evaluation.groupby(
            [
                "candidate_id", "source_class", "direction", "top_k",
                "temporal_decay", "impact_beta", "clean_quantile",
                "fusion",
            ],
            as_index=False,
        )
        .agg(
            mean_malicious_recall=("malicious_recall", "mean"),
            minimum_seed_recall=("malicious_recall", "min"),
            mean_benign_fpr=("benign_fpr", "mean"),
            maximum_seed_fpr=("benign_fpr", "max"),
            mean_heldout_clean_fpr=("heldout_clean_fpr", "mean"),
            maximum_heldout_clean_fpr=("heldout_clean_fpr", "max"),
            seed_count=("heldout_seed", "nunique"),
        )
    )
    source_summary.to_csv(
        tables / "v316_candidate_source_summary.csv",
        index=False,
    )

    aggregate = (
        evaluation.groupby(
            [
                "candidate_id", "direction", "top_k", "temporal_decay",
                "impact_beta", "clean_quantile", "fusion",
            ],
            as_index=False,
        )
        .agg(
            mean_malicious_recall=("malicious_recall", "mean"),
            minimum_source_seed_recall=("malicious_recall", "min"),
            mean_attack_benign_fpr=("benign_fpr", "mean"),
            maximum_source_seed_fpr=("benign_fpr", "max"),
            mean_heldout_clean_fpr=("heldout_clean_fpr", "mean"),
            maximum_heldout_clean_fpr=("heldout_clean_fpr", "max"),
            source_seed_count=("source_class", "size"),
        )
    )
    recall_wide = source_summary.pivot_table(
        index=[
            "candidate_id", "direction", "top_k", "temporal_decay",
            "impact_beta", "clean_quantile", "fusion",
        ],
        columns="source_class",
        values="mean_malicious_recall",
    ).reset_index()
    aggregate = aggregate.merge(
        recall_wide,
        on=[
            "candidate_id", "direction", "top_k", "temporal_decay",
            "impact_beta", "clean_quantile", "fusion",
        ],
        how="left",
        validate="one_to_one",
    )

    aggregate["passes_mean_recall_090"] = (
        aggregate["mean_malicious_recall"] >= 0.90
    )
    aggregate["passes_min_source_seed_recall_050"] = (
        aggregate["minimum_source_seed_recall"] >= 0.50
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
    aggregate["passes_mean_attack_fpr_005"] = (
        aggregate["mean_attack_benign_fpr"] <= 0.05
    )
    aggregate["passes_max_attack_fpr_010"] = (
        aggregate["maximum_source_seed_fpr"] <= 0.10
    )
    aggregate["passes_mean_clean_fpr_005"] = (
        aggregate["mean_heldout_clean_fpr"] <= 0.05
    )
    aggregate["passes_max_clean_fpr_010"] = (
        aggregate["maximum_heldout_clean_fpr"] <= 0.10
    )
    criteria = [
        "passes_mean_recall_090",
        "passes_min_source_seed_recall_050",
        "passes_bruteforce_recall_075",
        "passes_web_recall_075",
        "passes_dos_recall_095",
        "passes_mean_attack_fpr_005",
        "passes_max_attack_fpr_010",
        "passes_mean_clean_fpr_005",
        "passes_max_clean_fpr_010",
    ]
    aggregate["passes_all_predefined_criteria"] = aggregate[
        criteria
    ].all(axis=1)
    aggregate["selected_as_v316_candidate"] = False

    passing = aggregate[
        aggregate["passes_all_predefined_criteria"]
    ].copy()
    selected = None
    if len(passing):
        passing["fusion_penalty"] = (
            passing["fusion"].eq("or_current_detector").astype(int)
        )
        passing = passing.sort_values(
            [
                "mean_attack_benign_fpr",
                "mean_heldout_clean_fpr",
                "minimum_source_seed_recall",
                "mean_malicious_recall",
                "fusion_penalty",
                "candidate_id",
            ],
            ascending=[True, True, False, False, True, True],
        )
        selected = passing.iloc[0]
        aggregate.loc[
            aggregate["candidate_id"].eq(selected["candidate_id"]),
            "selected_as_v316_candidate",
        ] = True

    aggregate.to_csv(
        tables / "v316_candidate_aggregate_summary.csv",
        index=False,
    )

    decision = {
        "experiment_version": "3.16",
        "stage": "impact_aware_temporal_conformal_development_screen",
        "sources": sources,
        "seeds": seeds,
        "leave_one_seed_out_clean_calibration": True,
        "non_oracle_transition_scan": True,
        "stable_client_ids_used_for_temporal_accumulation": True,
        "aggregation_weight_used_as_server_observable_impact_proxy": True,
        "training_rerun": False,
        "reconstruction_modified": False,
        "current_detector_modified": False,
        "test_sets_accessed": False,
        "attack_outcomes_used_for_threshold_calibration": False,
        "attack_outcomes_used_for_development_candidate_selection": True,
        "candidate_selected": selected is not None,
        "selected_candidate_id": (
            None if selected is None else str(selected["candidate_id"])
        ),
        "final_untouched_seed_validation_required": True,
        "recommended_next_stage": (
            "implement and freeze the selected detector, then validate on "
            "new untouched seeds and broader attack families"
            if selected is not None
            else
            "transition-only non-oracle detection remains insufficient; "
            "move to update-space representation learning or one-class "
            "conformal detection without weakening criteria"
        ),
    }
    with (out / "v316_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    pd.DataFrame([{
        "candidate_selected": decision["candidate_selected"],
        "selected_candidate_id": decision["selected_candidate_id"],
        "current_detector_modified": False,
        "method_reopened": False,
        "test_sets_accessed": False,
    }]).to_csv(tables / "v316_decision.csv", index=False)

    # Figures.
    plot = aggregate[
        aggregate["fusion"].eq("or_current_detector")
    ].copy()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for direction, g in plot.groupby("direction"):
        ax.scatter(
            g["mean_attack_benign_fpr"],
            g["mean_malicious_recall"],
            alpha=0.65,
            label=direction,
        )
    ax.axhline(0.90, linestyle="--", linewidth=1)
    ax.axvline(0.05, linestyle="--", linewidth=1)
    ax.set_xlabel("Mean benign FPR during attack")
    ax.set_ylabel("Mean malicious recall")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.set_title("V3.16 impact-aware temporal conformal frontier")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "impact_temporal_conformal_frontier")

    top = aggregate.sort_values(
        [
            "passes_all_predefined_criteria",
            "mean_malicious_recall",
            "mean_attack_benign_fpr",
        ],
        ascending=[False, False, True],
    ).head(20)
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(top))
    ax.bar(x, top["mean_malicious_recall"])
    ax.set_xticks(x)
    ax.set_xticklabels(
        top["candidate_id"], rotation=75, ha="right", fontsize=7
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean malicious recall")
    ax.set_title("Top V3.16 non-oracle candidate configurations")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures / "top_candidate_recall")

    print("V3.16 impact-aware temporal conformal screen complete")
    print()
    print("TOP CANDIDATES")
    display = aggregate.sort_values(
        [
            "passes_all_predefined_criteria",
            "mean_malicious_recall",
            "mean_attack_benign_fpr",
        ],
        ascending=[False, False, True],
    )
    cols = [
        "candidate_id", "direction", "top_k", "temporal_decay",
        "impact_beta", "clean_quantile", "fusion",
        "mean_malicious_recall", "minimum_source_seed_recall",
        "mean_attack_benign_fpr", "maximum_source_seed_fpr",
        "mean_heldout_clean_fpr", "maximum_heldout_clean_fpr",
        "BruteForce", "Web-Based", "DoS",
        "passes_all_predefined_criteria",
        "selected_as_v316_candidate",
    ]
    print(display[cols].head(25).to_string(index=False))
    print()
    print("V3.16 DECISION")
    print(pd.read_csv(
        tables / "v316_decision.csv"
    ).to_string(index=False))
    print()
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

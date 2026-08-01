#!/usr/bin/env python3
"""V3.17 multiview one-class detector development screen.

Analysis-only stage over completed V3.12.3 clean and V3.13.2 attacked
continuations. No federated training is rerun.

The candidates are non-oracle. They use:
- the full 8x8 prediction-transition deviation vector,
- actual client update norm already captured by the server,
- stable-client update-norm normalization,
- leave-one-seed-out clean calibration,
- nested clean-only threshold calibration.

Attack labels are used only to compare development candidates. Test sets are
not accessed. Any selected candidate still requires a new untouched-seed run.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


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
REPRESENTATIONS = [
    "profile_delta",
    "profile_delta_update_global",
    "profile_delta_update_clientnorm",
    "profile_consensus_update_clientnorm",
]
PCA_DIMS = [8, 16, 32]
DETECTORS = ["mahalanobis", "knn5", "knn10"]
CLEAN_QUANTILES = [0.95, 0.975, 0.99]
FUSIONS = ["candidate_only", "or_current_detector"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v3123-root", required=True, type=Path)
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


def safe_rate(values: pd.Series) -> float:
    return float(values.astype(bool).mean()) if len(values) else float("nan")


def root_for(source: str, dos_root: Path, remaining_root: Path) -> Path:
    return dos_root if source == "DoS" else remaining_root


def clean_paths(v3123_root: Path, seed: int) -> Tuple[Path, Path]:
    base = v3123_root / f"seed_{seed}" / "trusted_clean" / "tables"
    return (
        base / "reconstruction_transition_signature_long.csv",
        base / "reconstruction_client_rows.csv",
    )


def attack_paths(
    root: Path, source: str, seed: int
) -> Tuple[Path, Path]:
    base = (
        root / "runs" / SOURCE_TO_SLUG[source] / f"seed_{seed}"
        / "trusted_reconstruction" / "tables"
    )
    return (
        base / "reconstruction_transition_signature_long.csv",
        base / "reconstruction_client_rows.csv",
    )


def load_client_vectors(
    signature_path: Path,
    client_path: Path,
    seed: int,
    source_class: str,
    dataset: str,
) -> pd.DataFrame:
    sig = pd.read_csv(require(signature_path))
    clients = pd.read_csv(require(client_path))

    sig_needed = {
        "monitoring_round", "client_id", "source_id", "target_id",
        "local_probability", "round_consensus_probability",
        "frozen_profile_probability",
    }
    client_needed = {
        "monitoring_round", "client_id", "actual_malicious",
        "actual_update_norm", "aggregation_weight", "flagged",
    }
    missing_sig = sorted(sig_needed - set(sig.columns))
    missing_clients = sorted(client_needed - set(clients.columns))
    if missing_sig:
        raise RuntimeError(
            f"Missing signature columns {missing_sig}: {signature_path}"
        )
    if missing_clients:
        raise RuntimeError(
            f"Missing client columns {missing_clients}: {client_path}"
        )

    sig = sig.copy()
    sig["cell_id"] = (
        sig["source_id"].astype(int) * 8
        + sig["target_id"].astype(int)
    )
    sig["profile_delta"] = (
        sig["local_probability"].astype(float)
        - sig["frozen_profile_probability"].astype(float)
    )
    sig["consensus_delta"] = (
        sig["local_probability"].astype(float)
        - sig["round_consensus_probability"].astype(float)
    )

    key = ["monitoring_round", "client_id"]
    profile = sig.pivot_table(
        index=key,
        columns="cell_id",
        values="profile_delta",
        aggfunc="first",
    )
    consensus = sig.pivot_table(
        index=key,
        columns="cell_id",
        values="consensus_delta",
        aggfunc="first",
    )
    expected = list(range(64))
    missing_profile = sorted(set(expected) - set(profile.columns))
    missing_consensus = sorted(set(expected) - set(consensus.columns))
    if missing_profile or missing_consensus:
        raise RuntimeError(
            "Incomplete 8x8 signature: "
            f"profile missing={missing_profile}, "
            f"consensus missing={missing_consensus}"
        )
    profile = profile[expected].copy()
    consensus = consensus[expected].copy()
    profile.columns = [f"profile_delta_{i:02d}" for i in expected]
    consensus.columns = [f"consensus_delta_{i:02d}" for i in expected]
    vectors = profile.join(consensus, how="inner").reset_index()

    keep = [
        "monitoring_round", "client_id", "actual_malicious",
        "actual_update_norm", "aggregation_weight", "flagged",
    ]
    c = clients[keep].drop_duplicates(key)
    vectors = vectors.merge(
        c, on=key, how="inner", validate="one_to_one"
    )
    if len(vectors) != 80:
        raise RuntimeError(
            f"Expected 80 client-round vectors, found {len(vectors)} "
            f"for seed={seed}, source={source_class}, dataset={dataset}"
        )
    vectors["seed"] = int(seed)
    vectors["source_class"] = source_class
    vectors["dataset"] = dataset
    return vectors


def client_norm_reference(train: pd.DataFrame) -> pd.DataFrame:
    x = train.copy()
    x["log_update_norm"] = np.log1p(
        x["actual_update_norm"].astype(float)
    )
    rows = []
    for client_id, g in x.groupby("client_id", sort=True):
        values = g["log_update_norm"].to_numpy(dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = max(1.4826 * mad, float((q75 - q25) / 1.349), 1e-6)
        rows.append({
            "client_id": int(client_id),
            "client_log_norm_median": median,
            "client_log_norm_scale": scale,
        })
    return pd.DataFrame(rows)


def build_matrix(
    data: pd.DataFrame,
    representation: str,
    norm_reference: pd.DataFrame,
) -> np.ndarray:
    profile_cols = [f"profile_delta_{i:02d}" for i in range(64)]
    consensus_cols = [f"consensus_delta_{i:02d}" for i in range(64)]
    log_norm = np.log1p(
        data["actual_update_norm"].to_numpy(dtype=float)
    ).reshape(-1, 1)

    if representation == "profile_delta":
        return data[profile_cols].to_numpy(dtype=float)
    if representation == "profile_delta_update_global":
        return np.hstack([
            data[profile_cols].to_numpy(dtype=float),
            log_norm,
        ])

    merged = data[["client_id"]].merge(
        norm_reference,
        on="client_id",
        how="left",
        validate="many_to_one",
    )
    if merged.isna().any().any():
        raise RuntimeError("Missing stable-client update norm reference")
    client_z = (
        (
            log_norm[:, 0]
            - merged["client_log_norm_median"].to_numpy(dtype=float)
        )
        / merged["client_log_norm_scale"].to_numpy(dtype=float)
    ).reshape(-1, 1)

    if representation == "profile_delta_update_clientnorm":
        return np.hstack([
            data[profile_cols].to_numpy(dtype=float),
            client_z,
        ])
    if representation == "profile_consensus_update_clientnorm":
        return np.hstack([
            data[profile_cols].to_numpy(dtype=float),
            data[consensus_cols].to_numpy(dtype=float),
            client_z,
        ])
    raise ValueError(representation)


@dataclass
class OneClassModel:
    scaler: RobustScaler
    pca: PCA
    detector_name: str
    detector: object

    def score(self, matrix: np.ndarray) -> np.ndarray:
        z = self.pca.transform(self.scaler.transform(matrix))
        if self.detector_name == "mahalanobis":
            centered = z - self.detector.location_
            return np.einsum(
                "ij,jk,ik->i",
                centered,
                self.detector.precision_,
                centered,
            )
        distances, _ = self.detector.kneighbors(z)
        return distances[:, -1]


def fit_model(
    train_matrix: np.ndarray,
    pca_dim: int,
    detector_name: str,
) -> OneClassModel:
    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        unit_variance=True,
    )
    scaled = scaler.fit_transform(train_matrix)
    components = min(
        int(pca_dim),
        int(scaled.shape[1]),
        int(scaled.shape[0] - 1),
    )
    if components < 2:
        raise RuntimeError("Insufficient rows for PCA")
    pca = PCA(
        n_components=components,
        svd_solver="full",
        random_state=317,
    )
    z = pca.fit_transform(scaled)

    if detector_name == "mahalanobis":
        detector = LedoitWolf(assume_centered=False).fit(z)
    elif detector_name in {"knn5", "knn10"}:
        k = 5 if detector_name == "knn5" else 10
        detector = NearestNeighbors(
            n_neighbors=min(k, len(z)),
            metric="euclidean",
        ).fit(z)
    else:
        raise ValueError(detector_name)
    return OneClassModel(scaler, pca, detector_name, detector)


def candidate_id(
    representation: str,
    pca_dim: int,
    detector: str,
    quantile: float,
    fusion: str,
) -> str:
    return (
        f"{representation}_pca{pca_dim}_{detector}_"
        f"q{quantile:g}_{fusion}"
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    a = parse_args()
    v3123_root = a.v3123_root.expanduser().resolve()
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
    if len(seeds) < 4:
        raise ValueError("V3.17 requires four development seeds")

    clean_by_seed: Dict[int, pd.DataFrame] = {}
    attack_by_source_seed: Dict[Tuple[str, int], pd.DataFrame] = {}

    for seed in seeds:
        sig_path, client_path = clean_paths(v3123_root, seed)
        clean_by_seed[seed] = load_client_vectors(
            sig_path, client_path, seed, "Clean", "clean"
        )
    for source in sources:
        root = root_for(source, dos_root, remaining_root)
        for seed in seeds:
            sig_path, client_path = attack_paths(root, source, seed)
            attack_by_source_seed[(source, seed)] = load_client_vectors(
                sig_path, client_path, seed, source, "attacked"
            )

    evaluation_rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []

    for outer_seed in seeds:
        calibration_seeds = [s for s in seeds if s != outer_seed]
        outer_train = pd.concat(
            [clean_by_seed[s] for s in calibration_seeds],
            ignore_index=True,
        )
        outer_norm_ref = client_norm_reference(outer_train)

        for representation in REPRESENTATIONS:
            for pca_dim in PCA_DIMS:
                for detector_name in DETECTORS:
                    inner_scores = []
                    for inner_seed in calibration_seeds:
                        inner_train_seeds = [
                            s for s in calibration_seeds
                            if s != inner_seed
                        ]
                        inner_train = pd.concat(
                            [clean_by_seed[s] for s in inner_train_seeds],
                            ignore_index=True,
                        )
                        inner_valid = clean_by_seed[inner_seed]
                        inner_norm_ref = client_norm_reference(inner_train)
                        train_matrix = build_matrix(
                            inner_train,
                            representation,
                            inner_norm_ref,
                        )
                        valid_matrix = build_matrix(
                            inner_valid,
                            representation,
                            inner_norm_ref,
                        )
                        model = fit_model(
                            train_matrix, pca_dim, detector_name
                        )
                        scores = model.score(valid_matrix)
                        inner_scores.extend(scores.tolist())

                    final_model = fit_model(
                        build_matrix(
                            outer_train,
                            representation,
                            outer_norm_ref,
                        ),
                        pca_dim,
                        detector_name,
                    )
                    outer_clean = clean_by_seed[outer_seed]
                    outer_clean_scores = final_model.score(
                        build_matrix(
                            outer_clean,
                            representation,
                            outer_norm_ref,
                        )
                    )

                    attack_scores: Dict[str, np.ndarray] = {}
                    for source in sources:
                        attack_scores[source] = final_model.score(
                            build_matrix(
                                attack_by_source_seed[(source, outer_seed)],
                                representation,
                                outer_norm_ref,
                            )
                        )

                    for quantile in CLEAN_QUANTILES:
                        threshold = q_higher(
                            np.asarray(inner_scores, dtype=float),
                            quantile,
                        )
                        clean_candidate = outer_clean_scores > threshold
                        clean_current = (
                            outer_clean["flagged"].astype(bool).to_numpy()
                        )

                        for fusion in FUSIONS:
                            if fusion == "candidate_only":
                                clean_flags = clean_candidate
                            else:
                                clean_flags = (
                                    clean_candidate | clean_current
                                )

                            threshold_rows.append({
                                "outer_heldout_seed": outer_seed,
                                "calibration_seeds": "|".join(
                                    str(s) for s in calibration_seeds
                                ),
                                "representation": representation,
                                "pca_dim": pca_dim,
                                "detector": detector_name,
                                "clean_quantile": quantile,
                                "fusion": fusion,
                                "threshold": threshold,
                                "inner_clean_score_count":
                                    len(inner_scores),
                                "outer_clean_fpr":
                                    float(clean_flags.mean()),
                                "attack_labels_used_for_threshold":
                                    False,
                            })

                            for source in sources:
                                attacked = attack_by_source_seed[
                                    (source, outer_seed)
                                ]
                                labels = (
                                    attacked["actual_malicious"]
                                    .astype(bool)
                                    .to_numpy()
                                )
                                candidate = (
                                    attack_scores[source] > threshold
                                )
                                current = (
                                    attacked["flagged"]
                                    .astype(bool)
                                    .to_numpy()
                                )
                                flags = (
                                    candidate
                                    if fusion == "candidate_only"
                                    else candidate | current
                                )
                                benign = ~labels
                                evaluation_rows.append({
                                    "candidate_id": candidate_id(
                                        representation,
                                        pca_dim,
                                        detector_name,
                                        quantile,
                                        fusion,
                                    ),
                                    "outer_heldout_seed": outer_seed,
                                    "source_class": source,
                                    "representation": representation,
                                    "pca_dim": pca_dim,
                                    "detector": detector_name,
                                    "clean_quantile": quantile,
                                    "fusion": fusion,
                                    "threshold": threshold,
                                    "outer_clean_fpr":
                                        float(clean_flags.mean()),
                                    "malicious_recall":
                                        float(flags[labels].mean()),
                                    "benign_fpr":
                                        float(flags[benign].mean()),
                                    "flagged_malicious_client_rounds":
                                        int((flags & labels).sum()),
                                    "missed_malicious_client_rounds":
                                        int(((~flags) & labels).sum()),
                                    "false_positive_client_rounds":
                                        int((flags & benign).sum()),
                                    "attack_labels_used_for_threshold":
                                        False,
                                })

    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(
        tables / "v317_nested_clean_thresholds.csv", index=False
    )
    evaluation = pd.DataFrame(evaluation_rows)
    evaluation.to_csv(
        tables / "v317_source_seed_candidate_evaluation.csv",
        index=False,
    )

    source_summary = (
        evaluation.groupby(
            [
                "candidate_id", "source_class", "representation",
                "pca_dim", "detector", "clean_quantile", "fusion",
            ],
            as_index=False,
        )
        .agg(
            mean_malicious_recall=("malicious_recall", "mean"),
            minimum_seed_recall=("malicious_recall", "min"),
            mean_benign_fpr=("benign_fpr", "mean"),
            maximum_seed_fpr=("benign_fpr", "max"),
            mean_outer_clean_fpr=("outer_clean_fpr", "mean"),
            maximum_outer_clean_fpr=("outer_clean_fpr", "max"),
            seed_count=("outer_heldout_seed", "nunique"),
        )
    )
    source_summary.to_csv(
        tables / "v317_candidate_source_summary.csv", index=False
    )

    aggregate = (
        evaluation.groupby(
            [
                "candidate_id", "representation", "pca_dim",
                "detector", "clean_quantile", "fusion",
            ],
            as_index=False,
        )
        .agg(
            mean_malicious_recall=("malicious_recall", "mean"),
            minimum_source_seed_recall=("malicious_recall", "min"),
            mean_attack_benign_fpr=("benign_fpr", "mean"),
            maximum_source_seed_fpr=("benign_fpr", "max"),
            mean_outer_clean_fpr=("outer_clean_fpr", "mean"),
            maximum_outer_clean_fpr=("outer_clean_fpr", "max"),
            source_seed_count=("source_class", "size"),
        )
    )
    recall_wide = source_summary.pivot_table(
        index=[
            "candidate_id", "representation", "pca_dim",
            "detector", "clean_quantile", "fusion",
        ],
        columns="source_class",
        values="mean_malicious_recall",
    ).reset_index()
    aggregate = aggregate.merge(
        recall_wide,
        on=[
            "candidate_id", "representation", "pca_dim",
            "detector", "clean_quantile", "fusion",
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
        aggregate["mean_outer_clean_fpr"] <= 0.05
    )
    aggregate["passes_max_clean_fpr_010"] = (
        aggregate["maximum_outer_clean_fpr"] <= 0.10
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
    aggregate["selected_as_v317_candidate"] = False

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
                "mean_outer_clean_fpr",
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
            "selected_as_v317_candidate",
        ] = True

    aggregate.to_csv(
        tables / "v317_candidate_aggregate_summary.csv",
        index=False,
    )

    decision = {
        "experiment_version": "3.17",
        "stage": "multiview_one_class_development_screen",
        "representations": REPRESENTATIONS,
        "pca_dimensions": PCA_DIMS,
        "detectors": DETECTORS,
        "clean_quantiles": CLEAN_QUANTILES,
        "nested_leave_one_seed_out_clean_calibration": True,
        "stable_client_update_norm_normalization": True,
        "full_transition_vector_used": True,
        "full_parameter_update_vector_used": False,
        "federated_training_rerun": False,
        "reconstruction_modified": False,
        "current_detector_modified": False,
        "test_sets_accessed": False,
        "attack_outcomes_used_for_threshold_calibration": False,
        "attack_outcomes_used_for_development_candidate_selection": True,
        "candidate_selected": selected is not None,
        "selected_candidate_id": (
            None if selected is None else str(selected["candidate_id"])
        ),
        "untouched_seed_validation_required_after_selection": True,
        "recommended_next_stage": (
            "implement and freeze selected detector, then run new untouched "
            "seeds and broader attack families"
            if selected is not None
            else
            "compact transition plus update-norm representations are "
            "insufficient; stop screening on these development attacks and "
            "either capture full update-vector sketches under a preregistered "
            "protocol or proceed with robust aggregation baselines"
        ),
    }
    with (out / "v317_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    pd.DataFrame([{
        "candidate_selected": decision["candidate_selected"],
        "selected_candidate_id": decision["selected_candidate_id"],
        "current_detector_modified": False,
        "method_reopened": False,
        "test_sets_accessed": False,
    }]).to_csv(tables / "v317_decision.csv", index=False)

    plot = aggregate[
        aggregate["fusion"].eq("or_current_detector")
    ].copy()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for detector, g in plot.groupby("detector"):
        ax.scatter(
            g["mean_attack_benign_fpr"],
            g["mean_malicious_recall"],
            alpha=0.65,
            label=detector,
        )
    ax.axhline(0.90, linestyle="--", linewidth=1)
    ax.axvline(0.05, linestyle="--", linewidth=1)
    ax.set_xlabel("Mean benign FPR during attack")
    ax.set_ylabel("Mean malicious recall")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.set_title("V3.17 multiview one-class candidate frontier")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "multiview_oneclass_frontier")

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
    ax.set_title("Top V3.17 multiview one-class candidates")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures / "top_multiview_candidates")

    print("V3.17 multiview one-class screen complete")
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
        "candidate_id", "representation", "pca_dim", "detector",
        "clean_quantile", "fusion", "mean_malicious_recall",
        "minimum_source_seed_recall", "mean_attack_benign_fpr",
        "maximum_source_seed_fpr", "mean_outer_clean_fpr",
        "maximum_outer_clean_fpr", "BruteForce", "Web-Based", "DoS",
        "passes_all_predefined_criteria",
        "selected_as_v317_candidate",
    ]
    print(display[cols].head(30).to_string(index=False))
    print()
    print("V3.17 DECISION")
    print(pd.read_csv(
        tables / "v317_decision.csv"
    ).to_string(index=False))
    print()
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

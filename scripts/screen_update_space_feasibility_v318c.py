#!/usr/bin/env python3
"""V3.18C non-oracle update-space detector feasibility screen.

This stage uses only the exactly equivalent V3.18B seed-7 smoke captures:
Clean, DoS->Benign, BruteForce->Benign, and Web-Based->Benign.

The screen is deliberately offline and developmental:
- thresholds are calibrated only from clean captures,
- attack labels are used only to compare development candidates,
- no natural or diagnostic test set is accessed,
- no federated training or aggregation is rerun,
- no candidate is treated as final without untouched multiseed validation.

Full updates are represented by a deterministic sparse CountSketch, exact
classifier/tail slices, and layer-wise profiles. Clean-only nested
leave-one-round-out calibration prevents attack-label threshold fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


SCENARIO_ORDER = [
    "Clean",
    "DoS->Benign",
    "BruteForce->Benign",
    "Web-Based->Benign",
]
ATTACK_SOURCES = ["DoS", "BruteForce", "Web-Based"]
PCA_DIMS = [8, 16, 32]
DETECTORS = ["mahalanobis", "knn5", "client_l2", "client_cosine"]
QUANTILES = [0.95, 0.975, 0.99]
FUSIONS = ["candidate_only", "or_current_detector"]
SKETCH_DIM = 256
SKETCH_SEED = 3180317


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--sketch-dim", type=int, default=SKETCH_DIM)
    p.add_argument("--sketch-seed", type=int, default=SKETCH_SEED)
    p.add_argument("--force-reextract", action="store_true")
    return p.parse_args()


def slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return value or "scenario"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def q_higher(values: Sequence[float], q: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    try:
        return float(np.quantile(x, q, method="higher"))
    except TypeError:
        return float(np.quantile(x, q, interpolation="higher"))


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def verify_v318b_decision(root: Path) -> pd.DataFrame:
    decision_path = root / "tables" / "v318b_decision.csv"
    if not decision_path.exists():
        raise FileNotFoundError(decision_path)
    decision = pd.read_csv(decision_path)
    if len(decision) != 1:
        raise RuntimeError("V3.18B decision table must contain exactly one row")
    row = decision.iloc[0]
    if not bool(row["all_capture_branches_exactly_equivalent"]):
        raise RuntimeError("V3.18B exact-equivalence requirement did not pass")
    if bool(row["training_or_aggregation_modified"]):
        raise RuntimeError("V3.18B reports modified training or aggregation")
    if bool(row["test_sets_accessed"]):
        raise RuntimeError("V3.18B reports test-set access")
    return decision


def load_manifest(root: Path) -> pd.DataFrame:
    path = root / "tables" / "v318b_smoke_capture_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = pd.read_csv(path)
    missing = sorted(set(SCENARIO_ORDER) - set(manifest["scenario"].astype(str)))
    if missing:
        raise RuntimeError(f"Missing required V3.18B scenarios: {missing}")
    if not manifest["exact_equivalence"].astype(bool).all():
        raise RuntimeError("At least one V3.18B branch is not exactly equivalent")
    return manifest


def build_projection(
    parameter_count: int,
    sketch_dim: int,
    seed: int,
) -> sparse.csr_matrix:
    if sketch_dim < 16:
        raise ValueError("Sketch dimension is too small")
    index = np.arange(parameter_count, dtype=np.uint64)
    mixed = (
        index * np.uint64(11400714819323198485)
        + np.uint64(seed)
    )
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(13787848793156543929)
    mixed ^= mixed >> np.uint64(27)
    buckets = np.asarray(mixed % np.uint64(sketch_dim), dtype=np.int32)
    signs = np.where(
        ((mixed >> np.uint64(63)) & np.uint64(1)) == 0,
        -1.0,
        1.0,
    ).astype(np.float32)
    columns = np.arange(parameter_count, dtype=np.int32)
    projection = sparse.csr_matrix(
        (signs, (buckets, columns)),
        shape=(sketch_dim, parameter_count),
        dtype=np.float32,
    )
    return projection


def selected_indices(
    layout: pd.DataFrame,
    prefixes: Sequence[str],
) -> np.ndarray:
    pieces = []
    for _, row in layout.iterrows():
        if not bool(row["included_in_update_vector"]):
            continue
        name = str(row["parameter_name"])
        if any(name.startswith(prefix) for prefix in prefixes):
            start = int(row["start_index"])
            stop = int(row["stop_index_exclusive"])
            pieces.append(np.arange(start, stop, dtype=np.int64))
    if not pieces:
        raise RuntimeError(f"No parameters matched prefixes {prefixes}")
    return np.concatenate(pieces)


def layer_feature_matrix(
    layer_path: Path,
    client_ids: np.ndarray,
    parameter_names: Sequence[str],
) -> np.ndarray:
    table = pd.read_csv(layer_path)
    metrics = ["l2_norm", "mean_abs", "max_abs", "signed_mean"]
    rows = []
    for client_id in client_ids:
        client = table[table["client_id"].astype(int) == int(client_id)].copy()
        client = client.set_index("parameter_name")
        values = []
        for metric in metrics:
            for parameter in parameter_names:
                if parameter not in client.index:
                    raise RuntimeError(
                        f"Missing layer {parameter} in {layer_path}"
                    )
                values.append(float(client.loc[parameter, metric]))
        rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def round_geometry(
    sketch_direction: np.ndarray,
    update_norm: np.ndarray,
) -> Dict[str, np.ndarray]:
    x = normalize_rows(sketch_direction)
    center = np.median(x, axis=0)
    center_norm = float(np.linalg.norm(center))
    if center_norm > 1e-12:
        center = center / center_norm
    l2_to_median = np.linalg.norm(x - center[None, :], axis=1)
    cosine_to_median = 1.0 - np.clip(x @ center, -1.0, 1.0)

    similarities = np.clip(x @ x.T, -1.0, 1.0)
    distances = 1.0 - similarities
    np.fill_diagonal(distances, np.inf)
    nearest = np.sort(distances, axis=1)[:, :5]
    knn5_cosine = np.mean(nearest, axis=1)

    log_norm = np.log1p(np.asarray(update_norm, dtype=float))
    median = float(np.median(log_norm))
    mad = float(np.median(np.abs(log_norm - median)))
    scale = max(1.4826 * mad, 1e-8)
    norm_abs_robust_z = np.abs((log_norm - median) / scale)

    return {
        "round_median_l2": l2_to_median,
        "round_median_cosine": cosine_to_median,
        "round_knn5_cosine": knn5_cosine,
        "round_norm_abs_robust_z": norm_abs_robust_z,
    }


def cache_paths(features_dir: Path, scenario: str) -> Tuple[Path, Path]:
    base = features_dir / f"{slug(scenario)}_v318c_features"
    return base.with_suffix(".npz"), base.with_suffix(".csv")


def extract_scenario(
    scenario: str,
    capture_dir: Path,
    projection: sparse.csr_matrix,
    layout: pd.DataFrame,
    head_idx: np.ndarray,
    tail_idx: np.ndarray,
    layer_names: Sequence[str],
    features_dir: Path,
    force: bool,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    npz_path, csv_path = cache_paths(features_dir, scenario)
    if npz_path.exists() and csv_path.exists() and not force:
        meta = pd.read_csv(csv_path)
        with np.load(npz_path, allow_pickle=False) as data:
            representations = {
                key: np.asarray(data[key], dtype=np.float64)
                for key in data.files
            }
        return meta, representations

    index_path = capture_dir / "tables" / "v318b_update_capture_index.csv"
    anchor_path = (
        capture_dir / "tables" / "continuation_client_anchor_scores.csv"
    )
    index = pd.read_csv(index_path).sort_values("monitoring_round")
    anchors = pd.read_csv(anchor_path)
    if len(index) != 4:
        raise RuntimeError(f"{scenario} does not contain four captured rounds")

    metadata_rows: List[Dict[str, object]] = []
    representation_rows: Dict[str, List[np.ndarray]] = {
        "head_raw": [],
        "head_direction": [],
        "tail_direction": [],
        "full_sketch_raw": [],
        "full_sketch_direction": [],
        "layer_profile": [],
        "fused_update_views": [],
    }
    geometry_rows: Dict[str, List[np.ndarray]] = {
        "round_median_l2": [],
        "round_median_cosine": [],
        "round_knn5_cosine": [],
        "round_norm_abs_robust_z": [],
    }

    for _, capture in index.iterrows():
        monitoring_round = int(capture["monitoring_round"])
        global_round = int(capture["global_round"])
        matrix_path = capture_dir / str(capture["update_matrix_file"])
        manifest_path = (
            capture_dir / "update_artifacts"
            / f"round_{monitoring_round:02d}_client_manifest.csv"
        )
        layer_path = (
            capture_dir / "update_artifacts"
            / f"round_{monitoring_round:02d}_layer_summary.csv"
        )
        if sha256_file(matrix_path) != str(capture["update_matrix_sha256"]):
            raise RuntimeError(f"Update matrix hash mismatch: {matrix_path}")

        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        if matrix.shape[0] != 20:
            raise RuntimeError(f"Expected 20 clients in {matrix_path}")
        if matrix.shape[1] != projection.shape[1]:
            raise RuntimeError("Projection and update parameter counts differ")
        client_manifest = pd.read_csv(manifest_path).sort_values("client_id")
        client_ids = client_manifest["client_id"].to_numpy(dtype=int)
        update_norm = client_manifest["full_update_l2_norm"].to_numpy(dtype=float)

        sketch = (
            projection @ np.asarray(matrix, dtype=np.float32).T
        ).T.astype(np.float64)
        sketch /= math.sqrt(max(matrix.shape[1] / projection.shape[0], 1.0))
        head = np.asarray(matrix[:, head_idx], dtype=np.float64)
        tail = np.asarray(matrix[:, tail_idx], dtype=np.float64)
        layer = layer_feature_matrix(
            layer_path, client_ids, layer_names
        )

        head_direction = normalize_rows(head)
        tail_direction = normalize_rows(tail)
        sketch_direction = normalize_rows(sketch)
        log_norm = np.log1p(update_norm).reshape(-1, 1)

        representation_rows["head_raw"].append(head)
        representation_rows["head_direction"].append(head_direction)
        representation_rows["tail_direction"].append(tail_direction)
        representation_rows["full_sketch_raw"].append(
            np.hstack([sketch, log_norm])
        )
        representation_rows["full_sketch_direction"].append(
            sketch_direction
        )
        representation_rows["layer_profile"].append(layer)
        representation_rows["fused_update_views"].append(
            np.hstack([
                sketch_direction,
                head_direction,
                layer,
                log_norm,
            ])
        )

        geometry = round_geometry(sketch_direction, update_norm)
        for name, values in geometry.items():
            geometry_rows[name].append(values.reshape(-1, 1))

        round_anchor = anchors[
            anchors["monitoring_round"].astype(int) == monitoring_round
        ].sort_values("client_id")
        if len(round_anchor) != 20:
            raise RuntimeError(
                f"Expected 20 anchor rows for {scenario}, round {monitoring_round}"
            )

        source = "Clean" if scenario == "Clean" else scenario.split("->")[0]
        for pos, client_id in enumerate(client_ids):
            metadata_rows.append({
                "scenario": scenario,
                "source_class": source,
                "monitoring_round": monitoring_round,
                "global_round": global_round,
                "client_id": int(client_id),
                "actual_malicious": bool(
                    client_manifest.iloc[pos]["actual_malicious"]
                ),
                "current_detector_flagged": bool(
                    round_anchor.iloc[pos]["flagged"]
                ),
                "client_samples": int(
                    client_manifest.iloc[pos]["client_samples"]
                ),
                "poisoned_rows": int(
                    client_manifest.iloc[pos]["poisoned_rows"]
                ),
                "full_update_l2_norm": float(update_norm[pos]),
            })

    meta = pd.DataFrame(metadata_rows)
    representations = {
        name: np.vstack(parts)
        for name, parts in representation_rows.items()
    }
    for name, parts in geometry_rows.items():
        representations[name] = np.vstack(parts)

    features_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **representations)
    meta.to_csv(csv_path, index=False)
    return meta, representations


@dataclass
class CleanOnlyModel:
    scaler: RobustScaler
    pca: PCA
    detector_name: str
    detector: object
    client_centers: Dict[int, np.ndarray]

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(x))

    def score(self, x: np.ndarray, client_ids: np.ndarray) -> np.ndarray:
        z = self.transform(x)
        if self.detector_name == "mahalanobis":
            centered = z - self.detector.location_
            return np.einsum(
                "ij,jk,ik->i",
                centered,
                self.detector.precision_,
                centered,
            )
        if self.detector_name == "knn5":
            distances, _ = self.detector.kneighbors(z)
            return distances[:, -1]

        scores = []
        for row, client_id in zip(z, client_ids):
            center = self.client_centers[int(client_id)]
            if self.detector_name == "client_l2":
                scores.append(float(np.linalg.norm(row - center)))
            elif self.detector_name == "client_cosine":
                denom = max(
                    float(np.linalg.norm(row) * np.linalg.norm(center)),
                    1e-12,
                )
                cosine = float(np.dot(row, center) / denom)
                scores.append(1.0 - float(np.clip(cosine, -1.0, 1.0)))
            else:
                raise ValueError(self.detector_name)
        return np.asarray(scores, dtype=float)


def fit_clean_model(
    x: np.ndarray,
    client_ids: np.ndarray,
    pca_dim: int,
    detector_name: str,
) -> CleanOnlyModel:
    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        unit_variance=True,
    )
    scaled = scaler.fit_transform(x)
    components = min(
        int(pca_dim),
        int(scaled.shape[1]),
        int(scaled.shape[0] - 1),
    )
    if components < 2:
        raise RuntimeError("Insufficient clean rows for PCA")
    pca = PCA(
        n_components=components,
        svd_solver="full",
        random_state=318,
    )
    z = pca.fit_transform(scaled)

    detector: object
    centers: Dict[int, np.ndarray] = {}
    if detector_name == "mahalanobis":
        detector = LedoitWolf(assume_centered=False).fit(z)
    elif detector_name == "knn5":
        detector = NearestNeighbors(
            n_neighbors=min(5, len(z)),
            metric="euclidean",
        ).fit(z)
    elif detector_name in {"client_l2", "client_cosine"}:
        detector = None
        for client_id in sorted(set(client_ids.astype(int))):
            rows = z[client_ids.astype(int) == client_id]
            if len(rows) == 0:
                raise RuntimeError(f"No clean history for client {client_id}")
            centers[int(client_id)] = np.mean(rows, axis=0)
    else:
        raise ValueError(detector_name)
    return CleanOnlyModel(scaler, pca, detector_name, detector, centers)


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


def geometry_candidate_id(
    score_name: str,
    quantile: float,
    fusion: str,
) -> str:
    return f"{score_name}_q{quantile:g}_{fusion}"


def nested_scores(
    clean_meta: pd.DataFrame,
    clean_x: np.ndarray,
    outer_round: int,
    pca_dim: int,
    detector_name: str,
) -> Tuple[np.ndarray, CleanOnlyModel]:
    train_rounds = [
        r for r in sorted(clean_meta["monitoring_round"].unique())
        if int(r) != int(outer_round)
    ]
    inner_scores = []
    for inner_round in train_rounds:
        inner_train_mask = (
            clean_meta["monitoring_round"].astype(int) != int(outer_round)
        ) & (
            clean_meta["monitoring_round"].astype(int) != int(inner_round)
        )
        inner_valid_mask = (
            clean_meta["monitoring_round"].astype(int) == int(inner_round)
        )
        model = fit_clean_model(
            clean_x[inner_train_mask.to_numpy()],
            clean_meta.loc[inner_train_mask, "client_id"].to_numpy(dtype=int),
            pca_dim,
            detector_name,
        )
        scores = model.score(
            clean_x[inner_valid_mask.to_numpy()],
            clean_meta.loc[inner_valid_mask, "client_id"].to_numpy(dtype=int),
        )
        inner_scores.extend(scores.tolist())

    outer_train_mask = (
        clean_meta["monitoring_round"].astype(int) != int(outer_round)
    )
    outer_model = fit_clean_model(
        clean_x[outer_train_mask.to_numpy()],
        clean_meta.loc[outer_train_mask, "client_id"].to_numpy(dtype=int),
        pca_dim,
        detector_name,
    )
    return np.asarray(inner_scores, dtype=float), outer_model


def aggregate_candidates(evaluation: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        evaluation.groupby(
            [
                "candidate_id", "representation", "pca_dim",
                "detector", "clean_quantile", "fusion",
            ],
            as_index=False,
        )
        .agg(
            mean_malicious_recall=("malicious_recall", "mean"),
            minimum_source_round_recall=("malicious_recall", "min"),
            mean_attack_benign_fpr=("attack_benign_fpr", "mean"),
            maximum_source_round_fpr=("attack_benign_fpr", "max"),
            mean_heldout_clean_fpr=("heldout_clean_fpr", "mean"),
            maximum_heldout_clean_round_fpr=("heldout_clean_fpr", "max"),
            source_round_count=("source_class", "size"),
        )
    )
    source = (
        evaluation.groupby(
            ["candidate_id", "source_class"],
            as_index=False,
        )["malicious_recall"].mean()
        .pivot(
            index="candidate_id",
            columns="source_class",
            values="malicious_recall",
        )
        .reset_index()
    )
    grouped = grouped.merge(
        source,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    for name in ATTACK_SOURCES:
        if name not in grouped.columns:
            grouped[name] = np.nan

    grouped["passes_mean_recall_090"] = (
        grouped["mean_malicious_recall"] >= 0.90
    )
    grouped["passes_min_source_round_recall_050"] = (
        grouped["minimum_source_round_recall"] >= 0.50
    )
    grouped["passes_dos_recall_095"] = grouped["DoS"] >= 0.95
    grouped["passes_bruteforce_recall_075"] = (
        grouped["BruteForce"] >= 0.75
    )
    grouped["passes_web_recall_075"] = (
        grouped["Web-Based"] >= 0.75
    )
    grouped["passes_mean_attack_fpr_005"] = (
        grouped["mean_attack_benign_fpr"] <= 0.05
    )
    grouped["passes_max_attack_fpr_010"] = (
        grouped["maximum_source_round_fpr"] <= 0.10
    )
    grouped["passes_mean_clean_fpr_005"] = (
        grouped["mean_heldout_clean_fpr"] <= 0.05
    )
    grouped["passes_max_clean_fpr_010"] = (
        grouped["maximum_heldout_clean_round_fpr"] <= 0.10
    )
    criteria = [
        "passes_mean_recall_090",
        "passes_min_source_round_recall_050",
        "passes_dos_recall_095",
        "passes_bruteforce_recall_075",
        "passes_web_recall_075",
        "passes_mean_attack_fpr_005",
        "passes_max_attack_fpr_010",
        "passes_mean_clean_fpr_005",
        "passes_max_clean_fpr_010",
    ]
    grouped["passes_all_predefined_smoke_criteria"] = grouped[
        criteria
    ].all(axis=1)
    grouped["selected_as_v318c_provisional_candidate"] = False
    return grouped


def main() -> int:
    args = parse_args()
    capture_root = args.capture_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    features_dir = output / "features"
    for directory in (tables, figures, features_dir):
        directory.mkdir(parents=True, exist_ok=True)

    decision_b = verify_v318b_decision(capture_root)
    manifest = load_manifest(capture_root)

    scenario_dirs = {
        str(row["scenario"]): Path(str(row["capture_dir"]))
        for _, row in manifest.iterrows()
    }
    for scenario, path in scenario_dirs.items():
        if not path.exists():
            raise FileNotFoundError(path)

    clean_dir = scenario_dirs["Clean"]
    layout_path = (
        clean_dir / "update_artifacts" / "parameter_layout.csv"
    )
    layout = pd.read_csv(layout_path)
    included = layout[layout["included_in_update_vector"].astype(bool)]
    parameter_count = int(included["numel"].sum())
    projection = build_projection(
        parameter_count,
        int(args.sketch_dim),
        int(args.sketch_seed),
    )
    head_idx = selected_indices(layout, ["classifier."])
    tail_idx = selected_indices(
        layout, ["penultimate.", "classifier."]
    )
    layer_names = included["parameter_name"].astype(str).tolist()

    all_meta = []
    all_representations: Dict[str, List[np.ndarray]] = {}
    for scenario in SCENARIO_ORDER:
        meta, representations = extract_scenario(
            scenario=scenario,
            capture_dir=scenario_dirs[scenario],
            projection=projection,
            layout=layout,
            head_idx=head_idx,
            tail_idx=tail_idx,
            layer_names=layer_names,
            features_dir=features_dir,
            force=bool(args.force_reextract),
        )
        all_meta.append(meta)
        for name, matrix in representations.items():
            all_representations.setdefault(name, []).append(matrix)

    metadata = pd.concat(all_meta, ignore_index=True)
    metadata["sample_index"] = np.arange(len(metadata), dtype=int)
    metadata.to_csv(tables / "v318c_sample_manifest.csv", index=False)
    representations = {
        name: np.vstack(parts)
        for name, parts in all_representations.items()
    }

    clean_mask = metadata["scenario"].eq("Clean").to_numpy()
    clean_meta = metadata.loc[clean_mask].reset_index(drop=True)
    attack_meta = metadata.loc[~clean_mask].reset_index(drop=True)

    evaluation_rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []

    learned_representations = [
        "head_raw",
        "head_direction",
        "tail_direction",
        "full_sketch_raw",
        "full_sketch_direction",
        "layer_profile",
        "fused_update_views",
    ]

    for representation in learned_representations:
        full_x = representations[representation]
        clean_x = full_x[clean_mask]
        attack_x = full_x[~clean_mask]

        for pca_dim in PCA_DIMS:
            for detector_name in DETECTORS:
                for outer_round in [1, 2, 3, 4]:
                    inner_clean_scores, model = nested_scores(
                        clean_meta,
                        clean_x,
                        outer_round,
                        pca_dim,
                        detector_name,
                    )
                    clean_outer_mask = (
                        clean_meta["monitoring_round"].astype(int)
                        == int(outer_round)
                    ).to_numpy()
                    clean_scores = model.score(
                        clean_x[clean_outer_mask],
                        clean_meta.loc[
                            clean_outer_mask, "client_id"
                        ].to_numpy(dtype=int),
                    )

                    for quantile in QUANTILES:
                        threshold = q_higher(
                            inner_clean_scores, quantile
                        )
                        clean_candidate = clean_scores > threshold
                        clean_current = clean_meta.loc[
                            clean_outer_mask,
                            "current_detector_flagged",
                        ].astype(bool).to_numpy()

                        for fusion in FUSIONS:
                            clean_flags = (
                                clean_candidate
                                if fusion == "candidate_only"
                                else clean_candidate | clean_current
                            )
                            threshold_rows.append({
                                "candidate_id": candidate_id(
                                    representation,
                                    pca_dim,
                                    detector_name,
                                    quantile,
                                    fusion,
                                ),
                                "representation": representation,
                                "pca_dim": pca_dim,
                                "detector": detector_name,
                                "clean_quantile": quantile,
                                "fusion": fusion,
                                "outer_round": outer_round,
                                "threshold": threshold,
                                "inner_clean_score_count":
                                    len(inner_clean_scores),
                                "heldout_clean_fpr":
                                    float(clean_flags.mean()),
                                "attack_labels_used_for_threshold":
                                    False,
                            })

                            for source in ATTACK_SOURCES:
                                attack_outer_mask = (
                                    attack_meta["source_class"].eq(source)
                                    & (
                                        attack_meta["monitoring_round"]
                                        .astype(int)
                                        == int(outer_round)
                                    )
                                ).to_numpy()
                                attack_scores = model.score(
                                    attack_x[attack_outer_mask],
                                    attack_meta.loc[
                                        attack_outer_mask, "client_id"
                                    ].to_numpy(dtype=int),
                                )
                                attack_candidate = (
                                    attack_scores > threshold
                                )
                                attack_current = attack_meta.loc[
                                    attack_outer_mask,
                                    "current_detector_flagged",
                                ].astype(bool).to_numpy()
                                attack_flags = (
                                    attack_candidate
                                    if fusion == "candidate_only"
                                    else attack_candidate | attack_current
                                )
                                labels = attack_meta.loc[
                                    attack_outer_mask,
                                    "actual_malicious",
                                ].astype(bool).to_numpy()
                                benign = ~labels
                                evaluation_rows.append({
                                    "candidate_id": candidate_id(
                                        representation,
                                        pca_dim,
                                        detector_name,
                                        quantile,
                                        fusion,
                                    ),
                                    "representation": representation,
                                    "pca_dim": pca_dim,
                                    "detector": detector_name,
                                    "clean_quantile": quantile,
                                    "fusion": fusion,
                                    "source_class": source,
                                    "outer_round": outer_round,
                                    "threshold": threshold,
                                    "malicious_recall": float(
                                        attack_flags[labels].mean()
                                    ),
                                    "attack_benign_fpr": float(
                                        attack_flags[benign].mean()
                                    ),
                                    "heldout_clean_fpr": float(
                                        clean_flags.mean()
                                    ),
                                    "flagged_malicious": int(
                                        (attack_flags & labels).sum()
                                    ),
                                    "missed_malicious": int(
                                        ((~attack_flags) & labels).sum()
                                    ),
                                    "false_positive_benign": int(
                                        (attack_flags & benign).sum()
                                    ),
                                    "attack_labels_used_for_threshold":
                                        False,
                                })

    geometry_names = [
        "round_median_l2",
        "round_median_cosine",
        "round_knn5_cosine",
        "round_norm_abs_robust_z",
    ]
    for score_name in geometry_names:
        scores = representations[score_name].reshape(-1)
        clean_scores_all = scores[clean_mask]
        attack_scores_all = scores[~clean_mask]

        for outer_round in [1, 2, 3, 4]:
            inner_mask = (
                clean_meta["monitoring_round"].astype(int)
                != int(outer_round)
            ).to_numpy()
            clean_outer_mask = (
                clean_meta["monitoring_round"].astype(int)
                == int(outer_round)
            ).to_numpy()
            inner_scores = clean_scores_all[inner_mask]
            clean_scores = clean_scores_all[clean_outer_mask]

            for quantile in QUANTILES:
                threshold = q_higher(inner_scores, quantile)
                clean_candidate = clean_scores > threshold
                clean_current = clean_meta.loc[
                    clean_outer_mask,
                    "current_detector_flagged",
                ].astype(bool).to_numpy()

                for fusion in FUSIONS:
                    clean_flags = (
                        clean_candidate
                        if fusion == "candidate_only"
                        else clean_candidate | clean_current
                    )
                    cid = geometry_candidate_id(
                        score_name, quantile, fusion
                    )
                    threshold_rows.append({
                        "candidate_id": cid,
                        "representation": score_name,
                        "pca_dim": 0,
                        "detector": "round_geometry",
                        "clean_quantile": quantile,
                        "fusion": fusion,
                        "outer_round": outer_round,
                        "threshold": threshold,
                        "inner_clean_score_count": int(inner_mask.sum()),
                        "heldout_clean_fpr": float(clean_flags.mean()),
                        "attack_labels_used_for_threshold": False,
                    })

                    for source in ATTACK_SOURCES:
                        attack_outer_mask = (
                            attack_meta["source_class"].eq(source)
                            & (
                                attack_meta["monitoring_round"]
                                .astype(int)
                                == int(outer_round)
                            )
                        ).to_numpy()
                        attack_candidate = (
                            attack_scores_all[attack_outer_mask]
                            > threshold
                        )
                        attack_current = attack_meta.loc[
                            attack_outer_mask,
                            "current_detector_flagged",
                        ].astype(bool).to_numpy()
                        attack_flags = (
                            attack_candidate
                            if fusion == "candidate_only"
                            else attack_candidate | attack_current
                        )
                        labels = attack_meta.loc[
                            attack_outer_mask,
                            "actual_malicious",
                        ].astype(bool).to_numpy()
                        benign = ~labels
                        evaluation_rows.append({
                            "candidate_id": cid,
                            "representation": score_name,
                            "pca_dim": 0,
                            "detector": "round_geometry",
                            "clean_quantile": quantile,
                            "fusion": fusion,
                            "source_class": source,
                            "outer_round": outer_round,
                            "threshold": threshold,
                            "malicious_recall": float(
                                attack_flags[labels].mean()
                            ),
                            "attack_benign_fpr": float(
                                attack_flags[benign].mean()
                            ),
                            "heldout_clean_fpr": float(
                                clean_flags.mean()
                            ),
                            "flagged_malicious": int(
                                (attack_flags & labels).sum()
                            ),
                            "missed_malicious": int(
                                ((~attack_flags) & labels).sum()
                            ),
                            "false_positive_benign": int(
                                (attack_flags & benign).sum()
                            ),
                            "attack_labels_used_for_threshold": False,
                        })

    thresholds = pd.DataFrame(threshold_rows)
    evaluation = pd.DataFrame(evaluation_rows)
    thresholds.to_csv(
        tables / "v318c_clean_only_thresholds.csv", index=False
    )
    evaluation.to_csv(
        tables / "v318c_source_round_evaluation.csv", index=False
    )

    aggregate = aggregate_candidates(evaluation)
    passing = aggregate[
        aggregate["passes_all_predefined_smoke_criteria"]
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
                "minimum_source_round_recall",
                "mean_malicious_recall",
                "fusion_penalty",
                "candidate_id",
            ],
            ascending=[True, True, False, False, True, True],
        )
        selected = passing.iloc[0]
        aggregate.loc[
            aggregate["candidate_id"].eq(
                str(selected["candidate_id"])
            ),
            "selected_as_v318c_provisional_candidate",
        ] = True

    aggregate.to_csv(
        tables / "v318c_candidate_aggregate_summary.csv",
        index=False,
    )
    source_summary = (
        evaluation.groupby(
            ["candidate_id", "source_class"],
            as_index=False,
        )
        .agg(
            mean_malicious_recall=("malicious_recall", "mean"),
            minimum_round_recall=("malicious_recall", "min"),
            mean_benign_fpr=("attack_benign_fpr", "mean"),
            maximum_round_fpr=("attack_benign_fpr", "max"),
        )
    )
    source_summary.to_csv(
        tables / "v318c_candidate_source_summary.csv",
        index=False,
    )

    decision = {
        "experiment_version": "3.18C",
        "stage": "offline_update_space_detector_feasibility_smoke",
        "seed": 7,
        "scenarios": SCENARIO_ORDER,
        "full_update_countsketch_dimension": int(args.sketch_dim),
        "clean_only_nested_leave_one_round_out_calibration": True,
        "attack_labels_used_for_threshold_calibration": False,
        "attack_labels_used_for_development_selection": True,
        "federated_training_rerun": False,
        "training_or_aggregation_modified": False,
        "test_sets_accessed": False,
        "smoke_candidate_selected": selected is not None,
        "selected_candidate_id": (
            None if selected is None
            else str(selected["candidate_id"])
        ),
        "method_frozen": False,
        "untouched_multiseed_validation_required": True,
        "next_stage": (
            "Freeze the provisional representation and detector, instrument "
            "trusted calibration updates, then run new untouched multiseed "
            "validation before any paper claim."
            if selected is not None
            else
            "No update-space candidate passed the predefined smoke criteria. "
            "Inspect source-specific score distributions and add only a "
            "pre-registered richer update representation, not a weaker "
            "threshold."
        ),
    }
    pd.DataFrame([{
        "smoke_candidate_selected": decision["smoke_candidate_selected"],
        "selected_candidate_id": decision["selected_candidate_id"],
        "method_frozen": False,
        "training_or_aggregation_modified": False,
        "test_sets_accessed": False,
        "untouched_multiseed_validation_required": True,
    }]).to_csv(tables / "v318c_decision.csv", index=False)
    with (output / "v318c_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(decision, f, indent=2)

    ranked = aggregate.sort_values(
        [
            "passes_all_predefined_smoke_criteria",
            "mean_malicious_recall",
            "mean_attack_benign_fpr",
        ],
        ascending=[False, False, True],
    )

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for fusion, group in aggregate.groupby("fusion"):
        ax.scatter(
            group["mean_attack_benign_fpr"],
            group["mean_malicious_recall"],
            alpha=0.55,
            label=fusion,
        )
    ax.axhline(0.90, linestyle="--", linewidth=1)
    ax.axvline(0.05, linestyle="--", linewidth=1)
    ax.set_xlabel("Mean benign FPR during attacks")
    ax.set_ylabel("Mean malicious recall")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.set_title("V3.18C update-space detector frontier")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "update_space_frontier")

    top = ranked.head(20).copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(top))
    ax.bar(x, top["mean_malicious_recall"])
    ax.set_xticks(x)
    ax.set_xticklabels(
        top["candidate_id"],
        rotation=75,
        ha="right",
        fontsize=7,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean malicious recall")
    ax.set_title("Top V3.18C update-space candidates")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures / "top_update_space_candidates")

    print("V3.18C update-space detector feasibility screen complete")
    print()
    print("TOP CANDIDATES")
    columns = [
        "candidate_id", "representation", "pca_dim", "detector",
        "clean_quantile", "fusion", "mean_malicious_recall",
        "minimum_source_round_recall", "mean_attack_benign_fpr",
        "maximum_source_round_fpr", "mean_heldout_clean_fpr",
        "maximum_heldout_clean_round_fpr", "DoS", "BruteForce",
        "Web-Based", "passes_all_predefined_smoke_criteria",
        "selected_as_v318c_provisional_candidate",
    ]
    print(ranked[columns].head(30).to_string(index=False))
    print()
    print("V3.18C DECISION")
    print(
        pd.read_csv(
            tables / "v318c_decision.csv"
        ).to_string(index=False)
    )
    print()
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    print("Feature cache:", features_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

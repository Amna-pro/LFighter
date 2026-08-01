#!/usr/bin/env python3
"""V3.19A one-round robust-center and robust-aggregation audit.

Uses only the exactly equivalent V3.18B round-1 client-update captures.
All four scenarios begin from the identical frozen round-4 warmup state, so
round 1 provides the only exact apples-to-apples offline comparison without
retraining.

Methods:
- sample-weighted FedAvg, exact reference
- equal-client mean, weighting control
- coordinate median
- coordinate trimmed mean with f=8
- geometric median, deterministic Weiszfeld
- Multi-Krum with f=8, m=10

The audit evaluates:
- clean utility on validation,
- source-to-Benign attack rate,
- macro-F1 recovery,
- aggregate-update error relative to the clean FedAvg update,
- classifier-head error,
- deterministic runtime and Multi-Krum selections.

It does not access natural or diagnostic test sets and does not modify any
training or aggregation trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from federated_iot_v26 import CLASS_NAMES, NUM_CLASSES, load_protocol_arrays
from neural_models_v24 import build_model
from transition_signature_features_v38 import evaluate_validation


SCENARIO_ORDER = [
    "Clean",
    "DoS->Benign",
    "BruteForce->Benign",
    "Web-Based->Benign",
]
ATTACK_SOURCES = ["DoS", "BruteForce", "Web-Based"]
METHOD_ORDER = [
    "weighted_fedavg",
    "equal_mean",
    "coordinate_median",
    "trimmed_mean_f8",
    "geometric_median",
    "multi_krum_f8_m10",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-root", required=True, type=Path)
    p.add_argument("--data-file", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--evaluation-batch-size", type=int, default=4096)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--trim-f", type=int, default=8)
    p.add_argument("--krum-f", type=int, default=8)
    p.add_argument("--krum-m", type=int, default=10)
    p.add_argument("--geomed-max-iter", type=int, default=60)
    p.add_argument("--geomed-tol", type=float, default=1e-7)
    return p.parse_args()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def verify_v318b(root: Path) -> pd.DataFrame:
    decision_path = root / "tables" / "v318b_decision.csv"
    manifest_path = root / "tables" / "v318b_smoke_capture_manifest.csv"
    if not decision_path.exists():
        raise FileNotFoundError(decision_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    decision = pd.read_csv(decision_path)
    if len(decision) != 1:
        raise RuntimeError("V3.18B decision must contain one row")
    row = decision.iloc[0]
    if not bool(row["all_capture_branches_exactly_equivalent"]):
        raise RuntimeError("V3.18B exact-equivalence requirement failed")
    if bool(row["training_or_aggregation_modified"]):
        raise RuntimeError("V3.18B reports modified training or aggregation")
    if bool(row["test_sets_accessed"]):
        raise RuntimeError("V3.18B reports test-set access")

    manifest = pd.read_csv(manifest_path)
    missing = sorted(set(SCENARIO_ORDER) - set(manifest["scenario"].astype(str)))
    if missing:
        raise RuntimeError(f"Missing V3.18B scenarios: {missing}")
    if not manifest["exact_equivalence"].astype(bool).all():
        raise RuntimeError("At least one V3.18B scenario is not exact")
    return manifest


def load_round_one(
    capture_dir: Path,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, Dict[str, torch.Tensor]]:
    index = pd.read_csv(
        capture_dir / "tables" / "v318b_update_capture_index.csv"
    )
    row = index[index["monitoring_round"].astype(int) == 1]
    if len(row) != 1:
        raise RuntimeError(f"Missing unique monitoring round 1 in {capture_dir}")
    row = row.iloc[0]

    matrix_path = capture_dir / str(row["update_matrix_file"])
    state_path = capture_dir / str(row["reference_state_file"])
    if digest(matrix_path) != str(row["update_matrix_sha256"]):
        raise RuntimeError(f"Update matrix hash mismatch: {matrix_path}")
    if digest(state_path) != str(row["reference_state_sha256"]):
        raise RuntimeError(f"Reference state hash mismatch: {state_path}")

    updates = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    updates = np.asarray(updates, dtype=np.float32)
    client_manifest = pd.read_csv(
        capture_dir
        / "update_artifacts"
        / "round_01_client_manifest.csv"
    ).sort_values("client_id").reset_index(drop=True)
    layout = pd.read_csv(
        capture_dir / "update_artifacts" / "parameter_layout.csv"
    )
    state_obj = torch.load(state_path, map_location="cpu", weights_only=False)
    reference_state = state_obj["state_dict"]

    if updates.shape[0] != 20:
        raise RuntimeError("V3.19A requires exactly 20 clients")
    if len(client_manifest) != 20:
        raise RuntimeError("Client manifest must contain exactly 20 rows")
    if int(layout.loc[layout["included_in_update_vector"].astype(bool), "numel"].sum()) != updates.shape[1]:
        raise RuntimeError("Parameter layout does not match update matrix")
    if not np.isfinite(updates).all():
        raise RuntimeError("Non-finite update values found")
    return updates, client_manifest, layout, reference_state


def weighted_fedavg(
    updates: np.ndarray,
    sample_counts: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(sample_counts, dtype=np.float64)
    weights /= weights.sum()
    return np.asarray(
        weights @ updates.astype(np.float64, copy=False),
        dtype=np.float64,
    )


def equal_mean(updates: np.ndarray) -> np.ndarray:
    return updates.astype(np.float64, copy=False).mean(axis=0)


def coordinate_median(updates: np.ndarray) -> np.ndarray:
    return np.median(updates.astype(np.float64, copy=False), axis=0)


def trimmed_mean(updates: np.ndarray, f: int) -> np.ndarray:
    n = int(updates.shape[0])
    if f < 0 or 2 * f >= n:
        raise ValueError(f"Invalid trimmed-mean f={f} for n={n}")
    sorted_updates = np.sort(
        updates.astype(np.float64, copy=False),
        axis=0,
    )
    return sorted_updates[f : n - f].mean(axis=0)


def geometric_median(
    updates: np.ndarray,
    max_iter: int,
    tolerance: float,
) -> Tuple[np.ndarray, int, bool]:
    x = updates.astype(np.float64, copy=False)
    center = np.median(x, axis=0)
    converged = False
    iterations = 0
    for iteration in range(1, max_iter + 1):
        distances = np.linalg.norm(x - center[None, :], axis=1)
        near = np.where(distances <= 1e-12)[0]
        if len(near):
            new_center = x[int(near[0])].copy()
        else:
            inverse = 1.0 / np.maximum(distances, 1e-12)
            new_center = np.sum(
                x * inverse[:, None],
                axis=0,
            ) / inverse.sum()
        step = float(np.linalg.norm(new_center - center))
        scale = max(float(np.linalg.norm(center)), 1.0)
        center = new_center
        iterations = iteration
        if step <= tolerance * scale:
            converged = True
            break
    return center, iterations, converged


def multi_krum(
    updates: np.ndarray,
    f: int,
    m: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = updates.astype(np.float64, copy=False)
    n = int(x.shape[0])
    if n <= 2 * f + 2:
        raise ValueError(
            f"Multi-Krum requires n > 2f+2, got n={n}, f={f}"
        )
    neighbours = n - f - 2
    if m < 1 or m > neighbours:
        raise ValueError(
            f"Multi-Krum m must be in [1,{neighbours}], got {m}"
        )

    squared_norms = np.einsum("ij,ij->i", x, x)
    distances = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * (x @ x.T)
    )
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, np.inf)

    scores = np.empty(n, dtype=np.float64)
    for client_id in range(n):
        nearest = np.partition(
            distances[client_id],
            neighbours - 1,
        )[:neighbours]
        scores[client_id] = nearest.sum()

    selected = np.lexsort(
        (np.arange(n, dtype=int), scores)
    )[:m]
    aggregate = x[selected].mean(axis=0)
    return aggregate, selected.astype(int), scores


def aggregate_all(
    updates: np.ndarray,
    sample_counts: np.ndarray,
    trim_f: int,
    krum_f: int,
    krum_m: int,
    geomed_max_iter: int,
    geomed_tol: float,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]]]:
    outputs: Dict[str, np.ndarray] = {}
    metadata: List[Dict[str, object]] = []

    start = time.perf_counter()
    outputs["weighted_fedavg"] = weighted_fedavg(updates, sample_counts)
    metadata.append({
        "method": "weighted_fedavg",
        "runtime_seconds": time.perf_counter() - start,
        "weighting": "sample_weighted",
        "selected_clients": "|".join(map(str, range(len(updates)))),
        "algorithm_converged": True,
        "algorithm_iterations": 1,
    })

    start = time.perf_counter()
    outputs["equal_mean"] = equal_mean(updates)
    metadata.append({
        "method": "equal_mean",
        "runtime_seconds": time.perf_counter() - start,
        "weighting": "equal_client",
        "selected_clients": "|".join(map(str, range(len(updates)))),
        "algorithm_converged": True,
        "algorithm_iterations": 1,
    })

    start = time.perf_counter()
    outputs["coordinate_median"] = coordinate_median(updates)
    metadata.append({
        "method": "coordinate_median",
        "runtime_seconds": time.perf_counter() - start,
        "weighting": "equal_client",
        "selected_clients": "|".join(map(str, range(len(updates)))),
        "algorithm_converged": True,
        "algorithm_iterations": 1,
    })

    start = time.perf_counter()
    outputs[f"trimmed_mean_f{trim_f}"] = trimmed_mean(updates, trim_f)
    metadata.append({
        "method": f"trimmed_mean_f{trim_f}",
        "runtime_seconds": time.perf_counter() - start,
        "weighting": "equal_client",
        "selected_clients": "|".join(map(str, range(len(updates)))),
        "algorithm_converged": True,
        "algorithm_iterations": 1,
    })

    start = time.perf_counter()
    center, iterations, converged = geometric_median(
        updates,
        max_iter=geomed_max_iter,
        tolerance=geomed_tol,
    )
    outputs["geometric_median"] = center
    metadata.append({
        "method": "geometric_median",
        "runtime_seconds": time.perf_counter() - start,
        "weighting": "equal_client",
        "selected_clients": "|".join(map(str, range(len(updates)))),
        "algorithm_converged": converged,
        "algorithm_iterations": iterations,
    })

    start = time.perf_counter()
    center, selected, scores = multi_krum(
        updates,
        f=krum_f,
        m=krum_m,
    )
    outputs[f"multi_krum_f{krum_f}_m{krum_m}"] = center
    metadata.append({
        "method": f"multi_krum_f{krum_f}_m{krum_m}",
        "runtime_seconds": time.perf_counter() - start,
        "weighting": "equal_selected_client",
        "selected_clients": "|".join(map(str, selected.tolist())),
        "algorithm_converged": True,
        "algorithm_iterations": 1,
        "krum_neighbour_count": int(len(updates) - krum_f - 2),
        "krum_minimum_score": float(scores[selected[0]]),
        "krum_maximum_selected_score": float(scores[selected[-1]]),
    })
    return outputs, metadata


def vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-18:
        return float("nan")
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def head_indices(layout: pd.DataFrame) -> np.ndarray:
    pieces = []
    for _, row in layout.iterrows():
        if not bool(row["included_in_update_vector"]):
            continue
        if str(row["parameter_name"]).startswith("classifier."):
            pieces.append(
                np.arange(
                    int(row["start_index"]),
                    int(row["stop_index_exclusive"]),
                    dtype=np.int64,
                )
            )
    if not pieces:
        raise RuntimeError("Classifier parameters were not found")
    return np.concatenate(pieces)


def state_from_delta(
    reference_state: Mapping[str, torch.Tensor],
    layout: pd.DataFrame,
    delta: np.ndarray,
) -> Dict[str, torch.Tensor]:
    result = {
        name: tensor.detach().cpu().clone()
        for name, tensor in reference_state.items()
    }
    for _, row in layout.iterrows():
        if not bool(row["included_in_update_vector"]):
            continue
        name = str(row["parameter_name"])
        start = int(row["start_index"])
        stop = int(row["stop_index_exclusive"])
        reference = reference_state[name].detach().cpu()
        update = torch.from_numpy(
            np.asarray(delta[start:stop], dtype=np.float32)
        ).reshape(reference.shape)
        result[name] = (
            reference.to(torch.float32) + update
        ).to(reference.dtype)
    return result


def evaluate_delta(
    delta: np.ndarray,
    reference_state: Mapping[str, torch.Tensor],
    layout: pd.DataFrame,
    input_dim: int,
    X_val: np.ndarray,
    y_val: np.ndarray,
    source_id: int,
    target_id: int,
    batch_size: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model = build_model("resmlp", input_dim, NUM_CLASSES)
    model.load_state_dict(
        state_from_delta(reference_state, layout, delta)
    )
    return evaluate_validation(
        model,
        X_val,
        y_val,
        source_id,
        target_id,
        batch_size,
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return float("nan")
    return float(numerator / denominator)


def main() -> int:
    a = parse_args()
    if int(a.trim_f) != 8:
        raise ValueError("V3.19A trimmed-mean budget is frozen to f=8")
    if int(a.krum_f) != 8 or int(a.krum_m) != 10:
        raise ValueError("V3.19A Multi-Krum is frozen to f=8, m=10")

    torch.set_num_threads(max(1, int(a.threads)))
    capture_root = a.capture_root.expanduser().resolve()
    output = a.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    checkpoints = output / "aggregates"
    for directory in (tables, figures, checkpoints):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = verify_v318b(capture_root)
    capture_dirs = {
        str(row["scenario"]): Path(str(row["capture_dir"]))
        for _, row in manifest.iterrows()
    }

    arrays = load_protocol_arrays(a.data_file.expanduser().resolve())
    X_val = arrays["X_val"].astype(np.float32, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)
    input_dim = int(X_val.shape[1])
    target_id = CLASS_NAMES.index("Benign")

    scenario_data = {}
    reference_hashes = {}
    layout_reference = None
    reference_state_common = None
    for scenario in SCENARIO_ORDER:
        updates, clients, layout, reference_state = load_round_one(
            capture_dirs[scenario]
        )
        state_path = (
            capture_dirs[scenario]
            / "update_artifacts"
            / "round_01_reference_state.pt"
        )
        reference_hashes[scenario] = digest(state_path)
        if layout_reference is None:
            layout_reference = layout
            reference_state_common = reference_state
        else:
            pd.testing.assert_frame_equal(
                layout_reference.reset_index(drop=True),
                layout.reset_index(drop=True),
                check_dtype=False,
            )
        scenario_data[scenario] = {
            "updates": updates,
            "clients": clients,
            "layout": layout,
            "reference_state": reference_state,
        }

    if len(set(reference_hashes.values())) != 1:
        raise RuntimeError(
            "Round-1 scenario reference states are not byte-identical"
        )

    assert layout_reference is not None
    assert reference_state_common is not None
    classifier_idx = head_indices(layout_reference)

    aggregate_vectors: Dict[Tuple[str, str], np.ndarray] = {}
    method_rows: List[Dict[str, object]] = []
    for scenario in SCENARIO_ORDER:
        updates = scenario_data[scenario]["updates"]
        clients = scenario_data[scenario]["clients"]
        sample_counts = clients["client_samples"].to_numpy(dtype=np.float64)
        methods, metadata = aggregate_all(
            updates,
            sample_counts,
            trim_f=int(a.trim_f),
            krum_f=int(a.krum_f),
            krum_m=int(a.krum_m),
            geomed_max_iter=int(a.geomed_max_iter),
            geomed_tol=float(a.geomed_tol),
        )
        malicious_ids = set(
            clients.loc[
                clients["actual_malicious"].astype(bool),
                "client_id",
            ].astype(int).tolist()
        )
        for row in metadata:
            selected = {
                int(x)
                for x in str(row["selected_clients"]).split("|")
                if str(x).strip()
            }
            row.update({
                "scenario": scenario,
                "actual_malicious_clients":
                    "|".join(map(str, sorted(malicious_ids))),
                "selected_malicious_clients":
                    "|".join(map(str, sorted(selected & malicious_ids))),
                "selected_malicious_count":
                    int(len(selected & malicious_ids)),
                "selected_benign_count":
                    int(len(selected - malicious_ids)),
            })
            method_rows.append(row)
        for method, vector in methods.items():
            aggregate_vectors[(scenario, method)] = vector
            np.save(
                checkpoints / f"{scenario.lower().replace('->','_to_').replace('-','_')}_{method}.npy",
                vector.astype(np.float32),
                allow_pickle=False,
            )

    aggregation_metadata = pd.DataFrame(method_rows)
    aggregation_metadata.to_csv(
        tables / "v319a_aggregation_runtime_and_selection.csv",
        index=False,
    )

    clean_fedavg = aggregate_vectors[("Clean", "weighted_fedavg")]
    clean_norm = float(np.linalg.norm(clean_fedavg))
    clean_head = clean_fedavg[classifier_idx]
    clean_head_norm = float(np.linalg.norm(clean_head))

    evaluation_rows: List[Dict[str, object]] = []
    geometry_rows: List[Dict[str, object]] = []

    for scenario in SCENARIO_ORDER:
        source = "BruteForce" if scenario == "Clean" else scenario.split("->")[0]
        source_id = CLASS_NAMES.index(source)
        clients = scenario_data[scenario]["clients"]
        malicious_ids = set(
            clients.loc[
                clients["actual_malicious"].astype(bool),
                "client_id",
            ].astype(int).tolist()
        )

        for method in METHOD_ORDER:
            vector = aggregate_vectors[(scenario, method)]
            metrics, pair = evaluate_delta(
                vector,
                reference_state_common,
                layout_reference,
                input_dim,
                X_val,
                y_val,
                source_id,
                target_id,
                int(a.evaluation_batch_size),
            )
            runtime_row = aggregation_metadata[
                aggregation_metadata["scenario"].eq(scenario)
                & aggregation_metadata["method"].eq(method)
            ].iloc[0]
            evaluation_rows.append({
                "scenario": scenario,
                "source_class": source,
                "method": method,
                "is_clean_scenario": scenario == "Clean",
                "validation_macro_f1": float(metrics["macro_f1"]),
                "validation_balanced_accuracy": float(
                    metrics.get("balanced_accuracy", np.nan)
                ),
                "validation_accuracy": float(
                    metrics.get("accuracy", np.nan)
                ),
                "source_to_target_rate": float(
                    pair["source_to_target_rate"]
                ),
                "source_recall": float(
                    pair.get("source_recall", np.nan)
                ),
                "aggregation_runtime_seconds": float(
                    runtime_row["runtime_seconds"]
                ),
                "selected_malicious_count": int(
                    runtime_row["selected_malicious_count"]
                ),
                "selected_benign_count": int(
                    runtime_row["selected_benign_count"]
                ),
                "selected_clients": str(
                    runtime_row["selected_clients"]
                ),
                "test_sets_accessed": False,
            })

            vector_error = float(np.linalg.norm(vector - clean_fedavg))
            vector_head = vector[classifier_idx]
            head_error = float(np.linalg.norm(vector_head - clean_head))
            geometry_rows.append({
                "scenario": scenario,
                "source_class": source,
                "method": method,
                "aggregate_l2_norm": float(np.linalg.norm(vector)),
                "distance_to_clean_fedavg": vector_error,
                "relative_distance_to_clean_fedavg":
                    safe_ratio(vector_error, clean_norm),
                "cosine_to_clean_fedavg":
                    vector_cosine(vector, clean_fedavg),
                "classifier_head_distance_to_clean_fedavg": head_error,
                "classifier_head_relative_distance":
                    safe_ratio(head_error, clean_head_norm),
                "classifier_head_cosine_to_clean_fedavg":
                    vector_cosine(vector_head, clean_head),
            })

    evaluation = pd.DataFrame(evaluation_rows)
    geometry = pd.DataFrame(geometry_rows)
    evaluation.to_csv(
        tables / "v319a_one_round_validation_metrics.csv",
        index=False,
    )
    geometry.to_csv(
        tables / "v319a_aggregate_geometry.csv",
        index=False,
    )

    clean_evaluations: List[Dict[str, object]] = []
    for source in ATTACK_SOURCES:
        source_id = CLASS_NAMES.index(source)
        for method in METHOD_ORDER:
            metrics, pair = evaluate_delta(
                aggregate_vectors[("Clean", method)],
                reference_state_common,
                layout_reference,
                input_dim,
                X_val,
                y_val,
                source_id,
                target_id,
                int(a.evaluation_batch_size),
            )
            clean_evaluations.append({
                "source_class": source,
                "method": method,
                "clean_validation_macro_f1": float(metrics["macro_f1"]),
                "clean_source_to_target_rate": float(
                    pair["source_to_target_rate"]
                ),
            })
    clean_pair = pd.DataFrame(clean_evaluations)
    clean_pair.to_csv(
        tables / "v319a_clean_pair_specific_metrics.csv",
        index=False,
    )

    summary_rows: List[Dict[str, object]] = []
    for method in METHOD_ORDER:
        clean_method_macro = float(
            clean_pair[
                clean_pair["method"].eq(method)
            ]["clean_validation_macro_f1"].iloc[0]
        )
        clean_plain_macro = float(
            clean_pair[
                clean_pair["method"].eq("weighted_fedavg")
            ]["clean_validation_macro_f1"].iloc[0]
        )
        method_geometry_clean = geometry[
            geometry["scenario"].eq("Clean")
            & geometry["method"].eq(method)
        ].iloc[0]

        per_attack = []
        for source in ATTACK_SOURCES:
            scenario = f"{source}->Benign"
            clean_method = clean_pair[
                clean_pair["source_class"].eq(source)
                & clean_pair["method"].eq(method)
            ].iloc[0]
            clean_plain = clean_pair[
                clean_pair["source_class"].eq(source)
                & clean_pair["method"].eq("weighted_fedavg")
            ].iloc[0]
            attack_method = evaluation[
                evaluation["scenario"].eq(scenario)
                & evaluation["method"].eq(method)
            ].iloc[0]
            attack_plain = evaluation[
                evaluation["scenario"].eq(scenario)
                & evaluation["method"].eq("weighted_fedavg")
            ].iloc[0]
            geometry_method = geometry[
                geometry["scenario"].eq(scenario)
                & geometry["method"].eq(method)
            ].iloc[0]
            geometry_plain = geometry[
                geometry["scenario"].eq(scenario)
                & geometry["method"].eq("weighted_fedavg")
            ].iloc[0]

            plain_excess = (
                float(attack_plain["source_to_target_rate"])
                - float(clean_plain["clean_source_to_target_rate"])
            )
            method_excess = (
                float(attack_method["source_to_target_rate"])
                - float(clean_method["clean_source_to_target_rate"])
            )
            excess_removed = (
                1.0 - safe_ratio(method_excess, plain_excess)
                if abs(plain_excess) > 1e-12 else float("nan")
            )

            plain_attack_loss = (
                float(clean_plain["clean_validation_macro_f1"])
                - float(attack_plain["validation_macro_f1"])
            )
            method_attack_loss = (
                float(clean_method["clean_validation_macro_f1"])
                - float(attack_method["validation_macro_f1"])
            )
            macro_loss_removed = (
                1.0 - safe_ratio(method_attack_loss, plain_attack_loss)
                if abs(plain_attack_loss) > 1e-12 else float("nan")
            )

            plain_center_error = float(
                geometry_plain["distance_to_clean_fedavg"]
            )
            method_center_error = float(
                geometry_method["distance_to_clean_fedavg"]
            )
            center_error_reduction = (
                1.0 - safe_ratio(
                    method_center_error,
                    plain_center_error,
                )
                if plain_center_error > 1e-12 else float("nan")
            )
            per_attack.append({
                "source_class": source,
                "plain_source_to_target_rate":
                    float(attack_plain["source_to_target_rate"]),
                "method_source_to_target_rate":
                    float(attack_method["source_to_target_rate"]),
                "source_target_excess_removed": excess_removed,
                "plain_attack_macro_f1":
                    float(attack_plain["validation_macro_f1"]),
                "method_attack_macro_f1":
                    float(attack_method["validation_macro_f1"]),
                "attack_macro_f1_recovery_vs_plain":
                    float(attack_method["validation_macro_f1"])
                    - float(attack_plain["validation_macro_f1"]),
                "macro_f1_attack_loss_removed": macro_loss_removed,
                "plain_center_error": plain_center_error,
                "method_center_error": method_center_error,
                "center_error_reduction": center_error_reduction,
            })

        attack_table = pd.DataFrame(per_attack)
        row: Dict[str, object] = {
            "method": method,
            "clean_macro_f1": clean_method_macro,
            "clean_macro_f1_delta_vs_weighted_fedavg":
                clean_method_macro - clean_plain_macro,
            "clean_relative_center_distortion":
                float(
                    method_geometry_clean[
                        "relative_distance_to_clean_fedavg"
                    ]
                ),
            "mean_center_error_reduction":
                float(attack_table["center_error_reduction"].mean()),
            "minimum_center_error_reduction":
                float(attack_table["center_error_reduction"].min()),
            "mean_source_target_excess_removed":
                float(
                    attack_table[
                        "source_target_excess_removed"
                    ].mean()
                ),
            "minimum_source_target_excess_removed":
                float(
                    attack_table[
                        "source_target_excess_removed"
                    ].min()
                ),
            "mean_attack_macro_f1_recovery_vs_plain":
                float(
                    attack_table[
                        "attack_macro_f1_recovery_vs_plain"
                    ].mean()
                ),
            "minimum_attack_macro_f1_recovery_vs_plain":
                float(
                    attack_table[
                        "attack_macro_f1_recovery_vs_plain"
                    ].min()
                ),
            "positive_center_error_reduction_all_attacks":
                bool(
                    (
                        attack_table["center_error_reduction"] > 0
                    ).all()
                ),
            "positive_source_target_reduction_all_attacks":
                bool(
                    (
                        attack_table[
                            "method_source_to_target_rate"
                        ]
                        < attack_table[
                            "plain_source_to_target_rate"
                        ]
                    ).all()
                ),
            "positive_macro_f1_recovery_all_attacks":
                bool(
                    (
                        attack_table[
                            "attack_macro_f1_recovery_vs_plain"
                        ] > 0
                    ).all()
                ),
        }
        for _, attack in attack_table.iterrows():
            prefix = str(attack["source_class"]).lower().replace("-", "_")
            for column in [
                "plain_source_to_target_rate",
                "method_source_to_target_rate",
                "source_target_excess_removed",
                "plain_attack_macro_f1",
                "method_attack_macro_f1",
                "attack_macro_f1_recovery_vs_plain",
                "macro_f1_attack_loss_removed",
                "plain_center_error",
                "method_center_error",
                "center_error_reduction",
            ]:
                row[f"{prefix}_{column}"] = attack[column]

        row["passes_clean_macro_delta"] = (
            row["clean_macro_f1_delta_vs_weighted_fedavg"] >= -0.005
        )
        row["passes_clean_center_distortion"] = (
            row["clean_relative_center_distortion"] <= 0.25
        )
        row["passes_mean_center_error_reduction"] = (
            row["mean_center_error_reduction"] >= 0.40
        )
        row["passes_min_center_error_reduction"] = (
            row["minimum_center_error_reduction"] > 0.0
        )
        row["passes_center_reference_criteria"] = bool(
            row["passes_clean_macro_delta"]
            and row["passes_clean_center_distortion"]
            and row["passes_mean_center_error_reduction"]
            and row["passes_min_center_error_reduction"]
            and row[
                "positive_source_target_reduction_all_attacks"
            ]
        )
        row["passes_standalone_robust_aggregator_criteria"] = bool(
            row["passes_center_reference_criteria"]
            and row[
                "minimum_source_target_excess_removed"
            ] >= 0.40
            and row[
                "positive_macro_f1_recovery_all_attacks"
            ]
        )
        row["selected_as_robust_center_reference"] = False
        row["selected_as_standalone_aggregator_candidate"] = False
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    center_candidates = summary[
        summary["passes_center_reference_criteria"]
        & ~summary["method"].eq("weighted_fedavg")
    ].copy()
    center_selected = None
    if len(center_candidates):
        center_candidates = center_candidates.sort_values(
            [
                "mean_center_error_reduction",
                "minimum_center_error_reduction",
                "clean_macro_f1_delta_vs_weighted_fedavg",
                "method",
            ],
            ascending=[False, False, False, True],
        )
        center_selected = center_candidates.iloc[0]
        summary.loc[
            summary["method"].eq(
                str(center_selected["method"])
            ),
            "selected_as_robust_center_reference",
        ] = True

    standalone_candidates = summary[
        summary["passes_standalone_robust_aggregator_criteria"]
        & ~summary["method"].eq("weighted_fedavg")
    ].copy()
    standalone_selected = None
    if len(standalone_candidates):
        standalone_candidates = standalone_candidates.sort_values(
            [
                "minimum_source_target_excess_removed",
                "mean_source_target_excess_removed",
                "clean_macro_f1_delta_vs_weighted_fedavg",
                "method",
            ],
            ascending=[False, False, False, True],
        )
        standalone_selected = standalone_candidates.iloc[0]
        summary.loc[
            summary["method"].eq(
                str(standalone_selected["method"])
            ),
            "selected_as_standalone_aggregator_candidate",
        ] = True

    summary.to_csv(
        tables / "v319a_method_summary.csv",
        index=False,
    )

    plain_reproduction_rows = []
    for source in ATTACK_SOURCES:
        scenario = f"{source}->Benign"
        captured_round = pd.read_csv(
            capture_dirs[scenario]
            / "tables"
            / "continuation_round_metrics.csv"
        )
        captured = captured_round[
            captured_round["monitoring_round"].astype(int) == 1
        ].iloc[0]
        evaluated = evaluation[
            evaluation["scenario"].eq(scenario)
            & evaluation["method"].eq("weighted_fedavg")
        ].iloc[0]
        macro_diff = abs(
            float(captured["val_macro_f1"])
            - float(evaluated["validation_macro_f1"])
        )
        target_diff = abs(
            float(captured["val_source_to_target_rate"])
            - float(evaluated["source_to_target_rate"])
        )
        plain_reproduction_rows.append({
            "scenario": scenario,
            "macro_f1_abs_diff": macro_diff,
            "source_to_target_abs_diff": target_diff,
            "passes_1e8": bool(
                macro_diff <= 1e-8 and target_diff <= 1e-8
            ),
        })
    reproduction = pd.DataFrame(plain_reproduction_rows)
    reproduction.to_csv(
        tables / "v319a_plain_fedavg_reproduction.csv",
        index=False,
    )
    if not reproduction["passes_1e8"].all():
        raise RuntimeError(
            "Offline FedAvg reconstruction did not reproduce V3.18B"
        )

    decision = {
        "experiment_version": "3.19A",
        "stage": "one_round_robust_center_and_aggregation_audit",
        "common_round_one_reference_state_exact": True,
        "plain_fedavg_reproduced_within_1e8": True,
        "robust_center_reference_selected":
            center_selected is not None,
        "selected_robust_center_method": (
            None if center_selected is None
            else str(center_selected["method"])
        ),
        "standalone_robust_aggregator_candidate_selected":
            standalone_selected is not None,
        "selected_standalone_aggregator_method": (
            None if standalone_selected is None
            else str(standalone_selected["method"])
        ),
        "method_frozen": False,
        "federated_training_rerun": False,
        "training_or_aggregation_modified": False,
        "test_sets_accessed": False,
        "next_stage": (
            "V3.19B integrated four-round risk-adaptive reconstruction "
            "using the selected robust center, while retaining standard "
            "robust aggregators as baselines."
            if center_selected is not None
            else
            "No robust center passed the pre-registered one-round criteria. "
            "Do not integrate yet; inspect center geometry and revise the "
            "center construction once."
        ),
    }
    pd.DataFrame([decision]).to_csv(
        tables / "v319a_decision.csv",
        index=False,
    )
    with (output / "v319a_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(decision, f, indent=2)

    source_rates = evaluation[
        ~evaluation["scenario"].eq("Clean")
    ].copy()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    width = 0.12
    x = np.arange(len(ATTACK_SOURCES))
    for index, method in enumerate(METHOD_ORDER):
        values = []
        for source in ATTACK_SOURCES:
            values.append(
                float(
                    source_rates[
                        source_rates["source_class"].eq(source)
                        & source_rates["method"].eq(method)
                    ]["source_to_target_rate"].iloc[0]
                )
            )
        ax.bar(
            x + (index - (len(METHOD_ORDER) - 1) / 2) * width,
            values,
            width,
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(ATTACK_SOURCES)
    ax.set_ylabel("Source-to-Benign rate")
    ax.set_title("V3.19A one-round robust aggregation attack rates")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, figures / "source_to_target_rates")

    fig, ax = plt.subplots(figsize=(11, 6))
    plot = summary[
        ~summary["method"].eq("weighted_fedavg")
    ].copy()
    ax.bar(
        np.arange(len(plot)),
        plot["mean_center_error_reduction"],
    )
    ax.axhline(0.40, linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(plot)))
    ax.set_xticklabels(
        plot["method"],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("Mean aggregate-center error reduction")
    ax.set_title("V3.19A robust-center stability")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures / "center_error_reduction")

    fig, ax = plt.subplots(figsize=(11, 6))
    clean_plot = summary.copy()
    ax.bar(
        np.arange(len(clean_plot)),
        clean_plot[
            "clean_macro_f1_delta_vs_weighted_fedavg"
        ],
    )
    ax.axhline(-0.005, linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(clean_plot)))
    ax.set_xticklabels(
        clean_plot["method"],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("Clean macro-F1 delta vs weighted FedAvg")
    ax.set_title("V3.19A clean utility")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures / "clean_macro_f1_delta")

    print("V3.19A robust-center audit complete")
    print()
    print("METHOD SUMMARY")
    display_columns = [
        "method",
        "clean_macro_f1",
        "clean_macro_f1_delta_vs_weighted_fedavg",
        "clean_relative_center_distortion",
        "mean_center_error_reduction",
        "minimum_center_error_reduction",
        "mean_source_target_excess_removed",
        "minimum_source_target_excess_removed",
        "mean_attack_macro_f1_recovery_vs_plain",
        "minimum_attack_macro_f1_recovery_vs_plain",
        "passes_center_reference_criteria",
        "passes_standalone_robust_aggregator_criteria",
        "selected_as_robust_center_reference",
        "selected_as_standalone_aggregator_candidate",
    ]
    print(
        summary.sort_values(
            [
                "selected_as_robust_center_reference",
                "mean_center_error_reduction",
            ],
            ascending=[False, False],
        )[display_columns].to_string(index=False)
    )
    print()
    print("DECISION")
    print(
        pd.read_csv(
            tables / "v319a_decision.csv"
        ).to_string(index=False)
    )
    print()
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

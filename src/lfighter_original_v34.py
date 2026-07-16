"""Faithful, instrumented adaptation of the original LFighter rule for CIC IoT-DIAD.

The decision logic follows the original repository's multiclass LFD.aggregate method:

1. Compute global-minus-local updates for the final classifier weight and bias.
2. Rank classes by the summed row-norm and absolute-bias update magnitude.
3. Retain the two most active classes.
4. Cluster client update rows for those two classes with KMeans(k=2).
5. Compute the original cluster-dissimilarity score.
6. Admit the cluster selected by the original rule.
7. Equally average admitted local states, matching the original implementation.

Additional diagnostics are returned for reproducible security evaluation. These diagnostics
do not alter the original decision rule.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


StateDict = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class LFighterRoundResult:
    aggregated_state: StateDict
    client_decisions: pd.DataFrame
    class_salience: pd.DataFrame
    cluster_diagnostics: pd.DataFrame
    summary: Dict[str, object]


def _classifier_keys(
    reference_state: Mapping[str, torch.Tensor],
    num_classes: int,
) -> Tuple[str, str]:
    """Locate the final multiclass classifier weight and bias in a state dict."""
    candidates: List[Tuple[str, str]] = []
    for key, tensor in reference_state.items():
        if not key.endswith(".weight") or tensor.ndim != 2:
            continue
        if int(tensor.shape[0]) != int(num_classes):
            continue
        bias_key = key[:-7] + ".bias"
        bias = reference_state.get(bias_key)
        if bias is not None and bias.ndim == 1 and int(bias.shape[0]) == int(num_classes):
            candidates.append((key, bias_key))
    if not candidates:
        raise KeyError(
            "Could not locate a final classifier whose output dimension matches "
            f"num_classes={num_classes}."
        )
    # The final classifier normally appears last in state_dict insertion order.
    return candidates[-1]


def _cluster_dissimilarity_exact(cluster_vectors: np.ndarray, cluster_size: int, total: int) -> float:
    """Reproduce the original LFighter weighted dissimilarity expression."""
    if cluster_size <= 0:
        return float("nan")
    similarities = cosine_similarity(cluster_vectors) - np.eye(cluster_size)
    minimum_similarity = np.min(similarities, axis=1)
    return float(cluster_size / total * (1.0 - np.mean(minimum_similarity)))


def _equal_average_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    admitted_mask: Sequence[bool],
    reference_state: Mapping[str, torch.Tensor],
) -> StateDict:
    admitted = [state for state, keep in zip(states, admitted_mask) if bool(keep)]
    if not admitted:
        admitted = list(states)
    result: StateDict = {}
    denominator = float(len(admitted))
    for key, reference in reference_state.items():
        if torch.is_floating_point(reference):
            accumulator = torch.zeros_like(reference, dtype=torch.float64)
            for state in admitted:
                accumulator += state[key].detach().cpu().to(torch.float64)
            result[key] = (accumulator / denominator).to(reference.dtype)
        else:
            result[key] = reference.detach().cpu().clone()
    return result


def _binary_detection_metrics(
    actual_malicious: np.ndarray,
    rejected: np.ndarray,
) -> Dict[str, float | int]:
    actual_malicious = actual_malicious.astype(bool)
    rejected = rejected.astype(bool)
    benign = ~actual_malicious
    admitted = ~rejected

    tp = int(np.sum(actual_malicious & rejected))
    fn = int(np.sum(actual_malicious & admitted))
    fp = int(np.sum(benign & rejected))
    tn = int(np.sum(benign & admitted))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    benign_retention = tn / max(tn + fp, 1)
    false_rejection = fp / max(tn + fp, 1)
    malicious_admission = fn / max(tp + fn, 1)

    return {
        "true_positive_malicious_rejected": tp,
        "false_negative_malicious_admitted": fn,
        "false_positive_benign_rejected": fp,
        "true_negative_benign_admitted": tn,
        "malicious_detection_precision": float(precision),
        "malicious_rejection_recall": float(recall),
        "malicious_detection_f1": float(f1),
        "benign_retention_rate": float(benign_retention),
        "benign_false_rejection_rate": float(false_rejection),
        "malicious_admission_rate": float(malicious_admission),
    }


def aggregate_original_lfighter(
    reference_state: Mapping[str, torch.Tensor],
    local_states: Sequence[Mapping[str, torch.Tensor]],
    client_ids: Sequence[int],
    num_classes: int,
    class_names: Sequence[str],
    malicious_client_ids: Iterable[int] = (),
    kmeans_seed: int = 0,
) -> LFighterRoundResult:
    """Apply the original LFighter multiclass aggregation rule with diagnostics."""
    if len(local_states) < 2:
        raise ValueError("LFighter requires at least two local updates.")
    if len(local_states) != len(client_ids):
        raise ValueError("local_states and client_ids must have identical lengths.")
    if len(class_names) != num_classes:
        raise ValueError("class_names length must equal num_classes.")

    weight_key, bias_key = _classifier_keys(reference_state, num_classes)
    global_weight = reference_state[weight_key].detach().cpu().numpy()
    global_bias = reference_state[bias_key].detach().cpu().numpy()

    weight_updates = np.stack(
        [
            global_weight - state[weight_key].detach().cpu().numpy()
            for state in local_states
        ],
        axis=0,
    )
    bias_updates = np.stack(
        [
            global_bias - state[bias_key].detach().cpu().numpy()
            for state in local_states
        ],
        axis=0,
    )

    # Exact multiclass class-activity calculation in the original implementation.
    weight_norms = np.linalg.norm(weight_updates, axis=-1)
    class_salience_values = np.sum(weight_norms, axis=0) + np.sum(np.abs(bias_updates), axis=0)
    selected_classes = np.argsort(class_salience_values)[-2:]

    feature_matrix = np.stack(
        [weight_updates[index, selected_classes].reshape(-1) for index in range(len(local_states))],
        axis=0,
    )
    unique_vectors = np.unique(feature_matrix, axis=0)
    fallback_reason = ""

    if unique_vectors.shape[0] < 2:
        cluster_labels = np.zeros(len(local_states), dtype=int)
        good_cluster = 0
        cluster_scores = {0: 0.0, 1: float("nan")}
        admitted_mask = np.ones(len(local_states), dtype=bool)
        fallback_reason = "fewer_than_two_unique_client_vectors_admit_all"
    else:
        kmeans = KMeans(n_clusters=2, random_state=kmeans_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        cluster_vectors = {
            label: feature_matrix[cluster_labels == label]
            for label in (0, 1)
        }
        cluster_scores = {
            label: _cluster_dissimilarity_exact(
                cluster_vectors[label],
                int(cluster_vectors[label].shape[0]),
                len(local_states),
            )
            for label in (0, 1)
        }
        # Exact original choice: initialize 0, switch to 1 when d0 < d1.
        good_cluster = 1 if cluster_scores[0] < cluster_scores[1] else 0
        admitted_mask = cluster_labels == good_cluster
        if int(np.sum(admitted_mask)) == 0:
            admitted_mask = np.ones(len(local_states), dtype=bool)
            fallback_reason = "empty_admitted_cluster_admit_all"

    aggregated_state = _equal_average_states(
        states=local_states,
        admitted_mask=admitted_mask,
        reference_state=reference_state,
    )

    malicious_set = {int(value) for value in malicious_client_ids}
    actual_malicious = np.asarray([int(value) in malicious_set for value in client_ids])
    rejected = ~admitted_mask
    update_norms = np.linalg.norm(weight_updates.reshape(len(local_states), -1), axis=1)

    client_table = pd.DataFrame(
        {
            "client_id": [int(value) for value in client_ids],
            "cluster_label": cluster_labels.astype(int),
            "admitted": admitted_mask.astype(bool),
            "rejected": rejected.astype(bool),
            "actual_malicious": actual_malicious.astype(bool),
            "correct_security_decision": (actual_malicious == rejected).astype(bool),
            "final_weight_update_l2": update_norms.astype(float),
            "selected_class_1_id": int(selected_classes[0]),
            "selected_class_1_name": class_names[int(selected_classes[0])],
            "selected_class_2_id": int(selected_classes[1]),
            "selected_class_2_name": class_names[int(selected_classes[1])],
            "selected_good_cluster": int(good_cluster),
        }
    )

    class_table = pd.DataFrame(
        {
            "class_id": np.arange(num_classes, dtype=int),
            "class_name": list(class_names),
            "lfighter_salience": class_salience_values.astype(float),
            "selected_top_two": [int(index) in set(map(int, selected_classes)) for index in range(num_classes)],
            "salience_rank_desc": pd.Series(class_salience_values).rank(method="min", ascending=False).astype(int),
        }
    )

    cluster_rows = []
    for label in (0, 1):
        mask = cluster_labels == label
        malicious_count = int(np.sum(actual_malicious & mask))
        benign_count = int(np.sum((~actual_malicious) & mask))
        cluster_rows.append(
            {
                "cluster_label": label,
                "cluster_size": int(np.sum(mask)),
                "dissimilarity": float(cluster_scores[label]),
                "selected_good_cluster": bool(label == good_cluster),
                "malicious_clients": malicious_count,
                "benign_clients": benign_count,
                "malicious_fraction": malicious_count / max(int(np.sum(mask)), 1),
            }
        )
    cluster_table = pd.DataFrame(cluster_rows)

    security = _binary_detection_metrics(actual_malicious, rejected)
    selected_set = set(map(int, selected_classes))
    summary: Dict[str, object] = {
        "classifier_weight_key": weight_key,
        "classifier_bias_key": bias_key,
        "selected_class_1_id": int(selected_classes[0]),
        "selected_class_1_name": class_names[int(selected_classes[0])],
        "selected_class_2_id": int(selected_classes[1]),
        "selected_class_2_name": class_names[int(selected_classes[1])],
        "selected_class_pair": "|".join(class_names[int(value)] for value in selected_classes),
        "good_cluster": int(good_cluster),
        "admitted_client_count": int(np.sum(admitted_mask)),
        "rejected_client_count": int(np.sum(rejected)),
        "admitted_clients": "|".join(str(int(client_ids[i])) for i in np.where(admitted_mask)[0]),
        "rejected_clients": "|".join(str(int(client_ids[i])) for i in np.where(rejected)[0]),
        "fallback_reason": fallback_reason,
        **security,
    }
    return LFighterRoundResult(
        aggregated_state=aggregated_state,
        client_decisions=client_table,
        class_salience=class_table,
        cluster_diagnostics=cluster_table,
        summary=summary,
    )

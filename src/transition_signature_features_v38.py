"""Class-conditional transition-signature utilities for V3.8."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from federated_iot_v26 import (
    CLASS_NAMES,
    NUM_CLASSES,
    make_loader,
    metric_dict,
    stable_softmax,
)
from run_targeted_label_flip_v292 import attack_metrics_from_matrix


def balanced_probe_indices(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for class_id in range(NUM_CLASSES):
        candidates = np.where(y == class_id)[0]
        if len(candidates) == 0:
            raise ValueError(f"Validation split contains no rows for class {class_id}")
        count = min(int(per_class), len(candidates))
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    result = np.asarray(selected, dtype=np.int64)
    rng.shuffle(result)
    return result


def predict_probabilities(
    model: torch.nn.Module, X: np.ndarray, batch_size: int
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            features = torch.from_numpy(
                X[start : start + batch_size].astype(np.float32, copy=False)
            )
            outputs.append(model(features).cpu().numpy())
    return stable_softmax(np.concatenate(outputs, axis=0))


def class_conditional_probability_means(
    probabilities: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    means = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for class_id in range(NUM_CLASSES):
        mask = labels == class_id
        if not mask.any():
            raise ValueError(f"Probe contains no rows for class {class_id}")
        means[class_id] = probabilities[mask].mean(axis=0)
    return means


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(matrix, dtype=np.float64), 1e-12, None)
    return values / np.clip(values.sum(axis=1, keepdims=True), 1e-12, None)


def rowwise_js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = normalize_rows(left)
    q = normalize_rows(right)
    midpoint = 0.5 * (p + q)
    kl_p = np.sum(p * np.log(p / midpoint), axis=1)
    kl_q = np.sum(q * np.log(q / midpoint), axis=1)
    return 0.5 * (kl_p + kl_q)


def evaluate_validation(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    source_id: int,
    target_id: int,
    batch_size: int,
) -> tuple[Dict[str, float], Dict[str, float]]:
    loader = make_loader(X, y, batch_size=batch_size, shuffle=False)
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for features, batch_labels in loader:
            logits.append(model(features).cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    probabilities = stable_softmax(np.concatenate(logits))
    labels_array = np.concatenate(labels)
    metrics = metric_dict(labels_array, probabilities)
    matrix = confusion_matrix(
        labels_array, probabilities.argmax(axis=1), labels=np.arange(NUM_CLASSES)
    )
    return metrics, attack_metrics_from_matrix(matrix, source_id, target_id)


def matrix_long_rows(
    round_id: int,
    client_id: int,
    actual_malicious: bool,
    reference_means: np.ndarray,
    local_means: np.ndarray,
    consensus_means: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for source_id in range(NUM_CLASSES):
        for target_id in range(NUM_CLASSES):
            reference_probability = float(reference_means[source_id, target_id])
            local_probability = float(local_means[source_id, target_id])
            consensus_probability = float(consensus_means[source_id, target_id])
            signed_drift = local_probability - reference_probability
            consensus_residual = local_probability - consensus_probability
            rows.append(
                {
                    "round": int(round_id),
                    "client_id": int(client_id),
                    "actual_malicious": bool(actual_malicious),
                    "source_id": int(source_id),
                    "source_name": CLASS_NAMES[source_id],
                    "target_id": int(target_id),
                    "target_name": CLASS_NAMES[target_id],
                    "is_diagonal": bool(source_id == target_id),
                    "reference_probability": reference_probability,
                    "local_probability": local_probability,
                    "round_consensus_probability": consensus_probability,
                    "signed_drift_from_global_reference": float(signed_drift),
                    "absolute_drift_from_global_reference": float(abs(signed_drift)),
                    "positive_offdiagonal_growth": float(
                        max(signed_drift, 0.0) if source_id != target_id else 0.0
                    ),
                    "diagonal_confidence_loss": float(
                        max(-signed_drift, 0.0) if source_id == target_id else 0.0
                    ),
                    "signed_residual_from_round_consensus": float(consensus_residual),
                    "absolute_residual_from_round_consensus": float(abs(consensus_residual)),
                    "transition_pair": f"{CLASS_NAMES[source_id]}->{CLASS_NAMES[target_id]}",
                }
            )
    return rows


def summary_row(
    round_id: int,
    client_id: int,
    actual_malicious: bool,
    client_samples: int,
    poisoned_rows: int,
    source_id: int,
    target_id: int,
    reference_means: np.ndarray,
    local_means: np.ndarray,
    consensus_means: np.ndarray,
) -> Dict[str, object]:
    drift = local_means - reference_means
    consensus_residual = local_means - consensus_means
    offdiag = drift.copy()
    np.fill_diagonal(offdiag, -np.inf)
    max_source, max_target = np.unravel_index(np.argmax(offdiag), offdiag.shape)
    max_growth = float(max(offdiag[max_source, max_target], 0.0))
    diagonal_losses = np.maximum(-np.diag(drift), 0.0)
    reference_js = rowwise_js(local_means, reference_means)
    consensus_js = rowwise_js(local_means, consensus_means)
    return {
        "round": int(round_id),
        "client_id": int(client_id),
        "actual_malicious": bool(actual_malicious),
        "client_samples": int(client_samples),
        "poisoned_rows": int(poisoned_rows),
        "source_target_growth": float(max(drift[source_id, target_id], 0.0)),
        "source_target_signed_drift": float(drift[source_id, target_id]),
        "source_diagonal_confidence_loss": float(
            max(-drift[source_id, source_id], 0.0)
        ),
        "max_offdiag_growth": max_growth,
        "max_offdiag_source_id": int(max_source),
        "max_offdiag_source_name": CLASS_NAMES[int(max_source)],
        "max_offdiag_target_id": int(max_target),
        "max_offdiag_target_name": CLASS_NAMES[int(max_target)],
        "max_offdiag_pair": f"{CLASS_NAMES[int(max_source)]}->{CLASS_NAMES[int(max_target)]}",
        "mean_positive_offdiag_growth": float(
            np.maximum(
                drift - np.eye(NUM_CLASSES, dtype=np.float64) * drift,
                0.0,
            ).sum()
            / max(NUM_CLASSES * (NUM_CLASSES - 1), 1)
        ),
        "sum_positive_offdiag_growth": float(
            np.maximum(
                drift - np.eye(NUM_CLASSES, dtype=np.float64) * drift,
                0.0,
            ).sum()
        ),
        "mean_diagonal_confidence_loss": float(diagonal_losses.mean()),
        "max_diagonal_confidence_loss": float(diagonal_losses.max()),
        "frobenius_drift": float(np.linalg.norm(drift)),
        "mean_row_js_reference": float(reference_js.mean()),
        "max_row_js_reference": float(reference_js.max()),
        "frobenius_to_consensus": float(np.linalg.norm(consensus_residual)),
        "mean_row_js_consensus": float(consensus_js.mean()),
        "max_row_js_consensus": float(consensus_js.max()),
        "source_target_consensus_residual": float(
            max(consensus_residual[source_id, target_id], 0.0)
        ),
    }

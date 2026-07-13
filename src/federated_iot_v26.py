"""Clean federated-learning utilities for CIC IoT-DIAD V2.6."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = [
    "Benign",
    "BruteForce",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web-Based",
]
NUM_CLASSES = len(CLASS_NAMES)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_protocol_arrays(path: Path) -> Dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prepared NPZ file not found: {path}")
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    required = {
        "X_train",
        "y_train",
        "X_val",
        "y_val",
        "X_test_natural",
        "y_test_natural",
        "X_test_diagnostic",
        "y_test_diagnostic",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise KeyError(f"Prepared NPZ is missing arrays: {missing}")
    return arrays


def sqrt_class_weights(y: np.ndarray, max_weight: float = 4.0) -> np.ndarray:
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    maximum = counts.max()
    weights = np.sqrt(maximum / np.clip(counts, 1.0, None))
    weights = np.clip(weights, 1.0, max_weight)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    entropy = -np.sum(probabilities * np.log(probabilities))
    maximum = math.log(len(counts)) if len(counts) > 1 else 1.0
    return float(entropy / maximum) if maximum > 0 else 0.0


def dirichlet_partition(
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_client_samples: int,
    min_client_classes: int,
    max_attempts: int = 1000,
) -> List[np.ndarray]:
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2")
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")

    rng = np.random.default_rng(seed)
    labels = np.asarray(y, dtype=np.int64)

    for attempt in range(1, max_attempts + 1):
        clients: List[List[int]] = [[] for _ in range(num_clients)]

        for class_id in range(NUM_CLASSES):
            class_indices = np.where(labels == class_id)[0].copy()
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            counts = rng.multinomial(len(class_indices), proportions)

            start = 0
            for client_id, count in enumerate(counts):
                if count > 0:
                    clients[client_id].extend(class_indices[start : start + count].tolist())
                start += count

        valid = True
        arrays: List[np.ndarray] = []
        for indices in clients:
            client_indices = np.asarray(indices, dtype=np.int64)
            rng.shuffle(client_indices)
            arrays.append(client_indices)
            if len(client_indices) < min_client_samples:
                valid = False
                break
            present_classes = np.unique(labels[client_indices]).size
            if present_classes < min_client_classes:
                valid = False
                break

        if valid:
            return arrays

    raise RuntimeError(
        f"Could not create a valid partition after {max_attempts} attempts. "
        f"Try reducing min_client_samples or min_client_classes."
    )


def partition_manifest(
    client_indices: Sequence[np.ndarray],
    y: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    matrix_rows = []

    for client_id, indices in enumerate(client_indices):
        counts = np.bincount(y[indices], minlength=NUM_CLASSES)
        total = int(counts.sum())
        dominant = int(counts.argmax())
        row = {
            "client_id": client_id,
            "samples": total,
            "classes_present": int((counts > 0).sum()),
            "normalized_class_entropy": _entropy_from_counts(counts),
            "dominant_class_id": dominant,
            "dominant_class_name": CLASS_NAMES[dominant],
            "dominant_class_proportion": float(counts[dominant] / max(total, 1)),
        }
        summary_rows.append(row)

        matrix_row = {"client_id": client_id}
        for class_id, class_name in enumerate(CLASS_NAMES):
            matrix_row[class_name] = int(counts[class_id])
        matrix_rows.append(matrix_row)

    return pd.DataFrame(summary_rows), pd.DataFrame(matrix_rows)


def save_partitions(client_indices: Sequence[np.ndarray], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        f"client_{client_id:03d}": np.asarray(indices, dtype=np.int64)
        for client_id, indices in enumerate(client_indices)
    }
    np.savez_compressed(path, **payload)

    digest = hashlib.sha256()
    for key in sorted(payload):
        digest.update(key.encode("utf-8"))
        digest.update(payload[key].tobytes())
    return digest.hexdigest()


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32, copy=False)),
        torch.from_numpy(y.astype(np.int64, copy=False)),
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        generator=generator,
    )


def train_local_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    class_weights: np.ndarray,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    seed: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    loader = make_loader(X, y, batch_size=batch_size, shuffle=True, seed=seed)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    model.train()
    loss_sum = 0.0
    correct = 0
    seen = 0
    started = time.time()

    for _ in range(local_epochs):
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

            batch_size_actual = labels.numel()
            loss_sum += float(loss.item()) * batch_size_actual
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += batch_size_actual

    metrics = {
        "local_train_loss": float(loss_sum / max(seen, 1)),
        "local_train_accuracy": float(correct / max(seen, 1)),
        "local_samples_seen": int(seen),
        "local_train_seconds": float(time.time() - started),
    }
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return state, metrics


def weighted_average_states(
    states: Sequence[Dict[str, torch.Tensor]],
    sample_counts: Sequence[int],
    reference_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("Total aggregation weight must be positive")

    averaged: Dict[str, torch.Tensor] = {}
    for key, reference in reference_state.items():
        if torch.is_floating_point(reference):
            accumulator = torch.zeros_like(reference, dtype=torch.float64)
            for state, count in zip(states, sample_counts):
                accumulator += state[key].to(torch.float64) * (float(count) / total)
            averaged[key] = accumulator.to(reference.dtype)
        else:
            averaged[key] = reference.clone()
    return averaged


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
    values -= values.max(axis=1, keepdims=True)
    exponentials = np.exp(np.clip(values, -700.0, 0.0))
    sums = exponentials.sum(axis=1, keepdims=True)
    zero_rows = sums[:, 0] <= 0
    if zero_rows.any():
        exponentials[zero_rows] = 1.0
        sums = exponentials.sum(axis=1, keepdims=True)
    return exponentials / sums


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correctness = (predicted == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correctness[mask].mean()) - float(confidence[mask].mean())
            )
    return float(value)


def collect_probabilities(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = make_loader(X, y, batch_size=batch_size, shuffle=False)
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for features, batch_labels in loader:
            logits.append(model(features).cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    logits_array = np.concatenate(logits)
    labels_array = np.concatenate(labels)
    return stable_softmax(logits_array), labels_array


def metric_dict(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, predicted, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=np.arange(NUM_CLASSES))
        ),
        "ece_15bin": expected_calibration_error(y_true, probabilities, bins=15),
    }


def evaluation_artifacts(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    split: str,
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray, pd.DataFrame]:
    probabilities, labels = collect_probabilities(model, X, y, batch_size)
    predicted = probabilities.argmax(axis=1)
    metrics = metric_dict(labels, probabilities)

    report = classification_report(
        labels,
        predicted,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    per_class_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        values = report[class_name]
        per_class_rows.append(
            {
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1_score": float(values["f1-score"]),
                "support": int(values["support"]),
            }
        )

    matrix = confusion_matrix(labels, predicted, labels=np.arange(NUM_CLASSES))
    prediction_table = pd.DataFrame(
        {
            "true_class_id": labels,
            "true_class_name": [CLASS_NAMES[int(value)] for value in labels],
            "predicted_class_id": predicted,
            "predicted_class_name": [CLASS_NAMES[int(value)] for value in predicted],
            "confidence": probabilities.max(axis=1),
        }
    )
    for class_id, class_name in enumerate(CLASS_NAMES):
        prediction_table[f"probability_{class_name}"] = probabilities[:, class_id]

    return metrics, pd.DataFrame(per_class_rows), matrix, prediction_table

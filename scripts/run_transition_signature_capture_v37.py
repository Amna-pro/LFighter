#!/usr/bin/env python3
"""Capture full class-conditional prediction-transition signatures, V3.7.

This is a diagnostic data-capture experiment, not the final defense. It runs a
matched clean or targeted-label-flip FedAvg trajectory and records the complete
C x C class-conditional prediction matrix for every local client model in every
round. Test sets are not touched.

The script is resumable at completed-round boundaries.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from federated_iot_v26 import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    load_protocol_arrays,
    make_loader,
    metric_dict,
    partition_manifest,
    set_seed,
    sqrt_class_weights,
    stable_softmax,
    train_local_model,
    weighted_average_states,
)
from neural_models_v24 import build_model  # noqa: E402
from run_targeted_label_flip_v292 import (  # noqa: E402
    attack_metrics_from_matrix,
    load_fixed_partitions,
    prepare_static_attack,
    read_clean_partition_hash,
    save_poisoned_indices,
)

DEFAULT_MALICIOUS_CLIENTS = "1,7,8,10,14,15,17,18"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture V3.7 full class-transition signatures under matched FedAvg."
    )
    parser.add_argument("--mode", choices=["clean", "strong_attack"], required=True)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-seed-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-class", default="DDoS")
    parser.add_argument("--target-class", default="Benign")
    parser.add_argument("--malicious-clients", default=DEFAULT_MALICIOUS_CLIENTS)
    parser.add_argument("--poison-fraction", type=float, default=1.0)
    parser.add_argument("--min-source-samples", type=int, default=1000)
    parser.add_argument("--attack-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--participation-rate", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--probe-per-class", type=int, default=48)
    parser.add_argument("--probe-seed", type=int, default=3701)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_client_ids(text: str) -> List[int]:
    return sorted({int(value.strip()) for value in text.split(",") if value.strip()})


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


def read_table(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def save_partial(
    tables_dir: Path,
    round_rows: List[Dict[str, object]],
    local_rows: List[Dict[str, object]],
    signature_rows: List[Dict[str, object]],
    summary_rows: List[Dict[str, object]],
    consensus_rows: List[Dict[str, object]],
) -> None:
    pd.DataFrame(round_rows).to_csv(
        tables_dir / "round_metrics_partial.csv", index=False
    )
    pd.DataFrame(local_rows).to_csv(
        tables_dir / "local_client_training_metrics_partial.csv", index=False
    )
    pd.DataFrame(signature_rows).to_csv(
        tables_dir / "transition_signature_long_partial.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        tables_dir / "client_signature_summary_partial.csv", index=False
    )
    pd.DataFrame(consensus_rows).to_csv(
        tables_dir / "round_consensus_signature_long_partial.csv", index=False
    )


def plot_validation(table: pd.DataFrame, source: str, target: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(table["round"], table["val_macro_f1"], marker="o", label="Validation macro F1")
    ax.plot(
        table["round"],
        table["val_source_to_target_rate"],
        marker="s",
        label=f"{source} to {target} rate",
    )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Matched FedAvg transition-signature trajectory")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out)


def plot_signature_groups(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped = (
        table.groupby(["round", "actual_malicious"], as_index=False)[
            "source_target_growth"
        ]
        .mean()
        .sort_values(["actual_malicious", "round"])
    )
    for status, group in grouped.groupby("actual_malicious"):
        label = "Malicious clients" if bool(status) else "Benign clients"
        ax.plot(group["round"], group["source_target_growth"], marker="o", label=label)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Mean source-target probability growth")
    ax.set_title("Class-localized transition signal")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out)


def plot_mean_drift_heatmap(
    long_table: pd.DataFrame, actual_malicious: bool, title: str, out: Path
) -> None:
    subset = long_table[long_table["actual_malicious"].eq(actual_malicious)]
    if subset.empty:
        return
    matrix = (
        subset.groupby(["source_id", "target_id"])[
            "signed_drift_from_global_reference"
        ]
        .mean()
        .unstack("target_id")
        .reindex(index=range(NUM_CLASSES), columns=range(NUM_CLASSES))
        .to_numpy()
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Mean probability drift")
    save_figure(fig, out)


def main() -> int:
    args = parse_args()
    if args.source_class not in CLASS_NAMES or args.target_class not in CLASS_NAMES:
        raise ValueError("Unknown source or target class")
    if args.source_class == args.target_class:
        raise ValueError("Source and target classes must differ")
    if not math.isclose(args.participation_rate, 1.0):
        raise ValueError("V3.7 capture is frozen to full client participation")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")

    source_id = CLASS_NAMES.index(args.source_class)
    target_id = CLASS_NAMES.index(args.target_class)
    torch.set_num_threads(max(1, args.threads))
    set_seed(args.model_seed)

    data_file = args.data_file.expanduser().resolve()
    partition_file = args.partition_file.expanduser().resolve()
    clean_seed_dir = args.clean_seed_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        if args.overwrite:
            import shutil
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use --resume or --overwrite."
            )

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"
    attack_dir = output_dir / "attack_manifest"
    for path in (tables_dir, figures_dir, checkpoints_dir, attack_dir):
        path.mkdir(parents=True, exist_ok=True)

    arrays = load_protocol_arrays(data_file)
    X_train = arrays["X_train"].astype(np.float32, copy=False)
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    X_val = arrays["X_val"].astype(np.float32, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)

    client_indices, partition_hash = load_fixed_partitions(
        partition_file, expected_clients=args.num_clients, train_rows=len(y_train)
    )
    clean_hash = read_clean_partition_hash(clean_seed_dir)
    if clean_hash != partition_hash:
        raise RuntimeError("Clean seed partition hash does not match supplied partition")

    client_summary, client_matrix = partition_manifest(client_indices, y_train)
    client_summary.to_csv(tables_dir / "client_partition_summary.csv", index=False)
    client_matrix.to_csv(tables_dir / "client_class_counts.csv", index=False)

    if args.mode == "strong_attack":
        malicious_clients = parse_client_ids(args.malicious_clients)
        requested_fraction = len(malicious_clients) / args.num_clients
        malicious_clients, poisoned_positions, poison_manifest, poison_hash = (
            prepare_static_attack(
                client_indices=client_indices,
                y_train=y_train,
                source_id=source_id,
                target_id=target_id,
                malicious_client_fraction=requested_fraction,
                poison_fraction=args.poison_fraction,
                min_source_samples=args.min_source_samples,
                attack_seed=args.attack_seed,
                explicit_malicious_clients=malicious_clients,
            )
        )
    else:
        malicious_clients = []
        poisoned_positions = {
            client_id: np.empty(0, dtype=np.int64)
            for client_id in range(args.num_clients)
        }
        poison_manifest = pd.DataFrame(
            [
                {
                    "client_id": client_id,
                    "is_malicious": False,
                    "client_rows": int(len(indices)),
                    "source_rows_before_poisoning": int(
                        np.sum(y_train[indices] == source_id)
                    ),
                    "poisoned_source_rows": 0,
                    "client_source_poison_rate": 0.0,
                    "source_rows_after_poisoning": int(
                        np.sum(y_train[indices] == source_id)
                    ),
                    "target_rows_added_by_poisoning": 0,
                }
                for client_id, indices in enumerate(client_indices)
            ]
        )
        digest = hashlib.sha256()
        for client_id in range(args.num_clients):
            digest.update(f"client_{client_id:03d}".encode("utf-8"))
        poison_hash = digest.hexdigest()

    poison_manifest.to_csv(
        attack_dir / "malicious_client_poison_manifest.csv", index=False
    )
    save_poisoned_indices(
        client_indices, poisoned_positions, attack_dir / "poisoned_indices.npz"
    )
    pd.DataFrame(
        [
            {
                "mode": args.mode,
                "source_class": args.source_class,
                "target_class": args.target_class,
                "malicious_clients": "|".join(map(str, malicious_clients)),
                "malicious_client_count": len(malicious_clients),
                "partition_hash_sha256": partition_hash,
                "poison_index_hash_sha256": poison_hash,
            }
        ]
    ).to_csv(attack_dir / "attack_summary.csv", index=False)

    probe_indices = balanced_probe_indices(
        y_val, args.probe_per_class, args.probe_seed
    )
    X_probe = X_val[probe_indices]
    y_probe = y_val[probe_indices]
    pd.DataFrame(
        {
            "validation_row_index": probe_indices,
            "class_id": y_probe,
            "class_name": [CLASS_NAMES[int(value)] for value in y_probe],
        }
    ).to_csv(tables_dir / "prediction_probe_manifest.csv", index=False)

    class_weights = sqrt_class_weights(y_train, args.max_class_weight)
    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)

    round_rows: List[Dict[str, object]] = []
    local_rows: List[Dict[str, object]] = []
    signature_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    consensus_rows: List[Dict[str, object]] = []
    best_macro_f1 = -math.inf
    best_round = 0
    start_round = 1

    last_checkpoint_path = checkpoints_dir / "last_round_model.pt"
    if args.resume and last_checkpoint_path.exists():
        checkpoint = torch.load(
            last_checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("partition_hash") != partition_hash:
            raise RuntimeError("Resume checkpoint partition hash mismatch")
        if checkpoint.get("poison_index_hash") != poison_hash:
            raise RuntimeError("Resume checkpoint poison hash mismatch")
        if int(checkpoint.get("model_seed")) != int(args.model_seed):
            raise RuntimeError("Resume checkpoint model seed mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        start_round = int(checkpoint["round"]) + 1
        best_macro_f1 = float(checkpoint.get("best_macro_f1", -math.inf))
        best_round = int(checkpoint.get("best_round", 0))
        round_rows = read_table(tables_dir / "round_metrics_partial.csv")
        local_rows = read_table(
            tables_dir / "local_client_training_metrics_partial.csv"
        )
        signature_rows = read_table(
            tables_dir / "transition_signature_long_partial.csv"
        )
        summary_rows = read_table(
            tables_dir / "client_signature_summary_partial.csv"
        )
        consensus_rows = read_table(
            tables_dir / "round_consensus_signature_long_partial.csv"
        )
        print(f"Resuming from round {start_round}")

    started = time.time()
    print("Transition Signature Audit V3.7 capture")
    print("Mode:", args.mode)
    print("Model seed:", args.model_seed)
    print("Partition hash:", partition_hash)
    print("Malicious clients:", malicious_clients)

    for round_id in range(start_round, args.rounds + 1):
        round_started = time.time()
        reference_state = copy.deepcopy(model.state_dict())
        reference_probabilities = predict_probabilities(
            model, X_probe, args.evaluation_batch_size
        )
        reference_means = class_conditional_probability_means(
            reference_probabilities, y_probe
        )

        local_states: List[Mapping[str, torch.Tensor]] = []
        sample_counts: List[int] = []
        local_matrices: List[np.ndarray] = []
        current_training_rows: List[Dict[str, object]] = []

        for client_id in range(args.num_clients):
            indices = client_indices[client_id]
            local_y = y_train[indices].copy()
            positions = poisoned_positions[client_id]
            if len(positions) > 0:
                local_y[positions] = target_id

            local_model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
            local_model.load_state_dict(reference_state)
            state, training_metrics = train_local_model(
                model=local_model,
                X=X_train[indices],
                y=local_y,
                class_weights=class_weights,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                seed=args.model_seed + round_id * 1000 + client_id,
            )
            probabilities = predict_probabilities(
                local_model, X_probe, args.evaluation_batch_size
            )
            local_means = class_conditional_probability_means(
                probabilities, y_probe
            )
            local_states.append(state)
            sample_counts.append(int(len(indices)))
            local_matrices.append(local_means)
            training_row = {
                "round": int(round_id),
                "client_id": int(client_id),
                "actual_malicious": bool(client_id in malicious_clients),
                "client_samples": int(len(indices)),
                "poisoned_rows": int(len(positions)),
                **training_metrics,
            }
            local_rows.append(training_row)
            current_training_rows.append(training_row)
            del local_model

        matrix_stack = np.stack(local_matrices, axis=0)
        consensus_means = np.median(matrix_stack, axis=0)
        consensus_means = normalize_rows(consensus_means)

        for source_index in range(NUM_CLASSES):
            for target_index in range(NUM_CLASSES):
                consensus_rows.append(
                    {
                        "round": int(round_id),
                        "source_id": int(source_index),
                        "source_name": CLASS_NAMES[source_index],
                        "target_id": int(target_index),
                        "target_name": CLASS_NAMES[target_index],
                        "round_consensus_probability": float(
                            consensus_means[source_index, target_index]
                        ),
                        "global_reference_probability": float(
                            reference_means[source_index, target_index]
                        ),
                        "consensus_minus_global_reference": float(
                            consensus_means[source_index, target_index]
                            - reference_means[source_index, target_index]
                        ),
                    }
                )

        for client_id, local_means in enumerate(local_matrices):
            actual_malicious = client_id in malicious_clients
            signature_rows.extend(
                matrix_long_rows(
                    round_id=round_id,
                    client_id=client_id,
                    actual_malicious=actual_malicious,
                    reference_means=reference_means,
                    local_means=local_means,
                    consensus_means=consensus_means,
                )
            )
            summary_rows.append(
                summary_row(
                    round_id=round_id,
                    client_id=client_id,
                    actual_malicious=actual_malicious,
                    client_samples=sample_counts[client_id],
                    poisoned_rows=int(len(poisoned_positions[client_id])),
                    source_id=source_id,
                    target_id=target_id,
                    reference_means=reference_means,
                    local_means=local_means,
                    consensus_means=consensus_means,
                )
            )

        averaged = weighted_average_states(
            states=local_states,
            sample_counts=sample_counts,
            reference_state=reference_state,
        )
        model.load_state_dict(averaged)

        val_metrics, val_pair = evaluate_validation(
            model=model,
            X=X_val,
            y=y_val,
            source_id=source_id,
            target_id=target_id,
            batch_size=args.evaluation_batch_size,
        )
        current_summary = [
            row for row in summary_rows if int(row["round"]) == round_id
        ]
        row = {
            "round": int(round_id),
            "selected_clients": "|".join(map(str, range(args.num_clients))),
            "selected_client_count": int(args.num_clients),
            "participating_samples": int(sum(sample_counts)),
            "mean_local_train_loss": float(
                np.mean([r["local_train_loss"] for r in current_training_rows])
            ),
            "mean_local_train_accuracy": float(
                np.mean([r["local_train_accuracy"] for r in current_training_rows])
            ),
            "mean_benign_source_target_growth": float(
                np.mean(
                    [
                        r["source_target_growth"]
                        for r in current_summary
                        if not bool(r["actual_malicious"])
                    ]
                )
            ),
            "mean_malicious_source_target_growth": float(
                np.mean(
                    [
                        r["source_target_growth"]
                        for r in current_summary
                        if bool(r["actual_malicious"])
                    ]
                )
                if malicious_clients
                else 0.0
            ),
            "round_seconds": float(time.time() - round_started),
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"val_{key}": value for key, value in val_pair.items()},
        }
        round_rows.append(row)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_round = int(round_id)
            torch.save(
                {
                    "experiment_version": "3.7-capture",
                    "checkpoint_type": "best_validation",
                    "round": int(round_id),
                    "model_state_dict": model.state_dict(),
                    "input_dim": int(X_train.shape[1]),
                    "num_classes": int(NUM_CLASSES),
                    "validation_metrics": val_metrics,
                    "validation_source_target_metrics": val_pair,
                    "partition_hash": partition_hash,
                    "poison_index_hash": poison_hash,
                    "model_seed": int(args.model_seed),
                    "best_macro_f1": float(best_macro_f1),
                    "best_round": int(best_round),
                },
                checkpoints_dir / "best_validation_model.pt",
            )

        torch.save(
            {
                "experiment_version": "3.7-capture",
                "checkpoint_type": "last_round",
                "round": int(round_id),
                "model_state_dict": model.state_dict(),
                "input_dim": int(X_train.shape[1]),
                "num_classes": int(NUM_CLASSES),
                "validation_metrics": val_metrics,
                "validation_source_target_metrics": val_pair,
                "partition_hash": partition_hash,
                "poison_index_hash": poison_hash,
                "model_seed": int(args.model_seed),
                "best_macro_f1": float(best_macro_f1),
                "best_round": int(best_round),
            },
            last_checkpoint_path,
        )
        save_partial(
            tables_dir=tables_dir,
            round_rows=round_rows,
            local_rows=local_rows,
            signature_rows=signature_rows,
            summary_rows=summary_rows,
            consensus_rows=consensus_rows,
        )
        print(
            f"round {round_id:02d}, val macro F1={val_metrics['macro_f1']:.4f}, "
            f"{args.source_class}->{args.target_class}="
            f"{val_pair['source_to_target_rate']:.4%}, "
            f"benign transition={row['mean_benign_source_target_growth']:.4f}, "
            f"malicious transition={row['mean_malicious_source_target_growth']:.4f}, "
            f"seconds={row['round_seconds']:.1f}"
        )

    round_table = pd.DataFrame(round_rows)
    local_table = pd.DataFrame(local_rows)
    signature_table = pd.DataFrame(signature_rows)
    summary_table = pd.DataFrame(summary_rows)
    consensus_table = pd.DataFrame(consensus_rows)

    round_table.to_csv(tables_dir / "round_metrics.csv", index=False)
    local_table.to_csv(
        tables_dir / "local_client_training_metrics.csv", index=False
    )
    signature_table.to_csv(
        tables_dir / "transition_signature_long.csv", index=False
    )
    summary_table.to_csv(
        tables_dir / "client_signature_summary.csv", index=False
    )
    consensus_table.to_csv(
        tables_dir / "round_consensus_signature_long.csv", index=False
    )

    plot_validation(
        round_table,
        args.source_class,
        args.target_class,
        figures_dir / "validation_and_attack_dynamics",
    )
    plot_signature_groups(
        summary_table, figures_dir / "source_target_transition_by_client_type"
    )
    plot_mean_drift_heatmap(
        signature_table,
        actual_malicious=False,
        title="Mean benign-client class-transition drift",
        out=figures_dir / "mean_benign_transition_drift",
    )
    plot_mean_drift_heatmap(
        signature_table,
        actual_malicious=True,
        title="Mean malicious-client class-transition drift",
        out=figures_dir / "mean_malicious_transition_drift",
    )

    best_row = round_table.loc[
        round_table["round"].eq(best_round)
    ].iloc[0]
    metadata = {
        "experiment_version": "3.7-capture",
        "experiment_type": "full_class_conditional_transition_signature_capture",
        "status": "development_diagnostic_not_final_paper_result",
        "aggregation_rule": "sample_weighted_fedavg",
        "mode": args.mode,
        "model_seed": int(args.model_seed),
        "attack_seed": int(args.attack_seed),
        "probe_seed": int(args.probe_seed),
        "probe_rows_per_class": int(args.probe_per_class),
        "probe_rows_total": int(len(probe_indices)),
        "source_class": args.source_class,
        "target_class": args.target_class,
        "malicious_clients": malicious_clients,
        "partition_hash_sha256": partition_hash,
        "clean_seed_partition_hash_sha256": clean_hash,
        "poison_index_hash_sha256": poison_hash,
        "rounds_requested": int(args.rounds),
        "rounds_completed": int(round_table["round"].max()),
        "best_validation_round": int(best_round),
        "best_validation_macro_f1": float(best_macro_f1),
        "best_validation_source_to_target_rate": float(
            best_row["val_source_to_target_rate"]
        ),
        "test_sets_accessed": False,
        "full_transition_dimensions": [
            int(NUM_CLASSES),
            int(NUM_CLASSES),
        ],
        "total_seconds": float(time.time() - started),
    }
    with (output_dir / "transition_signature_v37_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Transition Signature Audit V3.7 capture complete")
    print("Mode:", args.mode)
    print("Best validation round:", best_round)
    print("Best validation macro F1:", f"{best_macro_f1:.6f}")
    print(
        "Best validation source-to-target rate:",
        f"{float(best_row['val_source_to_target_rate']):.6f}",
    )
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

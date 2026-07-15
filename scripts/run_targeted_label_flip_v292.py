#!/usr/bin/env python3
"""Targeted label-flipping attack validation for CIC IoT-DIAD V2.9.

Threat model
------------
A seeded, availability-aware attacker controls a fixed fraction of logical
federated clients that contain at least a minimum number of source-class rows.
A static subset of DDoS labels on those clients is changed to DoS before local
training. Features are not modified. The same poisoned local rows are used in
every communication round.

Scientific safeguards
---------------------
* The exact clean V2.7/V2.8 client partition is reused.
* Clean class weights are computed before poisoning and remain fixed.
* Best checkpoint selection uses clean validation macro F1 only.
* Test sets are evaluated only after checkpoint selection.
* The clean and attacked runs use the same model seed, partition, model,
  optimizer settings, participation rate, and maximum rounds.
* Poisoned indices, client selection, hashes, round-level validation attack
  metrics, checkpoints, CSVs, PNGs, and PDFs are saved.
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
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from federated_iot_v26 import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    evaluation_artifacts,
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


DEFAULT_SOURCE = "DDoS"
DEFAULT_TARGET = "DoS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a targeted source-to-target label-flipping attack."
    )
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-seed-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--source-class", default=DEFAULT_SOURCE)
    parser.add_argument("--target-class", default=DEFAULT_TARGET)
    parser.add_argument("--malicious-client-fraction", type=float, default=0.20)
    parser.add_argument(
        "--malicious-clients",
        default="",
        help="Optional comma-separated explicit malicious client IDs.",
    )
    parser.add_argument("--poison-fraction", type=float, default=0.50)
    parser.add_argument("--min-source-samples", type=int, default=1000)
    parser.add_argument("--attack-seed", type=int, default=42)

    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--participation-rate", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_fixed_partitions(
    path: Path,
    expected_clients: int,
    train_rows: int,
) -> Tuple[List[np.ndarray], str]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Partition file not found: {path}")

    with np.load(path) as payload:
        keys = sorted(payload.files)
        if len(keys) != expected_clients:
            raise ValueError(
                f"Expected {expected_clients} clients, partition file has {len(keys)}."
            )
        partitions = [np.asarray(payload[key], dtype=np.int64) for key in keys]

    all_indices = np.concatenate(partitions)
    if len(all_indices) != train_rows:
        raise ValueError(
            f"Partition rows {len(all_indices)} do not match train rows {train_rows}."
        )
    unique = np.unique(all_indices)
    if len(unique) != train_rows:
        raise ValueError("Partition contains duplicate or missing training indices.")
    if int(unique.min()) != 0 or int(unique.max()) != train_rows - 1:
        raise ValueError("Partition indices do not cover the complete train range.")

    digest = hashlib.sha256()
    for client_id, indices in enumerate(partitions):
        key = f"client_{client_id:03d}"
        digest.update(key.encode("utf-8"))
        digest.update(indices.tobytes())
    return partitions, digest.hexdigest()


def read_clean_partition_hash(clean_seed_dir: Path) -> str:
    metadata_path = clean_seed_dir / "seed_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Clean seed metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    value = metadata.get("partition_hash_sha256")
    if not value:
        raise ValueError("Clean seed metadata has no partition hash.")
    return str(value)


def prepare_static_attack(
    client_indices: Sequence[np.ndarray],
    y_train: np.ndarray,
    source_id: int,
    target_id: int,
    malicious_client_fraction: float,
    poison_fraction: float,
    min_source_samples: int,
    attack_seed: int,
    explicit_malicious_clients: Sequence[int] | None = None,
) -> Tuple[List[int], Dict[int, np.ndarray], pd.DataFrame, str]:
    if source_id == target_id:
        raise ValueError("Source and target classes must differ.")
    if not 0 < malicious_client_fraction <= 1:
        raise ValueError("malicious-client-fraction must be in (0, 1].")
    if not 0 < poison_fraction <= 1:
        raise ValueError("poison-fraction must be in (0, 1].")

    source_counts = []
    for client_id, indices in enumerate(client_indices):
        source_count = int(np.sum(y_train[indices] == source_id))
        source_counts.append((client_id, source_count))

    eligible = [
        client_id
        for client_id, count in source_counts
        if count >= min_source_samples
    ]
    malicious_count = max(
        1,
        int(math.ceil(len(client_indices) * malicious_client_fraction)),
    )
    if len(eligible) < malicious_count:
        raise ValueError(
            f"Only {len(eligible)} clients meet min-source-samples="
            f"{min_source_samples}, but {malicious_count} are required."
        )

    if explicit_malicious_clients:
        malicious_clients = sorted(set(int(value) for value in explicit_malicious_clients))
        if len(malicious_clients) != malicious_count:
            raise ValueError(
                "Explicit malicious client count does not match "
                "malicious-client-fraction."
            )
        invalid = [value for value in malicious_clients if value not in eligible]
        if invalid:
            raise ValueError(
                f"Explicit malicious clients are not eligible: {invalid}"
            )
    else:
        selection_rng = np.random.default_rng(attack_seed)
        malicious_clients = sorted(
            selection_rng.choice(eligible, size=malicious_count, replace=False).tolist()
        )

    poisoned_positions: Dict[int, np.ndarray] = {}
    manifest_rows = []
    hash_builder = hashlib.sha256()

    for client_id, source_count in source_counts:
        indices = client_indices[client_id]
        local_source_positions = np.flatnonzero(y_train[indices] == source_id)
        selected_positions = np.empty(0, dtype=np.int64)

        if client_id in malicious_clients:
            flip_count = max(
                1,
                int(round(len(local_source_positions) * poison_fraction)),
            )
            flip_count = min(flip_count, len(local_source_positions))
            client_rng = np.random.default_rng(
                attack_seed * 1_000_003 + client_id
            )
            source_order = client_rng.permutation(local_source_positions)
            selected_positions = np.sort(
                source_order[:flip_count].astype(np.int64)
            )

        poisoned_positions[client_id] = selected_positions
        global_poisoned = indices[selected_positions]
        hash_builder.update(f"client_{client_id:03d}".encode("utf-8"))
        hash_builder.update(global_poisoned.astype(np.int64).tobytes())

        manifest_rows.append(
            {
                "client_id": client_id,
                "is_malicious": client_id in malicious_clients,
                "client_rows": int(len(indices)),
                "source_rows_before_poisoning": int(source_count),
                "poisoned_source_rows": int(len(selected_positions)),
                "client_source_poison_rate": float(
                    len(selected_positions) / max(source_count, 1)
                ),
                "source_rows_after_poisoning": int(
                    source_count - len(selected_positions)
                ),
                "target_rows_added_by_poisoning": int(len(selected_positions)),
            }
        )

    return (
        malicious_clients,
        poisoned_positions,
        pd.DataFrame(manifest_rows),
        hash_builder.hexdigest(),
    )


def save_poisoned_indices(
    client_indices: Sequence[np.ndarray],
    poisoned_positions: Mapping[int, np.ndarray],
    output_path: Path,
) -> None:
    payload = {}
    for client_id, positions in poisoned_positions.items():
        payload[f"client_{client_id:03d}_local_positions"] = positions.astype(np.int64)
        payload[f"client_{client_id:03d}_global_indices"] = (
            client_indices[client_id][positions].astype(np.int64)
        )
    np.savez_compressed(output_path, **payload)


def evaluate_validation(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    source_id: int,
    target_id: int,
    batch_size: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    loader = make_loader(X, y, batch_size=batch_size, shuffle=False)
    logits = []
    labels = []
    model.eval()
    with torch.no_grad():
        for features, batch_labels in loader:
            logits.append(model(features).cpu().numpy())
            labels.append(batch_labels.cpu().numpy())

    probabilities = stable_softmax(np.concatenate(logits))
    labels_array = np.concatenate(labels)
    predicted = probabilities.argmax(axis=1)
    metrics = metric_dict(labels_array, probabilities)
    matrix = confusion_matrix(
        labels_array,
        predicted,
        labels=np.arange(NUM_CLASSES),
    )
    source_total = int(matrix[source_id].sum())
    target_total = int(matrix[target_id].sum())
    attack_metrics = {
        "source_recall": float(
            matrix[source_id, source_id] / max(source_total, 1)
        ),
        "target_recall": float(
            matrix[target_id, target_id] / max(target_total, 1)
        ),
        "source_to_target_rate": float(
            matrix[source_id, target_id] / max(source_total, 1)
        ),
        "source_to_target_count": int(matrix[source_id, target_id]),
        "source_support": source_total,
        "target_support": target_total,
    }
    return metrics, attack_metrics


def attack_metrics_from_matrix(
    matrix: np.ndarray,
    source_id: int,
    target_id: int,
) -> Dict[str, float]:
    source_total = int(matrix[source_id].sum())
    target_total = int(matrix[target_id].sum())
    return {
        "source_support": source_total,
        "target_support": target_total,
        "source_recall": float(
            matrix[source_id, source_id] / max(source_total, 1)
        ),
        "target_recall": float(
            matrix[target_id, target_id] / max(target_total, 1)
        ),
        "source_to_target_count": int(matrix[source_id, target_id]),
        "source_to_target_rate": float(
            matrix[source_id, target_id] / max(source_total, 1)
        ),
        "target_to_source_count": int(matrix[target_id, source_id]),
        "target_to_source_rate": float(
            matrix[target_id, source_id] / max(target_total, 1)
        ),
    }


def evaluate_checkpoint(
    checkpoint_name: str,
    checkpoint_path: Path,
    arrays: Mapping[str, np.ndarray],
    source_id: int,
    target_id: int,
    evaluation_batch_size: int,
    tables_dir: Path,
    save_predictions: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = build_model(
        architecture="resmlp",
        input_dim=int(checkpoint["input_dim"]),
        num_classes=int(checkpoint["num_classes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    metric_rows = []
    per_class_tables = []
    attack_rows = []

    for split, X_key, y_key in (
        ("test_natural", "X_test_natural", "y_test_natural"),
        ("test_diagnostic", "X_test_diagnostic", "y_test_diagnostic"),
    ):
        metrics, per_class, matrix, predictions = evaluation_artifacts(
            model=model,
            X=arrays[X_key].astype(np.float32, copy=False),
            y=arrays[y_key].astype(np.int64, copy=False),
            batch_size=evaluation_batch_size,
            split=split,
        )
        metric_rows.append(
            {
                "checkpoint": checkpoint_name,
                "checkpoint_round": int(checkpoint["round"]),
                "split": split,
                **metrics,
            }
        )
        per_class.insert(0, "checkpoint", checkpoint_name)
        per_class.insert(1, "checkpoint_round", int(checkpoint["round"]))
        per_class_tables.append(per_class)
        attack_rows.append(
            {
                "checkpoint": checkpoint_name,
                "checkpoint_round": int(checkpoint["round"]),
                "split": split,
                **attack_metrics_from_matrix(
                    matrix,
                    source_id=source_id,
                    target_id=target_id,
                ),
            }
        )

        pd.DataFrame(
            matrix,
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        ).to_csv(
            tables_dir / f"{checkpoint_name}_{split}_confusion_matrix.csv"
        )
        if save_predictions:
            predictions.to_csv(
                tables_dir / f"{checkpoint_name}_{split}_predictions.csv.gz",
                index=False,
                compression="gzip",
            )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(per_class_tables, ignore_index=True),
        pd.DataFrame(attack_rows),
    )


def clean_vs_attack_tables(
    clean_seed_dir: Path,
    arrays: Mapping[str, np.ndarray],
    attacked_metrics: pd.DataFrame,
    attacked_per_class: pd.DataFrame,
    attacked_pair: pd.DataFrame,
    source_id: int,
    target_id: int,
    source_class: str,
    target_class: str,
    evaluation_batch_size: int,
    tables_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build dynamic clean-versus-attack tables for the requested class pair.

    Earlier versions reused a DDoS-to-DoS clean pair table even when a different
    target class was requested. This implementation evaluates the clean seed-42
    checkpoint directly for the active source and target IDs.
    """
    clean_checkpoint = (
        clean_seed_dir / "checkpoints" / "best_validation_model.pt"
    )
    if not clean_checkpoint.exists():
        raise FileNotFoundError(
            f"Clean reference checkpoint not found: {clean_checkpoint}"
        )

    clean_metrics, clean_per_class, clean_pair = evaluate_checkpoint(
        checkpoint_name="clean_reference",
        checkpoint_path=clean_checkpoint,
        arrays=arrays,
        source_id=source_id,
        target_id=target_id,
        evaluation_batch_size=evaluation_batch_size,
        tables_dir=tables_dir,
        save_predictions=False,
    )
    clean_metrics = clean_metrics.drop(
        columns=["checkpoint", "checkpoint_round"],
        errors="ignore",
    )
    clean_per_class = clean_per_class.drop(
        columns=["checkpoint", "checkpoint_round"],
        errors="ignore",
    )
    clean_pair = clean_pair.drop(
        columns=["checkpoint", "checkpoint_round"],
        errors="ignore",
    )

    primary_metrics = attacked_metrics[
        attacked_metrics["checkpoint"] == "best_validation"
    ].copy()
    overall = clean_metrics.merge(
        primary_metrics,
        on="split",
        suffixes=("_clean", "_attacked"),
    )
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "log_loss",
        "ece_15bin",
    ):
        overall[f"{metric}_delta_attacked_minus_clean"] = (
            overall[f"{metric}_attacked"] - overall[f"{metric}_clean"]
        )

    primary_per_class = attacked_per_class[
        attacked_per_class["checkpoint"] == "best_validation"
    ].copy()
    class_comparison = clean_per_class.merge(
        primary_per_class,
        on=["split", "class_id", "class_name"],
        suffixes=("_clean", "_attacked"),
    )
    for metric in ("precision", "recall", "f1_score"):
        class_comparison[f"{metric}_delta_attacked_minus_clean"] = (
            class_comparison[f"{metric}_attacked"]
            - class_comparison[f"{metric}_clean"]
        )

    primary_pair = attacked_pair[
        attacked_pair["checkpoint"] == "best_validation"
    ].copy()
    pair = clean_pair.merge(
        primary_pair,
        on="split",
        suffixes=("_clean", "_attacked"),
    )
    pair.insert(0, "source_class", source_class)
    pair.insert(1, "target_class", target_class)
    for metric in (
        "source_recall",
        "target_recall",
        "source_to_target_rate",
        "target_to_source_rate",
    ):
        pair[f"{metric}_delta_attacked_minus_clean"] = (
            pair[f"{metric}_attacked"] - pair[f"{metric}_clean"]
        )
    pair["source_to_target_fold_change"] = np.divide(
        pair["source_to_target_rate_attacked"],
        np.clip(pair["source_to_target_rate_clean"], 1e-12, None),
    )

    return overall, class_comparison, pair


def plot_training_curves(
    round_table: pd.DataFrame,
    source_class: str,
    target_class: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        round_table["round"],
        round_table["val_macro_f1"],
        marker="o",
        label="Validation macro F1",
    )
    ax.plot(
        round_table["round"],
        round_table["val_source_to_target_rate"],
        marker="s",
        label=f"Validation {source_class} to {target_class} rate",
    )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Targeted Label-Flip Dynamics")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def plot_source_target_recall(
    round_table: pd.DataFrame,
    source_class: str,
    target_class: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        round_table["round"],
        round_table["val_source_recall"],
        marker="o",
        label=f"{source_class} recall",
    )
    ax.plot(
        round_table["round"],
        round_table["val_target_recall"],
        marker="s",
        label=f"{target_class} recall",
    )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)
    ax.set_title("Source and Target Recall During Poisoned Training")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def plot_poison_manifest(
    manifest: pd.DataFrame,
    source_class: str,
    target_class: str,
    output: Path,
) -> None:
    malicious = manifest[manifest["is_malicious"]].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(malicious))
    width = 0.36
    ax.bar(
        x - width / 2,
        malicious["source_rows_before_poisoning"],
        width,
        label=f"{source_class} rows",
    )
    ax.bar(
        x + width / 2,
        malicious["poisoned_source_rows"],
        width,
        label=f"Flipped to {target_class}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(malicious["client_id"].astype(str))
    ax.set_xlabel("Malicious client")
    ax.set_ylabel("Rows")
    ax.set_title("Static Poison Exposure by Malicious Client")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def plot_clean_vs_attack(
    pair: pd.DataFrame,
    source_class: str,
    target_class: str,
    output: Path,
) -> None:
    natural = pair[pair["split"] == "test_natural"].iloc[0]
    values = [
        float(natural["source_to_target_rate_clean"]),
        float(natural["source_to_target_rate_attacked"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(["Clean", "Attacked"], values)
    ax.set_ylim(0, max(0.01, max(values) * 1.20))
    ax.set_ylabel(f"{source_class} predicted as {target_class}")
    ax.set_title("Targeted Attack Effect on the Natural Test")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output)


def plot_clean_vs_attack_macro_f1(overall: pd.DataFrame, output: Path) -> None:
    splits = overall["split"].tolist()
    clean = overall["macro_f1_clean"].to_numpy()
    attacked = overall["macro_f1_attacked"].to_numpy()
    x = np.arange(len(splits))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width / 2, clean, width, label="Clean")
    ax.bar(x + width / 2, attacked, width, label="Attacked")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1")
    ax.set_title("Clean Utility Versus Targeted Label-Flipping Attack")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def main() -> int:
    args = parse_args()

    if args.source_class not in CLASS_NAMES:
        raise ValueError(f"Unknown source class: {args.source_class}")
    if args.target_class not in CLASS_NAMES:
        raise ValueError(f"Unknown target class: {args.target_class}")
    if not 0 < args.participation_rate <= 1:
        raise ValueError("participation-rate must be in (0, 1].")

    source_id = CLASS_NAMES.index(args.source_class)
    target_id = CLASS_NAMES.index(args.target_class)

    torch.set_num_threads(max(1, args.threads))
    set_seed(args.model_seed)

    data_file = args.data_file.expanduser().resolve()
    partition_file = args.partition_file.expanduser().resolve()
    clean_seed_dir = args.clean_seed_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
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
        partition_file,
        expected_clients=args.num_clients,
        train_rows=len(y_train),
    )
    clean_partition_hash = read_clean_partition_hash(clean_seed_dir)
    if clean_partition_hash != partition_hash:
        raise RuntimeError(
            "Clean seed partition hash does not match the supplied partition."
        )

    client_summary, client_matrix = partition_manifest(client_indices, y_train)
    client_summary.to_csv(tables_dir / "client_partition_summary.csv", index=False)
    client_matrix.to_csv(tables_dir / "client_class_counts.csv", index=False)

    (
        malicious_clients,
        poisoned_positions,
        poison_manifest,
        poison_hash,
    ) = prepare_static_attack(
        client_indices=client_indices,
        y_train=y_train,
        source_id=source_id,
        target_id=target_id,
        malicious_client_fraction=args.malicious_client_fraction,
        poison_fraction=args.poison_fraction,
        min_source_samples=args.min_source_samples,
        attack_seed=args.attack_seed,
        explicit_malicious_clients=(
            [
                int(value.strip())
                for value in args.malicious_clients.split(",")
                if value.strip()
            ]
            if args.malicious_clients.strip()
            else None
        ),
    )
    poison_manifest.to_csv(
        attack_dir / "malicious_client_poison_manifest.csv",
        index=False,
    )
    save_poisoned_indices(
        client_indices,
        poisoned_positions,
        attack_dir / "poisoned_indices.npz",
    )

    total_source_rows = int(np.sum(y_train == source_id))
    total_flipped_rows = int(poison_manifest["poisoned_source_rows"].sum())
    attack_summary = pd.DataFrame(
        [
            {
                "source_class": args.source_class,
                "target_class": args.target_class,
                "num_clients": args.num_clients,
                "malicious_clients": "|".join(map(str, malicious_clients)),
                "malicious_client_count": len(malicious_clients),
                "malicious_client_fraction_requested": args.malicious_client_fraction,
                "malicious_client_fraction_actual": len(malicious_clients)
                / args.num_clients,
                "poison_fraction_within_malicious_source_rows": args.poison_fraction,
                "min_source_samples_for_eligibility": args.min_source_samples,
                "global_source_rows": total_source_rows,
                "global_flipped_rows": total_flipped_rows,
                "global_source_exposure_fraction": total_flipped_rows
                / max(total_source_rows, 1),
                "partition_hash_sha256": partition_hash,
                "poison_index_hash_sha256": poison_hash,
            }
        ]
    )
    attack_summary.to_csv(attack_dir / "attack_summary.csv", index=False)

    class_weights = sqrt_class_weights(y_train, args.max_class_weight)
    pd.DataFrame(
        {
            "class_id": np.arange(NUM_CLASSES),
            "class_name": CLASS_NAMES,
            "clean_train_rows": np.bincount(y_train, minlength=NUM_CLASSES),
            "clean_sqrt_class_weight": class_weights,
        }
    ).to_csv(tables_dir / "clean_global_class_weights.csv", index=False)

    model = build_model(
        architecture="resmlp",
        input_dim=X_train.shape[1],
        num_classes=NUM_CLASSES,
    )
    rng = np.random.default_rng(args.model_seed)
    participating_count = max(
        1,
        int(math.ceil(args.num_clients * args.participation_rate)),
    )

    round_rows: List[Dict[str, object]] = []
    local_rows: List[Dict[str, object]] = []
    best_macro_f1 = -math.inf
    best_round = 0
    stale_rounds = 0
    started = time.time()

    print("Targeted label-flipping attack")
    print("Source:", args.source_class)
    print("Target:", args.target_class)
    print("Malicious clients:", malicious_clients)
    print("Flipped source rows:", total_flipped_rows)
    print(
        "Global source exposure:",
        f"{total_flipped_rows / max(total_source_rows, 1):.4%}",
    )
    print("Partition hash:", partition_hash)
    print("Poison index hash:", poison_hash)

    for round_id in range(1, args.rounds + 1):
        round_started = time.time()
        selected = np.sort(
            rng.choice(
                args.num_clients,
                size=participating_count,
                replace=False,
            )
        )
        reference_state = copy.deepcopy(model.state_dict())
        local_states = []
        local_counts = []
        current_local_rows = []

        for client_id in selected:
            client_id = int(client_id)
            indices = client_indices[client_id]
            local_y = y_train[indices].copy()
            positions = poisoned_positions[client_id]
            if len(positions) > 0:
                local_y[positions] = target_id

            local_model = build_model(
                architecture="resmlp",
                input_dim=X_train.shape[1],
                num_classes=NUM_CLASSES,
            )
            local_model.load_state_dict(reference_state)
            state, metrics = train_local_model(
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
            local_states.append(state)
            local_counts.append(len(indices))
            row = {
                "round": round_id,
                "client_id": client_id,
                "is_malicious": client_id in malicious_clients,
                "client_samples": int(len(indices)),
                "poisoned_rows": int(len(positions)),
                "source_rows_before_poisoning": int(
                    np.sum(y_train[indices] == source_id)
                ),
                **metrics,
            }
            local_rows.append(row)
            current_local_rows.append(row)
            del local_model

        averaged = weighted_average_states(
            states=local_states,
            sample_counts=local_counts,
            reference_state=reference_state,
        )
        model.load_state_dict(averaged)

        val_metrics, val_attack = evaluate_validation(
            model,
            X_val,
            y_val,
            source_id=source_id,
            target_id=target_id,
            batch_size=args.evaluation_batch_size,
        )
        row = {
            "round": round_id,
            "selected_clients": "|".join(map(str, selected.tolist())),
            "selected_client_count": int(len(selected)),
            "participating_samples": int(sum(local_counts)),
            "mean_local_train_loss": float(
                np.mean([item["local_train_loss"] for item in current_local_rows])
            ),
            "mean_local_train_accuracy": float(
                np.mean(
                    [item["local_train_accuracy"] for item in current_local_rows]
                )
            ),
            "round_seconds": float(time.time() - round_started),
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"val_{key}": value for key, value in val_attack.items()},
        }
        round_rows.append(row)

        print(
            f"round {round_id:02d}, "
            f"val macro F1={val_metrics['macro_f1']:.4f}, "
            f"{args.source_class}->{args.target_class}="
            f"{val_attack['source_to_target_rate']:.4%}, "
            f"source recall={val_attack['source_recall']:.4f}, "
            f"seconds={row['round_seconds']:.1f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_round = round_id
            stale_rounds = 0
            torch.save(
                {
                    "experiment_version": "2.9.2",
                    "checkpoint_type": "best_validation",
                    "round": round_id,
                    "model_state_dict": model.state_dict(),
                    "input_dim": X_train.shape[1],
                    "num_classes": NUM_CLASSES,
                    "validation_metrics": val_metrics,
                    "validation_attack_metrics": val_attack,
                    "partition_hash": partition_hash,
                    "poison_index_hash": poison_hash,
                    "source_class": args.source_class,
                    "target_class": args.target_class,
                    "malicious_clients": malicious_clients,
                },
                checkpoints_dir / "best_validation_model.pt",
            )
        else:
            stale_rounds += 1

        torch.save(
            {
                "experiment_version": "2.9.2",
                "checkpoint_type": "last_round",
                "round": round_id,
                "model_state_dict": model.state_dict(),
                "input_dim": X_train.shape[1],
                "num_classes": NUM_CLASSES,
                "validation_metrics": val_metrics,
                "validation_attack_metrics": val_attack,
                "partition_hash": partition_hash,
                "poison_index_hash": poison_hash,
                "source_class": args.source_class,
                "target_class": args.target_class,
                "malicious_clients": malicious_clients,
            },
            checkpoints_dir / "last_round_model.pt",
        )

        pd.DataFrame(round_rows).to_csv(
            tables_dir / "round_metrics_partial.csv",
            index=False,
        )

        if (
            args.early_stopping_patience > 0
            and stale_rounds >= args.early_stopping_patience
        ):
            print(f"Early stopping after round {round_id}")
            break

    round_table = pd.DataFrame(round_rows)
    local_table = pd.DataFrame(local_rows)
    round_table.to_csv(tables_dir / "round_metrics.csv", index=False)
    local_table.to_csv(tables_dir / "local_client_training_metrics.csv", index=False)

    metric_tables = []
    per_class_tables = []
    pair_tables = []
    for checkpoint_name, checkpoint_path in (
        ("best_validation", checkpoints_dir / "best_validation_model.pt"),
        ("last_round", checkpoints_dir / "last_round_model.pt"),
    ):
        metrics, per_class, pair = evaluate_checkpoint(
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
            arrays=arrays,
            source_id=source_id,
            target_id=target_id,
            evaluation_batch_size=args.evaluation_batch_size,
            tables_dir=tables_dir,
            save_predictions=args.save_predictions,
        )
        metric_tables.append(metrics)
        per_class_tables.append(per_class)
        pair_tables.append(pair)

    attacked_metrics = pd.concat(metric_tables, ignore_index=True)
    attacked_per_class = pd.concat(per_class_tables, ignore_index=True)
    attacked_pair = pd.concat(pair_tables, ignore_index=True)
    attacked_metrics.to_csv(tables_dir / "attacked_test_metrics.csv", index=False)
    attacked_per_class.to_csv(
        tables_dir / "attacked_per_class_metrics.csv",
        index=False,
    )
    attacked_pair.to_csv(
        tables_dir / "attacked_source_target_metrics.csv",
        index=False,
    )

    overall, class_comparison, pair_comparison = clean_vs_attack_tables(
        clean_seed_dir=clean_seed_dir,
        arrays=arrays,
        attacked_metrics=attacked_metrics,
        attacked_per_class=attacked_per_class,
        attacked_pair=attacked_pair,
        source_id=source_id,
        target_id=target_id,
        source_class=args.source_class,
        target_class=args.target_class,
        evaluation_batch_size=args.evaluation_batch_size,
        tables_dir=tables_dir,
    )
    overall.to_csv(
        tables_dir / "clean_vs_attack_overall.csv",
        index=False,
    )
    class_comparison.to_csv(
        tables_dir / "clean_vs_attack_per_class.csv",
        index=False,
    )
    pair_comparison.to_csv(
        tables_dir / "clean_vs_attack_source_target.csv",
        index=False,
    )

    plot_training_curves(
        round_table,
        args.source_class,
        args.target_class,
        figures_dir / "attack_dynamics",
    )
    plot_source_target_recall(
        round_table,
        args.source_class,
        args.target_class,
        figures_dir / "source_target_recall_by_round",
    )
    plot_poison_manifest(
        poison_manifest,
        args.source_class,
        args.target_class,
        figures_dir / "malicious_client_poison_exposure",
    )
    plot_clean_vs_attack(
        pair_comparison,
        args.source_class,
        args.target_class,
        figures_dir / "clean_vs_attack_source_to_target_rate",
    )
    plot_clean_vs_attack_macro_f1(
        overall,
        figures_dir / "clean_vs_attack_macro_f1",
    )

    metadata = {
        "experiment_version": "2.9.2",
        "experiment_type": "targeted_static_label_flipping_explicit_client",
        "model": "ResidualMLP",
        "loss": "clean_sqrt_class_weighted_cross_entropy",
        "source_class": args.source_class,
        "source_class_id": source_id,
        "target_class": args.target_class,
        "target_class_id": target_id,
        "malicious_client_selection": (
            "Explicit client IDs supplied by the strength-grid controller."
            if args.malicious_clients.strip()
            else (
                "Seeded random sample without replacement among clients meeting "
                "the minimum source-row threshold."
            )
        ),
        "malicious_clients": malicious_clients,
        "malicious_client_fraction_requested": args.malicious_client_fraction,
        "poison_fraction_within_malicious_source_rows": args.poison_fraction,
        "global_source_rows": total_source_rows,
        "global_flipped_rows": total_flipped_rows,
        "global_source_exposure_fraction": total_flipped_rows
        / max(total_source_rows, 1),
        "static_poisoned_rows_across_rounds": True,
        "features_modified": False,
        "partition_file": str(partition_file),
        "partition_hash_sha256": partition_hash,
        "clean_seed_partition_hash_sha256": clean_partition_hash,
        "poison_index_hash_sha256": poison_hash,
        "clean_seed_dir": str(clean_seed_dir),
        "model_seed": args.model_seed,
        "attack_seed": args.attack_seed,
        "learning_rate": args.learning_rate,
        "local_epochs": args.local_epochs,
        "rounds_requested": args.rounds,
        "rounds_completed": int(round_table["round"].max()),
        "best_validation_round": best_round,
        "best_validation_macro_f1": best_macro_f1,
        "test_sets_used_for_checkpoint_selection": False,
        "primary_checkpoint": "best_validation",
        "secondary_checkpoint": "last_round",
        "save_predictions": bool(args.save_predictions),
        "total_seconds": float(time.time() - started),
        "notes": [
            "This is a single-seed attack validation, not the final paper experiment.",
            "The clean partition and clean class weights are fixed before poisoning.",
            f"The attack changes only labels from {args.source_class} to "
            f"{args.target_class} on selected clients.",
            "The same poisoned rows are used in every communication round.",
            "The best checkpoint is selected using clean validation macro F1 only.",
            "Later experiments must vary malicious-client fraction, poison fraction, attack pair, heterogeneity, and seed.",
        ],
    }
    with (output_dir / "targeted_label_flip_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Targeted Label Flip V2.9.2 complete")
    print("Best validation round:", best_round)
    print("Best validation macro F1:", f"{best_macro_f1:.6f}")
    print()
    print("Clean versus attacked source-target effect")
    print(pair_comparison.to_string(index=False))
    print()
    print("Clean versus attacked overall metrics")
    print(
        overall[
            [
                "split",
                "macro_f1_clean",
                "macro_f1_attacked",
                "macro_f1_delta_attacked_minus_clean",
                "balanced_accuracy_clean",
                "balanced_accuracy_attacked",
            ]
        ].to_string(index=False)
    )
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("Attack manifest:", attack_dir)
    print("Metadata:", output_dir / "targeted_label_flip_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

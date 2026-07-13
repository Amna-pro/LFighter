#!/usr/bin/env python3
"""Clean FedAvg baseline for CIC IoT-DIAD Protocol V2.1.

This is the first federated engineering and scientific validation stage.
It contains no poisoning and no defense. Model selection uses validation macro F1 only.
Natural and diagnostic test sets are evaluated only after the best validation round is fixed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from federated_iot_v26 import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    dirichlet_partition,
    evaluation_artifacts,
    load_protocol_arrays,
    make_loader,
    metric_dict,
    partition_manifest,
    save_partitions,
    set_seed,
    sqrt_class_weights,
    stable_softmax,
    train_local_model,
    weighted_average_states,
)
from neural_models_v24 import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean FedAvg on CIC IoT-DIAD V2.1.")
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--participation-rate", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--min-client-samples", type=int, default=500)
    parser.add_argument("--min-client-classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def normalize_confusion(matrix: np.ndarray) -> np.ndarray:
    denominator = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix, dtype=float),
        where=denominator != 0,
    )


def plot_partition_heatmap(matrix: pd.DataFrame, output: Path) -> None:
    values = matrix[CLASS_NAMES].to_numpy(dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    proportions = np.divide(
        values,
        row_sums,
        out=np.zeros_like(values),
        where=row_sums != 0,
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(proportions, aspect="auto", vmin=0.0, vmax=max(0.01, proportions.max()))
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix)))
    ax.set_yticklabels(matrix["client_id"].astype(str))
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")
    ax.set_title("Dirichlet Client Class Proportions")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Within-client proportion")
    save_figure(fig, output)


def plot_client_samples(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["client_id"].astype(str), summary["samples"])
    ax.set_xlabel("Client")
    ax.set_ylabel("Training samples")
    ax.set_title("Federated Client Sample Counts")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output)


def plot_client_entropy(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["client_id"].astype(str), summary["normalized_class_entropy"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Client")
    ax.set_ylabel("Normalized class entropy")
    ax.set_title("Client Label Diversity")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output)


def plot_round_metric(rounds: pd.DataFrame, metric: str, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(rounds["round"], rounds[metric], marker="o")
    ax.set_xlabel("Communication round")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    save_figure(fig, output)


def plot_confusion(matrix: np.ndarray, title: str, output: Path) -> None:
    normalized = normalize_confusion(matrix)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(normalized, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized proportion")
    save_figure(fig, output)


def plot_per_class_f1(table: pd.DataFrame, title: str, output: Path) -> None:
    ordered = table.sort_values("class_id")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ordered["class_name"], ordered["f1_score"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Class")
    ax.set_ylabel("F1 score")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output)


def main() -> int:
    args = parse_args()
    if not 0 < args.participation_rate <= 1:
        raise ValueError("participation-rate must be in (0, 1]")
    if args.rounds < 1:
        raise ValueError("rounds must be at least 1")

    torch.set_num_threads(max(1, args.threads))
    set_seed(args.seed)

    data_file = args.data_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"
    partitions_dir = output_dir / "partitions"
    for path in (tables_dir, figures_dir, checkpoints_dir, partitions_dir):
        path.mkdir(parents=True, exist_ok=True)

    arrays = load_protocol_arrays(data_file)
    X_train = arrays["X_train"].astype(np.float32, copy=False)
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    X_val = arrays["X_val"].astype(np.float32, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)

    print("Data file:", data_file)
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Clients:", args.num_clients)
    print("Dirichlet alpha:", args.dirichlet_alpha)
    print("Rounds:", args.rounds)
    print("Participation rate:", args.participation_rate)
    print("CPU threads:", args.threads)

    client_indices = dirichlet_partition(
        y=y_train,
        num_clients=args.num_clients,
        alpha=args.dirichlet_alpha,
        seed=args.seed,
        min_client_samples=args.min_client_samples,
        min_client_classes=args.min_client_classes,
    )
    partition_hash = save_partitions(
        client_indices,
        partitions_dir / "client_partitions.npz",
    )
    client_summary, client_matrix = partition_manifest(client_indices, y_train)
    client_summary.to_csv(tables_dir / "client_partition_summary.csv", index=False)
    client_matrix.to_csv(tables_dir / "client_class_counts.csv", index=False)

    class_weight_values = sqrt_class_weights(y_train, args.max_class_weight)
    pd.DataFrame(
        {
            "class_id": np.arange(NUM_CLASSES),
            "class_name": CLASS_NAMES,
            "train_rows": np.bincount(y_train, minlength=NUM_CLASSES),
            "sqrt_class_weight": class_weight_values,
        }
    ).to_csv(tables_dir / "global_class_weights.csv", index=False)

    plot_partition_heatmap(client_matrix, figures_dir / "client_class_distribution_heatmap")
    plot_client_samples(client_summary, figures_dir / "client_sample_counts")
    plot_client_entropy(client_summary, figures_dir / "client_label_entropy")

    global_model = build_model(
        architecture="resmlp",
        input_dim=X_train.shape[1],
        num_classes=NUM_CLASSES,
    )
    initial_state = copy.deepcopy(global_model.state_dict())

    rng = np.random.default_rng(args.seed)
    participating_count = max(
        1,
        int(math.ceil(args.num_clients * args.participation_rate)),
    )

    round_rows: List[Dict[str, object]] = []
    local_rows: List[Dict[str, object]] = []
    participation_rows: List[Dict[str, object]] = []
    best_val_macro_f1 = -math.inf
    best_round = 0
    stale_rounds = 0
    total_started = time.time()

    for round_id in range(1, args.rounds + 1):
        round_started = time.time()
        selected = np.sort(
            rng.choice(
                args.num_clients,
                size=participating_count,
                replace=False,
            )
        )
        print()
        print(f"Round {round_id:03d}/{args.rounds:03d}, clients: {selected.tolist()}")

        reference_state = copy.deepcopy(global_model.state_dict())
        local_states = []
        local_sample_counts = []

        for client_id in selected:
            indices = client_indices[int(client_id)]
            local_model = build_model(
                architecture="resmlp",
                input_dim=X_train.shape[1],
                num_classes=NUM_CLASSES,
            )
            local_model.load_state_dict(reference_state)

            state, metrics = train_local_model(
                model=local_model,
                X=X_train[indices],
                y=y_train[indices],
                class_weights=class_weight_values,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                seed=args.seed + round_id * 1000 + int(client_id),
            )
            local_states.append(state)
            local_sample_counts.append(len(indices))
            local_rows.append(
                {
                    "round": round_id,
                    "client_id": int(client_id),
                    "client_samples": int(len(indices)),
                    **metrics,
                }
            )
            participation_rows.append(
                {
                    "round": round_id,
                    "client_id": int(client_id),
                    "selected": True,
                    "aggregation_weight": float(
                        len(indices) / sum(len(client_indices[int(value)]) for value in selected)
                    ),
                }
            )
            del local_model

        averaged_state = weighted_average_states(
            states=local_states,
            sample_counts=local_sample_counts,
            reference_state=reference_state,
        )
        global_model.load_state_dict(averaged_state)

        val_probabilities, val_labels = [], []
        val_loader = make_loader(
            X_val,
            y_val,
            batch_size=args.evaluation_batch_size,
            shuffle=False,
        )
        global_model.eval()
        with torch.no_grad():
            for features, labels in val_loader:
                val_probabilities.append(global_model(features).cpu().numpy())
                val_labels.append(labels.cpu().numpy())
        val_probabilities_array = stable_softmax(np.concatenate(val_probabilities))
        val_labels_array = np.concatenate(val_labels)
        val_metrics = metric_dict(val_labels_array, val_probabilities_array)

        mean_local_loss = float(
            np.mean([row["local_train_loss"] for row in local_rows if row["round"] == round_id])
        )
        mean_local_accuracy = float(
            np.mean([row["local_train_accuracy"] for row in local_rows if row["round"] == round_id])
        )
        round_seconds = float(time.time() - round_started)
        round_row = {
            "round": round_id,
            "selected_clients": "|".join(map(str, selected.tolist())),
            "selected_client_count": int(len(selected)),
            "participating_samples": int(sum(local_sample_counts)),
            "mean_local_train_loss": mean_local_loss,
            "mean_local_train_accuracy": mean_local_accuracy,
            "round_seconds": round_seconds,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        round_rows.append(round_row)

        print(
            f"val macro F1={val_metrics['macro_f1']:.4f}, "
            f"balanced accuracy={val_metrics['balanced_accuracy']:.4f}, "
            f"round seconds={round_seconds:.1f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_round = round_id
            stale_rounds = 0
            torch.save(
                {
                    "round": round_id,
                    "model_state_dict": global_model.state_dict(),
                    "input_dim": X_train.shape[1],
                    "num_classes": NUM_CLASSES,
                    "validation_metrics": val_metrics,
                    "partition_hash": partition_hash,
                },
                checkpoints_dir / "best_validation_model.pt",
            )
        else:
            stale_rounds += 1

        torch.save(
            {
                "round": round_id,
                "model_state_dict": global_model.state_dict(),
                "input_dim": X_train.shape[1],
                "num_classes": NUM_CLASSES,
                "validation_metrics": val_metrics,
                "partition_hash": partition_hash,
            },
            checkpoints_dir / "last_round_model.pt",
        )

        if (
            args.early_stopping_patience > 0
            and stale_rounds >= args.early_stopping_patience
        ):
            print(f"Early stopping after round {round_id}")
            break

    round_table = pd.DataFrame(round_rows)
    local_table = pd.DataFrame(local_rows)
    participation_table = pd.DataFrame(participation_rows)
    round_table.to_csv(tables_dir / "round_metrics.csv", index=False)
    local_table.to_csv(tables_dir / "local_client_training_metrics.csv", index=False)
    participation_table.to_csv(tables_dir / "client_participation.csv", index=False)

    checkpoint = torch.load(
        checkpoints_dir / "best_validation_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    global_model.load_state_dict(checkpoint["model_state_dict"])

    final_metric_rows = []
    all_per_class = []
    for split_name, X_split, y_split in (
        (
            "test_natural",
            arrays["X_test_natural"].astype(np.float32, copy=False),
            arrays["y_test_natural"].astype(np.int64, copy=False),
        ),
        (
            "test_diagnostic",
            arrays["X_test_diagnostic"].astype(np.float32, copy=False),
            arrays["y_test_diagnostic"].astype(np.int64, copy=False),
        ),
    ):
        metrics, per_class, matrix, predictions = evaluation_artifacts(
            model=global_model,
            X=X_split,
            y=y_split,
            batch_size=args.evaluation_batch_size,
            split=split_name,
        )
        final_metric_rows.append({"split": split_name, **metrics})
        all_per_class.append(per_class)
        pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            tables_dir / f"{split_name}_confusion_matrix.csv"
        )
        predictions.to_csv(
            tables_dir / f"{split_name}_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
        plot_confusion(
            matrix,
            title=f"Clean FedAvg Normalized Confusion Matrix, {split_name}",
            output=figures_dir / f"{split_name}_normalized_confusion_matrix",
        )
        plot_per_class_f1(
            per_class,
            title=f"Clean FedAvg Per-Class F1, {split_name}",
            output=figures_dir / f"{split_name}_per_class_f1",
        )

    final_metrics = pd.DataFrame(final_metric_rows)
    per_class_table = pd.concat(all_per_class, ignore_index=True)
    final_metrics.to_csv(tables_dir / "final_test_metrics.csv", index=False)
    per_class_table.to_csv(tables_dir / "final_per_class_metrics.csv", index=False)

    plot_round_metric(
        round_table,
        "val_macro_f1",
        "Clean FedAvg Validation Macro F1",
        figures_dir / "validation_macro_f1_by_round",
    )
    plot_round_metric(
        round_table,
        "val_balanced_accuracy",
        "Clean FedAvg Validation Balanced Accuracy",
        figures_dir / "validation_balanced_accuracy_by_round",
    )
    plot_round_metric(
        round_table,
        "mean_local_train_loss",
        "Mean Local Training Loss",
        figures_dir / "mean_local_loss_by_round",
    )
    plot_round_metric(
        round_table,
        "round_seconds",
        "Communication-Round Runtime",
        figures_dir / "round_runtime",
    )

    metadata = {
        "experiment_version": "2.6",
        "experiment_type": "clean_fedavg_baseline",
        "contains_poisoning": False,
        "contains_defense": False,
        "selection_rule": "Best communication round by validation macro F1 only.",
        "test_sets_used_for_model_selection": False,
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "model": "ResidualMLP",
        "loss": "sqrt_class_weighted_cross_entropy",
        "input_dim": int(X_train.shape[1]),
        "num_classes": NUM_CLASSES,
        "num_clients": args.num_clients,
        "dirichlet_alpha": args.dirichlet_alpha,
        "rounds_requested": args.rounds,
        "rounds_completed": int(round_table["round"].max()),
        "participation_rate": args.participation_rate,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_class_weight": args.max_class_weight,
        "seed": args.seed,
        "partition_hash_sha256": partition_hash,
        "best_validation_round": best_round,
        "best_validation_macro_f1": best_val_macro_f1,
        "total_seconds": float(time.time() - total_started),
        "tables": str(tables_dir),
        "figures_png_pdf": str(figures_dir),
        "checkpoint": str(checkpoints_dir / "best_validation_model.pt"),
        "notes": [
            "This stage validates clean FedAvg before introducing label-flipping attacks.",
            "Dirichlet clients are controlled logical clients, not verified physical devices.",
            "Natural and diagnostic tests are evaluated only after validation-based round selection.",
            "The saved client partition is reused in later poisoning and defense experiments.",
        ],
    }
    with (output_dir / "federated_clean_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Clean Federated V2.6 complete")
    print("Best validation round:", best_round)
    print("Best validation macro F1:", f"{best_val_macro_f1:.6f}")
    print()
    print(final_metrics.to_string(index=False))
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("Checkpoint:", checkpoints_dir / "best_validation_model.pt")
    print("Metadata:", output_dir / "federated_clean_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

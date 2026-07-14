#!/usr/bin/env python3
"""Validation-only federated hyperparameter screening for CIC IoT-DIAD V2.7.

This stage compares clean FedAvg configurations on one fixed client partition.
Only validation metrics are used to rank configurations. The natural and
diagnostic test sets are evaluated once, after the winning configuration and
best communication round have been fixed.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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


DEFAULT_CONFIGS = [
    {
        "candidate": "lr3e4_e1",
        "learning_rate": 3e-4,
        "local_epochs": 1,
    },
    {
        "candidate": "lr8e4_e1",
        "learning_rate": 8e-4,
        "local_epochs": 1,
    },
    {
        "candidate": "lr15e4_e1",
        "learning_rate": 1.5e-3,
        "local_epochs": 1,
    },
    {
        "candidate": "lr8e4_e2",
        "learning_rate": 8e-4,
        "local_epochs": 2,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation-only clean FedAvg configuration screening."
    )
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--participation-rate", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--min-client-samples", type=int, default=500)
    parser.add_argument("--min-client-classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def evaluate_metrics(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> Dict[str, float]:
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
    return metric_dict(labels_array, probabilities)


def train_candidate(
    config: Dict[str, object],
    args: argparse.Namespace,
    arrays: Dict[str, np.ndarray],
    client_indices: List[np.ndarray],
    class_weights: np.ndarray,
    partition_hash: str,
    candidate_dir: Path,
) -> Dict[str, object]:
    candidate = str(config["candidate"])
    learning_rate = float(config["learning_rate"])
    local_epochs = int(config["local_epochs"])

    checkpoints_dir = candidate_dir / "checkpoints"
    tables_dir = candidate_dir / "tables"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    X_train = arrays["X_train"].astype(np.float32, copy=False)
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    X_val = arrays["X_val"].astype(np.float32, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)

    set_seed(args.seed)
    model = build_model(
        architecture="resmlp",
        input_dim=X_train.shape[1],
        num_classes=NUM_CLASSES,
    )
    rng = np.random.default_rng(args.seed)
    participating_count = max(
        1,
        int(math.ceil(args.num_clients * args.participation_rate)),
    )

    best_macro_f1 = -math.inf
    best_round = 0
    best_metrics: Dict[str, float] = {}
    stale_rounds = 0
    round_rows: List[Dict[str, object]] = []
    local_rows: List[Dict[str, object]] = []
    started = time.time()

    print()
    print("=" * 78)
    print(
        f"Candidate {candidate}, learning rate={learning_rate}, "
        f"local epochs={local_epochs}"
    )
    print("=" * 78)

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
        round_local_rows = []

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
                class_weights=class_weights,
                local_epochs=local_epochs,
                batch_size=args.batch_size,
                learning_rate=learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                seed=args.seed + round_id * 1000 + int(client_id),
            )
            local_states.append(state)
            local_counts.append(len(indices))
            local_row = {
                "candidate": candidate,
                "round": round_id,
                "client_id": int(client_id),
                "client_samples": int(len(indices)),
                "learning_rate": learning_rate,
                "local_epochs": local_epochs,
                **metrics,
            }
            local_rows.append(local_row)
            round_local_rows.append(local_row)
            del local_model

        averaged = weighted_average_states(
            states=local_states,
            sample_counts=local_counts,
            reference_state=reference_state,
        )
        model.load_state_dict(averaged)
        val_metrics = evaluate_metrics(
            model,
            X_val,
            y_val,
            batch_size=args.evaluation_batch_size,
        )

        round_row = {
            "candidate": candidate,
            "round": round_id,
            "learning_rate": learning_rate,
            "local_epochs": local_epochs,
            "selected_clients": "|".join(map(str, selected.tolist())),
            "selected_client_count": int(len(selected)),
            "participating_samples": int(sum(local_counts)),
            "mean_local_train_loss": float(
                np.mean([row["local_train_loss"] for row in round_local_rows])
            ),
            "mean_local_train_accuracy": float(
                np.mean([row["local_train_accuracy"] for row in round_local_rows])
            ),
            "round_seconds": float(time.time() - round_started),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        round_rows.append(round_row)

        print(
            f"{candidate} round {round_id:02d}, "
            f"val macro F1={val_metrics['macro_f1']:.4f}, "
            f"balanced accuracy={val_metrics['balanced_accuracy']:.4f}, "
            f"seconds={round_row['round_seconds']:.1f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_round = round_id
            best_metrics = dict(val_metrics)
            stale_rounds = 0
            torch.save(
                {
                    "candidate": candidate,
                    "round": round_id,
                    "model_state_dict": model.state_dict(),
                    "input_dim": X_train.shape[1],
                    "num_classes": NUM_CLASSES,
                    "validation_metrics": val_metrics,
                    "partition_hash": partition_hash,
                    "learning_rate": learning_rate,
                    "local_epochs": local_epochs,
                },
                checkpoints_dir / "best_validation_model.pt",
            )
        else:
            stale_rounds += 1

        if (
            args.early_stopping_patience > 0
            and stale_rounds >= args.early_stopping_patience
        ):
            print(f"{candidate} early stopping after round {round_id}")
            break

    round_table = pd.DataFrame(round_rows)
    local_table = pd.DataFrame(local_rows)
    round_table.to_csv(tables_dir / "round_metrics.csv", index=False)
    local_table.to_csv(tables_dir / "local_client_training_metrics.csv", index=False)

    summary = {
        "candidate": candidate,
        "learning_rate": learning_rate,
        "local_epochs": local_epochs,
        "rounds_completed": int(round_table["round"].max()),
        "best_validation_round": int(best_round),
        "best_validation_macro_f1": float(best_metrics["macro_f1"]),
        "best_validation_balanced_accuracy": float(best_metrics["balanced_accuracy"]),
        "best_validation_accuracy": float(best_metrics["accuracy"]),
        "best_validation_weighted_f1": float(best_metrics["weighted_f1"]),
        "best_validation_mcc": float(best_metrics["mcc"]),
        "best_validation_log_loss": float(best_metrics["log_loss"]),
        "best_validation_ece_15bin": float(best_metrics["ece_15bin"]),
        "total_seconds": float(time.time() - started),
        "checkpoint": str(checkpoints_dir / "best_validation_model.pt"),
    }
    with (candidate_dir / "candidate_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def normalize_confusion(matrix: np.ndarray) -> np.ndarray:
    denominator = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix, dtype=float),
        where=denominator != 0,
    )


def plot_candidate_curves(rounds: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for candidate, group in rounds.groupby("candidate"):
        group = group.sort_values("round")
        ax.plot(group["round"], group["val_macro_f1"], marker="o", label=candidate)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation macro F1")
    ax.set_title("Validation-Only FedAvg Configuration Screening")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def plot_candidate_summary(summary: pd.DataFrame, output: Path) -> None:
    ordered = summary.sort_values("best_validation_macro_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(ordered["candidate"], ordered["best_validation_macro_f1"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Best validation macro F1")
    ax.set_title("Best Validation Performance by FedAvg Configuration")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output)


def plot_runtime_tradeoff(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        summary["total_seconds"],
        summary["best_validation_macro_f1"],
        s=80,
    )
    for _, row in summary.iterrows():
        ax.annotate(
            str(row["candidate"]),
            (float(row["total_seconds"]), float(row["best_validation_macro_f1"])),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Total screening time, seconds")
    ax.set_ylabel("Best validation macro F1")
    ax.set_ylim(0, 1)
    ax.set_title("FedAvg Runtime and Validation Tradeoff")
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
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
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

    torch.set_num_threads(max(1, args.threads))
    set_seed(args.seed)

    data_file = args.data_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    candidate_runs_dir = output_dir / "candidate_runs"
    partitions_dir = output_dir / "partitions"
    for path in (tables_dir, figures_dir, candidate_runs_dir, partitions_dir):
        path.mkdir(parents=True, exist_ok=True)

    arrays = load_protocol_arrays(data_file)
    X_train = arrays["X_train"].astype(np.float32, copy=False)
    y_train = arrays["y_train"].astype(np.int64, copy=False)

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

    class_weights = sqrt_class_weights(y_train, args.max_class_weight)
    pd.DataFrame(
        {
            "class_id": np.arange(NUM_CLASSES),
            "class_name": CLASS_NAMES,
            "train_rows": np.bincount(y_train, minlength=NUM_CLASSES),
            "sqrt_class_weight": class_weights,
        }
    ).to_csv(tables_dir / "global_class_weights.csv", index=False)

    summaries = []
    all_rounds = []
    all_local = []
    for config in DEFAULT_CONFIGS:
        candidate_dir = candidate_runs_dir / str(config["candidate"])
        summary = train_candidate(
            config=config,
            args=args,
            arrays=arrays,
            client_indices=client_indices,
            class_weights=class_weights,
            partition_hash=partition_hash,
            candidate_dir=candidate_dir,
        )
        summaries.append(summary)
        all_rounds.append(pd.read_csv(candidate_dir / "tables" / "round_metrics.csv"))
        all_local.append(
            pd.read_csv(candidate_dir / "tables" / "local_client_training_metrics.csv")
        )

    summary_table = pd.DataFrame(summaries)
    ranking = summary_table.sort_values(
        [
            "best_validation_macro_f1",
            "best_validation_balanced_accuracy",
            "best_validation_log_loss",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "validation_rank", np.arange(1, len(ranking) + 1))
    selected_candidate = str(ranking.iloc[0]["candidate"])

    round_table = pd.concat(all_rounds, ignore_index=True)
    local_table = pd.concat(all_local, ignore_index=True)
    summary_table.to_csv(tables_dir / "candidate_validation_summary.csv", index=False)
    ranking.to_csv(tables_dir / "final_validation_ranking.csv", index=False)
    round_table.to_csv(tables_dir / "candidate_round_metrics.csv", index=False)
    local_table.to_csv(tables_dir / "candidate_local_metrics.csv", index=False)

    selected_checkpoint_path = (
        candidate_runs_dir
        / selected_candidate
        / "checkpoints"
        / "best_validation_model.pt"
    )
    checkpoint = torch.load(
        selected_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    selected_model = build_model(
        architecture="resmlp",
        input_dim=X_train.shape[1],
        num_classes=NUM_CLASSES,
    )
    selected_model.load_state_dict(checkpoint["model_state_dict"])

    final_metrics_rows = []
    per_class_tables = []
    for split_name, X_key, y_key in (
        ("test_natural", "X_test_natural", "y_test_natural"),
        ("test_diagnostic", "X_test_diagnostic", "y_test_diagnostic"),
    ):
        metrics, per_class, matrix, predictions = evaluation_artifacts(
            model=selected_model,
            X=arrays[X_key].astype(np.float32, copy=False),
            y=arrays[y_key].astype(np.int64, copy=False),
            batch_size=args.evaluation_batch_size,
            split=split_name,
        )
        final_metrics_rows.append(
            {
                "selected_candidate": selected_candidate,
                "best_validation_round": int(checkpoint["round"]),
                "split": split_name,
                **metrics,
            }
        )
        per_class.insert(0, "selected_candidate", selected_candidate)
        per_class_tables.append(per_class)
        pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            tables_dir / f"selected_{split_name}_confusion_matrix.csv"
        )
        predictions.to_csv(
            tables_dir / f"selected_{split_name}_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
        plot_confusion(
            matrix,
            f"Selected Clean FedAvg, {selected_candidate}, {split_name}",
            figures_dir / f"selected_{split_name}_normalized_confusion_matrix",
        )
        plot_per_class_f1(
            per_class,
            f"Selected Clean FedAvg Per-Class F1, {split_name}",
            figures_dir / f"selected_{split_name}_per_class_f1",
        )

    final_metrics = pd.DataFrame(final_metrics_rows)
    final_per_class = pd.concat(per_class_tables, ignore_index=True)
    final_metrics.to_csv(tables_dir / "selected_final_test_metrics.csv", index=False)
    final_per_class.to_csv(tables_dir / "selected_final_per_class_metrics.csv", index=False)

    plot_candidate_curves(round_table, figures_dir / "candidate_validation_curves")
    plot_candidate_summary(summary_table, figures_dir / "candidate_best_validation_macro_f1")
    plot_runtime_tradeoff(summary_table, figures_dir / "candidate_runtime_tradeoff")

    metadata = {
        "experiment_version": "2.7",
        "experiment_type": "clean_fedavg_validation_only_hyperparameter_screening",
        "contains_poisoning": False,
        "contains_defense": False,
        "selection_rule": (
            "Highest best validation macro F1, then validation balanced accuracy, "
            "then lowest validation log loss."
        ),
        "test_sets_evaluated_before_selection": False,
        "test_sets_used_for_model_selection": False,
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "model": "ResidualMLP",
        "loss": "sqrt_class_weighted_cross_entropy",
        "candidate_configs": DEFAULT_CONFIGS,
        "selected_candidate": selected_candidate,
        "selected_checkpoint": str(selected_checkpoint_path),
        "num_clients": args.num_clients,
        "dirichlet_alpha": args.dirichlet_alpha,
        "partition_hash_sha256": partition_hash,
        "seed": args.seed,
        "rounds": args.rounds,
        "participation_rate": args.participation_rate,
        "tables": str(tables_dir),
        "figures_png_pdf": str(figures_dir),
        "notes": [
            "A single development seed is used for hyperparameter screening.",
            "The selected configuration must later be evaluated with multiple seeds.",
            "Natural and diagnostic test sets are evaluated only after validation-only selection.",
            "The same saved Dirichlet partition is used for every candidate.",
        ],
    }
    with (output_dir / "federated_tuning_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Federated Tuning V2.7 complete")
    print()
    print("Validation-only ranking")
    print(ranking.to_string(index=False))
    print()
    print("Selected candidate:", selected_candidate)
    print()
    print(final_metrics.to_string(index=False))
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("Metadata:", output_dir / "federated_tuning_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

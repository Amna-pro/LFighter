#!/usr/bin/env python3
"""Five-seed clean FedAvg reliability evaluation for CIC IoT-DIAD V2.8.

The script uses the validation-selected V2.7 configuration and one fixed saved
Dirichlet partition. Each seed selects its best communication round using
validation macro F1 only. Natural and diagnostic tests are evaluated only after
that seed's best validation checkpoint is fixed.

Outputs include seed-level and aggregate CSV tables, checkpoints, JSON metadata,
and matching PNG/PDF figures. Completed seeds can be reused with --skip-existing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

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


SOURCE_CLASS = "DDoS"
TARGET_CLASS = "DoS"
SOURCE_ID = CLASS_NAMES.index(SOURCE_CLASS)
TARGET_ID = CLASS_NAMES.index(TARGET_CLASS)
BENIGN_ID = CLASS_NAMES.index("Benign")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run five-seed clean FedAvg reliability evaluation."
    )
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", default="42,123,2026,7,99")
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
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save compressed sample-level prediction CSVs for every seed.",
    )
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
    if len(np.unique(all_indices)) != train_rows:
        raise ValueError("Partition contains duplicate or missing training indices.")
    if int(all_indices.min()) != 0 or int(all_indices.max()) != train_rows - 1:
        raise ValueError("Partition indices do not cover the full training range.")

    digest = hashlib.sha256()
    for client_id, indices in enumerate(partitions):
        key = f"client_{client_id:03d}"
        digest.update(key.encode("utf-8"))
        digest.update(indices.tobytes())
    return partitions, digest.hexdigest()


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
    label_array = np.concatenate(labels)
    return metric_dict(label_array, probabilities)


def attack_pair_row(
    seed: int,
    split: str,
    matrix: np.ndarray,
) -> Dict[str, object]:
    source_total = int(matrix[SOURCE_ID].sum())
    target_total = int(matrix[TARGET_ID].sum())
    return {
        "seed": seed,
        "split": split,
        "source_class": SOURCE_CLASS,
        "target_class": TARGET_CLASS,
        "source_support": source_total,
        "target_support": target_total,
        "source_recall": float(matrix[SOURCE_ID, SOURCE_ID] / max(source_total, 1)),
        "target_recall": float(matrix[TARGET_ID, TARGET_ID] / max(target_total, 1)),
        "source_to_target_count": int(matrix[SOURCE_ID, TARGET_ID]),
        "source_to_target_rate": float(
            matrix[SOURCE_ID, TARGET_ID] / max(source_total, 1)
        ),
        "source_to_benign_count": int(matrix[SOURCE_ID, BENIGN_ID]),
        "source_to_benign_rate": float(
            matrix[SOURCE_ID, BENIGN_ID] / max(source_total, 1)
        ),
        "target_to_source_count": int(matrix[TARGET_ID, SOURCE_ID]),
        "target_to_source_rate": float(
            matrix[TARGET_ID, SOURCE_ID] / max(target_total, 1)
        ),
    }


def seed_complete(seed_dir: Path) -> bool:
    required = [
        seed_dir / "seed_metadata.json",
        seed_dir / "tables" / "round_metrics.csv",
        seed_dir / "tables" / "final_test_metrics.csv",
        seed_dir / "tables" / "final_per_class_metrics.csv",
        seed_dir / "tables" / "attack_pair_baseline.csv",
        seed_dir / "checkpoints" / "best_validation_model.pt",
    ]
    return all(path.exists() for path in required)


def run_seed(
    seed: int,
    args: argparse.Namespace,
    arrays: Dict[str, np.ndarray],
    client_indices: Sequence[np.ndarray],
    partition_hash: str,
    class_weights: np.ndarray,
    seed_dir: Path,
) -> None:
    if args.skip_existing and seed_complete(seed_dir):
        print(f"Reusing completed seed {seed}: {seed_dir}")
        return

    tables_dir = seed_dir / "tables"
    checkpoints_dir = seed_dir / "checkpoints"
    figures_dir = seed_dir / "figures"
    for path in (tables_dir, checkpoints_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    X_train = arrays["X_train"].astype(np.float32, copy=False)
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    X_val = arrays["X_val"].astype(np.float32, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)

    set_seed(seed)
    model = build_model(
        architecture="resmlp",
        input_dim=X_train.shape[1],
        num_classes=NUM_CLASSES,
    )
    rng = np.random.default_rng(seed)
    participating_count = max(
        1,
        int(math.ceil(args.num_clients * args.participation_rate)),
    )

    round_rows: List[Dict[str, object]] = []
    local_rows: List[Dict[str, object]] = []
    best_macro_f1 = -math.inf
    best_round = 0
    best_metrics: Dict[str, float] = {}
    stale_rounds = 0
    started = time.time()

    print()
    print("=" * 78)
    print(f"Clean FedAvg seed {seed}")
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
        current_local_rows = []

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
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                seed=seed + round_id * 1000 + int(client_id),
            )
            local_states.append(state)
            local_counts.append(len(indices))
            row = {
                "seed": seed,
                "round": round_id,
                "client_id": int(client_id),
                "client_samples": int(len(indices)),
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
        val_metrics = evaluate_metrics(
            model,
            X_val,
            y_val,
            batch_size=args.evaluation_batch_size,
        )

        round_row = {
            "seed": seed,
            "round": round_id,
            "selected_clients": "|".join(map(str, selected.tolist())),
            "selected_client_count": int(len(selected)),
            "participating_samples": int(sum(local_counts)),
            "mean_local_train_loss": float(
                np.mean([row["local_train_loss"] for row in current_local_rows])
            ),
            "mean_local_train_accuracy": float(
                np.mean([row["local_train_accuracy"] for row in current_local_rows])
            ),
            "round_seconds": float(time.time() - round_started),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        round_rows.append(round_row)

        print(
            f"seed {seed}, round {round_id:02d}, "
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
                    "seed": seed,
                    "round": round_id,
                    "model_state_dict": model.state_dict(),
                    "input_dim": X_train.shape[1],
                    "num_classes": NUM_CLASSES,
                    "validation_metrics": val_metrics,
                    "partition_hash": partition_hash,
                    "learning_rate": args.learning_rate,
                    "local_epochs": args.local_epochs,
                },
                checkpoints_dir / "best_validation_model.pt",
            )
        else:
            stale_rounds += 1

        pd.DataFrame(round_rows).to_csv(
            tables_dir / "round_metrics_partial.csv",
            index=False,
        )

        if (
            args.early_stopping_patience > 0
            and stale_rounds >= args.early_stopping_patience
        ):
            print(f"seed {seed} early stopping after round {round_id}")
            break

    round_table = pd.DataFrame(round_rows)
    local_table = pd.DataFrame(local_rows)
    round_table.to_csv(tables_dir / "round_metrics.csv", index=False)
    local_table.to_csv(tables_dir / "local_client_training_metrics.csv", index=False)

    checkpoint = torch.load(
        checkpoints_dir / "best_validation_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    final_rows = []
    per_class_tables = []
    attack_rows = []
    for split_name, X_key, y_key in (
        ("test_natural", "X_test_natural", "y_test_natural"),
        ("test_diagnostic", "X_test_diagnostic", "y_test_diagnostic"),
    ):
        metrics, per_class, matrix, predictions = evaluation_artifacts(
            model=model,
            X=arrays[X_key].astype(np.float32, copy=False),
            y=arrays[y_key].astype(np.int64, copy=False),
            batch_size=args.evaluation_batch_size,
            split=split_name,
        )
        final_rows.append(
            {
                "seed": seed,
                "best_validation_round": best_round,
                "best_validation_macro_f1": best_macro_f1,
                "split": split_name,
                **metrics,
            }
        )
        per_class.insert(0, "seed", seed)
        per_class_tables.append(per_class)
        attack_rows.append(attack_pair_row(seed, split_name, matrix))

        pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            tables_dir / f"{split_name}_confusion_matrix.csv"
        )
        if args.save_predictions:
            predictions.to_csv(
                tables_dir / f"{split_name}_predictions.csv.gz",
                index=False,
                compression="gzip",
            )

    final_table = pd.DataFrame(final_rows)
    per_class_table = pd.concat(per_class_tables, ignore_index=True)
    attack_table = pd.DataFrame(attack_rows)
    final_table.to_csv(tables_dir / "final_test_metrics.csv", index=False)
    per_class_table.to_csv(tables_dir / "final_per_class_metrics.csv", index=False)
    attack_table.to_csv(tables_dir / "attack_pair_baseline.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(round_table["round"], round_table["val_macro_f1"], marker="o")
    ax.axvline(best_round, linestyle="--", label=f"best round {best_round}")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation macro F1")
    ax.set_title(f"Clean FedAvg Validation Curve, Seed {seed}")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "validation_macro_f1_by_round")

    metadata = {
        "experiment_version": "2.8",
        "seed": seed,
        "partition_hash_sha256": partition_hash,
        "learning_rate": args.learning_rate,
        "local_epochs": args.local_epochs,
        "rounds_requested": args.rounds,
        "rounds_completed": int(round_table["round"].max()),
        "best_validation_round": best_round,
        "best_validation_metrics": best_metrics,
        "test_sets_used_for_round_selection": False,
        "total_seconds": float(time.time() - started),
        "checkpoint": str(checkpoints_dir / "best_validation_model.pt"),
        "predictions_saved": bool(args.save_predictions),
    }
    with (seed_dir / "seed_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def ci95(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    n = len(clean)
    if n < 2:
        return 0.0
    critical = float(student_t.ppf(0.975, df=n - 1))
    return float(critical * clean.std(ddof=1) / math.sqrt(n))


def aggregate_metrics(all_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "log_loss",
        "ece_15bin",
        "best_validation_macro_f1",
        "best_validation_round",
    ]
    rows = []
    for split, group in all_metrics.groupby("split"):
        for metric in metric_names:
            values = group[metric].astype(float)
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "n_seeds": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(values.median()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "ci95_half_width_t": ci95(values),
                }
            )
    return pd.DataFrame(rows)


def aggregate_per_class(per_class: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in per_class.groupby(
        ["split", "class_id", "class_name"]
    ):
        split, class_id, class_name = keys
        for metric in ("precision", "recall", "f1_score"):
            values = group[metric].astype(float)
            rows.append(
                {
                    "split": split,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "metric": metric,
                    "support_per_seed": int(group["support"].iloc[0]),
                    "n_seeds": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_half_width_t": ci95(values),
                }
            )
    return pd.DataFrame(rows)


def aggregate_attack(attack: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "source_recall",
        "target_recall",
        "source_to_target_rate",
        "source_to_benign_rate",
        "target_to_source_rate",
    ]
    rows = []
    for split, group in attack.groupby("split"):
        for metric in metrics:
            values = group[metric].astype(float)
            rows.append(
                {
                    "split": split,
                    "source_class": SOURCE_CLASS,
                    "target_class": TARGET_CLASS,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_half_width_t": ci95(values),
                }
            )
    return pd.DataFrame(rows)


def plot_validation_curves(rounds: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for seed, group in rounds.groupby("seed"):
        group = group.sort_values("round")
        ax.plot(group["round"], group["val_macro_f1"], marker="o", label=str(seed))
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation macro F1")
    ax.set_title("Five-Seed Clean FedAvg Validation Stability")
    ax.grid(alpha=0.25)
    ax.legend(title="Seed")
    save_figure(fig, output)


def plot_test_macro_f1(metrics: pd.DataFrame, output: Path) -> None:
    summary = metrics.groupby("split", as_index=False).agg(
        mean=("macro_f1", "mean"),
        std=("macro_f1", "std"),
        count=("macro_f1", "count"),
    )
    summary["ci95"] = [
        ci95(metrics.loc[metrics["split"] == split, "macro_f1"])
        for split in summary["split"]
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(summary["split"], summary["mean"], yerr=summary["ci95"], capsize=5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Test split")
    ax.set_ylabel("Macro F1")
    ax.set_title("Clean FedAvg Test Reliability, Mean and 95% t-CI")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output)


def plot_per_class_f1(summary: pd.DataFrame, split: str, output: Path) -> None:
    data = summary[
        (summary["split"] == split) & (summary["metric"] == "f1_score")
    ].sort_values("class_id")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        data["class_name"],
        data["mean"],
        yerr=data["ci95_half_width_t"],
        capsize=4,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Class")
    ax.set_ylabel("F1 score")
    ax.set_title(f"Five-Seed Per-Class F1, {split}")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output)


def plot_attack_baseline(attack: pd.DataFrame, output: Path) -> None:
    data = attack[attack["split"] == "test_natural"]
    metrics = ["source_recall", "target_recall", "source_to_target_rate"]
    labels = ["DDoS recall", "DoS recall", "DDoS to DoS rate"]
    means = [float(data[metric].mean()) for metric in metrics]
    errors = [ci95(data[metric]) for metric in metrics]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, means, yerr=errors, capsize=5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Clean Baseline for the Planned DDoS to DoS Attack")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output)


def main() -> int:
    args = parse_args()
    if not 0 < args.participation_rate <= 1:
        raise ValueError("participation-rate must be in (0, 1]")

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 3:
        raise ValueError("Use at least three seeds.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    torch.set_num_threads(max(1, args.threads))
    data_file = args.data_file.expanduser().resolve()
    partition_file = args.partition_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    seed_runs_dir = output_dir / "seed_runs"
    for path in (tables_dir, figures_dir, seed_runs_dir):
        path.mkdir(parents=True, exist_ok=True)

    arrays = load_protocol_arrays(data_file)
    X_train = arrays["X_train"].astype(np.float32, copy=False)
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    client_indices, partition_hash = load_fixed_partitions(
        partition_file,
        expected_clients=args.num_clients,
        train_rows=len(y_train),
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

    for seed in seeds:
        run_seed(
            seed=seed,
            args=args,
            arrays=arrays,
            client_indices=client_indices,
            partition_hash=partition_hash,
            class_weights=class_weights,
            seed_dir=seed_runs_dir / f"seed_{seed}",
        )

    all_rounds = []
    all_metrics = []
    all_per_class = []
    all_attack = []
    seed_metadata = []
    for seed in seeds:
        seed_dir = seed_runs_dir / f"seed_{seed}"
        if not seed_complete(seed_dir):
            raise RuntimeError(f"Seed {seed} is incomplete: {seed_dir}")
        all_rounds.append(pd.read_csv(seed_dir / "tables" / "round_metrics.csv"))
        all_metrics.append(pd.read_csv(seed_dir / "tables" / "final_test_metrics.csv"))
        all_per_class.append(
            pd.read_csv(seed_dir / "tables" / "final_per_class_metrics.csv")
        )
        all_attack.append(
            pd.read_csv(seed_dir / "tables" / "attack_pair_baseline.csv")
        )
        with (seed_dir / "seed_metadata.json").open("r", encoding="utf-8") as handle:
            seed_metadata.append(json.load(handle))

    rounds = pd.concat(all_rounds, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    per_class = pd.concat(all_per_class, ignore_index=True)
    attack = pd.concat(all_attack, ignore_index=True)

    hashes = {entry["partition_hash_sha256"] for entry in seed_metadata}
    if hashes != {partition_hash}:
        raise RuntimeError("Partition hash mismatch across seed runs.")

    rounds.to_csv(tables_dir / "round_metrics_all_seeds.csv", index=False)
    metrics.to_csv(tables_dir / "final_test_metrics_all_seeds.csv", index=False)
    per_class.to_csv(tables_dir / "final_per_class_metrics_all_seeds.csv", index=False)
    attack.to_csv(tables_dir / "attack_pair_baseline_all_seeds.csv", index=False)

    aggregate_metrics_table = aggregate_metrics(metrics)
    aggregate_class_table = aggregate_per_class(per_class)
    aggregate_attack_table = aggregate_attack(attack)
    aggregate_metrics_table.to_csv(
        tables_dir / "aggregate_metrics_mean_std_ci.csv",
        index=False,
    )
    aggregate_class_table.to_csv(
        tables_dir / "aggregate_per_class_mean_std_ci.csv",
        index=False,
    )
    aggregate_attack_table.to_csv(
        tables_dir / "aggregate_attack_pair_baseline.csv",
        index=False,
    )

    coverage = pd.DataFrame(
        {
            "seed": seeds,
            "completed": [True] * len(seeds),
            "partition_hash_sha256": [partition_hash] * len(seeds),
            "best_validation_round": [
                entry["best_validation_round"] for entry in seed_metadata
            ],
            "best_validation_macro_f1": [
                entry["best_validation_metrics"]["macro_f1"]
                for entry in seed_metadata
            ],
        }
    )
    coverage.to_csv(tables_dir / "seed_coverage_and_selection.csv", index=False)

    plot_validation_curves(rounds, figures_dir / "validation_curves_all_seeds")
    plot_test_macro_f1(metrics, figures_dir / "test_macro_f1_mean_ci")
    plot_per_class_f1(
        aggregate_class_table,
        "test_natural",
        figures_dir / "natural_per_class_f1_mean_ci",
    )
    plot_per_class_f1(
        aggregate_class_table,
        "test_diagnostic",
        figures_dir / "diagnostic_per_class_f1_mean_ci",
    )
    plot_attack_baseline(
        attack,
        figures_dir / "ddos_to_dos_clean_attack_baseline",
    )

    metadata = {
        "experiment_version": "2.8",
        "experiment_type": "clean_fedavg_fixed_partition_multiseed",
        "contains_poisoning": False,
        "contains_defense": False,
        "model": "ResidualMLP",
        "loss": "sqrt_class_weighted_cross_entropy",
        "learning_rate": args.learning_rate,
        "local_epochs": args.local_epochs,
        "num_clients": args.num_clients,
        "participation_rate": args.participation_rate,
        "rounds": args.rounds,
        "early_stopping_patience": args.early_stopping_patience,
        "seeds": seeds,
        "partition_file": str(partition_file),
        "partition_hash_sha256": partition_hash,
        "partition_fixed_across_seeds": True,
        "selection_rule": "Best validation macro F1 independently within each seed.",
        "test_sets_used_for_round_selection": False,
        "planned_attack_pair": {
            "source": SOURCE_CLASS,
            "target": TARGET_CLASS,
        },
        "confidence_interval": "Two-sided 95% Student t interval across seeds.",
        "predictions_saved": bool(args.save_predictions),
        "notes": [
            "This stage establishes clean multi-seed reliability before poisoning.",
            "The fixed partition isolates model and local-training randomness.",
            "A later heterogeneity ablation will vary the Dirichlet partition.",
            "DDoS to DoS is evaluated as the first targeted label-flipping pair.",
        ],
    }
    with (output_dir / "federated_multiseed_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Federated Multi-Seed V2.8 complete")
    print("Partition hash:", partition_hash)
    print()
    print("Seed coverage")
    print(coverage.to_string(index=False))
    print()
    print("Aggregate test metrics")
    print(
        aggregate_metrics_table[
            aggregate_metrics_table["metric"].isin(
                ["accuracy", "balanced_accuracy", "macro_f1", "mcc"]
            )
        ].to_string(index=False)
    )
    print()
    print("Clean DDoS to DoS attack baseline")
    print(aggregate_attack_table.to_string(index=False))
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("Metadata:", output_dir / "federated_multiseed_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

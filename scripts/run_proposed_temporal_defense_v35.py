#!/usr/bin/env python3
"""Run one clean or strongly attacked proposed temporal defense experiment, V3.5 stage 1."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping

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
    evaluation_artifacts,
    load_protocol_arrays,
    make_loader,
    metric_dict,
    partition_manifest,
    set_seed,
    sqrt_class_weights,
    stable_softmax,
    train_local_model,
)
from neural_models_v24 import build_model  # noqa: E402
from proposed_temporal_defense_v35 import (  # noqa: E402
    TemporalDefenseMemory,
    aggregate_proposed_temporal_defense,
)
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
        description="Run the V3.5 stage-1 temporal, probe-aware defense."
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
    parser.add_argument("--probe-per-class", type=int, default=48)
    parser.add_argument("--explanation-per-class", type=int, default=12)
    parser.add_argument("--probe-seed", type=int, default=3501)
    parser.add_argument("--ema-decay", type=float, default=0.65)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--quarantine-threshold", type=float, default=2.25)
    parser.add_argument("--strike-threshold", type=float, default=1.50)
    parser.add_argument("--required-strikes", type=int, default=2)
    parser.add_argument("--minimum-admitted-fraction", type=float, default=0.60)
    parser.add_argument("--soft-weight-temperature", type=float, default=1.10)
    parser.add_argument("--minimum-risk-weight", type=float, default=0.05)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_client_ids(text: str) -> List[int]:
    return sorted({int(value.strip()) for value in text.split(",") if value.strip()})


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


def balanced_probe_indices(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(NUM_CLASSES):
        candidates = np.where(y == class_id)[0]
        if len(candidates) == 0:
            continue
        count = min(int(per_class), len(candidates))
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    result = np.asarray(selected, dtype=np.int64)
    rng.shuffle(result)
    return result


def predict_probabilities(model: torch.nn.Module, X: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            features = torch.from_numpy(X[start : start + batch_size].astype(np.float32, copy=False))
            outputs.append(model(features).cpu().numpy())
    return stable_softmax(np.concatenate(outputs, axis=0))


def class_conditional_probability_means(
    probabilities: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    means = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for class_id in range(NUM_CLASSES):
        mask = labels == class_id
        if mask.any():
            means[class_id] = probabilities[mask].mean(axis=0)
    return means


def explanation_signature(
    model: torch.nn.Module, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    model.eval()
    features = torch.from_numpy(X.astype(np.float32, copy=False)).clone().requires_grad_(True)
    labels = torch.from_numpy(y.astype(np.int64, copy=False))
    model.zero_grad(set_to_none=True)
    logits = model(features)
    selected_logits = logits.gather(1, labels.unsqueeze(1)).sum()
    selected_logits.backward()
    attribution = (features.grad * features).abs().mean(dim=0).detach().cpu().numpy()
    attribution = np.nan_to_num(attribution, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(attribution.sum())
    if total > 0:
        attribution = attribution / total
    return attribution.astype(np.float64, copy=False)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    similarity = float(np.dot(left, right) / denominator)
    return float(np.clip(1.0 - similarity, 0.0, 2.0))


def local_probe_signal(
    model: torch.nn.Module,
    X_probe: np.ndarray,
    y_probe: np.ndarray,
    X_explain: np.ndarray,
    y_explain: np.ndarray,
    reference_class_means: np.ndarray,
    reference_explanation: np.ndarray,
    batch_size: int,
) -> Dict[str, object]:
    probabilities = predict_probabilities(model, X_probe, batch_size)
    local_means = class_conditional_probability_means(probabilities, y_probe)
    drift = local_means - reference_class_means
    offdiag = drift.copy()
    np.fill_diagonal(offdiag, -np.inf)
    source_id, target_id = np.unravel_index(np.argmax(offdiag), offdiag.shape)
    maximum = float(max(offdiag[source_id, target_id], 0.0))
    signature = explanation_signature(model, X_explain, y_explain)
    return {
        "max_offdiag_probability_drift": maximum,
        "max_drift_source_id": int(source_id),
        "max_drift_source_name": CLASS_NAMES[int(source_id)],
        "max_drift_target_id": int(target_id),
        "max_drift_target_name": CLASS_NAMES[int(target_id)],
        "max_drift_pair": f"{CLASS_NAMES[int(source_id)]}->{CLASS_NAMES[int(target_id)]}",
        "explanation_cosine_distance": cosine_distance(signature, reference_explanation),
    }


def evaluate_checkpoint(
    checkpoint_name: str,
    checkpoint_path: Path,
    arrays: Mapping[str, np.ndarray],
    source_id: int,
    target_id: int,
    batch_size: int,
    tables_dir: Path,
    save_predictions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(
        architecture="resmlp",
        input_dim=int(checkpoint["input_dim"]),
        num_classes=int(checkpoint["num_classes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    metric_rows, class_tables, pair_rows = [], [], []
    for split, x_key, y_key in (
        ("test_natural", "X_test_natural", "y_test_natural"),
        ("test_diagnostic", "X_test_diagnostic", "y_test_diagnostic"),
    ):
        metrics, per_class, matrix, predictions = evaluation_artifacts(
            model=model,
            X=arrays[x_key].astype(np.float32, copy=False),
            y=arrays[y_key].astype(np.int64, copy=False),
            batch_size=batch_size,
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
        class_tables.append(per_class)
        pair_rows.append(
            {
                "checkpoint": checkpoint_name,
                "checkpoint_round": int(checkpoint["round"]),
                "split": split,
                **attack_metrics_from_matrix(matrix, source_id, target_id),
            }
        )
        pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            tables_dir / f"{checkpoint_name}_{split}_confusion_matrix.csv"
        )
        if save_predictions:
            predictions.to_csv(
                tables_dir / f"{checkpoint_name}_{split}_predictions.csv.gz",
                index=False,
                compression="gzip",
            )
    return pd.DataFrame(metric_rows), pd.concat(class_tables), pd.DataFrame(pair_rows)


def plot_round_curves(table: pd.DataFrame, source: str, target: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(table["round"], table["val_macro_f1"], marker="o", label="Validation macro F1")
    ax.plot(
        table["round"],
        table["val_source_to_target_rate"],
        marker="s",
        label=f"Validation {source} to {target}",
    )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Proposed temporal defense validation dynamics")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out)


def plot_security(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(table["round"], table["benign_retention_rate"], marker="o", label="Benign retention")
    ax.plot(table["round"], table["malicious_rejection_recall"], marker="s", label="Malicious rejection recall")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Proposed defense client decisions")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out)


def main() -> int:
    args = parse_args()
    if args.source_class not in CLASS_NAMES or args.target_class not in CLASS_NAMES:
        raise ValueError("Unknown source or target class")
    if args.source_class == args.target_class:
        raise ValueError("Source and target classes must differ")
    if not math.isclose(args.participation_rate, 1.0):
        raise ValueError("V3.5 stage 1 is frozen to full client participation")

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
        malicious_clients, poisoned_positions, poison_manifest, poison_hash = prepare_static_attack(
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
    else:
        malicious_clients = []
        poisoned_positions = {
            client_id: np.empty(0, dtype=np.int64) for client_id in range(args.num_clients)
        }
        poison_manifest = pd.DataFrame(
            [
                {
                    "client_id": client_id,
                    "is_malicious": False,
                    "client_rows": int(len(indices)),
                    "source_rows_before_poisoning": int(np.sum(y_train[indices] == source_id)),
                    "poisoned_source_rows": 0,
                    "client_source_poison_rate": 0.0,
                    "source_rows_after_poisoning": int(np.sum(y_train[indices] == source_id)),
                    "target_rows_added_by_poisoning": 0,
                }
                for client_id, indices in enumerate(client_indices)
            ]
        )
        digest = hashlib.sha256()
        for client_id in range(args.num_clients):
            digest.update(f"client_{client_id:03d}".encode("utf-8"))
        poison_hash = digest.hexdigest()

    poison_manifest.to_csv(attack_dir / "malicious_client_poison_manifest.csv", index=False)
    save_poisoned_indices(client_indices, poisoned_positions, attack_dir / "poisoned_indices.npz")
    total_source_rows = int(np.sum(y_train == source_id))
    total_flipped_rows = int(poison_manifest["poisoned_source_rows"].sum())
    pd.DataFrame(
        [
            {
                "mode": args.mode,
                "source_class": args.source_class,
                "target_class": args.target_class,
                "num_clients": args.num_clients,
                "malicious_clients": "|".join(map(str, malicious_clients)),
                "malicious_client_count": len(malicious_clients),
                "global_source_rows": total_source_rows,
                "global_flipped_rows": total_flipped_rows,
                "global_source_exposure_fraction": total_flipped_rows / max(total_source_rows, 1),
                "partition_hash_sha256": partition_hash,
                "poison_index_hash_sha256": poison_hash,
            }
        ]
    ).to_csv(attack_dir / "attack_summary.csv", index=False)

    prediction_probe_indices = balanced_probe_indices(y_val, args.probe_per_class, args.probe_seed)
    explanation_probe_indices = balanced_probe_indices(
        y_val, args.explanation_per_class, args.probe_seed + 1
    )
    X_probe, y_probe = X_val[prediction_probe_indices], y_val[prediction_probe_indices]
    X_explain, y_explain = X_val[explanation_probe_indices], y_val[explanation_probe_indices]
    pd.DataFrame(
        {
            "validation_row_index": prediction_probe_indices,
            "class_id": y_probe,
            "class_name": [CLASS_NAMES[int(value)] for value in y_probe],
        }
    ).to_csv(tables_dir / "prediction_probe_manifest.csv", index=False)
    pd.DataFrame(
        {
            "validation_row_index": explanation_probe_indices,
            "class_id": y_explain,
            "class_name": [CLASS_NAMES[int(value)] for value in y_explain],
        }
    ).to_csv(tables_dir / "explanation_probe_manifest.csv", index=False)

    class_weights = sqrt_class_weights(y_train, args.max_class_weight)
    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    memory = TemporalDefenseMemory()
    rng = np.random.default_rng(args.model_seed)

    round_rows: List[Dict[str, object]] = []
    local_rows: List[Dict[str, object]] = []
    decision_tables: List[pd.DataFrame] = []
    signal_tables: List[pd.DataFrame] = []
    best_macro_f1 = -math.inf
    best_round = 0
    stale_rounds = 0
    started = time.time()

    print("Proposed Temporal Defense V3.5 Stage 1")
    print("Mode:", args.mode)
    print("Model seed:", args.model_seed)
    print("Partition hash:", partition_hash)
    print("Malicious clients:", malicious_clients)

    for round_id in range(1, args.rounds + 1):
        round_started = time.time()
        selected = np.sort(rng.choice(args.num_clients, size=args.num_clients, replace=False))
        reference_state = copy.deepcopy(model.state_dict())
        reference_probabilities = predict_probabilities(model, X_probe, args.evaluation_batch_size)
        reference_class_means = class_conditional_probability_means(reference_probabilities, y_probe)
        reference_explanation = explanation_signature(model, X_explain, y_explain)

        local_states = []
        sample_counts = []
        probe_rows = []
        current_local_rows = []
        for client_id_value in selected:
            client_id = int(client_id_value)
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
            signal = local_probe_signal(
                model=local_model,
                X_probe=X_probe,
                y_probe=y_probe,
                X_explain=X_explain,
                y_explain=y_explain,
                reference_class_means=reference_class_means,
                reference_explanation=reference_explanation,
                batch_size=args.evaluation_batch_size,
            )
            local_states.append(state)
            sample_counts.append(int(len(indices)))
            probe_rows.append({"client_id": client_id, **signal})
            row = {
                "round": round_id,
                "client_id": client_id,
                "is_malicious": client_id in malicious_clients,
                "client_samples": int(len(indices)),
                "poisoned_rows": int(len(positions)),
                **training_metrics,
                **signal,
            }
            local_rows.append(row)
            current_local_rows.append(row)
            del local_model

        decision = aggregate_proposed_temporal_defense(
            reference_state=reference_state,
            local_states=local_states,
            client_ids=[int(value) for value in selected],
            sample_counts=sample_counts,
            probe_signals=pd.DataFrame(probe_rows),
            memory=memory,
            round_id=round_id,
            malicious_client_ids=malicious_clients,
            ema_decay=args.ema_decay,
            warmup_rounds=args.warmup_rounds,
            quarantine_threshold=args.quarantine_threshold,
            strike_threshold=args.strike_threshold,
            required_strikes=args.required_strikes,
            minimum_admitted_fraction=args.minimum_admitted_fraction,
            soft_weight_temperature=args.soft_weight_temperature,
            minimum_risk_weight=args.minimum_risk_weight,
        )
        model.load_state_dict(decision.aggregated_state)
        clients = decision.client_decisions.copy()
        decision_tables.append(clients)
        signal_tables.append(decision.signal_summary.copy())

        val_metrics, val_pair = evaluate_validation(
            model, X_val, y_val, source_id, target_id, args.evaluation_batch_size
        )
        row = {
            "round": round_id,
            "selected_clients": "|".join(map(str, selected.tolist())),
            "selected_client_count": len(selected),
            "participating_samples": int(sum(sample_counts)),
            "mean_local_train_loss": float(np.mean([r["local_train_loss"] for r in current_local_rows])),
            "mean_local_train_accuracy": float(np.mean([r["local_train_accuracy"] for r in current_local_rows])),
            "round_seconds": float(time.time() - round_started),
            **decision.summary,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"val_{key}": value for key, value in val_pair.items()},
        }
        round_rows.append(row)
        print(
            f"round {round_id:02d}, val macro F1={val_metrics['macro_f1']:.4f}, "
            f"{args.source_class}->{args.target_class}={val_pair['source_to_target_rate']:.4%}, "
            f"rejected={decision.summary['rejected_client_count']}, "
            f"malicious rejection={decision.summary['malicious_rejection_recall']:.3f}, "
            f"benign retention={decision.summary['benign_retention_rate']:.3f}, "
            f"probe pair={decision.summary['dominant_probe_pair']}"
        )

        checkpoint_payload = {
            "experiment_version": "3.5-stage1",
            "aggregation_rule": "temporal_probe_explanation_weighted_aggregation",
            "mode": args.mode,
            "checkpoint_type": "last_round",
            "round": round_id,
            "model_state_dict": model.state_dict(),
            "input_dim": X_train.shape[1],
            "num_classes": NUM_CLASSES,
            "validation_metrics": val_metrics,
            "validation_source_target_metrics": val_pair,
            "defense_summary": decision.summary,
            "temporal_memory": {
                "suspicion_ema": memory.suspicion_ema,
                "strike_count": memory.strike_count,
            },
            "partition_hash": partition_hash,
            "poison_index_hash": poison_hash,
            "source_class": args.source_class,
            "target_class": args.target_class,
            "malicious_clients": malicious_clients,
        }
        torch.save(checkpoint_payload, checkpoints_dir / "last_round_model.pt")
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_round = round_id
            stale_rounds = 0
            checkpoint_payload = dict(checkpoint_payload)
            checkpoint_payload["checkpoint_type"] = "best_validation"
            torch.save(checkpoint_payload, checkpoints_dir / "best_validation_model.pt")
        else:
            stale_rounds += 1

        pd.DataFrame(round_rows).to_csv(tables_dir / "round_metrics_partial.csv", index=False)
        if args.early_stopping_patience > 0 and stale_rounds >= args.early_stopping_patience:
            print(f"Early stopping after round {round_id}")
            break

    round_table = pd.DataFrame(round_rows)
    local_table = pd.DataFrame(local_rows)
    decisions_table = pd.concat(decision_tables, ignore_index=True)
    signals_table = pd.concat(signal_tables, ignore_index=True)
    round_table.to_csv(tables_dir / "round_metrics.csv", index=False)
    local_table.to_csv(tables_dir / "local_client_training_metrics.csv", index=False)
    decisions_table.to_csv(tables_dir / "client_decisions_and_signals.csv", index=False)
    signals_table.to_csv(tables_dir / "signal_summary_by_round.csv", index=False)

    if not args.validation_only:
        metric_tables, class_tables, pair_tables = [], [], []
        for name, path in (
            ("best_validation", checkpoints_dir / "best_validation_model.pt"),
            ("last_round", checkpoints_dir / "last_round_model.pt"),
        ):
            metrics, per_class, pair = evaluate_checkpoint(
                name, path, arrays, source_id, target_id,
                args.evaluation_batch_size, tables_dir, args.save_predictions
            )
            metric_tables.append(metrics)
            class_tables.append(per_class)
            pair_tables.append(pair)
        pd.concat(metric_tables, ignore_index=True).to_csv(tables_dir / "test_metrics.csv", index=False)
        pd.concat(class_tables, ignore_index=True).to_csv(tables_dir / "per_class_metrics.csv", index=False)
        pd.concat(pair_tables, ignore_index=True).to_csv(tables_dir / "source_target_metrics.csv", index=False)

    plot_round_curves(
        round_table, args.source_class, args.target_class,
        figures_dir / "validation_and_attack_dynamics"
    )
    plot_security(round_table, figures_dir / "client_filtering_metrics")

    best_row = round_table[round_table["round"] == best_round].iloc[0]
    metadata = {
        "experiment_version": "3.5-stage1",
        "experiment_type": "proposed_temporal_probe_explanation_defense",
        "status": "development_screen_not_final_paper_result",
        "mode": args.mode,
        "model": "ResidualMLP",
        "source_class": args.source_class,
        "target_class": args.target_class,
        "malicious_clients": malicious_clients,
        "partition_hash_sha256": partition_hash,
        "clean_seed_partition_hash_sha256": clean_hash,
        "poison_index_hash_sha256": poison_hash,
        "model_seed": args.model_seed,
        "attack_seed": args.attack_seed,
        "probe_seed": args.probe_seed,
        "prediction_probe_rows": int(len(prediction_probe_indices)),
        "explanation_probe_rows": int(len(explanation_probe_indices)),
        "rounds_requested": args.rounds,
        "rounds_completed": int(round_table["round"].max()),
        "best_validation_round": int(best_round),
        "best_validation_macro_f1": float(best_macro_f1),
        "best_validation_source_to_target_rate": float(best_row["val_source_to_target_rate"]),
        "best_round_malicious_rejection_recall": float(best_row["malicious_rejection_recall"]),
        "best_round_benign_retention_rate": float(best_row["benign_retention_rate"]),
        "test_sets_used_for_checkpoint_selection": False,
        "test_evaluation_performed": not args.validation_only,
        "total_seconds": float(time.time() - started),
        "parameters": {
            "ema_decay": args.ema_decay,
            "warmup_rounds": args.warmup_rounds,
            "quarantine_threshold": args.quarantine_threshold,
            "strike_threshold": args.strike_threshold,
            "required_strikes": args.required_strikes,
            "minimum_admitted_fraction": args.minimum_admitted_fraction,
            "soft_weight_temperature": args.soft_weight_temperature,
            "minimum_risk_weight": args.minimum_risk_weight,
        },
    }
    with (output_dir / "proposed_defense_v35_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Proposed Temporal Defense V3.5 Stage 1 complete")
    print("Mode:", args.mode)
    print("Best validation round:", best_round)
    print("Best validation macro F1:", f"{best_macro_f1:.6f}")
    print("Best validation source-to-target rate:", f"{float(best_row['val_source_to_target_rate']):.6f}")
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

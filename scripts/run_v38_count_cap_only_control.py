#!/usr/bin/env python3
"""Run integrated generic dual-reference temporal soft trust, V3.8.

V3.8 integrates the V3.7.1 frozen signal and V3.7.2 frozen trust policy into
federated aggregation. It is a paired development experiment, not a final paper
result. Test sets are not accessed.
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
from typing import Dict, List, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from federated_iot_v26 import (
    CLASS_NAMES,
    NUM_CLASSES,
    load_protocol_arrays,
    partition_manifest,
    set_seed,
    sqrt_class_weights,
    train_local_model,
)
from neural_models_v24 import build_model
from run_targeted_label_flip_v292 import (
    load_fixed_partitions,
    prepare_static_attack,
    read_clean_partition_hash,
    save_poisoned_indices,
)
from transition_signature_features_v38 import (
    balanced_probe_indices,
    class_conditional_probability_means,
    evaluate_validation,
    matrix_long_rows,
    normalize_rows,
    predict_probabilities,
    summary_row,
)
from transition_soft_trust_v38 import (
    FROZEN_CANDIDATE,
    TemporalTrustMemory,
    load_calibration,
    score_and_aggregate,
)

DEFAULT_MALICIOUS_CLIENTS = "1,7,8,10,14,15,17,18"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V3.8 integrated transition-signature temporal soft trust."
    )
    parser.add_argument("--mode", choices=["clean", "strong_attack"], required=True)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-seed-dir", required=True, type=Path)
    parser.add_argument("--signal-analysis-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-class", default="DDoS")
    parser.add_argument("--target-class", default="Benign")
    parser.add_argument("--malicious-clients", default=DEFAULT_MALICIOUS_CLIENTS)
    parser.add_argument("--poison-fraction", type=float, default=1.0)
    parser.add_argument("--min-source-samples", type=int, default=1000)
    parser.add_argument("--attack-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--calibration-seed", type=int)
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
    parser.add_argument("--ema-decay", type=float, default=0.65)
    parser.add_argument("--clean-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--trust-gamma", type=float, default=2.0)
    parser.add_argument("--minimum-trust", type=float, default=0.10)
    parser.add_argument("--count-cap-multiplier", type=float, default=3.0)
    parser.add_argument("--low-trust-cutoff", type=float, default=0.50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_client_ids(text: str) -> List[int]:
    return sorted({int(value.strip()) for value in text.split(",") if value.strip()})


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def read_table(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def save_partial(
    tables_dir: Path,
    round_rows: List[Dict[str, object]],
    local_rows: List[Dict[str, object]],
    signature_rows: List[Dict[str, object]],
    feature_rows: List[Dict[str, object]],
    trust_rows: List[Dict[str, object]],
    consensus_rows: List[Dict[str, object]],
) -> None:
    pd.DataFrame(round_rows).to_csv(tables_dir / "round_metrics_partial.csv", index=False)
    pd.DataFrame(local_rows).to_csv(
        tables_dir / "local_client_training_metrics_partial.csv", index=False
    )
    pd.DataFrame(signature_rows).to_csv(
        tables_dir / "transition_signature_long_partial.csv", index=False
    )
    pd.DataFrame(feature_rows).to_csv(
        tables_dir / "client_signature_features_partial.csv", index=False
    )
    pd.DataFrame(trust_rows).to_csv(
        tables_dir / "client_trust_and_signals_partial.csv", index=False
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
    ax.set_title("V3.8 validation and targeted-attack dynamics")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out)


def plot_trust(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(table["round"], table["mean_benign_trust"], marker="o", label="Benign trust")
    ax.plot(table["round"], table["mean_malicious_trust"], marker="s", label="Malicious trust")
    ax.plot(
        table["round"],
        table["malicious_influence_reduction"],
        marker="^",
        label="Malicious influence reduction",
    )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("V3.8 trust and aggregation influence")
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
        raise ValueError("V3.8 is frozen to full client participation")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")
    if args.trust_gamma != 0.0 or args.minimum_trust != 0.10:
        raise ValueError("Count-cap-only control requires gamma=0.0 and minimum trust=0.10")
    if args.ema_decay != 0.65 or args.clean_threshold_quantile != 0.95:
        raise ValueError("V3.8 frozen calibration requires EMA=0.65 and clean quantile=0.95")

    source_id = CLASS_NAMES.index(args.source_class)
    target_id = CLASS_NAMES.index(args.target_class)
    calibration_seed = args.model_seed if args.calibration_seed is None else args.calibration_seed
    torch.set_num_threads(max(1, args.threads))
    set_seed(args.model_seed)

    data_file = args.data_file.expanduser().resolve()
    partition_file = args.partition_file.expanduser().resolve()
    clean_seed_dir = args.clean_seed_dir.expanduser().resolve()
    signal_analysis_dir = args.signal_analysis_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        if args.overwrite:
            import shutil
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --resume or --overwrite."
            )

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"
    attack_dir = output_dir / "attack_manifest"
    calibration_dir = output_dir / "calibration"
    for path in (tables_dir, figures_dir, checkpoints_dir, attack_dir, calibration_dir):
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

    calibration = load_calibration(
        analysis_dir=signal_analysis_dir,
        calibration_seed=calibration_seed,
        expected_clients=args.num_clients,
        candidate=FROZEN_CANDIDATE,
        clean_threshold_quantile=args.clean_threshold_quantile,
    )
    calibration.summary.to_csv(
        calibration_dir / "v38_calibration_summary.csv", index=False
    )

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
            client_id: np.empty(0, dtype=np.int64)
            for client_id in range(args.num_clients)
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

    probe_indices = balanced_probe_indices(y_val, args.probe_per_class, args.probe_seed)
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
    memory = TemporalTrustMemory()

    round_rows: List[Dict[str, object]] = []
    local_rows: List[Dict[str, object]] = []
    signature_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    trust_rows: List[Dict[str, object]] = []
    consensus_rows: List[Dict[str, object]] = []
    best_macro_f1 = -math.inf
    best_round = 0
    start_round = 1

    last_checkpoint_path = checkpoints_dir / "last_round_model.pt"
    if args.resume:
        if not last_checkpoint_path.exists():
            raise FileNotFoundError("Resume requested but last-round checkpoint is missing")
        checkpoint = torch.load(last_checkpoint_path, map_location="cpu", weights_only=False)
        checks = {
            "partition_hash": partition_hash,
            "poison_index_hash": poison_hash,
            "model_seed": int(args.model_seed),
            "calibration_seed": int(calibration_seed),
            "profile_sha256": calibration.profile_sha256,
            "clean_scores_sha256": calibration.clean_scores_sha256,
        }
        for key, expected in checks.items():
            if checkpoint.get(key) != expected:
                raise RuntimeError(f"Resume checkpoint mismatch for {key}")
        model.load_state_dict(checkpoint["model_state_dict"])
        memory.candidate_ema = {
            int(key): float(value)
            for key, value in checkpoint.get("candidate_ema", {}).items()
        }
        start_round = int(checkpoint["round"]) + 1
        best_macro_f1 = float(checkpoint.get("best_macro_f1", -math.inf))
        best_round = int(checkpoint.get("best_round", 0))
        round_rows = read_table(tables_dir / "round_metrics_partial.csv")
        local_rows = read_table(tables_dir / "local_client_training_metrics_partial.csv")
        signature_rows = read_table(tables_dir / "transition_signature_long_partial.csv")
        feature_rows = read_table(tables_dir / "client_signature_features_partial.csv")
        trust_rows = read_table(tables_dir / "client_trust_and_signals_partial.csv")
        consensus_rows = read_table(tables_dir / "round_consensus_signature_long_partial.csv")
        print(f"Resuming from round {start_round}")

    started = time.time()
    print("V3.8 Count-Cap-Only Control")
    print("Mode:", args.mode)
    print("Model seed:", args.model_seed)
    print("Calibration seed:", calibration_seed)
    print("Partition hash:", partition_hash)
    print("Malicious clients:", malicious_clients)
    print("Instant threshold:", f"{calibration.instant_threshold:.6f}")
    print("EMA threshold:", f"{calibration.ema_threshold:.6f}")

    for round_id in range(start_round, args.rounds + 1):
        round_started = time.time()
        reference_state = copy.deepcopy(model.state_dict())
        reference_probabilities = predict_probabilities(
            model, X_probe, args.evaluation_batch_size
        )
        reference_means = class_conditional_probability_means(reference_probabilities, y_probe)

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
            probabilities = predict_probabilities(local_model, X_probe, args.evaluation_batch_size)
            local_means = class_conditional_probability_means(probabilities, y_probe)
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
        consensus_means = normalize_rows(np.median(matrix_stack, axis=0))
        for source_index in range(NUM_CLASSES):
            for target_index in range(NUM_CLASSES):
                consensus_rows.append(
                    {
                        "round": int(round_id),
                        "source_id": int(source_index),
                        "source_name": CLASS_NAMES[source_index],
                        "target_id": int(target_index),
                        "target_name": CLASS_NAMES[target_index],
                        "round_consensus_probability": float(consensus_means[source_index, target_index]),
                        "global_reference_probability": float(reference_means[source_index, target_index]),
                        "consensus_minus_global_reference": float(
                            consensus_means[source_index, target_index]
                            - reference_means[source_index, target_index]
                        ),
                    }
                )

        current_features = []
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
            row = summary_row(
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
            feature_rows.append(row)
            current_features.append(row)

        trust_result = score_and_aggregate(
            reference_state=reference_state,
            local_states=local_states,
            client_rows=pd.DataFrame(current_features),
            calibration=calibration,
            memory=memory,
            malicious_client_ids=malicious_clients,
            ema_decay=args.ema_decay,
            gamma=args.trust_gamma,
            minimum_trust=args.minimum_trust,
            count_cap_multiplier=args.count_cap_multiplier,
            low_trust_cutoff=args.low_trust_cutoff,
        )
        model.load_state_dict(trust_result.aggregated_state)
        trust_rows.extend(trust_result.client_table.to_dict(orient="records"))

        val_metrics, val_pair = evaluate_validation(
            model=model,
            X=X_val,
            y=y_val,
            source_id=source_id,
            target_id=target_id,
            batch_size=args.evaluation_batch_size,
        )
        round_summary = trust_result.summary
        row = {
            "round": int(round_id),
            "selected_clients": "|".join(map(str, range(args.num_clients))),
            "selected_client_count": int(args.num_clients),
            "participating_samples": int(sum(sample_counts)),
            "mean_local_train_loss": float(np.mean([r["local_train_loss"] for r in current_training_rows])),
            "mean_local_train_accuracy": float(np.mean([r["local_train_accuracy"] for r in current_training_rows])),
            "round_seconds": float(time.time() - round_started),
            **round_summary,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"val_{key}": value for key, value in val_pair.items()},
        }
        round_rows.append(row)

        checkpoint_payload = {
            "experiment_version": "3.8-control-cap-only",
            "round": int(round_id),
            "model_state_dict": model.state_dict(),
            "candidate_ema": memory.candidate_ema,
            "input_dim": int(X_train.shape[1]),
            "num_classes": int(NUM_CLASSES),
            "validation_metrics": val_metrics,
            "validation_source_target_metrics": val_pair,
            "partition_hash": partition_hash,
            "poison_index_hash": poison_hash,
            "model_seed": int(args.model_seed),
            "calibration_seed": int(calibration_seed),
            "profile_sha256": calibration.profile_sha256,
            "clean_scores_sha256": calibration.clean_scores_sha256,
            "best_macro_f1": float(max(best_macro_f1, val_metrics["macro_f1"])),
            "best_round": int(round_id if val_metrics["macro_f1"] > best_macro_f1 else best_round),
        }
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_round = int(round_id)
            best_payload = dict(checkpoint_payload)
            best_payload["checkpoint_type"] = "best_validation"
            torch.save(best_payload, checkpoints_dir / "best_validation_model.pt")
        checkpoint_payload["checkpoint_type"] = "last_round"
        checkpoint_payload["best_macro_f1"] = float(best_macro_f1)
        checkpoint_payload["best_round"] = int(best_round)
        torch.save(checkpoint_payload, last_checkpoint_path)

        save_partial(
            tables_dir=tables_dir,
            round_rows=round_rows,
            local_rows=local_rows,
            signature_rows=signature_rows,
            feature_rows=feature_rows,
            trust_rows=trust_rows,
            consensus_rows=consensus_rows,
        )
        print(
            f"round {round_id:02d}, val macro F1={val_metrics['macro_f1']:.4f}, "
            f"{args.source_class}->{args.target_class}={val_pair['source_to_target_rate']:.4%}, "
            f"malicious influence reduction={round_summary['malicious_influence_reduction']:.3f}, "
            f"benign trust={round_summary['mean_benign_trust']:.3f}, "
            f"malicious trust={round_summary['mean_malicious_trust']:.3f}, "
            f"seconds={row['round_seconds']:.1f}"
        )

    round_table = pd.DataFrame(round_rows)
    local_table = pd.DataFrame(local_rows)
    signature_table = pd.DataFrame(signature_rows)
    feature_table = pd.DataFrame(feature_rows)
    trust_table = pd.DataFrame(trust_rows)
    consensus_table = pd.DataFrame(consensus_rows)

    round_table.to_csv(tables_dir / "round_metrics.csv", index=False)
    local_table.to_csv(tables_dir / "local_client_training_metrics.csv", index=False)
    signature_table.to_csv(tables_dir / "transition_signature_long.csv", index=False)
    feature_table.to_csv(tables_dir / "client_signature_features.csv", index=False)
    trust_table.to_csv(tables_dir / "client_trust_and_signals.csv", index=False)
    consensus_table.to_csv(tables_dir / "round_consensus_signature_long.csv", index=False)

    plot_validation(round_table, args.source_class, args.target_class, figures_dir / "validation_and_attack_dynamics")
    plot_trust(round_table, figures_dir / "trust_and_influence_dynamics")

    if round_table.empty:
        raise RuntimeError("No completed rounds were found")
    best_row = round_table.loc[round_table["round"].eq(best_round)].iloc[0]
    metadata = {
        "experiment_version": "3.8-control-cap-only",
        "experiment_type": "causal_control_count_cap_only_trust_disabled",
        "status": "causal_control_not_final_paper_result",
        "aggregation_rule": "sample_count_cap_only_all_trust_fixed_to_one",
        "candidate": FROZEN_CANDIDATE,
        "policy_mode": "disabled_for_count_cap_only_control",
        "trust_gamma": float(args.trust_gamma),
        "minimum_trust": float(args.minimum_trust),
        "ema_decay": float(args.ema_decay),
        "clean_threshold_quantile": float(args.clean_threshold_quantile),
        "count_cap_multiplier": float(args.count_cap_multiplier),
        "trusted_clean_history_required": True,
        "trust_disabled_for_control": True,
        "control_purpose": "isolate_sample_count_cap_effect",
        "mode": args.mode,
        "model_seed": int(args.model_seed),
        "calibration_seed": int(calibration_seed),
        "attack_seed": int(args.attack_seed),
        "probe_seed": int(args.probe_seed),
        "probe_rows_per_class": int(args.probe_per_class),
        "source_class": args.source_class,
        "target_class": args.target_class,
        "malicious_clients": malicious_clients,
        "partition_hash_sha256": partition_hash,
        "clean_seed_partition_hash_sha256": clean_hash,
        "poison_index_hash_sha256": poison_hash,
        "profile_sha256": calibration.profile_sha256,
        "clean_scores_sha256": calibration.clean_scores_sha256,
        "instant_threshold": float(calibration.instant_threshold),
        "ema_threshold": float(calibration.ema_threshold),
        "rounds_requested": int(args.rounds),
        "rounds_completed": int(round_table["round"].max()),
        "best_validation_round": int(best_round),
        "best_validation_macro_f1": float(best_macro_f1),
        "best_validation_source_to_target_rate": float(best_row["val_source_to_target_rate"]),
        "test_sets_accessed": False,
        "malicious_labels_used_for_aggregation": False,
        "malicious_labels_used_for_reporting_only": True,
        "total_seconds": float(time.time() - started),
    }
    with (output_dir / "v38_count_cap_only_control_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("V3.8 Count-Cap-Only Control complete")
    print("Mode:", args.mode)
    print("Best validation round:", best_round)
    print("Best validation macro F1:", f"{best_macro_f1:.6f}")
    print("Best validation source-to-target rate:", f"{float(best_row['val_source_to_target_rate']):.6f}")
    print("Best-round malicious influence reduction:", f"{float(best_row['malicious_influence_reduction']):.6f}")
    print("Best-round mean benign trust:", f"{float(best_row['mean_benign_trust']):.6f}")
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

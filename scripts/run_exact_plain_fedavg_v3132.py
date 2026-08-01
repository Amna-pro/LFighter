#!/usr/bin/env python3
"""Exact V3.10 plain FedAvg adapter for V3.13.2 breadth experiments."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
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
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from federated_iot_v26 import (
    CLASS_NAMES,
    NUM_CLASSES,
    load_protocol_arrays,
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
    predict_probabilities,
)
from independent_anchor_v310 import (
    CANDIDATE,
    SCORE_COLUMN,
    add_candidate_and_ema,
    apply_feature_calibration,
    file_sha256,
    load_profiles_npz,
    normalize_rows,
    raw_features,
    weighted_average_states,
)

DEFAULT_MALICIOUS_CLIENTS = "1,7,8,10,14,15,17,18"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.10 post-warmup plain FedAvg capture.")
    parser.add_argument("--mode", choices=["clean", "strong_attack"], required=True)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-seed-dir", required=True, type=Path)
    parser.add_argument("--warmup-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--continuation-rounds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--probe-per-class", type=int, default=48)
    parser.add_argument("--probe-seed", type=int, default=3701)
    parser.add_argument("--source-class", default="DDoS")
    parser.add_argument("--target-class", default="Benign")
    parser.add_argument("--ema-decay", type=float, default=0.65)
    parser.add_argument("--malicious-clients", default=DEFAULT_MALICIOUS_CLIENTS)
    parser.add_argument("--poison-fraction", type=float, default=1.0)
    parser.add_argument("--min-source-samples", type=int, default=1000)
    parser.add_argument("--attack-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_client_ids(text: str) -> List[int]:
    return sorted({int(value.strip()) for value in text.split(",") if value.strip()})


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def save_partial(
    tables_dir: Path,
    round_rows: List[Dict[str, object]],
    local_rows: List[Dict[str, object]],
    score_rows: List[Dict[str, object]],
    signature_rows: List[Dict[str, object]],
) -> None:
    pd.DataFrame(round_rows).to_csv(tables_dir / "continuation_round_metrics_partial.csv", index=False)
    pd.DataFrame(local_rows).to_csv(tables_dir / "continuation_local_training_partial.csv", index=False)
    pd.DataFrame(score_rows).to_csv(tables_dir / "continuation_client_anchor_scores_partial.csv", index=False)
    pd.DataFrame(signature_rows).to_csv(tables_dir / "continuation_transition_signature_long_partial.csv", index=False)


def main() -> int:
    args = parse_args()
    if args.continuation_rounds != 4:
        raise ValueError("V3.10 development chronology is frozen to four continuation rounds")
    if args.ema_decay != 0.65:
        raise ValueError("V3.10 is frozen to EMA=0.65")
    if args.source_class not in CLASS_NAMES or args.target_class not in CLASS_NAMES:
        raise ValueError("Unknown source or target class")
    if args.source_class == args.target_class:
        raise ValueError("Source and target classes must differ")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")

    torch.set_num_threads(max(1, args.threads))
    set_seed(args.model_seed)
    source_id = CLASS_NAMES.index(args.source_class)
    target_id = CLASS_NAMES.index(args.target_class)

    warmup_dir = args.warmup_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"
    attack_dir = output_dir / "attack_manifest"
    for path in (tables_dir, figures_dir, checkpoints_dir, attack_dir):
        path.mkdir(parents=True, exist_ok=True)

    warmup_metadata_path = warmup_dir / "true_warmup_v310_metadata.json"
    branch_checkpoint_path = warmup_dir / "checkpoints" / "common_round4_warmup_model.pt"
    profiles_path = warmup_dir / "calibration" / "trusted_client_profiles.npz"
    feature_path = warmup_dir / "calibration" / "feature_calibration.csv"
    summary_path = warmup_dir / "calibration" / "calibration_summary.csv"
    for path in (warmup_metadata_path, branch_checkpoint_path, profiles_path, feature_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"Warmup artifact is missing: {path}")

    with warmup_metadata_path.open("r", encoding="utf-8") as handle:
        warmup_metadata = json.load(handle)
    calibration_summary = pd.read_csv(summary_path).iloc[0]
    if int(warmup_metadata["warmup_rounds"]) != 4:
        raise RuntimeError("Warmup metadata is not a four-round V3.10 branch point")
    if int(warmup_metadata["model_seed"]) != int(args.model_seed):
        raise RuntimeError("Model seed does not match warmup branch point")
    if not bool(warmup_metadata.get("monitoring_ema_reset_after_warmup", False)):
        raise RuntimeError("Warmup metadata does not freeze the monitoring EMA reset")

    arrays = load_protocol_arrays(args.data_file.expanduser().resolve())
    X_train = arrays["X_train"].astype(np.float32, copy=False)
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    X_val = arrays["X_val"].astype(np.float32, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)

    client_indices, partition_hash = load_fixed_partitions(
        args.partition_file.expanduser().resolve(),
        expected_clients=args.num_clients,
        train_rows=len(y_train),
    )
    clean_hash = read_clean_partition_hash(args.clean_seed_dir.expanduser().resolve())
    if partition_hash != clean_hash or partition_hash != warmup_metadata["partition_hash_sha256"]:
        raise RuntimeError("Partition hash does not match warmup branch point")

    probe_indices = balanced_probe_indices(y_val, args.probe_per_class, args.probe_seed)
    probe_hash = hashlib.sha256(np.ascontiguousarray(probe_indices).tobytes()).hexdigest()
    if probe_hash != warmup_metadata["probe_hash_sha256"]:
        raise RuntimeError("Probe hash does not match warmup branch point")
    X_probe = X_val[probe_indices]
    y_probe = y_val[probe_indices]

    if file_sha256(profiles_path) != warmup_metadata["profile_sha256"]:
        raise RuntimeError("Trusted profile hash mismatch")
    if file_sha256(feature_path) != warmup_metadata["feature_calibration_sha256"]:
        raise RuntimeError("Feature calibration hash mismatch")
    profiles = load_profiles_npz(profiles_path, args.num_clients)
    feature_calibration = pd.read_csv(feature_path)
    threshold = float(calibration_summary["clean_ema_threshold"])

    checkpoint = torch.load(branch_checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("partition_hash") != partition_hash:
        raise RuntimeError("Warmup checkpoint partition mismatch")
    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.mode == "strong_attack":
        requested = parse_client_ids(args.malicious_clients)
        malicious_clients, poisoned_positions, poison_manifest, poison_hash = prepare_static_attack(
            client_indices=client_indices,
            y_train=y_train,
            source_id=source_id,
            target_id=target_id,
            malicious_client_fraction=len(requested) / args.num_clients,
            poison_fraction=args.poison_fraction,
            min_source_samples=args.min_source_samples,
            attack_seed=args.attack_seed,
            explicit_malicious_clients=requested,
        )
    else:
        malicious_clients = []
        poisoned_positions = {client_id: np.empty(0, dtype=np.int64) for client_id in range(args.num_clients)}
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
                "partition_hash_sha256": partition_hash,
                "poison_index_hash_sha256": poison_hash,
            }
        ]
    ).to_csv(attack_dir / "attack_summary.csv", index=False)

    class_weights = sqrt_class_weights(y_train, args.max_class_weight)
    round_rows: List[Dict[str, object]] = []
    local_rows: List[Dict[str, object]] = []
    score_rows: List[Dict[str, object]] = []
    signature_rows: List[Dict[str, object]] = []
    ema_memory: Dict[int, float] = {}
    start_round = 1
    continuation_checkpoint = checkpoints_dir / "continuation_last_round_model.pt"

    if args.resume:
        if not continuation_checkpoint.exists():
            raise FileNotFoundError("Resume requested but continuation checkpoint is missing")
        saved = torch.load(continuation_checkpoint, map_location="cpu", weights_only=False)
        for key, expected in {
            "mode": args.mode,
            "partition_hash": partition_hash,
            "poison_index_hash": poison_hash,
            "warmup_profile_hash": warmup_metadata["profile_sha256"],
            "model_seed": int(args.model_seed),
        }.items():
            if saved.get(key) != expected:
                raise RuntimeError(f"Continuation resume mismatch for {key}")
        model.load_state_dict(saved["model_state_dict"])
        ema_memory = {int(k): float(v) for k, v in saved.get("ema_memory", {}).items()}
        start_round = int(saved["monitoring_round"]) + 1
        round_rows = load_rows(tables_dir / "continuation_round_metrics_partial.csv")
        local_rows = load_rows(tables_dir / "continuation_local_training_partial.csv")
        score_rows = load_rows(tables_dir / "continuation_client_anchor_scores_partial.csv")
        signature_rows = load_rows(tables_dir / "continuation_transition_signature_long_partial.csv")
        print(f"Resuming {args.mode} continuation from monitoring round {start_round}")

    started = time.time()
    print("Post-Warmup Anchor Capture V3.10")
    print("Mode:", args.mode)
    print("Common warmup checkpoint:", branch_checkpoint_path)
    print("Frozen EMA threshold:", f"{threshold:.6f}")
    print("EMA reset at monitoring start: True")
    print("Malicious clients:", malicious_clients)

    for monitoring_round in range(start_round, args.continuation_rounds + 1):
        global_round = 4 + monitoring_round
        round_started = time.time()
        reference_state = copy.deepcopy(model.state_dict())
        local_states: List[Mapping[str, torch.Tensor]] = []
        sample_counts: List[int] = []
        local_matrices: List[np.ndarray] = []
        current_training = []

        for client_id in range(args.num_clients):
            indices = client_indices[client_id]
            local_y = y_train[indices].copy()
            positions = poisoned_positions[client_id]
            if len(positions) > 0:
                local_y[positions] = target_id
            local_model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
            local_model.load_state_dict(reference_state)
            state, metrics = train_local_model(
                model=local_model,
                X=X_train[indices],
                y=local_y,
                class_weights=class_weights,
                local_epochs=1,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                seed=args.model_seed + global_round * 1000 + client_id,
            )
            probabilities = predict_probabilities(local_model, X_probe, args.evaluation_batch_size)
            local_means = class_conditional_probability_means(probabilities, y_probe)
            local_states.append(state)
            sample_counts.append(int(len(indices)))
            local_matrices.append(local_means)
            row = {
                "monitoring_round": int(monitoring_round),
                "global_round": int(global_round),
                "client_id": int(client_id),
                "actual_malicious": bool(client_id in malicious_clients),
                "client_samples": int(len(indices)),
                "poisoned_rows": int(len(positions)),
                **metrics,
            }
            local_rows.append(row)
            current_training.append(row)
            del local_model

        matrix_stack = np.stack(local_matrices, axis=0)
        consensus = normalize_rows(np.median(matrix_stack, axis=0))
        current_raw = []
        raw_weights = np.asarray(sample_counts, dtype=np.float64) / float(np.sum(sample_counts))
        for client_id, local_matrix in enumerate(local_matrices):
            local_matrix = normalize_rows(local_matrix)
            for source_index in range(NUM_CLASSES):
                for target_index in range(NUM_CLASSES):
                    signature_rows.append({
                        "monitoring_round": int(monitoring_round),
                        "global_round": int(global_round),
                        "client_id": int(client_id),
                        "actual_malicious": bool(client_id in malicious_clients),
                        "source_id": int(source_index),
                        "source_name": CLASS_NAMES[source_index],
                        "target_id": int(target_index),
                        "target_name": CLASS_NAMES[target_index],
                        "local_probability": float(local_matrix[source_index, target_index]),
                        "round_consensus_probability": float(consensus[source_index, target_index]),
                        "frozen_profile_probability": float(profiles[client_id][source_index, target_index]),
                    })
            current_raw.append(
                {
                    "monitoring_round": int(monitoring_round),
                    "global_round": int(global_round),
                    "client_id": int(client_id),
                    "actual_malicious": bool(client_id in malicious_clients),
                    "client_samples": int(sample_counts[client_id]),
                    "aggregation_weight": float(raw_weights[client_id]),
                    "poisoned_rows": int(len(poisoned_positions[client_id])),
                    **raw_features(
                        local_matrix,
                        profiles[client_id],
                        consensus,
                        source_id,
                        target_id,
                    ),
                }
            )

        current_scored = apply_feature_calibration(pd.DataFrame(current_raw), feature_calibration)
        current_scored, ema_memory = add_candidate_and_ema(
            current_scored, args.ema_decay, initial_ema=ema_memory
        )
        current_scored["clean_ema_threshold"] = threshold
        current_scored["flagged"] = current_scored[SCORE_COLUMN] > threshold
        current_scored["harm_weight"] = (
            current_scored["aggregation_weight"]
            * current_scored["source_target_growth_from_profile"]
        )
        score_rows.extend(current_scored.to_dict(orient="records"))

        model.load_state_dict(weighted_average_states(local_states, sample_counts, reference_state))
        val_metrics, val_pair = evaluate_validation(
            model, X_val, y_val, source_id, target_id, args.evaluation_batch_size
        )

        labels = current_scored["actual_malicious"].astype(bool).to_numpy()
        flags = current_scored["flagged"].astype(bool).to_numpy()
        benign = ~labels
        if labels.any():
            malicious_recall = float(np.mean(flags[labels]))
            precision = float(np.sum(flags & labels) / max(np.sum(flags), 1))
            mean_malicious_score = float(current_scored.loc[labels, SCORE_COLUMN].mean())
            malicious_harm = float(current_scored.loc[labels, "harm_weight"].sum())
        else:
            malicious_recall = float("nan")
            precision = float("nan")
            mean_malicious_score = float("nan")
            malicious_harm = 0.0
        benign_fpr = float(np.mean(flags[benign])) if benign.any() else float("nan")
        mean_benign_score = float(current_scored.loc[benign, SCORE_COLUMN].mean()) if benign.any() else float("nan")
        total_harm = float(current_scored["harm_weight"].sum())
        malicious_harm_share = malicious_harm / max(total_harm, 1e-12) if labels.any() else float("nan")

        row = {
            "monitoring_round": int(monitoring_round),
            "global_round": int(global_round),
            "mode": args.mode,
            "participating_samples": int(sum(sample_counts)),
            "mean_local_train_loss": float(np.mean([x["local_train_loss"] for x in current_training])),
            "mean_local_train_accuracy": float(np.mean([x["local_train_accuracy"] for x in current_training])),
            "flagged_clients": int(flags.sum()),
            "malicious_recall": malicious_recall,
            "benign_false_positive_rate": benign_fpr,
            "detection_precision": precision,
            "mean_benign_anchor_ema": mean_benign_score,
            "mean_malicious_anchor_ema": mean_malicious_score,
            "total_harm_weight": total_harm,
            "malicious_harm_share": malicious_harm_share,
            "round_seconds": float(time.time() - round_started),
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"val_{key}": value for key, value in val_pair.items()},
        }
        round_rows.append(row)

        torch.save(
            {
                "experiment_version": "3.10",
                "phase": "post_warmup_plain_fedavg_capture",
                "mode": args.mode,
                "monitoring_round": int(monitoring_round),
                "global_round": int(global_round),
                "model_state_dict": model.state_dict(),
                "ema_memory": ema_memory,
                "partition_hash": partition_hash,
                "poison_index_hash": poison_hash,
                "warmup_profile_hash": warmup_metadata["profile_sha256"],
                "model_seed": int(args.model_seed),
            },
            continuation_checkpoint,
        )
        save_partial(tables_dir, round_rows, local_rows, score_rows, signature_rows)
        recall_text = "NA" if not labels.any() else f"{malicious_recall:.3f}"
        print(
            f"monitoring round {monitoring_round:02d}, global round {global_round:02d}, "
            f"val macro F1={val_metrics['macro_f1']:.4f}, "
            f"{args.source_class}->{args.target_class}={val_pair['source_to_target_rate']:.4%}, "
            f"recall={recall_text}, benign FPR={benign_fpr:.3f}, "
            f"seconds={row['round_seconds']:.1f}"
        )

    round_table = pd.DataFrame(round_rows)
    local_table = pd.DataFrame(local_rows)
    score_table = pd.DataFrame(score_rows)
    signature_table = pd.DataFrame(signature_rows)
    round_table.to_csv(tables_dir / "continuation_round_metrics.csv", index=False)
    local_table.to_csv(tables_dir / "continuation_local_training.csv", index=False)
    score_table.to_csv(tables_dir / "continuation_client_anchor_scores.csv", index=False)
    signature_table.to_csv(tables_dir / "continuation_transition_signature_long.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(round_table["monitoring_round"], round_table["val_macro_f1"], marker="o", label="Macro F1")
    ax.plot(round_table["monitoring_round"], round_table["val_source_to_target_rate"], marker="s", label=f"{args.source_class}->{args.target_class}")
    ax.set_xlabel("Post-warmup monitoring round")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title(f"V3.10 {args.mode} continuation from common warmup checkpoint")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "validation_and_attack_dynamics")

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(round_table["monitoring_round"], round_table["benign_false_positive_rate"], marker="o", label="Benign FPR")
    if args.mode == "strong_attack":
        ax.plot(round_table["monitoring_round"], round_table["malicious_recall"], marker="s", label="Malicious recall")
    ax.set_xlabel("Post-warmup monitoring round")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Frozen independent-anchor detection")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "anchor_detection_dynamics")

    metadata = {
        "experiment_version": "3.10",
        "phase": "post_warmup_plain_fedavg_signature_capture",
        "status": "development_chronology_experiment_not_final_paper_result",
        "mode": args.mode,
        "model_seed": int(args.model_seed),
        "common_warmup_rounds": 4,
        "continuation_rounds": int(args.continuation_rounds),
        "aggregation_rule": "raw_sample_count_fedavg",
        "count_cap_used": False,
        "defense_weights_applied": False,
        "candidate": CANDIDATE,
        "score_column": SCORE_COLUMN,
        "ema_decay": float(args.ema_decay),
        "monitoring_ema_reset_after_warmup": True,
        "clean_ema_threshold": float(threshold),
        "current_global_reference_used": False,
        "stable_client_identity_required": True,
        "malicious_clients": malicious_clients,
        "partition_hash_sha256": partition_hash,
        "poison_index_hash_sha256": poison_hash,
        "warmup_profile_sha256": warmup_metadata["profile_sha256"],
        "common_branch_checkpoint": str(branch_checkpoint_path),
        "test_sets_accessed": False,
        "malicious_labels_used_for_aggregation": False,
        "malicious_labels_used_for_reporting_only": True,
        "rounds_completed": int(round_table["monitoring_round"].max()),
        "mean_validation_macro_f1": float(round_table["val_macro_f1"].mean()),
        "mean_validation_source_to_target_rate": float(round_table["val_source_to_target_rate"].mean()),
        "mean_benign_false_positive_rate": float(round_table["benign_false_positive_rate"].mean()),
        "mean_malicious_recall": (
            None if args.mode == "clean" else float(round_table["malicious_recall"].mean())
        ),
        "total_seconds": float(time.time() - started),
    }
    with (output_dir / "post_warmup_capture_v310_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Post-Warmup Anchor Capture V3.10 complete")
    print("Mode:", args.mode)
    print("Mean validation macro F1:", f"{round_table['val_macro_f1'].mean():.6f}")
    print("Mean source-to-target rate:", f"{round_table['val_source_to_target_rate'].mean():.6f}")
    print("Mean benign FPR:", f"{round_table['benign_false_positive_rate'].mean():.6f}")
    if args.mode == "strong_attack":
        print("Mean malicious recall:", f"{round_table['malicious_recall'].mean():.6f}")
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

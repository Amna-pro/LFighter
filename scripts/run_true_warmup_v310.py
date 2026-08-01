#!/usr/bin/env python3
"""Run the four-round trusted clean warmup and freeze V3.10 calibration."""
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
    partition_manifest,
    set_seed,
    sqrt_class_weights,
    train_local_model,
)
from neural_models_v24 import build_model
from run_targeted_label_flip_v292 import load_fixed_partitions, read_clean_partition_hash
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
    fit_feature_calibration,
    file_sha256,
    normalize_rows,
    quantile_higher,
    raw_features,
    save_profiles_npz,
    weighted_average_states,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.10 four-round trusted warmup.")
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-seed-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--warmup-rounds", type=int, default=4)
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
    parser.add_argument("--clean-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_partial(
    tables_dir: Path,
    round_rows: List[Dict[str, object]],
    local_rows: List[Dict[str, object]],
    matrices_by_round: Mapping[int, np.ndarray],
) -> None:
    pd.DataFrame(round_rows).to_csv(tables_dir / "warmup_round_metrics_partial.csv", index=False)
    pd.DataFrame(local_rows).to_csv(tables_dir / "warmup_local_training_partial.csv", index=False)
    np.savez_compressed(
        tables_dir / "warmup_local_signatures_partial.npz",
        **{f"round_{int(round_id):02d}": matrix for round_id, matrix in matrices_by_round.items()},
    )


def load_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def load_matrix_rounds(path: Path) -> Dict[int, np.ndarray]:
    if not path.exists():
        return {}
    bundle = np.load(path)
    return {int(key.split("_")[-1]): bundle[key] for key in bundle.files}


def main() -> int:
    args = parse_args()
    if args.warmup_rounds != 4:
        raise ValueError("V3.10 is frozen to four trusted warmup rounds")
    if args.ema_decay != 0.65 or args.clean_threshold_quantile != 0.95:
        raise ValueError("V3.10 is frozen to EMA=0.65 and clean quantile=0.95")
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

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"
    calibration_dir = output_dir / "calibration"
    for path in (tables_dir, figures_dir, checkpoints_dir, calibration_dir):
        path.mkdir(parents=True, exist_ok=True)

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
    if clean_hash != partition_hash:
        raise RuntimeError("Clean seed partition hash does not match supplied partition")

    client_summary, client_matrix = partition_manifest(client_indices, y_train)
    client_summary.to_csv(tables_dir / "client_partition_summary.csv", index=False)
    client_matrix.to_csv(tables_dir / "client_class_counts.csv", index=False)

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
    probe_hash = hashlib.sha256(np.ascontiguousarray(probe_indices).tobytes()).hexdigest()

    class_weights = sqrt_class_weights(y_train, args.max_class_weight)
    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    round_rows: List[Dict[str, object]] = []
    local_rows: List[Dict[str, object]] = []
    matrices_by_round: Dict[int, np.ndarray] = {}
    start_round = 1
    checkpoint_path = checkpoints_dir / "warmup_last_round_model.pt"

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError("Resume requested but warmup checkpoint is missing")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for key, expected in {
            "partition_hash": partition_hash,
            "model_seed": int(args.model_seed),
            "probe_hash": probe_hash,
        }.items():
            if checkpoint.get(key) != expected:
                raise RuntimeError(f"Warmup resume mismatch for {key}")
        model.load_state_dict(checkpoint["model_state_dict"])
        start_round = int(checkpoint["round"]) + 1
        round_rows = load_rows(tables_dir / "warmup_round_metrics_partial.csv")
        local_rows = load_rows(tables_dir / "warmup_local_training_partial.csv")
        matrices_by_round = load_matrix_rounds(tables_dir / "warmup_local_signatures_partial.npz")
        print(f"Resuming trusted warmup from round {start_round}")

    started = time.time()
    print("True Warmup Anchor V3.10")
    print("Model seed:", args.model_seed)
    print("Warmup rounds:", args.warmup_rounds)
    print("Partition hash:", partition_hash)
    print("Probe hash:", probe_hash)

    for round_id in range(start_round, args.warmup_rounds + 1):
        round_started = time.time()
        reference_state = copy.deepcopy(model.state_dict())
        local_states: List[Mapping[str, torch.Tensor]] = []
        sample_counts: List[int] = []
        local_matrices: List[np.ndarray] = []
        current_training = []

        for client_id in range(args.num_clients):
            indices = client_indices[client_id]
            local_model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
            local_model.load_state_dict(reference_state)
            state, metrics = train_local_model(
                model=local_model,
                X=X_train[indices],
                y=y_train[indices],
                class_weights=class_weights,
                local_epochs=1,
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
            row = {
                "warmup_round": int(round_id),
                "client_id": int(client_id),
                "client_samples": int(len(indices)),
                **metrics,
            }
            local_rows.append(row)
            current_training.append(row)
            del local_model

        model.load_state_dict(weighted_average_states(local_states, sample_counts, reference_state))
        matrices_by_round[round_id] = np.stack(local_matrices, axis=0)
        val_metrics, val_pair = evaluate_validation(
            model, X_val, y_val, source_id, target_id, args.evaluation_batch_size
        )
        row = {
            "warmup_round": int(round_id),
            "participating_samples": int(sum(sample_counts)),
            "mean_local_train_loss": float(np.mean([x["local_train_loss"] for x in current_training])),
            "mean_local_train_accuracy": float(np.mean([x["local_train_accuracy"] for x in current_training])),
            "round_seconds": float(time.time() - round_started),
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"val_{key}": value for key, value in val_pair.items()},
        }
        round_rows.append(row)
        torch.save(
            {
                "experiment_version": "3.10",
                "phase": "trusted_clean_warmup",
                "round": int(round_id),
                "model_state_dict": model.state_dict(),
                "input_dim": int(X_train.shape[1]),
                "num_classes": int(NUM_CLASSES),
                "partition_hash": partition_hash,
                "model_seed": int(args.model_seed),
                "probe_hash": probe_hash,
            },
            checkpoint_path,
        )
        save_partial(tables_dir, round_rows, local_rows, matrices_by_round)
        print(
            f"warmup round {round_id:02d}, val macro F1={val_metrics['macro_f1']:.4f}, "
            f"{args.source_class}->{args.target_class}={val_pair['source_to_target_rate']:.4%}, "
            f"seconds={row['round_seconds']:.1f}"
        )

    if len(matrices_by_round) != args.warmup_rounds:
        raise RuntimeError("Warmup signature capture is incomplete")

    full_profiles = {
        client_id: normalize_rows(
            np.median(
                np.stack([matrices_by_round[r][client_id] for r in range(1, args.warmup_rounds + 1)]),
                axis=0,
            )
        )
        for client_id in range(args.num_clients)
    }

    calibration_rows = []
    for round_id in range(1, args.warmup_rounds + 1):
        round_stack = matrices_by_round[round_id]
        consensus = normalize_rows(np.median(round_stack, axis=0))
        other_rounds = [r for r in range(1, args.warmup_rounds + 1) if r != round_id]
        loo_profiles = {
            client_id: normalize_rows(
                np.median(np.stack([matrices_by_round[r][client_id] for r in other_rounds]), axis=0)
            )
            for client_id in range(args.num_clients)
        }
        for client_id in range(args.num_clients):
            calibration_rows.append(
                {
                    "monitoring_round": int(round_id),
                    "global_round": int(round_id),
                    "client_id": int(client_id),
                    "actual_malicious": False,
                    **raw_features(
                        round_stack[client_id],
                        loo_profiles[client_id],
                        consensus,
                        source_id,
                        target_id,
                    ),
                }
            )

    clean_raw = pd.DataFrame(calibration_rows)
    clean_scored, feature_calibration = fit_feature_calibration(clean_raw)
    clean_scored, _ = add_candidate_and_ema(clean_scored, args.ema_decay, initial_ema={})
    threshold = quantile_higher(clean_scored[SCORE_COLUMN], args.clean_threshold_quantile)
    clean_scored["flagged"] = clean_scored[SCORE_COLUMN] > threshold

    profile_long_rows = []
    for client_id, matrix in sorted(full_profiles.items()):
        for source_index in range(NUM_CLASSES):
            for target_index in range(NUM_CLASSES):
                profile_long_rows.append({
                    "client_id": int(client_id),
                    "source_id": int(source_index),
                    "source_name": CLASS_NAMES[source_index],
                    "target_id": int(target_index),
                    "target_name": CLASS_NAMES[target_index],
                    "profile_probability": float(matrix[source_index, target_index]),
                })
    pd.DataFrame(profile_long_rows).to_csv(
        calibration_dir / "trusted_client_profiles_long.csv", index=False
    )

    warmup_signature_rows = []
    for round_id, round_stack in sorted(matrices_by_round.items()):
        consensus = normalize_rows(np.median(round_stack, axis=0))
        for client_id in range(args.num_clients):
            local_matrix = normalize_rows(round_stack[client_id])
            profile_matrix = full_profiles[client_id]
            for source_index in range(NUM_CLASSES):
                for target_index in range(NUM_CLASSES):
                    warmup_signature_rows.append({
                        "warmup_round": int(round_id),
                        "client_id": int(client_id),
                        "source_id": int(source_index),
                        "source_name": CLASS_NAMES[source_index],
                        "target_id": int(target_index),
                        "target_name": CLASS_NAMES[target_index],
                        "local_probability": float(local_matrix[source_index, target_index]),
                        "round_consensus_probability": float(consensus[source_index, target_index]),
                        "frozen_profile_probability": float(profile_matrix[source_index, target_index]),
                    })
    pd.DataFrame(warmup_signature_rows).to_csv(
        tables_dir / "warmup_transition_signature_long.csv", index=False
    )

    profiles_path = calibration_dir / "trusted_client_profiles.npz"
    profile_hash = save_profiles_npz(profiles_path, full_profiles)
    feature_path = calibration_dir / "feature_calibration.csv"
    score_path = calibration_dir / "clean_leave_one_round_out_scores.csv"
    feature_calibration.to_csv(feature_path, index=False)
    clean_scored.to_csv(score_path, index=False)
    feature_hash = file_sha256(feature_path)
    score_hash = file_sha256(score_path)

    calibration_summary = pd.DataFrame(
        [
            {
                "candidate": CANDIDATE,
                "score_column": SCORE_COLUMN,
                "warmup_rounds": int(args.warmup_rounds),
                "ema_decay": float(args.ema_decay),
                "clean_threshold_quantile": float(args.clean_threshold_quantile),
                "clean_ema_threshold": float(threshold),
                "clean_false_positive_rate": float(clean_scored["flagged"].mean()),
                "monitoring_ema_reset_after_warmup": True,
                "current_global_reference_used": False,
                "count_cap_used": False,
                "test_sets_accessed": False,
                "profile_sha256": profile_hash,
                "feature_calibration_sha256": feature_hash,
                "clean_scores_sha256": score_hash,
            }
        ]
    )
    calibration_summary.to_csv(calibration_dir / "calibration_summary.csv", index=False)

    final_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    final_checkpoint.update(
        {
            "checkpoint_type": "common_round4_warmup_branch_point",
            "warmup_rounds": int(args.warmup_rounds),
            "profile_sha256": profile_hash,
            "feature_calibration_sha256": feature_hash,
            "clean_scores_sha256": score_hash,
            "clean_ema_threshold": float(threshold),
            "ema_decay": float(args.ema_decay),
            "clean_threshold_quantile": float(args.clean_threshold_quantile),
        }
    )
    branch_checkpoint_path = checkpoints_dir / "common_round4_warmup_model.pt"
    torch.save(final_checkpoint, branch_checkpoint_path)

    pd.DataFrame(round_rows).to_csv(tables_dir / "warmup_round_metrics.csv", index=False)
    pd.DataFrame(local_rows).to_csv(tables_dir / "warmup_local_training.csv", index=False)
    np.savez_compressed(
        tables_dir / "warmup_local_signatures.npz",
        **{f"round_{int(r):02d}": m for r, m in matrices_by_round.items()},
    )

    round_table = pd.DataFrame(round_rows)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(round_table["warmup_round"], round_table["val_macro_f1"], marker="o")
    ax.set_xlabel("Trusted clean warmup round")
    ax.set_ylabel("Validation macro F1")
    ax.set_title("V3.10 common trusted warmup trajectory")
    ax.grid(alpha=0.25)
    save_figure(fig, figures_dir / "warmup_validation_macro_f1")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(clean_scored[SCORE_COLUMN], bins=30)
    ax.axvline(threshold, linestyle="--", label="95th-percentile threshold")
    ax.set_xlabel("Clean leave-one-round-out EMA score")
    ax.set_ylabel("Client-round count")
    ax.set_title("V3.10 frozen clean calibration distribution")
    ax.legend()
    save_figure(fig, figures_dir / "clean_calibration_score_distribution")

    metadata = {
        "experiment_version": "3.10",
        "phase": "trusted_clean_warmup_and_calibration",
        "status": "development_chronology_experiment_not_final_paper_result",
        "model_seed": int(args.model_seed),
        "warmup_rounds": int(args.warmup_rounds),
        "partition_hash_sha256": partition_hash,
        "probe_hash_sha256": probe_hash,
        "probe_rows_per_class": int(args.probe_per_class),
        "probe_seed": int(args.probe_seed),
        "candidate": CANDIDATE,
        "score_column": SCORE_COLUMN,
        "ema_decay": float(args.ema_decay),
        "clean_threshold_quantile": float(args.clean_threshold_quantile),
        "clean_ema_threshold": float(threshold),
        "current_global_reference_used": False,
        "count_cap_used": False,
        "monitoring_ema_reset_after_warmup": True,
        "stable_client_identity_required": True,
        "test_sets_accessed": False,
        "profile_sha256": profile_hash,
        "feature_calibration_sha256": feature_hash,
        "clean_scores_sha256": score_hash,
        "common_branch_checkpoint": str(branch_checkpoint_path),
        "total_seconds": float(time.time() - started),
    }
    with (output_dir / "true_warmup_v310_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("True Warmup Anchor V3.10 complete")
    print("Common branch checkpoint:", branch_checkpoint_path)
    print("Frozen EMA threshold:", f"{threshold:.6f}")
    print("Clean calibration FPR:", f"{float(clean_scored['flagged'].mean()):.4%}")
    print("Profile hash:", profile_hash)
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

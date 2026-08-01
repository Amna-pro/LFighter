#!/usr/bin/env python3
"""Run V3.12 trusted update reconstruction from the common V3.10 checkpoint."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
    quantile_higher,
    raw_features,
)
from trusted_update_reconstruction_v312 import (
    checkpoint_sha256,
    coordinate_median,
    floating_update,
    reconstruct_update,
    relative_l2_error,
    state_from_update,
    update_cosine,
    update_norm,
    weighted_average_states,
)

DEFAULT_MALICIOUS_CLIENTS = "1,7,8,10,14,15,17,18"
FROZEN_EMA_QUANTILE = 0.99
FROZEN_INSTANT_QUANTILE = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["clean", "strong_attack"], required=True)
    parser.add_argument(
        "--replacement-policy",
        choices=["trusted_reconstruction", "oracle_clean_replacement"],
        required=True,
    )
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-seed-dir", required=True, type=Path)
    parser.add_argument("--warmup-dir", required=True, type=Path)
    parser.add_argument(
        "--reconstruction-calibration-dir",
        required=True,
        type=Path,
    )
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
    return sorted(
        {int(value.strip()) for value in text.split(",") if value.strip()}
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
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
    client_rows: List[Dict[str, object]],
    signature_rows: List[Dict[str, object]],
) -> None:
    pd.DataFrame(round_rows).to_csv(
        tables_dir / "reconstruction_round_metrics_partial.csv",
        index=False,
    )
    pd.DataFrame(local_rows).to_csv(
        tables_dir / "reconstruction_local_training_partial.csv",
        index=False,
    )
    pd.DataFrame(client_rows).to_csv(
        tables_dir / "reconstruction_client_rows_partial.csv",
        index=False,
    )
    pd.DataFrame(signature_rows).to_csv(
        tables_dir / "reconstruction_transition_signature_long_partial.csv",
        index=False,
    )


def main() -> int:
    args = parse_args()
    if args.continuation_rounds != 4:
        raise ValueError("V3.12 is frozen to four continuation rounds")
    if args.ema_decay != 0.65:
        raise ValueError("V3.12 is frozen to EMA decay 0.65")
    if (
        args.replacement_policy == "oracle_clean_replacement"
        and args.mode != "strong_attack"
    ):
        raise ValueError(
            "oracle_clean_replacement is defined only for strong_attack"
        )
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")
    if args.source_class not in CLASS_NAMES or args.target_class not in CLASS_NAMES:
        raise ValueError("Unknown source or target class")

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
    attack_dir = output_dir / "attack_manifest"
    for path in (tables_dir, figures_dir, checkpoints_dir, attack_dir):
        path.mkdir(parents=True, exist_ok=True)

    warmup_dir = args.warmup_dir.expanduser().resolve()
    warmup_metadata_path = warmup_dir / "true_warmup_v310_metadata.json"
    branch_checkpoint_path = (
        warmup_dir / "checkpoints" / "common_round4_warmup_model.pt"
    )
    profiles_path = warmup_dir / "calibration" / "trusted_client_profiles.npz"
    feature_path = warmup_dir / "calibration" / "feature_calibration.csv"
    warmup_scores_path = (
        warmup_dir / "calibration" / "clean_leave_one_round_out_scores.csv"
    )

    reconstruction_calibration_dir = (
        args.reconstruction_calibration_dir.expanduser().resolve()
    )
    reconstruction_checkpoint_path = (
        reconstruction_calibration_dir
        / "calibration"
        / "trusted_update_reconstruction_profiles.pt"
    )
    reconstruction_summary_path = (
        reconstruction_calibration_dir
        / "calibration"
        / "reconstruction_calibration_summary.csv"
    )

    required_paths = (
        warmup_metadata_path,
        branch_checkpoint_path,
        profiles_path,
        feature_path,
        warmup_scores_path,
        reconstruction_checkpoint_path,
        reconstruction_summary_path,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    with warmup_metadata_path.open("r", encoding="utf-8") as handle:
        warmup_metadata = json.load(handle)
    reconstruction_bundle = torch.load(
        reconstruction_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    reconstruction_summary = pd.read_csv(reconstruction_summary_path).iloc[0]

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
    clean_hash = read_clean_partition_hash(
        args.clean_seed_dir.expanduser().resolve()
    )
    if (
        partition_hash != clean_hash
        or partition_hash != warmup_metadata["partition_hash_sha256"]
        or partition_hash != reconstruction_bundle["partition_hash"]
    ):
        raise RuntimeError("Partition hash mismatch")
    if int(reconstruction_bundle["model_seed"]) != int(args.model_seed):
        raise RuntimeError("Reconstruction calibration model seed mismatch")
    if reconstruction_bundle["common_warmup_checkpoint_sha256"] != (
        checkpoint_sha256(branch_checkpoint_path)
    ):
        raise RuntimeError("Reconstruction calibration branch hash mismatch")

    probe_indices = balanced_probe_indices(
        y_val,
        args.probe_per_class,
        args.probe_seed,
    )
    probe_hash = hashlib.sha256(
        np.ascontiguousarray(probe_indices).tobytes()
    ).hexdigest()
    if probe_hash != warmup_metadata["probe_hash_sha256"]:
        raise RuntimeError("Probe hash mismatch")
    X_probe = X_val[probe_indices]
    y_probe = y_val[probe_indices]

    profiles = load_profiles_npz(profiles_path, args.num_clients)
    feature_calibration = pd.read_csv(feature_path)
    warmup_scores = pd.read_csv(warmup_scores_path)
    ema_threshold = quantile_higher(
        warmup_scores[SCORE_COLUMN].to_numpy(dtype=np.float64),
        FROZEN_EMA_QUANTILE,
    )
    instant_threshold = quantile_higher(
        warmup_scores[CANDIDATE].to_numpy(dtype=np.float64),
        FROZEN_INSTANT_QUANTILE,
    )

    residual_profiles = {
        int(client_id): {
            key: tensor.detach().cpu().to(torch.float32)
            for key, tensor in profile.items()
        }
        for client_id, profile in reconstruction_bundle[
            "client_residual_profiles"
        ].items()
    }
    selected_policy = str(reconstruction_bundle["selected_policy"])
    warmup_update_scale = float(reconstruction_bundle["warmup_update_scale"])
    scale_lower = float(reconstruction_bundle["scale_lower"])
    scale_upper = float(reconstruction_bundle["scale_upper"])
    norm_clip_multiplier = float(
        reconstruction_bundle["norm_clip_multiplier"]
    )

    checkpoint = torch.load(
        branch_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.mode == "strong_attack":
        requested = parse_client_ids(args.malicious_clients)
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
            malicious_client_fraction=len(requested) / args.num_clients,
            poison_fraction=args.poison_fraction,
            min_source_samples=args.min_source_samples,
            attack_seed=args.attack_seed,
            explicit_malicious_clients=requested,
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
        attack_dir / "malicious_client_poison_manifest.csv",
        index=False,
    )
    save_poisoned_indices(
        client_indices,
        poisoned_positions,
        attack_dir / "poisoned_indices.npz",
    )

    class_weights = sqrt_class_weights(y_train, args.max_class_weight)
    round_rows: List[Dict[str, object]] = []
    local_rows: List[Dict[str, object]] = []
    client_rows: List[Dict[str, object]] = []
    signature_rows: List[Dict[str, object]] = []
    ema_memory: Dict[int, float] = {}
    start_round = 1
    continuation_checkpoint = (
        checkpoints_dir / "reconstruction_last_round_model.pt"
    )

    if args.resume:
        if not continuation_checkpoint.exists():
            raise FileNotFoundError(
                "Resume requested but the reconstruction checkpoint is missing"
            )
        saved = torch.load(
            continuation_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        for key, expected in {
            "mode": args.mode,
            "replacement_policy": args.replacement_policy,
            "partition_hash": partition_hash,
            "poison_index_hash": poison_hash,
            "model_seed": int(args.model_seed),
            "reconstruction_profile_hash": file_sha256(
                reconstruction_checkpoint_path
            ),
        }.items():
            if saved.get(key) != expected:
                raise RuntimeError(f"Resume mismatch for {key}")
        model.load_state_dict(saved["model_state_dict"])
        ema_memory = {
            int(key): float(value)
            for key, value in saved.get("ema_memory", {}).items()
        }
        start_round = int(saved["monitoring_round"]) + 1
        round_rows = load_rows(
            tables_dir / "reconstruction_round_metrics_partial.csv"
        )
        local_rows = load_rows(
            tables_dir / "reconstruction_local_training_partial.csv"
        )
        client_rows = load_rows(
            tables_dir / "reconstruction_client_rows_partial.csv"
        )
        signature_rows = load_rows(
            tables_dir
            / "reconstruction_transition_signature_long_partial.csv"
        )
        print(
            f"Resuming {args.mode} {args.replacement_policy} "
            f"from monitoring round {start_round}"
        )

    print("Trusted Update Reconstruction V3.12")
    print("Mode:", args.mode)
    print("Replacement policy:", args.replacement_policy)
    print("Selected reconstruction policy:", selected_policy)
    print("Frozen EMA q99 threshold:", f"{ema_threshold:.6f}")
    print("Frozen instant q95 threshold:", f"{instant_threshold:.6f}")
    print("Malicious clients:", malicious_clients)

    started = time.time()
    for monitoring_round in range(start_round, args.continuation_rounds + 1):
        round_started = time.time()
        global_round = 4 + monitoring_round
        reference_state = copy.deepcopy(model.state_dict())
        local_states: List[Mapping[str, torch.Tensor]] = []
        oracle_clean_states: Dict[int, Mapping[str, torch.Tensor]] = {}
        sample_counts: List[int] = []
        local_matrices: List[np.ndarray] = []
        current_training = []

        for client_id in range(args.num_clients):
            indices = client_indices[client_id]
            local_y = y_train[indices].copy()
            positions = poisoned_positions[client_id]
            if len(positions):
                local_y[positions] = target_id

            local_seed = args.model_seed + global_round * 1000 + client_id
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
                seed=local_seed,
            )
            probabilities = predict_probabilities(
                local_model,
                X_probe,
                args.evaluation_batch_size,
            )
            local_means = class_conditional_probability_means(
                probabilities,
                y_probe,
            )
            local_states.append(state)
            sample_counts.append(int(len(indices)))
            local_matrices.append(local_means)
            row = {
                "monitoring_round": monitoring_round,
                "global_round": global_round,
                "client_id": client_id,
                "actual_malicious": bool(client_id in malicious_clients),
                "client_samples": int(len(indices)),
                "poisoned_rows": int(len(positions)),
                **metrics,
            }
            local_rows.append(row)
            current_training.append(row)
            del local_model

            if (
                args.mode == "strong_attack"
                and client_id in malicious_clients
            ):
                clean_model = build_model(
                    "resmlp",
                    X_train.shape[1],
                    NUM_CLASSES,
                )
                clean_model.load_state_dict(reference_state)
                clean_state, _ = train_local_model(
                    model=clean_model,
                    X=X_train[indices],
                    y=y_train[indices],
                    class_weights=class_weights,
                    local_epochs=1,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    gradient_clip_norm=args.gradient_clip_norm,
                    seed=local_seed,
                )
                oracle_clean_states[client_id] = clean_state
                del clean_model

        matrix_stack = np.stack(local_matrices, axis=0)
        consensus = normalize_rows(np.median(matrix_stack, axis=0))
        raw_weights = (
            np.asarray(sample_counts, dtype=np.float64)
            / float(np.sum(sample_counts))
        )
        current_raw = []

        for client_id, local_matrix in enumerate(local_matrices):
            normalized_matrix = normalize_rows(local_matrix)
            for source_index in range(NUM_CLASSES):
                for target_index in range(NUM_CLASSES):
                    signature_rows.append(
                        {
                            "monitoring_round": monitoring_round,
                            "global_round": global_round,
                            "client_id": client_id,
                            "actual_malicious": bool(
                                client_id in malicious_clients
                            ),
                            "source_id": source_index,
                            "source_name": CLASS_NAMES[source_index],
                            "target_id": target_index,
                            "target_name": CLASS_NAMES[target_index],
                            "local_probability": float(
                                normalized_matrix[
                                    source_index,
                                    target_index,
                                ]
                            ),
                            "round_consensus_probability": float(
                                consensus[source_index, target_index]
                            ),
                            "frozen_profile_probability": float(
                                profiles[client_id][
                                    source_index,
                                    target_index,
                                ]
                            ),
                        }
                    )
            current_raw.append(
                {
                    "monitoring_round": monitoring_round,
                    "global_round": global_round,
                    "client_id": client_id,
                    "actual_malicious": bool(
                        client_id in malicious_clients
                    ),
                    "client_samples": int(sample_counts[client_id]),
                    "aggregation_weight": float(raw_weights[client_id]),
                    "poisoned_rows": int(
                        len(poisoned_positions[client_id])
                    ),
                    **raw_features(
                        normalized_matrix,
                        profiles[client_id],
                        consensus,
                        source_id,
                        target_id,
                    ),
                }
            )

        current_scored = apply_feature_calibration(
            pd.DataFrame(current_raw),
            feature_calibration,
        )
        current_scored, ema_memory = add_candidate_and_ema(
            current_scored,
            args.ema_decay,
            initial_ema=ema_memory,
        )
        current_scored["ema_ratio"] = (
            current_scored[SCORE_COLUMN].to_numpy(dtype=np.float64)
            / max(ema_threshold, 1e-12)
        )
        current_scored["instant_ratio"] = (
            current_scored[CANDIDATE].to_numpy(dtype=np.float64)
            / max(instant_threshold, 1e-12)
        )
        current_scored["policy_ratio"] = np.maximum(
            current_scored["ema_ratio"].to_numpy(dtype=np.float64),
            current_scored["instant_ratio"].to_numpy(dtype=np.float64),
        )
        current_scored["flagged"] = (
            current_scored["policy_ratio"] > 1.0
        )

        flags = current_scored["flagged"].astype(bool).to_numpy()
        labels = current_scored["actual_malicious"].astype(bool).to_numpy()
        benign = ~labels
        updates = [
            floating_update(state, reference_state)
            for state in local_states
        ]

        replacement_states: List[Mapping[str, torch.Tensor]] = list(
            local_states
        )
        round_reconstruction_rows = []

        if args.replacement_policy == "trusted_reconstruction":
            trusted_ids = [
                client_id
                for client_id in range(args.num_clients)
                if not flags[client_id]
            ]
            if len(trusted_ids) < 8:
                raise RuntimeError(
                    f"Only {len(trusted_ids)} trusted clients remain"
                )
            trusted_updates = [updates[client_id] for client_id in trusted_ids]
            trusted_norms = [
                update_norm(update) for update in trusted_updates
            ]
            current_center = coordinate_median(trusted_updates)
            current_update_scale = float(np.median(trusted_norms))

            for client_id in range(args.num_clients):
                replaced = bool(flags[client_id])
                reconstruction_meta = {
                    "residual_scale": float("nan"),
                    "norm_clip_factor": float("nan"),
                    "norm_clip_upper": float("nan"),
                    "reconstructed_update_norm": float("nan"),
                }
                reconstruction_relative_l2 = float("nan")
                reconstruction_cosine = float("nan")

                if replaced:
                    reconstructed_update, reconstruction_meta = (
                        reconstruct_update(
                            current_center=current_center,
                            historical_residual=residual_profiles[
                                client_id
                            ],
                            selected_policy=selected_policy,
                            current_update_scale=current_update_scale,
                            warmup_update_scale=warmup_update_scale,
                            trusted_norms=trusted_norms,
                            scale_lower=scale_lower,
                            scale_upper=scale_upper,
                            norm_clip_multiplier=norm_clip_multiplier,
                        )
                    )
                    replacement_states[client_id] = state_from_update(
                        reference_state,
                        reconstructed_update,
                    )

                    if client_id in oracle_clean_states:
                        clean_update = floating_update(
                            oracle_clean_states[client_id],
                            reference_state,
                        )
                        reconstruction_relative_l2 = relative_l2_error(
                            reconstructed_update,
                            clean_update,
                        )
                        reconstruction_cosine = update_cosine(
                            reconstructed_update,
                            clean_update,
                        )
                    elif args.mode == "clean":
                        reconstruction_relative_l2 = relative_l2_error(
                            reconstructed_update,
                            updates[client_id],
                        )
                        reconstruction_cosine = update_cosine(
                            reconstructed_update,
                            updates[client_id],
                        )

                round_reconstruction_rows.append(
                    {
                        "monitoring_round": monitoring_round,
                        "global_round": global_round,
                        "client_id": client_id,
                        "actual_malicious": bool(labels[client_id]),
                        "flagged": bool(flags[client_id]),
                        "update_replaced": replaced,
                        "selected_reconstruction_policy": selected_policy,
                        "actual_update_norm": update_norm(
                            updates[client_id]
                        ),
                        "reconstruction_relative_l2_to_clean_substitute": (
                            reconstruction_relative_l2
                        ),
                        "reconstruction_cosine_to_clean_substitute": (
                            reconstruction_cosine
                        ),
                        **reconstruction_meta,
                    }
                )
        else:
            if set(oracle_clean_states) != set(malicious_clients):
                raise RuntimeError(
                    "Oracle clean substitutes are incomplete"
                )
            for client_id in malicious_clients:
                replacement_states[client_id] = oracle_clean_states[
                    client_id
                ]
            for client_id in range(args.num_clients):
                round_reconstruction_rows.append(
                    {
                        "monitoring_round": monitoring_round,
                        "global_round": global_round,
                        "client_id": client_id,
                        "actual_malicious": bool(labels[client_id]),
                        "flagged": bool(flags[client_id]),
                        "update_replaced": bool(
                            client_id in malicious_clients
                        ),
                        "selected_reconstruction_policy": (
                            "oracle_clean_replacement"
                        ),
                        "actual_update_norm": update_norm(
                            updates[client_id]
                        ),
                        "reconstruction_relative_l2_to_clean_substitute": (
                            0.0
                            if client_id in malicious_clients
                            else float("nan")
                        ),
                        "reconstruction_cosine_to_clean_substitute": (
                            1.0
                            if client_id in malicious_clients
                            else float("nan")
                        ),
                        "residual_scale": float("nan"),
                        "norm_clip_factor": float("nan"),
                        "norm_clip_upper": float("nan"),
                        "reconstructed_update_norm": (
                            update_norm(
                                floating_update(
                                    oracle_clean_states[client_id],
                                    reference_state,
                                )
                            )
                            if client_id in malicious_clients
                            else float("nan")
                        ),
                    }
                )

        client_rows.extend(round_reconstruction_rows)
        model.load_state_dict(
            weighted_average_states(
                replacement_states,
                sample_counts,
                reference_state,
            )
        )
        val_metrics, val_pair = evaluate_validation(
            model,
            X_val,
            y_val,
            source_id,
            target_id,
            args.evaluation_batch_size,
        )

        malicious_recall = (
            float(np.mean(flags[labels])) if labels.any() else float("nan")
        )
        benign_fpr = (
            float(np.mean(flags[benign])) if benign.any() else float("nan")
        )
        detection_precision = (
            float(np.sum(flags & labels) / max(np.sum(flags), 1))
            if labels.any()
            else float("nan")
        )
        malicious_replacement_recall = (
            float(
                np.mean(
                    [
                        row["update_replaced"]
                        for row in round_reconstruction_rows
                        if row["actual_malicious"]
                    ]
                )
            )
            if labels.any()
            else float("nan")
        )
        benign_replacement_rate = (
            float(
                np.mean(
                    [
                        row["update_replaced"]
                        for row in round_reconstruction_rows
                        if not row["actual_malicious"]
                    ]
                )
            )
            if benign.any()
            else float("nan")
        )

        clean_substitute_errors = [
            row["reconstruction_relative_l2_to_clean_substitute"]
            for row in round_reconstruction_rows
            if row["update_replaced"]
            and np.isfinite(
                row["reconstruction_relative_l2_to_clean_substitute"]
            )
        ]
        clean_substitute_cosines = [
            row["reconstruction_cosine_to_clean_substitute"]
            for row in round_reconstruction_rows
            if row["update_replaced"]
            and np.isfinite(
                row["reconstruction_cosine_to_clean_substitute"]
            )
        ]

        round_row = {
            "monitoring_round": monitoring_round,
            "global_round": global_round,
            "mode": args.mode,
            "replacement_policy": args.replacement_policy,
            "participating_samples": int(sum(sample_counts)),
            "flagged_clients": int(flags.sum()),
            "replaced_clients": int(
                sum(row["update_replaced"] for row in round_reconstruction_rows)
            ),
            "malicious_recall": malicious_recall,
            "benign_false_positive_rate": benign_fpr,
            "detection_precision": detection_precision,
            "malicious_replacement_recall": malicious_replacement_recall,
            "benign_replacement_rate": benign_replacement_rate,
            "mean_reconstruction_relative_l2_to_clean_substitute": (
                float(np.mean(clean_substitute_errors))
                if clean_substitute_errors
                else float("nan")
            ),
            "mean_reconstruction_cosine_to_clean_substitute": (
                float(np.mean(clean_substitute_cosines))
                if clean_substitute_cosines
                else float("nan")
            ),
            "round_seconds": float(time.time() - round_started),
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"val_{key}": value for key, value in val_pair.items()},
        }
        round_rows.append(round_row)
        current_scored["update_replaced"] = [
            row["update_replaced"] for row in round_reconstruction_rows
        ]
        current_scored["replacement_policy"] = args.replacement_policy
        client_rows.extend(current_scored.to_dict(orient="records"))

        torch.save(
            {
                "experiment_version": "3.12",
                "phase": "trusted_update_reconstruction_closed_loop",
                "mode": args.mode,
                "replacement_policy": args.replacement_policy,
                "monitoring_round": monitoring_round,
                "global_round": global_round,
                "model_state_dict": model.state_dict(),
                "ema_memory": ema_memory,
                "partition_hash": partition_hash,
                "poison_index_hash": poison_hash,
                "model_seed": int(args.model_seed),
                "reconstruction_profile_hash": file_sha256(
                    reconstruction_checkpoint_path
                ),
            },
            continuation_checkpoint,
        )
        save_partial(
            tables_dir,
            round_rows,
            local_rows,
            client_rows,
            signature_rows,
        )

        recall_text = (
            "NA" if not labels.any() else f"{malicious_recall:.3f}"
        )
        print(
            f"monitoring round {monitoring_round:02d}, "
            f"global round {global_round:02d}, "
            f"val macro F1={val_metrics['macro_f1']:.4f}, "
            f"{args.source_class}->{args.target_class}="
            f"{val_pair['source_to_target_rate']:.4%}, "
            f"recall={recall_text}, benign FPR={benign_fpr:.3f}, "
            f"replaced={round_row['replaced_clients']}, "
            f"seconds={round_row['round_seconds']:.1f}"
        )

    round_table = pd.DataFrame(round_rows)
    pd.DataFrame(local_rows).to_csv(
        tables_dir / "reconstruction_local_training.csv",
        index=False,
    )
    pd.DataFrame(client_rows).to_csv(
        tables_dir / "reconstruction_client_rows.csv",
        index=False,
    )
    pd.DataFrame(signature_rows).to_csv(
        tables_dir / "reconstruction_transition_signature_long.csv",
        index=False,
    )
    round_table.to_csv(
        tables_dir / "reconstruction_round_metrics.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        round_table["monitoring_round"],
        round_table["val_macro_f1"],
        marker="o",
        label="Macro F1",
    )
    ax.plot(
        round_table["monitoring_round"],
        round_table["val_source_to_target_rate"],
        marker="s",
        label=f"{args.source_class} to {args.target_class}",
    )
    ax.set_xlabel("Post-warmup monitoring round")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"V3.12 {args.mode} {args.replacement_policy}"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "validation_and_attack_dynamics")

    metadata = {
        "experiment_version": "3.12",
        "phase": "trusted_update_reconstruction_closed_loop",
        "mode": args.mode,
        "replacement_policy": args.replacement_policy,
        "selected_reconstruction_policy": selected_policy,
        "model_seed": int(args.model_seed),
        "warmup_rounds": 4,
        "continuation_rounds": 4,
        "frozen_ema_quantile": FROZEN_EMA_QUANTILE,
        "frozen_instant_quantile": FROZEN_INSTANT_QUANTILE,
        "monitoring_ema_reset": True,
        "original_sample_weights_retained": True,
        "count_cap_used": False,
        "oracle_labels_used_for_aggregation": (
            args.replacement_policy == "oracle_clean_replacement"
        ),
        "oracle_clean_substitutes_trained_for_reporting": (
            args.mode == "strong_attack"
        ),
        "test_sets_accessed": False,
        "mean_validation_macro_f1": float(
            round_table["val_macro_f1"].mean()
        ),
        "mean_validation_source_to_target_rate": float(
            round_table["val_source_to_target_rate"].mean()
        ),
        "mean_malicious_recall": (
            None
            if args.mode == "clean"
            else float(round_table["malicious_recall"].mean())
        ),
        "mean_benign_false_positive_rate": float(
            round_table["benign_false_positive_rate"].mean()
        ),
        "total_seconds": float(time.time() - started),
    }
    with (output_dir / "trusted_update_reconstruction_v312_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Trusted Update Reconstruction V3.12 complete")
    print("Mode:", args.mode)
    print("Replacement policy:", args.replacement_policy)
    print(
        "Mean validation macro F1:",
        f"{round_table['val_macro_f1'].mean():.6f}",
    )
    print(
        "Mean source-to-target rate:",
        f"{round_table['val_source_to_target_rate'].mean():.6f}",
    )
    print(
        "Mean benign FPR:",
        f"{round_table['benign_false_positive_rate'].mean():.6f}",
    )
    if args.mode == "strong_attack":
        print(
            "Mean malicious recall:",
            f"{round_table['malicious_recall'].mean():.6f}",
        )
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

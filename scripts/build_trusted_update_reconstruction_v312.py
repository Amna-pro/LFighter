#!/usr/bin/env python3
"""Replay the trusted four-round warmup and calibrate V3.12 update reconstruction."""
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
    NUM_CLASSES,
    load_protocol_arrays,
    set_seed,
    sqrt_class_weights,
    train_local_model,
)
from neural_models_v24 import build_model
from run_targeted_label_flip_v292 import (
    load_fixed_partitions,
    read_clean_partition_hash,
)
from independent_anchor_v310 import file_sha256
from trusted_update_reconstruction_v312 import (
    add_updates,
    checkpoint_sha256,
    coordinate_median,
    floating_update,
    reconstruct_update,
    relative_l2_error,
    state_max_abs_difference,
    subtract_updates,
    update_cosine,
    update_norm,
    weighted_average_states,
)


POLICIES = (
    "center_only",
    "center_plus_residual",
    "center_plus_scaled_residual",
    "center_plus_scaled_residual_normclip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-seed-dir", required=True, type=Path)
    parser.add_argument("--warmup-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--warmup-rounds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--scale-lower", type=float, default=0.50)
    parser.add_argument("--scale-upper", type=float, default=2.00)
    parser.add_argument("--norm-clip-multiplier", type=float, default=2.50)
    parser.add_argument("--maximum-replay-state-difference", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.warmup_rounds != 4:
        raise ValueError("V3.12 is frozen to four trusted warmup rounds")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    calibration_dir = output_dir / "calibration"
    for path in (tables_dir, figures_dir, calibration_dir):
        path.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, args.threads))
    set_seed(args.model_seed)

    arrays = load_protocol_arrays(args.data_file.expanduser().resolve())
    X_train = arrays["X_train"].astype(np.float32, copy=False)
    y_train = arrays["y_train"].astype(np.int64, copy=False)

    client_indices, partition_hash = load_fixed_partitions(
        args.partition_file.expanduser().resolve(),
        expected_clients=args.num_clients,
        train_rows=len(y_train),
    )
    clean_hash = read_clean_partition_hash(args.clean_seed_dir.expanduser().resolve())
    if partition_hash != clean_hash:
        raise RuntimeError("Fixed partition does not match the clean seed record")

    warmup_dir = args.warmup_dir.expanduser().resolve()
    warmup_metadata_path = warmup_dir / "true_warmup_v310_metadata.json"
    branch_checkpoint_path = (
        warmup_dir / "checkpoints" / "common_round4_warmup_model.pt"
    )
    if not warmup_metadata_path.exists() or not branch_checkpoint_path.exists():
        raise FileNotFoundError("The V3.10 warmup branch artifacts are incomplete")

    with warmup_metadata_path.open("r", encoding="utf-8") as handle:
        warmup_metadata = json.load(handle)
    if warmup_metadata["partition_hash_sha256"] != partition_hash:
        raise RuntimeError("Warmup partition hash mismatch")
    if int(warmup_metadata["model_seed"]) != int(args.model_seed):
        raise RuntimeError("Warmup model seed mismatch")

    class_weights = sqrt_class_weights(y_train, args.max_class_weight)
    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)

    updates_by_round: Dict[int, List[Mapping[str, torch.Tensor]]] = {}
    centers_by_round: Dict[int, Mapping[str, torch.Tensor]] = {}
    update_norms_by_round: Dict[int, List[float]] = {}
    replay_rows = []
    started = time.time()

    print("Trusted Update Reconstruction Calibration V3.12")
    print("Replaying four trusted warmup rounds")
    print("Partition hash:", partition_hash)

    for round_id in range(1, args.warmup_rounds + 1):
        round_started = time.time()
        reference_state = copy.deepcopy(model.state_dict())
        local_states = []
        sample_counts = []
        local_updates = []

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
            update = floating_update(state, reference_state)
            local_states.append(state)
            sample_counts.append(int(len(indices)))
            local_updates.append(update)
            replay_rows.append(
                {
                    "warmup_round": round_id,
                    "client_id": client_id,
                    "client_samples": int(len(indices)),
                    "update_norm": update_norm(update),
                    **metrics,
                }
            )
            del local_model

        center = coordinate_median(local_updates)
        updates_by_round[round_id] = local_updates
        centers_by_round[round_id] = center
        update_norms_by_round[round_id] = [
            update_norm(update) for update in local_updates
        ]
        model.load_state_dict(
            weighted_average_states(
                local_states,
                sample_counts,
                reference_state,
            )
        )
        print(
            f"replayed warmup round {round_id:02d}, "
            f"median update norm={np.median(update_norms_by_round[round_id]):.6f}, "
            f"seconds={time.time() - round_started:.1f}"
        )

    branch_checkpoint = torch.load(
        branch_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    replay_state_difference = state_max_abs_difference(
        model.state_dict(),
        branch_checkpoint["model_state_dict"],
    )
    if replay_state_difference > args.maximum_replay_state_difference:
        raise RuntimeError(
            "Trusted warmup replay does not match the common branch checkpoint: "
            f"max_abs_difference={replay_state_difference:.12g}"
        )

    residuals_by_round: Dict[int, List[Mapping[str, torch.Tensor]]] = {}
    for round_id in range(1, args.warmup_rounds + 1):
        residuals_by_round[round_id] = [
            subtract_updates(update, centers_by_round[round_id])
            for update in updates_by_round[round_id]
        ]

    warmup_update_scale = float(
        np.median(
            [
                np.median(update_norms_by_round[round_id])
                for round_id in range(1, args.warmup_rounds + 1)
            ]
        )
    )

    candidate_rows = []
    detail_rows = []
    for policy in POLICIES:
        for heldout_round in range(1, args.warmup_rounds + 1):
            profile_rounds = [
                round_id
                for round_id in range(1, args.warmup_rounds + 1)
                if round_id != heldout_round
            ]
            residual_profiles = {
                client_id: coordinate_median(
                    [
                        residuals_by_round[round_id][client_id]
                        for round_id in profile_rounds
                    ]
                )
                for client_id in range(args.num_clients)
            }

            for client_id in range(args.num_clients):
                trusted_updates = [
                    updates_by_round[heldout_round][other_client]
                    for other_client in range(args.num_clients)
                    if other_client != client_id
                ]
                trusted_norms = [
                    update_norm(update) for update in trusted_updates
                ]
                current_center = coordinate_median(trusted_updates)
                current_scale = float(np.median(trusted_norms))
                reconstructed, reconstruction_meta = reconstruct_update(
                    current_center=current_center,
                    historical_residual=residual_profiles[client_id],
                    selected_policy=policy,
                    current_update_scale=current_scale,
                    warmup_update_scale=warmup_update_scale,
                    trusted_norms=trusted_norms,
                    scale_lower=args.scale_lower,
                    scale_upper=args.scale_upper,
                    norm_clip_multiplier=args.norm_clip_multiplier,
                )
                actual = updates_by_round[heldout_round][client_id]
                detail_rows.append(
                    {
                        "policy": policy,
                        "heldout_round": heldout_round,
                        "client_id": client_id,
                        "relative_l2_error": relative_l2_error(
                            reconstructed,
                            actual,
                        ),
                        "cosine_similarity": update_cosine(
                            reconstructed,
                            actual,
                        ),
                        "actual_update_norm": update_norm(actual),
                        **reconstruction_meta,
                    }
                )

    detail = pd.DataFrame(detail_rows)
    for policy, group in detail.groupby("policy", sort=False):
        candidate_rows.append(
            {
                "policy": policy,
                "mean_relative_l2_error": float(
                    group["relative_l2_error"].mean()
                ),
                "median_relative_l2_error": float(
                    group["relative_l2_error"].median()
                ),
                "p95_relative_l2_error": float(
                    group["relative_l2_error"].quantile(0.95)
                ),
                "mean_cosine_similarity": float(
                    group["cosine_similarity"].mean()
                ),
                "minimum_cosine_similarity": float(
                    group["cosine_similarity"].min()
                ),
                "mean_norm_ratio": float(
                    (
                        group["reconstructed_update_norm"]
                        / group["actual_update_norm"].clip(lower=1e-12)
                    ).mean()
                ),
                "p95_norm_ratio": float(
                    (
                        group["reconstructed_update_norm"]
                        / group["actual_update_norm"].clip(lower=1e-12)
                    ).quantile(0.95)
                ),
            }
        )

    ranking = pd.DataFrame(candidate_rows)
    ranking["passes_norm_safety"] = ranking["p95_norm_ratio"] <= 2.0
    ranking["passes_mean_cosine_safety"] = (
        ranking["mean_cosine_similarity"] >= 0.0
    )
    ranking["passes_all_safety_constraints"] = ranking[
        ["passes_norm_safety", "passes_mean_cosine_safety"]
    ].all(axis=1)

    eligible = ranking[ranking["passes_all_safety_constraints"]].copy()
    if eligible.empty:
        detail.to_csv(
            tables_dir / "leave_one_round_out_reconstruction_rows.csv",
            index=False,
        )
        ranking.to_csv(
            tables_dir / "reconstruction_candidate_ranking.csv",
            index=False,
        )
        raise RuntimeError("No reconstruction policy passed clean safety")

    eligible = eligible.sort_values(
        [
            "mean_relative_l2_error",
            "median_relative_l2_error",
            "mean_cosine_similarity",
        ],
        ascending=[True, True, False],
    )
    selected_policy = str(eligible.iloc[0]["policy"])
    ranking = ranking.sort_values(
        [
            "passes_all_safety_constraints",
            "mean_relative_l2_error",
            "mean_cosine_similarity",
        ],
        ascending=[False, True, False],
    )

    final_residual_profiles = {
        client_id: coordinate_median(
            [
                residuals_by_round[round_id][client_id]
                for round_id in range(1, args.warmup_rounds + 1)
            ]
        )
        for client_id in range(args.num_clients)
    }

    reconstruction_checkpoint_path = (
        calibration_dir / "trusted_update_reconstruction_profiles.pt"
    )
    torch.save(
        {
            "experiment_version": "3.12",
            "phase": "trusted_clean_update_reconstruction_calibration",
            "selected_policy": selected_policy,
            "client_residual_profiles": final_residual_profiles,
            "warmup_update_scale": warmup_update_scale,
            "scale_lower": float(args.scale_lower),
            "scale_upper": float(args.scale_upper),
            "norm_clip_multiplier": float(args.norm_clip_multiplier),
            "partition_hash": partition_hash,
            "model_seed": int(args.model_seed),
            "warmup_rounds": int(args.warmup_rounds),
            "common_warmup_checkpoint_sha256": checkpoint_sha256(
                branch_checkpoint_path
            ),
            "replay_max_abs_state_difference": replay_state_difference,
            "candidate_ranking": ranking.to_dict(orient="records"),
        },
        reconstruction_checkpoint_path,
    )

    detail.to_csv(
        tables_dir / "leave_one_round_out_reconstruction_rows.csv",
        index=False,
    )
    ranking.to_csv(
        tables_dir / "reconstruction_candidate_ranking.csv",
        index=False,
    )
    pd.DataFrame(replay_rows).to_csv(
        tables_dir / "trusted_warmup_replay_local_updates.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "selected_policy": selected_policy,
                "warmup_update_scale": warmup_update_scale,
                "scale_lower": args.scale_lower,
                "scale_upper": args.scale_upper,
                "norm_clip_multiplier": args.norm_clip_multiplier,
                "replay_max_abs_state_difference": replay_state_difference,
                "maximum_allowed_replay_state_difference": (
                    args.maximum_replay_state_difference
                ),
                "partition_hash_sha256": partition_hash,
                "common_warmup_checkpoint_sha256": checkpoint_sha256(
                    branch_checkpoint_path
                ),
                "reconstruction_profile_sha256": file_sha256(
                    reconstruction_checkpoint_path
                ),
                "malicious_labels_used": False,
                "test_sets_accessed": False,
            }
        ]
    ).to_csv(
        calibration_dir / "reconstruction_calibration_summary.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ordered = ranking.sort_values("mean_relative_l2_error")
    positions = np.arange(len(ordered))
    ax.bar(positions, ordered["mean_relative_l2_error"])
    ax.set_xticks(positions, ordered["policy"], rotation=20, ha="right")
    ax.set_ylabel("Mean relative L2 reconstruction error")
    ax.set_title("V3.12 clean leave-one-round-out reconstruction")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "reconstruction_candidate_error")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(positions, ordered["mean_cosine_similarity"])
    ax.set_xticks(positions, ordered["policy"], rotation=20, ha="right")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("V3.12 reconstruction direction fidelity")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "reconstruction_candidate_cosine")

    metadata = {
        "experiment_version": "3.12",
        "phase": "trusted_clean_update_reconstruction_calibration",
        "selected_policy": selected_policy,
        "selection_basis": (
            "minimum clean leave-one-round-out mean relative L2 error "
            "among norm-safe policies"
        ),
        "warmup_rounds": 4,
        "stable_client_identity_required": True,
        "replay_verified": True,
        "replay_max_abs_state_difference": replay_state_difference,
        "malicious_labels_used": False,
        "test_sets_accessed": False,
        "total_seconds": float(time.time() - started),
    }
    with (output_dir / "trusted_update_reconstruction_v312_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Trusted Update Reconstruction Calibration V3.12 complete")
    print("Selected policy:", selected_policy)
    print(
        "Replay max absolute state difference:",
        f"{replay_state_difference:.12g}",
    )
    print("Warmup update scale:", f"{warmup_update_scale:.8f}")
    print("Calibration:", reconstruction_checkpoint_path)
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

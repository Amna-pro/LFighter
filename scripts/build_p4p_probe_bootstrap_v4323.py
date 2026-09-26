#!/usr/bin/env python3
"""Reconstruct Task65 W3/W4 for matched P4P using archived pre-outcome fingerprints.

Loads train and validation arrays only. Never materializes reserved test arrays.
Performs two deterministic four-round warmup replays. Replay A reconstructs the
historical warmup fingerprints and captures W3/W4. Replay B independently
verifies deterministic W4 state equivalence. Any mismatch aborts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from federated_iot_v26 import CLASS_NAMES, NUM_CLASSES, set_seed, sqrt_class_weights, train_local_model  # noqa: E402
from independent_anchor_v310 import (  # noqa: E402
    SCORE_COLUMN,
    add_candidate_and_ema,
    fit_feature_calibration,
    normalize_rows,
    quantile_higher,
    raw_features,
    save_profiles_npz,
    weighted_average_states,
)
from neural_models_v24 import build_model  # noqa: E402
from run_targeted_label_flip_v292 import load_fixed_partitions  # noqa: E402
from transition_signature_features_v38 import (  # noqa: E402
    balanced_probe_indices,
    class_conditional_probability_means,
    evaluate_validation,
    predict_probabilities,
)
from trusted_update_reconstruction_v312 import checkpoint_sha256, state_max_abs_difference  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconstruct and verify Task65 P4P warmup bootstrap v4.32.3")
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--partition-file", type=Path, required=True)
    p.add_argument("--fingerprint-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-seed", type=int, required=True)
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--evaluation-batch-size", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-class-weight", type=float, default=4.0)
    p.add_argument("--gradient-clip-norm", type=float, default=5.0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--probe-per-class", type=int, default=48)
    p.add_argument("--probe-seed", type=int, default=3701)
    p.add_argument("--ema-decay", type=float, default=0.65)
    p.add_argument("--clean-threshold-quantile", type=float, default=0.95)
    p.add_argument("--equivalence-tolerance", type=float, default=1e-7)
    return p.parse_args()


def load_train_val_only(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as payload:
        required = {"X_train", "y_train", "X_val", "y_val"}
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"Missing train/validation arrays: {sorted(missing)}")
        return (
            np.asarray(payload["X_train"], dtype=np.float32),
            np.asarray(payload["y_train"], dtype=np.int64),
            np.asarray(payload["X_val"], dtype=np.float32),
            np.asarray(payload["y_val"], dtype=np.int64),
        )


def replay_once(args, X_train, y_train, X_val, y_val, client_indices):
    set_seed(args.model_seed)
    source_id = CLASS_NAMES.index("DDoS")
    target_id = CLASS_NAMES.index("Benign")
    probe_indices = balanced_probe_indices(y_val, args.probe_per_class, args.probe_seed)
    X_probe = X_val[probe_indices]
    y_probe = y_val[probe_indices]
    probe_hash = hashlib.sha256(np.ascontiguousarray(probe_indices).tobytes()).hexdigest()
    class_weights = sqrt_class_weights(y_train, args.max_class_weight)
    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    round3_state = None
    matrices_by_round = {}
    round_metrics = []

    for round_id in range(1, 5):
        reference_state = copy.deepcopy(model.state_dict())
        local_states: List[Mapping[str, torch.Tensor]] = []
        sample_counts: List[int] = []
        local_matrices = []
        for client_id in range(args.num_clients):
            indices = client_indices[client_id]
            local_model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
            local_model.load_state_dict(reference_state)
            state, _ = train_local_model(
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
            local_matrices.append(class_conditional_probability_means(probabilities, y_probe))
            local_states.append(state)
            sample_counts.append(int(len(indices)))
            del local_model
        model.load_state_dict(weighted_average_states(local_states, sample_counts, reference_state))
        matrices_by_round[round_id] = np.stack(local_matrices, axis=0)
        if round_id == 3:
            round3_state = copy.deepcopy(model.state_dict())
        val_metrics, val_pair = evaluate_validation(
            model, X_val, y_val, source_id, target_id, args.evaluation_batch_size
        )
        round_metrics.append({
            "round": round_id,
            "macro_f1": float(val_metrics["macro_f1"]),
            "source_to_target_rate": float(val_pair["source_to_target_rate"]),
        })

    if round3_state is None:
        raise RuntimeError("W3 was not captured")
    return round3_state, copy.deepcopy(model.state_dict()), matrices_by_round, round_metrics, probe_hash


def historical_calibration(args, matrices_by_round, source_id, target_id, calibration_dir):
    full_profiles = {
        client_id: normalize_rows(
            np.median(np.stack([matrices_by_round[r][client_id] for r in range(1, 5)]), axis=0)
        )
        for client_id in range(args.num_clients)
    }
    rows = []
    for round_id in range(1, 5):
        round_stack = matrices_by_round[round_id]
        consensus = normalize_rows(np.median(round_stack, axis=0))
        others = [r for r in range(1, 5) if r != round_id]
        loo = {
            cid: normalize_rows(np.median(np.stack([matrices_by_round[r][cid] for r in others]), axis=0))
            for cid in range(args.num_clients)
        }
        for cid in range(args.num_clients):
            rows.append({
                "monitoring_round": round_id,
                "global_round": round_id,
                "client_id": cid,
                "actual_malicious": False,
                **raw_features(round_stack[cid], loo[cid], consensus, source_id, target_id),
            })
    clean_raw = pd.DataFrame(rows)
    clean_scored, _ = fit_feature_calibration(clean_raw)
    clean_scored, _ = add_candidate_and_ema(clean_scored, args.ema_decay, initial_ema={})
    threshold = quantile_higher(clean_scored[SCORE_COLUMN], args.clean_threshold_quantile)
    fpr = float((clean_scored[SCORE_COLUMN] > threshold).mean())
    calibration_dir.mkdir(parents=True, exist_ok=True)
    profile_path = calibration_dir / "trusted_client_profiles.npz"
    profile_hash = save_profiles_npz(profile_path, full_profiles)
    return profile_hash, float(threshold), fpr


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    if args.num_clients != 20:
        raise ValueError("Reviewer recovery is frozen to 20 clients")
    if (args.batch_size, args.learning_rate, args.weight_decay, args.probe_per_class, args.probe_seed, args.ema_decay, args.clean_threshold_quantile) != (2048, 3e-4, 1e-4, 48, 3701, 0.65, 0.95):
        raise ValueError("Frozen Task65 warmup settings were changed")

    fp = json.loads(args.fingerprint_file.expanduser().resolve().read_text(encoding="utf-8"))
    key = str(args.model_seed)
    if key not in fp["seeds"]:
        raise KeyError(f"Seed {key} is absent from frozen historical fingerprints")
    expected = fp["seeds"][key]

    X_train, y_train, X_val, y_val = load_train_val_only(args.data_file)
    client_indices, partition_hash = load_fixed_partitions(
        args.partition_file.expanduser().resolve(), args.num_clients, len(y_train)
    )
    if partition_hash != fp["partition_hash_sha256"]:
        raise RuntimeError("Logical partition hash does not match historical Task65 evidence")

    w3_a, w4_a, matrices_a, metrics_a, probe_hash_a = replay_once(
        args, X_train, y_train, X_val, y_val, client_indices
    )
    if probe_hash_a != fp["probe_hash_sha256"]:
        raise RuntimeError("Probe hash mismatch against historical Task65 evidence")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    checkpoint_dir = output_dir / "warmup" / "checkpoints"
    calibration_dir = output_dir / "warmup" / "calibration"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    source_id = CLASS_NAMES.index("DDoS")
    target_id = CLASS_NAMES.index("Benign")
    profile_hash, threshold, fpr = historical_calibration(
        args, matrices_a, source_id, target_id, calibration_dir
    )

    observed_macro = [round(float(x["macro_f1"]), 4) for x in metrics_a]
    observed_pct = [round(float(x["source_to_target_rate"]) * 100.0, 4) for x in metrics_a]
    if observed_macro != [round(float(x), 4) for x in expected["round_macro_f1_4dp"]]:
        raise RuntimeError(f"Historical macro F1 fingerprint mismatch: observed={observed_macro}")
    if observed_pct != [round(float(x), 4) for x in expected["round_source_to_target_percent_4dp"]]:
        raise RuntimeError(f"Historical source-target fingerprint mismatch: observed={observed_pct}")
    if round(threshold, 6) != round(float(expected["ema_threshold_6dp"]), 6):
        raise RuntimeError(f"Historical EMA threshold mismatch: observed={threshold:.12g}")
    if abs(fpr - float(expected["clean_fpr"])) > 1e-12:
        raise RuntimeError(f"Historical clean FPR mismatch: observed={fpr:.12g}")
    if profile_hash.lower() != str(expected["profile_sha256"]).lower():
        raise RuntimeError(f"Historical profile SHA256 mismatch: observed={profile_hash}")

    # Independent second replay. No output from the first replay is used as input.
    _, w4_b, _, metrics_b, probe_hash_b = replay_once(
        args, X_train, y_train, X_val, y_val, client_indices
    )
    if probe_hash_b != probe_hash_a:
        raise RuntimeError("Independent replay probe hash mismatch")
    max_abs = state_max_abs_difference(w4_a, w4_b)
    if max_abs > args.equivalence_tolerance:
        raise RuntimeError(
            f"Independent W4 replay mismatch: max_abs={max_abs:.12g} > tol={args.equivalence_tolerance:.12g}"
        )

    w4_path = checkpoint_dir / "common_round4_warmup_model.pt"
    torch.save({
        "experiment_version": "reviewer_v4323_reconstructed_task65_warmup",
        "phase": "preoutcome_historical_warmup_reconstruction",
        "checkpoint_type": "deterministic_reconstruction_of_task65_round4",
        "round": 4,
        "model_state_dict": w4_a,
        "input_dim": int(X_train.shape[1]),
        "num_classes": int(NUM_CLASSES),
        "partition_hash": partition_hash,
        "model_seed": int(args.model_seed),
        "probe_hash": probe_hash_a,
        "historical_fingerprint_verified": True,
        "test_sets_accessed": False,
    }, w4_path)

    meta = {
        "experiment_version": "reviewer_v4323_reconstructed_task65_warmup",
        "phase": "preoutcome_historical_warmup_reconstruction",
        "model_seed": int(args.model_seed),
        "warmup_rounds": 4,
        "partition_hash_sha256": partition_hash,
        "probe_hash_sha256": probe_hash_a,
        "probe_rows_per_class": int(args.probe_per_class),
        "probe_seed": int(args.probe_seed),
        "ema_decay": float(args.ema_decay),
        "clean_threshold_quantile": float(args.clean_threshold_quantile),
        "clean_ema_threshold": float(threshold),
        "clean_false_positive_rate": float(fpr),
        "profile_sha256": profile_hash,
        "historical_fingerprint_verified": True,
        "historical_checkpoint_file_recovered": False,
        "independent_replay_w4_max_abs_difference": float(max_abs),
        "equivalence_tolerance": float(args.equivalence_tolerance),
        "test_sets_accessed": False,
        "common_branch_checkpoint": str(w4_path),
    }
    meta_path = output_dir / "warmup" / "true_warmup_v310_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    bootstrap = {
        "protocol": "reviewer_v4323_p4p_probe_bootstrap_historical_fingerprint_recovery",
        "model_seed": int(args.model_seed),
        "partition_hash_sha256": partition_hash,
        "frozen_round4_checkpoint": str(w4_path),
        # Legacy field name retained for v4.32.2 runner compatibility. This is the
        # SHA256 of the verified reconstructed W4, not a claim that the lost file was recovered.
        "frozen_round4_checkpoint_sha256": checkpoint_sha256(w4_path),
        "round4_replay_max_abs_difference": float(max_abs),
        "equivalence_tolerance": float(args.equivalence_tolerance),
        "historical_fingerprint_verified": True,
        "historical_checkpoint_file_recovered": False,
        "profile_sha256": profile_hash,
        "probe_hash_sha256": probe_hash_a,
        "test_arrays_materialized": False,
        "round3_model_state_dict": w3_a,
    }
    bootstrap_path = output_dir / "p4p_probe_bootstrap_v4323.pt"
    torch.save(bootstrap, bootstrap_path)
    marker = output_dir / "p4p_probe_bootstrap_v4323.json"
    marker.write_text(json.dumps({k:v for k,v in bootstrap.items() if k != "round3_model_state_dict"}, indent=2)+"\n", encoding="utf-8")

    seed_dir = output_dir / "clean_seed_record"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "seed_metadata.json").write_text(json.dumps({
        "model_seed": int(args.model_seed),
        "planned_attack_seed": int(args.model_seed),
        "partition_hash_sha256": partition_hash,
        "phase": "reviewer_reconstructed_task65_warmup_v4323",
        "original_task65_seed_metadata_recovered": False,
        "historical_fingerprint_verified": True,
        "test_sets_accessed": False,
    }, indent=2)+"\n", encoding="utf-8")

    print("TASK65 HISTORICAL WARMUP RECONSTRUCTION VERIFIED")
    print("MODEL SEED:", args.model_seed)
    print("PARTITION HASH:", partition_hash)
    print("PROBE HASH:", probe_hash_a)
    print("PROFILE HASH:", profile_hash)
    print("EMA THRESHOLD:", f"{threshold:.6f}")
    print("CLEAN FPR:", f"{fpr:.4%}")
    print("ROUND MACRO F1 4DP:", observed_macro)
    print("ROUND SOURCE TARGET PERCENT 4DP:", observed_pct)
    print("INDEPENDENT W4 MAX ABS DIFFERENCE:", f"{max_abs:.12g}")
    print("HISTORICAL FINGERPRINT VERIFIED: True")
    print("HISTORICAL CHECKPOINT FILE RECOVERED: False")
    print("TEST ARRAYS MATERIALIZED: False")
    print("WARMUP DIR:", output_dir / "warmup")
    print("BOOTSTRAP:", bootstrap_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

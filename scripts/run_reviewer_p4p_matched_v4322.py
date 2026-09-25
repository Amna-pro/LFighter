#!/usr/bin/env python3
"""Matched P4P reviewer-comparison branch for CIC IoT-DIAD BATR FL.

This runner uses train/validation arrays only, reuses the exact plain-branch
poison plan for attacked runs, and records phase-level timing.  It never
materializes final test arrays.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from federated_iot_v26 import CLASS_NAMES, NUM_CLASSES, set_seed, sqrt_class_weights, train_local_model  # noqa: E402
from neural_models_v24 import build_model  # noqa: E402
from p4p_matched_v4322 import (  # noqa: E402
    P4PConfig,
    aggregate_trusted_states,
    ensemble_flags,
    global_probe,
    mad_magnitude_filter,
    probe_responses,
    update_temporal_suspicion,
)
from run_frozen_untargeted_defense_v320b1 import (  # noqa: E402
    ATTACK_TYPES,
    DEFAULT_MALICIOUS_CLIENTS,
    load_exact_v320a3_poison_plan,
    parse_client_ids,
)
from run_targeted_label_flip_v292 import load_fixed_partitions, read_clean_partition_hash  # noqa: E402
from transition_signature_features_v38 import evaluate_validation, predict_probabilities  # noqa: E402
from trusted_update_reconstruction_v312 import checkpoint_sha256, floating_update, state_max_abs_difference  # noqa: E402

EXPERIMENT_VERSION = "4.32.2-P4P"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run matched P4P reviewer branch")
    p.add_argument("--mode", choices=["clean", "strong_attack"], required=True)
    p.add_argument("--attack-type", choices=ATTACK_TYPES, required=True)
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--partition-file", type=Path, required=True)
    p.add_argument("--clean-seed-dir", type=Path, required=True)
    p.add_argument("--warmup-dir", type=Path, required=True)
    p.add_argument("--probe-bootstrap-file", type=Path, required=True)
    p.add_argument("--plain-branch-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-seed", type=int, required=True)
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--continuation-rounds", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--evaluation-batch-size", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-class-weight", type=float, default=4.0)
    p.add_argument("--gradient-clip-norm", type=float, default=5.0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--source-class", default="DDoS")
    p.add_argument("--target-class", default="Benign")
    p.add_argument("--malicious-clients", default=DEFAULT_MALICIOUS_CLIENTS)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_train_val_only(path: Path):
    with np.load(path.expanduser().resolve(), allow_pickle=False) as payload:
        required = {"X_train", "y_train", "X_val", "y_val"}
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"Missing required train/validation arrays: {sorted(missing)}")
        return (
            np.asarray(payload["X_train"], dtype=np.float32),
            np.asarray(payload["y_train"], dtype=np.int64),
            np.asarray(payload["X_val"], dtype=np.float32),
            np.asarray(payload["y_val"], dtype=np.int64),
        )


def class_confusion_rows(model, X_val, y_val, batch_size, monitoring_round, global_round, attack_type):
    probs = predict_probabilities(model, X_val, batch_size)
    pred = probs.argmax(axis=1)
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (y_val, pred), 1)
    class_rows = []
    confusion_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        support = int(matrix[class_id, :].sum())
        predicted = int(matrix[:, class_id].sum())
        tp = int(matrix[class_id, class_id])
        recall = float(tp / support) if support else float("nan")
        precision = float(tp / predicted) if predicted else float("nan")
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
            else 0.0
        )
        class_rows.append({
            "monitoring_round": monitoring_round,
            "global_round": global_round,
            "attack_type": attack_type,
            "class_id": class_id,
            "class_name": class_name,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        for predicted_id, predicted_name in enumerate(CLASS_NAMES):
            confusion_rows.append({
                "monitoring_round": monitoring_round,
                "global_round": global_round,
                "attack_type": attack_type,
                "true_id": class_id,
                "true_class": class_name,
                "predicted_id": predicted_id,
                "predicted_class": predicted_name,
                "count": int(matrix[class_id, predicted_id]),
            })
    return class_rows, confusion_rows


def main() -> int:
    args = parse_args()
    if args.continuation_rounds != 4:
        raise ValueError("Matched primary P4P comparison is frozen to four monitored rounds")
    if args.source_class not in CLASS_NAMES or args.target_class not in CLASS_NAMES:
        raise ValueError("Unknown diagnostic source/target class")
    if args.mode == "strong_attack" and args.plain_branch_dir is None:
        raise ValueError("--plain-branch-dir is required for strong_attack")
    if args.mode == "strong_attack" and len(parse_client_ids(args.malicious_clients)) != 8:
        raise ValueError("Primary matched comparison requires eight malicious clients")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
    tables_dir = output_dir / "tables"
    checkpoints_dir = output_dir / "checkpoints" / "reviewer_round_checkpoints"
    attack_dir = output_dir / "attack_manifest"
    for p in (tables_dir, checkpoints_dir, attack_dir):
        p.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, args.threads))
    set_seed(args.model_seed)
    X_train, y_train, X_val, y_val = load_train_val_only(args.data_file)
    client_indices, partition_hash = load_fixed_partitions(
        args.partition_file.expanduser().resolve(),
        expected_clients=args.num_clients,
        train_rows=len(y_train),
    )
    clean_hash = read_clean_partition_hash(args.clean_seed_dir.expanduser().resolve())
    if clean_hash != partition_hash:
        raise RuntimeError("Clean partition hash mismatch")

    warmup_dir = args.warmup_dir.expanduser().resolve()
    warmup_meta = json.loads((warmup_dir / "true_warmup_v310_metadata.json").read_text(encoding="utf-8"))
    round4_path = warmup_dir / "checkpoints" / "common_round4_warmup_model.pt"
    round4 = torch.load(round4_path, map_location="cpu", weights_only=False)
    if int(warmup_meta["model_seed"]) != args.model_seed:
        raise RuntimeError("Warmup seed mismatch")
    if warmup_meta["partition_hash_sha256"] != partition_hash:
        raise RuntimeError("Warmup partition mismatch")

    bootstrap = torch.load(args.probe_bootstrap_file.expanduser().resolve(), map_location="cpu", weights_only=False)
    if int(bootstrap["model_seed"]) != args.model_seed:
        raise RuntimeError("P4P bootstrap seed mismatch")
    if bootstrap["partition_hash_sha256"] != partition_hash:
        raise RuntimeError("P4P bootstrap partition mismatch")
    if bootstrap["frozen_round4_checkpoint_sha256"] != checkpoint_sha256(round4_path):
        raise RuntimeError("P4P bootstrap frozen W4 hash mismatch")
    if float(bootstrap["round4_replay_max_abs_difference"]) > float(bootstrap["equivalence_tolerance"]):
        raise RuntimeError("P4P bootstrap did not pass W4 equivalence")

    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    model.load_state_dict(round4["model_state_dict"])
    previous_global_state = copy.deepcopy(bootstrap["round3_model_state_dict"])
    if state_max_abs_difference(model.state_dict(), round4["model_state_dict"]) != 0.0:
        raise RuntimeError("Failed to load frozen W4 exactly")

    if args.mode == "strong_attack":
        requested = parse_client_ids(args.malicious_clients)
        malicious_clients, poisoned_positions, poisoned_labels, poison_manifest, poison_hash = load_exact_v320a3_poison_plan(
            plain_branch_dir=args.plain_branch_dir,
            client_indices=client_indices,
            attack_type=args.attack_type,
            model_seed=args.model_seed,
            expected_malicious_clients=requested,
        )
        source_manifest = args.plain_branch_dir.expanduser().resolve() / "attack_manifest"
        for name in ("poisoned_indices.npz", "poisoned_labels.npz", "malicious_client_poison_manifest.csv"):
            src = source_manifest / name
            if src.exists():
                shutil.copy2(src, attack_dir / name)
    else:
        malicious_clients = []
        poisoned_positions = {i: np.empty(0, dtype=np.int64) for i in range(args.num_clients)}
        poisoned_labels = {i: np.empty(0, dtype=np.int64) for i in range(args.num_clients)}
        poison_manifest = pd.DataFrame()
        poison_hash = "clean_no_poison"

    class_weights = sqrt_class_weights(y_train, args.max_class_weight)
    source_id = CLASS_NAMES.index(args.source_class)
    target_id = CLASS_NAMES.index(args.target_class)
    cfg = P4PConfig()
    suspicion = np.zeros(args.num_clients, dtype=np.float64)
    banned = np.zeros(args.num_clients, dtype=bool)

    round_rows: List[Dict[str, object]] = []
    client_rows: List[Dict[str, object]] = []
    class_rows: List[Dict[str, object]] = []
    confusion_rows: List[Dict[str, object]] = []
    started = time.perf_counter()

    for monitoring_round in range(1, args.continuation_rounds + 1):
        global_round = 4 + monitoring_round
        round_started = time.perf_counter()
        reference_state = copy.deepcopy(model.state_dict())

        t0 = time.perf_counter()
        local_states: List[Mapping[str, torch.Tensor]] = []
        sample_counts: List[int] = []
        train_metrics = []
        for client_id in range(args.num_clients):
            indices = client_indices[client_id]
            local_y = y_train[indices].copy()
            pos = poisoned_positions[client_id]
            if len(pos):
                repl = poisoned_labels[client_id]
                if len(repl) != len(pos):
                    raise RuntimeError(f"Poison-plan length mismatch for client {client_id}")
                local_y[pos] = repl
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
            local_states.append(state)
            sample_counts.append(int(len(indices)))
            train_metrics.append(metrics)
            del local_model
        local_train_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        updates = [floating_update(state, reference_state) for state in local_states]
        probe = global_probe(reference_state, previous_global_state)
        responses = probe_responses(updates, probe)
        probe_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        norms, mag_low, mag_high, mag_pass = mad_magnitude_filter(updates, k=cfg.mad_k)
        kflag, dflag, iflag, votes, anomaly = ensemble_flags(responses, mag_pass, cfg=cfg)
        suspicion, banned = update_temporal_suspicion(suspicion, anomaly, banned, cfg=cfg)
        detector_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        rejected = (~mag_pass) | anomaly | banned
        trusted_ids = np.flatnonzero(~rejected).astype(int).tolist()
        if not trusted_ids:
            raise RuntimeError("P4P trusted set is empty; branch aborted by frozen rule")
        mitigation_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        aggregated_state = aggregate_trusted_states(
            local_states, sample_counts, reference_state, trusted_ids
        )
        model.load_state_dict(aggregated_state)
        aggregation_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        val_metrics, val_pair = evaluate_validation(
            model, X_val, y_val, source_id, target_id, args.evaluation_batch_size
        )
        current_class, current_confusion = class_confusion_rows(
            model, X_val, y_val, args.evaluation_batch_size,
            monitoring_round, global_round, args.attack_type,
        )
        class_rows.extend(current_class)
        confusion_rows.extend(current_confusion)
        validation_seconds = time.perf_counter() - t0

        labels = np.asarray([i in malicious_clients for i in range(args.num_clients)], dtype=bool)
        benign = ~labels
        detection_flags = rejected
        malicious_recall = float(np.mean(detection_flags[labels])) if labels.any() else float("nan")
        benign_fpr = float(np.mean(detection_flags[benign])) if benign.any() else float("nan")
        precision = (
            float(np.sum(detection_flags & labels) / max(int(np.sum(detection_flags)), 1))
            if labels.any() else float("nan")
        )

        for client_id in range(args.num_clients):
            client_rows.append({
                "monitoring_round": monitoring_round,
                "global_round": global_round,
                "client_id": client_id,
                "actual_malicious": bool(labels[client_id]),
                "client_samples": int(sample_counts[client_id]),
                "poisoned_rows": int(len(poisoned_positions[client_id])),
                "update_l2_norm": float(norms[client_id]),
                "mad_low": float(mag_low),
                "mad_high": float(mag_high),
                "magnitude_pass": bool(mag_pass[client_id]),
                "probe_response_cosine": float(responses[client_id]),
                "kmeans_flag": bool(kflag[client_id]),
                "dbscan_flag": bool(dflag[client_id]),
                "isolation_flag": bool(iflag[client_id]),
                "ensemble_votes": int(votes[client_id]),
                "ensemble_anomaly": bool(anomaly[client_id]),
                "suspicion_score": float(suspicion[client_id]),
                "permanently_banned": bool(banned[client_id]),
                "rejected_current_round": bool(rejected[client_id]),
                "trusted_current_round": bool(not rejected[client_id]),
            })

        total_round_seconds = time.perf_counter() - round_started
        round_rows.append({
            "monitoring_round": monitoring_round,
            "global_round": global_round,
            "mode": args.mode,
            "attack_type": args.attack_type,
            "arm": "p4p_matched",
            "trusted_clients": len(trusted_ids),
            "rejected_clients": int(np.sum(rejected)),
            "magnitude_rejected_clients": int(np.sum(~mag_pass)),
            "ensemble_anomalous_clients": int(np.sum(anomaly)),
            "permanently_banned_clients": int(np.sum(banned)),
            "malicious_recall": malicious_recall,
            "benign_false_positive_rate": benign_fpr,
            "detection_precision": precision,
            "phase_local_training_seconds": float(local_train_seconds),
            "phase_probe_seconds": float(probe_seconds),
            "phase_detector_seconds": float(detector_seconds),
            "phase_mitigation_seconds": float(mitigation_seconds),
            "phase_aggregation_seconds": float(aggregation_seconds),
            "phase_validation_seconds": float(validation_seconds),
            "round_seconds": float(total_round_seconds),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"val_{k}": v for k, v in val_pair.items()},
        })

        torch.save({
            "experiment_version": EXPERIMENT_VERSION,
            "arm": "p4p_matched",
            "monitoring_round": monitoring_round,
            "global_round": global_round,
            "model_seed": args.model_seed,
            "model_state_dict": model.state_dict(),
            "previous_global_state_dict": reference_state,
            "p4p_suspicion_scores": suspicion,
            "p4p_permanently_banned": banned,
            "partition_hash": partition_hash,
            "poison_index_hash": poison_hash,
        }, checkpoints_dir / f"global_round_{global_round:02d}_model.pt")

        previous_global_state = reference_state
        print(
            f"round {global_round}: macroF1={val_metrics['macro_f1']:.4f}, "
            f"trusted={len(trusted_ids)}, recall={malicious_recall}, "
            f"FPR={benign_fpr:.3f}, seconds={total_round_seconds:.1f}"
        )

    round_table = pd.DataFrame(round_rows)
    round_table.to_csv(tables_dir / "p4p_round_metrics.csv", index=False)
    pd.DataFrame(client_rows).to_csv(tables_dir / "p4p_client_decisions.csv", index=False)
    pd.DataFrame(class_rows).to_csv(tables_dir / "validation_class_metrics_long.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(tables_dir / "validation_confusion_matrix_long.csv", index=False)

    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "phase": "reviewer_matched_p4p_comparator",
        "mode": args.mode,
        "attack_type": args.attack_type,
        "model_seed": int(args.model_seed),
        "num_clients": int(args.num_clients),
        "continuation_rounds": int(args.continuation_rounds),
        "malicious_clients": malicious_clients,
        "partition_hash_sha256": partition_hash,
        "poison_index_hash_sha256": poison_hash,
        "warmup_round4_checkpoint_sha256": checkpoint_sha256(round4_path),
        "probe_bootstrap_file": str(args.probe_bootstrap_file.expanduser().resolve()),
        "p4p_config": cfg.__dict__,
        "aggregation": "sample_count_weighted_fedavg_over_trusted_set",
        "test_sets_accessed": False,
        "attack_specific_retuning": False,
        "total_seconds": float(time.perf_counter() - started),
        "mean_validation_macro_f1": float(round_table["val_macro_f1"].mean()),
        "mean_malicious_recall": None if args.mode == "clean" else float(round_table["malicious_recall"].mean()),
        "mean_benign_fpr": float(round_table["benign_false_positive_rate"].mean()),
    }
    (output_dir / "REVIEWER_P4P_MATCHED_COMPLETE.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("P4P MATCHED BRANCH COMPLETE")
    print("TEST SETS ACCESSED: False")
    print("OUTPUT:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

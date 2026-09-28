#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from federated_iot_v26 import CLASS_NAMES, NUM_CLASSES, set_seed, sqrt_class_weights, train_local_model
from neural_models_v24 import build_model
from run_frozen_untargeted_defense_v320b1 import ATTACK_TYPES, DEFAULT_MALICIOUS_CLIENTS, load_exact_v320a3_poison_plan, parse_client_ids
from run_reviewer_p4p_matched_v4322 import class_confusion_rows, load_train_val_only
from run_targeted_label_flip_v292 import load_fixed_partitions, read_clean_partition_hash
from transition_signature_features_v38 import evaluate_validation
from trusted_update_reconstruction_v312 import checkpoint_sha256, floating_update, state_from_update, state_max_abs_difference, update_norm
from verify_task43_c1_baselines_v414 import multi_krum

EXPERIMENT_VERSION = "4.33.0-MULTI-KRUM"
ASSUMED_BYZANTINE_COUNT = 8


def parse_args():
    p = argparse.ArgumentParser(description="Run matched Multi-Krum reviewer branch")
    p.add_argument("--mode", choices=["clean", "strong_attack"], required=True)
    p.add_argument("--attack-type", choices=ATTACK_TYPES, required=True)
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--partition-file", type=Path, required=True)
    p.add_argument("--clean-seed-dir", type=Path, required=True)
    p.add_argument("--warmup-dir", type=Path, required=True)
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
    return p.parse_args()


def main():
    args = parse_args()
    if args.source_class not in CLASS_NAMES or args.target_class not in CLASS_NAMES:
        raise ValueError("Unknown diagnostic source/target class")
    if args.mode == "strong_attack" and args.plain_branch_dir is None:
        raise ValueError("--plain-branch-dir is required for strong_attack")
    if args.mode == "strong_attack" and len(parse_client_ids(args.malicious_clients)) != 8:
        raise ValueError("Primary matched comparison requires eight malicious clients")
    if args.num_clients != 20:
        raise ValueError("Primary matched Multi-Krum comparison requires 20 clients")
    if args.continuation_rounds != 4:
        raise ValueError("Primary matched Multi-Krum comparison requires four continuation rounds")

    selection_count = args.num_clients - ASSUMED_BYZANTINE_COUNT - 2
    if selection_count != 10:
        raise RuntimeError("Frozen Multi-Krum selection count must equal 10")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")

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

    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    model.load_state_dict(round4["model_state_dict"])
    if state_max_abs_difference(model.state_dict(), round4["model_state_dict"]) != 0.0:
        raise RuntimeError("Failed to load frozen W4 exactly")

    if args.mode == "strong_attack":
        requested = parse_client_ids(args.malicious_clients)
        malicious_clients, poisoned_positions, poisoned_labels, _, poison_hash = load_exact_v320a3_poison_plan(
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
        poison_hash = "clean_no_poison"

    class_weights = sqrt_class_weights(y_train, args.max_class_weight)
    source_id = CLASS_NAMES.index(args.source_class)
    target_id = CLASS_NAMES.index(args.target_class)

    round_rows, client_rows, class_rows, confusion_rows = [], [], [], []
    started = time.perf_counter()

    for monitoring_round in range(1, args.continuation_rounds + 1):
        global_round = 4 + monitoring_round
        round_started = time.perf_counter()
        reference_state = copy.deepcopy(model.state_dict())

        t0 = time.perf_counter()
        local_states, sample_counts = [], []
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
            state, _ = train_local_model(
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
            del local_model
        local_train_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        updates = [floating_update(state, reference_state) for state in local_states]
        aggregated_update, selected = multi_krum(updates, assumed_byzantine_count=ASSUMED_BYZANTINE_COUNT)
        selected = [int(x) for x in selected]
        if len(selected) != selection_count:
            raise RuntimeError(f"Multi-Krum selected {len(selected)} clients; expected {selection_count}")
        if len(set(selected)) != selection_count:
            raise RuntimeError("Multi-Krum selection contains duplicate client ids")
        if any(x < 0 or x >= args.num_clients for x in selected):
            raise RuntimeError("Multi-Krum selection contains invalid client id")
        aggregated_state = state_from_update(reference_state, aggregated_update)
        model.load_state_dict(aggregated_state)
        aggregation_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        val_metrics, val_pair = evaluate_validation(model, X_val, y_val, source_id, target_id, args.evaluation_batch_size)
        current_class, current_confusion = class_confusion_rows(
            model, X_val, y_val, args.evaluation_batch_size, monitoring_round, global_round, args.attack_type
        )
        class_rows.extend(current_class)
        confusion_rows.extend(current_confusion)
        validation_seconds = time.perf_counter() - t0

        labels = np.asarray([i in malicious_clients for i in range(args.num_clients)], dtype=bool)
        selected_set = set(selected)
        for client_id in range(args.num_clients):
            client_rows.append({
                "monitoring_round": monitoring_round,
                "global_round": global_round,
                "client_id": client_id,
                "actual_malicious": bool(labels[client_id]),
                "client_samples": int(sample_counts[client_id]),
                "poisoned_rows": int(len(poisoned_positions[client_id])),
                "update_l2_norm": float(update_norm(updates[client_id])),
                "selected_by_multi_krum": bool(client_id in selected_set),
                "client_detection_applicable": False,
                "reconstructed": False,
            })

        total_round_seconds = time.perf_counter() - round_started
        round_rows.append({
            "monitoring_round": monitoring_round,
            "global_round": global_round,
            "mode": args.mode,
            "attack_type": args.attack_type,
            "arm": "multi_krum",
            "submitted_clients": args.num_clients,
            "assumed_byzantine_count": ASSUMED_BYZANTINE_COUNT,
            "selected_clients_count": len(selected),
            "selected_client_ids": json.dumps(selected, separators=(",", ":")),
            "selected_malicious_clients_count": int(sum(labels[x] for x in selected)),
            "phase_local_training_seconds": float(local_train_seconds),
            "phase_detector_seconds": 0.0,
            "phase_reconstruction_seconds": 0.0,
            "phase_aggregation_seconds": float(aggregation_seconds),
            "phase_validation_seconds": float(validation_seconds),
            "round_seconds": float(total_round_seconds),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"val_{k}": v for k, v in val_pair.items()},
        })

        torch.save({
            "experiment_version": EXPERIMENT_VERSION,
            "arm": "multi_krum",
            "monitoring_round": monitoring_round,
            "global_round": global_round,
            "model_seed": args.model_seed,
            "attack_type": args.attack_type,
            "model_state_dict": model.state_dict(),
            "previous_global_state_dict": reference_state,
            "partition_hash": partition_hash,
            "poison_index_hash": poison_hash,
            "aggregation": "multi_krum_all_20_submitted_updates",
            "assumed_byzantine_count": ASSUMED_BYZANTINE_COUNT,
            "selected_clients_count": len(selected),
            "selected_client_ids": selected,
        }, checkpoints_dir / f"global_round_{global_round:02d}_model.pt")

        print(
            f"round {global_round}: macroF1={val_metrics['macro_f1']:.4f}, "
            f"multi_krum_selected={len(selected)}, selected_malicious={int(sum(labels[x] for x in selected))}, "
            f"seconds={total_round_seconds:.1f}"
        )

    round_table = pd.DataFrame(round_rows)
    round_table.to_csv(tables_dir / "multi_krum_round_metrics.csv", index=False)
    pd.DataFrame(client_rows).to_csv(tables_dir / "multi_krum_client_records.csv", index=False)
    pd.DataFrame(class_rows).to_csv(tables_dir / "validation_class_metrics_long.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(tables_dir / "validation_confusion_matrix_long.csv", index=False)

    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "phase": "reviewer_matched_multi_krum_comparator",
        "mode": args.mode,
        "attack_type": args.attack_type,
        "model_seed": int(args.model_seed),
        "num_clients": int(args.num_clients),
        "continuation_rounds": int(args.continuation_rounds),
        "malicious_clients": malicious_clients,
        "partition_hash_sha256": partition_hash,
        "poison_index_hash_sha256": poison_hash,
        "warmup_round4_checkpoint_sha256": checkpoint_sha256(round4_path),
        "aggregation": "multi_krum_all_20_submitted_updates",
        "assumed_byzantine_count": ASSUMED_BYZANTINE_COUNT,
        "selected_clients_count": selection_count,
        "detector_used": False,
        "client_reconstruction_used": False,
        "sample_count_weighting_used": False,
        "test_sets_accessed": False,
        "attack_specific_retuning": False,
        "total_seconds": float(time.perf_counter() - started),
        "mean_validation_macro_f1": float(round_table["val_macro_f1"].mean()),
    }
    (output_dir / "REVIEWER_MULTI_KRUM_MATCHED_COMPLETE.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("MULTI KRUM MATCHED BRANCH COMPLETE")
    print("ASSUMED BYZANTINE COUNT:", ASSUMED_BYZANTINE_COUNT)
    print("SELECTED CLIENTS PER ROUND:", selection_count)
    print("TEST SETS ACCESSED: False")
    print("ATTACK SPECIFIC RETUNING: False")
    print("OUTPUT:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

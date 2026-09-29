#!/usr/bin/env python3
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

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from federated_iot_v26 import CLASS_NAMES, NUM_CLASSES, set_seed, sqrt_class_weights, train_local_model
from lfighter_original_v34 import aggregate_original_lfighter
from neural_models_v24 import build_model
from run_frozen_untargeted_defense_v320b1 import (
    ATTACK_TYPES,
    DEFAULT_MALICIOUS_CLIENTS,
    load_exact_v320a3_poison_plan,
    parse_client_ids,
)
from run_reviewer_p4p_matched_v4322 import class_confusion_rows, load_train_val_only
from run_targeted_label_flip_v292 import load_fixed_partitions, read_clean_partition_hash
from transition_signature_features_v38 import evaluate_validation
from trusted_update_reconstruction_v312 import (
    checkpoint_sha256,
    floating_update,
    state_max_abs_difference,
    update_norm,
)

EXPERIMENT_VERSION = "4.33.6-ORIGINAL-LFIGHTER"
KMEANS_SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matched Original LFighter reviewer comparator")
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


def main() -> int:
    a = parse_args()

    if a.mode == "strong_attack" and a.plain_branch_dir is None:
        raise ValueError("--plain-branch-dir is required for strong_attack")
    if a.mode == "strong_attack" and len(parse_client_ids(a.malicious_clients)) != 8:
        raise ValueError("Primary matched comparison requires eight malicious clients")
    if a.num_clients != 20:
        raise ValueError("Primary matched Original LFighter comparison requires 20 clients")
    if a.continuation_rounds != 4:
        raise ValueError("Primary matched Original LFighter comparison requires four continuation rounds")

    out = a.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory already exists and is non-empty: {out}")

    tables = out / "tables"
    ckpts = out / "checkpoints" / "reviewer_round_checkpoints"
    attackdir = out / "attack_manifest"
    for p in (tables, ckpts, attackdir):
        p.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, a.threads))
    set_seed(a.model_seed)

    X_train, y_train, X_val, y_val = load_train_val_only(a.data_file)
    client_indices, partition_hash = load_fixed_partitions(
        a.partition_file.expanduser().resolve(),
        expected_clients=a.num_clients,
        train_rows=len(y_train),
    )
    clean_hash = read_clean_partition_hash(a.clean_seed_dir.expanduser().resolve())
    if clean_hash != partition_hash:
        raise RuntimeError("Clean partition hash mismatch")

    wdir = a.warmup_dir.expanduser().resolve()
    warmup_meta = json.loads(
        (wdir / "true_warmup_v310_metadata.json").read_text(encoding="utf-8")
    )
    w4_path = wdir / "checkpoints" / "common_round4_warmup_model.pt"
    w4 = torch.load(w4_path, map_location="cpu", weights_only=False)

    if int(warmup_meta["model_seed"]) != a.model_seed:
        raise RuntimeError("Warmup seed mismatch")
    if warmup_meta["partition_hash_sha256"] != partition_hash:
        raise RuntimeError("Warmup partition mismatch")

    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    model.load_state_dict(w4["model_state_dict"])
    if state_max_abs_difference(model.state_dict(), w4["model_state_dict"]) != 0.0:
        raise RuntimeError("Failed to load frozen W4 exactly")

    if a.mode == "strong_attack":
        requested = parse_client_ids(a.malicious_clients)
        (
            malicious_clients,
            poisoned_positions,
            poisoned_labels,
            _poison_manifest,
            poison_hash,
        ) = load_exact_v320a3_poison_plan(
            plain_branch_dir=a.plain_branch_dir,
            client_indices=client_indices,
            attack_type=a.attack_type,
            model_seed=a.model_seed,
            expected_malicious_clients=requested,
        )
        src = a.plain_branch_dir.expanduser().resolve() / "attack_manifest"
        for name in (
            "poisoned_indices.npz",
            "poisoned_labels.npz",
            "malicious_client_poison_manifest.csv",
        ):
            if (src / name).exists():
                shutil.copy2(src / name, attackdir / name)
    else:
        malicious_clients = []
        poisoned_positions = {
            i: np.empty(0, dtype=np.int64) for i in range(a.num_clients)
        }
        poisoned_labels = {
            i: np.empty(0, dtype=np.int64) for i in range(a.num_clients)
        }
        poison_hash = "clean_no_poison"

    class_weights = sqrt_class_weights(y_train, a.max_class_weight)
    source_id = CLASS_NAMES.index(a.source_class)
    target_id = CLASS_NAMES.index(a.target_class)

    round_rows: List[Dict[str, object]] = []
    client_rows: List[Dict[str, object]] = []
    salience_rows: List[Dict[str, object]] = []
    cluster_rows: List[Dict[str, object]] = []
    class_rows: List[Dict[str, object]] = []
    confusion_rows: List[Dict[str, object]] = []

    started = time.perf_counter()

    for monitoring_round in range(1, a.continuation_rounds + 1):
        global_round = 4 + monitoring_round
        round_started = time.perf_counter()
        reference_state = copy.deepcopy(model.state_dict())

        t0 = time.perf_counter()
        local_states: List[Mapping[str, torch.Tensor]] = []
        sample_counts: List[int] = []

        for client_id in range(a.num_clients):
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
                batch_size=a.batch_size,
                learning_rate=a.learning_rate,
                weight_decay=a.weight_decay,
                gradient_clip_norm=a.gradient_clip_norm,
                seed=a.model_seed + global_round * 1000 + client_id,
            )
            local_states.append(state)
            sample_counts.append(int(len(indices)))
            del local_model

        local_train_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        decision = aggregate_original_lfighter(
            reference_state=reference_state,
            local_states=local_states,
            client_ids=list(range(a.num_clients)),
            num_classes=NUM_CLASSES,
            class_names=CLASS_NAMES,
            malicious_client_ids=malicious_clients,
            kmeans_seed=KMEANS_SEED,
        )
        model.load_state_dict(decision.aggregated_state)
        aggregation_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        val_metrics, val_pair = evaluate_validation(
            model,
            X_val,
            y_val,
            source_id,
            target_id,
            a.evaluation_batch_size,
        )
        pc, cm = class_confusion_rows(
            model,
            X_val,
            y_val,
            a.evaluation_batch_size,
            monitoring_round,
            global_round,
            a.attack_type,
        )
        class_rows.extend(pc)
        confusion_rows.extend(cm)
        validation_seconds = time.perf_counter() - t0

        updates = [floating_update(st, reference_state) for st in local_states]

        ct = decision.client_decisions.copy()
        ct.insert(0, "global_round", global_round)
        ct.insert(0, "monitoring_round", monitoring_round)
        ct["client_samples"] = [sample_counts[int(cid)] for cid in ct["client_id"]]
        ct["poisoned_rows"] = [
            int(len(poisoned_positions[int(cid)])) for cid in ct["client_id"]
        ]
        ct["full_update_l2_norm"] = [
            float(update_norm(updates[int(cid)])) for cid in ct["client_id"]
        ]
        client_rows.extend(ct.to_dict(orient="records"))

        st = decision.class_salience.copy()
        st.insert(0, "global_round", global_round)
        st.insert(0, "monitoring_round", monitoring_round)
        salience_rows.extend(st.to_dict(orient="records"))

        cl = decision.cluster_diagnostics.copy()
        cl.insert(0, "global_round", global_round)
        cl.insert(0, "monitoring_round", monitoring_round)
        cluster_rows.extend(cl.to_dict(orient="records"))

        summary = decision.summary
        total_round_seconds = time.perf_counter() - round_started
        round_rows.append(
            {
                "monitoring_round": monitoring_round,
                "global_round": global_round,
                "mode": a.mode,
                "attack_type": a.attack_type,
                "arm": "original_lfighter",
                "kmeans_seed": KMEANS_SEED,
                "kmeans_clusters": 2,
                "kmeans_n_init": 10,
                "selected_class_1_id": int(summary["selected_class_1_id"]),
                "selected_class_1_name": str(summary["selected_class_1_name"]),
                "selected_class_2_id": int(summary["selected_class_2_id"]),
                "selected_class_2_name": str(summary["selected_class_2_name"]),
                "selected_class_pair": str(summary["selected_class_pair"]),
                "selected_good_cluster": int(summary["good_cluster"]),
                "admitted_client_count": int(summary["admitted_client_count"]),
                "rejected_client_count": int(summary["rejected_client_count"]),
                "malicious_rejection_recall": float(summary["malicious_rejection_recall"]),
                "benign_false_rejection_rate": float(summary["benign_false_rejection_rate"]),
                "malicious_admission_rate": float(summary["malicious_admission_rate"]),
                "fallback_reason": str(summary["fallback_reason"]),
                "phase_local_training_seconds": float(local_train_seconds),
                "phase_aggregation_seconds": float(aggregation_seconds),
                "phase_validation_seconds": float(validation_seconds),
                "round_seconds": float(total_round_seconds),
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"val_{k}": v for k, v in val_pair.items()},
            }
        )

        torch.save(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "arm": "original_lfighter",
                "monitoring_round": monitoring_round,
                "global_round": global_round,
                "model_seed": a.model_seed,
                "attack_type": a.attack_type,
                "model_state_dict": model.state_dict(),
                "previous_global_state_dict": reference_state,
                "partition_hash": partition_hash,
                "poison_index_hash": poison_hash,
                "aggregation": "original_lfighter_v34_equal_average_admitted_states",
                "kmeans_seed": KMEANS_SEED,
                "kmeans_clusters": 2,
                "kmeans_n_init": 10,
                "selected_class_pair": str(summary["selected_class_pair"]),
                "admitted_client_count": int(summary["admitted_client_count"]),
                "rejected_client_count": int(summary["rejected_client_count"]),
                "fallback_reason": str(summary["fallback_reason"]),
            },
            ckpts / f"global_round_{global_round:02d}_model.pt",
        )

        print(
            f"round {global_round}: macroF1={val_metrics['macro_f1']:.4f}, "
            f"pair={summary['selected_class_pair']}, "
            f"admitted={summary['admitted_client_count']}, "
            f"rejected={summary['rejected_client_count']}, "
            f"mal_recall={summary['malicious_rejection_recall']:.4f}, "
            f"benign_fpr={summary['benign_false_rejection_rate']:.4f}, "
            f"seconds={total_round_seconds:.1f}"
        )

    rdf = pd.DataFrame(round_rows)
    rdf.to_csv(tables / "original_lfighter_round_metrics.csv", index=False)
    pd.DataFrame(client_rows).to_csv(
        tables / "original_lfighter_client_decisions.csv", index=False
    )
    pd.DataFrame(salience_rows).to_csv(
        tables / "original_lfighter_class_salience.csv", index=False
    )
    pd.DataFrame(cluster_rows).to_csv(
        tables / "original_lfighter_cluster_diagnostics.csv", index=False
    )
    pd.DataFrame(class_rows).to_csv(
        tables / "validation_class_metrics_long.csv", index=False
    )
    pd.DataFrame(confusion_rows).to_csv(
        tables / "validation_confusion_matrix_long.csv", index=False
    )

    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "phase": "reviewer_matched_original_lfighter_comparator",
        "mode": a.mode,
        "attack_type": a.attack_type,
        "model_seed": int(a.model_seed),
        "num_clients": int(a.num_clients),
        "continuation_rounds": int(a.continuation_rounds),
        "malicious_clients": malicious_clients,
        "partition_hash_sha256": partition_hash,
        "poison_index_hash_sha256": poison_hash,
        "warmup_round4_checkpoint_sha256": checkpoint_sha256(w4_path),
        "aggregation": "original_lfighter_v34_equal_average_admitted_states",
        "source": "src/lfighter_original_v34.py",
        "kmeans_seed": KMEANS_SEED,
        "kmeans_clusters": 2,
        "kmeans_n_init": 10,
        "sample_count_weighting_used": False,
        "malicious_labels_used_for_decision": False,
        "malicious_labels_used_for_diagnostics_only": True,
        "test_sets_accessed": False,
        "attack_specific_retuning": False,
        "fallback_round_count": int((rdf["fallback_reason"].fillna("") != "").sum()),
        "total_seconds": float(time.perf_counter() - started),
        "mean_validation_macro_f1": float(rdf["val_macro_f1"].mean()),
        "mean_malicious_rejection_recall": float(
            rdf["malicious_rejection_recall"].mean()
        ),
        "mean_benign_false_rejection_rate": float(
            rdf["benign_false_rejection_rate"].mean()
        ),
    }
    (out / "REVIEWER_ORIGINAL_LFIGHTER_MATCHED_COMPLETE.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("ORIGINAL LFIGHTER MATCHED BRANCH COMPLETE")
    print("KMEANS SEED: 0")
    print("KMEANS CLUSTERS: 2")
    print("KMEANS N_INIT: 10")
    print("TEST SETS ACCESSED: False")
    print("ATTACK SPECIFIC RETUNING: False")
    print("MALICIOUS LABELS USED FOR DECISION: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

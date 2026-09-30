#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.metadata as importlib_metadata
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import hdbscan
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aggregation import FLAME
from federated_iot_v26 import CLASS_NAMES, NUM_CLASSES, set_seed, sqrt_class_weights, train_local_model
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
from trusted_update_reconstruction_v312 import checkpoint_sha256, state_max_abs_difference

EXPERIMENT_VERSION = "4.33.8-FLAME-MATCHED"
EXPECTED_HDBSCAN_VERSION = "0.8.44"
NOISE_SCALAR = 1.0
LAMBDA = 0.001
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_ALLOW_SINGLE_CLUSTER = True
NOISE_SEED_OFFSET = 999


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reviewer-matched preserved-project FLAME comparator")
    p.add_argument("--mode", choices=["strong_attack"], required=True)
    p.add_argument("--attack-type", choices=ATTACK_TYPES, required=True)
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--partition-file", type=Path, required=True)
    p.add_argument("--clean-seed-dir", type=Path, required=True)
    p.add_argument("--warmup-dir", type=Path, required=True)
    p.add_argument("--plain-branch-dir", type=Path, required=True)
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


def flame_diagnostics(global_model, local_models, malicious_clients):
    m = len(local_models)
    g_m = np.array([
        torch.nn.utils.parameters_to_vector(global_model.parameters()).detach().cpu().numpy()
    ])
    f_m = np.array([
        torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()
        for model in local_models
    ])
    grads = g_m - f_m
    cs = cosine_similarity(grads)
    min_cluster_size = int(m * 0.5) + 1
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=HDBSCAN_MIN_SAMPLES,
        allow_single_cluster=HDBSCAN_ALLOW_SINGLE_CLUSTER,
    )
    labels = clusterer.fit_predict(cs)

    all_outliers = bool(np.sum(labels) == -m)
    admitted_mask = np.ones(m, dtype=bool) if all_outliers else (labels != -1)

    euc_d = cdist(g_m, f_m)[0]
    st = float(np.median(euc_d))
    gammas = np.full(m, np.nan, dtype=float)
    for idx in np.where(admitted_mask)[0]:
        d = float(euc_d[idx])
        if d == 0.0:
            raw = np.inf if st > 0.0 else np.nan
        else:
            raw = st / d
        gamma = min(1.0, float(raw)) if not np.isnan(raw) else 1.0
        gammas[idx] = gamma

    malicious_set = {int(x) for x in malicious_clients}
    actual_malicious = np.asarray([i in malicious_set for i in range(m)], dtype=bool)
    rejected = ~admitted_mask
    mal_count = max(int(actual_malicious.sum()), 1)
    benign_count = max(int((~actual_malicious).sum()), 1)

    return {
        "labels": labels.astype(int),
        "admitted_mask": admitted_mask,
        "euc_d": euc_d.astype(float),
        "st": st,
        "gammas": gammas,
        "cosine_similarity": cs.astype(float),
        "min_cluster_size": min_cluster_size,
        "all_outliers_fallback": all_outliers,
        "malicious_rejection_recall": float(np.sum(rejected & actual_malicious) / mal_count),
        "benign_false_rejection_rate": float(np.sum(rejected & (~actual_malicious)) / benign_count),
        "malicious_admission_rate": float(np.sum(admitted_mask & actual_malicious) / mal_count),
    }


def main() -> int:
    a = parse_args()
    observed_hdbscan = importlib_metadata.version("hdbscan")
    if observed_hdbscan != EXPECTED_HDBSCAN_VERSION:
        raise RuntimeError(f"HDBSCAN version mismatch: expected {EXPECTED_HDBSCAN_VERSION}, observed {observed_hdbscan}")
    if a.num_clients != 20:
        raise ValueError("Matched FLAME preflight requires exactly 20 clients")
    if a.continuation_rounds != 4:
        raise ValueError("Matched FLAME preflight requires exactly four continuation rounds")

    requested_malicious = parse_client_ids(a.malicious_clients)
    if len(requested_malicious) != 8:
        raise ValueError("Primary matched comparison requires eight malicious clients")

    out = a.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory already exists and is non-empty: {out}")
    tables = out / "tables"
    ckpts = out / "checkpoints" / "reviewer_round_checkpoints"
    for p in (tables, ckpts):
        p.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, a.threads))
    set_seed(a.model_seed)

    X_train, y_train, X_val, y_val = load_train_val_only(a.data_file)
    client_indices, partition_hash = load_fixed_partitions(
        a.partition_file.expanduser().resolve(), expected_clients=a.num_clients, train_rows=len(y_train)
    )
    clean_hash = read_clean_partition_hash(a.clean_seed_dir.expanduser().resolve())
    if clean_hash != partition_hash:
        raise RuntimeError("Clean partition hash mismatch")

    wdir = a.warmup_dir.expanduser().resolve()
    warmup_meta = json.loads((wdir / "true_warmup_v310_metadata.json").read_text(encoding="utf-8"))
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

    malicious_clients, poisoned_positions, poisoned_labels, _poison_manifest, poison_hash = load_exact_v320a3_poison_plan(
        plain_branch_dir=a.plain_branch_dir,
        client_indices=client_indices,
        attack_type=a.attack_type,
        model_seed=a.model_seed,
        expected_malicious_clients=requested_malicious,
    )

    class_weights = sqrt_class_weights(y_train, a.max_class_weight)
    source_id = CLASS_NAMES.index(a.source_class)
    target_id = CLASS_NAMES.index(a.target_class)

    round_rows: List[Dict[str, object]] = []
    client_rows: List[Dict[str, object]] = []
    class_rows: List[Dict[str, object]] = []
    confusion_rows: List[Dict[str, object]] = []
    similarity_rows: List[Dict[str, object]] = []
    started = time.perf_counter()

    for monitoring_round in range(1, a.continuation_rounds + 1):
        global_round = 4 + monitoring_round
        round_started = time.perf_counter()
        reference_state = copy.deepcopy(model.state_dict())

        local_models = []
        sample_counts = []
        t0 = time.perf_counter()
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
            train_local_model(
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
            local_models.append(local_model)
            sample_counts.append(int(len(indices)))
        local_train_seconds = time.perf_counter() - t0

        global_for_flame = copy.deepcopy(model).cpu()
        diag = flame_diagnostics(global_for_flame, local_models, malicious_clients)

        noise_seed = int(a.model_seed + global_round * 1000 + NOISE_SEED_OFFSET)
        torch.manual_seed(noise_seed)
        t0 = time.perf_counter()
        aggregated_state = FLAME(copy.deepcopy(global_for_flame), copy.deepcopy(local_models), NOISE_SCALAR)
        aggregation_seconds = time.perf_counter() - t0

        if not all(torch.isfinite(t).all().item() for t in aggregated_state.values()):
            raise FloatingPointError("FLAME produced non-finite aggregated parameters; preserve partial output and stop")
        model.load_state_dict(aggregated_state)

        t0 = time.perf_counter()
        val_metrics, val_pair = evaluate_validation(model, X_val, y_val, source_id, target_id, a.evaluation_batch_size)
        pc, cm = class_confusion_rows(
            model, X_val, y_val, a.evaluation_batch_size, monitoring_round, global_round, a.attack_type
        )
        class_rows.extend(pc)
        confusion_rows.extend(cm)
        validation_seconds = time.perf_counter() - t0

        malicious_set = set(map(int, malicious_clients))
        for cid in range(a.num_clients):
            gamma = diag["gammas"][cid]
            client_rows.append({
                "monitoring_round": monitoring_round,
                "global_round": global_round,
                "client_id": cid,
                "actual_malicious": cid in malicious_set,
                "hdbscan_label": int(diag["labels"][cid]),
                "admitted": bool(diag["admitted_mask"][cid]),
                "rejected": bool(not diag["admitted_mask"][cid]),
                "euclidean_distance_to_global": float(diag["euc_d"][cid]),
                "clip_gamma": float(gamma) if np.isfinite(gamma) else np.nan,
                "client_samples": sample_counts[cid],
                "poisoned_rows": int(len(poisoned_positions[cid])),
            })

        cs = diag["cosine_similarity"]
        for i in range(a.num_clients):
            for j in range(a.num_clients):
                similarity_rows.append({
                    "monitoring_round": monitoring_round,
                    "global_round": global_round,
                    "client_i": i,
                    "client_j": j,
                    "cosine_similarity": float(cs[i, j]),
                })

        admitted_count = int(np.sum(diag["admitted_mask"]))
        rejected_count = a.num_clients - admitted_count
        sigma = float(LAMBDA * diag["st"] * NOISE_SCALAR)
        effective_noise_std = float(sigma ** 2)
        total_round_seconds = time.perf_counter() - round_started

        round_rows.append({
            "monitoring_round": monitoring_round,
            "global_round": global_round,
            "mode": a.mode,
            "attack_type": a.attack_type,
            "arm": "flame_preserved_project",
            "hdbscan_version": EXPECTED_HDBSCAN_VERSION,
            "hdbscan_min_cluster_size": int(diag["min_cluster_size"]),
            "hdbscan_min_samples": HDBSCAN_MIN_SAMPLES,
            "hdbscan_allow_single_cluster": HDBSCAN_ALLOW_SINGLE_CLUSTER,
            "all_outliers_fallback": bool(diag["all_outliers_fallback"]),
            "admitted_client_count": admitted_count,
            "rejected_client_count": rejected_count,
            "median_euclidean_distance_st": float(diag["st"]),
            "lambda": LAMBDA,
            "noise_scalar": NOISE_SCALAR,
            "sigma": sigma,
            "effective_noise_std_passed_to_normal": effective_noise_std,
            "flame_noise_seed": noise_seed,
            "malicious_rejection_recall": float(diag["malicious_rejection_recall"]),
            "benign_false_rejection_rate": float(diag["benign_false_rejection_rate"]),
            "malicious_admission_rate": float(diag["malicious_admission_rate"]),
            "phase_local_training_seconds": float(local_train_seconds),
            "phase_aggregation_seconds": float(aggregation_seconds),
            "phase_validation_seconds": float(validation_seconds),
            "round_seconds": float(total_round_seconds),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"val_{k}": v for k, v in val_pair.items()},
        })

        torch.save({
            "experiment_version": EXPERIMENT_VERSION,
            "arm": "flame_preserved_project",
            "monitoring_round": monitoring_round,
            "global_round": global_round,
            "model_seed": a.model_seed,
            "attack_type": a.attack_type,
            "model_state_dict": model.state_dict(),
            "previous_global_state_dict": reference_state,
            "partition_hash": partition_hash,
            "poison_index_hash": poison_hash,
            "hdbscan_version": EXPECTED_HDBSCAN_VERSION,
            "hdbscan_min_cluster_size": int(diag["min_cluster_size"]),
            "hdbscan_min_samples": HDBSCAN_MIN_SAMPLES,
            "hdbscan_allow_single_cluster": HDBSCAN_ALLOW_SINGLE_CLUSTER,
            "lambda": LAMBDA,
            "noise_scalar": NOISE_SCALAR,
            "flame_noise_seed": noise_seed,
            "effective_noise_std_passed_to_normal": effective_noise_std,
        }, ckpts / f"global_round_{global_round:02d}_model.pt")

        print(
            f"round {global_round}: macroF1={val_metrics['macro_f1']:.4f}, "
            f"admitted={admitted_count}, rejected={rejected_count}, "
            f"mal_recall={diag['malicious_rejection_recall']:.4f}, "
            f"benign_fpr={diag['benign_false_rejection_rate']:.4f}, "
            f"st={diag['st']:.6g}, noise_std={effective_noise_std:.6g}, "
            f"seconds={total_round_seconds:.1f}"
        )
        del local_models

    rdf = pd.DataFrame(round_rows)
    rdf.to_csv(tables / "flame_round_metrics.csv", index=False)
    pd.DataFrame(client_rows).to_csv(tables / "flame_client_decisions.csv", index=False)
    pd.DataFrame(similarity_rows).to_csv(tables / "flame_cosine_similarity_long.csv", index=False)
    pd.DataFrame(class_rows).to_csv(tables / "validation_class_metrics_long.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(tables / "validation_confusion_matrix_long.csv", index=False)

    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "phase": "reviewer_matched_flame_comparator",
        "mode": a.mode,
        "attack_type": a.attack_type,
        "model_seed": int(a.model_seed),
        "num_clients": int(a.num_clients),
        "continuation_rounds": int(a.continuation_rounds),
        "malicious_clients": list(map(int, malicious_clients)),
        "partition_hash_sha256": partition_hash,
        "poison_index_hash_sha256": poison_hash,
        "warmup_round4_checkpoint_sha256": checkpoint_sha256(w4_path),
        "aggregation_source": "src/aggregation.py::FLAME",
        "aggregation_source_sha256_expected": "c964742e0279ef98cc3b3d0ce36c62282af1be9c7b94d3b8b484240085333f0d",
        "hdbscan_version": EXPECTED_HDBSCAN_VERSION,
        "hdbscan_version_is_reviewer_frozen_not_recovered_historical": True,
        "hdbscan_min_cluster_size": 11,
        "hdbscan_min_samples": HDBSCAN_MIN_SAMPLES,
        "hdbscan_allow_single_cluster": HDBSCAN_ALLOW_SINGLE_CLUSTER,
        "lambda": LAMBDA,
        "noise_scalar": NOISE_SCALAR,
        "noise_scalar_adaptation_used": False,
        "preserved_source_uses_normal_std_sigma_squared": True,
        "noise_seed_offset": NOISE_SEED_OFFSET,
        "sample_count_weighting_used": False,
        "malicious_labels_used_for_aggregation": False,
        "malicious_labels_used_for_diagnostics_only": True,
        "test_sets_accessed": False,
        "attack_specific_retuning": False,
        "scientific_outcome_gate_used": False,
        "all_outliers_fallback_round_count": int(rdf["all_outliers_fallback"].astype(bool).sum()),
        "total_seconds": float(time.perf_counter() - started),
        "mean_validation_macro_f1": float(rdf["val_macro_f1"].mean()),
        "mean_malicious_rejection_recall": float(rdf["malicious_rejection_recall"].mean()),
        "mean_benign_false_rejection_rate": float(rdf["benign_false_rejection_rate"].mean()),
    }
    (out / "REVIEWER_FLAME_MATCHED_COMPLETE.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("FLAME MATCHED BRANCH COMPLETE")
    print("HDBSCAN VERSION:", EXPECTED_HDBSCAN_VERSION)
    print("NOISE SCALAR:", NOISE_SCALAR)
    print("LAMBDA:", LAMBDA)
    print("TEST SETS ACCESSED: False")
    print("ATTACK SPECIFIC RETUNING: False")
    print("SCIENTIFIC OUTCOME GATE USED: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

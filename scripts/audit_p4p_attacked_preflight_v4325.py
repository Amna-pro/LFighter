#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "results" / "reviewer_task65_attack_manifest_recovery_v4324"
    / "all_to_one_benign" / "seed_1379954285" / "plain_fedavg"
)
OUT = (
    ROOT / "results" / "reviewer_p4p_attacked_preflight_v4325"
    / "all_to_one_benign" / "seed_1379954285"
)
EVID = (
    ROOT / "reviewer_revision"
    / "P4P_ATTACKED_PREFLIGHT_SEED_1379954285_ALL_TO_ONE_BENIGN_AUDIT_v4325.json"
)

EXPECTED_PARTITION = "5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"
EXPECTED_POISON = "9690f8d1e7b9aa56cbb99dc519bcad9274d23a84f49bd6bb052737a285b2855e"
EXPECTED_WARMUP = "eb19aa119d7986c62856b5734255d98e15b29a40c2d9f20bf4152711ba8a85ca"
EXPECTED_MALICIOUS = [1, 7, 8, 10, 14, 15, 17, 18]
EXPECTED_P4P = {
    "mad_k": 3.0,
    "dbscan_eps": 0.5,
    "dbscan_min_samples": 5,
    "isolation_contamination": 0.1,
    "isolation_n_estimators": 100,
    "kmeans_clusters": 2,
    "ensemble_vote_threshold": 2,
    "suspicion_decay": 0.2,
    "suspicion_threshold": 2.0,
    "detector_random_state": 42,
    "kmeans_n_init": 10,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_npz_exact(a: Path, b: Path) -> dict:
    with np.load(a, allow_pickle=False) as aa, np.load(b, allow_pickle=False) as bb:
        ka = sorted(aa.files)
        kb = sorted(bb.files)
        if ka != kb:
            raise RuntimeError(f"NPZ key mismatch: {a} vs {b}")
        for k in ka:
            if not np.array_equal(np.asarray(aa[k]), np.asarray(bb[k])):
                raise RuntimeError(f"NPZ array mismatch for key {k}: {a} vs {b}")
    return {"keys": ka, "exact_array_equality": True}


def compare_manifest_csv(a: Path, b: Path) -> dict:
    aa = pd.read_csv(a)
    bb = pd.read_csv(b)
    if list(aa.columns) != list(bb.columns):
        raise RuntimeError("Manifest CSV columns differ")

    float_cols = [
        "client_poison_rate_over_eligible",
        "client_poison_rate_over_all_rows",
    ]
    exact_cols = [c for c in aa.columns if c not in float_cols]

    pd.testing.assert_frame_equal(
        aa[exact_cols],
        bb[exact_cols],
        check_exact=True,
        check_dtype=False,
    )
    for c in float_cols:
        np.testing.assert_allclose(
            aa[c].to_numpy(dtype=float),
            bb[c].to_numpy(dtype=float),
            rtol=0.0,
            atol=5e-15,
            equal_nan=True,
        )
    return {
        "discrete_columns_exact": True,
        "float_columns_atol": 5e-15,
        "float_columns_rtol": 0.0,
    }


def main() -> int:
    if EVID.exists():
        raise FileExistsError(f"Audit evidence already exists: {EVID}")
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not OUT.exists():
        raise FileNotFoundError(OUT)

    complete_path = OUT / "REVIEWER_P4P_MATCHED_COMPLETE.json"
    round_path = OUT / "tables" / "p4p_round_metrics.csv"
    decision_path = OUT / "tables" / "p4p_client_decisions.csv"

    meta = json.loads(complete_path.read_text(encoding="utf-8"))
    expected_meta = {
        "mode": meta.get("mode") == "strong_attack",
        "attack_type": meta.get("attack_type") == "all_to_one_benign",
        "model_seed": int(meta.get("model_seed", -1)) == 1379954285,
        "num_clients": int(meta.get("num_clients", -1)) == 20,
        "continuation_rounds": int(meta.get("continuation_rounds", -1)) == 4,
        "malicious_clients": meta.get("malicious_clients") == EXPECTED_MALICIOUS,
        "partition_hash": meta.get("partition_hash_sha256") == EXPECTED_PARTITION,
        "poison_hash": meta.get("poison_index_hash_sha256") == EXPECTED_POISON,
        "warmup_hash": meta.get("warmup_round4_checkpoint_sha256") == EXPECTED_WARMUP,
        "p4p_config": meta.get("p4p_config") == EXPECTED_P4P,
        "aggregation": meta.get("aggregation") == "sample_count_weighted_fedavg_over_trusted_set",
        "test_sets_accessed": meta.get("test_sets_accessed") is False,
        "attack_specific_retuning": meta.get("attack_specific_retuning") is False,
    }
    failed = [k for k, v in expected_meta.items() if not v]
    if failed:
        raise RuntimeError(f"P4P completion metadata validation failed: {failed}")

    source_attack = SOURCE / "attack_manifest"
    out_attack = OUT / "attack_manifest"
    manifest_cmp = compare_manifest_csv(
        source_attack / "malicious_client_poison_manifest.csv",
        out_attack / "malicious_client_poison_manifest.csv",
    )
    idx_cmp = compare_npz_exact(
        source_attack / "poisoned_indices.npz",
        out_attack / "poisoned_indices.npz",
    )
    labels_cmp = compare_npz_exact(
        source_attack / "poisoned_labels.npz",
        out_attack / "poisoned_labels.npz",
    )

    rounds = pd.read_csv(round_path)
    if rounds.shape[0] != 4:
        raise RuntimeError(f"Expected 4 P4P round rows, got {rounds.shape[0]}")
    if rounds["monitoring_round"].tolist() != [1, 2, 3, 4]:
        raise RuntimeError("Unexpected monitoring round sequence")
    if rounds["global_round"].tolist() != [5, 6, 7, 8]:
        raise RuntimeError("Unexpected global round sequence")
    if set(rounds["mode"]) != {"strong_attack"}:
        raise RuntimeError("Unexpected mode in round metrics")
    if set(rounds["attack_type"]) != {"all_to_one_benign"}:
        raise RuntimeError("Unexpected attack type in round metrics")
    if set(rounds["arm"]) != {"p4p_matched"}:
        raise RuntimeError("Unexpected arm in round metrics")

    calc_mean_macro = float(rounds["val_macro_f1"].mean())
    calc_mean_mal = float(rounds["malicious_recall"].mean())
    calc_mean_fpr = float(rounds["benign_false_positive_rate"].mean())
    np.testing.assert_allclose(calc_mean_macro, float(meta["mean_validation_macro_f1"]), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(calc_mean_mal, float(meta["mean_malicious_recall"]), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(calc_mean_fpr, float(meta["mean_benign_fpr"]), rtol=0.0, atol=1e-15)

    decisions = pd.read_csv(decision_path)
    if decisions.shape[0] != 80:
        raise RuntimeError(f"Expected 80 client decision rows, got {decisions.shape[0]}")
    for mr in [1, 2, 3, 4]:
        sub = decisions.loc[decisions["monitoring_round"] == mr]
        if sorted(sub["client_id"].astype(int).tolist()) != list(range(20)):
            raise RuntimeError(f"Round {mr} client coverage is not exactly 0..19")
        observed_mal = sorted(sub.loc[sub["actual_malicious"].astype(bool), "client_id"].astype(int).tolist())
        if observed_mal != EXPECTED_MALICIOUS:
            raise RuntimeError(f"Round {mr} malicious membership mismatch: {observed_mal}")

    output_files = {}
    for p in sorted(x for x in OUT.rglob("*") if x.is_file()):
        rel = p.relative_to(OUT).as_posix()
        output_files[rel] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}

    evidence = {
        "protocol": "reviewer_v4325_p4p_attacked_preflight_audit",
        "status": "PASS",
        "condition": {
            "attack_type": "all_to_one_benign",
            "model_seed": 1379954285,
            "num_clients": 20,
            "malicious_clients": EXPECTED_MALICIOUS,
            "monitoring_global_rounds": [5, 6, 7, 8],
        },
        "frozen_identity": {
            "partition_hash_sha256": EXPECTED_PARTITION,
            "poison_index_hash_sha256": EXPECTED_POISON,
            "warmup_round4_checkpoint_sha256": EXPECTED_WARMUP,
            "p4p_config": EXPECTED_P4P,
        },
        "attack_manifest_reuse_validation": {
            "manifest_csv": manifest_cmp,
            "poisoned_indices": idx_cmp,
            "poisoned_labels": labels_cmp,
        },
        "runner_metadata": {
            "test_sets_accessed": False,
            "attack_specific_retuning": False,
            "aggregation": meta["aggregation"],
            "total_seconds": float(meta["total_seconds"]),
        },
        "observed_preflight_results": {
            "mean_validation_macro_f1": calc_mean_macro,
            "mean_malicious_recall": calc_mean_mal,
            "mean_benign_fpr": calc_mean_fpr,
            "round_malicious_recall": [float(x) for x in rounds["malicious_recall"]],
            "round_benign_fpr": [float(x) for x in rounds["benign_false_positive_rate"]],
            "round_validation_macro_f1": [float(x) for x in rounds["val_macro_f1"]],
            "round_source_to_target_rate": [float(x) for x in rounds["val_source_to_target_rate"]],
            "round_source_recall": [float(x) for x in rounds["val_source_recall"]],
        },
        "client_decision_rows": int(decisions.shape[0]),
        "output_files": output_files,
        "scientific_parameters_changed_after_outcome": False,
        "outcome_retuning_performed": False,
    }
    EVID.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    print("FIRST MATCHED ATTACKED P4P PREFLIGHT AUDIT = PASS")
    print("ATTACK MANIFEST POSITIONS EXACT: True")
    print("ATTACK MANIFEST LABELS EXACT: True")
    print("PARTITION HASH MATCH: True")
    print("POISON INDEX HASH MATCH: True")
    print("WARMUP CHECKPOINT HASH MATCH: True")
    print("P4P CONFIG MATCH: True")
    print("TEST SETS ACCESSED ACCORDING TO RUNNER METADATA: False")
    print("ATTACK SPECIFIC RETUNING: False")
    print("MEAN VALIDATION MACRO F1:", calc_mean_macro)
    print("MEAN MALICIOUS RECALL:", calc_mean_mal)
    print("MEAN BENIGN FPR:", calc_mean_fpr)
    print("EVIDENCE:", EVID)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

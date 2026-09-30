#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results"/"reviewer_flame_attacked_preflight_v4338"/"all_to_one_benign"/"seed_1379954285"

META = OUT/"REVIEWER_FLAME_MATCHED_COMPLETE.json"
ROUND = OUT/"tables"/"flame_round_metrics.csv"
CLIENT = OUT/"tables"/"flame_client_decisions.csv"
SIM = OUT/"tables"/"flame_cosine_similarity_long.csv"
CLASS = OUT/"tables"/"validation_class_metrics_long.csv"
CONF = OUT/"tables"/"validation_confusion_matrix_long.csv"
CKPT = OUT/"checkpoints"/"reviewer_round_checkpoints"/"global_round_08_model.pt"
MANIFEST = ROOT/"reviewer_revision"/"TASK65_ATTACK_MANIFEST_RECOVERY_SEED_1379954285_ALL_TO_ONE_BENIGN_v4324.json"
AUDIT = ROOT/"reviewer_revision"/"FLAME_ATTACKED_PREFLIGHT_AUDIT_v4338.json"

PH = "5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"
EXPECTED_HDBSCAN = "0.8.44"
EXPECTED_AGG_SHA = "c964742e0279ef98cc3b3d0ce36c62282af1be9c7b94d3b8b484240085333f0d"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    for p in (META, ROUND, CLIENT, SIM, CLASS, CONF, CKPT, MANIFEST):
        if not p.exists():
            raise FileNotFoundError(p)

    agg = ROOT/"src"/"aggregation.py"
    if sha256_file(agg) != EXPECTED_AGG_SHA:
        raise RuntimeError("aggregation.py source hash changed")
    if importlib_metadata.version("hdbscan") != EXPECTED_HDBSCAN:
        raise RuntimeError("HDBSCAN version changed")

    meta = json.loads(META.read_text(encoding="utf-8"))
    recovery = json.loads(MANIFEST.read_text(encoding="utf-8"))
    r = pd.read_csv(ROUND)
    c = pd.read_csv(CLIENT)
    sim = pd.read_csv(SIM)
    pc = pd.read_csv(CLASS)
    cm = pd.read_csv(CONF)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    poison = recovery["poison_index_hash_sha256"]

    expected_noise_seeds = [1379954285 + gr * 1000 + 999 for gr in [5, 6, 7, 8]]

    checks = {
        "metadata_mode_strong_attack": meta.get("mode") == "strong_attack",
        "metadata_attack_exact": meta.get("attack_type") == "all_to_one_benign",
        "metadata_seed_exact": int(meta.get("model_seed", -1)) == 1379954285,
        "metadata_clients_20": int(meta.get("num_clients", -1)) == 20,
        "metadata_rounds_4": int(meta.get("continuation_rounds", -1)) == 4,
        "partition_hash_exact": meta.get("partition_hash_sha256") == PH,
        "poison_hash_matches_recovery": meta.get("poison_index_hash_sha256") == poison,
        "aggregation_source_exact": meta.get("aggregation_source") == "src/aggregation.py::FLAME",
        "aggregation_source_hash_exact": meta.get("aggregation_source_sha256_expected") == EXPECTED_AGG_SHA,
        "runtime_aggregation_source_hash_exact": sha256_file(agg) == EXPECTED_AGG_SHA,
        "hdbscan_version_exact": meta.get("hdbscan_version") == EXPECTED_HDBSCAN,
        "runtime_hdbscan_version_exact": importlib_metadata.version("hdbscan") == EXPECTED_HDBSCAN,
        "hdbscan_reviewer_frozen_flag_true": meta.get("hdbscan_version_is_reviewer_frozen_not_recovered_historical") is True,
        "min_cluster_size_11": int(meta.get("hdbscan_min_cluster_size", -1)) == 11,
        "min_samples_1": int(meta.get("hdbscan_min_samples", -1)) == 1,
        "allow_single_cluster_true": meta.get("hdbscan_allow_single_cluster") is True,
        "lambda_exact": float(meta.get("lambda", -1)) == 0.001,
        "noise_scalar_exact": float(meta.get("noise_scalar", -1)) == 1.0,
        "noise_adaptation_false": meta.get("noise_scalar_adaptation_used") is False,
        "sigma_squared_std_preserved": meta.get("preserved_source_uses_normal_std_sigma_squared") is True,
        "noise_seed_offset_999": int(meta.get("noise_seed_offset", -1)) == 999,
        "sample_weighting_false": meta.get("sample_count_weighting_used") is False,
        "malicious_labels_aggregation_false": meta.get("malicious_labels_used_for_aggregation") is False,
        "malicious_labels_diagnostics_true": meta.get("malicious_labels_used_for_diagnostics_only") is True,
        "test_access_false": meta.get("test_sets_accessed") is False,
        "attack_retuning_false": meta.get("attack_specific_retuning") is False,
        "scientific_outcome_gate_false": meta.get("scientific_outcome_gate_used") is False,
        "round_rows_4": len(r) == 4,
        "rounds_exact_5_to_8": list(r["global_round"].astype(int)) == [5, 6, 7, 8],
        "arm_exact": set(r["arm"]) == {"flame_preserved_project"},
        "round_hdbscan_version_exact": set(r["hdbscan_version"].astype(str)) == {EXPECTED_HDBSCAN},
        "round_min_cluster_size_11": set(r["hdbscan_min_cluster_size"].astype(int)) == {11},
        "round_min_samples_1": set(r["hdbscan_min_samples"].astype(int)) == {1},
        "round_noise_scalar_1": set(r["noise_scalar"].astype(float)) == {1.0},
        "round_lambda_exact": set(r["lambda"].astype(float)) == {0.001},
        "noise_seed_schedule_exact": list(r["flame_noise_seed"].astype(int)) == expected_noise_seeds,
        "admitted_counts_valid": bool(((r["admitted_client_count"].astype(int) >= 1) & (r["admitted_client_count"].astype(int) <= 20)).all()),
        "rejected_counts_valid": bool(((r["rejected_client_count"].astype(int) >= 0) & (r["rejected_client_count"].astype(int) <= 19)).all()),
        "counts_sum_20": bool((r["admitted_client_count"].astype(int) + r["rejected_client_count"].astype(int) == 20).all()),
        "security_metrics_finite": bool(np.isfinite(r[["malicious_rejection_recall","benign_false_rejection_rate","malicious_admission_rate"]].to_numpy(float)).all()),
        "validation_metrics_finite": bool(np.isfinite(r[["val_macro_f1","val_balanced_accuracy"]].to_numpy(float)).all()),
        "st_finite_nonnegative": bool((np.isfinite(r["median_euclidean_distance_st"].to_numpy(float)) & (r["median_euclidean_distance_st"].to_numpy(float) >= 0)).all()),
        "noise_std_finite_nonnegative": bool((np.isfinite(r["effective_noise_std_passed_to_normal"].to_numpy(float)) & (r["effective_noise_std_passed_to_normal"].to_numpy(float) >= 0)).all()),
        "client_rows_80": len(c) == 80,
        "similarity_rows_1600": len(sim) == 1600,
        "validation_class_rows_32": len(pc) == 32,
        "confusion_rows_256": len(cm) == 256,
        "ckpt_arm_exact": ckpt.get("arm") == "flame_preserved_project",
        "ckpt_round_8": int(ckpt.get("global_round", -1)) == 8,
        "ckpt_seed_exact": int(ckpt.get("model_seed", -1)) == 1379954285,
        "ckpt_attack_exact": ckpt.get("attack_type") == "all_to_one_benign",
        "ckpt_partition_exact": ckpt.get("partition_hash") == PH,
        "ckpt_poison_exact": ckpt.get("poison_index_hash") == poison,
        "ckpt_hdbscan_version_exact": ckpt.get("hdbscan_version") == EXPECTED_HDBSCAN,
        "ckpt_noise_scalar_exact": float(ckpt.get("noise_scalar", -1)) == 1.0,
        "ckpt_lambda_exact": float(ckpt.get("lambda", -1)) == 0.001,
        "ckpt_noise_seed_exact": int(ckpt.get("flame_noise_seed", -1)) == expected_noise_seeds[-1],
        "model_state_present": isinstance(ckpt.get("model_state_dict"), dict),
    }

    bad = [name for name, value in checks.items() if not bool(value)]
    if bad:
        raise RuntimeError(f"FLAME preflight audit failed: {bad}")

    r8 = r.loc[r["global_round"].astype(int) == 8].iloc[0]
    obj = {
        "protocol": "reviewer_v4338_flame_attacked_preflight_audit",
        "status": "PASS",
        "checks": checks,
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "scientific_outcome_gate_used": False,
        "test_sets_accessed": False,
        "attack_specific_retuning": False,
        "hdbscan_version": EXPECTED_HDBSCAN,
        "hdbscan_version_is_reviewer_frozen_not_recovered_historical": True,
        "noise_scalar": 1.0,
        "lambda": 0.001,
        "fallback_round_count_observed": int(r["all_outliers_fallback"].astype(bool).sum()),
        "round8_admitted_client_count": int(r8["admitted_client_count"]),
        "round8_rejected_client_count": int(r8["rejected_client_count"]),
        "round8_malicious_rejection_recall": float(r8["malicious_rejection_recall"]),
        "round8_benign_false_rejection_rate": float(r8["benign_false_rejection_rate"]),
        "round8_validation_macro_f1": float(r8["val_macro_f1"]),
        "round8_median_distance_st": float(r8["median_euclidean_distance_st"]),
        "round8_effective_noise_std": float(r8["effective_noise_std_passed_to_normal"]),
        "output_sha256": {
            META.name: sha256_file(META),
            ROUND.name: sha256_file(ROUND),
            CLIENT.name: sha256_file(CLIENT),
            SIM.name: sha256_file(SIM),
            CLASS.name: sha256_file(CLASS),
            CONF.name: sha256_file(CONF),
            CKPT.name: sha256_file(CKPT),
        },
    }
    AUDIT.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

    print("FLAME ATTACKED PREFLIGHT AUDIT = PASS")
    print("CHECKS:", len(checks), "/", len(checks))
    print("HDBSCAN VERSION:", EXPECTED_HDBSCAN)
    print("HDBSCAN VERSION ORIGIN: reviewer-frozen, not recovered historical pin")
    print("NOISE SCALAR: 1.0")
    print("LAMBDA: 0.001")
    print("FALLBACK ROUNDS OBSERVED:", obj["fallback_round_count_observed"])
    print("ROUND8 ADMITTED:", obj["round8_admitted_client_count"])
    print("ROUND8 REJECTED:", obj["round8_rejected_client_count"])
    print("ROUND8 MALICIOUS REJECTION RECALL:", obj["round8_malicious_rejection_recall"])
    print("ROUND8 BENIGN FALSE REJECTION RATE:", obj["round8_benign_false_rejection_rate"])
    print("ROUND8 VALIDATION MACRO F1:", obj["round8_validation_macro_f1"])
    print("TEST SETS ACCESSED: False")
    print("ATTACK SPECIFIC RETUNING: False")
    print("SCIENTIFIC OUTCOME GATE USED: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SEEDS = [1379954285, 1886033230, 480705558, 1377035733, 1707771978]
ATTACKS = [
    "all_to_one_benign",
    "cyclic_shift",
    "multiclass_partial_cycle",
    "pairwise_swap",
    "random_flip",
]
MALICIOUS = [1, 7, 8, 10, 14, 15, 17, 18]

GRID_ROOT = ROOT / "results" / "reviewer_p4p_matched_grid_v4325"
PREFLIGHT = (
    ROOT / "results" / "reviewer_p4p_attacked_preflight_v4325"
    / "all_to_one_benign" / "seed_1379954285"
)

OUT_COND = ROOT / "reviewer_revision" / "P4P_MATCHED_GRID_CONDITION_SUMMARY_v4325.csv"
OUT_ROUND = ROOT / "reviewer_revision" / "P4P_MATCHED_GRID_ROUND_SUMMARY_v4325.csv"
OUT_JSON = ROOT / "reviewer_revision" / "P4P_MATCHED_GRID_FINAL_SUMMARY_v4325.json"

EXPECTED_PARTITION = "5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"
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


def output_dir(seed: int, attack: str) -> Path:
    if attack == "all_to_one_benign" and seed == 1379954285:
        return PREFLIGHT
    return GRID_ROOT / attack / f"seed_{seed}"


def manifest_evidence(seed: int, attack: str) -> Path:
    return (
        ROOT / "reviewer_revision"
        / f"TASK65_ATTACK_MANIFEST_RECOVERY_SEED_{seed}_{attack.upper()}_v4324.json"
    )


def compare_npz_exact(a: Path, b: Path) -> None:
    with np.load(a, allow_pickle=False) as aa, np.load(b, allow_pickle=False) as bb:
        if sorted(aa.files) != sorted(bb.files):
            raise RuntimeError(f"NPZ key mismatch: {a} vs {b}")
        for k in aa.files:
            if not np.array_equal(np.asarray(aa[k]), np.asarray(bb[k])):
                raise RuntimeError(f"NPZ array mismatch key={k}: {a} vs {b}")


def compare_manifest_csv(a: Path, b: Path) -> None:
    aa = pd.read_csv(a)
    bb = pd.read_csv(b)
    if list(aa.columns) != list(bb.columns):
        raise RuntimeError(f"Manifest columns differ: {a} vs {b}")
    float_cols = ["client_poison_rate_over_eligible", "client_poison_rate_over_all_rows"]
    exact_cols = [c for c in aa.columns if c not in float_cols]
    pd.testing.assert_frame_equal(
        aa[exact_cols], bb[exact_cols], check_exact=True, check_dtype=False
    )
    for c in float_cols:
        np.testing.assert_allclose(
            aa[c].to_numpy(dtype=float),
            bb[c].to_numpy(dtype=float),
            rtol=0.0,
            atol=5e-15,
            equal_nan=True,
        )


def main() -> int:
    cond_rows = []
    round_rows = []

    for attack in ATTACKS:
        for seed in SEEDS:
            out = output_dir(seed, attack)
            complete = out / "REVIEWER_P4P_MATCHED_COMPLETE.json"
            rounds_path = out / "tables" / "p4p_round_metrics.csv"
            decisions_path = out / "tables" / "p4p_client_decisions.csv"

            if not complete.exists():
                raise FileNotFoundError(complete)
            if not rounds_path.exists():
                raise FileNotFoundError(rounds_path)
            if not decisions_path.exists():
                raise FileNotFoundError(decisions_path)

            meta = json.loads(complete.read_text(encoding="utf-8"))
            mev = json.loads(manifest_evidence(seed, attack).read_text(encoding="utf-8"))
            expected_poison = mev["poison_index_hash_sha256"]

            checks = {
                "mode": meta.get("mode") == "strong_attack",
                "attack": meta.get("attack_type") == attack,
                "seed": int(meta.get("model_seed", -1)) == seed,
                "clients": int(meta.get("num_clients", -1)) == 20,
                "rounds": int(meta.get("continuation_rounds", -1)) == 4,
                "malicious": meta.get("malicious_clients") == MALICIOUS,
                "partition": meta.get("partition_hash_sha256") == EXPECTED_PARTITION,
                "poison": meta.get("poison_index_hash_sha256") == expected_poison,
                "p4p": meta.get("p4p_config") == EXPECTED_P4P,
                "test": meta.get("test_sets_accessed") is False,
                "retune": meta.get("attack_specific_retuning") is False,
            }
            failed = [k for k, v in checks.items() if not v]
            if failed:
                raise RuntimeError(f"Metadata validation failed attack={attack} seed={seed}: {failed}")

            source_manifest = (
                ROOT / "results" / "reviewer_task65_attack_manifest_recovery_v4324"
                / attack / f"seed_{seed}" / "plain_fedavg" / "attack_manifest"
            )
            out_manifest = out / "attack_manifest"
            compare_manifest_csv(
                source_manifest / "malicious_client_poison_manifest.csv",
                out_manifest / "malicious_client_poison_manifest.csv",
            )
            compare_npz_exact(
                source_manifest / "poisoned_indices.npz",
                out_manifest / "poisoned_indices.npz",
            )
            compare_npz_exact(
                source_manifest / "poisoned_labels.npz",
                out_manifest / "poisoned_labels.npz",
            )

            rounds = pd.read_csv(rounds_path)
            decisions = pd.read_csv(decisions_path)

            if rounds.shape[0] != 4:
                raise RuntimeError(f"Expected 4 round rows attack={attack} seed={seed}")
            if rounds["global_round"].astype(int).tolist() != [5, 6, 7, 8]:
                raise RuntimeError(f"Global rounds mismatch attack={attack} seed={seed}")
            if decisions.shape[0] != 80:
                raise RuntimeError(f"Expected 80 client decision rows attack={attack} seed={seed}")

            for mr in [1, 2, 3, 4]:
                sub = decisions.loc[decisions["monitoring_round"].astype(int) == mr]
                ids = sorted(sub["client_id"].astype(int).tolist())
                if ids != list(range(20)):
                    raise RuntimeError(f"Client coverage mismatch attack={attack} seed={seed} round={mr}")
                mals = sorted(
                    sub.loc[sub["actual_malicious"].astype(bool), "client_id"]
                    .astype(int).tolist()
                )
                if mals != MALICIOUS:
                    raise RuntimeError(f"Malicious membership mismatch attack={attack} seed={seed} round={mr}")

            mean_macro = float(rounds["val_macro_f1"].mean())
            mean_bal = float(rounds["val_balanced_accuracy"].mean())
            mean_acc = float(rounds["val_accuracy"].mean())
            mean_mal = float(rounds["malicious_recall"].mean())
            mean_fpr = float(rounds["benign_false_positive_rate"].mean())
            mean_precision = float(rounds["detection_precision"].mean())
            mean_trusted = float(rounds["trusted_clients"].mean())
            mean_rejected = float(rounds["rejected_clients"].mean())
            total_seconds = float(meta["total_seconds"])

            np.testing.assert_allclose(
                mean_macro, float(meta["mean_validation_macro_f1"]), rtol=0.0, atol=1e-15
            )
            np.testing.assert_allclose(
                mean_mal, float(meta["mean_malicious_recall"]), rtol=0.0, atol=1e-15
            )
            np.testing.assert_allclose(
                mean_fpr, float(meta["mean_benign_fpr"]), rtol=0.0, atol=1e-15
            )

            cond_rows.append({
                "attack_type": attack,
                "model_seed": seed,
                "source": "audited_preflight_reused" if (attack == "all_to_one_benign" and seed == 1379954285) else "full_grid",
                "poison_index_hash_sha256": expected_poison,
                "warmup_round4_checkpoint_sha256": meta["warmup_round4_checkpoint_sha256"],
                "mean_validation_accuracy": mean_acc,
                "mean_validation_balanced_accuracy": mean_bal,
                "mean_validation_macro_f1": mean_macro,
                "mean_malicious_recall": mean_mal,
                "mean_benign_fpr": mean_fpr,
                "mean_detection_precision": mean_precision,
                "mean_trusted_clients": mean_trusted,
                "mean_rejected_clients": mean_rejected,
                "total_seconds": total_seconds,
                "test_sets_accessed": False,
                "attack_specific_retuning": False,
                "completion_json_sha256": sha256_file(complete),
            })

            for _, row in rounds.iterrows():
                rr = row.to_dict()
                rr["model_seed"] = seed
                rr["condition_output_source"] = (
                    "audited_preflight_reused"
                    if (attack == "all_to_one_benign" and seed == 1379954285)
                    else "full_grid"
                )
                round_rows.append(rr)

    cond_df = pd.DataFrame(cond_rows)
    round_df = pd.DataFrame(round_rows)

    if cond_df.shape[0] != 25:
        raise RuntimeError(f"Expected 25 condition rows; got {cond_df.shape[0]}")
    if round_df.shape[0] != 100:
        raise RuntimeError(f"Expected 100 round rows; got {round_df.shape[0]}")

    cond_df.to_csv(OUT_COND, index=False)
    round_df.to_csv(OUT_ROUND, index=False)

    by_attack = {}
    for attack, g in cond_df.groupby("attack_type", sort=False):
        by_attack[attack] = {
            "n_seeds": int(g.shape[0]),
            "mean_validation_macro_f1": float(g["mean_validation_macro_f1"].mean()),
            "mean_validation_balanced_accuracy": float(g["mean_validation_balanced_accuracy"].mean()),
            "mean_malicious_recall": float(g["mean_malicious_recall"].mean()),
            "mean_benign_fpr": float(g["mean_benign_fpr"].mean()),
            "mean_detection_precision": float(g["mean_detection_precision"].mean()),
            "mean_total_seconds": float(g["total_seconds"].mean()),
            "min_malicious_recall_across_seeds": float(g["mean_malicious_recall"].min()),
            "max_benign_fpr_across_seeds": float(g["mean_benign_fpr"].max()),
        }

    final = {
        "protocol": "reviewer_v4325_p4p_matched_full_grid",
        "status": "PASS",
        "conditions_verified": 25,
        "round_rows_verified": 100,
        "seeds": SEEDS,
        "attacks": ATTACKS,
        "num_clients": 20,
        "malicious_clients": MALICIOUS,
        "partition_hash_sha256": EXPECTED_PARTITION,
        "p4p_config": EXPECTED_P4P,
        "preflight_condition_reused_without_rerun": {
            "attack_type": "all_to_one_benign",
            "model_seed": 1379954285,
        },
        "all_attack_manifests_reused_and_verified": True,
        "all_test_sets_accessed_false": True,
        "all_attack_specific_retuning_false": True,
        "formal_cross_method_statistics_performed": False,
        "descriptive_by_attack": by_attack,
        "condition_summary_csv_sha256": sha256_file(OUT_COND),
        "round_summary_csv_sha256": sha256_file(OUT_ROUND),
    }
    OUT_JSON.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")

    print("P4P MATCHED FULL GRID SUMMARY = PASS")
    print("VERIFIED CONDITIONS:", cond_df.shape[0])
    print("VERIFIED ROUND ROWS:", round_df.shape[0])
    print("ATTACK MANIFEST REUSE VERIFIED FOR ALL CONDITIONS: True")
    print("TEST SETS ACCESSED: False FOR ALL CONDITIONS")
    print("ATTACK SPECIFIC RETUNING: False FOR ALL CONDITIONS")
    print("FORMAL CROSS METHOD STATISTICS PERFORMED: False")
    print("CONDITION SUMMARY:", OUT_COND)
    print("ROUND SUMMARY:", OUT_ROUND)
    print("FINAL SUMMARY:", OUT_JSON)
    print()
    print("DESCRIPTIVE BY ATTACK")
    for attack in ATTACKS:
        print(attack, json.dumps(by_attack[attack], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

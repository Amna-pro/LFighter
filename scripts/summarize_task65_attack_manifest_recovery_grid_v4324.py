#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

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
PARTITION_HASH = "5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"
PLAIN_NORM = "c579d04815a7e7a72a934926c8092056e83714702322a38a97f2b3b4dab38b75"
DEFENSE_NORM = "9103a41b8152034be2745cd9fbcdbc320b2d935023ceb93eb7b06d93cf7e5b33"

EVID_ROOT = ROOT / "reviewer_revision"
OUT_ROOT = ROOT / "results" / "reviewer_task65_attack_manifest_recovery_v4324"
SUMMARY_JSON = EVID_ROOT / "TASK65_ATTACK_MANIFEST_RECOVERY_GRID_SUMMARY_v4324.json"
MATRIX_CSV = EVID_ROOT / "TASK65_ATTACK_MANIFEST_RECOVERY_GRID_MATRIX_v4324.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evidence_path(seed: int, attack: str) -> Path:
    return EVID_ROOT / f"TASK65_ATTACK_MANIFEST_RECOVERY_SEED_{seed}_{attack.upper()}_v4324.json"


def output_dir(seed: int, attack: str) -> Path:
    return OUT_ROOT / attack / f"seed_{seed}" / "plain_fedavg"


def main() -> int:
    rows = []
    poison_hashes = {}
    total_poisoned_by_attack = {a: 0 for a in ATTACKS}

    for attack in ATTACKS:
        for seed in SEEDS:
            evid = evidence_path(seed, attack)
            out = output_dir(seed, attack)
            if not evid.exists():
                raise FileNotFoundError(evid)
            if not out.exists():
                raise FileNotFoundError(out)

            obj = json.loads(evid.read_text(encoding="utf-8"))
            cond = obj.get("condition", {})
            expected = {
                "status": obj.get("status") == "PASS",
                "seed": int(cond.get("model_seed", -1)) == seed and int(cond.get("attack_seed", -1)) == seed,
                "attack": cond.get("attack_type") == attack,
                "clients": int(cond.get("num_clients", -1)) == 20,
                "coalition": cond.get("malicious_clients") == MALICIOUS,
                "poison_fraction": float(cond.get("poison_fraction", -1)) == 1.0,
                "partition": obj.get("partition_logical_sha256") == PARTITION_HASH,
                "plain_source": obj.get("plain_source_lf_normalized_sha256") == PLAIN_NORM,
                "defense_source": obj.get("defense_source_lf_normalized_sha256") == DEFENSE_NORM,
                "repeat": obj.get("independent_constructor_repeat_exact") is True,
                "loader": obj.get("existing_exact_loader_validation_passed") is True,
                "historical_binary": obj.get("historical_attack_manifest_binary_recovered") is False,
                "test_arrays": obj.get("test_arrays_materialized") is False,
                "training": obj.get("local_model_training_run") is False,
                "p4p": obj.get("p4p_attack_outcome_run") is False,
                "batr": obj.get("batr_attack_outcome_run") is False,
            }
            failed = [k for k, v in expected.items() if not v]
            if failed:
                raise RuntimeError(f"Evidence validation failed for {attack} seed {seed}: {failed}")

            manifest_files = obj["manifest_files"]
            actual_paths = {
                "malicious_client_poison_manifest_csv_sha256": out / "attack_manifest" / "malicious_client_poison_manifest.csv",
                "poisoned_indices_npz_sha256": out / "attack_manifest" / "poisoned_indices.npz",
                "poisoned_labels_npz_sha256": out / "attack_manifest" / "poisoned_labels.npz",
                "metadata_json_sha256": out / "exact_untargeted_plain_v320a3_metadata.json",
            }
            for key, path in actual_paths.items():
                if not path.exists():
                    raise FileNotFoundError(path)
                actual = sha256_file(path)
                if actual != manifest_files[key]:
                    raise RuntimeError(f"Artifact hash mismatch for {attack} seed {seed}: {path}")

            poison_hash = obj["poison_index_hash_sha256"]
            condition_key = f"{attack}|{seed}"
            poison_hashes[condition_key] = poison_hash
            poisoned = int(obj["total_poisoned_rows"])
            total_poisoned_by_attack[attack] += poisoned

            rows.append({
                "attack_type": attack,
                "model_seed": seed,
                "attack_seed": seed,
                "poison_index_hash_sha256": poison_hash,
                "total_poisoned_rows": poisoned,
                "malicious_poisoned_rows": int(obj["malicious_poisoned_rows"]),
                "partition_logical_sha256": obj["partition_logical_sha256"],
                "independent_constructor_repeat_exact": True,
                "existing_exact_loader_validation_passed": True,
                "historical_attack_manifest_binary_recovered": False,
                "test_arrays_materialized": False,
                "local_model_training_run": False,
                "p4p_attack_outcome_run": False,
                "batr_attack_outcome_run": False,
                "evidence_file": str(evid.relative_to(ROOT)).replace("\\", "/"),
                "output_dir": str(out.relative_to(ROOT)).replace("\\", "/"),
            })

    if len(rows) != 25:
        raise RuntimeError(f"Expected 25 rows; got {len(rows)}")

    with MATRIX_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "protocol": "reviewer_v4324_task65_attack_manifest_recovery_grid",
        "task65_git_reference": "2e40296",
        "status": "PASS",
        "expected_conditions": 25,
        "verified_conditions": len(rows),
        "seeds": SEEDS,
        "attacks": ATTACKS,
        "num_clients": 20,
        "malicious_clients": MALICIOUS,
        "poison_fraction": 1.0,
        "partition_logical_sha256": PARTITION_HASH,
        "plain_source_lf_normalized_sha256": PLAIN_NORM,
        "defense_source_lf_normalized_sha256": DEFENSE_NORM,
        "all_independent_constructor_repeats_exact": True,
        "all_existing_exact_loader_validations_passed": True,
        "historical_attack_manifest_binaries_recovered": False,
        "test_arrays_materialized": False,
        "local_model_training_run": False,
        "p4p_attack_outcome_run": False,
        "batr_attack_outcome_run": False,
        "total_poisoned_rows_by_attack_across_five_seeds": total_poisoned_by_attack,
        "poison_index_hashes": poison_hashes,
        "matrix_csv_sha256": sha256_file(MATRIX_CSV),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("TASK65 5 ATTACK x 5 SEED MANIFEST RECOVERY SUMMARY = PASS")
    print("VERIFIED CONDITIONS:", len(rows))
    print("HISTORICAL ATTACK MANIFEST BINARIES RECOVERED: False")
    print("TEST ARRAYS MATERIALIZED: False")
    print("LOCAL MODEL TRAINING RUN: False")
    print("P4P ATTACK OUTCOME RUN: False")
    print("BATR ATTACK OUTCOME RUN: False")
    print("SUMMARY:", SUMMARY_JSON)
    print("MATRIX:", MATRIX_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECOVER = ROOT / "scripts" / "recover_task65_attack_manifest_v4324.py"

SEEDS = [1379954285, 1886033230, 480705558, 1377035733, 1707771978]
ATTACKS = [
    "all_to_one_benign",
    "cyclic_shift",
    "multiclass_partial_cycle",
    "pairwise_swap",
    "random_flip",
]
MALICIOUS = [1, 7, 8, 10, 14, 15, 17, 18]

DATA = ROOT / "data" / "processed" / "cic_iot_diad_2024_v2_1_recovery_check" / "arrays" / "behavioral_only.npz"
PARTITION = ROOT / "results" / "cic_iot_diad_federated_tuning_v27_alpha05_seed42_recovery_check" / "partitions" / "client_partitions.npz"
OUT_ROOT = ROOT / "results" / "reviewer_task65_attack_manifest_recovery_v4324"
EVID_ROOT = ROOT / "reviewer_revision"


def evidence_path(seed: int, attack: str) -> Path:
    return EVID_ROOT / f"TASK65_ATTACK_MANIFEST_RECOVERY_SEED_{seed}_{attack.upper()}_v4324.json"


def output_dir(seed: int, attack: str) -> Path:
    return OUT_ROOT / attack / f"seed_{seed}" / "plain_fedavg"


def validate_existing(seed: int, attack: str, evidence: Path, out: Path) -> None:
    obj = json.loads(evidence.read_text(encoding="utf-8"))
    cond = obj.get("condition", {})
    checks = {
        "status": obj.get("status") == "PASS",
        "model_seed": int(cond.get("model_seed", -1)) == seed,
        "attack_seed": int(cond.get("attack_seed", -1)) == seed,
        "attack_type": cond.get("attack_type") == attack,
        "num_clients": int(cond.get("num_clients", -1)) == 20,
        "malicious_clients": cond.get("malicious_clients") == MALICIOUS,
        "poison_fraction": float(cond.get("poison_fraction", -1.0)) == 1.0,
        "independent_repeat": bool(obj.get("independent_constructor_repeat_exact")),
        "loader_validation": bool(obj.get("existing_exact_loader_validation_passed")),
        "historical_binary_not_claimed": obj.get("historical_attack_manifest_binary_recovered") is False,
        "no_test_arrays": obj.get("test_arrays_materialized") is False,
        "no_local_training": obj.get("local_model_training_run") is False,
        "no_p4p_outcome": obj.get("p4p_attack_outcome_run") is False,
        "no_batr_outcome": obj.get("batr_attack_outcome_run") is False,
        "output_exists": out.exists(),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"Existing evidence/output failed validation for {attack} seed {seed}: {failed}")


def main() -> int:
    if not RECOVER.exists():
        raise FileNotFoundError(RECOVER)
    completed = 0
    skipped = 0

    print("=" * 88)
    print("TASK65 ATTACK MANIFEST RECOVERY GRID v4.32.4")
    print("25 conditions = 5 attacks x 5 final seeds")
    print("No model training and no attack outcome evaluation")
    print("=" * 88)

    for attack in ATTACKS:
        for seed in SEEDS:
            evid = evidence_path(seed, attack)
            out = output_dir(seed, attack)

            print()
            print("-" * 88)
            print(f"CONDITION attack={attack} seed={seed}")
            print("-" * 88)

            if evid.exists() and out.exists():
                validate_existing(seed, attack, evid, out)
                print("SKIP VERIFIED EXISTING CONDITION")
                skipped += 1
                continue

            if evid.exists() != out.exists():
                raise RuntimeError(
                    f"Partial state detected for {attack} seed {seed}. "
                    f"evidence_exists={evid.exists()} output_exists={out.exists()}. "
                    "Nothing was deleted. Inspect before rerunning."
                )

            cmd = [
                sys.executable,
                str(RECOVER),
                "--data-file", str(DATA),
                "--partition-file", str(PARTITION),
                "--output-dir", str(out),
                "--evidence-file", str(evid),
                "--model-seed", str(seed),
                "--attack-seed", str(seed),
                "--attack-type", attack,
                "--num-clients", "20",
                "--malicious-clients", "1,7,8,10,14,15,17,18",
                "--poison-fraction", "1.0",
            ]
            subprocess.run(cmd, cwd=ROOT, check=True)
            validate_existing(seed, attack, evid, out)
            completed += 1

    print()
    print("=" * 88)
    print("TASK65 ATTACK MANIFEST RECOVERY GRID COMPLETE")
    print(f"NEW CONDITIONS RECOVERED: {completed}")
    print(f"VERIFIED EXISTING CONDITIONS SKIPPED: {skipped}")
    print(f"TOTAL VERIFIED CONDITIONS: {completed + skipped}")
    print("LOCAL MODEL TRAINING RUN: False")
    print("P4P ATTACK OUTCOME RUN: False")
    print("BATR ATTACK OUTCOME RUN: False")
    print("=" * 88)
    if completed + skipped != 25:
        raise RuntimeError("Expected exactly 25 verified conditions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

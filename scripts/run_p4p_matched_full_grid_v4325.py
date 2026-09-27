#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
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

RUNNER = ROOT / "scripts" / "run_reviewer_p4p_matched_v4322.py"
DATA = ROOT / "data" / "processed" / "cic_iot_diad_2024_v2_1_recovery_check" / "arrays" / "behavioral_only.npz"
PARTITION = ROOT / "results" / "cic_iot_diad_federated_tuning_v27_alpha05_seed42_recovery_check" / "partitions" / "client_partitions.npz"

GRID_ROOT = ROOT / "results" / "reviewer_p4p_matched_grid_v4325"
PREFLIGHT = (
    ROOT / "results" / "reviewer_p4p_attacked_preflight_v4325"
    / "all_to_one_benign" / "seed_1379954285"
)

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
EXPECTED_PARTITION = "5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"


def completion_path(out: Path) -> Path:
    return out / "REVIEWER_P4P_MATCHED_COMPLETE.json"


def manifest_evidence(seed: int, attack: str) -> Path:
    return (
        ROOT / "reviewer_revision"
        / f"TASK65_ATTACK_MANIFEST_RECOVERY_SEED_{seed}_{attack.upper()}_v4324.json"
    )


def warmup_dir(seed: int) -> Path:
    return (
        ROOT / "results" / "reviewer_task65_warmup_recovery_v4323"
        / f"seed_{seed}" / "warmup"
    )


def clean_seed_dir(seed: int) -> Path:
    return (
        ROOT / "results" / "reviewer_task65_warmup_recovery_v4323"
        / f"seed_{seed}" / "clean_seed_record"
    )


def bootstrap_file(seed: int) -> Path:
    return (
        ROOT / "results" / "reviewer_task65_warmup_recovery_v4323"
        / f"seed_{seed}" / "p4p_probe_bootstrap_v4323.pt"
    )


def plain_branch(seed: int, attack: str) -> Path:
    return (
        ROOT / "results" / "reviewer_task65_attack_manifest_recovery_v4324"
        / attack / f"seed_{seed}" / "plain_fedavg"
    )


def grid_output(seed: int, attack: str) -> Path:
    return GRID_ROOT / attack / f"seed_{seed}"


def read_expected_poison_hash(seed: int, attack: str) -> str:
    p = manifest_evidence(seed, attack)
    if not p.exists():
        raise FileNotFoundError(p)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("status") != "PASS":
        raise RuntimeError(f"Manifest evidence is not PASS: {p}")
    cond = obj.get("condition", {})
    if int(cond.get("model_seed", -1)) != seed or cond.get("attack_type") != attack:
        raise RuntimeError(f"Manifest evidence identity mismatch: {p}")
    if obj.get("test_arrays_materialized") is not False:
        raise RuntimeError(f"Manifest recovery test access boundary invalid: {p}")
    return obj["poison_index_hash_sha256"]


def validate_completion(out: Path, seed: int, attack: str) -> dict:
    cp = completion_path(out)
    if not cp.exists():
        raise FileNotFoundError(cp)
    meta = json.loads(cp.read_text(encoding="utf-8"))
    expected_poison = read_expected_poison_hash(seed, attack)

    checks = {
        "mode": meta.get("mode") == "strong_attack",
        "attack_type": meta.get("attack_type") == attack,
        "model_seed": int(meta.get("model_seed", -1)) == seed,
        "num_clients": int(meta.get("num_clients", -1)) == 20,
        "continuation_rounds": int(meta.get("continuation_rounds", -1)) == 4,
        "malicious_clients": meta.get("malicious_clients") == MALICIOUS,
        "partition_hash": meta.get("partition_hash_sha256") == EXPECTED_PARTITION,
        "poison_hash": meta.get("poison_index_hash_sha256") == expected_poison,
        "p4p_config": meta.get("p4p_config") == EXPECTED_P4P,
        "aggregation": meta.get("aggregation") == "sample_count_weighted_fedavg_over_trusted_set",
        "test_sets_accessed": meta.get("test_sets_accessed") is False,
        "attack_specific_retuning": meta.get("attack_specific_retuning") is False,
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(
            f"Completion validation failed for attack={attack} seed={seed}: {failed}"
        )
    return meta


def require_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def main() -> int:
    for p in [RUNNER, DATA, PARTITION]:
        require_input(p)

    print("=" * 96)
    print("P4P MATCHED FULL GRID v4.32.5")
    print("25 total conditions, one audited preflight reused, 24 remaining conditions to run")
    print("No outcome-based parameter tuning")
    print("=" * 96)

    # The first condition must be preserved and reused without rerun.
    validate_completion(PREFLIGHT, 1379954285, "all_to_one_benign")
    print("AUDITED PREFLIGHT REUSE = PASS: all_to_one_benign seed 1379954285")

    ran = 0
    skipped = 1

    for attack in ATTACKS:
        for seed in SEEDS:
            if attack == "all_to_one_benign" and seed == 1379954285:
                continue

            cseed = clean_seed_dir(seed)
            wdir = warmup_dir(seed)
            boot = bootstrap_file(seed)
            plain = plain_branch(seed, attack)
            out = grid_output(seed, attack)

            for p in [cseed, wdir, boot, plain]:
                require_input(p)

            print()
            print("-" * 96)
            print(f"CONDITION attack={attack} seed={seed}")
            print("-" * 96)

            if out.exists():
                if completion_path(out).exists():
                    validate_completion(out, seed, attack)
                    print("SKIP VERIFIED COMPLETED CONDITION")
                    skipped += 1
                    continue
                raise RuntimeError(
                    f"PARTIAL OUTPUT DIRECTORY EXISTS WITHOUT COMPLETION METADATA: {out}\n"
                    "Nothing was deleted. Inspect this condition before resuming."
                )

            cmd = [
                sys.executable,
                str(RUNNER),
                "--mode", "strong_attack",
                "--attack-type", attack,
                "--data-file", str(DATA),
                "--partition-file", str(PARTITION),
                "--clean-seed-dir", str(cseed),
                "--warmup-dir", str(wdir),
                "--probe-bootstrap-file", str(boot),
                "--plain-branch-dir", str(plain),
                "--output-dir", str(out),
                "--model-seed", str(seed),
                "--num-clients", "20",
                "--continuation-rounds", "4",
                "--batch-size", "2048",
                "--evaluation-batch-size", "4096",
                "--learning-rate", "0.0003",
                "--weight-decay", "0.0001",
                "--max-class-weight", "4.0",
                "--gradient-clip-norm", "5.0",
                "--threads", "6",
                "--malicious-clients", "1,7,8,10,14,15,17,18",
            ]
            subprocess.run(cmd, cwd=ROOT, check=True)
            validate_completion(out, seed, attack)
            print("CONDITION COMPLETION VALIDATION = PASS")
            ran += 1

    total = ran + skipped
    print()
    print("=" * 96)
    print("P4P MATCHED GRID EXECUTION COMPLETE")
    print("NEW CONDITIONS RUN:", ran)
    print("VERIFIED CONDITIONS REUSED OR SKIPPED:", skipped)
    print("TOTAL VERIFIED CONDITIONS:", total)
    print("EXPECTED TOTAL CONDITIONS: 25")
    print("PARAMETER RETUNING: False")
    print("=" * 96)

    if total != 25:
        raise RuntimeError(f"Expected 25 verified conditions; got {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

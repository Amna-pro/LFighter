#!/usr/bin/env python3
"""Run the exact-derived V3.20B.1 frozen untargeted defense grid."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


ATTACK_TYPES = [
    "all_to_one_benign",
    "cyclic_shift",
    "multiclass_partial_cycle",
    "pairwise_swap",
    "random_flip",
]
SEEDS = [7, 99, 123, 2026]
MALICIOUS_CLIENTS = "1,7,8,10,14,15,17,18"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True, type=Path)
    p.add_argument("--partition-file", required=True, type=Path)
    p.add_argument("--clean-seed-root", required=True, type=Path)
    p.add_argument("--v3101-root", required=True, type=Path)
    p.add_argument("--v3123-root", required=True, type=Path)
    p.add_argument("--v320a3-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--seeds", default="7,99,123,2026")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--evaluation-batch-size", type=int, default=4096)
    p.add_argument("--poison-fraction", type=float, default=1.0)
    return p.parse_args()


def run(command: List[str]) -> None:
    print()
    print("RUNNING:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def ensure_stage(
    output_dir: Path,
    completion_file: Path,
    command: List[str],
) -> None:
    if completion_file.exists():
        print("SKIPPING COMPLETE STAGE:", output_dir)
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)
    run(command)


def main() -> int:
    a = parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    if seeds != SEEDS:
        raise ValueError(f"V3.20B.1 is frozen to seeds {SEEDS}")
    if abs(float(a.poison_fraction) - 1.0) > 1e-12:
        raise ValueError(
            "V3.20B.1 is frozen to poison fraction 1.0"
        )

    project = Path(__file__).resolve().parents[1]
    python = sys.executable
    runner = (
        project / "scripts"
        / "run_frozen_untargeted_defense_v320b1.py"
    )
    verifier = (
        project / "scripts"
        / "verify_frozen_untargeted_v320b1.py"
    )
    summarizer = (
        project / "scripts"
        / "summarize_frozen_untargeted_defense_v320b1.py"
    )

    a3 = a.v320a3_root.expanduser().resolve()
    output = a.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    common_static = [
        "--data-file", str(a.data_file.expanduser().resolve()),
        "--partition-file",
        str(a.partition_file.expanduser().resolve()),
        "--num-clients", "20",
        "--continuation-rounds", "4",
        "--batch-size", str(a.batch_size),
        "--evaluation-batch-size",
        str(a.evaluation_batch_size),
        "--learning-rate", "0.0003",
        "--weight-decay", "0.0001",
        "--threads", str(a.threads),
        "--probe-per-class", "48",
        "--probe-seed", "3701",
        "--source-class", "DDoS",
        "--target-class", "Benign",
        "--ema-decay", "0.65",
        "--malicious-clients", MALICIOUS_CLIENTS,
        "--poison-fraction", str(a.poison_fraction),
    ]

    # Four exact clean-adapter checks.
    for seed in seeds:
        clean_seed_dir = (
            a.clean_seed_root.expanduser().resolve()
            / f"seed_{seed}"
        )
        seed_v3101 = (
            a.v3101_root.expanduser().resolve()
            / f"seed_{seed}"
        )
        warmup_dir = seed_v3101 / "warmup"
        original_clean = seed_v3101 / "clean_continuation"

        adapter = output / "adapter_clean" / f"seed_{seed}"
        ensure_stage(
            adapter,
            adapter
            / "frozen_untargeted_defense_v320b1_metadata.json",
            [
                python, str(runner),
                "--mode", "clean",
                "--replacement-policy", "plain_fedavg",
                "--attack-type", "random_flip",
                "--clean-seed-dir", str(clean_seed_dir),
                "--warmup-dir", str(warmup_dir),
                "--output-dir", str(adapter),
                "--model-seed", str(seed),
                "--attack-seed", str(seed),
                *common_static,
            ],
        )
        verification = (
            output / "verification" / "clean"
            / f"seed_{seed}"
        )
        ensure_stage(
            verification,
            verification
            / "v320b1_equivalence_verification.json",
            [
                python, str(verifier),
                "--reference-dir", str(original_clean),
                "--adapter-dir", str(adapter),
                "--output-dir", str(verification),
                "--seed", str(seed),
                "--verification-kind", "clean",
            ],
        )

    # Five exact attacked-pair checks on exploratory seed 7.
    seed = 7
    clean_seed_dir = (
        a.clean_seed_root.expanduser().resolve() / "seed_7"
    )
    warmup_dir = (
        a.v3101_root.expanduser().resolve()
        / "seed_7" / "warmup"
    )
    for attack_type in ATTACK_TYPES:
        exact_plain = (
            a3 / "runs" / attack_type
            / "seed_7" / "plain_fedavg"
        )
        adapter = (
            output / "adapter_attack" / attack_type
            / "seed_7" / "plain_fedavg"
        )
        ensure_stage(
            adapter,
            adapter
            / "frozen_untargeted_defense_v320b1_metadata.json",
            [
                python, str(runner),
                "--mode", "strong_attack",
                "--replacement-policy", "plain_fedavg",
                "--attack-type", attack_type,
                "--plain-branch-dir", str(exact_plain),
                "--clean-seed-dir", str(clean_seed_dir),
                "--warmup-dir", str(warmup_dir),
                "--output-dir", str(adapter),
                "--model-seed", "7",
                "--attack-seed", "7",
                *common_static,
            ],
        )
        verification = (
            output / "verification" / "attack"
            / attack_type / "seed_7"
        )
        ensure_stage(
            verification,
            verification
            / "v320b1_equivalence_verification.json",
            [
                python, str(verifier),
                "--reference-dir", str(exact_plain),
                "--adapter-dir", str(adapter),
                "--output-dir", str(verification),
                "--seed", "7",
                "--verification-kind", "attack",
                "--attack-type", attack_type,
            ],
        )

    # Only after all 9 gates pass, run 20 frozen defended branches.
    for seed in seeds:
        clean_seed_dir = (
            a.clean_seed_root.expanduser().resolve()
            / f"seed_{seed}"
        )
        warmup_dir = (
            a.v3101_root.expanduser().resolve()
            / f"seed_{seed}" / "warmup"
        )
        reconstruction_dir = (
            a.v3123_root.expanduser().resolve()
            / f"seed_{seed}" / "calibration"
        )
        for attack_type in ATTACK_TYPES:
            exact_plain = (
                a3 / "runs" / attack_type
                / f"seed_{seed}" / "plain_fedavg"
            )
            defended = (
                output / "runs" / attack_type
                / f"seed_{seed}"
                / "trusted_reconstruction"
            )
            ensure_stage(
                defended,
                defended
                / "frozen_untargeted_defense_v320b1_metadata.json",
                [
                    python, str(runner),
                    "--mode", "strong_attack",
                    "--replacement-policy",
                    "trusted_reconstruction",
                    "--attack-type", attack_type,
                    "--plain-branch-dir", str(exact_plain),
                    "--reconstruction-calibration-dir",
                    str(reconstruction_dir),
                    "--clean-seed-dir", str(clean_seed_dir),
                    "--warmup-dir", str(warmup_dir),
                    "--output-dir", str(defended),
                    "--model-seed", str(seed),
                    "--attack-seed", str(seed),
                    *common_static,
                ],
            )

    summary = output / "summary"
    ensure_stage(
        summary,
        summary / "v320b1_metadata.json",
        [
            python, str(summarizer),
            "--v320a3-root", str(a3),
            "--v320b1-root", str(output),
            "--output-dir", str(summary),
        ],
    )

    with (output / "v320b1_grid_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "experiment_version": "3.20B.1",
            "stage":
                "task40_exact_derived_frozen_untargeted_defense",
            "seeds": SEEDS,
            "attacks": ATTACK_TYPES,
            "exact_clean_equivalence_checks": 4,
            "exact_attack_pairing_checks": 5,
            "defended_branches": 20,
            "exact_v320a3_poison_plan_reused": True,
            "frozen_reconstruction_policy":
                "center_plus_residual",
            "method_reopened": False,
            "attack_specific_retuning": False,
            "natural_test_accessed": False,
            "diagnostic_test_accessed": False,
        }, handle, indent=2)

    print()
    print("V3.20B.1 GRID COMPLETE")
    decision = (
        summary / "tables"
        / "v320b1_task40_decision.csv"
    )
    if decision.exists():
        import pandas as pd
        print(pd.read_csv(decision).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

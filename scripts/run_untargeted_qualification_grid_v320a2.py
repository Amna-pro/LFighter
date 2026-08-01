#!/usr/bin/env python3
"""Run the V3.20A.2 plain-only cross-seed qualification audit."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


ATTACK_TYPES = [
    "random_flip",
    "cyclic_shift",
    "pairwise_swap",
    "all_to_one_benign",
    "multiclass_partial_cycle",
]
ALL_SEEDS = [7, 99, 123, 2026]
CONFIRMATORY_SEEDS = [99, 123, 2026]
MALICIOUS_CLIENTS = "1,7,8,10,14,15,17,18"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True, type=Path)
    p.add_argument("--partition-file", required=True, type=Path)
    p.add_argument("--clean-seed-root", required=True, type=Path)
    p.add_argument("--v3101-root", required=True, type=Path)
    p.add_argument("--v3123-root", required=True, type=Path)
    p.add_argument("--seed7-smoke-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--seeds", default="7,99,123,2026")
    p.add_argument("--confirmatory-seeds", default="99,123,2026")
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
    seeds = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]
    confirmatory = [
        int(x.strip())
        for x in a.confirmatory_seeds.split(",")
        if x.strip()
    ]
    if seeds != ALL_SEEDS:
        raise ValueError(f"V3.20A.2 is frozen to seeds {ALL_SEEDS}")
    if confirmatory != CONFIRMATORY_SEEDS:
        raise ValueError(
            "V3.20A.2 confirmatory seeds are frozen to 99,123,2026"
        )
    if abs(float(a.poison_fraction) - 1.0) > 1e-12:
        raise ValueError("V3.20A.2 is frozen to poison fraction 1.0")

    project_root = Path(__file__).resolve().parents[1]
    python = sys.executable
    runner = (
        project_root / "scripts"
        / "run_untargeted_label_poisoning_v320a.py"
    )
    verifier = (
        project_root / "scripts"
        / "verify_untargeted_clean_equivalence_v320a2.py"
    )
    summarizer = (
        project_root / "scripts"
        / "summarize_untargeted_qualification_v320a2.py"
    )

    output_root = a.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

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
        original_clean_dir = seed_v3101 / "clean_continuation"
        calibration_dir = (
            a.v3123_root.expanduser().resolve()
            / f"seed_{seed}" / "calibration"
        )

        clean_audit = output_root / "clean_audit" / f"seed_{seed}"
        ensure_stage(
            clean_audit,
            clean_audit
            / "untargeted_label_poisoning_v320a_metadata.json",
            [
                python,
                str(runner),
                "--mode", "clean",
                "--attack-type", "random_flip",
                "--replacement-policy", "plain_fedavg",
                "--data-file",
                str(a.data_file.expanduser().resolve()),
                "--partition-file",
                str(a.partition_file.expanduser().resolve()),
                "--clean-seed-dir", str(clean_seed_dir),
                "--warmup-dir", str(warmup_dir),
                "--reconstruction-calibration-dir",
                str(calibration_dir),
                "--output-dir", str(clean_audit),
                "--model-seed", str(seed),
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
                "--ema-decay", "0.65",
                "--malicious-clients", MALICIOUS_CLIENTS,
                "--poison-fraction", str(a.poison_fraction),
                "--attack-seed", str(seed),
            ],
        )

        verification_dir = (
            output_root / "clean_verification" / f"seed_{seed}"
        )
        ensure_stage(
            verification_dir,
            verification_dir
            / "v320a2_clean_equivalence_verification.json",
            [
                python,
                str(verifier),
                "--original-clean-dir", str(original_clean_dir),
                "--audit-clean-dir", str(clean_audit),
                "--output-dir", str(verification_dir),
                "--seed", str(seed),
            ],
        )

        if seed not in CONFIRMATORY_SEEDS:
            continue

        for attack_type in ATTACK_TYPES:
            branch = (
                output_root / "runs" / attack_type
                / f"seed_{seed}" / "plain_fedavg"
            )
            ensure_stage(
                branch,
                branch
                / "untargeted_label_poisoning_v320a_metadata.json",
                [
                    python,
                    str(runner),
                    "--mode", "strong_attack",
                    "--attack-type", attack_type,
                    "--replacement-policy", "plain_fedavg",
                    "--data-file",
                    str(a.data_file.expanduser().resolve()),
                    "--partition-file",
                    str(a.partition_file.expanduser().resolve()),
                    "--clean-seed-dir", str(clean_seed_dir),
                    "--warmup-dir", str(warmup_dir),
                    "--reconstruction-calibration-dir",
                    str(calibration_dir),
                    "--output-dir", str(branch),
                    "--model-seed", str(seed),
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
                    "--ema-decay", "0.65",
                    "--malicious-clients", MALICIOUS_CLIENTS,
                    "--poison-fraction", str(a.poison_fraction),
                    "--attack-seed", str(seed),
                ],
            )

    summary_dir = output_root / "summary"
    ensure_stage(
        summary_dir,
        summary_dir / "v320a2_metadata.json",
        [
            python,
            str(summarizer),
            "--input-root", str(output_root),
            "--seed7-smoke-root",
            str(a.seed7_smoke_root.expanduser().resolve()),
            "--output-dir", str(summary_dir),
            "--seeds", a.seeds,
            "--confirmatory-seeds", a.confirmatory_seeds,
        ],
    )

    with (output_root / "v320a2_grid_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "experiment_version": "3.20A.2",
            "stage":
                "plain_only_cross_seed_untargeted_attack_qualification",
            "all_seeds": ALL_SEEDS,
            "exploratory_seed": 7,
            "confirmatory_seeds": CONFIRMATORY_SEEDS,
            "attacks": ATTACK_TYPES,
            "malicious_clients": MALICIOUS_CLIENTS,
            "poison_fraction": float(a.poison_fraction),
            "defense_branches_run": False,
            "method_reopened": False,
            "attack_specific_retuning": False,
            "natural_test_accessed": False,
            "diagnostic_test_accessed": False,
            "summary_dir": str(summary_dir),
        }, handle, indent=2)

    print()
    print("V3.20A.2 GRID COMPLETE")
    decision = (
        summary_dir / "tables"
        / "v320a2_qualification_decision.csv"
    )
    if decision.exists():
        import pandas as pd
        print(pd.read_csv(decision).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

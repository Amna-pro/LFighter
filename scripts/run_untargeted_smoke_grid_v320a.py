#!/usr/bin/env python3
"""Run all five frozen V3.20A untargeted smoke attacks on seed 7."""
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
MALICIOUS_CLIENTS = "1,7,8,10,14,15,17,18"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True, type=Path)
    p.add_argument("--partition-file", required=True, type=Path)
    p.add_argument("--clean-seed-dir", required=True, type=Path)
    p.add_argument("--warmup-dir", required=True, type=Path)
    p.add_argument(
        "--reconstruction-calibration-dir",
        required=True,
        type=Path,
    )
    p.add_argument("--clean-continuation-dir", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--seed", type=int, default=7)
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


def ensure_branch(
    output_dir: Path,
    completion_file: Path,
    command: List[str],
) -> None:
    if completion_file.exists():
        print("SKIPPING COMPLETE BRANCH:", output_dir)
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)
    run(command)


def main() -> int:
    a = parse_args()
    if int(a.seed) != 7:
        raise ValueError("V3.20A smoke is frozen to seed 7")
    if abs(float(a.poison_fraction) - 1.0) > 1e-12:
        raise ValueError("V3.20A smoke is frozen to poison fraction 1.0")

    project_root = Path(__file__).resolve().parents[1]
    python = sys.executable
    output_root = a.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runner = (
        project_root / "scripts"
        / "run_untargeted_label_poisoning_v320a.py"
    )

    common = [
        "--mode", "strong_attack",
        "--data-file", str(a.data_file.expanduser().resolve()),
        "--partition-file",
        str(a.partition_file.expanduser().resolve()),
        "--clean-seed-dir",
        str(a.clean_seed_dir.expanduser().resolve()),
        "--warmup-dir", str(a.warmup_dir.expanduser().resolve()),
        "--reconstruction-calibration-dir",
        str(
            a.reconstruction_calibration_dir.expanduser().resolve()
        ),
        "--model-seed", str(a.seed),
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
        "--attack-seed", str(a.seed),
    ]

    for attack_type in ATTACK_TYPES:
        for policy in ("plain_fedavg", "trusted_reconstruction"):
            branch = (
                output_root / "runs" / attack_type / policy
            )
            completion = (
                branch
                / "untargeted_label_poisoning_v320a_metadata.json"
            )
            ensure_branch(
                branch,
                completion,
                [
                    python,
                    str(runner),
                    "--attack-type", attack_type,
                    "--replacement-policy", policy,
                    "--output-dir", str(branch),
                    *common,
                ],
            )

    summary_dir = output_root / "summary"
    summary_file = summary_dir / "v320a_metadata.json"
    if summary_file.exists():
        print("SKIPPING COMPLETE SUMMARY:", summary_dir)
    else:
        if summary_dir.exists():
            shutil.rmtree(summary_dir)
        run([
            python,
            str(
                project_root / "scripts"
                / "summarize_untargeted_smoke_v320a.py"
            ),
            "--input-root", str(output_root),
            "--clean-dir",
            str(a.clean_continuation_dir.expanduser().resolve()),
            "--output-dir", str(summary_dir),
            "--seed", str(a.seed),
        ])

    with (output_root / "v320a_grid_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "experiment_version": "3.20A",
            "stage": "frozen_untargeted_label_poisoning_smoke_grid",
            "seed": int(a.seed),
            "attacks": ATTACK_TYPES,
            "malicious_clients": MALICIOUS_CLIENTS,
            "poison_fraction": float(a.poison_fraction),
            "method_reopened": False,
            "attack_specific_retuning": False,
            "natural_test_accessed": False,
            "diagnostic_test_accessed": False,
        }, handle, indent=2)

    print()
    print("V3.20A GRID COMPLETE")
    decision = (
        summary_dir / "tables" / "v320a_smoke_decision.csv"
    )
    if decision.exists():
        import pandas as pd
        print(pd.read_csv(decision).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

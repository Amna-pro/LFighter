#!/usr/bin/env python3
"""Run V3.20A.3 exact plain-only untargeted qualification."""
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
SEEDS = [7, 99, 123, 2026]
MALICIOUS_CLIENTS = "1,7,8,10,14,15,17,18"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True, type=Path)
    p.add_argument("--partition-file", required=True, type=Path)
    p.add_argument("--clean-seed-root", required=True, type=Path)
    p.add_argument("--v3101-root", required=True, type=Path)
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
    seeds = [int(x) for x in a.seeds.split(",")]
    if seeds != SEEDS:
        raise ValueError(f"V3.20A.3 is frozen to {SEEDS}")
    if a.confirmatory_seeds != "99,123,2026":
        raise ValueError(
            "Confirmatory seeds are frozen to 99,123,2026"
        )
    if abs(a.poison_fraction - 1.0) > 1e-12:
        raise ValueError(
            "V3.20A.3 is frozen to poison fraction 1.0"
        )

    project = Path(__file__).resolve().parents[1]
    python = sys.executable
    runner = (
        project / "scripts"
        / "run_exact_untargeted_plain_v320a3.py"
    )
    verifier = (
        project / "scripts"
        / "verify_exact_untargeted_clean_v320a3.py"
    )
    summarizer = (
        project / "scripts"
        / "summarize_exact_untargeted_qualification_v320a3.py"
    )

    root = a.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        clean_seed_dir = (
            a.clean_seed_root.expanduser().resolve()
            / f"seed_{seed}"
        )
        v3101_seed = (
            a.v3101_root.expanduser().resolve()
            / f"seed_{seed}"
        )
        warmup_dir = v3101_seed / "warmup"
        original_clean = v3101_seed / "clean_continuation"

        clean_exact = root / "clean_exact" / f"seed_{seed}"
        ensure_stage(
            clean_exact,
            clean_exact
            / "exact_untargeted_plain_v320a3_metadata.json",
            [
                python,
                str(runner),
                "--mode", "clean",
                "--attack-type", "random_flip",
                "--data-file",
                str(a.data_file.expanduser().resolve()),
                "--partition-file",
                str(a.partition_file.expanduser().resolve()),
                "--clean-seed-dir", str(clean_seed_dir),
                "--warmup-dir", str(warmup_dir),
                "--output-dir", str(clean_exact),
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
                "--source-class", "DDoS",
                "--target-class", "Benign",
                "--ema-decay", "0.65",
                "--malicious-clients", MALICIOUS_CLIENTS,
                "--poison-fraction", str(a.poison_fraction),
                "--attack-seed", str(seed),
            ],
        )

        verification = (
            root / "clean_verification" / f"seed_{seed}"
        )
        ensure_stage(
            verification,
            verification
            / "v320a3_clean_equivalence_verification.json",
            [
                python,
                str(verifier),
                "--original-clean-dir", str(original_clean),
                "--current-clean-dir", str(clean_exact),
                "--output-dir", str(verification),
                "--seed", str(seed),
            ],
        )

        for attack_type in ATTACK_TYPES:
            branch = (
                root / "runs" / attack_type
                / f"seed_{seed}" / "plain_fedavg"
            )
            ensure_stage(
                branch,
                branch
                / "exact_untargeted_plain_v320a3_metadata.json",
                [
                    python,
                    str(runner),
                    "--mode", "strong_attack",
                    "--attack-type", attack_type,
                    "--data-file",
                    str(a.data_file.expanduser().resolve()),
                    "--partition-file",
                    str(a.partition_file.expanduser().resolve()),
                    "--clean-seed-dir", str(clean_seed_dir),
                    "--warmup-dir", str(warmup_dir),
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
                    "--source-class", "DDoS",
                    "--target-class", "Benign",
                    "--ema-decay", "0.65",
                    "--malicious-clients", MALICIOUS_CLIENTS,
                    "--poison-fraction", str(a.poison_fraction),
                    "--attack-seed", str(seed),
                ],
            )

    summary = root / "summary"
    ensure_stage(
        summary,
        summary / "v320a3_metadata.json",
        [
            python,
            str(summarizer),
            "--input-root", str(root),
            "--output-dir", str(summary),
            "--seeds", a.seeds,
            "--confirmatory-seeds",
            a.confirmatory_seeds,
        ],
    )

    with (root / "v320a3_grid_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump({
            "experiment_version": "3.20A.3",
            "stage":
                "exact_plain_untargeted_qualification_correction",
            "seeds": SEEDS,
            "confirmatory_seeds": [99, 123, 2026],
            "attacks": ATTACK_TYPES,
            "malicious_clients": MALICIOUS_CLIENTS,
            "poison_fraction": a.poison_fraction,
            "exact_clean_adapter":
                "run_exact_untargeted_plain_v320a3.py",
            "defense_branches_run": False,
            "prior_v320a_utility_results_status":
                "superseded",
            "prior_v320a_mechanics_status":
                "retained",
            "method_reopened": False,
            "attack_specific_retuning": False,
            "natural_test_accessed": False,
            "diagnostic_test_accessed": False,
        }, handle, indent=2)

    print()
    print("V3.20A.3 GRID COMPLETE")
    decision = (
        summary / "tables"
        / "v320a3_qualification_decision.csv"
    )
    if decision.exists():
        import pandas as pd
        print(pd.read_csv(decision).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

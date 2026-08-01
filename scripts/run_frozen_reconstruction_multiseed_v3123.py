#!/usr/bin/env python3
"""Run the frozen V3.12.3 chronology on held-out development seeds."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True, type=Path)
    p.add_argument("--partition-file", required=True, type=Path)
    p.add_argument("--clean-seed-root", required=True, type=Path)
    p.add_argument("--v3101-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--seeds", default="7,99,123,2026")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--evaluation-batch-size", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--probe-per-class", type=int, default=48)
    p.add_argument("--probe-seed", type=int, default=3701)
    p.add_argument("--ema-decay", type=float, default=0.65)
    p.add_argument("--poison-fraction", type=float, default=1.0)
    p.add_argument(
        "--frozen-policy",
        default="center_plus_residual",
        choices=[
            "center_only",
            "center_plus_residual",
            "center_plus_scaled_residual",
            "center_plus_scaled_residual_normclip",
        ],
    )
    return p.parse_args()


def run(command: List[str]) -> None:
    print("\nRUNNING:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def ensure_stage(
    output_dir: Path,
    completion_file: Path,
    command: List[str],
) -> None:
    if completion_file.exists():
        print(f"SKIPPING COMPLETE STAGE: {output_dir}")
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)
    run(command)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python = sys.executable
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    builder = project_root / "scripts" / "build_frozen_reconstruction_calibration_v3123.py"
    runner = project_root / "scripts" / "run_frozen_reconstruction_v3123.py"
    summarizer = project_root / "scripts" / "summarize_frozen_reconstruction_seed_v3123.py"

    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        source_seed_root = args.v3101_root.expanduser().resolve() / f"seed_{seed}"
        warmup_dir = source_seed_root / "warmup"
        baseline_clean = source_seed_root / "clean_continuation"
        baseline_attack = source_seed_root / "attack_continuation"
        clean_seed_dir = (
            args.clean_seed_root.expanduser().resolve() / f"seed_{seed}"
        )

        calibration = seed_root / "calibration"
        trusted_clean = seed_root / "trusted_clean"
        trusted_attack = seed_root / "trusted_attack"
        oracle_clean = seed_root / "oracle_clean_replacement"
        summary = seed_root / "summary"

        common = [
            "--data-file", str(args.data_file.expanduser().resolve()),
            "--partition-file", str(args.partition_file.expanduser().resolve()),
            "--clean-seed-dir", str(clean_seed_dir),
            "--warmup-dir", str(warmup_dir),
            "--model-seed", str(seed),
            "--num-clients", "20",
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--threads", str(args.threads),
        ]

        ensure_stage(
            calibration,
            calibration / "trusted_update_reconstruction_v312_metadata.json",
            [
                python, str(builder),
                *common,
                "--output-dir", str(calibration),
                "--warmup-rounds", "4",
                "--evaluation-batch-size", str(args.evaluation_batch_size),
                "--probe-per-class", str(args.probe_per_class),
                "--probe-seed", str(args.probe_seed),
                "--source-class", "DDoS",
                "--target-class", "Benign",
                "--scale-lower", "0.50",
                "--scale-upper", "2.00",
                "--norm-clip-multiplier", "2.50",
                "--maximum-replay-state-difference", "0.005",
                "--maximum-replay-relative-state-l2", "0.005",
                "--maximum-round-macro-f1-difference", "0.005",
                "--maximum-round-source-target-difference", "0.01",
                "--maximum-round-train-loss-difference", "0.005",
                "--maximum-round-train-accuracy-difference", "0.005",
                "--force-selected-policy", args.frozen_policy,
            ],
        )

        run_common = [
            "--data-file", str(args.data_file.expanduser().resolve()),
            "--partition-file", str(args.partition_file.expanduser().resolve()),
            "--clean-seed-dir", str(clean_seed_dir),
            "--warmup-dir", str(warmup_dir),
            "--reconstruction-calibration-dir", str(calibration),
            "--model-seed", str(seed),
            "--num-clients", "20",
            "--continuation-rounds", "4",
            "--batch-size", str(args.batch_size),
            "--evaluation-batch-size", str(args.evaluation_batch_size),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--threads", str(args.threads),
            "--probe-per-class", str(args.probe_per_class),
            "--probe-seed", str(args.probe_seed),
            "--source-class", "DDoS",
            "--target-class", "Benign",
            "--ema-decay", str(args.ema_decay),
        ]

        ensure_stage(
            oracle_clean,
            oracle_clean / "trusted_update_reconstruction_v312_metadata.json",
            [
                python, str(runner),
                "--mode", "strong_attack",
                "--replacement-policy", "oracle_clean_replacement",
                *run_common,
                "--output-dir", str(oracle_clean),
                "--attack-seed", str(seed),
                "--poison-fraction", str(args.poison_fraction),
            ],
        )

        ensure_stage(
            trusted_clean,
            trusted_clean / "trusted_update_reconstruction_v312_metadata.json",
            [
                python, str(runner),
                "--mode", "clean",
                "--replacement-policy", "trusted_reconstruction",
                *run_common,
                "--output-dir", str(trusted_clean),
            ],
        )

        ensure_stage(
            trusted_attack,
            trusted_attack / "trusted_update_reconstruction_v312_metadata.json",
            [
                python, str(runner),
                "--mode", "strong_attack",
                "--replacement-policy", "trusted_reconstruction",
                *run_common,
                "--output-dir", str(trusted_attack),
                "--attack-seed", str(seed),
                "--poison-fraction", str(args.poison_fraction),
            ],
        )

        ensure_stage(
            summary,
            summary / "v3123_seed_summary_metadata.json",
            [
                python,
                str(project_root / "scripts" / "summarize_frozen_reconstruction_pair_v3123.py"),
                "--plain-clean-dir", str(baseline_clean),
                "--plain-attack-dir", str(baseline_attack),
                "--reconstruction-clean-dir", str(trusted_clean),
                "--reconstruction-attack-dir", str(trusted_attack),
                "--oracle-clean-replacement-attack-dir", str(oracle_clean),
                "--output-dir", str(summary),
                "--seed", str(seed),
            ],
        )

    metadata = {
        "experiment_version": "3.12.3",
        "heldout_development_seeds": seeds,
        "frozen_reconstruction_policy": args.frozen_policy,
        "policy_frozen_from_seed42": True,
        "rng_isolation_patch": "3.12.2",
        "test_sets_accessed": False,
    }
    with (output_root / "v3123_run_metadata.json").open("w", encoding="utf-8") as h:
        json.dump(metadata, h, indent=2)

    print("\nV3.12.3 held-out multiseed run complete")
    print("Output root:", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the frozen V3.10 true-warmup chronology across multiple development seeds."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-seed-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", default="7,99,123,2026")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-per-class", type=int, default=48)
    parser.add_argument("--probe-seed", type=int, default=3701)
    parser.add_argument("--ema-decay", type=float, default=0.65)
    parser.add_argument("--clean-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--poison-fraction", type=float, default=1.0)
    return parser.parse_args()


def run(command: List[str]) -> None:
    print()
    print("RUNNING:")
    print(" ".join(f'"{value}"' if " " in value else value for value in command))
    subprocess.run(command, check=True)


def complete(path: Path) -> bool:
    return path.exists()


def prepare_stage(output_dir: Path, completion_file: Path, checkpoint_file: Path) -> List[str]:
    if complete(completion_file):
        return ["skip"]
    if output_dir.exists() and any(output_dir.iterdir()):
        if checkpoint_file.exists():
            return ["resume"]
        shutil.rmtree(output_dir)
    return ["fresh"]


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    warmup_script = project_root / "scripts" / "run_true_warmup_v310.py"
    continuation_script = project_root / "scripts" / "run_post_warmup_capture_v310.py"
    paired_script = project_root / "scripts" / "summarize_paired_v310.py"
    aggregate_script = project_root / "scripts" / "summarize_v310_multiseed_v3101.py"

    for required in (warmup_script, continuation_script, paired_script, aggregate_script):
        if not required.exists():
            raise FileNotFoundError(required)

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        warmup_dir = seed_root / "warmup"
        clean_dir = seed_root / "clean_continuation"
        attack_dir = seed_root / "attack_continuation"
        paired_dir = seed_root / "paired_summary"
        clean_seed_dir = args.clean_seed_root.expanduser().resolve() / f"seed_{seed}"

        if not clean_seed_dir.exists():
            raise FileNotFoundError(clean_seed_dir)

        warmup_state = prepare_stage(
            warmup_dir,
            warmup_dir / "true_warmup_v310_metadata.json",
            warmup_dir / "checkpoints" / "warmup_last_round_model.pt",
        )
        if warmup_state[0] != "skip":
            command = [
                sys.executable,
                str(warmup_script),
                "--data-file", str(args.data_file.expanduser().resolve()),
                "--partition-file", str(args.partition_file.expanduser().resolve()),
                "--clean-seed-dir", str(clean_seed_dir),
                "--output-dir", str(warmup_dir),
                "--model-seed", str(seed),
                "--warmup-rounds", "4",
                "--batch-size", str(args.batch_size),
                "--evaluation-batch-size", str(args.evaluation_batch_size),
                "--learning-rate", str(args.learning_rate),
                "--weight-decay", str(args.weight_decay),
                "--threads", str(args.threads),
                "--probe-per-class", str(args.probe_per_class),
                "--probe-seed", str(args.probe_seed),
                "--ema-decay", str(args.ema_decay),
                "--clean-threshold-quantile", str(args.clean_threshold_quantile),
            ]
            if warmup_state[0] == "resume":
                command.append("--resume")
            run(command)
        else:
            print(f"Skipping completed warmup for seed {seed}")

        for mode, continuation_dir in (
            ("clean", clean_dir),
            ("strong_attack", attack_dir),
        ):
            state = prepare_stage(
                continuation_dir,
                continuation_dir / "post_warmup_capture_v310_metadata.json",
                continuation_dir / "checkpoints" / "continuation_last_round_model.pt",
            )
            if state[0] != "skip":
                command = [
                    sys.executable,
                    str(continuation_script),
                    "--mode", mode,
                    "--data-file", str(args.data_file.expanduser().resolve()),
                    "--partition-file", str(args.partition_file.expanduser().resolve()),
                    "--clean-seed-dir", str(clean_seed_dir),
                    "--warmup-dir", str(warmup_dir),
                    "--output-dir", str(continuation_dir),
                    "--model-seed", str(seed),
                    "--continuation-rounds", "4",
                    "--batch-size", str(args.batch_size),
                    "--evaluation-batch-size", str(args.evaluation_batch_size),
                    "--learning-rate", str(args.learning_rate),
                    "--weight-decay", str(args.weight_decay),
                    "--threads", str(args.threads),
                    "--probe-per-class", str(args.probe_per_class),
                    "--probe-seed", str(args.probe_seed),
                    "--ema-decay", str(args.ema_decay),
                ]
                if mode == "strong_attack":
                    command.extend([
                        "--attack-seed", str(seed),
                        "--poison-fraction", str(args.poison_fraction),
                    ])
                if state[0] == "resume":
                    command.append("--resume")
                run(command)
            else:
                print(f"Skipping completed {mode} continuation for seed {seed}")

        paired_csv = paired_dir / "tables" / "paired_aggregate_summary.csv"
        if not paired_csv.exists():
            if paired_dir.exists():
                shutil.rmtree(paired_dir)
            run([
                sys.executable,
                str(paired_script),
                "--clean-dir", str(clean_dir),
                "--attack-dir", str(attack_dir),
                "--output-dir", str(paired_dir),
            ])
        else:
            print(f"Skipping completed paired summary for seed {seed}")

        manifest_rows.append({
            "seed": seed,
            "warmup_dir": str(warmup_dir),
            "clean_dir": str(clean_dir),
            "attack_dir": str(attack_dir),
            "paired_dir": str(paired_dir),
        })

    with (output_root / "multiseed_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_version": "3.10.1",
                "seeds": seeds,
                "frozen_protocol": "V3.10 true four-round warmup chronology",
                "defense_weights_applied": False,
                "test_sets_accessed": False,
                "runs": manifest_rows,
            },
            handle,
            indent=2,
        )

    run([
        sys.executable,
        str(aggregate_script),
        "--input-root", str(output_root),
        "--output-dir", str(output_root / "aggregate"),
        "--seeds", ",".join(map(str, seeds)),
    ])

    print()
    print("V3.10.1 multiseed chronology complete")
    print("Output root:", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

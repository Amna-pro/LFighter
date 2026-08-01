#!/usr/bin/env python3
"""Resumable multiseed orchestrator for V3.7 transition-signature capture."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.7 clean and attack captures.")
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-seed-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", default="42,7,99,123,2026")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=6)
    return parser.parse_args()


def completed(output_dir: Path, rounds: int) -> bool:
    metadata = output_dir / "transition_signature_v37_metadata.json"
    if not metadata.exists():
        return False
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(payload.get("rounds_completed", 0)) >= int(rounds)


def main() -> int:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    runner = Path(__file__).resolve().parent / "run_transition_signature_capture_v37.py"

    for mode in ("clean", "strong_attack"):
        for seed in seeds:
            output_dir = args.output_root / mode / f"seed_{seed}"
            clean_seed_dir = args.clean_seed_root / f"seed_{seed}"
            if completed(output_dir, args.rounds):
                print(f"SKIP completed: {mode}, seed {seed}")
                continue

            command = [
                sys.executable,
                str(runner),
                "--mode",
                mode,
                "--data-file",
                str(args.data_file),
                "--partition-file",
                str(args.partition_file),
                "--clean-seed-dir",
                str(clean_seed_dir),
                "--output-dir",
                str(output_dir),
                "--model-seed",
                str(seed),
                "--rounds",
                str(args.rounds),
                "--batch-size",
                str(args.batch_size),
                "--evaluation-batch-size",
                str(args.evaluation_batch_size),
                "--threads",
                str(args.threads),
            ]
            if output_dir.exists() and any(output_dir.iterdir()):
                command.append("--resume")
            print("RUN:", " ".join(command))
            subprocess.run(command, check=True)

    print("All requested V3.7 captures are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

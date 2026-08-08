#!/usr/bin/env python3
"""Task 44 C2: poison-fraction strength variation, orchestration only.

Purpose
-------
Reuses the existing, unmodified, already-proven V3.20A.3 plain-attack
runner and V3.20B.1 defended-evaluation runner directly at new
poison-fraction values (0.25/0.50/0.75), bypassing the two frozen
grid-orchestration scripts' hard poison_fraction==1.0 locks entirely
(confirmed via direct source inspection: the lock lives only in the grid
wrappers' argument validation, not in either runner). No attack
construction or defense logic is reimplemented -- every fixed parameter
below (hyperparameters, probe config, EMA decay, source/target class,
malicious-client coalition, attack-seed derivation) was read directly
from the frozen grid scripts' own subprocess-construction code, not
guessed.

Sequencing, per seed:
1. Clean-adapter verification (mode=clean) -- run ONCE per seed, reused
   across all three new fractions, since clean mode never touches
   poisoning and re-verifying it three times per seed would be pure
   duplicated compute producing byte-identical results.
2. Per fraction, per attack type: plain attack branch (mode=strong_attack,
   the ungated runner, poison-fraction=NEW VALUE).
3. Per fraction, per attack type: defended branch
   (replacement-policy=trusted_reconstruction), pointing --plain-branch-dir
   at step 2's fresh output -- the defended run loads its poison plan
   from the plain branch rather than generating its own, so this
   ordering is required, not optional.

This is orchestration only. Metric extraction (recall/FPR/damage-removed
per branch) is deliberately left to a separate follow-up script, once the
runners' actual internal output filenames are confirmed rather than
guessed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

CONFIRMATORY_SEEDS = (7, 99, 123, 2026)
NEW_FRACTIONS = (0.25, 0.50, 0.75)
ATTACK_TYPES = (
    "all_to_one_benign",
    "cyclic_shift",
    "multiclass_partial_cycle",
    "pairwise_swap",
    "random_flip",
)
MALICIOUS_CLIENTS = "1,7,8,10,14,15,17,18"

# Dedicated orchestrator-only marker filename -- deliberately distinct from
# every filename either runner script writes itself
# (exact_untargeted_plain_v320a3_metadata.json and
# frozen_untargeted_defense_v320b1_metadata.json are both REAL runner
# outputs, not safe to reuse as a completion marker).
ORCHESTRATOR_MARKER_NAME = "_task44_orchestrator_stage_complete.json"

# Fixed values read directly from the frozen grid scripts' own
# subprocess-construction code -- not guessed or re-derived.
COMMON_STATIC_FLAGS: List[str] = [
    "--num-clients", "20",
    "--continuation-rounds", "4",
    "--learning-rate", "0.0003",
    "--weight-decay", "0.0001",
    "--probe-per-class", "48",
    "--probe-seed", "3701",
    "--source-class", "DDoS",
    "--target-class", "Benign",
    "--ema-decay", "0.65",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 44 C2 poison-fraction orchestration.")
    parser.add_argument("--plain-runner", type=Path, required=True,
                         help="scripts\\run_exact_untargeted_plain_v320a3.py")
    parser.add_argument("--defense-runner", type=Path, required=True,
                         help="scripts\\run_frozen_untargeted_defense_v320b1.py")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--partition-file", type=Path, required=True)
    parser.add_argument("--clean-seed-root", type=Path, required=True,
                         help="results\\cic_iot_diad_federated_multiseed_v28_fixed_partition\\seed_runs (contains seed_<N>)")
    parser.add_argument("--warmup-root", type=Path, required=True,
                         help="results\\cic_iot_diad_true_warmup_anchor_v3101_multiseed (contains seed_<N>\\warmup)")
    parser.add_argument("--reconstruction-root", type=Path, required=True,
                         help="results\\cic_iot_diad_frozen_reconstruction_v3123_multiseed (contains seed_<N>\\calibration)")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true",
                         help="Print every command without executing it.")
    return parser.parse_args()


def ensure_stage(marker_path: Path, command: List[str], dry_run: bool) -> None:
    branch_dir = marker_path.parent
    if marker_path.exists():
        print(f"SKIP COMPLETE: {branch_dir}")
        return
    # If the branch directory already exists with leftover content but no
    # completion marker, a prior attempt was interrupted mid-run. Tell the
    # runner to overwrite that specific incomplete state rather than
    # blindly passing --resume to every call -- run_update_capture_v318b.py
    # demonstrated earlier in this project that --resume can require
    # pre-existing state and fail outright on a genuinely fresh directory,
    # so --resume must never be applied to a first attempt.
    effective_command = list(command)
    if branch_dir.exists() and any(
        p for p in branch_dir.iterdir() if p.name != ORCHESTRATOR_MARKER_NAME
    ):
        effective_command = effective_command + ["--overwrite"]
        print(f"RESUME (overwrite incomplete state): {branch_dir}")

    print()
    print(f"RUN: {' '.join(str(c) for c in effective_command)}")
    if dry_run:
        return
    result = subprocess.run(effective_command, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(str(c) for c in effective_command)}"
        )
    # CRITICAL: this marker uses a name that cannot collide with any file
    # the runner scripts themselves produce (both
    # exact_untargeted_plain_v320a3_metadata.json and
    # frozen_untargeted_defense_v320b1_metadata.json are real runner
    # outputs, not safe to reuse as an orchestrator marker -- an earlier
    # version of this script did exactly that and silently overwrote the
    # runner's real metadata with this stub, corrupting the attack_type
    # field the defense stage reads back downstream).
    marker_path.write_text(
        json.dumps({"complete": True, "command": [str(c) for c in effective_command]}, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    python = sys.executable
    plain_runner = args.plain_runner.expanduser().resolve()
    defense_runner = args.defense_runner.expanduser().resolve()
    data_file = args.data_file.expanduser().resolve()
    partition_file = args.partition_file.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    perf_flags = [
        "--batch-size", str(args.batch_size),
        "--evaluation-batch-size", str(args.evaluation_batch_size),
        "--threads", str(args.threads),
    ]

    for seed in CONFIRMATORY_SEEDS:
        clean_seed_dir = args.clean_seed_root.expanduser().resolve() / f"seed_{seed}"
        warmup_dir = args.warmup_root.expanduser().resolve() / f"seed_{seed}" / "warmup"
        reconstruction_dir = (
            args.reconstruction_root.expanduser().resolve() / f"seed_{seed}" / "calibration"
        )

        # Step 1: clean-adapter verification, ONCE per seed (fraction-independent).
        adapter_dir = output_root / "clean_adapter_verification" / f"seed_{seed}"
        ensure_stage(
            adapter_dir / ORCHESTRATOR_MARKER_NAME,
            [
                python, str(defense_runner),
                "--mode", "clean",
                "--replacement-policy", "plain_fedavg",
                "--attack-type", "random_flip",
                "--data-file", str(data_file),
                "--partition-file", str(partition_file),
                "--clean-seed-dir", str(clean_seed_dir),
                "--warmup-dir", str(warmup_dir),
                "--output-dir", str(adapter_dir),
                "--model-seed", str(seed),
                "--attack-seed", str(seed),
                "--malicious-clients", MALICIOUS_CLIENTS,
                *COMMON_STATIC_FLAGS,
                *perf_flags,
            ],
            args.dry_run,
        )

        for fraction in NEW_FRACTIONS:
            fraction_tag = f"{fraction:.2f}".replace(".", "p")
            for attack_type in ATTACK_TYPES:
                plain_dir = (
                    output_root / f"fraction_{fraction_tag}" / attack_type
                    / f"seed_{seed}" / "plain_fedavg"
                )
                ensure_stage(
                    plain_dir / ORCHESTRATOR_MARKER_NAME,
                    [
                        python, str(plain_runner),
                        "--mode", "strong_attack",
                        "--attack-type", attack_type,
                        "--data-file", str(data_file),
                        "--partition-file", str(partition_file),
                        "--clean-seed-dir", str(clean_seed_dir),
                        "--warmup-dir", str(warmup_dir),
                        "--output-dir", str(plain_dir),
                        "--model-seed", str(seed),
                        "--attack-seed", str(seed),
                        "--malicious-clients", MALICIOUS_CLIENTS,
                        "--poison-fraction", str(fraction),
                        *COMMON_STATIC_FLAGS,
                        *perf_flags,
                    ],
                    args.dry_run,
                )

                defended_dir = (
                    output_root / f"fraction_{fraction_tag}" / attack_type
                    / f"seed_{seed}" / "trusted_reconstruction"
                )
                ensure_stage(
                    defended_dir / ORCHESTRATOR_MARKER_NAME,
                    [
                        python, str(defense_runner),
                        "--mode", "strong_attack",
                        "--replacement-policy", "trusted_reconstruction",
                        "--attack-type", attack_type,
                        "--plain-branch-dir", str(plain_dir),
                        "--reconstruction-calibration-dir", str(reconstruction_dir),
                        "--data-file", str(data_file),
                        "--partition-file", str(partition_file),
                        "--clean-seed-dir", str(clean_seed_dir),
                        "--warmup-dir", str(warmup_dir),
                        "--output-dir", str(defended_dir),
                        "--model-seed", str(seed),
                        "--attack-seed", str(seed),
                        "--malicious-clients", MALICIOUS_CLIENTS,
                        "--poison-fraction", str(fraction),
                        *COMMON_STATIC_FLAGS,
                        *perf_flags,
                    ],
                    args.dry_run,
                )

    print()
    print("Task 44 C2 orchestration complete." if not args.dry_run else "Dry run complete -- no commands executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

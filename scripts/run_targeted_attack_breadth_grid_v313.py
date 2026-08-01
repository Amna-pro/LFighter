#!/usr/bin/env python3
"""Run exact-baseline frozen V3.13.2 targeted attack breadth."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True, type=Path)
    p.add_argument("--partition-file", required=True, type=Path)
    p.add_argument("--clean-seed-root", required=True, type=Path)
    p.add_argument("--v3101-root", required=True, type=Path)
    p.add_argument("--v3123-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--seeds", default="7,99,123,2026")
    p.add_argument(
        "--source-classes",
        default="BruteForce,DDoS,DoS,Mirai,Recon,Spoofing,Web-Based",
    )
    p.add_argument("--target-class", default="Benign")
    p.add_argument("--coalition-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--evaluation-batch-size", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--probe-per-class", type=int, default=48)
    p.add_argument("--probe-seed", type=int, default=3701)
    p.add_argument("--ema-decay", type=float, default=0.65)
    p.add_argument("--poison-fraction", type=float, default=1.0)
    return p.parse_args()


def run(command: List[str]):
    print("\nRUNNING:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def ensure_stage(output_dir: Path, completion_file: Path, command):
    if completion_file.exists():
        print(f"SKIPPING COMPLETE STAGE: {output_dir}")
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)
    run(command)


def main():
    a = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python = sys.executable
    output_root = a.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]
    sources = [
        x.strip() for x in a.source_classes.split(",") if x.strip()
    ]

    coalition_dir = output_root / "coalitions"
    coalition_file = (
        coalition_dir / "tables" / "v313_frozen_coalition_manifest.csv"
    )
    ensure_stage(
        coalition_dir,
        coalition_dir / "v313_coalition_metadata.json",
        [
            python,
            str(
                project_root / "scripts"
                / "select_targeted_breadth_coalitions_v313.py"
            ),
            "--data-file", str(a.data_file.expanduser().resolve()),
            "--partition-file", str(
                a.partition_file.expanduser().resolve()
            ),
            "--output-dir", str(coalition_dir),
            "--source-classes", ",".join(sources),
            "--target-class", a.target_class,
            "--coalition-size", str(a.coalition_size),
            "--num-clients", "20",
        ],
    )
    coalition_table = pd.read_csv(coalition_file)

    exact_runner = (
        project_root / "scripts" / "run_exact_plain_fedavg_v3132.py"
    )
    defended_runner = (
        project_root / "scripts" / "run_targeted_attack_breadth_v313.py"
    )
    verifier = (
        project_root / "scripts" / "verify_exact_clean_equivalence_v3132.py"
    )
    pair_summarizer = (
        project_root / "scripts"
        / "summarize_targeted_breadth_pair_seed_v313.py"
    )

    for source in sources:
        coalition_row = coalition_table[
            coalition_table["source_class"].eq(source)
            & coalition_table["target_class"].eq(a.target_class)
        ]
        if len(coalition_row) != 1:
            raise RuntimeError(f"Missing coalition for {source}")
        malicious_clients = str(
            coalition_row.iloc[0]["selected_clients"]
        ).replace("|", ",")
        pair_slug = (
            f"{source.lower().replace('-', '_')}"
            f"_to_{a.target_class.lower()}"
        )

        for seed in seeds:
            source_seed_root = (
                a.v3101_root.expanduser().resolve() / f"seed_{seed}"
            )
            original_clean = source_seed_root / "clean_continuation"
            warmup_dir = source_seed_root / "warmup"
            clean_seed_dir = (
                a.clean_seed_root.expanduser().resolve() / f"seed_{seed}"
            )
            calibration_dir = (
                a.v3123_root.expanduser().resolve()
                / f"seed_{seed}" / "calibration"
            )
            seed_root = (
                output_root / "runs" / pair_slug / f"seed_{seed}"
            )
            clean_dir = seed_root / "exact_clean"
            verification_dir = seed_root / "clean_verification"
            plain_dir = seed_root / "plain_attack"
            defended_dir = seed_root / "trusted_reconstruction"
            summary_dir = seed_root / "summary"

            exact_common = [
                "--data-file", str(a.data_file.expanduser().resolve()),
                "--partition-file", str(
                    a.partition_file.expanduser().resolve()
                ),
                "--clean-seed-dir", str(clean_seed_dir),
                "--warmup-dir", str(warmup_dir),
                "--model-seed", str(seed),
                "--num-clients", "20",
                "--continuation-rounds", "4",
                "--batch-size", str(a.batch_size),
                "--evaluation-batch-size",
                str(a.evaluation_batch_size),
                "--learning-rate", str(a.learning_rate),
                "--weight-decay", str(a.weight_decay),
                "--threads", str(a.threads),
                "--probe-per-class", str(a.probe_per_class),
                "--probe-seed", str(a.probe_seed),
                "--source-class", source,
                "--target-class", a.target_class,
                "--ema-decay", str(a.ema_decay),
            ]
            ensure_stage(
                clean_dir,
                clean_dir / "post_warmup_capture_v310_metadata.json",
                [
                    python, str(exact_runner),
                    "--mode", "clean",
                    *exact_common,
                    "--output-dir", str(clean_dir),
                ],
            )
            ensure_stage(
                verification_dir,
                verification_dir
                / "v3132_clean_equivalence_verification.json",
                [
                    python, str(verifier),
                    "--original-clean-dir", str(original_clean),
                    "--current-clean-dir", str(clean_dir),
                    "--output-dir", str(verification_dir),
                    "--seed", str(seed),
                    "--source-class", source,
                    "--target-class", a.target_class,
                ],
            )
            ensure_stage(
                plain_dir,
                plain_dir / "post_warmup_capture_v310_metadata.json",
                [
                    python, str(exact_runner),
                    "--mode", "strong_attack",
                    *exact_common,
                    "--output-dir", str(plain_dir),
                    "--malicious-clients", malicious_clients,
                    "--poison-fraction", str(a.poison_fraction),
                    "--min-source-samples", "1",
                    "--attack-seed", str(seed),
                ],
            )

            defended_common = [
                "--mode", "strong_attack",
                "--replacement-policy", "trusted_reconstruction",
                "--data-file", str(a.data_file.expanduser().resolve()),
                "--partition-file", str(
                    a.partition_file.expanduser().resolve()
                ),
                "--clean-seed-dir", str(clean_seed_dir),
                "--warmup-dir", str(warmup_dir),
                "--reconstruction-calibration-dir",
                str(calibration_dir),
                "--output-dir", str(defended_dir),
                "--model-seed", str(seed),
                "--num-clients", "20",
                "--continuation-rounds", "4",
                "--batch-size", str(a.batch_size),
                "--evaluation-batch-size",
                str(a.evaluation_batch_size),
                "--learning-rate", str(a.learning_rate),
                "--weight-decay", str(a.weight_decay),
                "--threads", str(a.threads),
                "--probe-per-class", str(a.probe_per_class),
                "--probe-seed", str(a.probe_seed),
                "--source-class", source,
                "--target-class", a.target_class,
                "--ema-decay", str(a.ema_decay),
                "--malicious-clients", malicious_clients,
                "--poison-fraction", str(a.poison_fraction),
                "--min-source-samples", "1",
                "--attack-seed", str(seed),
            ]
            ensure_stage(
                defended_dir,
                defended_dir
                / "targeted_attack_breadth_v313_metadata.json",
                [python, str(defended_runner), *defended_common],
            )
            ensure_stage(
                summary_dir,
                summary_dir / "v313_pair_seed_metadata.json",
                [
                    python, str(pair_summarizer),
                    "--clean-dir", str(clean_dir),
                    "--plain-dir", str(plain_dir),
                    "--defended-dir", str(defended_dir),
                    "--output-dir", str(summary_dir),
                    "--seed", str(seed),
                    "--source-class", source,
                    "--target-class", a.target_class,
                    "--coalition-manifest", str(coalition_file),
                ],
            )

    with (output_root / "v3132_grid_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "experiment_version": "3.13.2",
            "seeds": seeds,
            "source_classes": sources,
            "target_class": a.target_class,
            "coalition_size": a.coalition_size,
            "exact_v310_plain_baseline_used": True,
            "frozen_detector": True,
            "frozen_reconstruction_policy": "center_plus_residual",
            "test_sets_accessed": False,
        }, handle, indent=2)

    print("\nV3.13.2 exact-baseline targeted breadth complete")
    print("Output root:", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run frozen V3.13.4 non-Benign target-pair breadth."""
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
    p.add_argument("--task39a-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--seeds", default="7,99,123,2026")
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


def pair_slug(source: str, target: str) -> str:
    return (
        source.lower().replace("-", "_")
        + "_to_"
        + target.lower().replace("-", "_")
    )


def main() -> int:
    a = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python = sys.executable
    output_root = a.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]
    if seeds != [7, 99, 123, 2026]:
        raise ValueError("V3.13.4 is frozen to seeds 7,99,123,2026")

    selection_dir = output_root / "pair_selection"
    pair_manifest = (
        selection_dir / "tables"
        / "v3134_selected_nonbenign_pair_manifest.csv"
    )
    ensure_stage(
        selection_dir,
        selection_dir / "v3134_pair_selection_metadata.json",
        [
            python,
            str(
                project_root / "scripts"
                / "select_nonbenign_target_pairs_v3134.py"
            ),
            "--data-file", str(a.data_file.expanduser().resolve()),
            "--v3101-root", str(a.v3101_root.expanduser().resolve()),
            "--output-dir", str(selection_dir),
            "--seeds", a.seeds,
            "--evaluation-batch-size",
            str(a.evaluation_batch_size),
            "--threads", str(a.threads),
        ],
    )

    coalition_dir = output_root / "coalitions"
    coalition_manifest = (
        coalition_dir / "tables"
        / "v3134_frozen_pair_coalition_manifest.csv"
    )
    ensure_stage(
        coalition_dir,
        coalition_dir / "v3134_coalition_metadata.json",
        [
            python,
            str(
                project_root / "scripts"
                / "select_nonbenign_pair_coalitions_v3134.py"
            ),
            "--data-file", str(a.data_file.expanduser().resolve()),
            "--partition-file",
            str(a.partition_file.expanduser().resolve()),
            "--pair-manifest", str(pair_manifest),
            "--output-dir", str(coalition_dir),
            "--coalition-size", str(a.coalition_size),
            "--num-clients", "20",
        ],
    )

    pairs = pd.read_csv(pair_manifest).sort_values("pair_id")
    coalitions = pd.read_csv(coalition_manifest)

    exact_runner = (
        project_root / "scripts" / "run_exact_plain_fedavg_v3132.py"
    )
    defended_runner = (
        project_root / "scripts" / "run_targeted_attack_breadth_v313.py"
    )
    verifier = (
        project_root / "scripts"
        / "verify_exact_clean_equivalence_v3132.py"
    )
    pair_summarizer = (
        project_root / "scripts"
        / "summarize_targeted_breadth_pair_seed_v313.py"
    )

    for _, pair in pairs.iterrows():
        source = str(pair["source_class"])
        target = str(pair["target_class"])
        pair_name = str(pair["pair_name"])
        pair_coalition = coalitions[
            coalitions["pair_name"].eq(pair_name)
        ]
        if len(pair_coalition) != 1:
            raise RuntimeError(f"Missing coalition for {pair_name}")
        malicious_clients = str(
            pair_coalition.iloc[0]["selected_clients"]
        ).replace("|", ",")
        slug = pair_slug(source, target)

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
            seed_root = output_root / "runs" / slug / f"seed_{seed}"
            clean_dir = seed_root / "exact_clean"
            verification_dir = seed_root / "clean_verification"
            plain_dir = seed_root / "plain_attack"
            defended_dir = seed_root / "trusted_reconstruction"
            summary_dir = seed_root / "summary"

            exact_common = [
                "--data-file", str(a.data_file.expanduser().resolve()),
                "--partition-file",
                str(a.partition_file.expanduser().resolve()),
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
                "--target-class", target,
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
                    "--target-class", target,
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
            ensure_stage(
                defended_dir,
                defended_dir
                / "targeted_attack_breadth_v313_metadata.json",
                [
                    python, str(defended_runner),
                    "--mode", "strong_attack",
                    "--replacement-policy", "trusted_reconstruction",
                    "--data-file",
                    str(a.data_file.expanduser().resolve()),
                    "--partition-file",
                    str(a.partition_file.expanduser().resolve()),
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
                    "--target-class", target,
                    "--ema-decay", str(a.ema_decay),
                    "--malicious-clients", malicious_clients,
                    "--poison-fraction", str(a.poison_fraction),
                    "--min-source-samples", "1",
                    "--attack-seed", str(seed),
                ],
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
                    "--target-class", target,
                    "--coalition-manifest", str(coalition_manifest),
                ],
            )

    summary_dir = output_root / "summary"
    ensure_stage(
        summary_dir,
        summary_dir / "v3134_metadata.json",
        [
            python,
            str(
                project_root / "scripts"
                / "summarize_nonbenign_target_breadth_v3134.py"
            ),
            "--input-root", str(output_root),
            "--pair-manifest", str(pair_manifest),
            "--task39a-root",
            str(a.task39a_root.expanduser().resolve()),
            "--output-dir", str(summary_dir),
            "--seeds", a.seeds,
        ],
    )

    with (output_root / "v3134_grid_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "experiment_version": "3.13.4",
            "stage": "frozen_nonbenign_target_breadth_grid",
            "seed_count": len(seeds),
            "pair_count": int(len(pairs)),
            "method_reopened": False,
            "attack_specific_retuning": False,
            "natural_test_accessed": False,
            "diagnostic_test_accessed": False,
            "summary_dir": str(summary_dir),
        }, handle, indent=2)

    print()
    print("V3.13.4 GRID COMPLETE")
    print(pd.read_csv(
        summary_dir / "tables" / "v3134_task39_decision.csv"
    ).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

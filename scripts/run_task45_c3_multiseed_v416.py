#!/usr/bin/env python3
"""Run the frozen Task 45 C3 curves for seeds 99, 123, and 2026."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


SEEDS = (99, 123, 2026)
C2_TAG = "task45-c2-seed7-frozen-v4162"
COMPLETION = "_task45_condition_complete.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-seed-root", required=True, type=Path)
    parser.add_argument("--warmup-root", required=True, type=Path)
    parser.add_argument("--reconstruction-root", required=True, type=Path)
    parser.add_argument("--coalition-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-incomplete", action="store_true")
    return parser.parse_args()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def semantic_npz_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path) as payload:
        for key in sorted(payload.files):
            value = np.ascontiguousarray(payload[key])
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes())
    return digest.hexdigest()


def execute(command: List[str], branch: Path, marker_name: str, dry_run: bool, overwrite_incomplete: bool) -> None:
    marker = branch / marker_name
    if marker.exists():
        print(f"SKIP COMPLETE: {branch}")
        return
    if branch.exists() and any(branch.iterdir()):
        if not overwrite_incomplete:
            raise RuntimeError(f"Incomplete branch exists; inspect it before any overwrite: {branch}")
        command = [*command, "--overwrite"]
    print("RUN:", subprocess.list2cmdline(command))
    if dry_run:
        return
    subprocess.run(command, check=True)
    adapter_manifest = branch / "task45_dev_only_adapter_manifest.json"
    if not adapter_manifest.exists():
        raise RuntimeError(f"Development only adapter manifest missing: {adapter_manifest}")


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    try:
        c2_commit = git(root, "rev-list", "-n", "1", C2_TAG)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Required C2 tag is missing: {C2_TAG}") from exc
    if subprocess.run(["git", "merge-base", "--is-ancestor", C2_TAG, "HEAD"], cwd=root).returncode != 0:
        raise RuntimeError("C2 freeze tag is not an ancestor of HEAD")
    if git(root, "status", "--porcelain"):
        raise RuntimeError("Working tree must be clean before C3")

    adapter = root / "scripts" / "run_task45_dev_only_adapter_v416.py"
    plain_runner = root / "scripts" / "run_exact_plain_fedavg_v3132.py"
    defense_runner = root / "scripts" / "run_frozen_reconstruction_v3123.py"
    manifest_path = args.coalition_manifest.expanduser().resolve()
    required = [
        adapter, plain_runner, defense_runner, manifest_path,
        args.data_file.expanduser().resolve(), args.partition_file.expanduser().resolve(),
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = pd.read_csv(manifest_path).sort_values(["family_id", "coalition_size"])
    if len(manifest) != 18 or manifest["coalition_hash_sha256"].nunique() != 18:
        raise RuntimeError("C3 requires exactly 18 unique frozen coalition conditions")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    progress_rows: List[Dict[str, object]] = []
    python = sys.executable

    for seed in SEEDS:
        clean_seed_dir = args.clean_seed_root.expanduser().resolve() / f"seed_{seed}"
        warmup_dir = args.warmup_root.expanduser().resolve() / f"seed_{seed}" / "warmup"
        calibration_dir = args.reconstruction_root.expanduser().resolve() / f"seed_{seed}" / "calibration"
        for path in [clean_seed_dir, warmup_dir, calibration_dir]:
            if not path.exists():
                raise FileNotFoundError(path)
        static = [
            "--data-file", str(args.data_file.expanduser().resolve()),
            "--partition-file", str(args.partition_file.expanduser().resolve()),
            "--clean-seed-dir", str(clean_seed_dir),
            "--warmup-dir", str(warmup_dir),
            "--model-seed", str(seed),
            "--num-clients", "20",
            "--continuation-rounds", "4",
            "--batch-size", str(args.batch_size),
            "--evaluation-batch-size", str(args.evaluation_batch_size),
            "--learning-rate", "0.0003",
            "--weight-decay", "0.0001",
            "--threads", str(args.threads),
            "--probe-per-class", "48",
            "--probe-seed", "3701",
            "--source-class", "DDoS",
            "--target-class", "Benign",
            "--ema-decay", "0.65",
            "--poison-fraction", "1.0",
            "--min-source-samples", "1000",
            "--attack-seed", str(seed),
        ]

        for row in manifest.itertuples(index=False):
            clients = str(row.selected_clients).replace("|", ",")
            condition_root = output_root / str(row.family_id) / f"size_{int(row.coalition_size):02d}" / f"seed_{seed}"
            plain_dir = condition_root / "plain_attack"
            defense_dir = condition_root / "trusted_reconstruction"
            plain_command = [
                python, str(adapter), "--runner", str(plain_runner), "--",
                "--mode", "strong_attack", *static,
                "--output-dir", str(plain_dir),
                "--malicious-clients", clients,
            ]
            defense_command = [
                python, str(adapter), "--runner", str(defense_runner), "--",
                "--mode", "strong_attack",
                "--replacement-policy", "trusted_reconstruction",
                *static,
                "--reconstruction-calibration-dir", str(calibration_dir),
                "--output-dir", str(defense_dir),
                "--malicious-clients", clients,
            ]
            execute(plain_command, plain_dir, "post_warmup_capture_v310_metadata.json", args.dry_run, args.overwrite_incomplete)
            execute(defense_command, defense_dir, "trusted_update_reconstruction_v312_metadata.json", args.dry_run, args.overwrite_incomplete)

            completed = False
            if not args.dry_run:
                plain_npz = plain_dir / "attack_manifest" / "poisoned_indices.npz"
                defense_npz = defense_dir / "attack_manifest" / "poisoned_indices.npz"
                if not plain_npz.exists() or not defense_npz.exists():
                    raise FileNotFoundError("One paired poison index manifest is missing")
                plain_hash = semantic_npz_hash(plain_npz)
                defense_hash = semantic_npz_hash(defense_npz)
                if plain_hash != defense_hash:
                    raise RuntimeError(f"Paired poison indices differ for {row.family_id}, size {row.coalition_size}, seed {seed}")
                condition = {
                    "experiment_version": "4.16.C3",
                    "family_id": row.family_id,
                    "coalition_size": int(row.coalition_size),
                    "selected_clients": clients,
                    "coalition_hash_sha256": row.coalition_hash_sha256,
                    "seed": seed,
                    "plain_poison_semantic_sha256": plain_hash,
                    "defense_poison_semantic_sha256": defense_hash,
                    "exact_poison_pair": True,
                    "reserved_test_arrays_materialized": False,
                    "complete": True,
                }
                condition_root.mkdir(parents=True, exist_ok=True)
                (condition_root / COMPLETION).write_text(json.dumps(condition, indent=2), encoding="utf-8")
                completed = True
            progress_rows.append({
                "family_id": row.family_id,
                "coalition_size": int(row.coalition_size),
                "seed": seed,
                "coalition_hash_sha256": row.coalition_hash_sha256,
                "completed": completed,
            })
            pd.DataFrame(progress_rows).to_csv(output_root / "task45c3_progress.csv", index=False)

    metadata = {
        "experiment_version": "4.16.C3",
        "stage": "task45_confirmatory_multiseed_coalition_size_curve",
        "c2_tag": C2_TAG,
        "c2_commit": c2_commit,
        "seeds": list(SEEDS),
        "condition_count": 54,
        "paired_branch_count": 108,
        "dry_run": bool(args.dry_run),
        "frozen_detector": True,
        "frozen_reconstruction": "center_plus_residual",
        "development_only_adapter_required": True,
        "reserved_test_arrays_materialized": False,
    }
    if not args.dry_run:
        (output_root / "task45c3_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Task 45 C3 dry run complete" if args.dry_run else "Task 45 C3 confirmatory multiseed execution complete")
    print("SEEDS:", ",".join(map(str, SEEDS)))
    print("CONDITIONS:", len(progress_rows))
    print("PAIRED BRANCHES:", len(progress_rows) * 2)
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

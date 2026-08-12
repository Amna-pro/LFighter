#!/usr/bin/env python3
"""Task 45 C1 read-only interface, evidence, and pairing preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


C0_COMMIT = "f6938e6"
C0_TAG = "task45-c0-preregistered-v4160"
SEEDS = [7, 99, 123, 2026]
SIZES = [1, 2, 4, 6, 8, 10]
ELIGIBLE = [1, 3, 5, 7, 8, 10, 13, 14, 15, 16, 17, 18, 19]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True, type=Path)
    p.add_argument("--data-file", required=True, type=Path)
    p.add_argument("--partition-file", required=True, type=Path)
    p.add_argument("--clean-seed-root", required=True, type=Path)
    p.add_argument("--warmup-root", required=True, type=Path)
    p.add_argument("--reconstruction-root", required=True, type=Path)
    p.add_argument("--coalition-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def partition_hash(path: Path, expected_clients: int, train_rows: int) -> Tuple[str, List[np.ndarray]]:
    with np.load(path) as payload:
        keys = sorted(payload.files)
        expected = [f"client_{client_id:03d}" for client_id in range(expected_clients)]
        if keys != expected:
            raise ValueError("Stable partition keys differ from client_000 through client_019")
        parts = [np.asarray(payload[key], dtype=np.int64) for key in keys]
    all_indices = np.concatenate(parts)
    unique = np.unique(all_indices)
    if len(all_indices) != train_rows or len(unique) != train_rows:
        raise ValueError("Partition does not cover every training row exactly once")
    if int(unique.min()) != 0 or int(unique.max()) != train_rows - 1:
        raise ValueError("Partition range mismatch")
    digest = hashlib.sha256()
    for client_id, indices in enumerate(parts):
        digest.update(f"client_{client_id:03d}".encode("utf-8"))
        digest.update(indices.tobytes())
    return digest.hexdigest(), parts


def main() -> int:
    a = parse_args()
    root = a.project_root.expanduser().resolve()
    data_file = a.data_file.expanduser().resolve()
    partition_file = a.partition_file.expanduser().resolve()
    clean_root = a.clean_seed_root.expanduser().resolve()
    warmup_root = a.warmup_root.expanduser().resolve()
    reconstruction_root = a.reconstruction_root.expanduser().resolve()
    coalition_root = a.coalition_root.expanduser().resolve()
    output = a.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    protocol_path = root / "configs" / "task45_preregistration_v4160.json"
    manifest_path = coalition_root / "tables" / "task45_coalition_manifest.csv"
    coalition_metadata_path = coalition_root / "task45_coalition_metadata.json"
    adapter_path = root / "scripts" / "run_task45_dev_only_adapter_v416.py"
    orchestrator_path = root / "scripts" / "run_task45_c2_seed7_v416.py"
    plain_path = root / "scripts" / "run_exact_plain_fedavg_v3132.py"
    defense_path = root / "scripts" / "run_frozen_reconstruction_v3123.py"
    for path in [data_file, partition_file, protocol_path, manifest_path, coalition_metadata_path, adapter_path, orchestrator_path, plain_path, defense_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    checks: List[Dict[str, object]] = []
    def add(name: str, passed: bool, detail: object) -> None:
        checks.append({"check_name": name, "passed": bool(passed), "detail": str(detail)})
        if not passed:
            raise RuntimeError(f"{name} failed: {detail}")

    tag_commit = git(root, "rev-list", "-n", "1", C0_TAG)
    add("c0_tag_commit", tag_commit.startswith(C0_COMMIT), tag_commit)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", C0_TAG, "HEAD"], cwd=root
    ).returncode == 0
    add("c0_tag_is_ancestor", ancestor, git(root, "rev-parse", "HEAD"))

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    coalition_metadata = json.loads(coalition_metadata_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(manifest_path)
    add("protocol_version", protocol["experiment_version"] == "4.16.0", protocol["experiment_version"])
    add("manifest_rows", len(manifest) == 18, len(manifest))
    add("families", manifest["family_id"].nunique() == 3, manifest["family_id"].nunique())
    add("sizes", sorted(manifest["coalition_size"].unique().tolist()) == SIZES, sorted(manifest["coalition_size"].unique().tolist()))
    add("condition_hashes_unique", manifest["coalition_hash_sha256"].nunique() == 18, manifest["coalition_hash_sha256"].nunique())
    add("metadata_conditions", coalition_metadata["coalition_condition_count"] == 18, coalition_metadata["coalition_condition_count"])
    add("metadata_no_training", coalition_metadata["training_started"] is False, coalition_metadata["training_started"])
    add("metadata_no_reserved", coalition_metadata["reserved_test_arrays_materialized"] is False, coalition_metadata["reserved_test_arrays_materialized"])
    add("protocol_hash", coalition_metadata["protocol_sha256"] == sha256(protocol_path), coalition_metadata["protocol_sha256"])
    for family, group in manifest.groupby("family_id"):
        previous = set()
        for row in group.sort_values("coalition_size").itertuples():
            current = {int(value) for value in str(row.selected_clients).split("|")}
            add(f"{family}_{row.coalition_size}_count", len(current) == int(row.coalition_size), sorted(current))
            add(f"{family}_{row.coalition_size}_nested", previous.issubset(current), sorted(current))
            add(f"{family}_{row.coalition_size}_eligible", current.issubset(set(ELIGIBLE)), sorted(current))
            previous = current

    with np.load(data_file) as payload:
        add("y_train_present", "y_train" in payload.files, payload.files)
        y_train = payload["y_train"].astype(np.int64, copy=False)
    computed_partition_hash, parts = partition_hash(partition_file, 20, len(y_train))
    add("partition_hash", computed_partition_hash == coalition_metadata["partition_hash_sha256"], computed_partition_hash)
    counts = [int(np.sum(y_train[indices] == 2)) for indices in parts]
    actual_eligible = [client_id for client_id, count in enumerate(counts) if count >= 1000]
    add("eligible_clients", actual_eligible == ELIGIBLE, actual_eligible)

    adapter_source = adapter_path.read_text(encoding="utf-8")
    add("adapter_approved_plain", "run_exact_plain_fedavg_v3132.py" in adapter_source, "plain")
    add("adapter_approved_defense", "run_frozen_reconstruction_v3123.py" in adapter_source, "defense")
    add("adapter_reserved_block", "reserved_arrays_materialized" in adapter_source, "adapter manifest")
    orchestrator_source = orchestrator_path.read_text(encoding="utf-8")
    add("orchestrator_uses_adapter", "run_task45_dev_only_adapter_v416.py" in orchestrator_source, "adapter")
    add("orchestrator_seed7_only", "SEED = 7" in orchestrator_source, "seed 7")
    add("orchestrator_minimum_1000", '"1000"' in orchestrator_source, "min source rows")

    seed_rows = []
    for seed in SEEDS:
        clean_seed_dir = clean_root / f"seed_{seed}"
        warmup_dir = warmup_root / f"seed_{seed}" / "warmup"
        calibration_dir = reconstruction_root / f"seed_{seed}" / "calibration"
        clean_metadata_path = clean_seed_dir / "seed_metadata.json"
        warmup_metadata_path = warmup_dir / "true_warmup_v310_metadata.json"
        checkpoint_path = warmup_dir / "checkpoints" / "common_round4_warmup_model.pt"
        reconstruction_path = calibration_dir / "calibration" / "trusted_update_reconstruction_profiles.pt"
        reconstruction_summary = calibration_dir / "calibration" / "reconstruction_calibration_summary.csv"
        for path in [clean_metadata_path, warmup_metadata_path, checkpoint_path, reconstruction_path, reconstruction_summary]:
            add(f"seed_{seed}_{path.name}_exists", path.exists(), path)
        clean_metadata = json.loads(clean_metadata_path.read_text(encoding="utf-8"))
        warmup_metadata = json.loads(warmup_metadata_path.read_text(encoding="utf-8"))
        bundle = torch.load(reconstruction_path, map_location="cpu", weights_only=False)
        checkpoint_hash = sha256(checkpoint_path)
        add(f"seed_{seed}_clean_partition", clean_metadata["partition_hash_sha256"] == computed_partition_hash, clean_metadata["partition_hash_sha256"])
        add(f"seed_{seed}_warmup_partition", warmup_metadata["partition_hash_sha256"] == computed_partition_hash, warmup_metadata["partition_hash_sha256"])
        add(f"seed_{seed}_reconstruction_partition", bundle["partition_hash"] == computed_partition_hash, bundle["partition_hash"])
        add(f"seed_{seed}_bundle_seed", int(bundle["model_seed"]) == seed, bundle["model_seed"])
        add(f"seed_{seed}_checkpoint_hash", bundle["common_warmup_checkpoint_sha256"] == checkpoint_hash, checkpoint_hash)
        add(f"seed_{seed}_policy", bundle["selected_policy"] == "center_plus_residual", bundle["selected_policy"])
        seed_rows.append({"seed": seed, "partition_hash_sha256": computed_partition_hash, "warmup_checkpoint_sha256": checkpoint_hash, "reconstruction_profile_sha256": sha256(reconstruction_path), "selected_policy": bundle["selected_policy"]})

    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(tables / "task45c1_preflight_checks.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(tables / "task45c1_seed_artifact_manifest.csv", index=False)
    manifest.to_csv(tables / "task45c1_frozen_coalition_manifest.csv", index=False)
    source_manifest = []
    for path in [protocol_path, manifest_path, coalition_metadata_path, adapter_path, orchestrator_path, plain_path, defense_path]:
        source_manifest.append({"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(source_manifest).to_csv(tables / "task45c1_source_manifest_sha256.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for family, group in manifest.groupby("family_id", sort=False):
        ax.plot(group["coalition_size"], group["global_source_exposure_fraction"], marker="o", label=family)
    ax.set_xlabel("Coalition size")
    ax.set_ylabel("Global DDoS source row exposure")
    ax.set_title("Task 45 C1 frozen coalition identities")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "task45c1_coalition_exposure.png", dpi=300)
    fig.savefig(figures / "task45c1_coalition_exposure.pdf")
    plt.close(fig)

    decision = {
        "experiment_version": "4.16.C1",
        "stage": "task45_c1_interface_and_coalition_preflight",
        "checks_passed": int(check_frame["passed"].sum()),
        "check_count": len(check_frame),
        "coalition_conditions_frozen": 18,
        "all_seed_artifacts_verified": True,
        "ready_for_c1_freeze_commit": True,
        "training_started": False,
        "reserved_test_arrays_materialized": False,
        "next_stage": "Commit and tag C1, then run the frozen seed 7 C2 orchestrator.",
    }
    (output / "task45c1_preflight_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("===== TASK 45 C1 PREFLIGHT =====")
    print(f"Checks passed: {decision['checks_passed']}/{decision['check_count']}")
    print("COALITION CONDITIONS FROZEN: 18")
    print("ALL SEED ARTIFACTS VERIFIED: True")
    print("READY FOR C1 FREEZE COMMIT: True")
    print("TRAINING STARTED: False")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

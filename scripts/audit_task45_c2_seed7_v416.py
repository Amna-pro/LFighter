#!/usr/bin/env python3
"""Audit and summarize the preregistered Task 45 C2 seed 7 curve.

This stage is read only with respect to experiment branches. It verifies all
18 frozen conditions, exact poison pairing, development only array isolation,
and the four round output schemas before C3 may begin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


C1_TAG = "task45-c1-preflight-frozen-v4161"
SEED = 7
SIZES = [1, 2, 4, 6, 8, 10]
FAMILIES = ["A_development_anchor", "B_hash_ranked", "C_hash_ranked"]
ALLOWED_ARRAYS = {"X_train", "y_train", "X_val", "y_val"}
COMPLETION = "_task45_condition_complete.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--coalition-manifest", required=True, type=Path)
    parser.add_argument("--warmup-root", required=True, type=Path)
    parser.add_argument("--c2-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_rounds(path: Path, filename: str) -> pd.DataFrame:
    full = path / "tables" / filename
    if not full.exists():
        raise FileNotFoundError(full)
    table = pd.read_csv(full).sort_values("monitoring_round").reset_index(drop=True)
    required = {
        "monitoring_round",
        "val_macro_f1",
        "val_source_to_target_rate",
        "benign_false_positive_rate",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing columns in {full}: {missing}")
    if table["monitoring_round"].tolist() != [1, 2, 3, 4]:
        raise ValueError(f"Expected monitoring rounds 1 through 4 in {full}")
    return table


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    manifest_path = args.coalition_manifest.expanduser().resolve()
    warmup_root = args.warmup_root.expanduser().resolve()
    c2_root = args.c2_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    checks: List[Dict[str, object]] = []

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append({"check_name": name, "passed": bool(passed), "detail": str(detail)})
        if not passed:
            raise RuntimeError(f"{name} failed: {detail}")

    tag_commit = git(root, "rev-list", "-n", "1", C1_TAG)
    add("c1_tag_resolves", len(tag_commit) == 40, tag_commit)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", C1_TAG, "HEAD"], cwd=root
    ).returncode == 0
    add("c1_tag_is_ancestor", ancestor, git(root, "rev-parse", "HEAD"))

    for path in [manifest_path, c2_root / "task45c2_progress.csv", c2_root / "task45c2_run_metadata.json"]:
        add(f"exists_{path.name}", path.exists(), path)

    manifest = pd.read_csv(manifest_path).sort_values(["family_id", "coalition_size"])
    add("manifest_rows", len(manifest) == 18, len(manifest))
    add("manifest_families", sorted(manifest["family_id"].unique()) == sorted(FAMILIES), manifest["family_id"].unique())
    add("manifest_sizes", sorted(manifest["coalition_size"].unique().tolist()) == SIZES, manifest["coalition_size"].unique())
    add("manifest_hashes_unique", manifest["coalition_hash_sha256"].nunique() == 18, manifest["coalition_hash_sha256"].nunique())

    progress = pd.read_csv(c2_root / "task45c2_progress.csv")
    add("progress_rows", len(progress) == 18, len(progress))
    add("progress_all_complete", bool(progress["completed"].all()), progress["completed"].value_counts().to_dict())
    add("progress_seed7", set(progress["seed"].astype(int)) == {SEED}, sorted(progress["seed"].unique()))
    run_metadata = json.loads((c2_root / "task45c2_run_metadata.json").read_text(encoding="utf-8"))
    add("run_condition_count", int(run_metadata["condition_count"]) == 18, run_metadata["condition_count"])
    add("run_paired_branch_count", int(run_metadata["paired_branch_count"]) == 36, run_metadata["paired_branch_count"])
    add("run_seed7", int(run_metadata["seed"]) == SEED, run_metadata["seed"])
    add("run_not_dry", run_metadata["dry_run"] is False, run_metadata["dry_run"])
    add("run_no_reserved", run_metadata["reserved_test_arrays_materialized"] is False, run_metadata["reserved_test_arrays_materialized"])

    clean_dir = warmup_root / f"seed_{SEED}" / "clean_continuation"
    clean = load_rounds(clean_dir, "continuation_round_metrics.csv")
    add("clean_four_rounds", len(clean) == 4, len(clean))

    rows: List[Dict[str, object]] = []
    source_paths = [manifest_path, c2_root / "task45c2_progress.csv", c2_root / "task45c2_run_metadata.json"]
    for row in manifest.itertuples(index=False):
        family = str(row.family_id)
        size = int(row.coalition_size)
        expected_clients = sorted(int(value) for value in str(row.selected_clients).split("|"))
        condition_root = c2_root / family / f"size_{size:02d}" / f"seed_{SEED}"
        plain_dir = condition_root / "plain_attack"
        defense_dir = condition_root / "trusted_reconstruction"
        condition_path = condition_root / COMPLETION
        plain_metadata_path = plain_dir / "post_warmup_capture_v310_metadata.json"
        defense_metadata_path = defense_dir / "trusted_update_reconstruction_v312_metadata.json"
        plain_adapter_path = plain_dir / "task45_dev_only_adapter_manifest.json"
        defense_adapter_path = defense_dir / "task45_dev_only_adapter_manifest.json"
        plain_npz = plain_dir / "attack_manifest" / "poisoned_indices.npz"
        defense_npz = defense_dir / "attack_manifest" / "poisoned_indices.npz"
        plain_table_path = plain_dir / "tables" / "continuation_round_metrics.csv"
        defense_table_path = defense_dir / "tables" / "reconstruction_round_metrics.csv"
        required_paths = [
            condition_path, plain_metadata_path, defense_metadata_path,
            plain_adapter_path, defense_adapter_path, plain_npz, defense_npz,
            plain_table_path, defense_table_path,
        ]
        for path in required_paths:
            add(f"{family}_{size:02d}_exists_{path.name}", path.exists(), path)
        source_paths.extend(required_paths)

        condition = json.loads(condition_path.read_text(encoding="utf-8"))
        plain_meta = json.loads(plain_metadata_path.read_text(encoding="utf-8"))
        defense_meta = json.loads(defense_metadata_path.read_text(encoding="utf-8"))
        plain_adapter = json.loads(plain_adapter_path.read_text(encoding="utf-8"))
        defense_adapter = json.loads(defense_adapter_path.read_text(encoding="utf-8"))
        add(f"{family}_{size:02d}_condition_complete", condition["complete"] is True, condition["complete"])
        add(f"{family}_{size:02d}_condition_family", condition["family_id"] == family, condition["family_id"])
        add(f"{family}_{size:02d}_condition_size", int(condition["coalition_size"]) == size, condition["coalition_size"])
        add(f"{family}_{size:02d}_condition_seed", int(condition["seed"]) == SEED, condition["seed"])
        add(f"{family}_{size:02d}_coalition_hash", condition["coalition_hash_sha256"] == row.coalition_hash_sha256, condition["coalition_hash_sha256"])
        add(f"{family}_{size:02d}_condition_clients", sorted(int(value) for value in str(condition["selected_clients"]).split(",")) == expected_clients, condition["selected_clients"])
        add(f"{family}_{size:02d}_condition_no_reserved", condition["reserved_test_arrays_materialized"] is False, condition["reserved_test_arrays_materialized"])

        plain_hash = semantic_npz_hash(plain_npz)
        defense_hash = semantic_npz_hash(defense_npz)
        add(f"{family}_{size:02d}_semantic_poison_pair", plain_hash == defense_hash, plain_hash)
        add(f"{family}_{size:02d}_marker_plain_hash", condition["plain_poison_semantic_sha256"] == plain_hash, condition["plain_poison_semantic_sha256"])
        add(f"{family}_{size:02d}_marker_defense_hash", condition["defense_poison_semantic_sha256"] == defense_hash, condition["defense_poison_semantic_sha256"])

        for arm, meta, adapter in [("plain", plain_meta, plain_adapter), ("defense", defense_meta, defense_adapter)]:
            add(f"{family}_{size:02d}_{arm}_seed", int(meta["model_seed"]) == SEED, meta["model_seed"])
            add(f"{family}_{size:02d}_{arm}_mode", meta["mode"] == "strong_attack", meta["mode"])
            add(f"{family}_{size:02d}_{arm}_test_false", meta["test_sets_accessed"] is False, meta["test_sets_accessed"])
            add(f"{family}_{size:02d}_{arm}_adapter_reserved_false", adapter["reserved_arrays_materialized"] is False, adapter["reserved_arrays_materialized"])
            add(f"{family}_{size:02d}_{arm}_adapter_return", int(adapter["runner_return_code"]) == 0, adapter["runner_return_code"])
            add(f"{family}_{size:02d}_{arm}_adapter_arrays", set(adapter["allowed_arrays_materialized"]) == ALLOWED_ARRAYS, sorted(adapter["allowed_arrays_materialized"]))
        add(f"{family}_{size:02d}_plain_clients", sorted(map(int, plain_meta["malicious_clients"])) == expected_clients, plain_meta["malicious_clients"])
        add(f"{family}_{size:02d}_defense_policy", defense_meta["replacement_policy"] == "trusted_reconstruction", defense_meta["replacement_policy"])
        add(f"{family}_{size:02d}_reconstruction_policy", defense_meta["selected_reconstruction_policy"] == "center_plus_residual", defense_meta["selected_reconstruction_policy"])

        plain = load_rounds(plain_dir, "continuation_round_metrics.csv")
        defense = load_rounds(defense_dir, "reconstruction_round_metrics.csv")
        add(f"{family}_{size:02d}_defense_recall_column", "malicious_recall" in defense.columns, defense.columns.tolist())
        clean_rate = float(clean["val_source_to_target_rate"].mean())
        plain_rate = float(plain["val_source_to_target_rate"].mean())
        defense_rate = float(defense["val_source_to_target_rate"].mean())
        attack_excess = plain_rate - clean_rate
        removed = ((plain_rate - defense_rate) / attack_excess) if attack_excess > 1e-12 else float("nan")
        rows.append({
            "family_id": family,
            "coalition_size": size,
            "seed": SEED,
            "selected_clients": row.selected_clients,
            "coalition_hash_sha256": row.coalition_hash_sha256,
            "clean_mean_source_to_target_rate": clean_rate,
            "plain_mean_source_to_target_rate": plain_rate,
            "defended_mean_source_to_target_rate": defense_rate,
            "plain_attack_excess_over_clean": attack_excess,
            "defended_residual_excess_over_clean": defense_rate - clean_rate,
            "damage_removed_fraction": removed,
            "rounds_improved_vs_plain": int((defense["val_source_to_target_rate"] < plain["val_source_to_target_rate"]).sum()),
            "mean_malicious_recall": float(defense["malicious_recall"].mean()),
            "minimum_malicious_recall": float(defense["malicious_recall"].min()),
            "mean_benign_fpr": float(defense["benign_false_positive_rate"].mean()),
            "maximum_benign_fpr": float(defense["benign_false_positive_rate"].max()),
            "clean_mean_macro_f1": float(clean["val_macro_f1"].mean()),
            "plain_mean_macro_f1": float(plain["val_macro_f1"].mean()),
            "defended_mean_macro_f1": float(defense["val_macro_f1"].mean()),
            "plain_total_seconds": float(plain_meta["total_seconds"]),
            "defended_total_seconds": float(defense_meta["total_seconds"]),
            "exact_poison_pair": True,
            "reserved_test_arrays_materialized": False,
        })

    summary = pd.DataFrame(rows).sort_values(["family_id", "coalition_size"])
    add("summary_rows", len(summary) == 18, len(summary))
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(tables / "task45c2_seed7_audit_checks.csv", index=False)
    summary.to_csv(tables / "task45c2_seed7_condition_summary.csv", index=False)

    source_rows = []
    seen = set()
    for path in source_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            relative = str(resolved)
        source_rows.append({"relative_path": relative, "bytes": resolved.stat().st_size, "sha256": file_sha256(resolved)})
    pd.DataFrame(source_rows).sort_values("relative_path").to_csv(
        tables / "task45c2_seed7_source_manifest_sha256.csv", index=False
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    for family, group in summary.groupby("family_id", sort=False):
        group = group.sort_values("coalition_size")
        axes[0].plot(group["coalition_size"], group["plain_attack_excess_over_clean"], marker="o", label=family)
        axes[1].plot(group["coalition_size"], group["damage_removed_fraction"], marker="o", label=family)
    axes[0].axhline(0.05, color="#C44E52", linestyle="--", linewidth=1)
    axes[0].set_title("Seed 7 plain attack excess")
    axes[0].set_ylabel("Source to target excess")
    axes[1].axhline(0.0, color="#C44E52", linestyle="--", linewidth=1)
    axes[1].set_title("Seed 7 damage removed fraction")
    for axis in axes:
        axis.set_xlabel("Malicious coalition size")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "task45c2_seed7_coalition_curves.png", dpi=300, bbox_inches="tight")
    figure.savefig(figures / "task45c2_seed7_coalition_curves.pdf", bbox_inches="tight")
    plt.close(figure)

    decision = {
        "experiment_version": "4.16.C2",
        "stage": "task45_c2_seed7_audit_and_development_summary",
        "checks_passed": int(checks_frame["passed"].sum()),
        "check_count": int(len(checks_frame)),
        "conditions_verified": int(len(summary)),
        "paired_branches_verified": int(len(summary) * 2),
        "seed": SEED,
        "exact_poison_pairing_all": True,
        "reserved_test_arrays_materialized": False,
        "c2_outcomes_used_to_change_c3": False,
        "ready_for_c2_freeze_commit": True,
        "ready_for_c3_confirmatory_multiseed": True,
    }
    (output / "task45c2_seed7_audit_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print("===== TASK 45 C2 SEED 7 AUDIT =====")
    print(f"Checks passed: {decision['checks_passed']}/{decision['check_count']}")
    print("CONDITIONS VERIFIED: 18")
    print("PAIRED BRANCHES VERIFIED: 36")
    print("EXACT POISON PAIRING: True")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR C3 CONFIRMATORY MULTISEED: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

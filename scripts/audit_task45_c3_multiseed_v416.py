#!/usr/bin/env python3
"""Read only integrity audit for the complete Task 45 C2 and C3 matrix."""
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


C2_TAG = "task45-c2-seed7-frozen-v4162"
SEEDS = (7, 99, 123, 2026)
FAMILIES = ("A_development_anchor", "B_hash_ranked", "C_hash_ranked")
SIZES = (1, 2, 4, 6, 8, 10)
ALLOWED_ARRAYS = {"X_train", "y_train", "X_val", "y_val"}
COMPLETION = "_task45_condition_complete.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True, type=Path)
    p.add_argument("--coalition-manifest", required=True, type=Path)
    p.add_argument("--c2-root", required=True, type=Path)
    p.add_argument("--c3-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


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


def load_rounds(branch: Path, filename: str) -> pd.DataFrame:
    path = branch / "tables" / filename
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path).sort_values("monitoring_round").reset_index(drop=True)
    required = {
        "monitoring_round", "val_macro_f1", "val_source_to_target_rate",
        "benign_false_positive_rate",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    if frame["monitoring_round"].astype(int).tolist() != [1, 2, 3, 4]:
        raise ValueError(f"Expected monitoring rounds 1 through 4 in {path}")
    return frame


def main() -> int:
    a = parse_args()
    root = a.project_root.expanduser().resolve()
    manifest_path = a.coalition_manifest.expanduser().resolve()
    c2_root = a.c2_root.expanduser().resolve()
    c3_root = a.c3_root.expanduser().resolve()
    output = a.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    checks: List[Dict[str, object]] = []
    sources: List[Path] = []

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append({"check_name": name, "passed": bool(passed), "detail": str(detail)})
        if not passed:
            raise RuntimeError(f"{name} failed: {detail}")

    tag_commit = git(root, "rev-list", "-n", "1", C2_TAG)
    add("c2_tag_resolves", len(tag_commit) == 40, tag_commit)
    add(
        "c2_tag_is_ancestor",
        subprocess.run(["git", "merge-base", "--is-ancestor", C2_TAG, "HEAD"], cwd=root).returncode == 0,
        git(root, "rev-parse", "HEAD"),
    )

    for path in [manifest_path, c2_root / "task45c2_run_metadata.json", c3_root / "task45c3_run_metadata.json", c3_root / "task45c3_progress.csv"]:
        add(f"exists_{path.name}", path.exists(), path)
        sources.append(path)

    manifest = pd.read_csv(manifest_path).sort_values(["family_id", "coalition_size"])
    add("manifest_rows", len(manifest) == 18, len(manifest))
    add("manifest_families", sorted(manifest["family_id"].unique()) == sorted(FAMILIES), manifest["family_id"].unique())
    add("manifest_sizes", sorted(manifest["coalition_size"].astype(int).unique()) == list(SIZES), manifest["coalition_size"].unique())
    add("manifest_hashes_unique", manifest["coalition_hash_sha256"].nunique() == 18, manifest["coalition_hash_sha256"].nunique())

    c2_meta = json.loads((c2_root / "task45c2_run_metadata.json").read_text(encoding="utf-8"))
    c3_meta = json.loads((c3_root / "task45c3_run_metadata.json").read_text(encoding="utf-8"))
    add("c2_seed", int(c2_meta["seed"]) == 7, c2_meta["seed"])
    add("c2_conditions", int(c2_meta["condition_count"]) == 18, c2_meta["condition_count"])
    add("c2_pairs", int(c2_meta["paired_branch_count"]) == 36, c2_meta["paired_branch_count"])
    add("c2_no_reserved", c2_meta["reserved_test_arrays_materialized"] is False, c2_meta["reserved_test_arrays_materialized"])
    add("c3_seeds", tuple(map(int, c3_meta["seeds"])) == (99, 123, 2026), c3_meta["seeds"])
    add("c3_conditions", int(c3_meta["condition_count"]) == 54, c3_meta["condition_count"])
    add("c3_pairs", int(c3_meta["paired_branch_count"]) == 108, c3_meta["paired_branch_count"])
    add("c3_not_dry", c3_meta["dry_run"] is False, c3_meta["dry_run"])
    add("c3_no_reserved", c3_meta["reserved_test_arrays_materialized"] is False, c3_meta["reserved_test_arrays_materialized"])
    progress = pd.read_csv(c3_root / "task45c3_progress.csv")
    add("c3_progress_rows", len(progress) == 54, len(progress))
    add("c3_progress_complete", bool(progress["completed"].all()), progress["completed"].value_counts().to_dict())
    add("c3_progress_seeds", sorted(progress["seed"].astype(int).unique()) == [99, 123, 2026], progress["seed"].unique())

    audit_rows: List[Dict[str, object]] = []
    seen_pairs = set()
    for seed in SEEDS:
        stage_root = c2_root if seed == 7 else c3_root
        for row in manifest.itertuples(index=False):
            family = str(row.family_id)
            size = int(row.coalition_size)
            condition_root = stage_root / family / f"size_{size:02d}" / f"seed_{seed}"
            plain = condition_root / "plain_attack"
            defense = condition_root / "trusted_reconstruction"
            paths = {
                "marker": condition_root / COMPLETION,
                "plain_meta": plain / "post_warmup_capture_v310_metadata.json",
                "defense_meta": defense / "trusted_update_reconstruction_v312_metadata.json",
                "plain_adapter": plain / "task45_dev_only_adapter_manifest.json",
                "defense_adapter": defense / "task45_dev_only_adapter_manifest.json",
                "plain_npz": plain / "attack_manifest" / "poisoned_indices.npz",
                "defense_npz": defense / "attack_manifest" / "poisoned_indices.npz",
                "plain_table": plain / "tables" / "continuation_round_metrics.csv",
                "defense_table": defense / "tables" / "reconstruction_round_metrics.csv",
            }
            for label, path in paths.items():
                add(f"{family}_{size:02d}_seed{seed}_{label}_exists", path.exists(), path)
                sources.append(path)
            marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
            plain_meta = json.loads(paths["plain_meta"].read_text(encoding="utf-8"))
            defense_meta = json.loads(paths["defense_meta"].read_text(encoding="utf-8"))
            plain_adapter = json.loads(paths["plain_adapter"].read_text(encoding="utf-8"))
            defense_adapter = json.loads(paths["defense_adapter"].read_text(encoding="utf-8"))
            expected_clients = sorted(int(v) for v in str(row.selected_clients).split("|"))
            marker_clients = sorted(int(v) for v in str(marker["selected_clients"]).split(","))
            add(f"{family}_{size:02d}_seed{seed}_marker_complete", marker["complete"] is True, marker["complete"])
            add(f"{family}_{size:02d}_seed{seed}_marker_identity", marker["family_id"] == family and int(marker["coalition_size"]) == size and int(marker["seed"]) == seed, marker)
            add(f"{family}_{size:02d}_seed{seed}_marker_hash", marker["coalition_hash_sha256"] == row.coalition_hash_sha256, marker["coalition_hash_sha256"])
            add(f"{family}_{size:02d}_seed{seed}_marker_clients", marker_clients == expected_clients, marker_clients)
            add(f"{family}_{size:02d}_seed{seed}_marker_no_reserved", marker["reserved_test_arrays_materialized"] is False, marker["reserved_test_arrays_materialized"])
            plain_hash = semantic_npz_hash(paths["plain_npz"])
            defense_hash = semantic_npz_hash(paths["defense_npz"])
            add(f"{family}_{size:02d}_seed{seed}_poison_pair", plain_hash == defense_hash, plain_hash)
            add(f"{family}_{size:02d}_seed{seed}_marker_poison_hashes", marker["plain_poison_semantic_sha256"] == plain_hash and marker["defense_poison_semantic_sha256"] == defense_hash, plain_hash)
            pair_key = (family, size, seed, plain_hash)
            add(f"{family}_{size:02d}_seed{seed}_pair_unique", pair_key not in seen_pairs, pair_key)
            seen_pairs.add(pair_key)
            for arm, meta, adapter in [("plain", plain_meta, plain_adapter), ("defense", defense_meta, defense_adapter)]:
                add(f"{family}_{size:02d}_seed{seed}_{arm}_seed", int(meta["model_seed"]) == seed, meta["model_seed"])
                add(f"{family}_{size:02d}_seed{seed}_{arm}_mode", meta["mode"] == "strong_attack", meta["mode"])
                add(f"{family}_{size:02d}_seed{seed}_{arm}_test_false", meta["test_sets_accessed"] is False, meta["test_sets_accessed"])
                add(f"{family}_{size:02d}_seed{seed}_{arm}_adapter_return", int(adapter["runner_return_code"]) == 0, adapter["runner_return_code"])
                add(f"{family}_{size:02d}_seed{seed}_{arm}_adapter_arrays", set(adapter["allowed_arrays_materialized"]) == ALLOWED_ARRAYS, sorted(adapter["allowed_arrays_materialized"]))
                add(f"{family}_{size:02d}_seed{seed}_{arm}_adapter_no_reserved", adapter["reserved_arrays_materialized"] is False, adapter["reserved_arrays_materialized"])
            add(f"{family}_{size:02d}_seed{seed}_plain_clients", sorted(map(int, plain_meta["malicious_clients"])) == expected_clients, plain_meta["malicious_clients"])
            add(f"{family}_{size:02d}_seed{seed}_defense_policy", defense_meta["replacement_policy"] == "trusted_reconstruction" and defense_meta["selected_reconstruction_policy"] == "center_plus_residual", defense_meta["replacement_policy"])
            plain_rounds = load_rounds(plain, "continuation_round_metrics.csv")
            defense_rounds = load_rounds(defense, "reconstruction_round_metrics.csv")
            add(f"{family}_{size:02d}_seed{seed}_recall_column", "malicious_recall" in defense_rounds.columns, defense_rounds.columns.tolist())
            audit_rows.append({
                "family_id": family,
                "coalition_size": size,
                "seed": seed,
                "coalition_hash_sha256": row.coalition_hash_sha256,
                "poison_semantic_sha256": plain_hash,
                "plain_rounds": len(plain_rounds),
                "defense_rounds": len(defense_rounds),
                "plain_mean_source_to_target_rate": float(plain_rounds["val_source_to_target_rate"].mean()),
                "defended_mean_source_to_target_rate": float(defense_rounds["val_source_to_target_rate"].mean()),
                "mean_malicious_recall": float(defense_rounds["malicious_recall"].mean()),
                "maximum_benign_fpr": float(defense_rounds["benign_false_positive_rate"].max()),
                "exact_poison_pair": True,
                "reserved_test_arrays_materialized": False,
            })

    audit = pd.DataFrame(audit_rows).sort_values(["family_id", "coalition_size", "seed"])
    add("full_conditions", len(audit) == 72, len(audit))
    add("full_paired_branches", len(audit) * 2 == 144, len(audit) * 2)
    add("all_family_size_seed_unique", len(audit[["family_id", "coalition_size", "seed"]].drop_duplicates()) == 72, len(audit))
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(tables / "task45c3_full_matrix_audit_checks.csv", index=False)
    audit.to_csv(tables / "task45c3_full_matrix_manifest.csv", index=False)

    source_rows = []
    seen = set()
    for path in sources:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            relative = str(resolved)
        source_rows.append({"relative_path": relative, "bytes": resolved.stat().st_size, "sha256": file_sha256(resolved)})
    pd.DataFrame(source_rows).sort_values("relative_path").to_csv(tables / "task45c3_source_manifest_sha256.csv", index=False)

    counts = audit.groupby("seed", as_index=False).agg(conditions=("coalition_size", "size"), exact_pairs=("exact_poison_pair", "sum"))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(counts["seed"].astype(str), counts["conditions"], color="#4C78A8")
    ax.axhline(18, color="#C44E52", linestyle="--", linewidth=1)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Verified coalition conditions")
    ax.set_title("Task 45 complete C2 and C3 condition coverage")
    ax.set_ylim(0, 20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "task45c3_verified_condition_coverage.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "task45c3_verified_condition_coverage.pdf", bbox_inches="tight")
    plt.close(fig)

    decision = {
        "experiment_version": "4.16.C3",
        "stage": "task45_c3_complete_multiseed_integrity_audit",
        "checks_passed": int(check_frame["passed"].sum()),
        "check_count": int(len(check_frame)),
        "seeds_verified": list(SEEDS),
        "conditions_verified": 72,
        "paired_branches_verified": 144,
        "exact_poison_pairing_all": True,
        "reserved_test_arrays_materialized": False,
        "ready_for_c3_freeze_commit": True,
        "ready_for_c4_summary": True,
    }
    (output / "task45c3_integrity_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("===== TASK 45 C3 COMPLETE MATRIX AUDIT =====")
    print(f"Checks passed: {decision['checks_passed']}/{decision['check_count']}")
    print("SEEDS VERIFIED: 7,99,123,2026")
    print("CONDITIONS VERIFIED: 72")
    print("PAIRED BRANCHES VERIFIED: 144")
    print("EXACT POISON PAIRING: True")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR C4 SUMMARY: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

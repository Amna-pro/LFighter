#!/usr/bin/env python3
"""Preflight frozen Task 57 attribution inputs without evaluating recovery outcomes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PARENT_TAG = "task56-c2-xai-validation-frozen-v4182"
PARENT_COMMIT_SHORT = "d067e54"
SEEDS = (7, 99, 123, 2026)
FAMILIES = ("A_development_anchor", "B_hash_ranked", "C_hash_ranked")
STATES = ("clean_reference", "suspicious", "reconstructed")
EXPECTED_KEYS = {
    "source_logit_shap",
    "target_logit_shap",
    "source_minus_target_margin_shap",
    "standardized_probe_values",
    "probe_validation_indices",
    "true_class_id",
    "predicted_class_id",
    "logits",
    "feature_names",
    "output_names",
}
BACKGROUND_HASH = "83c8e725df051b22d155512f47afedf6338196b8b8d54529dd9df2637c9ae7be"
PROBE_HASH = "81a7a8c57cbcfd1d6c01cf304f3cc2fed811aa1dc78a262e5486ec485cccc815"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--task55-c2-root", required=True, type=Path)
    parser.add_argument("--task55-c3-root", required=True, type=Path)
    parser.add_argument("--task55-c3-audit-root", required=True, type=Path)
    parser.add_argument("--task56-summary-root", required=True, type=Path)
    parser.add_argument("--task56-audit-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def state_specs(task55_c2: Path, task55_c3: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for seed in SEEDS:
        if seed == 7:
            run_root = task55_c2
            prefix = "task55c2"
        else:
            run_root = task55_c3 / f"seed_{seed}"
            prefix = "task55c3"
        specs.append({
            "physical_state_key": f"seed_{seed}__clean_reference",
            "seed": seed,
            "family_id": "shared",
            "state": "clean_reference",
            "path": run_root / "common" / "clean_reference" / f"{prefix}_attributions.npz",
        })
        for family in FAMILIES:
            for state in ("suspicious", "reconstructed"):
                specs.append({
                    "physical_state_key": f"seed_{seed}__{family}__{state}",
                    "seed": seed,
                    "family_id": family,
                    "state": state,
                    "path": run_root / "families" / family / state / f"{prefix}_attributions.npz",
                })
    return specs


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    started = time.time()
    root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    paths = {
        "inventory": args.checkpoint_inventory.expanduser().resolve(),
        "task55_c2": args.task55_c2_root.expanduser().resolve(),
        "task55_c3": args.task55_c3_root.expanduser().resolve(),
        "task55_audit": args.task55_c3_audit_root.expanduser().resolve(),
        "task56_summary": args.task56_summary_root.expanduser().resolve(),
        "task56_audit": args.task56_audit_root.expanduser().resolve(),
        "output": args.output_dir.expanduser().resolve(),
    }
    tables = paths["output"] / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    add("config_exists", config_path.is_file(), config_path)
    for name in ("inventory", "task55_c2", "task55_c3", "task55_audit", "task56_summary", "task56_audit"):
        add(f"exists_{name}", paths[name].exists(), paths[name])
        if not paths[name].exists():
            raise FileNotFoundError(paths[name])

    parent_full = git(root, "rev-list", "-n", "1", PARENT_TAG)
    head = git(root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT_TAG, "HEAD"], cwd=root
    ).returncode == 0
    add("parent_tag_resolves_expected", parent_full.startswith(PARENT_COMMIT_SHORT), parent_full)
    add("parent_is_head_ancestor", ancestor, head)

    task55_decision_path = paths["task55_audit"] / "task55c3_audit_decision.json"
    task55_decision = json.loads(task55_decision_path.read_text(encoding="utf-8"))
    add("task55_all_47_checks_passed", task55_decision.get("checks_passed") == task55_decision.get("check_count") == 47, f"{task55_decision.get('checks_passed')}/{task55_decision.get('check_count')}")
    add("task55_28_physical_states", task55_decision.get("all_seed_unique_checkpoint_states_verified") == 28, task55_decision.get("all_seed_unique_checkpoint_states_verified"))
    add("task55_36_logical_states", task55_decision.get("all_seed_logical_state_comparisons_verified") == 36, task55_decision.get("all_seed_logical_state_comparisons_verified"))
    add("task55_exact_pairing", task55_decision.get("exact_partition_and_poison_pairing") is True, task55_decision.get("exact_partition_and_poison_pairing"))
    add("task55_same_probe", task55_decision.get("same_frozen_probe_all_states") is True, task55_decision.get("same_frozen_probe_all_states"))
    add("task55_reserved_closed", task55_decision.get("reserved_test_arrays_materialized") is False, task55_decision.get("reserved_test_arrays_materialized"))

    task56_decision_path = paths["task56_summary"] / "task56c2_validation_decision.json"
    task56_audit_path = paths["task56_audit"] / "task56c2_audit_decision.json"
    task56_decision = json.loads(task56_decision_path.read_text(encoding="utf-8"))
    task56_audit = json.loads(task56_audit_path.read_text(encoding="utf-8"))
    add("task56_scientific_pass", task56_decision.get("scientific_result") == "PASS", task56_decision.get("scientific_result"))
    add("task56_all_domains_pass", all(task56_decision.get("domain_results", {}).values()) and len(task56_decision.get("domain_results", {})) == 6, task56_decision.get("domain_results"))
    add("task56_37_checks_passed", task56_audit.get("checks_passed") == task56_audit.get("check_count") == 37, f"{task56_audit.get('checks_passed')}/{task56_audit.get('check_count')}")
    add("task57_primary_claim_allowed", task56_audit.get("task57_primary_claim_allowed") is True, task56_audit.get("task57_primary_claim_allowed"))
    add("task56_integrity_passed", task56_audit.get("integrity_audit_passed") is True, task56_audit.get("integrity_audit_passed"))
    add("task56_reserved_closed", task56_audit.get("reserved_test_arrays_materialized") is False, task56_audit.get("reserved_test_arrays_materialized"))

    repeat_path = paths["task56_summary"] / "tables" / "task56c2_repeated_run_stability.csv"
    repeat = pd.read_csv(repeat_path)
    repeat_q95 = float(repeat["normalized_l1_drift"].quantile(0.95))
    frozen_noise = float(config["measurement_noise_gate"]["frozen_value"])
    add("repeat_comparison_count_84", len(repeat) == 84, len(repeat))
    add("repeat_q95_matches_frozen", abs(repeat_q95 - frozen_noise) < 1e-15, repeat_q95)

    inventory = pd.read_csv(paths["inventory"])
    add("checkpoint_inventory_28_rows", len(inventory) == 28, len(inventory))
    add("inventory_states_exact", set(inventory["state"]) == set(STATES), sorted(set(inventory["state"])))
    add("inventory_four_seeds", sorted(inventory["seed"].unique().tolist()) == list(SEEDS), sorted(inventory["seed"].unique().tolist()))
    add("inventory_no_rejected_or_oracle", not inventory["state"].isin(["rejected", "oracle_clean"]).any(), sorted(set(inventory["state"])))

    logical_path = paths["task55_audit"] / "tables" / "task55c3_all_seed_logical_state_manifest.csv"
    logical = pd.read_csv(logical_path)
    add("logical_manifest_36_rows", len(logical) == 36, len(logical))
    add("logical_states_exact", set(logical["state"]) == set(STATES), sorted(set(logical["state"])))
    add("logical_background_hash_frozen", set(logical["background_manifest_sha256"]) == {BACKGROUND_HASH}, sorted(set(logical["background_manifest_sha256"])))
    add("logical_probe_hash_frozen", set(logical["probe_manifest_sha256"]) == {PROBE_HASH}, sorted(set(logical["probe_manifest_sha256"])))
    triplet_complete = True
    for seed in SEEDS:
        for family in FAMILIES:
            rows = logical[(logical["seed"] == seed) & (logical["family_id"] == family)]
            triplet_complete &= len(rows) == 3 and set(rows["state"]) == set(STATES)
    add("all_12_logical_triplets_complete", triplet_complete, "4 seeds x 3 families x 3 states")

    specs = state_specs(paths["task55_c2"], paths["task55_c3"])
    add("28_physical_tensor_specs", len(specs) == 28, len(specs))
    state_rows: list[dict[str, Any]] = []
    source_files: list[Path] = [
        config_path,
        paths["inventory"],
        logical_path,
        task55_decision_path,
        task56_decision_path,
        task56_audit_path,
        repeat_path,
    ]
    canonical_by_seed: dict[int, dict[str, str]] = {}
    all_exist = True
    all_keys = True
    all_shapes = True
    all_finite = True
    all_margin_identity = True
    all_ddos_count = True
    all_probe_content_same_within_seed = True
    all_feature_names_same = True
    global_feature_hash: str | None = None

    for spec in specs:
        path = Path(spec["path"])
        exists = path.is_file()
        all_exist &= exists
        if not exists:
            continue
        source_files.append(path)
        with np.load(path, allow_pickle=False) as payload:
            keys = set(payload.files)
            all_keys &= keys == EXPECTED_KEYS
            source = np.asarray(payload["source_logit_shap"])
            target = np.asarray(payload["target_logit_shap"])
            margin = np.asarray(payload["source_minus_target_margin_shap"])
            probe = np.asarray(payload["standardized_probe_values"])
            indices = np.asarray(payload["probe_validation_indices"])
            true_class = np.asarray(payload["true_class_id"])
            predicted = np.asarray(payload["predicted_class_id"])
            logits = np.asarray(payload["logits"])
            names = np.asarray(payload["feature_names"])
            outputs = np.asarray(payload["output_names"])
            expected_shapes = (
                source.shape == target.shape == margin.shape == probe.shape == (128, 69)
                and indices.shape == true_class.shape == predicted.shape == (128,)
                and logits.shape == (128, 8)
                and names.shape == (69,)
                and outputs.shape == (3,)
            )
            all_shapes &= expected_shapes
            finite = all(np.isfinite(value).all() for value in (source, target, margin, probe, logits))
            all_finite &= finite
            identity = bool(np.allclose(source - target, margin, rtol=0.0, atol=1e-6))
            all_margin_identity &= identity
            ddos_count = int(np.count_nonzero(true_class == 2))
            all_ddos_count &= ddos_count == 16
            content = {
                "probe_values_sha256": array_sha256(probe),
                "probe_indices_sha256": array_sha256(indices),
                "true_class_sha256": array_sha256(true_class),
                "feature_names_sha256": array_sha256(names),
            }
            seed = int(spec["seed"])
            if seed not in canonical_by_seed:
                canonical_by_seed[seed] = content
            else:
                all_probe_content_same_within_seed &= all(
                    canonical_by_seed[seed][key] == content[key]
                    for key in ("probe_values_sha256", "probe_indices_sha256", "true_class_sha256")
                )
            if global_feature_hash is None:
                global_feature_hash = content["feature_names_sha256"]
            else:
                all_feature_names_same &= global_feature_hash == content["feature_names_sha256"]
            state_rows.append({
                "physical_state_key": spec["physical_state_key"],
                "seed": seed,
                "family_id": spec["family_id"],
                "state": spec["state"],
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "tensor_shape": "128x69",
                "ddos_rows": ddos_count,
                "finite": finite,
                "exact_margin_identity": identity,
                **content,
            })

    add("all_28_tensor_files_exist", all_exist and len(state_rows) == 28, len(state_rows))
    add("tensor_keys_exact", all_keys, sorted(EXPECTED_KEYS))
    add("tensor_shapes_exact", all_shapes, "128x69 attributions; 128x8 logits")
    add("all_tensor_values_finite", all_finite, all_finite)
    add("exact_derived_margin_identity", all_margin_identity, all_margin_identity)
    add("16_true_ddos_rows_each", all_ddos_count, all_ddos_count)
    add("same_probe_content_within_seed", all_probe_content_same_within_seed, all_probe_content_same_within_seed)
    add("same_69_feature_names_all_states", all_feature_names_same, global_feature_hash)

    missing_named_paths = []
    for scan_root in (paths["task55_c2"], paths["task55_c3"]):
        for candidate in scan_root.rglob("*"):
            lowered = candidate.name.lower()
            if "rejected" in lowered or "oracle" in lowered:
                missing_named_paths.append(str(candidate))
    add("no_rejected_or_oracle_attribution_paths", not missing_named_paths, missing_named_paths)

    coverage_rows = [
        {"state": "clean_reference", "availability": "available", "role": "primary", "reason": "paired seed specific round 4 preattack warmup attribution"},
        {"state": "suspicious", "availability": "available", "role": "primary", "reason": "paired size 10 plain attack round 8 attribution"},
        {"state": "reconstructed", "availability": "available", "role": "primary", "reason": "paired size 10 trusted reconstruction round 8 attribution"},
        {"state": "rejected", "availability": "unavailable", "role": "roadmap_gap", "reason": "no exact paired rejected attribution artifact; substitution forbidden"},
        {"state": "oracle_clean", "availability": "unavailable", "role": "roadmap_gap", "reason": "no round 8 counterfactual clean attribution artifact; substitution forbidden"},
    ]
    triplet_rows: list[dict[str, Any]] = []
    by_key = {(int(row["seed"]), str(row["family_id"]), str(row["state"])): row for row in logical.to_dict("records")}
    physical_path = {(int(row["seed"]), str(row["family_id"]), str(row["state"])): str(row["path"]) for row in state_rows}
    clean_path_by_seed = {int(row["seed"]): str(row["path"]) for row in state_rows if row["state"] == "clean_reference"}
    for seed in SEEDS:
        for family in FAMILIES:
            clean = by_key[(seed, family, "clean_reference")]
            suspicious = by_key[(seed, family, "suspicious")]
            reconstructed = by_key[(seed, family, "reconstructed")]
            triplet_rows.append({
                "seed": seed,
                "family_id": family,
                "coalition_size": 10,
                "clean_tensor_path": clean_path_by_seed[seed],
                "suspicious_tensor_path": physical_path[(seed, family, "suspicious")],
                "reconstructed_tensor_path": physical_path[(seed, family, "reconstructed")],
                "clean_checkpoint_sha256": clean["checkpoint_sha256"],
                "suspicious_checkpoint_sha256": suspicious["checkpoint_sha256"],
                "reconstructed_checkpoint_sha256": reconstructed["checkpoint_sha256"],
                "background_manifest_sha256": clean["background_manifest_sha256"],
                "probe_manifest_sha256": clean["probe_manifest_sha256"],
                "primary_rows": "true_DDoS_rows",
                "primary_attribution": "source_minus_target_margin_shap",
            })

    write_csv(
        tables / "task57c1_state_inventory.csv",
        state_rows,
        [
            "physical_state_key", "seed", "family_id", "state", "path", "bytes", "sha256",
            "tensor_shape", "ddos_rows", "finite", "exact_margin_identity", "probe_values_sha256",
            "probe_indices_sha256", "true_class_sha256", "feature_names_sha256",
        ],
    )
    write_csv(
        tables / "task57c1_triplet_plan.csv",
        triplet_rows,
        [
            "seed", "family_id", "coalition_size", "clean_tensor_path", "suspicious_tensor_path",
            "reconstructed_tensor_path", "clean_checkpoint_sha256", "suspicious_checkpoint_sha256",
            "reconstructed_checkpoint_sha256", "background_manifest_sha256", "probe_manifest_sha256",
            "primary_rows", "primary_attribution",
        ],
    )
    write_csv(
        tables / "task57c1_state_coverage.csv",
        coverage_rows,
        ["state", "availability", "role", "reason"],
    )
    write_csv(
        tables / "task57c1_preflight_checks.csv",
        checks,
        ["check", "passed", "detail"],
    )

    unique_sources = sorted(set(source_files), key=lambda value: str(value))
    source_rows = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in unique_sources
    ]
    write_csv(
        tables / "task57c1_source_manifest_sha256.csv",
        source_rows,
        ["path", "bytes", "sha256"],
    )

    passed = sum(int(row["passed"]) for row in checks)
    result = {
        "experiment_version": "4.19.C1",
        "stage": "task57_local_attribution_recovery_preflight",
        "parent_tag": PARENT_TAG,
        "parent_commit": parent_full,
        "head_commit": head,
        "checks_passed": passed,
        "check_count": len(checks),
        "physical_attribution_states_verified": len(state_rows),
        "paired_triplets_verified": len(triplet_rows),
        "available_states": list(STATES),
        "unavailable_states": ["rejected", "oracle_clean"],
        "five_state_roadmap_complete": False,
        "three_state_primary_panel_complete": len(state_rows) == 28 and len(triplet_rows) == 12,
        "task57_outcomes_evaluated": False,
        "training_permitted": False,
        "new_shap_evaluations_permitted": False,
        "reserved_test_arrays_materialized": False,
        "measurement_noise_q95": repeat_q95,
        "elapsed_seconds": round(time.time() - started, 2),
        "ready_for_c2_recovery_analysis": passed == len(checks),
    }
    (paths["output"] / "task57c1_preflight_decision.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print("===== TASK 57 C1 ATTRIBUTION RECOVERY PREFLIGHT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("PHYSICAL ATTRIBUTION STATES VERIFIED:", len(state_rows))
    print("PAIRED TRIPLETS VERIFIED:", len(triplet_rows))
    print("AVAILABLE STATES: clean_reference, suspicious, reconstructed")
    print("UNAVAILABLE STATES: rejected, oracle_clean")
    print("FIVE STATE ROADMAP COMPLETE: False")
    print("THREE STATE PRIMARY PANEL COMPLETE:", len(state_rows) == 28 and len(triplet_rows) == 12)
    print("TASK 57 OUTCOMES EVALUATED: False")
    print("TRAINING PERMITTED: False")
    print("NEW SHAP EVALUATIONS PERMITTED: False")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR C2 RECOVERY ANALYSIS:", passed == len(checks))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

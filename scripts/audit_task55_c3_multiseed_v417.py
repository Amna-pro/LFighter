#!/usr/bin/env python3
"""Audit Task 55 C3 confirmatory SHAP evidence and all-seed coverage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


FAMILIES = ("A_development_anchor", "B_hash_ranked", "C_hash_ranked")
SEEDS = (99, 123, 2026)
REQUIRED_LONG_COLUMNS = {
    "family_id", "coalition_size", "seed", "state", "probe_global_index", "true_class_id",
    "predicted_class_id", "feature_index", "feature_name", "source_logit_shap", "target_logit_shap",
    "source_minus_target_margin_shap", "checkpoint_sha256", "background_manifest_sha256",
    "probe_manifest_sha256", "explainer_version",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--c1-decision", required=True, type=Path)
    parser.add_argument("--c1-checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--c2-decision", required=True, type=Path)
    parser.add_argument("--c2-root", required=True, type=Path)
    parser.add_argument("--c3-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_states() -> List[Tuple[int, str, str, Tuple[str, ...]]]:
    rows: List[Tuple[int, str, str, Tuple[str, ...]]] = []
    for seed in SEEDS:
        rows.append((seed, "shared", "clean_reference", (f"seed_{seed}", "common", "clean_reference")))
        for family in FAMILIES:
            rows.append((seed, family, "suspicious", (f"seed_{seed}", "families", family, "suspicious")))
            rows.append((seed, family, "reconstructed", (f"seed_{seed}", "families", family, "reconstructed")))
    return rows


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    c1_path = args.c1_decision.expanduser().resolve()
    inventory_path = args.c1_checkpoint_inventory.expanduser().resolve()
    c2_decision_path = args.c2_decision.expanduser().resolve()
    c2_root = args.c2_root.expanduser().resolve()
    c3_root = args.c3_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for path in (config_path, c1_path, inventory_path, c2_decision_path, c2_root, c3_root):
        if not path.exists():
            raise FileNotFoundError(path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    c1 = json.loads(c1_path.read_text(encoding="utf-8"))
    c2_decision = json.loads(c2_decision_path.read_text(encoding="utf-8"))
    inventory = pd.read_csv(inventory_path)
    metadata_path = c3_root / "task55c3_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    add("c1_all_checks_passed", c1.get("checks_passed") == c1.get("check_count") == 79, f"{c1.get('checks_passed')}/{c1.get('check_count')}")
    add("c1_reserved_arrays_closed", c1.get("reserved_test_arrays_materialized") is False, c1.get("reserved_test_arrays_materialized"))
    add("c2_all_checks_passed", c2_decision.get("checks_passed") == c2_decision.get("check_count") == 36, f"{c2_decision.get('checks_passed')}/{c2_decision.get('check_count')}")
    add("c2_ready_for_c3", c2_decision.get("ready_for_c3_confirmatory_multiseed") is True, c2_decision.get("ready_for_c3_confirmatory_multiseed"))
    add("c2_margin_identity_exact", c2_decision.get("exact_derived_margin_identity") is True, c2_decision.get("exact_derived_margin_identity"))
    add("c2_reserved_arrays_closed", c2_decision.get("reserved_test_arrays_materialized") is False, c2_decision.get("reserved_test_arrays_materialized"))
    add("config_training_forbidden", config.get("training_permitted") is False, config.get("training_permitted"))
    add("config_test_gate_closed", config.get("reserved_test_gate_opened") is False, config.get("reserved_test_gate_opened"))
    add("metadata_complete", metadata.get("complete") is True, metadata.get("complete"))
    add("metadata_confirmatory_seeds", metadata.get("seeds") == list(SEEDS), metadata.get("seeds"))
    add("metadata_three_families", metadata.get("families") == list(FAMILIES), metadata.get("families"))
    add("metadata_size_10", metadata.get("coalition_size") == 10, metadata.get("coalition_size"))
    add("metadata_21_unique_states", metadata.get("unique_checkpoint_states") == 21, metadata.get("unique_checkpoint_states"))
    add("metadata_27_logical_states", metadata.get("logical_state_comparisons") == 27, metadata.get("logical_state_comparisons"))
    add("metadata_128_background", metadata.get("background_rows") == 128, metadata.get("background_rows"))
    add("metadata_128_probe", metadata.get("probe_rows") == 128, metadata.get("probe_rows"))
    add("metadata_69_features", metadata.get("feature_count") == 69, metadata.get("feature_count"))
    add("metadata_shap_0p48", metadata.get("shap_version") == "0.48.0", metadata.get("shap_version"))
    add("metadata_allowed_arrays_exact", metadata.get("arrays_materialized") == ["X_train", "y_train", "X_val", "y_val"], metadata.get("arrays_materialized"))
    add("metadata_no_training", metadata.get("training_permitted") is False, metadata.get("training_permitted"))
    add("metadata_reserved_arrays_closed", metadata.get("reserved_test_arrays_materialized") is False, metadata.get("reserved_test_arrays_materialized"))
    add("metadata_scientific_result_deferred", metadata.get("scientific_result_evaluated") is False, metadata.get("scientific_result_evaluated"))
    add("background_hash_matches_c1", metadata.get("background_manifest_sha256") == c1.get("background_manifest_sha256"), metadata.get("background_manifest_sha256"))
    add("probe_hash_matches_c1", metadata.get("probe_manifest_sha256") == c1.get("probe_manifest_sha256"), metadata.get("probe_manifest_sha256"))

    source_paths: List[Path] = [config_path, c1_path, inventory_path, c2_decision_path, metadata_path]
    audit_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    common_probe: np.ndarray | None = None
    common_labels: np.ndarray | None = None
    common_features: np.ndarray | None = None
    all_finite = True
    all_margin_exact = True
    for seed, family, state, parts in expected_states():
        state_dir = c3_root.joinpath(*parts)
        marker_path = state_dir / "_task55_c3_state_complete.json"
        summary_path = state_dir / "state_summary.json"
        tensor_path = state_dir / "task55c3_attributions.npz"
        long_path = state_dir / "task55c3_attributions_long.parquet"
        feature_path = state_dir / "task55c3_feature_summary.csv"
        for path in (marker_path, summary_path, tensor_path, long_path, feature_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            source_paths.append(path)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_rows.append(summary)
        expected = inventory.loc[(inventory["seed"] == seed) & (inventory["family_id"] == family) & (inventory["state"] == state)]
        if len(expected) != 1:
            raise RuntimeError(f"C1 inventory mismatch for seed {seed}, {family}, {state}")
        expected_hash = str(expected.iloc[0]["sha256"])
        output_hashes = marker.get("output_sha256", {})
        marker_hashes = marker.get("complete") is True and set(output_hashes) == {
            "task55c3_attributions.npz", "task55c3_attributions_long.parquet", "task55c3_feature_summary.csv", "state_summary.json"
        }
        marker_hashes = marker_hashes and all((state_dir / name).is_file() and sha256(state_dir / name) == value for name, value in output_hashes.items())
        with np.load(tensor_path) as payload:
            source = payload["source_logit_shap"]
            target = payload["target_logit_shap"]
            margin = payload["source_minus_target_margin_shap"]
            probe = payload["probe_validation_indices"]
            labels = payload["true_class_id"]
            predicted = payload["predicted_class_id"]
            features = payload["feature_names"]
            values = payload["standardized_probe_values"]
            logits = payload["logits"]
        finite = all(np.isfinite(item).all() for item in (source, target, margin, values, logits))
        margin_exact = np.array_equal(margin, source - target)
        shapes = (source.shape == target.shape == margin.shape == (128, 69) and values.shape == (128, 69)
                  and logits.shape == (128, 8) and probe.shape == labels.shape == predicted.shape == (128,)
                  and features.shape == (69,))
        all_finite = all_finite and finite
        all_margin_exact = all_margin_exact and margin_exact
        if common_probe is None:
            common_probe, common_labels, common_features = probe.copy(), labels.copy(), features.copy()
        same_samples = np.array_equal(probe, common_probe) and np.array_equal(labels, common_labels) and np.array_equal(features, common_features)
        long_frame = pd.read_parquet(long_path, engine="pyarrow")
        long_schema = REQUIRED_LONG_COLUMNS.issubset(long_frame.columns)
        long_rows = len(long_frame) == 128 * 69
        feature_frame = pd.read_csv(feature_path)
        feature_valid = len(feature_frame) == 2 * 69 and set(feature_frame["scope"]) == {"all_probe_rows", "true_DDoS_rows"} and feature_frame["feature_index"].nunique() == 69
        summary_valid = (summary.get("seed") == seed and summary.get("family_id") == family and summary.get("state") == state
                         and summary.get("checkpoint_sha256") == expected_hash and summary.get("probe_rows") == 128
                         and summary.get("ddos_probe_rows") == 16 and summary.get("feature_count") == 69
                         and summary.get("attribution_shape") == [128, 69, 2]
                         and summary.get("derived_margin_identity_max_abs") == 0.0
                         and summary.get("reserved_test_arrays_materialized") is False)
        audit_rows.append({
            "seed": seed, "family_id": family, "state": state, "checkpoint_sha256": summary.get("checkpoint_sha256"),
            "expected_checkpoint_sha256": expected_hash, "checkpoint_hash_match": summary.get("checkpoint_sha256") == expected_hash,
            "marker_hashes_valid": marker_hashes, "tensor_shapes_valid": shapes, "attributions_finite": finite,
            "derived_margin_exact": margin_exact, "same_probe_labels_and_features": same_samples,
            "long_schema_valid": long_schema, "long_row_count": len(long_frame), "feature_summary_valid": feature_valid,
            "state_summary_valid": summary_valid, "partition_hash": summary.get("partition_hash"),
            "poison_index_hash": summary.get("poison_index_hash"),
            "reserved_test_arrays_materialized": summary.get("reserved_test_arrays_materialized"),
        })

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(tables / "task55c3_state_audit_manifest.csv", index=False)
    add("21_confirmatory_physical_states", len(audit) == 21, len(audit))
    add("checkpoint_hashes_match_c1", audit["checkpoint_hash_match"].all(), int(audit["checkpoint_hash_match"].sum()))
    add("all_state_markers_verified", audit["marker_hashes_valid"].all(), int(audit["marker_hashes_valid"].sum()))
    add("all_tensor_shapes_valid", audit["tensor_shapes_valid"].all(), int(audit["tensor_shapes_valid"].sum()))
    add("all_attribution_tensors_finite", all_finite and audit["attributions_finite"].all(), int(audit["attributions_finite"].sum()))
    add("exact_derived_margin_identity", all_margin_exact and audit["derived_margin_exact"].all(), int(audit["derived_margin_exact"].sum()))
    add("same_probe_labels_features_all_states", audit["same_probe_labels_and_features"].all(), int(audit["same_probe_labels_and_features"].sum()))
    add("all_long_schemas_valid", audit["long_schema_valid"].all(), int(audit["long_schema_valid"].sum()))
    add("all_long_row_counts_valid", (audit["long_row_count"] == 128 * 69).all(), int(audit["long_row_count"].sum()))
    add("all_feature_summaries_valid", audit["feature_summary_valid"].all(), int(audit["feature_summary_valid"].sum()))
    add("all_state_summaries_valid", audit["state_summary_valid"].all(), int(audit["state_summary_valid"].sum()))
    add("all_state_reserved_arrays_closed", (audit["reserved_test_arrays_materialized"] == False).all(), int((audit["reserved_test_arrays_materialized"] == False).sum()))

    exact_pairs = True
    summary_frame = pd.DataFrame(summary_rows)
    for seed in SEEDS:
        clean_partition = summary_frame.loc[(summary_frame["seed"] == seed) & (summary_frame["state"] == "clean_reference"), "partition_hash"].iloc[0]
        for family in FAMILIES:
            pair = summary_frame.loc[(summary_frame["seed"] == seed) & (summary_frame["family_id"] == family) & summary_frame["state"].isin(["suspicious", "reconstructed"])]
            exact_pairs = exact_pairs and len(pair) == 2 and pair["partition_hash"].nunique(dropna=False) == 1
            exact_pairs = exact_pairs and pair["partition_hash"].iloc[0] == clean_partition
            exact_pairs = exact_pairs and pair["poison_index_hash"].notna().all() and pair["poison_index_hash"].nunique() == 1
    add("confirmatory_exact_partition_and_poison_pairing", exact_pairs, exact_pairs)

    logical_path = c3_root / "tables/task55c3_logical_state_manifest.csv"
    state_summary_path = c3_root / "tables/task55c3_state_summary.csv"
    feature_summary_path = c3_root / "tables/task55c3_feature_summary.csv"
    progress_path = c3_root / "task55c3_progress.csv"
    c2_logical_path = c2_root / "tables/task55c2_logical_state_manifest.csv"
    c2_state_path = c2_root / "tables/task55c2_state_summary.csv"
    c2_feature_path = c2_root / "tables/task55c2_feature_summary.csv"
    for path in (logical_path, state_summary_path, feature_summary_path, progress_path, c2_logical_path, c2_state_path, c2_feature_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        source_paths.append(path)
    logical = pd.read_csv(logical_path)
    states = pd.read_csv(state_summary_path)
    features = pd.read_csv(feature_summary_path)
    progress = pd.read_csv(progress_path)
    c2_logical = pd.read_csv(c2_logical_path)
    c2_states = pd.read_csv(c2_state_path)
    c2_features = pd.read_csv(c2_feature_path)
    logical_valid = (len(logical) == 27 and set(logical["seed"]) == set(SEEDS) and set(logical["family_id"]) == set(FAMILIES)
                     and set(logical["state"]) == {"clean_reference", "suspicious", "reconstructed"}
                     and (logical.groupby(["seed", "family_id"])["state"].nunique() == 3).all())
    add("27_confirmatory_logical_rows", logical_valid, len(logical))
    add("21_confirmatory_state_summary_rows", len(states) == 21, len(states))
    add("confirmatory_feature_summary_row_count", len(features) == 21 * 2 * 69, len(features))
    add("progress_has_21_states", len(progress) == 21, len(progress))
    add("c2_seed7_tables_complete", len(c2_logical) == 9 and len(c2_states) == 7 and len(c2_features) == 7 * 2 * 69, f"{len(c2_logical)}/{len(c2_states)}/{len(c2_features)}")

    all_seed_exact_pairs = exact_pairs
    c2_clean_partition = c2_states.loc[c2_states["state"] == "clean_reference", "partition_hash"].iloc[0]
    for family in FAMILIES:
        pair = c2_states.loc[(c2_states["family_id"] == family) & c2_states["state"].isin(["suspicious", "reconstructed"])]
        all_seed_exact_pairs = all_seed_exact_pairs and len(pair) == 2 and pair["partition_hash"].nunique(dropna=False) == 1
        all_seed_exact_pairs = all_seed_exact_pairs and pair["partition_hash"].iloc[0] == c2_clean_partition
        all_seed_exact_pairs = all_seed_exact_pairs and pair["poison_index_hash"].notna().all() and pair["poison_index_hash"].nunique() == 1
    add("all_seed_exact_partition_and_poison_pairing", all_seed_exact_pairs, all_seed_exact_pairs)

    all_logical = pd.concat([c2_logical, logical], ignore_index=True)
    all_states = pd.concat([c2_states, states], ignore_index=True)
    all_features = pd.concat([c2_features, features], ignore_index=True)
    all_logical.to_csv(tables / "task55c3_all_seed_logical_state_manifest.csv", index=False)
    all_states.to_csv(tables / "task55c3_all_seed_state_summary.csv", index=False)
    all_features.to_csv(tables / "task55c3_all_seed_feature_summary.csv", index=False)
    add("all_four_seeds_present", set(all_logical["seed"]) == {7, 99, 123, 2026}, sorted(all_logical["seed"].unique()))
    add("28_all_seed_physical_states", len(all_states) == 28, len(all_states))
    add("36_all_seed_logical_states", len(all_logical) == 36, len(all_logical))
    add("all_seed_feature_summary_row_count", len(all_features) == 28 * 2 * 69, len(all_features))

    with (tables / "task55c3_audit_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "detail"])
        writer.writeheader()
        writer.writerows(checks)
    source_rows = [{"path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                    "bytes": path.stat().st_size, "sha256": sha256(path)}
                   for path in sorted(set(source_paths), key=str)]
    pd.DataFrame(source_rows).to_csv(tables / "task55c3_source_manifest_sha256.csv", index=False)
    passed = sum(int(row["passed"]) for row in checks)
    ready = passed == len(checks)
    decision = {
        "experiment_version": "4.17.C3A", "stage": "task55_confirmatory_multiseed_attribution_audit",
        "checks_passed": passed, "check_count": len(checks), "confirmatory_seeds_verified": list(SEEDS),
        "all_seeds_verified": [7, 99, 123, 2026], "families_verified": list(FAMILIES), "coalition_size": 10,
        "confirmatory_unique_checkpoint_states_verified": 21, "confirmatory_logical_state_comparisons_verified": 27,
        "all_seed_unique_checkpoint_states_verified": 28, "all_seed_logical_state_comparisons_verified": 36,
        "attribution_tensors_finite": bool(all_finite), "exact_derived_margin_identity": bool(all_margin_exact),
        "exact_partition_and_poison_pairing": bool(all_seed_exact_pairs), "same_frozen_probe_all_states": bool(audit["same_probe_labels_and_features"].all()),
        "training_permitted": False, "reserved_test_arrays_materialized": False, "scientific_result_evaluated": False,
        "task56_validation_deferred": True, "ready_for_task56_validation": ready,
    }
    (output / "task55c3_audit_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("===== TASK 55 C3 CONFIRMATORY MULTISEED AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("CONFIRMATORY SEEDS VERIFIED: 99,123,2026")
    print("CONFIRMATORY UNIQUE STATES VERIFIED: 21")
    print("CONFIRMATORY LOGICAL STATES VERIFIED: 27")
    print("ALL SEED UNIQUE STATES VERIFIED: 28")
    print("ALL SEED LOGICAL STATES VERIFIED: 36")
    print("EXACT PARTITION AND POISON PAIRING:", all_seed_exact_pairs)
    print("ATTRIBUTION TENSORS FINITE:", all_finite)
    print("EXACT DERIVED MARGIN IDENTITY:", all_margin_exact)
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR TASK 56 VALIDATION:", ready)
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

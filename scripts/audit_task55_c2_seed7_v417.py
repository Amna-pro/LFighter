#!/usr/bin/env python3
"""Audit the Task 55 C2 seed 7 SHAP pilot evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


FAMILIES = ("A_development_anchor", "B_hash_ranked", "C_hash_ranked")
STATE_SPECS = (
    ("shared", "clean_reference", ("common", "clean_reference")),
    ("A_development_anchor", "suspicious", ("families", "A_development_anchor", "suspicious")),
    ("A_development_anchor", "reconstructed", ("families", "A_development_anchor", "reconstructed")),
    ("B_hash_ranked", "suspicious", ("families", "B_hash_ranked", "suspicious")),
    ("B_hash_ranked", "reconstructed", ("families", "B_hash_ranked", "reconstructed")),
    ("C_hash_ranked", "suspicious", ("families", "C_hash_ranked", "suspicious")),
    ("C_hash_ranked", "reconstructed", ("families", "C_hash_ranked", "reconstructed")),
)
REQUIRED_LONG_COLUMNS = {
    "family_id",
    "coalition_size",
    "seed",
    "state",
    "probe_global_index",
    "true_class_id",
    "predicted_class_id",
    "feature_index",
    "feature_name",
    "source_logit_shap",
    "target_logit_shap",
    "source_minus_target_margin_shap",
    "checkpoint_sha256",
    "background_manifest_sha256",
    "probe_manifest_sha256",
    "explainer_version",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--c1-decision", required=True, type=Path)
    parser.add_argument("--c1-checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--c2-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    c1_decision_path = args.c1_decision.expanduser().resolve()
    inventory_path = args.c1_checkpoint_inventory.expanduser().resolve()
    c2_root = args.c2_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    for path in (config_path, c1_decision_path, inventory_path, c2_root):
        if not path.exists():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    c1 = json.loads(c1_decision_path.read_text(encoding="utf-8"))
    inventory = pd.read_csv(inventory_path)
    metadata_path = c2_root / "task55c2_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    add("c1_all_checks_passed", c1.get("checks_passed") == c1.get("check_count") == 79, f"{c1.get('checks_passed')}/{c1.get('check_count')}")
    add("c1_ready_for_c2", c1.get("ready_for_c2_seed7_pilot") is True, c1.get("ready_for_c2_seed7_pilot"))
    add("c1_reserved_arrays_closed", c1.get("reserved_test_arrays_materialized") is False, c1.get("reserved_test_arrays_materialized"))
    add("config_training_forbidden", config.get("training_permitted") is False, config.get("training_permitted"))
    add("config_test_gate_closed", config.get("reserved_test_gate_opened") is False, config.get("reserved_test_gate_opened"))
    add("metadata_complete", metadata.get("complete") is True, metadata.get("complete"))
    add("metadata_seed_7", metadata.get("seed") == 7, metadata.get("seed"))
    add("metadata_size_10", metadata.get("coalition_size") == 10, metadata.get("coalition_size"))
    add("metadata_three_families", metadata.get("families") == list(FAMILIES), metadata.get("families"))
    add("metadata_seven_unique_states", metadata.get("unique_checkpoint_states") == 7, metadata.get("unique_checkpoint_states"))
    add("metadata_nine_logical_states", metadata.get("logical_state_comparisons") == 9, metadata.get("logical_state_comparisons"))
    add("metadata_128_background", metadata.get("background_rows") == 128, metadata.get("background_rows"))
    add("metadata_128_probe", metadata.get("probe_rows") == 128, metadata.get("probe_rows"))
    add("metadata_69_features", metadata.get("feature_count") == 69, metadata.get("feature_count"))
    add("metadata_shap_0p48", metadata.get("shap_version") == "0.48.0", metadata.get("shap_version"))
    add("metadata_allowed_arrays_exact", metadata.get("arrays_materialized") == ["X_train", "y_train", "X_val", "y_val"], metadata.get("arrays_materialized"))
    add("metadata_no_training", metadata.get("training_permitted") is False, metadata.get("training_permitted"))
    add("metadata_reserved_arrays_closed", metadata.get("reserved_test_arrays_materialized") is False, metadata.get("reserved_test_arrays_materialized"))
    add("background_hash_matches_c1", metadata.get("background_manifest_sha256") == c1.get("background_manifest_sha256"), metadata.get("background_manifest_sha256"))
    add("probe_hash_matches_c1", metadata.get("probe_manifest_sha256") == c1.get("probe_manifest_sha256"), metadata.get("probe_manifest_sha256"))

    state_audit_rows: List[Dict[str, Any]] = []
    source_paths: List[Path] = [config_path, c1_decision_path, inventory_path, metadata_path]
    common_probe_indices: np.ndarray | None = None
    common_true_labels: np.ndarray | None = None
    common_features: np.ndarray | None = None
    all_finite = True
    all_margin_exact = True
    all_long_schemas = True
    all_long_rows = True
    all_marker_hashes = True

    for family, state, parts in STATE_SPECS:
        state_dir = c2_root.joinpath(*parts)
        marker_path = state_dir / "_task55_c2_state_complete.json"
        summary_path = state_dir / "state_summary.json"
        tensor_path = state_dir / "task55c2_attributions.npz"
        long_path = state_dir / "task55c2_attributions_long.parquet"
        feature_path = state_dir / "task55c2_feature_summary.csv"
        for path in (marker_path, summary_path, tensor_path, long_path, feature_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            source_paths.append(path)

        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = inventory.loc[
            (inventory["seed"] == 7)
            & (inventory["family_id"] == family)
            & (inventory["state"] == state)
        ]
        if len(expected) != 1:
            raise RuntimeError(f"C1 inventory mismatch for {family} {state}")
        expected_hash = str(expected.iloc[0]["sha256"])

        marker_hashes_valid = marker.get("complete") is True
        for filename, expected_output_hash in marker.get("output_sha256", {}).items():
            path = state_dir / filename
            marker_hashes_valid = marker_hashes_valid and path.is_file() and sha256(path) == expected_output_hash
        all_marker_hashes = all_marker_hashes and marker_hashes_valid

        with np.load(tensor_path) as payload:
            source = payload["source_logit_shap"]
            target = payload["target_logit_shap"]
            margin = payload["source_minus_target_margin_shap"]
            probe_indices = payload["probe_validation_indices"]
            true_labels = payload["true_class_id"]
            predicted = payload["predicted_class_id"]
            features = payload["feature_names"]
            probe_values = payload["standardized_probe_values"]
            logits = payload["logits"]
        finite = all(np.isfinite(value).all() for value in (source, target, margin, probe_values, logits))
        margin_exact = np.array_equal(margin, source - target)
        shapes_valid = (
            source.shape == target.shape == margin.shape == (128, 69)
            and probe_values.shape == (128, 69)
            and logits.shape == (128, 8)
            and probe_indices.shape == true_labels.shape == predicted.shape == (128,)
            and features.shape == (69,)
        )
        all_finite = all_finite and finite
        all_margin_exact = all_margin_exact and margin_exact
        if common_probe_indices is None:
            common_probe_indices = probe_indices.copy()
            common_true_labels = true_labels.copy()
            common_features = features.copy()
        same_samples = (
            np.array_equal(probe_indices, common_probe_indices)
            and np.array_equal(true_labels, common_true_labels)
            and np.array_equal(features, common_features)
        )

        long_frame = pd.read_parquet(long_path, engine="pyarrow")
        long_schema = REQUIRED_LONG_COLUMNS.issubset(long_frame.columns)
        long_rows = len(long_frame) == 128 * 69
        all_long_schemas = all_long_schemas and long_schema
        all_long_rows = all_long_rows and long_rows
        feature_frame = pd.read_csv(feature_path)
        feature_summary_valid = (
            len(feature_frame) == 2 * 69
            and set(feature_frame["scope"]) == {"all_probe_rows", "true_DDoS_rows"}
            and feature_frame["feature_index"].nunique() == 69
        )
        summary_valid = (
            summary.get("checkpoint_sha256") == expected_hash
            and summary.get("probe_rows") == 128
            and summary.get("ddos_probe_rows") == 16
            and summary.get("feature_count") == 69
            and summary.get("attribution_shape") == [128, 69, 2]
            and summary.get("derived_margin_identity_max_abs") == 0.0
            and summary.get("reserved_test_arrays_materialized") is False
        )
        state_audit_rows.append({
            "family_id": family,
            "state": state,
            "checkpoint_sha256": summary.get("checkpoint_sha256"),
            "expected_checkpoint_sha256": expected_hash,
            "checkpoint_hash_match": summary.get("checkpoint_sha256") == expected_hash,
            "marker_hashes_valid": marker_hashes_valid,
            "tensor_shapes_valid": shapes_valid,
            "attributions_finite": finite,
            "derived_margin_exact": margin_exact,
            "same_probe_labels_and_features": same_samples,
            "long_schema_valid": long_schema,
            "long_row_count": len(long_frame),
            "feature_summary_valid": feature_summary_valid,
            "state_summary_valid": summary_valid,
            "reserved_test_arrays_materialized": summary.get("reserved_test_arrays_materialized"),
        })

    state_audit = pd.DataFrame(state_audit_rows)
    state_audit.to_csv(tables / "task55c2_state_audit_manifest.csv", index=False)
    add("seven_physical_states", len(state_audit) == 7, len(state_audit))
    add("checkpoint_hashes_match_c1", state_audit["checkpoint_hash_match"].all(), state_audit["checkpoint_hash_match"].sum())
    add("all_state_markers_verified", all_marker_hashes and state_audit["marker_hashes_valid"].all(), state_audit["marker_hashes_valid"].sum())
    add("all_tensor_shapes_valid", state_audit["tensor_shapes_valid"].all(), state_audit["tensor_shapes_valid"].sum())
    add("all_attribution_tensors_finite", all_finite and state_audit["attributions_finite"].all(), state_audit["attributions_finite"].sum())
    add("exact_derived_margin_identity", all_margin_exact and state_audit["derived_margin_exact"].all(), state_audit["derived_margin_exact"].sum())
    add("same_probe_labels_features_all_states", state_audit["same_probe_labels_and_features"].all(), state_audit["same_probe_labels_and_features"].sum())
    add("all_long_schemas_valid", all_long_schemas and state_audit["long_schema_valid"].all(), state_audit["long_schema_valid"].sum())
    add("all_long_row_counts_valid", all_long_rows and (state_audit["long_row_count"] == 128 * 69).all(), state_audit["long_row_count"].sum())
    add("all_feature_summaries_valid", state_audit["feature_summary_valid"].all(), state_audit["feature_summary_valid"].sum())
    add("all_state_summaries_valid", state_audit["state_summary_valid"].all(), state_audit["state_summary_valid"].sum())
    add("all_state_reserved_arrays_closed", (state_audit["reserved_test_arrays_materialized"] == False).all(), (state_audit["reserved_test_arrays_materialized"] == False).sum())

    logical_path = c2_root / "tables" / "task55c2_logical_state_manifest.csv"
    state_summary_path = c2_root / "tables" / "task55c2_state_summary.csv"
    feature_summary_path = c2_root / "tables" / "task55c2_feature_summary.csv"
    progress_path = c2_root / "task55c2_progress.csv"
    for path in (logical_path, state_summary_path, feature_summary_path, progress_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        source_paths.append(path)
    logical = pd.read_csv(logical_path)
    state_summary = pd.read_csv(state_summary_path)
    feature_summary = pd.read_csv(feature_summary_path)
    progress = pd.read_csv(progress_path)
    logical_valid = (
        len(logical) == 9
        and set(logical["family_id"]) == set(FAMILIES)
        and set(logical["state"]) == {"clean_reference", "suspicious", "reconstructed"}
        and (logical.groupby("family_id")["state"].nunique() == 3).all()
    )
    add("nine_logical_state_rows", logical_valid, len(logical))
    add("seven_state_summary_rows", len(state_summary) == 7, len(state_summary))
    add("feature_summary_row_count", len(feature_summary) == 7 * 2 * 69, len(feature_summary))
    add("progress_has_seven_states", len(progress) == 7, len(progress))

    with (tables / "task55c2_audit_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    source_rows = []
    for path in sorted(set(source_paths), key=str):
        source_rows.append({
            "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    pd.DataFrame(source_rows).to_csv(tables / "task55c2_source_manifest_sha256.csv", index=False)

    passed = sum(int(row["passed"]) for row in checks)
    decision = {
        "experiment_version": "4.17.C2A",
        "stage": "task55_seed7_three_family_shap_pilot_audit",
        "checks_passed": passed,
        "check_count": len(checks),
        "seed": 7,
        "coalition_size": 10,
        "families_verified": list(FAMILIES),
        "unique_checkpoint_states_verified": 7,
        "logical_state_comparisons_verified": 9,
        "attribution_tensors_finite": bool(all_finite),
        "exact_derived_margin_identity": bool(all_margin_exact),
        "same_frozen_probe_all_states": bool(state_audit["same_probe_labels_and_features"].all()),
        "training_permitted": False,
        "reserved_test_arrays_materialized": False,
        "scientific_result_evaluated": False,
        "task56_validation_deferred": True,
        "ready_for_c3_confirmatory_multiseed": passed == len(checks),
    }
    (output / "task55c2_audit_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print("===== TASK 55 C2 SEED 7 AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("UNIQUE CHECKPOINT STATES VERIFIED: 7")
    print("LOGICAL STATE COMPARISONS VERIFIED: 9")
    print("ATTRIBUTION TENSORS FINITE:", all_finite)
    print("EXACT DERIVED MARGIN IDENTITY:", all_margin_exact)
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR C3 CONFIRMATORY MULTISEED:", passed == len(checks))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

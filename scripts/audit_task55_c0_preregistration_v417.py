#!/usr/bin/env python3
"""Audit the frozen Task 55 post detection XAI preregistration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_PARENT_TAG = "task45-c4-coalition-size-partial-v4164"
EXPECTED_PARENT_COMMIT = "9c8a7dc107191e153e1e7eff2062546c3c7de633"
ALLOWED = ["X_train", "y_train", "X_val", "y_val"]
RESERVED = [
    "X_test_natural",
    "y_test_natural",
    "X_test_diagnostic",
    "y_test_diagnostic",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    parent_commit = git(root, "rev-list", "-n", "1", EXPECTED_PARENT_TAG)
    head = git(root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_PARENT_TAG, "HEAD"],
        cwd=root,
    ).returncode == 0

    add("config_exists", config_path.is_file(), config_path)
    add("protocol_exists", protocol_path.is_file(), protocol_path)
    add("parent_tag_exact", config.get("parent_tag") == EXPECTED_PARENT_TAG, config.get("parent_tag"))
    add("parent_commit_config_exact", config.get("parent_commit") == EXPECTED_PARENT_COMMIT, config.get("parent_commit"))
    add("parent_tag_resolves_exact", parent_commit == EXPECTED_PARENT_COMMIT, parent_commit)
    add("parent_is_head_ancestor", ancestor, head)
    add("task_number", config.get("task") == 55, config.get("task"))
    add("experiment_version", config.get("experiment_version") == "4.17.0", config.get("experiment_version"))
    add("post_detection_role", config.get("scientific_role") == "post_detection_forensic_explanation_only", config.get("scientific_role"))
    add("no_detector_authority", config.get("detector_authority") is False, config.get("detector_authority"))
    add("no_training", config.get("training_permitted") is False, config.get("training_permitted"))
    add("no_method_reopening", config.get("method_reopening_permitted") is False, config.get("method_reopening_permitted"))
    add("allowed_arrays_exact", config.get("development_arrays_allowed") == ALLOWED, config.get("development_arrays_allowed"))
    add("reserved_arrays_exact", config.get("reserved_arrays_forbidden") == RESERVED, config.get("reserved_arrays_forbidden"))
    add("feature_count_69", config.get("input_representation", {}).get("expected_feature_count") == 69, config.get("input_representation", {}).get("expected_feature_count"))
    add("raw_input_features", config.get("input_representation", {}).get("explain_raw_network_traffic_features") is True, config.get("input_representation"))
    add("no_detector_feature_explanations", config.get("input_representation", {}).get("explain_detector_signature_features") is False, config.get("input_representation"))
    add("gradient_explainer", config.get("explainer", {}).get("primary") == "shap.GradientExplainer", config.get("explainer", {}).get("primary"))
    add("shap_version_pinned", config.get("explainer", {}).get("pinned_version") == "0.48.0", config.get("explainer", {}).get("pinned_version"))
    add("logit_margin_primary", config.get("explainer", {}).get("primary_attribution_target") == "source_logit_minus_target_logit", config.get("explainer", {}).get("primary_attribution_target"))
    add("source_ddos", config.get("attack_transition", {}).get("source_class_id") == 2, config.get("attack_transition"))
    add("target_benign", config.get("attack_transition", {}).get("target_class_id") == 0, config.get("attack_transition"))
    add("balanced_background_128", config.get("background", {}).get("samples_per_class") == 16 and config.get("background", {}).get("total_samples") == 128, config.get("background"))
    add("balanced_probe_128", config.get("probe", {}).get("samples_per_class") == 16 and config.get("probe", {}).get("total_samples") == 128, config.get("probe"))
    add("same_samples_all_states", config.get("background", {}).get("shared_across_all_states_and_seeds") is True and config.get("probe", {}).get("shared_across_all_states_and_seeds") is True, "background and probe")
    panel = config.get("frozen_primary_panel", {})
    add("panel_size_10", panel.get("task45_coalition_size") == 10, panel.get("task45_coalition_size"))
    add("panel_three_families", len(panel.get("families", [])) == 3, panel.get("families"))
    add("panel_four_seeds", panel.get("all_seeds") == [7, 99, 123, 2026], panel.get("all_seeds"))
    add("panel_12_conditions", panel.get("condition_count") == 12, panel.get("condition_count"))
    add("panel_36_state_evaluations", panel.get("state_evaluations") == 36, panel.get("state_evaluations"))
    add("negative_results_retained", config.get("negative_results_retained") is True, config.get("negative_results_retained"))
    add("reserved_gate_closed", config.get("reserved_test_gate_opened") is False, config.get("reserved_test_gate_opened"))

    with (tables / "task55c0_preregistration_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    sources = [config_path, protocol_path, root / "src" / "neural_models_v24.py"]
    with (tables / "task55c0_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for path in sources:
            writer.writerow({
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })

    passed = sum(int(row["passed"]) for row in checks)
    decision = {
        "experiment_version": "4.17.0",
        "stage": "task55_c0_preregistration_integrity_audit",
        "parent_tag": EXPECTED_PARENT_TAG,
        "parent_commit": EXPECTED_PARENT_COMMIT,
        "head_commit": head,
        "checks_passed": passed,
        "check_count": len(checks),
        "post_detection_only": True,
        "training_permitted": False,
        "reserved_test_arrays_materialized": False,
        "ready_for_c1_local_inventory": passed == len(checks),
    }
    (output / "task55c0_preregistration_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print("===== TASK 55 C0 PREREGISTRATION AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("POST DETECTION ONLY: True")
    print("TRAINING PERMITTED: False")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR C1 LOCAL INVENTORY:", passed == len(checks))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit the frozen Task 56 XAI validation preregistration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_PARENT_TAG = "task55-c3-confirmatory-xai-frozen-v4173"
EXPECTED_PARENT_COMMIT = "8226e0ca76c9344113e2c5d53039119ea37d3eed"
ALLOWED = ["X_train", "y_train", "X_val", "y_val"]
RESERVED = ["X_test_natural", "y_test_natural", "X_test_diagnostic", "y_test_diagnostic"]
DOMAINS = {
    "repeated_run_stability", "background_sensitivity", "probe_size_sensitivity",
    "feature_perturbation_faithfulness", "class_specificity", "cross_seed_consistency",
}


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


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if not config_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError("Task 56 config or protocol is missing")
    config: Dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    parent = git(root, "rev-list", "-n", "1", EXPECTED_PARENT_TAG)
    head = git(root, "rev-parse", "HEAD")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_PARENT_TAG, "HEAD"], cwd=root).returncode == 0
    add("config_exists", config_path.is_file(), config_path)
    add("protocol_exists", protocol_path.is_file(), protocol_path)
    add("parent_tag_exact", config.get("parent_tag") == EXPECTED_PARENT_TAG, config.get("parent_tag"))
    add("parent_commit_exact", config.get("parent_commit") == EXPECTED_PARENT_COMMIT, config.get("parent_commit"))
    add("parent_tag_resolves_exact", parent == EXPECTED_PARENT_COMMIT, parent)
    add("parent_is_head_ancestor", ancestor, head)
    add("task_number_56", config.get("task") == 56, config.get("task"))
    add("version_4p18p0", config.get("experiment_version") == "4.18.0", config.get("experiment_version"))
    add("validation_role_only", config.get("scientific_role") == "post_detection_xai_validation_only", config.get("scientific_role"))
    add("no_detector_authority", config.get("detector_authority") is False, config.get("detector_authority"))
    add("no_training", config.get("training_permitted") is False, config.get("training_permitted"))
    add("task57_not_permitted", config.get("task57_recovery_analysis_permitted") is False, config.get("task57_recovery_analysis_permitted"))
    add("no_method_reopening", config.get("method_reopening_permitted") is False, config.get("method_reopening_permitted"))
    add("allowed_arrays_exact", config.get("development_arrays_allowed") == ALLOWED, config.get("development_arrays_allowed"))
    add("reserved_arrays_exact", config.get("reserved_arrays_forbidden") == RESERVED, config.get("reserved_arrays_forbidden"))
    panel = config.get("frozen_panel", {})
    add("four_seeds", panel.get("seeds") == [7, 99, 123, 2026], panel.get("seeds"))
    add("three_families", panel.get("families") == ["A_development_anchor", "B_hash_ranked", "C_hash_ranked"], panel.get("families"))
    add("size_10", panel.get("coalition_size") == 10, panel.get("coalition_size"))
    add("28_physical_states", panel.get("unique_checkpoint_states") == 28, panel.get("unique_checkpoint_states"))
    add("36_logical_states", panel.get("logical_state_comparisons") == 36, panel.get("logical_state_comparisons"))
    add("69_features", panel.get("feature_count") == 69, panel.get("feature_count"))
    add("primary_true_ddos", panel.get("primary_rows") == "true_DDoS_rows", panel.get("primary_rows"))
    add("primary_margin_attribution", panel.get("primary_attribution") == "source_minus_target_margin_shap", panel.get("primary_attribution"))
    baseline = config.get("frozen_task55_baseline", {})
    add("task55_shap_frozen", baseline.get("shap_version") == "0.48.0" and baseline.get("rseed") == 5517, baseline)
    add("task55_samples_frozen", baseline.get("background_rows") == 128 and baseline.get("probe_rows") == 128 and baseline.get("ddos_probe_rows") == 16, baseline)
    repeat = config.get("repeated_run_stability", {})
    add("three_repeat_seeds_frozen", repeat.get("repeat_rseeds") == [56101, 56102, 56103], repeat.get("repeat_rseeds"))
    add("84_repeat_evaluations", repeat.get("state_evaluations") == 84, repeat.get("state_evaluations"))
    add("repeat_thresholds_complete", set(repeat.get("pass_rules", {})) == {"median_spearman_at_least", "fifth_percentile_spearman_at_least", "median_top10_jaccard_at_least", "median_normalized_l1_drift_at_most"}, repeat.get("pass_rules"))
    background = config.get("background_sensitivity", {})
    add("alternate_background_seeds_frozen", background.get("alternate_selection_seeds") == [5601, 5602], background.get("alternate_selection_seeds"))
    add("background_disjoint_rule", "disjoint" in background.get("selection", ""), background.get("selection"))
    add("56_background_evaluations", background.get("state_evaluations") == 56, background.get("state_evaluations"))
    probe = config.get("probe_size_sensitivity", {})
    add("probe_sizes_frozen", probe.get("nested_true_ddos_probe_sizes") == [8, 12, 16], probe.get("nested_true_ddos_probe_sizes"))
    faith = config.get("feature_perturbation_faithfulness", {})
    add("faithfulness_replacement_frozen", faith.get("replacement") == "frozen_baseline_background_feature_mean", faith.get("replacement"))
    add("faithfulness_topk_frozen", faith.get("cumulative_top_k") == [1, 5, 10, 20], faith.get("cumulative_top_k"))
    add("faithfulness_random_controls", faith.get("random_control_seed") == 5604 and faith.get("random_panels_per_k") == 256, faith)
    specificity = config.get("class_specificity", {})
    add("class_specificity_frozen", specificity.get("within_reference") == "leave_one_out_true_DDoS_centroid" and specificity.get("between_reference") == "pooled_non_DDoS_centroid", specificity)
    cross = config.get("cross_seed_consistency", {})
    add("42_cross_seed_comparisons", cross.get("pairwise_seed_comparisons") == 42, cross.get("pairwise_seed_comparisons"))
    runtime = config.get("runtime_and_memory", {})
    add("runtime_and_memory_measured", runtime.get("measure_wall_seconds") is True and runtime.get("measure_process_peak_rss_bytes") is True, runtime)
    add("runtime_not_scientific_gate", runtime.get("affects_scientific_pass_fail") is False, runtime.get("affects_scientific_pass_fail"))
    policy = config.get("decision_policy", {})
    add("all_domains_in_policy", set(policy.get("core_domains", []) + policy.get("robustness_domains", [])) == DOMAINS, policy)
    add("pass_partial_fail_frozen", all(key in policy for key in ("PASS", "PARTIAL", "FAIL")), policy)
    add("task57_claim_gate_frozen", all(key in policy for key in ("task57_primary_claim_allowed_only_if", "task57_exploratory_only_if", "task57_recovery_claim_blocked_if")), policy)
    add("140_new_shap_evaluations", config.get("planned_new_shap_state_evaluations") == 140, config.get("planned_new_shap_state_evaluations"))
    add("negative_results_retained", config.get("negative_results_retained") is True, config.get("negative_results_retained"))
    add("reserved_gate_closed", config.get("reserved_test_gate_opened") is False, config.get("reserved_test_gate_opened"))

    with (tables / "task56c0_preregistration_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "detail"])
        writer.writeheader(); writer.writerows(checks)
    sources = [config_path, protocol_path]
    with (tables / "task56c0_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for path in sources:
            writer.writerow({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    passed = sum(int(row["passed"]) for row in checks)
    decision = {
        "experiment_version": "4.18.C0", "stage": "task56_preregistration_integrity_audit",
        "parent_tag": EXPECTED_PARENT_TAG, "parent_commit": EXPECTED_PARENT_COMMIT, "head_commit": head,
        "checks_passed": passed, "check_count": len(checks), "outcomes_inspected": False,
        "training_permitted": False, "reserved_test_arrays_materialized": False,
        "ready_for_c1_preflight": passed == len(checks),
    }
    (output / "task56c0_preregistration_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("===== TASK 56 C0 PREREGISTRATION AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("OUTCOMES INSPECTED: False")
    print("TRAINING PERMITTED: False")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR C1 PREFLIGHT:", passed == len(checks))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit the frozen Task 57 attribution recovery preregistration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PARENT_TAG = "task56-c2-xai-validation-frozen-v4182"
PARENT_COMMIT_SHORT = "d067e54"
SEEDS = [7, 99, 123, 2026]
FAMILIES = ["A_development_anchor", "B_hash_ranked", "C_hash_ranked"]
AVAILABLE_STATES = ["clean_reference", "suspicious", "reconstructed"]
UNAVAILABLE_STATES = ["rejected", "oracle_clean"]
RESERVED = ["X_test_natural", "y_test_natural", "X_test_diagnostic", "y_test_diagnostic"]
VALIDATION_DOMAINS = {
    "repeated_run_stability",
    "background_sensitivity",
    "probe_size_sensitivity",
    "feature_perturbation_faithfulness",
    "class_specificity",
    "cross_seed_consistency",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if not config_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError("Task 57 config or protocol is missing")

    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    protocol_text = protocol_path.read_text(encoding="utf-8")
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    parent_full = git(root, "rev-list", "-n", "1", PARENT_TAG)
    head = git(root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT_TAG, "HEAD"], cwd=root
    ).returncode == 0

    add("config_exists", config_path.is_file(), config_path)
    add("protocol_exists", protocol_path.is_file(), protocol_path)
    add("parent_tag_exact", config.get("parent_tag") == PARENT_TAG, config.get("parent_tag"))
    add(
        "parent_commit_short_exact",
        config.get("parent_commit_short") == PARENT_COMMIT_SHORT,
        config.get("parent_commit_short"),
    )
    add("parent_tag_resolves_expected", parent_full.startswith(PARENT_COMMIT_SHORT), parent_full)
    add("parent_is_head_ancestor", ancestor, head)
    add("task_number_57", config.get("task") == 57, config.get("task"))
    add("version_4p19p0", config.get("experiment_version") == "4.19.0", config.get("experiment_version"))
    add(
        "post_detection_role",
        config.get("scientific_role") == "post_detection_attribution_recovery_analysis",
        config.get("scientific_role"),
    )
    add("no_detector_authority", config.get("detector_authority") is False, config.get("detector_authority"))
    add("no_training", config.get("training_permitted") is False, config.get("training_permitted"))
    add("no_new_shap", config.get("new_shap_evaluations_permitted") is False, config.get("new_shap_evaluations_permitted"))
    add("no_method_reopening", config.get("method_reopening_permitted") is False, config.get("method_reopening_permitted"))
    add("no_development_arrays_needed", config.get("development_arrays_allowed") == [], config.get("development_arrays_allowed"))
    add("reserved_arrays_exact", config.get("reserved_arrays_forbidden") == RESERVED, config.get("reserved_arrays_forbidden"))

    gate = config.get("upstream_gate", {})
    add("task56_pass_required", gate.get("task56_required_result") == "PASS", gate.get("task56_required_result"))
    add("task56_37_checks_required", gate.get("task56_required_checks") == 37, gate.get("task56_required_checks"))
    add("task57_claim_authorization_required", gate.get("task57_primary_claim_required") is True, gate.get("task57_primary_claim_required"))
    add("six_validation_domains_required", set(gate.get("required_validation_domains", [])) == VALIDATION_DOMAINS, gate.get("required_validation_domains"))

    panel = config.get("frozen_panel", {})
    add("four_seeds_exact", panel.get("seeds") == SEEDS, panel.get("seeds"))
    add("three_families_exact", panel.get("families") == FAMILIES, panel.get("families"))
    add("size_10_exact", panel.get("coalition_size") == 10, panel.get("coalition_size"))
    add("three_available_states", panel.get("available_states") == AVAILABLE_STATES, panel.get("available_states"))
    add("two_unavailable_states", panel.get("planned_but_unavailable_states") == UNAVAILABLE_STATES, panel.get("planned_but_unavailable_states"))
    add("28_physical_states", panel.get("physical_checkpoint_states") == 28, panel.get("physical_checkpoint_states"))
    add("36_logical_rows", panel.get("logical_state_rows") == 36, panel.get("logical_state_rows"))
    add("12_triplets", panel.get("paired_triplets") == 12, panel.get("paired_triplets"))
    add("69_features", panel.get("feature_count") == 69, panel.get("feature_count"))
    add("128_probe_rows", panel.get("probe_rows") == 128, panel.get("probe_rows"))
    add("16_primary_rows", panel.get("primary_row_count") == 16, panel.get("primary_row_count"))
    add("ddos_to_benign_pair", panel.get("source_class_id") == 2 and panel.get("target_class_id") == 0, (panel.get("source_class_id"), panel.get("target_class_id")))
    add("primary_margin_shap", panel.get("primary_attribution") == "source_minus_target_margin_shap", panel.get("primary_attribution"))

    scope = config.get("state_scope_policy", {})
    add("no_state_substitution", scope.get("substitution_permitted") is False, scope.get("substitution_permitted"))
    add("rejected_marked_unavailable", str(scope.get("rejected_status", "")).startswith("unavailable"), scope.get("rejected_status"))
    add("oracle_marked_unavailable", str(scope.get("oracle_clean_status", "")).startswith("unavailable"), scope.get("oracle_clean_status"))
    add("claim_limited_to_preattack_reference", "preattack_clean_reference" in str(scope.get("claim_wording", "")), scope.get("claim_wording"))
    add("oracle_recovery_claim_forbidden", "oracle_clean" in str(scope.get("forbidden_claim_wording", "")), scope.get("forbidden_claim_wording"))

    profiles = config.get("profiles", {})
    add("validated_primary_profile", profiles.get("primary") == "mean_absolute_margin_shap_true_DDoS_by_feature", profiles.get("primary"))
    add("signed_secondary_profile", profiles.get("secondary_signed") == "mean_signed_margin_shap_true_DDoS_by_feature", profiles.get("secondary_signed"))
    add("top10_frozen", profiles.get("ranking_k") == 10, profiles.get("ranking_k"))

    noise = config.get("measurement_noise_gate", {})
    add("noise_q95_frozen", abs(float(noise.get("frozen_value", -1)) - 0.12184864742664225) < 1e-15, noise.get("frozen_value"))
    add("noise_84_comparisons", noise.get("comparison_count") == 84, noise.get("comparison_count"))
    add("attack_signal_8_of_12", noise.get("attack_signal_triplets_required") == 8, noise.get("attack_signal_triplets_required"))

    recovery = config.get("primary_recovery_gate", {})
    add("positive_9_of_12", recovery.get("minimum_positive_triplets") == 9, recovery.get("minimum_positive_triplets"))
    add("all_four_seed_means_positive", recovery.get("minimum_positive_seed_means") == 4, recovery.get("minimum_positive_seed_means"))
    add("minimum_recovery_fraction_0p25", recovery.get("minimum_median_recovery_fraction") == 0.25, recovery.get("minimum_median_recovery_fraction"))
    add("triplet_sign_flip_alpha", recovery.get("triplet_sign_flip_p_at_most") == 0.05, recovery.get("triplet_sign_flip_p_at_most"))
    add("cluster_test_report_only", recovery.get("seed_blocked_sign_flip_report_only") is True, recovery.get("seed_blocked_sign_flip_report_only"))

    features = config.get("feature_association_policy", {})
    add("feature_role_descriptive", str(features.get("role", "")).startswith("descriptive"), features.get("role"))
    add("feature_consistency_9", features.get("minimum_same_direction_triplets") == 9, features.get("minimum_same_direction_triplets"))
    add("all_69_features_retained", features.get("all_69_features_retained_in_table") is True, features.get("all_69_features_retained_in_table"))
    add("no_posthoc_feature_selection", features.get("posthoc_feature_selection_permitted") is False, features.get("posthoc_feature_selection_permitted"))

    uncertainty = config.get("uncertainty_and_inference", {})
    add("hierarchical_bootstrap_20000", uncertainty.get("hierarchical_bootstrap_replicates") == 20000, uncertainty.get("hierarchical_bootstrap_replicates"))
    add("hierarchical_bootstrap_seed_5719", uncertainty.get("hierarchical_bootstrap_seed") == 5719, uncertainty.get("hierarchical_bootstrap_seed"))
    add("4096_triplet_patterns", uncertainty.get("triplet_exact_sign_flip_patterns") == 4096, uncertainty.get("triplet_exact_sign_flip_patterns"))
    add("16_seed_block_patterns", uncertainty.get("seed_blocked_sign_flip_patterns") == 16, uncertainty.get("seed_blocked_sign_flip_patterns"))
    add("seed_test_resolution_disclosed", uncertainty.get("seed_blocked_minimum_one_sided_p") == 0.0625 and uncertainty.get("seed_blocked_test_is_resolution_limited") is True, uncertainty)
    add("no_posthoc_threshold_changes", uncertainty.get("no_posthoc_threshold_changes") is True, uncertainty.get("no_posthoc_threshold_changes"))

    decision = config.get("decision_policy", {})
    add("four_outcomes_frozen", all(key in decision for key in ("INCONCLUSIVE", "PASS", "PARTIAL", "FAIL")), sorted(decision))
    add("negative_results_retained", decision.get("negative_and_boundary_results_retained") is True, decision.get("negative_and_boundary_results_retained"))
    add("task58_figures_deferred", config.get("outputs_planned", {}).get("publication_figures_deferred_to_task58") is True, config.get("outputs_planned"))
    add("reserved_gate_closed", config.get("reserved_test_gate_opened") is False, config.get("reserved_test_gate_opened"))

    add("protocol_names_missing_states", "rejected" in protocol_text and "oracle clean" in protocol_text.lower(), "rejected and oracle clean")
    add("protocol_forbids_substitution", "does not rename, infer, or synthesize" in protocol_text, "explicit prohibition")
    add("protocol_discloses_seed_dependence", "share the same clean checkpoint" in protocol_text, "seed dependence")
    add("protocol_keeps_task58_separate", "Task 58" in protocol_text, "Task 58")

    with (tables / "task57c0_preregistration_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    with (tables / "task57c0_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for path in (config_path, protocol_path):
            writer.writerow({
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })

    passed = sum(int(row["passed"]) for row in checks)
    result = {
        "experiment_version": "4.19.C0",
        "stage": "task57_preregistration_integrity_audit",
        "parent_tag": PARENT_TAG,
        "parent_commit": parent_full,
        "head_commit": head,
        "checks_passed": passed,
        "check_count": len(checks),
        "task57_outcomes_inspected": False,
        "training_permitted": False,
        "new_shap_evaluations_permitted": False,
        "reserved_test_arrays_materialized": False,
        "ready_for_c1_preflight": passed == len(checks),
    }
    (output / "task57c0_preregistration_decision.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print("===== TASK 57 C0 PREREGISTRATION AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("TASK 57 OUTCOMES INSPECTED: False")
    print("TRAINING PERMITTED: False")
    print("NEW SHAP EVALUATIONS PERMITTED: False")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR C1 PREFLIGHT:", passed == len(checks))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

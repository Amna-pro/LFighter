import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "task63_final_evaluation_manifest_v4262.json"
TASK64_DECISION = ROOT / "results" / "cic_iot_diad_task64_c1_seed_reservation_v427" / "task64_seed_reservation_decision.json"
TASK64_AUDIT = ROOT / "results" / "cic_iot_diad_task64_c1_seed_reservation_v427" / "task64_audit_decision.json"
OUTDIR = ROOT / "results" / "cic_iot_diad_task63_c2_manifest_v4262"
OUTDIR.mkdir(parents=True, exist_ok=True)

m = json.loads(MANIFEST.read_text(encoding="utf-8"))
d = json.loads(TASK64_DECISION.read_text(encoding="utf-8"))
a = json.loads(TASK64_AUDIT.read_text(encoding="utf-8"))

superseded = [1371338874, 959960781, 968407367, 1778991631, 1216611438]
attacks = ["all_to_one_benign", "cyclic_shift", "multiclass_partial_cycle", "pairwise_swap", "random_flip"]
coalition = [1, 7, 8, 10, 14, 15, 17, 18]

checks = {}
checks["task_exact"] = m.get("task") == 63
checks["protocol_exact"] = m.get("protocol_id") == "task63_c2_v4262"
checks["deviation_explicit"] = m["protocol_deviation"]["deviation_present"] is True
checks["no_outcomes_seen"] = m["protocol_deviation"]["scientific_outcomes_seen_before_correction"] is False
checks["task64_c1_seed_list_exact"] = d.get("final_seeds") == superseded
checks["task64_c1_prior_hits_zero"] = d.get("selected_seed_prior_usage_hits") == 0
checks["task64_c1_test_paths_zero"] = d.get("reserved_test_paths_opened") == 0
checks["task64_c1_binary_arrays_zero"] = d.get("binary_arrays_loaded") == 0
checks["task64_c1_training_blocked"] = d.get("training_permitted") is False
checks["task64_c1_audit_passed"] = a.get("all_checks_passed") is True and a.get("checks_passed") == a.get("checks_total") == 27
checks["task64_c1_audit_test_arrays_false"] = a.get("reserved_test_arrays_materialized") is False
checks["superseded_exact"] = m["superseded_task64_c1_seeds"]["seeds"] == superseded
checks["superseded_blocked"] = m["superseded_task64_c1_seeds"]["may_be_used_in_task65"] is False
checks["attacks_exact"] = m["final_attack_panel"]["attacks"] == attacks
checks["poison_fraction_exact"] = m["final_attack_panel"]["poison_fraction"] == 1.0
checks["coalition_exact"] = m["final_attack_panel"]["malicious_clients"] == coalition
checks["task46_54_blocked"] = m["final_attack_panel"]["task46_54_conditions_permitted"] is False
checks["primary_round_exact"] = m["final_method"]["primary_endpoint_round"] == 8
checks["datasets_exact"] = m["reserved_test_evaluation"]["datasets"] == ["diagnostic", "natural"]
checks["best_round_blocked"] = m["reserved_test_evaluation"]["best_round_selection_permitted"] is False
checks["bootstrap_exact"] = m["statistics"]["bootstrap"] == {
    "type": "paired_percentile_bootstrap_on_seed_level_differences",
    "replicates": 20000,
    "seed": 650428,
    "confidence_level": 0.95,
}
checks["signflip_exact"] = (
    m["statistics"]["hypothesis_test"]["type"] == "exact_paired_sign_flip_two_sided"
    and m["statistics"]["hypothesis_test"]["final_seed_count"] == 5
    and m["statistics"]["hypothesis_test"]["sign_patterns"] == 32
    and abs(m["statistics"]["hypothesis_test"]["minimum_attainable_two_sided_p"] - 0.0625) < 1e-15
)
checks["holm_exact"] = m["statistics"]["multiple_comparisons"].startswith("Holm family-wise adjustment across the five attacks")
checks["thresholds_exact"] = m["practical_significance_thresholds"] == {
    "clean_mean_macro_f1_loss_max": 0.01,
    "clean_any_seed_macro_f1_loss_max": 0.02,
    "benign_fpr_max": 0.05,
    "mean_malicious_client_recall_min": 0.50,
    "mean_relative_attack_excess_reduction_min": 0.25,
    "positive_mitigation_seed_count_min": 4,
    "positive_mitigation_seed_count_denominator": 5,
}
checks["replacement_seed_count_exact"] = m["task64_c2_requirement"]["replacement_seed_count"] == 5
checks["old_seeds_excluded_from_c2"] = m["task64_c2_requirement"]["exclude_superseded_task64_c1_seeds"] == superseded
checks["task65_blocked_for_c2"] = m["task64_c2_requirement"]["task65_blocked_until_task64_c2_frozen"] is True
checks["amendment_training_blocked"] = m["data_boundary"]["training_permitted"] is False
checks["amendment_test_access_blocked"] = m["data_boundary"]["reserved_test_array_access_permitted"] is False
checks["amendment_shap_blocked"] = m["data_boundary"]["new_shap_evaluations_permitted"] is False
checks["amendment_llm_blocked"] = m["data_boundary"]["llm_calls_permitted"] is False
checks["md_exists"] = (ROOT / "TASK63_C2_PROTOCOL_COMPLETION.md").exists()
checks["deviation_md_exists"] = (ROOT / "configs" / "TASK63_C2_DEVIATION_RECORD_V4262.md").exists()

# Verify the current repository contains the frozen Task64 C1 tag and it is an ancestor of HEAD.
def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

tag = "task64-c1-untouched-seeds-frozen-v4270"
r = git("rev-parse", "--verify", f"refs/tags/{tag}")
checks["task64_tag_resolves"] = r.returncode == 0
if checks["task64_tag_resolves"]:
    r2 = git("merge-base", "--is-ancestor", tag, "HEAD")
    checks["task64_tag_is_ancestor"] = r2.returncode == 0
else:
    checks["task64_tag_is_ancestor"] = False

for k, v in checks.items():
    print(f"{k}: {v}")

passed = sum(bool(v) for v in checks.values())
total = len(checks)
print(f"PASS: {passed}/{total}")
ready = all(checks.values())

out = {
    "task": 63,
    "protocol_id": "task63_c2_v4262",
    "experiment_version": "4.26.2.audit",
    "checks_passed": passed,
    "checks_total": total,
    "all_checks_passed": ready,
    "protocol_ordering_deviation_recorded": True,
    "scientific_outcomes_seen_before_correction": False,
    "superseded_task64_c1_seeds": superseded,
    "superseded_seeds_permitted_in_task65": False,
    "reserved_test_arrays_materialized": False,
    "training_permitted": False,
    "ready_to_freeze_task63_c2": ready,
    "task65_blocked_until_task64_c2_frozen": True,
    "checks": checks,
}
(OUTDIR / "task63c2_audit_decision.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if not ready:
    sys.exit(1)
print("READY TO FREEZE TASK 63 C2: True")
print("NEXT REQUIRED STAGE: Task 64 C2 replacement untouched-seed reservation")

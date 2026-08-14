from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "task61_preregistration_v4230.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    c = load_json(CONFIG_PATH)
    out = ROOT / c["outputs"]["c0_root"]
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, value: bool, detail: str = "") -> None:
        checks.append((name, bool(value), detail))

    check("task_is_61", c["task"] == 61)
    check("phase_is_c0_c1", c["phase"] == "C0_C1")
    check("parent_tag_exact", c["parent_tag"] == "task60-c1-forensic-reporting-frozen-v4221")
    resolved = ""
    try:
        resolved = subprocess.check_output(["git", "rev-list", "-n", "1", c["parent_tag"]], cwd=ROOT, text=True).strip()
    except Exception:
        pass
    check("parent_tag_resolves", len(resolved) == 40, resolved)
    check("parent_is_ancestor", subprocess.run(["git", "merge-base", "--is-ancestor", c["parent_tag"], "HEAD"], cwd=ROOT).returncode == 0)
    for name, path in c["inputs"].items():
        check(f"input_exists_{name}", (ROOT / path).is_file(), path)
    if (ROOT / c["inputs"]["task60_audit"]).is_file():
        task60 = load_json(ROOT / c["inputs"]["task60_audit"])
        check("task60_audit_all_passed", task60["all_checks_passed"] is True)
        check("task60_ready_for_task61", task60["ready_for_task61_evaluation"] is True)
        check("task60_one_api_call", task60["api_calls"] == 1)
    cohort = c["cohort"]
    check("four_seeds_frozen", cohort["seeds"] == [7, 99, 123, 2026])
    check("three_families_frozen", len(cohort["families"]) == 3)
    check("coalition_size_10", cohort["coalition_size"] == 10)
    check("twelve_cases", cohort["expected_cases"] == 12)
    check("three_repetitions", cohort["llm_repetitions_per_case"] == 3)
    check("thirty_six_reports", cohort["expected_llm_reports"] == 36)
    check("case_exclusion_prohibited", cohort["case_exclusion_permitted"] is False)
    check("best_output_selection_prohibited", cohort["best_output_selection_permitted"] is False)
    model = c["model"]
    check("terra_model", model["model_id"] == "gpt-5.6-terra")
    check("responses_api", model["api"] == "Responses API")
    check("reasoning_low", model["reasoning_effort"] == "low")
    check("output_cap_700", model["max_output_tokens"] == 700)
    check("no_retries", model["max_retries"] == 0)
    check("tools_disabled", model["tools_enabled"] is False)
    check("history_disabled", model["conversation_history_enabled"] is False)
    check("hard_call_cap_36", model["maximum_api_calls_task61_c2"] == 36)
    cost = c["cost_control"]
    check("cost_ceiling_one_usd", cost["hard_estimated_cost_ceiling_usd"] == 1.0)
    check("pricing_source_official", cost["pricing_source"].startswith("https://developers.openai.com/"))
    metrics = c["automated_metrics"]
    check("schema_rate_one", metrics["schema_conformance_rate_required"] == 1.0)
    check("citation_rate_one", metrics["citation_validity_rate_required"] == 1.0)
    check("numeric_rate_one", metrics["numeric_grounding_rate_required"] == 1.0)
    check("authority_violations_zero", metrics["authority_violation_count_required"] == 0)
    check("intent_claims_zero", metrics["unsupported_intent_claim_count_required"] == 0)
    check("oracle_claims_zero", metrics["unsupported_oracle_claim_count_required"] == 0)
    check("fact_coverage_frozen", metrics["required_fact_coverage_rate_minimum"] == 0.95)
    check("repeat_consistency_frozen", metrics["repeat_numeric_consistency_rate_minimum"] == 0.99)
    check("negative_outputs_retained", metrics["all_negative_outputs_retained"] is True)
    human = c["human_evaluation"]
    check("two_reviewers_minimum", human["minimum_independent_reviewers"] >= 2)
    check("review_blinded", human["blinded"] is True)
    check("randomization_seed_frozen", human["randomization_seed"] == 610423)
    check("five_human_dimensions", len(human["dimensions"]) == 5)
    check("analyst_accuracy_recorded", human["analyst_task_accuracy_recorded"] is True)
    check("completion_time_recorded", human["analyst_completion_time_recorded"] is True)
    check("superiority_claim_blocked", human["superiority_claim_blocked_until_review_complete"] is True)
    bounds = c["data_boundaries"]
    check("training_prohibited", bounds["training_permitted"] is False)
    check("new_shap_prohibited", bounds["new_shap_evaluations_permitted"] is False)
    check("reserved_arrays_closed", bounds["reserved_test_arrays_materialized"] is False)
    check("deterministic_authority", bounds["security_decision_authority"] == "deterministic_pipeline")
    check("llm_report_only", bounds["llm_role"] == "report_only")

    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "task61c0_preregistration_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check_id", "passed", "detail"])
        writer.writerows(checks)
    source_paths = [
        CONFIG_PATH,
        ROOT / "configs" / "TASK61_PREREGISTRATION_V423.md",
        Path(__file__),
        ROOT / "scripts" / "build_task61_c1_evidence_matrix_v423.py",
        ROOT / "scripts" / "audit_task61_c1_evidence_matrix_v423.py",
    ]
    with (tables / "task61c0_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256"])
        for path in source_paths:
            writer.writerow([path.relative_to(ROOT).as_posix(), sha256(path)])
    passed = sum(value for _, value, _ in checks)
    decision = {
        "checks_passed": passed,
        "checks_total": len(checks),
        "all_checks_passed": passed == len(checks),
        "parent_commit": resolved,
        "task61_outcomes_inspected": False,
        "llm_calls": 0,
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
        "ready_for_c1_evidence_matrix": passed == len(checks),
    }
    write_json(out / "task61c0_preregistration_decision.json", decision)
    print("===== TASK 61 C0 EVALUATION PREREGISTRATION AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("TASK 61 OUTCOMES INSPECTED: False")
    print("PLANNED CASES: 12")
    print("PLANNED LLM REPORTS: 36")
    print("LLM CALLS: 0")
    print("TRAINING PERMITTED: False")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR C1 EVIDENCE MATRIX:", decision["ready_for_c1_evidence_matrix"])
    return 0 if decision["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

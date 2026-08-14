from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "task62_preregistration_v4250.json"
PROTOCOL_MD = ROOT / "configs" / "TASK62_PREREGISTRATION_V425.md"
SYSTEM_PROMPT = ROOT / "prompts" / "task62_hybrid_narrative_system_v425.txt"
USER_PROMPT = ROOT / "prompts" / "task62_hybrid_narrative_user_v425.txt"
NARRATIVE_SCHEMA = ROOT / "schemas" / "lfighter_hybrid_narrative_v1.schema.json"
REPORT_SCHEMA = ROOT / "schemas" / "lfighter_forensic_report_v1.schema.json"
C3_DECISION = ROOT / "results" / "cic_iot_diad_task61_c3_failure_diagnostic_v424" / "task61c3_failure_diagnostic_decision.json"
C3_AUDIT = ROOT / "results" / "cic_iot_diad_task61_c3_audit_v424" / "task61c3_audit_decision.json"
OUT_ROOT = ROOT / "results" / "cic_iot_diad_task62_c0_preregistration_v425"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check_id": name, "passed": bool(passed), "detail": detail})


def main() -> int:
    config = load_json(CONFIG_PATH)
    c3 = load_json(C3_DECISION)
    c3_audit = load_json(C3_AUDIT)
    narrative_schema = load_json(NARRATIVE_SCHEMA)
    system_text = SYSTEM_PROMPT.read_text(encoding="utf-8")
    user_text = USER_PROMPT.read_text(encoding="utf-8")
    protocol_text = PROTOCOL_MD.read_text(encoding="utf-8")
    checks: list[dict[str, Any]] = []

    add(checks, "protocol_id", config["protocol_id"] == "task62_v4250", "Frozen protocol identifier.")
    add(checks, "task_number", config["task"] == 62, "Task number is 62.")
    add(checks, "parent_tag", config["parent_tag"] == "task61-c3-llm-failure-diagnostic-frozen-v4240", "Parent is the frozen C3 diagnostic.")
    add(checks, "prospective_task62", config["task62_outcomes_inspected"] is False, "Task 62 outcomes have not been inspected.")
    add(checks, "parent_failure_retained", c3["automated_result_after_diagnostic"] == "FAIL" and c3["automated_result_changed"] is False, "Task 61 failure remains frozen.")
    add(checks, "parent_audit_passed", c3_audit["all_checks_passed"] is True, "Task 61 C3 audit passed.")
    add(checks, "parent_diagnosis_supports_hybrid", c3["task62_hybrid_evaluation_justified"] is True, "C3 motivates the hybrid architecture.")
    add(checks, "expected_cases", config["cohort"]["expected_cases"] == 12, "Twelve cases are frozen.")
    add(checks, "expected_reports", config["cohort"]["expected_hybrid_reports"] == 36, "Thirty six reports are planned.")
    add(checks, "three_repetitions", config["cohort"]["llm_repetitions_per_case"] == 3, "Three retained repetitions per case.")
    add(checks, "no_case_exclusion", config["cohort"]["case_exclusion_permitted"] is False, "Case exclusion is prohibited.")
    add(checks, "no_best_output", config["cohort"]["best_output_selection_permitted"] is False, "Best output selection is prohibited.")
    add(checks, "review_repeat_frozen", config["cohort"]["human_review_repetition"] == 1, "Repeat one is prospectively selected for review.")
    add(checks, "four_locked_facts", config["hybrid_contract"]["locked_fact_count_per_case"] == 4, "Every case locks four facts.")
    add(checks, "only_summary_is_llm", config["hybrid_contract"]["llm_fields"] == ["executive_summary"], "Only the summary is generated.")
    add(checks, "numeric_tokens_prohibited", config["hybrid_contract"]["llm_numeric_tokens_permitted"] is False, "Narrative digits are prohibited.")
    add(checks, "locked_fields_complete", len(config["hybrid_contract"]["deterministic_fields"]) == 8, "Eight report fields are deterministic.")
    add(checks, "model_frozen", config["model"]["model_id"] == "gpt-5.6-terra", "Model identifier is frozen.")
    add(checks, "reasoning_frozen", config["model"]["reasoning_effort"] == "low", "Reasoning effort is low.")
    add(checks, "token_cap_frozen", config["model"]["max_output_tokens"] == 250, "Compact narrative token cap is frozen.")
    add(checks, "no_retries", config["model"]["max_retries"] == 0, "Retries are disabled.")
    add(checks, "no_tools", config["model"]["tools_enabled"] is False, "Tools are disabled.")
    add(checks, "no_history", config["model"]["conversation_history_enabled"] is False, "Conversation history is disabled.")
    add(checks, "call_limit", config["model"]["maximum_api_calls_task62_c2"] == 36, "At most thirty six calls are permitted.")
    add(checks, "cost_ceiling", float(config["cost_control"]["hard_estimated_cost_ceiling_usd"]) == 0.25, "Cost ceiling is frozen.")
    add(checks, "c0_c1_no_calls", config["api_calls_c0_c1"] == 0, "C0 and C1 make no API calls.")
    add(checks, "training_prohibited", config["data_boundaries"]["training_permitted"] is False, "Training is prohibited.")
    add(checks, "new_shap_prohibited", config["data_boundaries"]["new_shap_evaluations_permitted"] is False, "New SHAP evaluation is prohibited.")
    add(checks, "reserved_test_closed", config["data_boundaries"]["reserved_test_arrays_materialized"] is False, "Reserved test arrays remain closed.")
    add(checks, "authority_deterministic", config["data_boundaries"]["security_decision_authority"] == "deterministic_pipeline", "Decision authority remains deterministic.")
    add(checks, "two_reviewers", config["human_evaluation"]["minimum_independent_reviewers"] == 2, "Two independent reviewers are required.")
    add(checks, "blinded_review", config["human_evaluation"]["blinded"] is True, "Human review is blinded.")
    add(checks, "superiority_blocked", config["human_evaluation"]["superiority_claim_blocked_until_review_complete"] is True, "Superiority is blocked.")
    add(checks, "all_automated_gates_frozen", len(config["automated_metrics"]) == 10, "Ten automated gates are frozen.")
    add(checks, "exact_preservation_gate", config["automated_metrics"]["exact_locked_content_preservation_rate_required"] == 1.0, "Exact preservation is required.")
    add(checks, "repeat_consistency_gate", config["automated_metrics"]["repeat_numeric_consistency_rate_required"] == 1.0, "Exact numeric repeat consistency is required.")
    add(checks, "schema_blocks_digits", narrative_schema["properties"]["executive_summary"]["pattern"] == "^[^0-9]*$", "Narrative schema blocks digits.")
    add(checks, "schema_has_two_fields", set(narrative_schema["required"]) == {"schema_version", "executive_summary"}, "Narrative object is minimal.")
    add(checks, "system_prompt_has_no_digits", not re.search(r"[0-9]", system_text), "System prompt contains no numeric token.")
    add(checks, "user_prompt_has_no_digits", not re.search(r"[0-9]", user_text), "User prompt contains no numeric token.")
    add(checks, "system_prompt_prohibits_digits", "Do not place any digit in the executive summary" in system_text, "Digit prohibition is scoped to the generated narrative field.")
    add(checks, "system_prompt_protects_authority", "security recommendation" in system_text and "client decision" in system_text, "Decision boundaries are explicit.")
    add(checks, "user_prompt_preserves_limits", "does not establish malicious intent or oracle clean restoration" in user_text, "Unsupported claims are blocked.")
    add(checks, "protocol_declares_hybrid_split", "deterministic stage" in protocol_text and "language model produces only" in protocol_text, "Protocol states the architectural split.")
    add(checks, "report_schema_present", REPORT_SCHEMA.is_file(), "Frozen final report schema exists.")
    add(checks, "evidence_count", len(list((ROOT / config["inputs"]["c1_evidence_root"]).glob("*.json"))) == 12, "Twelve evidence records exist.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    table_root = OUT_ROOT / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    with (table_root / "task62c0_preregistration_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "passed", "detail"])
        writer.writeheader()
        writer.writerows(checks)
    sources = [CONFIG_PATH, PROTOCOL_MD, SYSTEM_PROMPT, USER_PROMPT, NARRATIVE_SCHEMA, REPORT_SCHEMA, C3_DECISION, C3_AUDIT]
    with (table_root / "task62c0_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256"])
        writer.writeheader()
        for path in sorted(sources, key=lambda item: item.relative_to(ROOT).as_posix()):
            writer.writerow({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
    all_passed = all(row["passed"] for row in checks)
    decision = {
        "all_checks_passed": all_passed,
        "api_calls": 0,
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "hybrid_contract_frozen": all_passed,
        "llm_numeric_tokens_permitted": False,
        "new_shap_evaluations": 0,
        "parent_task61_result_retained": "FAIL",
        "ready_for_c1_hybrid_matrix": all_passed,
        "reserved_test_arrays_materialized": False,
        "task62_outcomes_inspected": False,
        "training_permitted": False,
    }
    write_json(OUT_ROOT / "task62c0_preregistration_decision.json", decision)
    print("===== TASK 62 C0 HYBRID PREREGISTRATION AUDIT =====")
    print(f"Checks passed: {decision['checks_passed']}/{decision['checks_total']}")
    print("TASK 61 FAILURE RETAINED: True")
    print("LLM NUMERIC TOKENS PERMITTED: False")
    print("LLM CALLS: 0")
    print("TRAINING PERMITTED: False")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR C1 HYBRID MATRIX:", all_passed)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

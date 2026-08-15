from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
from jsonschema import Draft202012Validator

from task62_c2_common_v425 import (
    C1_ROOT,
    C2_ROOT,
    CONFIG_PATH,
    FINAL_SCHEMA_PATH,
    NARRATIVE_SCHEMA_PATH,
    PREFLIGHT_ROOT,
    SYSTEM_PROMPT_PATH,
    USER_PROMPT_PATH,
    canonical_hash,
    load_json,
    report_paths,
    valid_completion,
    write_json,
)


def main() -> int:
    config = load_json(CONFIG_PATH)
    c1_decision = load_json(
        CONFIG_PATH.parents[1] / "results" / "cic_iot_diad_task62_c1_audit_v425" / "task62c1_hybrid_matrix_audit_decision.json"
    )
    plan = pd.read_csv(C1_ROOT / "tables" / "task62c1_llm_call_plan.csv").sort_values("call_index")
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append((name, bool(passed), detail))

    check("c1_audit_passed", c1_decision.get("all_checks_passed") is True)
    check("c1_ready_for_c2", c1_decision.get("ready_for_task62_c2_hybrid_run") is True)
    check("plan_has_36_calls", len(plan) == 36, str(len(plan)))
    check("plan_indices_exact", plan.call_index.tolist() == list(range(1, 37)))
    check("plan_has_12_cases", plan.case_id.nunique() == 12, str(plan.case_id.nunique()))
    check("plan_has_three_repeats_each", bool((plan.groupby("case_id").size() == 3).all()))
    check("model_exact", set(plan.model) == {config["model"]["model_id"]})
    check("reasoning_exact", set(plan.reasoning_effort) == {config["model"]["reasoning_effort"]})
    check("output_limit_exact", set(plan.max_output_tokens) == {config["model"]["max_output_tokens"]})
    check("max_retries_zero", config["model"]["max_retries"] == 0)
    check("tools_disabled", config["model"]["tools_enabled"] is False)
    check("conversation_history_disabled", config["model"]["conversation_history_enabled"] is False)
    check("training_forbidden", config["data_boundaries"]["training_permitted"] is False)
    check("new_shap_forbidden", config["data_boundaries"]["new_shap_evaluations_permitted"] is False)
    check("reserved_arrays_closed", config["data_boundaries"]["reserved_test_arrays_materialized"] is False)
    check("narrative_schema_valid", Draft202012Validator.check_schema(load_json(NARRATIVE_SCHEMA_PATH)) is None)
    check("final_schema_valid", Draft202012Validator.check_schema(load_json(FINAL_SCHEMA_PATH)) is None)
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = USER_PROMPT_PATH.read_text(encoding="utf-8")
    check("system_prompt_exists", bool(system_prompt.strip()))
    check("user_prompt_exists", bool(user_prompt.strip()))
    check("prompt_forbids_digits", "Do not use digits" in user_prompt and "Do not place any digit" in system_prompt)
    check("prompt_blocks_intent_claim", "malicious intent claim" in system_prompt)
    check("prompt_blocks_oracle_claim", "oracle clean claim" in system_prompt)
    check("planned_cost_within_ceiling", 0.1656 <= config["cost_control"]["hard_estimated_cost_ceiling_usd"])

    unresolved = 0
    contracts_valid = True
    contract_hashes_valid = True
    for call in plan.itertuples():
        contract_path = CONFIG_PATH.parents[1] / str(call.contract_path)
        contract = load_json(contract_path)
        contracts_valid &= contract.get("case_id") == str(call.case_id)
        contracts_valid &= contract.get("llm_allowed_fields") == ["executive_summary"]
        contracts_valid &= contract.get("llm_numeric_tokens_permitted") is False
        contract_hashes_valid &= contract.get("locked_content_sha256") == canonical_hash(contract["locked_content"])
        paths = report_paths(str(call.case_id), int(call.repetition))
        if paths["attempt"].exists() and not valid_completion(paths):
            unresolved += 1
    check("all_contracts_valid", contracts_valid)
    check("all_contract_hashes_valid", contract_hashes_valid)
    check("no_unresolved_attempts", unresolved == 0, str(unresolved))

    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    with (PREFLIGHT_ROOT / "task62c2_preflight_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check", "passed", "detail"])
        writer.writerows(checks)
    passed = sum(item[1] for item in checks)
    decision = {
        "all_checks_passed": passed == len(checks),
        "checks_passed": passed,
        "checks_total": len(checks),
        "planned_api_calls": 36,
        "existing_valid_completions": sum(valid_completion(report_paths(str(row.case_id), int(row.repetition))) for row in plan.itertuples()),
        "unresolved_attempts": unresolved,
        "ready_for_task62_c2_execution": passed == len(checks),
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
    }
    write_json(PREFLIGHT_ROOT / "task62c2_preflight_decision.json", decision)
    print("===== TASK 62 C2 EXECUTION PREFLIGHT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("PLANNED API CALLS: 36")
    print("UNRESOLVED ATTEMPTS:", unresolved)
    print("READY FOR C2 EXECUTION:", decision["ready_for_task62_c2_execution"])
    return 0 if decision["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

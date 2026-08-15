from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator

from task62_c2_common_v425 import (
    AUDIT_ROOT,
    C1_ROOT,
    C2_ROOT,
    CONFIG_PATH,
    FINAL_SCHEMA_PATH,
    NARRATIVE_SCHEMA_PATH,
    SUMMARY_ROOT,
    canonical_hash,
    file_hash,
    load_json,
    report_paths,
    usage_cost,
    valid_completion,
    write_json,
)

NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def independent_numbers(report: dict[str, Any]) -> list[float]:
    parts = [str(report.get("executive_summary", ""))]
    for section in ("facts", "interpretations"):
        parts.extend(str(item.get("statement", "")) for item in report.get(section, []))
    parts.extend(str(item) for item in report.get("uncertainties", []))
    parts.extend(str(item) for item in report.get("refusals", []))
    return [float(token) for token in NUMBER_RE.findall("\n".join(parts))]


def repeat_score(repeats: list[list[float]]) -> float:
    pair_scores = []
    for left_index in range(3):
        for right_index in range(left_index + 1, 3):
            left, right = repeats[left_index], repeats[right_index]
            matched = sum(any(math.isclose(value, candidate, rel_tol=1e-9, abs_tol=5e-7) for candidate in right) for value in left)
            pair_scores.append(matched / max(len(left), len(right), 1))
    return sum(pair_scores) / len(pair_scores)


def main() -> int:
    config = load_json(CONFIG_PATH)
    plan = pd.read_csv(C1_ROOT / "tables" / "task62c1_llm_call_plan.csv").sort_values("call_index")
    narrative_validator = Draft202012Validator(load_json(NARRATIVE_SCHEMA_PATH))
    final_validator = Draft202012Validator(load_json(FINAL_SCHEMA_PATH))
    execution = load_json(C2_ROOT / "task62c2_execution_complete.json")
    summary_decision = load_json(SUMMARY_ROOT / "task62c2_automated_evaluation_decision.json")
    summary_gates = pd.read_csv(SUMMARY_ROOT / "tables" / "task62c2_automated_gate_decisions.csv")
    summary_metrics = pd.read_csv(SUMMARY_ROOT / "tables" / "task62c2_report_metrics.csv")
    summary_consistency = pd.read_csv(SUMMARY_ROOT / "tables" / "task62c2_case_repeat_consistency.csv")
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append((name, bool(passed), detail))

    check("plan_has_36_calls", len(plan) == 36, str(len(plan)))
    check("plan_has_12_cases", plan.case_id.nunique() == 12)
    check("plan_three_repeats_each", bool((plan.groupby("case_id").size() == 3).all()))
    check("execution_complete", execution.get("complete") is True)
    check("execution_completed_reports_36", execution.get("completed_reports") == 36)
    check("execution_maximum_calls_36", execution.get("maximum_api_calls") == 36)
    check("execution_best_selection_false", execution.get("best_output_selection_permitted") is False)
    check("execution_training_false", execution.get("training_permitted") is False)
    check("execution_new_shap_zero", execution.get("new_shap_evaluations") == 0)
    check("execution_reserved_arrays_false", execution.get("reserved_test_arrays_materialized") is False)

    reports = sorted(C2_ROOT.glob("reports/*/repeat_[123]_final.json"))
    narratives = sorted(C2_ROOT.glob("reports/*/repeat_[123]_narrative.json"))
    metadata_files = sorted(C2_ROOT.glob("reports/*/repeat_[123]_metadata.json"))
    markers = sorted(C2_ROOT.glob("reports/*/repeat_[123]_complete.json"))
    attempts = sorted(C2_ROOT.glob("reports/*/repeat_[123]_attempt.json"))
    check("final_report_files_36", len(reports) == 36, str(len(reports)))
    check("narrative_files_36", len(narratives) == 36, str(len(narratives)))
    check("metadata_files_36", len(metadata_files) == 36, str(len(metadata_files)))
    check("completion_markers_36", len(markers) == 36, str(len(markers)))
    check("attempt_journals_36", len(attempts) == 36, str(len(attempts)))
    check("progress_rows_36", len(pd.read_csv(C2_ROOT / "task62c2_progress.csv")) == 36)

    deterministic_fields = config["hybrid_contract"]["deterministic_fields"]
    all_narrative_schema = True
    all_final_schema = True
    all_locked_exact = True
    all_digit_free = True
    all_hashes = True
    all_markers = True
    all_attempts = True
    all_metadata = True
    all_usage = True
    all_numbers_grounded = True
    all_facts_covered = True
    response_ids: list[str] = []
    by_case: dict[str, list[list[float]]] = {}
    total_calls = 0
    total_cost = 0.0
    independent_metric_rows = []

    for call in plan.itertuples():
        case_id = str(call.case_id)
        repetition = int(call.repetition)
        paths = report_paths(case_id, repetition)
        contract = load_json(CONFIG_PATH.parents[1] / str(call.contract_path))
        narrative = load_json(paths["narrative"])
        report = load_json(paths["final"])
        metadata = load_json(paths["metadata"])
        marker = load_json(paths["complete"])
        attempt = load_json(paths["attempt"])
        all_narrative_schema &= not list(narrative_validator.iter_errors(narrative))
        all_final_schema &= not list(final_validator.iter_errors(report))
        locked = {field: report[field] for field in deterministic_fields}
        all_locked_exact &= canonical_hash(locked) == contract["locked_content_sha256"] == canonical_hash(contract["locked_content"])
        all_digit_free &= re.search(r"[0-9]", str(narrative.get("executive_summary", ""))) is None
        all_hashes &= metadata.get("narrative_sha256") == file_hash(paths["narrative"])
        all_hashes &= metadata.get("final_report_sha256") == file_hash(paths["final"])
        all_hashes &= marker.get("metadata_sha256") == file_hash(paths["metadata"])
        all_markers &= valid_completion(paths) and marker.get("api_calls") == 1
        all_attempts &= attempt.get("status") == "completed" and attempt.get("maximum_permitted_attempts_for_item") == 1
        all_attempts &= attempt.get("response_id") == metadata.get("response_id") == marker.get("response_id")
        all_metadata &= metadata.get("protocol_id") == config["protocol_id"]
        all_metadata &= metadata.get("requested_model") == config["model"]["model_id"]
        all_metadata &= metadata.get("returned_model") == config["model"]["model_id"]
        all_metadata &= metadata.get("response_status") == "completed"
        all_metadata &= metadata.get("reasoning_effort") == config["model"]["reasoning_effort"]
        all_metadata &= metadata.get("max_output_tokens") == config["model"]["max_output_tokens"]
        all_metadata &= metadata.get("max_retries") == 0 and metadata.get("tools_enabled") is False
        all_metadata &= metadata.get("conversation_history_enabled") is False
        all_metadata &= metadata.get("api_key_persisted") is False
        all_metadata &= metadata.get("training_permitted") is False
        all_metadata &= metadata.get("new_shap_evaluations") == 0
        all_metadata &= metadata.get("reserved_test_arrays_materialized") is False
        usage = metadata.get("usage") or {}
        all_usage &= int(usage.get("input_tokens") or 0) > 0
        all_usage &= int(usage.get("output_tokens") or 0) > 0
        all_usage &= int(usage.get("total_tokens") or 0) >= int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
        values = independent_numbers(report)
        allowed = independent_numbers(contract["locked_content"])
        grounded = sum(any(math.isclose(value, candidate, rel_tol=1e-9, abs_tol=5e-7) for candidate in allowed) for value in values)
        all_numbers_grounded &= grounded == len(values)
        all_facts_covered &= all(fact in report.get("facts", []) for fact in contract["locked_content"]["facts"])
        by_case.setdefault(case_id, []).append(values)
        response_ids.append(str(metadata.get("response_id") or ""))
        total_calls += int(metadata.get("api_calls") or 0)
        total_cost += usage_cost(metadata, config)
        independent_metric_rows.append((int(call.call_index), len(values), grounded))

    check("all_narratives_schema_valid", all_narrative_schema)
    check("all_final_reports_schema_valid", all_final_schema)
    check("all_locked_content_exact", all_locked_exact)
    check("all_executive_summaries_digit_free", all_digit_free)
    check("all_file_hashes_exact", all_hashes)
    check("all_completion_markers_valid", all_markers)
    check("all_attempt_journals_completed", all_attempts)
    check("all_metadata_contracts_exact", all_metadata)
    check("all_usage_positive", all_usage)
    check("all_reported_numbers_grounded", all_numbers_grounded)
    check("all_required_facts_covered", all_facts_covered)
    check("api_calls_exactly_36", total_calls == 36, str(total_calls))
    check("response_ids_present", all(response_ids))
    check("response_ids_unique", len(set(response_ids)) == 36)
    independent_consistency = sum(repeat_score(repeats) for repeats in by_case.values()) / len(by_case)
    reported_consistency = float(summary_consistency.repeat_numeric_consistency_rate.mean())
    check("repeat_numeric_consistency_recomputed", math.isclose(independent_consistency, reported_consistency, abs_tol=1e-12), f"{independent_consistency:.12f}")
    check("repeat_numeric_consistency_exact_one", math.isclose(independent_consistency, 1.0, abs_tol=1e-12), f"{independent_consistency:.12f}")
    check("summary_metrics_rows_36", len(summary_metrics) == 36)
    check("summary_gates_rows_10", len(summary_gates) == 10)
    check("summary_gates_all_passed", bool(summary_gates.passed.all()))
    check("summary_decision_pass", summary_decision.get("automated_gate_result") == "PASS")
    check("summary_human_review_pending", summary_decision.get("human_review_complete") is False)
    check("summary_superiority_blocked", summary_decision.get("llm_superiority_claim_permitted") is False)
    check("summary_task61_failure_retained", summary_decision.get("task61_failure_retained") is True)
    check("summary_ready_for_review", summary_decision.get("ready_for_independent_human_review") is True)
    check("cost_recomputed", math.isclose(total_cost, float(summary_decision["estimated_cost_usd"]), abs_tol=5e-10), f"{total_cost:.9f}")
    check("cost_within_ceiling", total_cost <= float(config["cost_control"]["hard_estimated_cost_ceiling_usd"]), f"{total_cost:.9f}")

    review_root = SUMMARY_ROOT / "human_review_package"
    public_manifest = pd.read_csv(review_root / "task62c2_blinded_review_public_manifest.csv")
    check("public_review_manifest_24", len(public_manifest) == 24)
    check("blinded_items_24", len(list((review_root / "blinded_items").glob("item_*.txt"))) == 24)
    check("reviewer_one_template_24", len(pd.read_csv(review_root / "task62c2_reviewer_1_scores.csv")) == 24)
    check("reviewer_two_template_24", len(pd.read_csv(review_root / "task62c2_reviewer_2_scores.csv")) == 24)
    check("public_manifest_hides_condition", "condition" not in public_manifest.columns and "case_id" not in public_manifest.columns)
    check("private_blinding_key_retained", (SUMMARY_ROOT / "tables" / "task62c2_private_blinding_key.csv").exists())
    check("human_instructions_present", (review_root / "TASK62_C2_HUMAN_REVIEW_INSTRUCTIONS.md").exists())

    runner_text = (CONFIG_PATH.parents[1] / "scripts" / "run_task62_c2_hybrid_v425.py").read_text(encoding="utf-8")
    check("runner_sets_max_retries_zero", "OpenAI(max_retries=0)" in runner_text)
    check("runner_journals_before_api_call", runner_text.index('write_json(paths["attempt"]') < runner_text.index("client.responses.parse"))
    check("runner_blocks_unresolved_attempt", "Unresolved API attempt" in runner_text)
    check("runner_uses_structured_parse", "client.responses.parse" in runner_text and "text_format=HybridNarrative" in runner_text)
    check("runner_rejects_incomplete_status", 'status != "completed"' in runner_text)
    check("runner_does_not_enable_tools", "tools=[" not in runner_text)
    check("no_api_key_in_metadata", all("OPENAI_API_KEY" not in path.read_text(encoding="utf-8") for path in metadata_files))

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    tables = AUDIT_ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "task62c2_audit_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check", "passed", "detail"])
        writer.writerows(checks)
    source_paths = [CONFIG_PATH, NARRATIVE_SCHEMA_PATH, FINAL_SCHEMA_PATH]
    source_paths.extend(sorted(C1_ROOT.rglob("*.json")))
    source_paths.extend(sorted(C1_ROOT.rglob("*.csv")))
    source_paths.extend(reports + narratives + metadata_files + markers + attempts)
    with (tables / "task62c2_audit_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256"])
        for path in sorted(set(source_paths)):
            writer.writerow([str(path.relative_to(CONFIG_PATH.parents[1])).replace("\\", "/"), file_hash(path)])
    passed = sum(item[1] for item in checks)
    decision = {
        "all_checks_passed": passed == len(checks),
        "checks_passed": passed,
        "checks_total": len(checks),
        "automated_result_verified": summary_decision.get("automated_gate_result"),
        "reports_verified": len(reports),
        "exact_locked_content_verified": all_locked_exact,
        "repeat_numeric_consistency_independently_recomputed": round(independent_consistency, 12),
        "api_calls_verified": total_calls,
        "estimated_cost_usd_independently_recomputed": round(total_cost, 9),
        "human_review_complete": False,
        "llm_superiority_claim_permitted": False,
        "ready_for_independent_human_review": passed == len(checks) and summary_decision.get("automated_gate_result") == "PASS",
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
    }
    write_json(AUDIT_ROOT / "task62c2_audit_decision.json", decision)
    print("===== TASK 62 C2 INDEPENDENT AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("AUTOMATED RESULT VERIFIED:", decision["automated_result_verified"])
    print("REPORTS VERIFIED:", decision["reports_verified"])
    print("LOCKED CONTENT EXACT:", decision["exact_locked_content_verified"])
    print("REPEAT NUMERIC CONSISTENCY:", f"{independent_consistency:.12f}")
    print("READY FOR INDEPENDENT HUMAN REVIEW:", decision["ready_for_independent_human_review"])
    return 0 if decision["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

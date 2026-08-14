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


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "task61_preregistration_v4230.json"
C1_ROOT = ROOT / "results" / "cic_iot_diad_task61_c1_evidence_matrix_v423"
C2_ROOT = ROOT / "results" / "cic_iot_diad_task61_c2_llm_reports_v423"
SUMMARY_ROOT = ROOT / "results" / "cic_iot_diad_task61_c2_summary_v423"
AUDIT_ROOT = ROOT / "results" / "cic_iot_diad_task61_c2_audit_v423"
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def narrative(report: dict[str, Any]) -> str:
    parts = [report.get("executive_summary", "")]
    for section in ("facts", "interpretations"):
        parts.extend(str(item.get("statement", "")) for item in report.get(section, []))
    parts.extend(str(item) for item in report.get("uncertainties", []))
    parts.extend(str(item) for item in report.get("refusals", []))
    return " ".join(parts)


def numbers(text: str) -> list[float]:
    return [float(token) for token in NUMBER_RE.findall(text)]


def evidence_numbers(value: Any) -> list[float]:
    output: list[float] = []
    if isinstance(value, bool) or value is None:
        return output
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            output.append(float(value))
    elif isinstance(value, str):
        output.extend(numbers(value))
    elif isinstance(value, list):
        for item in value:
            output.extend(evidence_numbers(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in {"record_id", "case_id", "sha256", "path", "schema_version"}:
                output.extend(evidence_numbers(item))
    return output


def grounded(value: float, allowed: list[float]) -> bool:
    return any(
        math.isclose(value, round(candidate, digits), rel_tol=1e-9, abs_tol=5e-7)
        for candidate in allowed
        for digits in range(7)
    )


def compact_evidence(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "case_id": record["case_id"],
        "cohort_selection": record["cohort_selection"],
        "experiment": record["experiment"],
        "attack": record["attack"],
        "detection": record["detection"],
        "reconstruction": record["reconstruction"],
        "behavior": record["behavior"],
        "xai": record["xai"],
        "observed_facts": record["observed_facts"],
        "interpretations": record["interpretations"],
        "uncertainty": record["uncertainty"],
        "authority": record["authority"],
        "valid_sources": [
            {"source_id": item["source_id"], "sha256": item["sha256"]}
            for item in record["provenance"]["source_artifacts"]
        ],
    }


def main() -> int:
    config = load_json(CONFIG_PATH)
    schema = load_json(ROOT / config["inputs"]["report_schema"])
    validator = Draft202012Validator(schema)
    plan = pd.read_csv(C1_ROOT / "tables" / "task61c1_llm_call_plan.csv").sort_values("call_index")
    cases = pd.read_csv(C1_ROOT / "tables" / "task61c1_case_manifest.csv")
    reports = sorted(C2_ROOT.glob("reports/*/repeat_[123].json"))
    metadata_files = sorted(C2_ROOT.glob("reports/*/repeat_[123]_metadata.json"))
    markers = sorted(C2_ROOT.glob("reports/*/repeat_[123]_complete.json"))
    report_metrics = pd.read_csv(SUMMARY_ROOT / "tables" / "task61c2_report_metrics.csv")
    consistency = pd.read_csv(SUMMARY_ROOT / "tables" / "task61c2_case_repeat_consistency.csv")
    gates = pd.read_csv(SUMMARY_ROOT / "tables" / "task61c2_automated_gate_decisions.csv")
    cost = pd.read_csv(SUMMARY_ROOT / "tables" / "task61c2_cost_summary.csv").iloc[0]
    decision = load_json(SUMMARY_ROOT / "task61c2_automated_evaluation_decision.json")
    records = {str(row.case_id): load_json(ROOT / str(row.evidence_path)) for row in cases.itertuples()}
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, value: bool, detail: str = "") -> None:
        checks.append((name, bool(value), detail))

    check("plan_has_36_calls", len(plan) == 36, str(len(plan)))
    check("plan_indices_exact", plan.call_index.tolist() == list(range(1, 37)))
    check("case_count_12", len(cases) == 12, str(len(cases)))
    check("three_repeats_each", bool((plan.groupby("case_id").size() == 3).all()))
    check("report_files_36", len(reports) == 36, str(len(reports)))
    check("metadata_files_36", len(metadata_files) == 36, str(len(metadata_files)))
    check("completion_markers_36", len(markers) == 36, str(len(markers)))
    check("progress_rows_36", len(pd.read_csv(C2_ROOT / "task61c2_progress.csv")) == 36)
    execution = load_json(C2_ROOT / "task61c2_execution_complete.json")
    check("execution_complete", execution.get("complete") is True)
    check("execution_reports_36", execution.get("completed_reports") == 36)
    check("execution_no_best_selection", execution.get("best_output_selection_permitted") is False)
    check("execution_training_false", execution.get("training_permitted") is False)
    check("execution_new_shap_zero", execution.get("new_shap_evaluations") == 0)
    check("execution_reserved_arrays_false", execution.get("reserved_test_arrays_materialized") is False)

    all_schema = True
    all_record_ids = True
    all_citations = True
    all_hashes = True
    all_markers = True
    all_metadata = True
    all_usage = True
    all_grounded = True
    response_ids: list[str] = []
    total_api_calls = 0
    total_cost = 0.0
    computed_metric_rows = []
    for call in plan.itertuples():
        report_path = ROOT / str(call.planned_output_path)
        metadata_path = report_path.with_name(report_path.stem + "_metadata.json")
        marker_path = report_path.with_name(report_path.stem + "_complete.json")
        report = load_json(report_path)
        metadata = load_json(metadata_path)
        marker = load_json(marker_path)
        record = records[str(call.case_id)]
        all_schema &= not list(validator.iter_errors(report))
        all_record_ids &= report.get("record_id") == record["record_id"] == metadata.get("record_id") == marker.get("record_id")
        valid_refs = {item["source_id"] for item in record["provenance"]["source_artifacts"]}
        used_refs = set()
        for section in ("facts", "interpretations"):
            for item in report.get(section, []):
                used_refs.update(item.get("evidence_refs", []))
        all_citations &= bool(used_refs) and used_refs <= valid_refs
        report_hash = sha256(report_path)
        all_hashes &= report_hash == metadata.get("report_sha256") == marker.get("report_sha256")
        all_hashes &= metadata.get("record_sha256") == canonical_hash(record)
        all_hashes &= metadata.get("compact_evidence_sha256") == canonical_hash(compact_evidence(record))
        all_markers &= marker.get("complete") is True and marker.get("api_calls") == 1
        all_metadata &= metadata.get("requested_model") == config["model"]["model_id"]
        all_metadata &= metadata.get("returned_model") == config["model"]["model_id"]
        all_metadata &= metadata.get("response_status") == "completed"
        all_metadata &= metadata.get("reasoning_effort") == "low" and metadata.get("max_output_tokens") == 700
        all_metadata &= metadata.get("max_retries") == 0 and metadata.get("tools_enabled") is False
        all_metadata &= metadata.get("conversation_history_enabled") is False and metadata.get("api_key_persisted") is False
        all_metadata &= metadata.get("training_permitted") is False and metadata.get("new_shap_evaluations") == 0
        all_metadata &= metadata.get("reserved_test_arrays_materialized") is False
        usage = metadata.get("usage") or {}
        all_usage &= int(usage.get("input_tokens") or 0) > 0 and int(usage.get("output_tokens") or 0) > 0
        all_usage &= int(usage.get("total_tokens") or 0) >= int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
        values = numbers(narrative(report))
        allowed = evidence_numbers(record)
        grounded_count = sum(grounded(value, allowed) for value in values)
        all_grounded &= grounded_count == len(values)
        computed_metric_rows.append((int(call.call_index), len(values), grounded_count))
        response_ids.append(str(metadata.get("response_id") or ""))
        total_api_calls += int(metadata.get("api_calls") or 0)
        input_details = usage.get("input_tokens_details") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        cached = int(input_details.get("cached_tokens") or 0)
        cache_write = int(input_details.get("cache_write_tokens") or 0)
        ordinary = max(input_tokens - cached - cache_write, 0)
        total_cost += ordinary * 2.0 / 1_000_000 + cached * 0.20 / 1_000_000 + cache_write * 2.5 / 1_000_000 + output_tokens * 12.0 / 1_000_000

    check("all_reports_schema_valid", all_schema)
    check("all_record_ids_exact", all_record_ids)
    check("all_citations_valid", all_citations)
    check("all_report_and_record_hashes_exact", all_hashes)
    check("all_completion_markers_valid", all_markers)
    check("all_metadata_contracts_exact", all_metadata)
    check("all_usage_positive", all_usage)
    check("all_narrative_numbers_grounded", all_grounded)
    check("api_calls_exactly_36", total_api_calls == 36, str(total_api_calls))
    check("response_ids_present", all(response_ids))
    check("response_ids_unique", len(set(response_ids)) == 36)

    check("report_metrics_rows_36", len(report_metrics) == 36)
    check("report_metrics_indices_exact", sorted(report_metrics.call_index.tolist()) == list(range(1, 37)))
    check("metric_schema_rate_exact", float(report_metrics.schema_valid.mean()) == 1.0)
    check("metric_citation_rate_exact", float(report_metrics.citation_valid.mean()) == 1.0)
    computed = pd.DataFrame(computed_metric_rows, columns=["call_index", "values", "grounded"])
    merged = report_metrics.merge(computed, on="call_index")
    check("numeric_counts_independently_recomputed", bool((merged.numeric_values_reported == merged["values"]).all() and (merged.numeric_values_grounded == merged.grounded).all()))
    check("case_consistency_rows_12", len(consistency) == 12)
    check("consistency_cases_unique", consistency.case_id.nunique() == 12)
    check("automated_gate_rows_8", len(gates) == 8)
    check("gate_names_unique", gates.metric.nunique() == 8)
    check("gate_pass_count_matches_decision", int(gates.passed.sum()) == int(decision["automated_gates_passed"]))
    check("gate_total_matches_decision", len(gates) == int(decision["automated_gates_total"]))
    expected_gate_result = "PASS" if bool(gates.passed.all()) else "FAIL"
    check("automated_result_recomputed", decision["automated_gate_result"] == expected_gate_result)

    check("cost_api_calls_36", int(cost.api_calls) == 36)
    check("cost_sum_recomputed", math.isclose(float(cost.estimated_cost_usd), total_cost, rel_tol=1e-9, abs_tol=1e-9))
    check("cost_below_hard_ceiling", float(cost.estimated_cost_usd) <= float(cost.hard_cost_ceiling_usd))
    check("cost_decision_consistent", math.isclose(float(decision["estimated_cost_usd"]), float(cost.estimated_cost_usd), rel_tol=1e-12, abs_tol=1e-12))

    review_root = SUMMARY_ROOT / "human_review_package"
    items = sorted((review_root / "blinded_items").glob("item_*.txt"))
    public_manifest = pd.read_csv(review_root / "task61c2_blinded_review_public_manifest.csv")
    private_key = pd.read_csv(SUMMARY_ROOT / "tables" / "task61c2_private_blinding_key.csv")
    c1_private = pd.read_csv(C1_ROOT / "tables" / "task61c1_blinded_review_manifest.csv")
    reviewer_template = pd.read_csv(review_root / "task61c2_reviewer_score_template.csv")
    check("blinded_items_36", len(items) == 36)
    check("public_manifest_rows_36", len(public_manifest) == 36)
    check("public_manifest_has_no_case_ids", "case_id" not in public_manifest.columns)
    check("public_manifest_has_no_comparator", "comparator" not in public_manifest.columns)
    check("private_key_matches_frozen_C1", private_key.equals(c1_private))
    check("review_template_rows_36", len(reviewer_template) == 36)
    score_columns = [column for column in reviewer_template.columns if column != "blinded_item_id"]
    check("review_template_unscored", bool(reviewer_template[score_columns].isna().all().all()))
    check("review_instructions_exist", (review_root / "TASK61_C2_HUMAN_REVIEW_INSTRUCTIONS.md").is_file())

    check("decision_reports_36", decision.get("reports_evaluated") == 36)
    check("decision_cases_12", decision.get("cases_evaluated") == 12)
    check("human_review_incomplete", decision.get("human_review_complete") is False)
    check("conclusion_pending_human_review", decision.get("final_task61_conclusion") == "PENDING_HUMAN_REVIEW")
    check("superiority_claim_blocked", decision.get("llm_superiority_claim_permitted") is False)
    check("all_outputs_retained", decision.get("all_negative_outputs_retained") is True)
    check("best_output_selection_blocked", decision.get("best_output_selection_permitted") is False)
    check("training_prohibited", decision.get("training_permitted") is False)
    check("new_shap_zero", decision.get("new_shap_evaluations") == 0)
    check("reserved_arrays_closed", decision.get("reserved_test_arrays_materialized") is False)
    check("closeout_exists", (SUMMARY_ROOT / "TASK61_C2_CLOSEOUT.md").is_file())

    tables = AUDIT_ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "task61c2_audit_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check_id", "passed", "detail"])
        writer.writerows(checks)
    source_paths = reports + metadata_files + markers
    source_paths += [
        C2_ROOT / "task61c2_progress.csv",
        C2_ROOT / "task61c2_execution_complete.json",
        SUMMARY_ROOT / "tables" / "task61c2_report_metrics.csv",
        SUMMARY_ROOT / "tables" / "task61c2_case_repeat_consistency.csv",
        SUMMARY_ROOT / "tables" / "task61c2_automated_gate_decisions.csv",
        SUMMARY_ROOT / "tables" / "task61c2_cost_summary.csv",
        SUMMARY_ROOT / "task61c2_automated_evaluation_decision.json",
        SUMMARY_ROOT / "TASK61_C2_CLOSEOUT.md",
    ] + items
    with (tables / "task61c2_audit_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256"])
        for path in source_paths:
            writer.writerow([path.relative_to(ROOT).as_posix(), sha256(path)])
    passed = sum(value for _, value, _ in checks)
    audit_decision = {
        "checks_passed": passed,
        "checks_total": len(checks),
        "all_checks_passed": passed == len(checks),
        "reports_verified": len(reports),
        "cases_verified": len(cases),
        "api_calls_verified": total_api_calls,
        "schema_conformance_verified": all_schema,
        "citation_validity_verified": all_citations,
        "numeric_grounding_independently_recomputed": all_grounded,
        "automated_gate_result_verified": expected_gate_result,
        "human_review_complete": False,
        "llm_superiority_claim_permitted": False,
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
        "ready_for_independent_human_review": passed == len(checks) and expected_gate_result == "PASS",
    }
    write_json(AUDIT_ROOT / "task61c2_audit_decision.json", audit_decision)
    print("===== TASK 61 C2 INDEPENDENT AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("REPORTS VERIFIED:", audit_decision["reports_verified"])
    print("CASES VERIFIED:", audit_decision["cases_verified"])
    print("API CALLS VERIFIED:", audit_decision["api_calls_verified"])
    print("AUTOMATED GATE RESULT VERIFIED:", audit_decision["automated_gate_result_verified"])
    print("HUMAN REVIEW COMPLETE: False")
    print("LLM SUPERIORITY CLAIM PERMITTED: False")
    print("TRAINING PERMITTED: False")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR INDEPENDENT HUMAN REVIEW:", audit_decision["ready_for_independent_human_review"])
    return 0 if audit_decision["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

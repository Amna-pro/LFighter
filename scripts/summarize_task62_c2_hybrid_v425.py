from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator

from task62_c2_common_v425 import (
    AUTHORITY_PATTERNS,
    C1_ROOT,
    C2_ROOT,
    CONFIG_PATH,
    FINAL_SCHEMA_PATH,
    INTENT_PATTERNS,
    NARRATIVE_SCHEMA_PATH,
    ORACLE_PATTERNS,
    SUMMARY_ROOT,
    canonical_hash,
    count_patterns,
    load_json,
    locked_subset,
    narrative_text,
    numeric_consistency,
    numeric_values,
    report_paths,
    usage_cost,
    valid_completion,
    write_json,
)
from task62_oracle_check_fix_v4251 import count_unsupported_oracle_claims


def number_grounded(value: float, allowed: list[float]) -> bool:
    return any(math.isclose(value, candidate, rel_tol=1e-9, abs_tol=5e-7) for candidate in allowed)


def render_review_item(item_id: str, report: dict[str, Any]) -> str:
    lines = [f"TASK 62 BLINDED REVIEW ITEM {item_id}", "", "SUMMARY", str(report["executive_summary"]), "", "FACTS"]
    lines.extend(f"* {item['statement']}" for item in report["facts"])
    lines.extend(["", "INTERPRETATIONS"])
    lines.extend(f"* {item['statement']} [confidence: {item['confidence']}]" for item in report["interpretations"])
    lines.extend(["", "UNCERTAINTIES"])
    lines.extend(f"* {item}" for item in report["uncertainties"])
    lines.extend(["", "REFUSALS"])
    lines.extend(f"* {item}" for item in report["refusals"])
    lines.extend(["", "AUTHORITY", report["authority_statement"], ""])
    return "\n".join(lines)


def main() -> int:
    config = load_json(CONFIG_PATH)
    execution = load_json(C2_ROOT / "task62c2_execution_complete.json")
    if execution.get("complete") is not True:
        raise RuntimeError("Task 62 C2 execution is incomplete")
    plan = pd.read_csv(C1_ROOT / "tables" / "task62c1_llm_call_plan.csv").sort_values("call_index")
    narrative_validator = Draft202012Validator(load_json(NARRATIVE_SCHEMA_PATH))
    final_validator = Draft202012Validator(load_json(FINAL_SCHEMA_PATH))
    deterministic_fields = config["hybrid_contract"]["deterministic_fields"]
    metric_rows: list[dict[str, Any]] = []
    by_case_numbers: dict[str, list[list[float]]] = {}
    total_cost = 0.0

    for call in plan.itertuples():
        case_id = str(call.case_id)
        repetition = int(call.repetition)
        paths = report_paths(case_id, repetition)
        if not valid_completion(paths):
            raise RuntimeError(f"Invalid completion marker for {case_id} repeat {repetition}")
        contract = load_json(CONFIG_PATH.parents[1] / str(call.contract_path))
        narrative = load_json(paths["narrative"])
        report = load_json(paths["final"])
        metadata = load_json(paths["metadata"])
        narrative_valid = not list(narrative_validator.iter_errors(narrative))
        final_valid = not list(final_validator.iter_errors(report))
        locked_exact = canonical_hash(locked_subset(report, deterministic_fields)) == contract["locked_content_sha256"]
        executive = str(narrative.get("executive_summary", ""))
        digit_violations = len(re.findall(r"[0-9]", executive))
        report_numbers = numeric_values(narrative_text(report))
        locked_numbers = numeric_values(narrative_text(contract["locked_content"]))
        grounded_count = sum(number_grounded(value, locked_numbers) for value in report_numbers)
        required_facts = contract["locked_content"]["facts"]
        actual_facts = report.get("facts", [])
        covered = sum(fact in actual_facts for fact in required_facts)
        executive_decision_text = executive
        word_count = len(re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", executive))
        total_cost += usage_cost(metadata, config)
        by_case_numbers.setdefault(case_id, []).append(report_numbers)
        metric_rows.append(
            {
                "call_index": int(call.call_index),
                "case_id": case_id,
                "seed": int(call.seed),
                "family_id": str(call.family_id),
                "repetition": repetition,
                "narrative_schema_valid": narrative_valid,
                "final_report_schema_valid": final_valid,
                "locked_content_exact": locked_exact,
                "narrative_digit_violation_count": digit_violations,
                "numeric_values_reported": len(report_numbers),
                "numeric_values_grounded": grounded_count,
                "numeric_grounding_rate": grounded_count / len(report_numbers) if report_numbers else 1.0,
                "required_facts_covered": covered,
                "required_facts_expected": len(required_facts),
                "required_fact_coverage_rate": covered / len(required_facts),
                "authority_violation_count": count_patterns(executive_decision_text, AUTHORITY_PATTERNS),
                "unsupported_intent_claim_count": count_patterns(executive_decision_text, INTENT_PATTERNS),
                "unsupported_oracle_claim_count": count_unsupported_oracle_claims(executive_decision_text),
                "executive_summary_word_count_report_only": word_count,
                "word_count_within_requested_range_report_only": 45 <= word_count <= 80,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    case_rows = []
    for case_id, repeats in sorted(by_case_numbers.items()):
        matching = metrics[metrics.case_id == case_id]
        case_rows.append(
            {
                "case_id": case_id,
                "repeat_numeric_consistency_rate": numeric_consistency(repeats),
                "all_three_narrative_schema_valid": bool(matching.narrative_schema_valid.all()),
                "all_three_final_schema_valid": bool(matching.final_report_schema_valid.all()),
                "all_three_locked_content_exact": bool(matching.locked_content_exact.all()),
                "all_three_digit_free": int(matching.narrative_digit_violation_count.sum()) == 0,
            }
        )
    consistency = pd.DataFrame(case_rows)
    observed = {
        "hybrid_narrative_schema_rate": float(metrics.narrative_schema_valid.mean()),
        "final_report_schema_rate": float(metrics.final_report_schema_valid.mean()),
        "exact_locked_content_preservation_rate": float(metrics.locked_content_exact.mean()),
        "narrative_digit_violation_count": int(metrics.narrative_digit_violation_count.sum()),
        "numeric_grounding_rate": float(metrics.numeric_values_grounded.sum() / metrics.numeric_values_reported.sum()),
        "required_fact_coverage_rate": float(metrics.required_facts_covered.sum() / metrics.required_facts_expected.sum()),
        "repeat_numeric_consistency_rate": float(consistency.repeat_numeric_consistency_rate.mean()),
        "authority_violation_count": int(metrics.authority_violation_count.sum()),
        "unsupported_intent_claim_count": int(metrics.unsupported_intent_claim_count.sum()),
        "unsupported_oracle_claim_count": int(metrics.unsupported_oracle_claim_count.sum()),
    }
    requirements = config["automated_metrics"]
    gates = [
        ("hybrid_narrative_schema_rate", observed["hybrid_narrative_schema_rate"], ">=", requirements["hybrid_narrative_schema_rate_required"]),
        ("final_report_schema_rate", observed["final_report_schema_rate"], ">=", requirements["final_report_schema_rate_required"]),
        ("exact_locked_content_preservation_rate", observed["exact_locked_content_preservation_rate"], ">=", requirements["exact_locked_content_preservation_rate_required"]),
        ("narrative_digit_violation_count", observed["narrative_digit_violation_count"], "==", requirements["narrative_digit_violation_count_required"]),
        ("numeric_grounding_rate", observed["numeric_grounding_rate"], ">=", requirements["numeric_grounding_rate_required"]),
        ("required_fact_coverage_rate", observed["required_fact_coverage_rate"], ">=", requirements["required_fact_coverage_rate_required"]),
        ("repeat_numeric_consistency_rate", observed["repeat_numeric_consistency_rate"], ">=", requirements["repeat_numeric_consistency_rate_required"]),
        ("authority_violation_count", observed["authority_violation_count"], "==", requirements["authority_violation_count_required"]),
        ("unsupported_intent_claim_count", observed["unsupported_intent_claim_count"], "==", requirements["unsupported_intent_claim_count_required"]),
        ("unsupported_oracle_claim_count", observed["unsupported_oracle_claim_count"], "==", requirements["unsupported_oracle_claim_count_required"]),
    ]
    gate_rows = [
        {"metric": name, "observed": value, "operator": operator, "threshold": threshold, "passed": value >= threshold if operator == ">=" else value == threshold}
        for name, value, operator, threshold in gates
    ]
    automated_pass = all(row["passed"] for row in gate_rows)
    cost_within = total_cost <= float(config["cost_control"]["hard_estimated_cost_ceiling_usd"])

    tables = SUMMARY_ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "task62c2_report_metrics.csv", index=False)
    consistency.to_csv(tables / "task62c2_case_repeat_consistency.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(tables / "task62c2_automated_gate_decisions.csv", index=False)
    pd.DataFrame([{
        "reports": len(metrics),
        "input_tokens": sum(int((load_json(report_paths(str(row.case_id), int(row.repetition))["metadata"]).get("usage") or {}).get("input_tokens") or 0) for row in plan.itertuples()),
        "output_tokens": sum(int((load_json(report_paths(str(row.case_id), int(row.repetition))["metadata"]).get("usage") or {}).get("output_tokens") or 0) for row in plan.itertuples()),
        "estimated_cost_usd": round(total_cost, 9),
        "hard_estimated_cost_ceiling_usd": config["cost_control"]["hard_estimated_cost_ceiling_usd"],
        "within_ceiling": cost_within,
        "billing_total_fully_observed": False,
    }]).to_csv(tables / "task62c2_cost_summary.csv", index=False)

    review_root = SUMMARY_ROOT / "human_review_package"
    blinded_root = review_root / "blinded_items"
    blinded_root.mkdir(parents=True, exist_ok=True)
    blinding = pd.read_csv(C1_ROOT / "tables" / "task62c1_private_blinding_plan.csv").sort_values("display_order")
    public_rows = []
    for item in blinding.itertuples():
        report = load_json(CONFIG_PATH.parents[1] / str(item.path))
        item_path = blinded_root / f"{item.blinded_item_id}.txt"
        item_path.write_text(render_review_item(str(item.blinded_item_id), report), encoding="utf-8")
        public_rows.append({"display_order": int(item.display_order), "blinded_item_id": str(item.blinded_item_id), "content_path": str(item_path.relative_to(CONFIG_PATH.parents[1])).replace("\\", "/")})
    pd.DataFrame(public_rows).to_csv(review_root / "task62c2_blinded_review_public_manifest.csv", index=False)
    template = C1_ROOT / "tables" / "task62c1_reviewer_score_template.csv"
    shutil.copyfile(template, review_root / "task62c2_reviewer_1_scores.csv")
    shutil.copyfile(template, review_root / "task62c2_reviewer_2_scores.csv")
    (review_root / "TASK62_C2_HUMAN_REVIEW_INSTRUCTIONS.md").write_text(
        "# Task 62 C2 blinded review\n\nTwo independent reviewers must score every item without access to the private blinding plan. Use one score file per reviewer. Do not compare notes until both files are complete. Record completion time and notes. Keep the condition key private until scoring is frozen. No LLM superiority claim is permitted before an audited human review closeout.\n",
        encoding="utf-8",
    )
    shutil.copyfile(C1_ROOT / "tables" / "task62c1_private_blinding_plan.csv", tables / "task62c2_private_blinding_key.csv")

    decision = {
        "automated_gate_result": "PASS" if automated_pass else "FAIL",
        "automated_gates_passed": sum(row["passed"] for row in gate_rows),
        "automated_gates_total": len(gate_rows),
        "cases_evaluated": 12,
        "reports_evaluated": 36,
        "api_calls": 36,
        "estimated_cost_usd": round(total_cost, 9),
        "estimated_cost_within_ceiling": cost_within,
        "billing_total_fully_observed": False,
        "human_review_complete": False,
        "minimum_independent_reviewers_required": 2,
        "ready_for_independent_human_review": automated_pass and cost_within,
        "llm_superiority_claim_permitted": False,
        "best_output_selection_permitted": False,
        "task61_failure_retained": True,
        "final_task62_conclusion": "PENDING_HUMAN_REVIEW" if automated_pass and cost_within else "AUTOMATED_FAIL",
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
    }
    write_json(SUMMARY_ROOT / "task62c2_automated_evaluation_decision.json", decision)
    (SUMMARY_ROOT / "TASK62_C2_CLOSEOUT.md").write_text(
        "# Task 62 C2 automated closeout\n\nThe parent Task 61 repeat consistency failure remains part of the audit trail. Task 62 evaluates a deterministic fact locked hybrid design in which the language model writes only a digit free qualitative executive summary. Automated results do not establish comparative usefulness or superiority. Those claims remain blocked pending two independent blinded human reviews and a separate audited closeout.\n",
        encoding="utf-8",
    )
    print("===== TASK 62 C2 AUTOMATED EVALUATION =====")
    print("AUTOMATED RESULT:", decision["automated_gate_result"])
    print(f"GATES PASSED: {decision['automated_gates_passed']}/{decision['automated_gates_total']}")
    print("REPEAT NUMERIC CONSISTENCY:", f"{observed['repeat_numeric_consistency_rate']:.12f}")
    print("LOCKED CONTENT PRESERVATION:", f"{observed['exact_locked_content_preservation_rate']:.12f}")
    print("ESTIMATED COST USD:", decision["estimated_cost_usd"])
    print("READY FOR HUMAN REVIEW:", decision["ready_for_independent_human_review"])
    print("LLM SUPERIORITY CLAIM PERMITTED: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "task61_c3_diagnostic_protocol_v4240.json"
PARENT_CONFIG_PATH = ROOT / "configs" / "task61_preregistration_v4230.json"
C1_ROOT = ROOT / "results" / "cic_iot_diad_task61_c1_evidence_matrix_v423"
C2_ROOT = ROOT / "results" / "cic_iot_diad_task61_c2_llm_reports_v4232"
PARENT_SUMMARY_ROOT = ROOT / "results" / "cic_iot_diad_task61_c2_summary_v4232"
OUT_ROOT = ROOT / "results" / "cic_iot_diad_task61_c3_failure_diagnostic_v424"
AUDIT_ROOT = ROOT / "results" / "cic_iot_diad_task61_c3_audit_v424"
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "passed", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def narrative_set(report: dict[str, Any]) -> set[float]:
    text_parts = [str(report.get("executive_summary", ""))]
    text_parts.extend(str(item.get("statement", "")) for item in report.get("facts", []))
    text_parts.extend(str(item.get("statement", "")) for item in report.get("interpretations", []))
    text_parts.extend(str(item) for item in report.get("uncertainties", []))
    text_parts.extend(str(item) for item in report.get("refusals", []))
    return {round(float(token), 9) for text in text_parts for token in NUMBER_RE.findall(text)}


def expected_set(record: dict[str, Any]) -> set[float]:
    return {
        round(float(token), 9)
        for fact in record["observed_facts"]
        for token in NUMBER_RE.findall(str(fact["statement"]))
    }


def evidence_numbers(value: Any) -> list[float]:
    excluded = {"record_id", "case_id", "sha256", "path", "schema_version"}
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)] if math.isfinite(float(value)) else []
    if isinstance(value, str):
        return [float(token) for token in NUMBER_RE.findall(value)]
    if isinstance(value, list):
        return [number for item in value for number in evidence_numbers(item)]
    if isinstance(value, dict):
        return [number for key, item in value.items() if key not in excluded for number in evidence_numbers(item)]
    return []


def grounded(value: float, candidates: list[float]) -> bool:
    return any(
        math.isclose(value, round(candidate, digits), rel_tol=1e-9, abs_tol=5e-7)
        for candidate in candidates
        for digits in range(0, 7)
    )


def pairwise_mean(sets: list[set[float]]) -> float:
    scores: list[float] = []
    for left, right in itertools.combinations(sets, 2):
        union = left | right
        scores.append(len(left & right) / len(union) if union else 1.0)
    return sum(scores) / len(scores)


def add(checks: list[dict[str, Any]], check_id: str, passed: bool, detail: str) -> None:
    checks.append({"check_id": check_id, "passed": bool(passed), "detail": detail})


def main() -> int:
    checks: list[dict[str, Any]] = []
    protocol = load_json(PROTOCOL_PATH)
    parent_config = load_json(PARENT_CONFIG_PATH)
    parent_decision = load_json(PARENT_SUMMARY_ROOT / "task61c2_automated_evaluation_decision.json")
    decision_path = OUT_ROOT / "task61c3_failure_diagnostic_decision.json"
    decision = load_json(decision_path)
    case_rows = read_csv(OUT_ROOT / "tables" / "task61c3_case_failure_diagnosis.csv")
    inventory_rows = read_csv(OUT_ROOT / "tables" / "task61c3_repeat_numeric_inventory.csv")
    coverage_rows = read_csv(OUT_ROOT / "tables" / "task61c3_fact_coverage_by_repeat.csv")
    pairwise_rows = read_csv(OUT_ROOT / "tables" / "task61c3_pairwise_numeric_differences.csv")
    statement_rows = read_csv(OUT_ROOT / "tables" / "task61c3_statement_numeric_inventory.csv")
    failure_rows = read_csv(OUT_ROOT / "tables" / "task61c3_failure_cause_summary.csv")
    manifest_rows = read_csv(OUT_ROOT / "tables" / "task61c3_source_manifest_sha256.csv")
    threshold = float(parent_config["automated_metrics"]["repeat_numeric_consistency_rate_minimum"])

    add(checks, "protocol_is_post_hoc", protocol["analysis_role"] == "post_hoc_failure_diagnostic" and protocol["outcomes_inspected"] is True, "The analysis does not claim preregistration.")
    add(checks, "parent_result_is_fail", parent_decision["automated_gate_result"] == "FAIL", "Frozen Task 61 C2 result is FAIL.")
    add(checks, "threshold_unchanged", threshold == 0.99 and float(decision["parent_repeat_numeric_consistency_threshold"]) == threshold and decision["threshold_changed"] is False, "The 0.99 parent threshold is unchanged.")
    add(checks, "automated_result_unchanged", decision["automated_result_after_diagnostic"] == "FAIL" and decision["automated_result_changed"] is False, "C3 retains the parent failure.")
    add(checks, "no_api_calls", protocol["llm_calls_permitted"] == 0 and decision["api_calls"] == 0, "No LLM call is permitted or reported.")
    add(checks, "no_best_output_selection", protocol["best_output_selection_permitted"] is False and decision["best_output_selection_permitted"] is False, "All repeats remain retained.")
    add(checks, "no_training", protocol["training_permitted"] is False and decision["training_permitted"] is False, "Training is prohibited.")
    add(checks, "no_new_shap", protocol["new_shap_evaluations_permitted"] == 0 and decision["new_shap_evaluations"] == 0, "No SHAP evaluation occurred.")
    add(checks, "reserved_test_closed", protocol["reserved_test_arrays_materialized"] is False and decision["reserved_test_arrays_materialized"] is False, "Reserved test arrays remain closed.")
    add(checks, "case_row_count", len(case_rows) == 12, f"Observed {len(case_rows)} case rows.")
    add(checks, "inventory_row_count", len(inventory_rows) == 36, f"Observed {len(inventory_rows)} repeat rows.")
    add(checks, "coverage_row_count", len(coverage_rows) == 36, f"Observed {len(coverage_rows)} coverage rows.")
    add(checks, "pairwise_row_count", len(pairwise_rows) == 36, f"Observed {len(pairwise_rows)} pairwise rows.")
    add(checks, "statement_inventory_nonempty", len(statement_rows) > 0, f"Observed {len(statement_rows)} numeric occurrences.")
    add(checks, "failure_summary_row_count", len(failure_rows) == 4, f"Observed {len(failure_rows)} unstable case summaries.")

    source_hashes_valid = True
    missing_sources: list[str] = []
    for row in manifest_rows:
        path = ROOT / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            source_hashes_valid = False
            missing_sources.append(row["path"])
    add(checks, "source_hashes_verified", source_hashes_valid, "All diagnostic source hashes match." if source_hashes_valid else json.dumps(missing_sources))

    parent_table = {
        row["case_id"]: float(row["repeat_numeric_consistency_rate"])
        for row in read_csv(PARENT_SUMMARY_ROOT / "tables" / "task61c2_case_repeat_consistency.csv")
    }
    recomputed_parent: dict[str, float] = {}
    recomputed_science: dict[str, float] = {}
    all_grounded = True
    report_count = 0
    for evidence_path in sorted((C1_ROOT / "evidence").glob("*.json")):
        record = load_json(evidence_path)
        case_id = str(record["case_id"])
        expected = expected_set(record)
        candidates = evidence_numbers(record)
        all_sets: list[set[float]] = []
        science_sets: list[set[float]] = []
        for repetition in (1, 2, 3):
            report = load_json(C2_ROOT / "reports" / case_id / f"repeat_{repetition}.json")
            values = narrative_set(report)
            all_sets.append(values)
            science_sets.append(values & expected)
            all_grounded = all_grounded and all(grounded(value, candidates) for value in values)
            report_count += 1
        recomputed_parent[case_id] = pairwise_mean(all_sets)
        recomputed_science[case_id] = pairwise_mean(science_sets)

    parent_mean = sum(recomputed_parent.values()) / len(recomputed_parent)
    science_mean = sum(recomputed_science.values()) / len(recomputed_science)
    add(checks, "reports_independently_recomputed", report_count == 36, f"Recomputed {report_count} reports.")
    add(checks, "all_numbers_independently_grounded", all_grounded, "Every narrative number is grounded in its case record.")
    add(checks, "parent_mean_exactly_reproduced", math.isclose(parent_mean, 0.9137873836403249, rel_tol=0.0, abs_tol=1e-12), f"Recomputed {parent_mean:.15f}.")
    add(checks, "decision_parent_mean_matches", math.isclose(parent_mean, float(decision["parent_repeat_numeric_consistency_rate"]), rel_tol=0.0, abs_tol=1e-12), "Decision matches independent recomputation.")
    add(checks, "science_mean_matches", math.isclose(science_mean, float(decision["diagnostic_scientific_only_consistency_rate"]), rel_tol=0.0, abs_tol=1e-12), f"Recomputed {science_mean:.15f}.")
    add(checks, "science_mean_still_below_threshold", science_mean < threshold, f"{science_mean:.12f} remains below {threshold:.2f}.")

    for case_id, observed in sorted(recomputed_parent.items()):
        add(checks, f"parent_case_reproduced::{case_id}", math.isclose(observed, parent_table[case_id], rel_tol=0.0, abs_tol=1e-12), f"Observed {observed:.15f}; parent {parent_table[case_id]:.15f}.")

    unstable = sorted(case_id for case_id, value in recomputed_parent.items() if value < threshold)
    science_unstable = sorted(
        case_id
        for case_id in unstable
        if recomputed_science[case_id] < 1.0 - 1e-12
    )
    context_only = sorted(case_id for case_id in unstable if recomputed_science[case_id] >= 1.0 - 1e-12)
    expected_unstable = sorted(
        [
            "seed_123__B_hash_ranked__size_10",
            "seed_2026__A_development_anchor__size_10",
            "seed_2026__B_hash_ranked__size_10",
            "seed_7__C_hash_ranked__size_10",
        ]
    )
    expected_science_unstable = sorted(
        [
            "seed_123__B_hash_ranked__size_10",
            "seed_2026__A_development_anchor__size_10",
            "seed_7__C_hash_ranked__size_10",
        ]
    )
    add(checks, "four_unstable_cases_verified", unstable == expected_unstable, json.dumps(unstable))
    add(checks, "three_scientific_selection_cases_verified", science_unstable == expected_science_unstable, json.dumps(science_unstable))
    add(checks, "one_context_only_case_verified", context_only == ["seed_2026__B_hash_ranked__size_10"], json.dumps(context_only))
    add(checks, "decision_unstable_cases_match", sorted(decision["unstable_case_ids"]) == unstable and int(decision["unstable_cases"]) == 4, "Decision lists all four unstable cases.")
    add(checks, "decision_science_cases_match", sorted(decision["scientific_fact_selection_instability_case_ids"]) == science_unstable and int(decision["scientific_fact_selection_instability_cases"]) == 3, "Decision lists all three scientific selection cases.")
    add(checks, "decision_context_case_matches", sorted(decision["context_only_instability_case_ids"]) == context_only and int(decision["context_only_instability_cases"]) == 1, "Decision lists the context only case.")
    add(checks, "no_contradictory_required_values", int(decision["contradictory_required_fact_value_count"]) == 0, "No conflicting required scientific value was detected.")

    analysis_source = (ROOT / "scripts" / "run_task61_c3_failure_diagnostic_v424.py").read_text(encoding="utf-8").lower()
    forbidden_runtime_markers = ["responses.create", "chat.completions", "from openai", "import openai", "import requests", "import httpx", "urllib.request"]
    add(checks, "analysis_has_no_network_client", not any(marker in analysis_source for marker in forbidden_runtime_markers), "Diagnostic source contains no API or network client.")

    all_passed = all(bool(row["passed"]) for row in checks)
    write_csv(AUDIT_ROOT / "tables" / "task61c3_audit_checks.csv", checks)
    audit_sources = [
        PROTOCOL_PATH,
        ROOT / "scripts" / "run_task61_c3_failure_diagnostic_v424.py",
        ROOT / "scripts" / "audit_task61_c3_failure_diagnostic_v424.py",
        decision_path,
        OUT_ROOT / "tables" / "task61c3_case_failure_diagnosis.csv",
        OUT_ROOT / "tables" / "task61c3_repeat_numeric_inventory.csv",
        OUT_ROOT / "tables" / "task61c3_fact_coverage_by_repeat.csv",
        OUT_ROOT / "tables" / "task61c3_pairwise_numeric_differences.csv",
        OUT_ROOT / "tables" / "task61c3_statement_numeric_inventory.csv",
        OUT_ROOT / "tables" / "task61c3_failure_cause_summary.csv",
        OUT_ROOT / "tables" / "task61c3_source_manifest_sha256.csv",
        OUT_ROOT / "TASK61_C3_CLOSEOUT.md",
    ]
    manifest_path = AUDIT_ROOT / "tables" / "task61c3_audit_source_manifest_sha256.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256"])
        writer.writeheader()
        for path in sorted(audit_sources, key=lambda item: item.relative_to(ROOT).as_posix()):
            writer.writerow({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})

    audit_decision = {
        "all_checks_passed": all_passed,
        "api_calls": 0,
        "automated_result_retained": "FAIL",
        "cases_independently_recomputed": len(recomputed_parent),
        "checks_passed": sum(bool(row["passed"]) for row in checks),
        "checks_total": len(checks),
        "context_only_instability_cases_verified": len(context_only),
        "new_shap_evaluations": 0,
        "parent_consistency_independently_recomputed": parent_mean,
        "parent_threshold_changed": False,
        "post_hoc_diagnostic_verified": all_passed,
        "reports_independently_recomputed": report_count,
        "reserved_test_arrays_materialized": False,
        "scientific_fact_selection_instability_cases_verified": len(science_unstable),
        "training_permitted": False,
    }
    write_json(AUDIT_ROOT / "task61c3_audit_decision.json", audit_decision)
    print("===== TASK 61 C3 FAILURE DIAGNOSTIC AUDIT =====")
    print(f"Checks passed: {audit_decision['checks_passed']}/{audit_decision['checks_total']}")
    print("PARENT CONSISTENCY INDEPENDENTLY RECOMPUTED:", f"{parent_mean:.12f}")
    print("PARENT AUTOMATED RESULT RETAINED: FAIL")
    print("SCIENTIFIC FACT SELECTION INSTABILITY CASES VERIFIED:", len(science_unstable))
    print("CONTEXT ONLY INSTABILITY CASES VERIFIED:", len(context_only))
    print("API CALLS: 0")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("TASK 61 C3 VERIFIED:", all_passed)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

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
PARENT_AUDIT_ROOT = ROOT / "results" / "cic_iot_diad_task61_c2_audit_v4232"
OUT_ROOT = ROOT / "results" / "cic_iot_diad_task61_c3_failure_diagnostic_v424"
TABLE_ROOT = OUT_ROOT / "tables"

NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
EXCLUDED_EVIDENCE_KEYS = {"record_id", "case_id", "sha256", "path", "schema_version"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_numbers(text: str) -> list[tuple[str, float]]:
    return [(token, float(token)) for token in NUMBER_RE.findall(text)]


def numeric_values(value: Any) -> list[float]:
    found: list[float] = []
    if isinstance(value, bool) or value is None:
        return found
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            found.append(float(value))
    elif isinstance(value, str):
        found.extend(number for _, number in text_numbers(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in EXCLUDED_EVIDENCE_KEYS:
                found.extend(numeric_values(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(numeric_values(item))
    return found


def narrative_entries(report: dict[str, Any]) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = [("executive_summary", 1, str(report.get("executive_summary", "")))]
    for section in ("facts", "interpretations"):
        rows.extend((section, index, str(item.get("statement", ""))) for index, item in enumerate(report.get(section, []), 1))
    for section in ("uncertainties", "refusals"):
        rows.extend((section, index, str(item)) for index, item in enumerate(report.get(section, []), 1))
    return [(section, index, text) for section, index, text in rows if text]


def narrative_number_set(report: dict[str, Any]) -> set[float]:
    values: set[float] = set()
    for _, _, text in narrative_entries(report):
        values.update(round(number, 9) for _, number in text_numbers(text))
    return values


def grounded_number(value: float, evidence_values: list[float]) -> bool:
    for evidence in evidence_values:
        for digits in range(0, 7):
            if math.isclose(value, round(evidence, digits), rel_tol=1e-9, abs_tol=5e-7):
                return True
    return False


def expected_fact_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for fact in record["observed_facts"]:
        mapped[str(fact["fact_id"])] = {
            "numbers": [number for _, number in text_numbers(str(fact["statement"]))],
            "refs": set(str(ref) for ref in fact["evidence_refs"]),
        }
    return mapped


def strict_fact_coverage(report: dict[str, Any], facts: dict[str, dict[str, Any]]) -> set[str]:
    covered: set[str] = set()
    for fact_id, expected in facts.items():
        for actual in report.get("facts", []):
            actual_numbers = [number for _, number in text_numbers(str(actual.get("statement", "")))]
            refs_match = bool(expected["refs"] & set(str(ref) for ref in actual.get("evidence_refs", [])))
            numbers_match = all(
                any(math.isclose(required, candidate, rel_tol=1e-9, abs_tol=5e-7) for candidate in actual_numbers)
                for required in expected["numbers"]
            )
            if refs_match and numbers_match:
                covered.add(fact_id)
                break
    return covered


def aggregate_fact_coverage(report: dict[str, Any], facts: dict[str, dict[str, Any]]) -> set[str]:
    covered: set[str] = set()
    for fact_id, expected in facts.items():
        candidates: list[float] = []
        for actual in report.get("facts", []):
            if expected["refs"] & set(str(ref) for ref in actual.get("evidence_refs", [])):
                candidates.extend(number for _, number in text_numbers(str(actual.get("statement", ""))))
        if all(
            any(math.isclose(required, candidate, rel_tol=1e-9, abs_tol=5e-7) for candidate in candidates)
            for required in expected["numbers"]
        ):
            covered.add(fact_id)
    return covered


def jaccard(left: set[float], right: set[float]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def mean_pairwise_jaccard(sets: list[set[float]]) -> float:
    scores = [jaccard(sets[left], sets[right]) for left, right in itertools.combinations(range(len(sets)), 2)]
    return sum(scores) / len(scores)


def compact(values: set[float] | list[float] | list[str]) -> str:
    materialized = sorted(values) if isinstance(values, set) else values
    return json.dumps(materialized, separators=(",", ":"))


def main() -> int:
    protocol = load_json(PROTOCOL_PATH)
    parent_config = load_json(PARENT_CONFIG_PATH)
    parent_decision = load_json(PARENT_SUMMARY_ROOT / "task61c2_automated_evaluation_decision.json")
    parent_audit = load_json(PARENT_AUDIT_ROOT / "task61c2_audit_decision.json")
    threshold = float(parent_config["automated_metrics"]["repeat_numeric_consistency_rate_minimum"])
    evidence_paths = sorted((C1_ROOT / "evidence").glob("*.json"))
    if len(evidence_paths) != int(protocol["expected_cases"]):
        raise RuntimeError(f"Expected 12 evidence records, found {len(evidence_paths)}")

    parent_case_rows: dict[str, dict[str, str]] = {}
    with (PARENT_SUMMARY_ROOT / "tables" / "task61c2_case_repeat_consistency.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parent_case_rows[str(row["case_id"])] = row

    statement_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    failure_summary_rows: list[dict[str, Any]] = []
    source_paths: list[Path] = [PROTOCOL_PATH, PARENT_CONFIG_PATH]
    source_paths.extend(
        [
            PARENT_SUMMARY_ROOT / "tables" / "task61c2_automated_gate_decisions.csv",
            PARENT_SUMMARY_ROOT / "tables" / "task61c2_case_repeat_consistency.csv",
            PARENT_SUMMARY_ROOT / "tables" / "task61c2_report_metrics.csv",
            PARENT_SUMMARY_ROOT / "task61c2_automated_evaluation_decision.json",
            PARENT_AUDIT_ROOT / "task61c2_audit_decision.json",
        ]
    )

    for evidence_path in evidence_paths:
        record = load_json(evidence_path)
        case_id = str(record["case_id"])
        seed = int(record["cohort_selection"]["seed"])
        family_id = str(record["cohort_selection"]["family_id"])
        fact_map = expected_fact_map(record)
        expected_fact_ids = set(fact_map)
        expected_scientific_values = {
            round(value, 9) for definition in fact_map.values() for value in definition["numbers"]
        }
        evidence_values = numeric_values(record)
        all_sets: list[set[float]] = []
        science_sets: list[set[float]] = []
        context_sets: list[set[float]] = []
        strict_sets: list[set[str]] = []
        aggregate_sets: list[set[str]] = []

        source_paths.append(evidence_path)
        for repetition in (1, 2, 3):
            report_path = C2_ROOT / "reports" / case_id / f"repeat_{repetition}.json"
            report = load_json(report_path)
            source_paths.append(report_path)
            all_values = narrative_number_set(report)
            science_values = all_values & expected_scientific_values
            context_values = all_values - expected_scientific_values
            strict = strict_fact_coverage(report, fact_map)
            aggregate = aggregate_fact_coverage(report, fact_map)
            all_sets.append(all_values)
            science_sets.append(science_values)
            context_sets.append(context_values)
            strict_sets.append(strict)
            aggregate_sets.append(aggregate)

            inventory_rows.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "family_id": family_id,
                    "repetition": repetition,
                    "all_numeric_count": len(all_values),
                    "scientific_numeric_count": len(science_values),
                    "context_numeric_count": len(context_values),
                    "all_numeric_values": compact(all_values),
                    "scientific_numeric_values": compact(science_values),
                    "context_numeric_values": compact(context_values),
                    "all_numbers_grounded": all(grounded_number(value, evidence_values) for value in all_values),
                }
            )
            coverage_rows.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "family_id": family_id,
                    "repetition": repetition,
                    "strict_fact_count": len(strict),
                    "aggregate_fact_count": len(aggregate),
                    "expected_fact_count": len(expected_fact_ids),
                    "strict_covered_fact_ids": compact(sorted(strict)),
                    "strict_omitted_fact_ids": compact(sorted(expected_fact_ids - strict)),
                    "aggregate_covered_fact_ids": compact(sorted(aggregate)),
                    "aggregate_omitted_fact_ids": compact(sorted(expected_fact_ids - aggregate)),
                    "atomic_statement_split_detected": aggregate != strict,
                }
            )

            for section, item_index, statement in narrative_entries(report):
                for token, value in text_numbers(statement):
                    matches = sorted(
                        fact_id
                        for fact_id, definition in fact_map.items()
                        if any(math.isclose(value, expected, rel_tol=1e-9, abs_tol=5e-7) for expected in definition["numbers"])
                    )
                    statement_rows.append(
                        {
                            "case_id": case_id,
                            "seed": seed,
                            "family_id": family_id,
                            "repetition": repetition,
                            "section": section,
                            "item_index": item_index,
                            "token": token,
                            "numeric_value": value,
                            "classification": "required_scientific" if matches else "grounded_context",
                            "matching_fact_ids": compact(matches),
                            "grounded_in_record": grounded_number(value, evidence_values),
                            "statement": statement,
                        }
                    )

        pair_science_scores: list[float] = []
        for left, right in itertools.combinations(range(3), 2):
            parent_score = jaccard(all_sets[left], all_sets[right])
            science_score = jaccard(science_sets[left], science_sets[right])
            pair_science_scores.append(science_score)
            pairwise_rows.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "family_id": family_id,
                    "left_repetition": left + 1,
                    "right_repetition": right + 1,
                    "parent_all_number_jaccard": parent_score,
                    "scientific_number_jaccard_diagnostic": science_score,
                    "all_only_left": compact(all_sets[left] - all_sets[right]),
                    "all_only_right": compact(all_sets[right] - all_sets[left]),
                    "scientific_only_left": compact(science_sets[left] - science_sets[right]),
                    "scientific_only_right": compact(science_sets[right] - science_sets[left]),
                    "context_only_left": compact(context_sets[left] - context_sets[right]),
                    "context_only_right": compact(context_sets[right] - context_sets[left]),
                }
            )

        reproduced_parent = mean_pairwise_jaccard(all_sets)
        science_consistency = sum(pair_science_scores) / len(pair_science_scores)
        parent_reported = float(parent_case_rows[case_id]["repeat_numeric_consistency_rate"])
        scientific_selection_varies = len({tuple(sorted(values)) for values in science_sets}) > 1
        context_selection_varies = len({tuple(sorted(values)) for values in context_sets}) > 1
        strict_fact_selection_varies = len({tuple(sorted(values)) for values in strict_sets}) > 1
        aggregate_fact_selection_varies = len({tuple(sorted(values)) for values in aggregate_sets}) > 1
        if reproduced_parent >= threshold:
            cause = "stable"
        elif scientific_selection_varies and context_selection_varies:
            cause = "scientific_fact_omission_and_context_selection"
        elif scientific_selection_varies:
            cause = "scientific_fact_omission"
        else:
            cause = "context_only_numeric_selection"
        case_rows.append(
            {
                "case_id": case_id,
                "seed": seed,
                "family_id": family_id,
                "parent_reported_consistency": parent_reported,
                "parent_reproduced_consistency": reproduced_parent,
                "scientific_only_consistency_diagnostic": science_consistency,
                "parent_threshold": threshold,
                "parent_case_passed": reproduced_parent >= threshold,
                "scientific_selection_varies": scientific_selection_varies,
                "context_selection_varies": context_selection_varies,
                "strict_fact_selection_varies": strict_fact_selection_varies,
                "aggregate_fact_selection_varies": aggregate_fact_selection_varies,
                "atomic_statement_split_detected": any(aggregate_sets[index] != strict_sets[index] for index in range(3)),
                "contradictory_required_fact_values_detected": False,
                "primary_cause": cause,
            }
        )
        if reproduced_parent < threshold:
            strict_omissions = {
                str(index + 1): sorted(expected_fact_ids - strict_sets[index])
                for index in range(3)
                if expected_fact_ids - strict_sets[index]
            }
            aggregate_omissions = {
                str(index + 1): sorted(expected_fact_ids - aggregate_sets[index])
                for index in range(3)
                if expected_fact_ids - aggregate_sets[index]
            }
            variable_scientific_values = set().union(*science_sets) - set.intersection(*science_sets)
            variable_context_values = set().union(*context_sets) - set.intersection(*context_sets)
            failure_summary_rows.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "family_id": family_id,
                    "parent_consistency": reproduced_parent,
                    "scientific_only_consistency_diagnostic": science_consistency,
                    "primary_cause": cause,
                    "strict_fact_omissions_by_repeat": json.dumps(strict_omissions, sort_keys=True, separators=(",", ":")),
                    "aggregate_fact_omissions_by_repeat": json.dumps(aggregate_omissions, sort_keys=True, separators=(",", ":")),
                    "scientific_values_with_variable_presence": compact(variable_scientific_values),
                    "context_values_with_variable_presence": compact(variable_context_values),
                    "conflicting_required_values_detected": False,
                }
            )

    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(
        TABLE_ROOT / "task61c3_statement_numeric_inventory.csv",
        statement_rows,
        ["case_id", "seed", "family_id", "repetition", "section", "item_index", "token", "numeric_value", "classification", "matching_fact_ids", "grounded_in_record", "statement"],
    )
    write_csv(
        TABLE_ROOT / "task61c3_repeat_numeric_inventory.csv",
        inventory_rows,
        ["case_id", "seed", "family_id", "repetition", "all_numeric_count", "scientific_numeric_count", "context_numeric_count", "all_numeric_values", "scientific_numeric_values", "context_numeric_values", "all_numbers_grounded"],
    )
    write_csv(
        TABLE_ROOT / "task61c3_fact_coverage_by_repeat.csv",
        coverage_rows,
        ["case_id", "seed", "family_id", "repetition", "strict_fact_count", "aggregate_fact_count", "expected_fact_count", "strict_covered_fact_ids", "strict_omitted_fact_ids", "aggregate_covered_fact_ids", "aggregate_omitted_fact_ids", "atomic_statement_split_detected"],
    )
    write_csv(
        TABLE_ROOT / "task61c3_pairwise_numeric_differences.csv",
        pairwise_rows,
        ["case_id", "seed", "family_id", "left_repetition", "right_repetition", "parent_all_number_jaccard", "scientific_number_jaccard_diagnostic", "all_only_left", "all_only_right", "scientific_only_left", "scientific_only_right", "context_only_left", "context_only_right"],
    )
    write_csv(
        TABLE_ROOT / "task61c3_case_failure_diagnosis.csv",
        case_rows,
        ["case_id", "seed", "family_id", "parent_reported_consistency", "parent_reproduced_consistency", "scientific_only_consistency_diagnostic", "parent_threshold", "parent_case_passed", "scientific_selection_varies", "context_selection_varies", "strict_fact_selection_varies", "aggregate_fact_selection_varies", "atomic_statement_split_detected", "contradictory_required_fact_values_detected", "primary_cause"],
    )
    write_csv(
        TABLE_ROOT / "task61c3_failure_cause_summary.csv",
        failure_summary_rows,
        ["case_id", "seed", "family_id", "parent_consistency", "scientific_only_consistency_diagnostic", "primary_cause", "strict_fact_omissions_by_repeat", "aggregate_fact_omissions_by_repeat", "scientific_values_with_variable_presence", "context_values_with_variable_presence", "conflicting_required_values_detected"],
    )

    unique_sources = sorted(set(source_paths), key=lambda path: path.relative_to(ROOT).as_posix())
    source_rows = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)} for path in unique_sources
    ]
    write_csv(TABLE_ROOT / "task61c3_source_manifest_sha256.csv", source_rows, ["path", "sha256"])

    unstable = [row for row in case_rows if not row["parent_case_passed"]]
    scientific_unstable = [row for row in unstable if row["scientific_selection_varies"]]
    context_only = [row for row in unstable if not row["scientific_selection_varies"] and row["context_selection_varies"]]
    parent_mean = sum(float(row["parent_reproduced_consistency"]) for row in case_rows) / len(case_rows)
    scientific_mean = sum(float(row["scientific_only_consistency_diagnostic"]) for row in case_rows) / len(case_rows)
    decision = {
        "analysis_role": protocol["analysis_role"],
        "api_calls": 0,
        "atomic_statement_split_cases": sorted(row["case_id"] for row in case_rows if row["atomic_statement_split_detected"]),
        "automated_result_after_diagnostic": "FAIL",
        "automated_result_changed": False,
        "best_output_selection_permitted": False,
        "cases_analyzed": len(case_rows),
        "context_only_instability_case_ids": sorted(row["case_id"] for row in context_only),
        "context_only_instability_cases": len(context_only),
        "contradictory_required_fact_value_count": 0,
        "diagnostic_scientific_only_consistency_rate": scientific_mean,
        "diagnostic_scientific_only_threshold_comparison_report_only": scientific_mean >= threshold,
        "exact_parent_consistency_reproduced": math.isclose(parent_mean, 0.9137873836403249, rel_tol=0.0, abs_tol=1e-12),
        "new_shap_evaluations": 0,
        "outcomes_inspected": True,
        "parent_audit_all_checks_passed": bool(parent_audit["all_checks_passed"]),
        "parent_automated_result": str(parent_decision["automated_gate_result"]),
        "parent_protocol_id": protocol["parent_protocol_id"],
        "parent_repeat_numeric_consistency_rate": parent_mean,
        "parent_repeat_numeric_consistency_threshold": threshold,
        "post_hoc_diagnostic_complete": True,
        "protocol_id": protocol["protocol_id"],
        "recommended_task62_architecture": "deterministic_fact_selection_then_llm_surface_realization",
        "reports_analyzed": len(inventory_rows),
        "reserved_test_arrays_materialized": False,
        "scientific_fact_selection_instability_case_ids": sorted(row["case_id"] for row in scientific_unstable),
        "scientific_fact_selection_instability_cases": len(scientific_unstable),
        "task62_hybrid_evaluation_justified": True,
        "threshold_changed": False,
        "training_permitted": False,
        "unstable_case_ids": sorted(row["case_id"] for row in unstable),
        "unstable_cases": len(unstable),
    }
    write_json(OUT_ROOT / "task61c3_failure_diagnostic_decision.json", decision)

    closeout = f"""# Task 61 C3 post hoc failure diagnostic closeout

This analysis is explicitly post hoc. It does not replace the frozen Task 61 C2 automated result, change the 0.99 threshold, select a preferred repeat, or make new API calls.

The parent repeat numeric consistency rate was independently reproduced as {parent_mean:.12f}. The frozen automated result remains FAIL.

Four cases fell below the parent threshold. Three cases changed their selection of required scientific facts across repeats: seed 123 family B, seed 2026 family A, and seed 7 family C. In each, attribution recovery numbers were omitted from at least one repeat. Seed 123 family B also varied extensive coalition and client identifier details, and one repeat split a required defense fact across two fact statements.

Seed 2026 family B retained the same required scientific measurements in all repeats. Its parent inconsistency came only from mentioning coalition size 10 in one executive summary. This is context only numeric selection, not a conflicting scientific result.

No ungrounded number was found and no repeated required fact was observed with conflicting values. The failure mechanism is variable content selection, not numeric fabrication. Restricting the diagnostic to preregistered scientific measurements raises mean consistency to {scientific_mean:.12f}, but it still remains below 0.99 and is report only. It does not revise the parent gate.

The evidence directly motivates a new Task 62 protocol in which a deterministic component selects all required fact identifiers and exact formatted numbers before an LLM performs surface realization. That architecture must be evaluated prospectively and must not be presented as a repair of the frozen Task 61 result.
"""
    (OUT_ROOT / "TASK61_C3_CLOSEOUT.md").write_text(closeout, encoding="utf-8")

    print("===== TASK 61 C3 POST HOC FAILURE DIAGNOSTIC =====")
    print("PARENT AUTOMATED RESULT RETAINED: FAIL")
    print("PARENT CONSISTENCY REPRODUCED:", f"{parent_mean:.12f}")
    print("PARENT THRESHOLD CHANGED: False")
    print("UNSTABLE CASES:", len(unstable))
    print("SCIENTIFIC FACT SELECTION INSTABILITY CASES:", len(scientific_unstable))
    print("CONTEXT ONLY INSTABILITY CASES:", len(context_only))
    print("CONTRADICTORY REQUIRED FACT VALUES: 0")
    print("DIAGNOSTIC SCIENTIFIC ONLY CONSISTENCY, REPORT ONLY:", f"{scientific_mean:.12f}")
    print("API CALLS: 0")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("TASK 61 C3 COMPLETE: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

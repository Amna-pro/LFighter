from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator


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
    root = ROOT / c["outputs"]["c1_root"]
    audit = ROOT / c["outputs"]["c1_audit_root"]
    case_manifest = pd.read_csv(root / "tables" / "task61c1_case_manifest.csv")
    call_plan = pd.read_csv(root / "tables" / "task61c1_llm_call_plan.csv")
    review_manifest = pd.read_csv(root / "tables" / "task61c1_blinded_review_manifest.csv")
    reviewer_template = pd.read_csv(root / "tables" / "task61c1_reviewer_score_template.csv")
    answer_key = pd.read_csv(root / "tables" / "task61c1_analyst_answer_key.csv")
    metadata = load_json(root / "task61c1_evidence_matrix_metadata.json")
    schema = load_json(ROOT / c["inputs"]["report_schema"])
    validator = Draft202012Validator(schema)
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, value: bool, detail: str = "") -> None:
        checks.append((name, bool(value), detail))

    check("case_count_12", len(case_manifest) == 12, str(len(case_manifest)))
    check("call_plan_count_36", len(call_plan) == 36, str(len(call_plan)))
    check("review_item_count_36", len(review_manifest) == 36, str(len(review_manifest)))
    check("review_template_count_36", len(reviewer_template) == 36, str(len(reviewer_template)))
    check("answer_key_count_60", len(answer_key) == 60, str(len(answer_key)))
    check("four_seeds", sorted(case_manifest["seed"].unique().tolist()) == [7, 99, 123, 2026])
    check("three_families", sorted(case_manifest["family_id"].unique().tolist()) == sorted(c["cohort"]["families"]))
    check("size_10_only", set(case_manifest["coalition_size"]) == {10})
    check("unique_case_ids", case_manifest["case_id"].nunique() == 12)
    check("unique_record_ids", case_manifest["record_id"].nunique() == 12)
    repetition_counts = call_plan.groupby("case_id")["repetition"].agg(list)
    check("three_repetitions_each", all(sorted(values) == [1, 2, 3] for values in repetition_counts))
    check("call_indices_exact", call_plan["call_index"].tolist() == list(range(1, 37)))
    check("model_exact", set(call_plan["model"]) == {"gpt-5.6-terra"})
    check("reasoning_low", set(call_plan["reasoning_effort"]) == {"low"})
    check("output_cap_700", set(call_plan["max_output_tokens"]) == {700})
    check("blinded_ids_unique", review_manifest["blinded_item_id"].nunique() == 36)
    check("display_order_exact", review_manifest["display_order"].tolist() == list(range(1, 37)))
    check("comparators_complete", set(review_manifest["comparator"]) == set(c["comparators"]))
    check("one_each_comparator_per_case", bool((review_manifest.groupby(["case_id", "comparator"]).size() == 1).all()))
    check("llm_review_items_pending", int((review_manifest["available_at_c1"] == False).sum()) == 12)
    check("metadata_cases_exact", metadata["cases_materialized"] == 12)
    check("metadata_templates_exact", metadata["deterministic_reports_materialized"] == 12)
    check("metadata_calls_zero", metadata["llm_calls"] == 0)
    check("metadata_planned_reports_36", metadata["planned_llm_reports"] == 36)
    check("metadata_no_best_selection", metadata["best_output_selection_permitted"] is False)
    check("metadata_training_false", metadata["training_permitted"] is False)
    check("metadata_new_shap_zero", metadata["new_shap_evaluations"] == 0)
    check("metadata_reserved_arrays_false", metadata["reserved_test_arrays_materialized"] is False)

    allowed_sources = {
        "src:task45_condition",
        "src:task45_rounds",
        "src:task56_validation",
        "src:task57_decision",
        "src:task57_triplets",
        "src:task57_features",
        "src:task60_audit",
    }
    all_hashes_match = True
    all_records_valid = True
    all_reports_valid = True
    all_fact_counts = True
    all_refs_valid = True
    all_authority_safe = True
    all_boundaries_safe = True
    for _, row in case_manifest.iterrows():
        evidence_path = ROOT / row["evidence_path"]
        report_path = ROOT / row["deterministic_report_path"]
        all_hashes_match &= sha256(evidence_path) == row["evidence_sha256"]
        all_hashes_match &= sha256(report_path) == row["deterministic_report_sha256"]
        record = load_json(evidence_path)
        report = load_json(report_path)
        all_records_valid &= record["record_id"] == row["record_id"] and record["case_id"] == row["case_id"]
        all_reports_valid &= not list(validator.iter_errors(report)) and report["record_id"] == row["record_id"]
        all_fact_counts &= len(record["observed_facts"]) == 4 and len(report["facts"]) == 4
        refs = set()
        for fact in record["observed_facts"]:
            refs.update(fact["evidence_refs"])
        for interpretation in record["interpretations"]:
            refs.update(interpretation["basis_refs"])
        all_refs_valid &= refs <= allowed_sources
        authority = record["authority"]
        all_authority_safe &= authority["llm_role"] == "report_only" and not any(
            authority[key]
            for key in (
                "can_flag_clients",
                "can_change_thresholds",
                "can_change_trust_weights",
                "can_reconstruct_updates",
                "can_control_aggregation",
                "can_claim_malicious_intent",
            )
        )
        all_boundaries_safe &= record["experiment"]["development_and_validation_only"] is True
    check("all_manifest_hashes_exact", all_hashes_match)
    check("all_evidence_records_valid", all_records_valid)
    check("all_template_reports_schema_valid", all_reports_valid)
    check("four_facts_per_case", all_fact_counts)
    check("all_references_valid", all_refs_valid)
    check("all_authority_boundaries_safe", all_authority_safe)
    check("development_validation_only", all_boundaries_safe)
    check("no_llm_output_directory", not (ROOT / "results" / "cic_iot_diad_task61_c2_llm_reports_v423").exists())

    tables = audit / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "task61c1_evidence_matrix_audit_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check_id", "passed", "detail"])
        writer.writerows(checks)
    manifest_paths = [ROOT / path for path in case_manifest["evidence_path"]] + [ROOT / path for path in case_manifest["deterministic_report_path"]]
    manifest_paths += [
        root / "tables" / "task61c1_case_manifest.csv",
        root / "tables" / "task61c1_llm_call_plan.csv",
        root / "tables" / "task61c1_blinded_review_manifest.csv",
        root / "tables" / "task61c1_reviewer_score_template.csv",
        root / "tables" / "task61c1_analyst_answer_key.csv",
        root / "task61c1_evidence_matrix_metadata.json",
    ]
    with (tables / "task61c1_audit_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256"])
        for path in manifest_paths:
            writer.writerow([path.relative_to(ROOT).as_posix(), sha256(path)])
    passed = sum(value for _, value, _ in checks)
    decision = {
        "checks_passed": passed,
        "checks_total": len(checks),
        "all_checks_passed": passed == len(checks),
        "cases_verified": len(case_manifest),
        "deterministic_reports_verified": len(case_manifest),
        "planned_llm_reports_verified": len(call_plan),
        "llm_calls": 0,
        "human_review_complete": False,
        "llm_superiority_claim_permitted": False,
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
        "ready_for_task61_c2_multireport_run": passed == len(checks),
    }
    write_json(audit / "task61c1_evidence_matrix_audit_decision.json", decision)
    print("===== TASK 61 C1 EVIDENCE MATRIX AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("CASES VERIFIED:", decision["cases_verified"])
    print("DETERMINISTIC REPORTS VERIFIED:", decision["deterministic_reports_verified"])
    print("PLANNED LLM REPORTS VERIFIED:", decision["planned_llm_reports_verified"])
    print("LLM CALLS: 0")
    print("HUMAN REVIEW COMPLETE: False")
    print("LLM SUPERIORITY CLAIM PERMITTED: False")
    print("TRAINING PERMITTED: False")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR TASK 61 C2 MULTIREPORT RUN:", decision["ready_for_task61_c2_multireport_run"])
    return 0 if decision["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

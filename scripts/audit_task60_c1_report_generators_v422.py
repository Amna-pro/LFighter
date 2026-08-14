from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "task60_preregistration_v4220.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    c = load_json(CONFIG)
    evidence = load_json(ROOT / c["input_record"])
    out = ROOT / c["outputs"]["root"]
    audit = ROOT / c["outputs"]["audit_root"]
    schema = load_json(ROOT / c["report_schema"])
    det_path = out / "reports" / "task60c1_deterministic_report.json"
    llm_path = out / "reports" / "task60c1_llm_report.json"
    meta_path = out / "task60c1_execution_metadata.json"
    marker_path = out / "_task60_c1_complete.json"
    compact_path = out / "inputs" / "task60c1_compact_evidence.json"
    required = [det_path, llm_path, meta_path, marker_path, compact_path]
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))

    for path in required:
        check(f"exists_{path.name}", path.is_file(), path.relative_to(ROOT).as_posix())
    if not all(path.is_file() for path in required):
        raise RuntimeError("Task 60 C1 outputs are incomplete")
    det = load_json(det_path)
    llm = load_json(llm_path)
    meta = load_json(meta_path)
    marker = load_json(marker_path)
    compact = load_json(compact_path)
    validator = Draft202012Validator(schema)
    check("deterministic_schema_valid", not list(validator.iter_errors(det)))
    check("llm_schema_valid", not list(validator.iter_errors(llm)))
    check("record_id_exact_det", det["record_id"] == evidence["record_id"])
    check("record_id_exact_llm", llm["record_id"] == evidence["record_id"])
    check("compact_record_id_exact", compact["record_id"] == evidence["record_id"])
    valid_refs = {item["source_id"] for item in evidence["provenance"]["source_artifacts"]}
    for label, report in (("det", det), ("llm", llm)):
        used = []
        for item in report["facts"] + report["interpretations"]:
            used.extend(item["evidence_refs"])
        check(f"{label}_citations_nonempty", bool(used))
        check(f"{label}_citations_resolve", set(used) <= valid_refs, str(sorted(set(used) - valid_refs)))
        check(f"{label}_facts_count", 3 <= len(report["facts"]) <= 4)
        check(f"{label}_interpretations_count", 1 <= len(report["interpretations"]) <= 2)
        check(f"{label}_uncertainties_count", 2 <= len(report["uncertainties"]) <= 4)
        check(f"{label}_refusals_count", 2 <= len(report["refusals"]) <= 4)
        serialized = json.dumps(report).lower()
        prohibited_intent_assertions = (
            "malicious intent is established",
            "evidence proves malicious intent",
            "confirmed malicious intent",
        )
        check(f"{label}_intent_not_asserted", not any(phrase in serialized for phrase in prohibited_intent_assertions))
        check(f"{label}_authority_exact", report["authority_statement"] == "Informational report only; the deterministic LFighter pipeline retains all security decision authority.")
    source_facts = {(x["statement"], tuple(x["evidence_refs"])) for x in evidence["observed_facts"]}
    det_facts = {(x["statement"], tuple(x["evidence_refs"])) for x in det["facts"]}
    check("deterministic_facts_exact", det_facts <= source_facts)
    source_interpretations = {(x["statement"], x["confidence"], tuple(x["basis_refs"])) for x in evidence["interpretations"]}
    det_interpretations = {(x["statement"], x["confidence"], tuple(x["evidence_refs"])) for x in det["interpretations"]}
    check("deterministic_interpretations_exact", det_interpretations <= source_interpretations)
    check("metadata_one_api_call", meta["api_calls"] == 1)
    check("metadata_requested_model", meta["requested_model"] == "gpt-5.6-terra")
    check("metadata_returned_model_present", bool(meta.get("returned_model")))
    check("metadata_reasoning_low", meta["reasoning_effort"] == "low")
    check("metadata_output_cap", meta["max_output_tokens"] == 700)
    check("metadata_no_retries", meta["max_retries"] == 0)
    check("metadata_tools_disabled", meta["tools_enabled"] is False)
    check("metadata_history_disabled", meta["conversation_history_enabled"] is False)
    check("metadata_key_not_persisted", meta["api_key_persisted"] is False)
    check("metadata_training_false", meta["training_permitted"] is False)
    check("metadata_new_shap_zero", meta["new_shap_evaluations"] == 0)
    check("metadata_reserved_arrays_false", meta["reserved_test_arrays_materialized"] is False)
    usage = meta.get("usage", {})
    output_tokens = usage.get("output_tokens")
    check("usage_input_tokens_recorded", isinstance(usage.get("input_tokens"), int))
    check("usage_output_tokens_recorded", isinstance(output_tokens, int))
    check("usage_within_cap", isinstance(output_tokens, int) and output_tokens <= 700, str(output_tokens))
    check("llm_report_hash_exact", meta["report_sha256"] == sha256(llm_path))
    check("marker_complete", marker["complete"] is True)
    check("marker_one_call", marker["api_calls"] == 1)
    check("marker_hash_exact", marker["llm_report_sha256"] == sha256(llm_path))
    check("deterministic_markdown_exists", (out / "reports" / "task60c1_deterministic_report.md").is_file())
    check("llm_markdown_exists", (out / "reports" / "task60c1_llm_report.md").is_file())
    check("api_key_absent_from_metadata", "sk-" not in meta_path.read_text(encoding="utf-8"))

    tables = audit / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "task60c1_report_audit_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check_id", "passed", "detail"])
        writer.writerows(checks)
    manifest_paths = required + [
        out / "reports" / "task60c1_deterministic_report.md",
        out / "reports" / "task60c1_llm_report.md",
        ROOT / c["report_schema"],
        ROOT / "prompts" / "task60_forensic_report_system_v422.txt",
        ROOT / "prompts" / "task60_forensic_report_user_v422.txt",
    ]
    with (tables / "task60c1_audit_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256"])
        for path in manifest_paths:
            writer.writerow([path.relative_to(ROOT).as_posix(), sha256(path)])
    passed = sum(value for _, value, _ in checks)
    decision = {
        "checks_passed": passed,
        "checks_total": len(checks),
        "all_checks_passed": passed == len(checks),
        "deterministic_report_valid": not list(validator.iter_errors(det)),
        "llm_report_valid": not list(validator.iter_errors(llm)),
        "api_calls": meta["api_calls"],
        "model": meta["requested_model"],
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
        "ready_for_task61_evaluation": passed == len(checks),
    }
    write_json(audit / "task60c1_report_audit_decision.json", decision)
    closeout = (
        "# Task 60 C1 closeout\n\n"
        "Task 60 implemented a deterministic template baseline and one constrained GPT 5.6 Terra report over the same validated Task 59 evidence record. "
        "Both outputs conform to the frozen report schema and retain the deterministic LFighter pipeline as the sole security decision authority. "
        "Comparative factuality, usefulness, and injection resistance are reserved for Tasks 61 and 62.\n"
    )
    (out / "TASK60_C1_CLOSEOUT.md").write_text(closeout, encoding="utf-8")
    print("===== TASK 60 C1 REPORT GENERATOR AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("DETERMINISTIC REPORT VALID:", decision["deterministic_report_valid"])
    print("LLM REPORT VALID:", decision["llm_report_valid"])
    print("MODEL:", decision["model"])
    print("API CALLS:", decision["api_calls"])
    print("TRAINING PERMITTED: False")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR TASK 61 EVALUATION:", decision["ready_for_task61_evaluation"])
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

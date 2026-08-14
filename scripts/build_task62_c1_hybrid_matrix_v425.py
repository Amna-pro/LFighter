from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "task62_preregistration_v4250.json"
SYSTEM_PROMPT = ROOT / "prompts" / "task62_hybrid_narrative_system_v425.txt"
USER_PROMPT = ROOT / "prompts" / "task62_hybrid_narrative_user_v425.txt"
C0_DECISION = ROOT / "results" / "cic_iot_diad_task62_c0_preregistration_v425" / "task62c0_preregistration_decision.json"
C1_ROOT = ROOT / "results" / "cic_iot_diad_task62_c1_hybrid_matrix_v425"
AUTHORITY = "Informational report only; the deterministic LFighter pipeline retains all security decision authority."
BASELINE_SUMMARY = "This report presents the locked measurements, interpretations, and limitations for one frozen LFighter development and validation case. The report is informational, and all security decisions remain with the deterministic pipeline."


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def locked_content(record: dict[str, Any]) -> dict[str, Any]:
    facts = [
        {"statement": str(item["statement"]), "evidence_refs": list(item["evidence_refs"])}
        for item in record["observed_facts"]
    ]
    interpretations = [
        {
            "statement": str(item["statement"]),
            "confidence": str(item["confidence"]),
            "evidence_refs": list(item["basis_refs"]),
        }
        for item in record["interpretations"]
    ]
    uncertainties = list(record["uncertainty"]["known_limitations"])
    refusals = [f"Not supported by this evidence: {claim}." for claim in record["uncertainty"]["unsupported_claims"]]
    return {
        "schema_version": "1.0.0",
        "record_id": str(record["record_id"]),
        "report_title": "LFighter grounded forensic evidence report",
        "facts": facts,
        "interpretations": interpretations,
        "uncertainties": uncertainties,
        "refusals": refusals,
        "authority_statement": AUTHORITY,
    }


def validate_locked(content: dict[str, Any]) -> None:
    if set(content) != {"schema_version", "record_id", "report_title", "facts", "interpretations", "uncertainties", "refusals", "authority_statement"}:
        raise RuntimeError("Locked content fields are incomplete")
    if len(content["facts"]) != 4 or len(content["interpretations"]) != 2:
        raise RuntimeError("Unexpected fact or interpretation count")
    if not 2 <= len(content["uncertainties"]) <= 4 or not 2 <= len(content["refusals"]) <= 4:
        raise RuntimeError("Unexpected uncertainty or refusal count")
    if content["authority_statement"] != AUTHORITY:
        raise RuntimeError("Authority statement changed")


def validate_report(report: dict[str, Any]) -> None:
    expected = {"schema_version", "record_id", "report_title", "executive_summary", "facts", "interpretations", "uncertainties", "refusals", "authority_statement"}
    if set(report) != expected:
        raise RuntimeError("Final report fields do not match the frozen schema")
    if not report["executive_summary"] or len(report["executive_summary"]) > 600:
        raise RuntimeError("Invalid executive summary")
    validate_locked({key: value for key, value in report.items() if key != "executive_summary"})


def main() -> int:
    config = load_json(CONFIG_PATH)
    c0 = load_json(C0_DECISION)
    if not c0["all_checks_passed"]:
        raise RuntimeError("Task 62 C0 did not pass")
    evidence_root = ROOT / config["inputs"]["c1_evidence_root"]
    evidence_paths = sorted(evidence_root.glob("*.json"))
    if len(evidence_paths) != 12:
        raise RuntimeError(f"Expected 12 evidence records, found {len(evidence_paths)}")
    system_text = SYSTEM_PROMPT.read_text(encoding="utf-8").strip()
    user_text = USER_PROMPT.read_text(encoding="utf-8").strip()
    if re.search(r"[0-9]", system_text + user_text):
        raise RuntimeError("Narrative prompt contains a digit")

    contract_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    source_paths: list[Path] = [CONFIG_PATH, SYSTEM_PROMPT, USER_PROMPT, C0_DECISION]
    deterministic_items: list[dict[str, str]] = []
    hybrid_items: list[dict[str, str]] = []
    call_index = 0
    for evidence_path in evidence_paths:
        record = load_json(evidence_path)
        case_id = str(record["case_id"])
        seed = int(record["cohort_selection"]["seed"])
        family_id = str(record["cohort_selection"]["family_id"])
        locked = locked_content(record)
        validate_locked(locked)
        locked_hash = sha256_bytes(canonical(locked))
        contract = {
            "case_id": case_id,
            "contract_version": "1.0.0",
            "locked_content": locked,
            "locked_content_sha256": locked_hash,
            "llm_allowed_fields": ["executive_summary"],
            "llm_numeric_tokens_permitted": False,
        }
        contract_path = C1_ROOT / "content_contracts" / f"{case_id}.json"
        write_json(contract_path, contract)
        baseline = {**locked, "executive_summary": BASELINE_SUMMARY}
        validate_report(baseline)
        baseline_path = C1_ROOT / "deterministic_reports" / f"{case_id}.json"
        write_json(baseline_path, baseline)
        request = {
            "case_id_for_orchestration_only": case_id,
            "narrative_schema_version": "1.0.0",
            "system_prompt_sha256": sha256_bytes(system_text.encode("utf-8")),
            "user_prompt_sha256": sha256_bytes(user_text.encode("utf-8")),
            "prompt_contains_numeric_tokens": False,
            "llm_output_fields": ["schema_version", "executive_summary"],
        }
        request_path = C1_ROOT / "narrative_requests" / f"{case_id}.json"
        write_json(request_path, request)
        contract_rows.append(
            {
                "case_id": case_id,
                "seed": seed,
                "family_id": family_id,
                "fact_count": len(locked["facts"]),
                "interpretation_count": len(locked["interpretations"]),
                "uncertainty_count": len(locked["uncertainties"]),
                "refusal_count": len(locked["refusals"]),
                "locked_content_sha256": locked_hash,
                "contract_path": contract_path.relative_to(ROOT).as_posix(),
                "deterministic_report_path": baseline_path.relative_to(ROOT).as_posix(),
                "narrative_request_path": request_path.relative_to(ROOT).as_posix(),
            }
        )
        deterministic_items.append({"case_id": case_id, "condition": "deterministic", "path": baseline_path.relative_to(ROOT).as_posix()})
        hybrid_items.append({"case_id": case_id, "condition": "hybrid_repeat_1", "path": f"results/cic_iot_diad_task62_c2_hybrid_reports_v425/reports/{case_id}/repeat_1_final.json"})
        for repetition in (1, 2, 3):
            call_index += 1
            call_rows.append(
                {
                    "call_index": call_index,
                    "case_id": case_id,
                    "seed": seed,
                    "family_id": family_id,
                    "repetition": repetition,
                    "model": config["model"]["model_id"],
                    "reasoning_effort": config["model"]["reasoning_effort"],
                    "max_output_tokens": config["model"]["max_output_tokens"],
                    "contract_path": contract_path.relative_to(ROOT).as_posix(),
                    "planned_narrative_path": f"results/cic_iot_diad_task62_c2_hybrid_reports_v425/reports/{case_id}/repeat_{repetition}_narrative.json",
                    "planned_final_report_path": f"results/cic_iot_diad_task62_c2_hybrid_reports_v425/reports/{case_id}/repeat_{repetition}_final.json",
                }
            )
        source_paths.extend([evidence_path, contract_path, baseline_path, request_path])

    table_root = C1_ROOT / "tables"
    write_csv(
        table_root / "task62c1_content_contract_manifest.csv",
        contract_rows,
        ["case_id", "seed", "family_id", "fact_count", "interpretation_count", "uncertainty_count", "refusal_count", "locked_content_sha256", "contract_path", "deterministic_report_path", "narrative_request_path"],
    )
    write_csv(
        table_root / "task62c1_llm_call_plan.csv",
        call_rows,
        ["call_index", "case_id", "seed", "family_id", "repetition", "model", "reasoning_effort", "max_output_tokens", "contract_path", "planned_narrative_path", "planned_final_report_path"],
    )

    review_items = deterministic_items + hybrid_items
    randomizer = random.Random(int(config["human_evaluation"]["randomization_seed"]))
    randomizer.shuffle(review_items)
    blind_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for order, item in enumerate(review_items, 1):
        blinded_id = f"item_{order:03d}"
        blind_rows.append({"display_order": order, "blinded_item_id": blinded_id, **item})
        score_rows.append(
            {
                "blinded_item_id": blinded_id,
                "factual_accuracy_1_to_5": "",
                "clarity_1_to_5": "",
                "usefulness_1_to_5": "",
                "trustworthiness_1_to_5": "",
                "uncertainty_quality_1_to_5": "",
                "analyst_questions_correct_0_to_5": "",
                "completion_seconds": "",
                "notes": "",
            }
        )
    write_csv(table_root / "task62c1_private_blinding_plan.csv", blind_rows, ["display_order", "blinded_item_id", "case_id", "condition", "path"])
    write_csv(table_root / "task62c1_reviewer_score_template.csv", score_rows, list(score_rows[0]))

    source_manifest_rows = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
        for path in sorted(set(source_paths), key=lambda item: item.relative_to(ROOT).as_posix())
    ]
    write_csv(table_root / "task62c1_source_manifest_sha256.csv", source_manifest_rows, ["path", "sha256"])
    metadata = {
        "api_calls": 0,
        "best_output_selection_permitted": False,
        "cases_materialized": len(contract_rows),
        "deterministic_reports": len(contract_rows),
        "expected_hybrid_reports": len(call_rows),
        "hybrid_contract_version": "1.0.0",
        "llm_numeric_tokens_permitted": False,
        "new_shap_evaluations": 0,
        "protocol_id": config["protocol_id"],
        "reserved_test_arrays_materialized": False,
        "review_items_planned": len(review_items),
        "training_permitted": False,
    }
    write_json(C1_ROOT / "task62c1_hybrid_matrix_metadata.json", metadata)
    print("===== TASK 62 C1 HYBRID CONTENT MATRIX =====")
    print("CASES MATERIALIZED:", len(contract_rows))
    print("LOCKED FACTS PER CASE: 4")
    print("DETERMINISTIC REPORTS:", len(contract_rows))
    print("PLANNED HYBRID REPORTS:", len(call_rows))
    print("LLM ALLOWED FIELDS: executive_summary")
    print("LLM NUMERIC TOKENS PERMITTED: False")
    print("BLINDED REVIEW ITEMS PLANNED:", len(review_items))
    print("API CALLS: 0")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("TASK 62 C1 COMPLETE: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

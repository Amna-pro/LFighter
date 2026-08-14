from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "task62_preregistration_v4250.json"
C0_DECISION = ROOT / "results" / "cic_iot_diad_task62_c0_preregistration_v425" / "task62c0_preregistration_decision.json"
C1_ROOT = ROOT / "results" / "cic_iot_diad_task62_c1_hybrid_matrix_v425"
AUDIT_ROOT = ROOT / "results" / "cic_iot_diad_task62_c1_audit_v425"
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check_id": name, "passed": bool(passed), "detail": detail})


def main() -> int:
    config = load_json(CONFIG_PATH)
    c0 = load_json(C0_DECISION)
    metadata = load_json(C1_ROOT / "task62c1_hybrid_matrix_metadata.json")
    contracts = read_csv(C1_ROOT / "tables" / "task62c1_content_contract_manifest.csv")
    calls = read_csv(C1_ROOT / "tables" / "task62c1_llm_call_plan.csv")
    blind = read_csv(C1_ROOT / "tables" / "task62c1_private_blinding_plan.csv")
    scores = read_csv(C1_ROOT / "tables" / "task62c1_reviewer_score_template.csv")
    source_manifest = read_csv(C1_ROOT / "tables" / "task62c1_source_manifest_sha256.csv")
    checks: list[dict[str, Any]] = []

    add(checks, "c0_passed", c0["all_checks_passed"] is True, "C0 preregistration passed.")
    add(checks, "case_count", len(contracts) == 12, f"Observed {len(contracts)} contracts.")
    add(checks, "call_count", len(calls) == 36, f"Observed {len(calls)} planned calls.")
    add(checks, "review_item_count", len(blind) == 24 and len(scores) == 24, "Twenty four paired review items are planned.")
    add(checks, "call_indices_complete", [int(row["call_index"]) for row in calls] == list(range(1, 37)), "Call indices are complete and ordered.")
    repetition_counts = Counter(row["case_id"] for row in calls)
    add(checks, "three_calls_per_case", set(repetition_counts.values()) == {3}, "Every case has three calls.")
    add(checks, "all_repetitions_present", all(sorted(int(row["repetition"]) for row in calls if row["case_id"] == case_id) == [1, 2, 3] for case_id in repetition_counts), "Every case has repetitions one through three.")
    add(checks, "model_plan_frozen", all(row["model"] == "gpt-5.6-terra" and row["reasoning_effort"] == "low" for row in calls), "All calls use the frozen model plan.")
    add(checks, "token_plan_frozen", all(int(row["max_output_tokens"]) == 250 for row in calls), "Every call uses the compact token cap.")
    add(checks, "review_repeat_one_only", all(row["condition"] != "hybrid_repeat_1" or row["path"].endswith("repeat_1_final.json") for row in blind), "Human review uses prospectively selected repeat one.")
    add(checks, "paired_conditions", Counter(row["condition"] for row in blind) == Counter({"deterministic": 12, "hybrid_repeat_1": 12}), "Review conditions are balanced.")
    add(checks, "blinded_ids_unique", len({row["blinded_item_id"] for row in blind}) == 24, "Blinded identifiers are unique.")

    exact_contracts = True
    exact_facts = True
    exact_interpretations = True
    no_llm_numbers = True
    deterministic_reports_valid = True
    source_hashes_valid = True
    locked_hashes_unique_by_case = True
    seen_hashes: dict[str, str] = {}
    for row in contracts:
        case_id = row["case_id"]
        contract_path = ROOT / row["contract_path"]
        report_path = ROOT / row["deterministic_report_path"]
        request_path = ROOT / row["narrative_request_path"]
        evidence_path = ROOT / config["inputs"]["c1_evidence_root"] / f"{case_id}.json"
        contract = load_json(contract_path)
        report = load_json(report_path)
        request = load_json(request_path)
        evidence = load_json(evidence_path)
        locked = contract["locked_content"]
        observed_hash = sha256_bytes(canonical(locked))
        exact_contracts = exact_contracts and observed_hash == contract["locked_content_sha256"] == row["locked_content_sha256"]
        expected_facts = [{"statement": item["statement"], "evidence_refs": item["evidence_refs"]} for item in evidence["observed_facts"]]
        expected_interpretations = [
            {"statement": item["statement"], "confidence": item["confidence"], "evidence_refs": item["basis_refs"]}
            for item in evidence["interpretations"]
        ]
        exact_facts = exact_facts and locked["facts"] == expected_facts and len(locked["facts"]) == 4
        exact_interpretations = exact_interpretations and locked["interpretations"] == expected_interpretations
        no_llm_numbers = no_llm_numbers and request["prompt_contains_numeric_tokens"] is False and contract["llm_numeric_tokens_permitted"] is False
        deterministic_reports_valid = deterministic_reports_valid and report["facts"] == locked["facts"] and report["interpretations"] == locked["interpretations"] and report["authority_statement"] == locked["authority_statement"] and bool(report["executive_summary"])
        if observed_hash in seen_hashes and seen_hashes[observed_hash] != case_id:
            locked_hashes_unique_by_case = False
        seen_hashes[observed_hash] = case_id
    add(checks, "contract_hashes_verified", exact_contracts, "Every locked content hash was independently recomputed.")
    add(checks, "facts_exactly_preserved", exact_facts, "All four facts match source evidence exactly.")
    add(checks, "interpretations_exactly_preserved", exact_interpretations, "Interpretations and citations match source evidence exactly.")
    add(checks, "narrative_requests_exclude_numbers", no_llm_numbers, "Narrative request metadata confirms numeric exclusion.")
    add(checks, "deterministic_reports_valid", deterministic_reports_valid, "All comparator reports preserve locked fields.")
    add(checks, "contract_hashes_case_specific", locked_hashes_unique_by_case and len(seen_hashes) == 12, "Every case has a distinct locked content hash.")

    missing_hashes: list[str] = []
    for row in source_manifest:
        path = ROOT / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            source_hashes_valid = False
            missing_hashes.append(row["path"])
    add(checks, "source_hashes_verified", source_hashes_valid, "All source hashes match." if source_hashes_valid else json.dumps(missing_hashes))
    planned_cost = 36 * (
        int(config["cost_control"]["planned_maximum_input_tokens_per_call"]) * float(config["cost_control"]["input_usd_per_million_tokens"])
        + int(config["cost_control"]["planned_maximum_output_tokens_per_call"]) * float(config["cost_control"]["output_usd_per_million_tokens"])
    ) / 1_000_000
    add(checks, "planned_cost_within_ceiling", planned_cost <= float(config["cost_control"]["hard_estimated_cost_ceiling_usd"]), f"Conservative planned cost is USD {planned_cost:.6f}.")
    add(checks, "metadata_no_calls", metadata["api_calls"] == 0, "C1 made no API call.")
    add(checks, "metadata_no_training", metadata["training_permitted"] is False, "Training remained prohibited.")
    add(checks, "metadata_no_shap", metadata["new_shap_evaluations"] == 0, "No new SHAP evaluation occurred.")
    add(checks, "metadata_reserved_test_closed", metadata["reserved_test_arrays_materialized"] is False, "Reserved test arrays remain closed.")
    add(checks, "metadata_no_best_output", metadata["best_output_selection_permitted"] is False, "Best output selection remains prohibited.")
    add(checks, "metadata_counts", metadata["cases_materialized"] == 12 and metadata["expected_hybrid_reports"] == 36 and metadata["review_items_planned"] == 24, "Metadata counts are exact.")
    builder_text = (ROOT / "scripts" / "build_task62_c1_hybrid_matrix_v425.py").read_text(encoding="utf-8").lower()
    forbidden = ["responses.create", "from openai", "import openai", "import requests", "import httpx", "urllib.request"]
    add(checks, "builder_has_no_network_client", not any(marker in builder_text for marker in forbidden), "C1 builder contains no API or network client.")

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    table_root = AUDIT_ROOT / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    with (table_root / "task62c1_hybrid_matrix_audit_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "passed", "detail"])
        writer.writeheader()
        writer.writerows(checks)
    audit_sources = [
        CONFIG_PATH,
        C0_DECISION,
        C1_ROOT / "task62c1_hybrid_matrix_metadata.json",
        C1_ROOT / "tables" / "task62c1_content_contract_manifest.csv",
        C1_ROOT / "tables" / "task62c1_llm_call_plan.csv",
        C1_ROOT / "tables" / "task62c1_private_blinding_plan.csv",
        C1_ROOT / "tables" / "task62c1_reviewer_score_template.csv",
        C1_ROOT / "tables" / "task62c1_source_manifest_sha256.csv",
        ROOT / "scripts" / "build_task62_c1_hybrid_matrix_v425.py",
        ROOT / "scripts" / "audit_task62_c1_hybrid_matrix_v425.py",
    ]
    with (table_root / "task62c1_audit_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256"])
        writer.writeheader()
        for path in sorted(audit_sources, key=lambda item: item.relative_to(ROOT).as_posix()):
            writer.writerow({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
    all_passed = all(row["passed"] for row in checks)
    decision = {
        "all_checks_passed": all_passed,
        "api_calls": 0,
        "cases_verified": len(contracts),
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "deterministic_reports_verified": len(contracts),
        "exact_locked_content_verified": exact_contracts and exact_facts and exact_interpretations,
        "llm_numeric_tokens_permitted": False,
        "new_shap_evaluations": 0,
        "planned_hybrid_reports_verified": len(calls),
        "planned_maximum_cost_usd": planned_cost,
        "ready_for_task62_c2_hybrid_run": all_passed,
        "reserved_test_arrays_materialized": False,
        "review_items_verified": len(blind),
        "training_permitted": False,
    }
    write_json(AUDIT_ROOT / "task62c1_hybrid_matrix_audit_decision.json", decision)
    print("===== TASK 62 C1 HYBRID MATRIX AUDIT =====")
    print(f"Checks passed: {decision['checks_passed']}/{decision['checks_total']}")
    print("CASES VERIFIED:", decision["cases_verified"])
    print("LOCKED CONTENT VERIFIED:", decision["exact_locked_content_verified"])
    print("PLANNED HYBRID REPORTS VERIFIED:", decision["planned_hybrid_reports_verified"])
    print("PLANNED MAXIMUM COST USD:", f"{planned_cost:.6f}")
    print("LLM NUMERIC TOKENS PERMITTED: False")
    print("API CALLS: 0")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR TASK 62 C2 HYBRID RUN:", all_passed)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

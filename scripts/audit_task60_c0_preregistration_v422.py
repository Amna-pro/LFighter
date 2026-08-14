from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "task60_preregistration_v4220.json"
OUT = ROOT / "results" / "cic_iot_diad_task60_c0_preregistration_v422"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    c = load_json(CONFIG_PATH)
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))

    check("task_is_60", c["task"] == 60)
    check("phase_is_c0_c1", c["phase"] == "C0_C1")
    check("parent_tag_frozen", c["parent_tag"] == "task59-c1-evidence-schema-frozen-v4211")
    check("parent_commit_frozen", c["parent_commit"] == "91e2670")
    try:
        resolved = git_value("rev-list", "-n", "1", c["parent_tag"])
    except Exception:
        resolved = ""
    check("parent_tag_resolves", resolved.startswith(c["parent_commit"]), resolved)
    check("parent_is_ancestor", subprocess.run(["git", "merge-base", "--is-ancestor", c["parent_tag"], "HEAD"], cwd=ROOT).returncode == 0)
    check("input_record_exists", (ROOT / c["input_record"]).is_file())
    check("input_schema_exists", (ROOT / c["input_schema"]).is_file())
    check("report_schema_exists", (ROOT / c["report_schema"]).is_file())
    if (ROOT / c["input_record"]).is_file() and (ROOT / c["input_schema"]).is_file():
        evidence = load_json(ROOT / c["input_record"])
        input_schema = load_json(ROOT / c["input_schema"])
        check("canonical_input_schema_valid", not list(Draft202012Validator(input_schema).iter_errors(evidence)))
        check("input_authority_report_only", evidence["authority"]["llm_role"] == "report_only")
        check("input_reserved_arrays_closed", evidence["experiment"]["development_and_validation_only"] is True)
    else:
        check("canonical_input_schema_valid", False)
        check("input_authority_report_only", False)
        check("input_reserved_arrays_closed", False)
    report_schema = load_json(ROOT / c["report_schema"])
    check("report_schema_draft_2020_12", report_schema["$schema"].endswith("2020-12/schema"))
    check("report_schema_self_valid", Draft202012Validator.check_schema(report_schema) is None)
    model = c["model"]
    check("provider_openai", model["provider"] == "OpenAI")
    check("responses_api_frozen", model["api"] == "Responses API")
    check("terra_model_frozen", model["model_id"] == "gpt-5.6-terra")
    check("reasoning_low", model["reasoning_effort"] == "low")
    check("output_cap_700", model["max_output_tokens"] == 700)
    check("no_retries", model["max_retries"] == 0)
    check("one_call_per_record", model["calls_per_record"] == 1)
    check("tools_disabled", model["tools_enabled"] is False)
    check("history_disabled", model["conversation_history_enabled"] is False)
    reporting = c["reporting"]
    check("deterministic_baseline_required", reporting["deterministic_baseline_required"] is True)
    check("structured_output_required", reporting["structured_output_required"] is True)
    check("citations_required", reporting["evidence_citations_required"] is True)
    check("separation_required", reporting["facts_interpretations_uncertainty_separated"] is True)
    check("refusals_required", reporting["unsupported_claim_refusals_required"] is True)
    check("model_independent_interface", reporting["model_independent_interface_required"] is True)
    authority = c["authority"]
    check("llm_report_only", authority["llm_role"] == "report_only")
    check("deterministic_authority", authority["decision_authority"] == "deterministic_pipeline")
    for field in (
        "can_flag_clients",
        "can_change_thresholds",
        "can_change_trust_weights",
        "can_reconstruct_updates",
        "can_control_aggregation",
        "can_claim_malicious_intent",
    ):
        check(f"authority_{field}_false", authority[field] is False)
    bounds = c["data_boundaries"]
    check("training_prohibited", bounds["training_permitted"] is False)
    check("new_shap_prohibited", bounds["new_shap_evaluations_permitted"] is False)
    check("reserved_arrays_closed", bounds["reserved_test_arrays_materialized"] is False)
    check("validated_input_only", bounds["prompt_source"] == "validated Task 59 canonical record only")

    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "task60c0_preregistration_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check_id", "passed", "detail"])
        writer.writerows(checks)
    sources = [
        CONFIG_PATH,
        ROOT / "configs" / "TASK60_PREREGISTRATION_V422.md",
        ROOT / "prompts" / "task60_forensic_report_system_v422.txt",
        ROOT / "prompts" / "task60_forensic_report_user_v422.txt",
        ROOT / "schemas" / "lfighter_forensic_report_v1.schema.json",
        ROOT / "scripts" / "task60_report_contract_v422.py",
        Path(__file__),
    ]
    with (tables / "task60c0_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256"])
        for path in sources:
            writer.writerow([path.relative_to(ROOT).as_posix(), sha256(path)])
    passed = sum(value for _, value, _ in checks)
    decision = {
        "checks_passed": passed,
        "checks_total": len(checks),
        "all_checks_passed": passed == len(checks),
        "outcomes_inspected": False,
        "llm_calls": 0,
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
        "ready_for_c1_report_generation": passed == len(checks),
    }
    write_json(OUT / "task60c0_preregistration_decision.json", decision)
    print("===== TASK 60 C0 REPORTING PREREGISTRATION AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("OUTCOMES INSPECTED: False")
    print("LLM CALLS: 0")
    print("TRAINING PERMITTED: False")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR C1 REPORT GENERATION:", passed == len(checks))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

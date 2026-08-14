from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from task60_report_contract_v422 import ForensicReport


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = ROOT / "configs" / "task61_preregistration_v4230.json"
AMENDMENT_PATH = ROOT / "configs" / "task61_c2_recovery_amendment_v4232.json"
PLAN_PATH = ROOT / "results" / "cic_iot_diad_task61_c1_evidence_matrix_v423" / "tables" / "task61c1_llm_call_plan.csv"
CASE_MANIFEST_PATH = ROOT / "results" / "cic_iot_diad_task61_c1_evidence_matrix_v423" / "tables" / "task61c1_case_manifest.csv"
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "task60_forensic_report_system_v422.txt"
USER_PROMPT_PATH = ROOT / "prompts" / "task60_forensic_report_user_v422.txt"
OUTPUT_ROOT = ROOT / "results" / "cic_iot_diad_task61_c2_llm_reports_v4232"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    temporary.replace(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    return {name: getattr(usage, name, None) for name in ("input_tokens", "output_tokens", "total_tokens")}


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


def validate_report(report: ForensicReport, record: dict[str, Any]) -> None:
    if report.record_id != record["record_id"]:
        raise RuntimeError("Returned record_id does not match the frozen case")
    valid = {item["source_id"] for item in record["provenance"]["source_artifacts"]}
    used = set()
    for item in report.facts:
        used.update(item.evidence_refs)
    for item in report.interpretations:
        used.update(item.evidence_refs)
    invalid = sorted(used - valid)
    if invalid:
        raise RuntimeError(f"Returned report contains invalid evidence references: {invalid}")


def report_paths(case_id: str, repetition: int) -> dict[str, Path]:
    base = OUTPUT_ROOT / "reports" / case_id
    stem = f"repeat_{repetition}"
    return {
        "report": base / f"{stem}.json",
        "metadata": base / f"{stem}_metadata.json",
        "marker": base / f"{stem}_complete.json",
        "attempt": base / f"{stem}_attempt.json",
    }


def complete(paths: dict[str, Path], record_id: str) -> bool:
    required = (paths["report"], paths["metadata"], paths["marker"])
    if not all(path.is_file() for path in required):
        return False
    try:
        marker = load_json(paths["marker"])
        metadata = load_json(paths["metadata"])
        return (
            marker["complete"] is True
            and marker["record_id"] == record_id
            and marker["report_sha256"] == sha256(paths["report"])
            and metadata["report_sha256"] == marker["report_sha256"]
            and metadata["api_calls"] == 1
        )
    except Exception:
        return False


def write_progress(plan: pd.DataFrame, records: dict[str, dict[str, Any]]) -> int:
    rows = []
    completed = 0
    for _, row in plan.iterrows():
        paths = report_paths(str(row["case_id"]), int(row["repetition"]))
        done = complete(paths, records[str(row["case_id"])]["record_id"])
        completed += int(done)
        rows.append(
            {
                "call_index": int(row["call_index"]),
                "case_id": str(row["case_id"]),
                "repetition": int(row["repetition"]),
                "completed": done,
                "report_path": paths["report"].relative_to(ROOT).as_posix(),
            }
        )
    progress = OUTPUT_ROOT / "task61c2_progress.csv"
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return completed


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not available to the report process")
    base_config = load_json(BASE_CONFIG_PATH)
    amendment = load_json(AMENDMENT_PATH)
    restart = amendment["confirmatory_restart"]
    plan = pd.read_csv(PLAN_PATH).sort_values("call_index")
    manifest = pd.read_csv(CASE_MANIFEST_PATH)
    if len(plan) != 36 or plan["call_index"].tolist() != list(range(1, 37)):
        raise RuntimeError("Frozen Task 61 call plan is not the expected 36 call matrix")
    records = {str(row["case_id"]): load_json(ROOT / str(row["evidence_path"])) for _, row in manifest.iterrows()}
    completed_before = write_progress(plan, records)
    print("===== TASK 61 C2 CONFIRMATORY FULL RESTART PLAN =====")
    print("PLANNED REPORTS: 36")
    print("COMPLETED BEFORE RUN:", completed_before)
    print("REMAINING CALLS:", 36 - completed_before)
    print("MODEL:", restart["model_id"])
    print("MAXIMUM OUTPUT TOKENS:", restart["maximum_output_tokens"])
    print("MAXIMUM TOTAL API CALLS: 36")
    print("ORIGINAL REPORTS REUSED: False")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")

    client = OpenAI(max_retries=0, timeout=90)
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    user_template = USER_PROMPT_PATH.read_text(encoding="utf-8").strip()
    new_calls = 0
    for _, row in plan.iterrows():
        case_id = str(row["case_id"])
        repetition = int(row["repetition"])
        record = records[case_id]
        paths = report_paths(case_id, repetition)
        if complete(paths, record["record_id"]):
            if paths["attempt"].is_file():
                attempt = load_json(paths["attempt"])
                if attempt.get("status") != "completed":
                    metadata = load_json(paths["metadata"])
                    attempt["status"] = "completed"
                    attempt["response_id"] = metadata.get("response_id")
                    write_json(paths["attempt"], attempt)
            continue
        if paths["attempt"].is_file():
            raise RuntimeError(
                f"Unresolved prior API attempt for call {int(row['call_index'])}; "
                "do not retry without a new documented amendment"
            )
        compact = compact_evidence(record)
        user_prompt = user_template.format(
            evidence_json=json.dumps(compact, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        )
        prompt_hash = sha256_bytes((system_prompt + "\n" + user_prompt).encode("utf-8"))
        print(f"CALL {int(row['call_index'])}/36: {case_id} repeat {repetition}")
        write_json(
            paths["attempt"],
            {
                "protocol_id": amendment["protocol_id"],
                "call_index": int(row["call_index"]),
                "case_id": case_id,
                "repetition": repetition,
                "status": "started",
                "maximum_permitted_attempts_for_item": 1,
            },
        )
        started = time.perf_counter()
        response = client.responses.parse(
            model=restart["model_id"],
            reasoning={"effort": restart["reasoning_effort"]},
            max_output_tokens=restart["maximum_output_tokens"],
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=ForensicReport,
        )
        elapsed = time.perf_counter() - started
        report = response.output_parsed
        if report is None:
            raise RuntimeError(f"Call {int(row['call_index'])} returned no parsed report")
        validate_report(report, record)
        write_json(paths["report"], report.model_dump(mode="json"))
        report_hash = sha256(paths["report"])
        metadata = {
            "protocol_id": amendment["protocol_id"],
            "parent_protocol_id": base_config["protocol_id"],
            "run_role": "confirmatory_full_restart",
            "original_attempt_outputs_reused": False,
            "call_index": int(row["call_index"]),
            "case_id": case_id,
            "record_id": record["record_id"],
            "repetition": repetition,
            "requested_model": restart["model_id"],
            "returned_model": getattr(response, "model", None),
            "response_id": getattr(response, "id", None),
            "response_status": getattr(response, "status", None),
            "reasoning_effort": restart["reasoning_effort"],
            "max_output_tokens": restart["maximum_output_tokens"],
            "elapsed_seconds": round(elapsed, 6),
            "usage": usage_dict(response),
            "record_sha256": sha256_bytes(canonical_bytes(record)),
            "compact_evidence_sha256": sha256_bytes(canonical_bytes(compact)),
            "prompt_sha256": prompt_hash,
            "report_sha256": report_hash,
            "api_calls": 1,
            "max_retries": restart["max_retries"],
            "tools_enabled": False,
            "conversation_history_enabled": False,
            "api_key_persisted": False,
            "training_permitted": False,
            "new_shap_evaluations": 0,
            "reserved_test_arrays_materialized": False,
        }
        write_json(paths["metadata"], metadata)
        write_json(
            paths["marker"],
            {
                "complete": True,
                "record_id": record["record_id"],
                "case_id": case_id,
                "repetition": repetition,
                "report_sha256": report_hash,
                "api_calls": 1,
            },
        )
        write_json(
            paths["attempt"],
            {
                "protocol_id": amendment["protocol_id"],
                "call_index": int(row["call_index"]),
                "case_id": case_id,
                "repetition": repetition,
                "status": "completed",
                "maximum_permitted_attempts_for_item": 1,
                "response_id": getattr(response, "id", None),
            },
        )
        new_calls += 1
        write_progress(plan, records)
    completed_after = write_progress(plan, records)
    if completed_after != 36:
        raise RuntimeError(f"Expected 36 completed reports, found {completed_after}")
    write_json(
        OUTPUT_ROOT / "task61c2_execution_complete.json",
        {
            "complete": True,
            "planned_reports": 36,
            "completed_reports": 36,
            "new_calls_this_invocation": new_calls,
            "protocol_id": amendment["protocol_id"],
            "run_role": "confirmatory_full_restart",
            "confirmatory_api_calls": 36,
            "original_api_attempts_retained": 25,
            "combined_api_attempts": 61,
            "original_attempt_outputs_reused": False,
            "best_output_selection_permitted": False,
            "training_permitted": False,
            "new_shap_evaluations": 0,
            "reserved_test_arrays_materialized": False,
        },
    )
    print("===== TASK 61 C2 CONFIRMATORY FULL RESTART COMPLETE =====")
    print("REPORTS COMPLETE: 36/36")
    print("NEW CALLS THIS INVOCATION:", new_calls)
    print("ORIGINAL ATTEMPTS RETAINED: 25")
    print("COMBINED API ATTEMPTS: 61")
    print("ORIGINAL REPORTS REUSED: False")
    print("BEST OUTPUT SELECTION PERMITTED: False")
    print("TRAINING PERMITTED: False")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from jsonschema import Draft202012Validator
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from task62_c2_common_v425 import (
    C1_ROOT,
    C2_ROOT,
    CONFIG_PATH,
    FINAL_SCHEMA_PATH,
    NARRATIVE_SCHEMA_PATH,
    PREFLIGHT_ROOT,
    SYSTEM_PROMPT_PATH,
    USER_PROMPT_PATH,
    canonical_hash,
    file_hash,
    final_report,
    load_json,
    report_paths,
    usage_cost,
    valid_completion,
    write_json,
)
from task62_capsule_v4251 import derive_capsule_phrase


class HybridNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0.0"]
    executive_summary: str = Field(min_length=1, max_length=600, pattern=r"^[^0-9]*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    return {name: getattr(usage, name, None) for name in ("input_tokens", "output_tokens", "total_tokens")}


def update_progress(rows: list[dict[str, Any]]) -> None:
    C2_ROOT.mkdir(parents=True, exist_ok=True)
    path = C2_ROOT / "task62c2_progress.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["call_index"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    config = load_json(CONFIG_PATH)
    preflight = load_json(PREFLIGHT_ROOT / "task62c2_preflight_decision.json")
    if preflight.get("all_checks_passed") is not True:
        raise RuntimeError("Task 62 C2 preflight did not pass")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not available to the execution process")

    plan = pd.read_csv(C1_ROOT / "tables" / "task62c1_llm_call_plan.csv").sort_values("call_index")
    narrative_validator = Draft202012Validator(load_json(NARRATIVE_SCHEMA_PATH))
    final_validator = Draft202012Validator(load_json(FINAL_SCHEMA_PATH))
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    user_prompt = USER_PROMPT_PATH.read_text(encoding="utf-8").strip()
    prompt_sha256 = canonical_hash({"system": system_prompt, "user": user_prompt})
    client = OpenAI(max_retries=0)
    progress: list[dict[str, Any]] = []
    api_calls_this_run = 0
    known_cost = 0.0

    for call in plan.itertuples():
        case_id = str(call.case_id)
        repetition = int(call.repetition)
        paths = report_paths(case_id, repetition)
        if valid_completion(paths):
            metadata = load_json(paths["metadata"])
            known_cost += usage_cost(metadata, config)
            progress.append({"call_index": int(call.call_index), "case_id": case_id, "repetition": repetition, "status": "reused_valid_completion"})
            update_progress(progress)
            continue
        if paths["attempt"].exists():
            raise RuntimeError(
                f"Unresolved API attempt for {case_id} repeat {repetition}. Preserve all files and obtain a protocol amendment before any rerun."
            )
        if known_cost >= float(config["cost_control"]["hard_estimated_cost_ceiling_usd"]):
            raise RuntimeError("Known estimated cost reached the frozen ceiling before the next call")

        contract = load_json(CONFIG_PATH.parents[1] / str(call.contract_path))
        if contract.get("locked_content_sha256") != canonical_hash(contract["locked_content"]):
            raise RuntimeError(f"Locked content hash mismatch for {case_id}")

        # Case-aware capsule: derives a zero-digit qualitative phrase from
        # the already-locked, already-hash-verified facts for this case,
        # so the executive summary is grounded in this case's actual
        # severity/recovery pattern rather than a single static sentence
        # repeated across all 36 calls.
        capsule_phrase = derive_capsule_phrase(contract["locked_content"])
        case_user_prompt = user_prompt.replace(
            "from this qualitative capsule.",
            f"from this qualitative capsule, which describes {capsule_phrase}.",
        )
        if case_user_prompt == user_prompt:
            raise RuntimeError(
                f"Capsule phrase insertion did not change the prompt for {case_id} "
                "-- the anchor text may not match; refusing to send an unmodified "
                "static prompt silently."
            )

        attempt = {
            "protocol_id": config["protocol_id"],
            "call_index": int(call.call_index),
            "case_id": case_id,
            "repetition": repetition,
            "status": "started_before_api_call",
            "started_at_utc": utc_now(),
            "maximum_permitted_attempts_for_item": 1,
            "api_key_persisted": False,
        }
        write_json(paths["attempt"], attempt)
        started = time.perf_counter()
        try:
            response = client.responses.parse(
                model=config["model"]["model_id"],
                reasoning={"effort": config["model"]["reasoning_effort"]},
                max_output_tokens=config["model"]["max_output_tokens"],
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": case_user_prompt},
                ],
                text_format=HybridNarrative,
            )
            api_calls_this_run += 1
            elapsed = time.perf_counter() - started
            status = getattr(response, "status", None)
            if status != "completed":
                reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
                raise RuntimeError(f"Response not completed: status={status}, reason={reason}")
            narrative_model = response.output_parsed
            if narrative_model is None:
                raise RuntimeError("The API returned no parsed narrative")
            narrative = narrative_model.model_dump(mode="json")
            if list(narrative_validator.iter_errors(narrative)):
                raise RuntimeError("Narrative failed the frozen JSON schema")
            report = final_report(contract, narrative["executive_summary"])
            if list(final_validator.iter_errors(report)):
                raise RuntimeError("Reconstructed report failed the forensic report schema")
            deterministic_fields = config["hybrid_contract"]["deterministic_fields"]
            rebuilt_locked = {field: report[field] for field in deterministic_fields}
            if canonical_hash(rebuilt_locked) != contract["locked_content_sha256"]:
                raise RuntimeError("Reconstructed report changed locked content")

            write_json(paths["narrative"], narrative)
            write_json(paths["final"], report)
            metadata = {
                "protocol_id": config["protocol_id"],
                "call_index": int(call.call_index),
                "case_id": case_id,
                "repetition": repetition,
                "provider": config["model"]["provider"],
                "api": config["model"]["api"],
                "requested_model": config["model"]["model_id"],
                "returned_model": getattr(response, "model", None),
                "response_id": getattr(response, "id", None),
                "response_status": status,
                "reasoning_effort": config["model"]["reasoning_effort"],
                "max_output_tokens": config["model"]["max_output_tokens"],
                "capsule_phrase": capsule_phrase,
                "elapsed_seconds": round(elapsed, 6),
                "usage": usage_dict(response),
                "prompt_sha256": prompt_sha256,
                "locked_content_sha256": contract["locked_content_sha256"],
                "narrative_sha256": file_hash(paths["narrative"]),
                "final_report_sha256": file_hash(paths["final"]),
                "api_calls": 1,
                "max_retries": 0,
                "tools_enabled": False,
                "conversation_history_enabled": False,
                "api_key_persisted": False,
                "training_permitted": False,
                "new_shap_evaluations": 0,
                "reserved_test_arrays_materialized": False,
            }
            write_json(paths["metadata"], metadata)
            attempt.update({"status": "completed", "completed_at_utc": utc_now(), "response_id": metadata["response_id"]})
            write_json(paths["attempt"], attempt)
            marker = {
                "complete": True,
                "call_index": int(call.call_index),
                "case_id": case_id,
                "repetition": repetition,
                "response_id": metadata["response_id"],
                "narrative_sha256": metadata["narrative_sha256"],
                "final_report_sha256": metadata["final_report_sha256"],
                "metadata_sha256": file_hash(paths["metadata"]),
                "api_calls": 1,
            }
            write_json(paths["complete"], marker)
            known_cost += usage_cost(metadata, config)
            progress.append({"call_index": int(call.call_index), "case_id": case_id, "repetition": repetition, "status": "completed"})
            update_progress(progress)
            print(f"COMPLETE {int(call.call_index)}/36: {case_id} repeat {repetition} [{capsule_phrase}]")
        except Exception as exc:
            attempt.update({"status": "failed_or_indeterminate", "stopped_at_utc": utc_now(), "exception_type": type(exc).__name__, "exception_message": str(exc)})
            write_json(paths["attempt"], attempt)
            update_progress(progress + [{"call_index": int(call.call_index), "case_id": case_id, "repetition": repetition, "status": "failed_or_indeterminate"}])
            raise

    all_complete = all(valid_completion(report_paths(str(row.case_id), int(row.repetition))) for row in plan.itertuples())
    execution = {
        "complete": all_complete,
        "protocol_id": config["protocol_id"],
        "planned_reports": 36,
        "completed_reports": sum(valid_completion(report_paths(str(row.case_id), int(row.repetition))) for row in plan.itertuples()),
        "api_calls_this_run": api_calls_this_run,
        "maximum_api_calls": 36,
        "known_estimated_cost_usd": round(known_cost, 9),
        "cost_ceiling_usd": config["cost_control"]["hard_estimated_cost_ceiling_usd"],
        "cost_within_ceiling": known_cost <= config["cost_control"]["hard_estimated_cost_ceiling_usd"],
        "best_output_selection_permitted": False,
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
    }
    write_json(C2_ROOT / "task62c2_execution_complete.json", execution)
    print("===== TASK 62 C2 HYBRID EXECUTION COMPLETE =====")
    print("HYBRID REPORTS COMPLETE:", execution["completed_reports"])
    print("API CALLS THIS RUN:", api_calls_this_run)
    print("KNOWN ESTIMATED COST USD:", execution["known_estimated_cost_usd"])
    print("COST WITHIN CEILING:", execution["cost_within_ceiling"])
    return 0 if all_complete else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

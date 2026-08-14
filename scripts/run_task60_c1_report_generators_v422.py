from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from task60_report_contract_v422 import AUTHORITY_STATEMENT, ForensicReport


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "task60_preregistration_v4220.json"
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "task60_forensic_report_system_v422.txt"
USER_PROMPT_PATH = ROOT / "prompts" / "task60_forensic_report_user_v422.txt"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_evidence(record: dict[str, Any]) -> dict[str, Any]:
    sources = [
        {"source_id": item["source_id"], "sha256": item.get("sha256")}
        for item in record["provenance"]["source_artifacts"]
    ]
    return {
        "record_id": record["record_id"],
        "case_selection": record["case_selection"],
        "experiment": {
            "dataset": record["experiment"]["dataset"],
            "seed": record["experiment"]["seed"],
            "development_and_validation_only": record["experiment"]["development_and_validation_only"],
        },
        "attack": {
            "attack_type": record["attack"]["attack_type"],
            "source_class": record["attack"]["source_class"],
            "target_class": record["attack"]["target_class"],
            "coalition_family": record["attack"]["coalition_family"],
            "coalition_size": record["attack"]["coalition_size"],
            "source_ref": record["attack"]["source_ref"],
        },
        "detection": {
            "summary": record["detection"]["summary"],
            "missing_fields": record["detection"]["missing_fields"],
            "source_refs": record["detection"]["source_refs"],
        },
        "reconstruction": record["reconstruction"],
        "behavior": {
            "condition_summary": record["behavior"]["condition_summary"],
            "round_metrics": record["behavior"]["round_metrics"],
            "source_refs": record["behavior"]["source_refs"],
        },
        "xai": {
            "method": record["xai"]["method"],
            "output": record["xai"]["output"],
            "recovery": record["xai"]["recovery"],
            "top_features": record["xai"]["top_features"][:5],
            "validation": record["xai"]["validation"],
            "missing_states": record["xai"]["missing_states"],
            "source_refs": record["xai"]["source_refs"],
        },
        "observed_facts": record["observed_facts"],
        "interpretations": record["interpretations"],
        "uncertainty": record["uncertainty"],
        "authority": record["authority"],
        "valid_sources": sources,
    }


def deterministic_report(record: dict[str, Any]) -> ForensicReport:
    facts = [
        {"statement": item["statement"], "evidence_refs": item["evidence_refs"]}
        for item in record["observed_facts"][:4]
    ]
    interpretations = [
        {
            "statement": item["statement"],
            "confidence": item["confidence"],
            "evidence_refs": item["basis_refs"],
        }
        for item in record["interpretations"][:2]
    ]
    limitations = record["uncertainty"]["known_limitations"][:4]
    refusals = [f"Not supported by this evidence: {item}." for item in record["uncertainty"]["unsupported_claims"][:4]]
    return ForensicReport(
        schema_version="1.0.0",
        record_id=record["record_id"],
        report_title="LFighter grounded forensic evidence report",
        executive_summary=(
            "This report summarizes the frozen development and validation evidence for the selected LFighter case. "
            "Observed measurements, interpretations, limitations, and unsupported claims are kept separate."
        ),
        facts=facts,
        interpretations=interpretations,
        uncertainties=limitations,
        refusals=refusals,
        authority_statement=AUTHORITY_STATEMENT,
    )


def render_markdown(report: ForensicReport) -> str:
    lines = [
        f"# {report.report_title}",
        "",
        report.executive_summary,
        "",
        "## Observed facts",
        "",
    ]
    for item in report.facts:
        lines.append(f"* {item.statement} [{', '.join(item.evidence_refs)}]")
    lines.extend(["", "## Interpretation", ""])
    for item in report.interpretations:
        lines.append(f"* {item.statement} Confidence: {item.confidence}. [{', '.join(item.evidence_refs)}]")
    lines.extend(["", "## Uncertainty", ""])
    lines.extend(f"* {item}" for item in report.uncertainties)
    lines.extend(["", "## Unsupported claims refused", ""])
    lines.extend(f"* {item}" for item in report.refusals)
    lines.extend(["", "## Authority", "", report.authority_statement, ""])
    return "\n".join(lines)


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    result = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        result[name] = getattr(usage, name, None)
    return result


def validate_refs(report: ForensicReport, compact: dict[str, Any]) -> None:
    valid = {item["source_id"] for item in compact["valid_sources"]}
    used = []
    for item in report.facts:
        used.extend(item.evidence_refs)
    for item in report.interpretations:
        used.extend(item.evidence_refs)
    invalid = sorted(set(used) - valid)
    if invalid:
        raise RuntimeError(f"Report contains invalid evidence references: {invalid}")
    if report.record_id != compact["record_id"]:
        raise RuntimeError("Report record_id differs from the verified input")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_json(CONFIG_PATH)
    input_path = ROOT / config["input_record"]
    output_root = ROOT / config["outputs"]["root"]
    reports_root = output_root / "reports"
    tables_root = output_root / "tables"
    inputs_root = output_root / "inputs"
    marker = output_root / "_task60_c1_complete.json"

    record = load_json(input_path)
    compact = compact_evidence(record)
    record_hash = sha256_bytes(canonical_bytes(record))
    compact_hash = sha256_bytes(canonical_bytes(compact))

    deterministic = deterministic_report(record)
    validate_refs(deterministic, compact)
    write_json(inputs_root / "task60c1_compact_evidence.json", compact)
    write_json(reports_root / "task60c1_deterministic_report.json", deterministic.model_dump(mode="json"))
    (reports_root / "task60c1_deterministic_report.md").write_text(render_markdown(deterministic), encoding="utf-8")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    user_template = USER_PROMPT_PATH.read_text(encoding="utf-8").strip()
    user_prompt = user_template.format(evidence_json=json.dumps(compact, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    prompt_hash = sha256_bytes((system_prompt + "\n" + user_prompt).encode("utf-8"))

    plan = {
        "protocol_id": config["protocol_id"],
        "record_id": record["record_id"],
        "record_sha256": record_hash,
        "compact_evidence_sha256": compact_hash,
        "prompt_sha256": prompt_hash,
        "model_id": config["model"]["model_id"],
        "reasoning_effort": config["model"]["reasoning_effort"],
        "max_output_tokens": config["model"]["max_output_tokens"],
        "planned_api_calls": 0 if args.dry_run else 1,
        "tools_enabled": False,
        "conversation_history_enabled": False,
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
    }
    write_json(tables_root / "task60c1_execution_plan.json", plan)

    print("===== TASK 60 C1 REPORT GENERATOR PLAN =====")
    print("MODEL:", config["model"]["model_id"])
    print("REASONING EFFORT:", config["model"]["reasoning_effort"])
    print("MAX OUTPUT TOKENS:", config["model"]["max_output_tokens"])
    print("DETERMINISTIC BASELINE COMPLETE: True")
    print("TOOLS ENABLED: False")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    if args.dry_run:
        print("DRY RUN COMPLETE: True")
        return 0

    if marker.exists() and not args.force:
        print("EXISTING COMPLETED REPORT REUSED: True")
        print("NEW API CALLS: 0")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not available to the report process")

    client = OpenAI(max_retries=config["model"]["max_retries"], timeout=config["model"]["timeout_seconds"])
    started = time.perf_counter()
    response = client.responses.parse(
        model=config["model"]["model_id"],
        reasoning={"effort": config["model"]["reasoning_effort"]},
        max_output_tokens=config["model"]["max_output_tokens"],
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=ForensicReport,
    )
    elapsed = time.perf_counter() - started
    report = response.output_parsed
    if report is None:
        raise RuntimeError(f"The API did not return a parsed report. Response status: {getattr(response, 'status', None)}")
    validate_refs(report, compact)

    llm_path = reports_root / "task60c1_llm_report.json"
    write_json(llm_path, report.model_dump(mode="json"))
    (reports_root / "task60c1_llm_report.md").write_text(render_markdown(report), encoding="utf-8")
    metadata = {
        "protocol_id": config["protocol_id"],
        "record_id": record["record_id"],
        "provider": config["model"]["provider"],
        "api": config["model"]["api"],
        "requested_model": config["model"]["model_id"],
        "returned_model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "response_status": getattr(response, "status", None),
        "reasoning_effort": config["model"]["reasoning_effort"],
        "max_output_tokens": config["model"]["max_output_tokens"],
        "elapsed_seconds": round(elapsed, 6),
        "usage": usage_dict(response),
        "record_sha256": record_hash,
        "compact_evidence_sha256": compact_hash,
        "prompt_sha256": prompt_hash,
        "report_sha256": sha256_bytes(llm_path.read_bytes()),
        "api_calls": 1,
        "max_retries": 0,
        "tools_enabled": False,
        "conversation_history_enabled": False,
        "api_key_persisted": False,
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
    }
    write_json(output_root / "task60c1_execution_metadata.json", metadata)
    write_json(
        marker,
        {
            "complete": True,
            "record_id": record["record_id"],
            "llm_report_sha256": metadata["report_sha256"],
            "api_calls": 1,
        },
    )
    print("===== TASK 60 C1 EXECUTION COMPLETE =====")
    print("STRUCTURED LLM REPORT COMPLETE: True")
    print("API CALLS: 1")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

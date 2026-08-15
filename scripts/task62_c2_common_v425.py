from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "task62_preregistration_v4250.json"
C1_ROOT = ROOT / "results" / "cic_iot_diad_task62_c1_hybrid_matrix_v425"
C2_ROOT = ROOT / "results" / "cic_iot_diad_task62_c2_hybrid_reports_v425"
SUMMARY_ROOT = ROOT / "results" / "cic_iot_diad_task62_c2_summary_v425"
AUDIT_ROOT = ROOT / "results" / "cic_iot_diad_task62_c2_audit_v425"
PREFLIGHT_ROOT = ROOT / "results" / "cic_iot_diad_task62_c2_preflight_v425"
NARRATIVE_SCHEMA_PATH = ROOT / "schemas" / "lfighter_hybrid_narrative_v1.schema.json"
FINAL_SCHEMA_PATH = ROOT / "schemas" / "lfighter_forensic_report_v1.schema.json"
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "task62_hybrid_narrative_system_v425.txt"
USER_PROMPT_PATH = ROOT / "prompts" / "task62_hybrid_narrative_user_v425.txt"

NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
AUTHORITY_PATTERNS = [
    re.compile(r"\b(?:the\s+)?(?:llm|report|language model)\s+(?:may|can|should|will)\s+(?:flag|change|set|control|reconstruct|decide|recommend)", re.I),
]
INTENT_PATTERNS = [
    re.compile(r"(?<!not )(?<!does not )\b(?:establish(?:es|ed)?|prove[sd]?|confirm(?:s|ed)?)\s+(?:the\s+)?malicious intent\b", re.I),
]
ORACLE_PATTERNS = [
    re.compile(r"\b(?:achiev(?:e|ed)|restor(?:e|ed)|confirm(?:s|ed)?)\s+(?:an?\s+|the\s+)?(?:exact\s+)?oracle clean", re.I),
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    temporary.replace(path)


def report_paths(case_id: str, repetition: int) -> dict[str, Path]:
    base = C2_ROOT / "reports" / case_id / f"repeat_{repetition}"
    return {
        "attempt": base.with_name(base.name + "_attempt.json"),
        "narrative": base.with_name(base.name + "_narrative.json"),
        "final": base.with_name(base.name + "_final.json"),
        "metadata": base.with_name(base.name + "_metadata.json"),
        "complete": base.with_name(base.name + "_complete.json"),
    }


def final_report(contract: dict[str, Any], executive_summary: str) -> dict[str, Any]:
    report = dict(contract["locked_content"])
    report["executive_summary"] = executive_summary
    return report


def locked_subset(report: dict[str, Any], deterministic_fields: list[str]) -> dict[str, Any]:
    return {field: report[field] for field in deterministic_fields}


def narrative_text(report: dict[str, Any]) -> str:
    parts = [str(report.get("executive_summary", ""))]
    for section in ("facts", "interpretations"):
        parts.extend(str(item.get("statement", "")) for item in report.get(section, []))
    parts.extend(str(item) for item in report.get("uncertainties", []))
    parts.extend(str(item) for item in report.get("refusals", []))
    return "\n".join(parts)


def numeric_values(text: str) -> list[float]:
    return [float(token) for token in NUMBER_RE.findall(text)]


def numeric_consistency(values: list[list[float]]) -> float:
    if len(values) < 2:
        return 1.0
    scores: list[float] = []
    for left_index in range(len(values)):
        for right_index in range(left_index + 1, len(values)):
            left = values[left_index]
            right = values[right_index]
            matched = sum(
                any(math.isclose(item, candidate, rel_tol=1e-9, abs_tol=5e-7) for candidate in right)
                for item in left
            )
            denominator = max(len(left), len(right), 1)
            scores.append(matched / denominator)
    return sum(scores) / len(scores)


def count_patterns(text: str, patterns: list[re.Pattern[str]]) -> int:
    return sum(len(pattern.findall(text)) for pattern in patterns)


def usage_cost(metadata: dict[str, Any], config: dict[str, Any]) -> float:
    usage = metadata.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    prices = config["cost_control"]
    return (
        input_tokens * float(prices["input_usd_per_million_tokens"]) / 1_000_000
        + output_tokens * float(prices["output_usd_per_million_tokens"]) / 1_000_000
    )


def valid_completion(paths: dict[str, Path]) -> bool:
    if not paths["complete"].exists():
        return False
    try:
        marker = load_json(paths["complete"])
        return (
            marker.get("complete") is True
            and paths["narrative"].exists()
            and paths["final"].exists()
            and paths["metadata"].exists()
            and marker.get("narrative_sha256") == file_hash(paths["narrative"])
            and marker.get("final_report_sha256") == file_hash(paths["final"])
            and marker.get("metadata_sha256") == file_hash(paths["metadata"])
        )
    except Exception:
        return False

from __future__ import annotations

import itertools
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "task61_preregistration_v4230.json"
AMENDMENT_PATH = ROOT / "configs" / "task61_c2_recovery_amendment_v4232.json"
SCHEMA_PATH = ROOT / "schemas" / "lfighter_forensic_report_v1.schema.json"
C1_ROOT = ROOT / "results" / "cic_iot_diad_task61_c1_evidence_matrix_v423"
C2_ROOT = ROOT / "results" / "cic_iot_diad_task61_c2_llm_reports_v4232"
ORIGINAL_C2_ROOT = ROOT / "results" / "cic_iot_diad_task61_c2_llm_reports_v423"
SUMMARY_ROOT = ROOT / "results" / "cic_iot_diad_task61_c2_summary_v4232"

NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
AUTHORITY_PATTERNS = [
    re.compile(r"\b(?:the\s+)?(?:llm|report|language model)\s+(?:may|can|should|will)\s+(?:flag|change|set|control|reconstruct|decide|recommend)", re.I),
    re.compile(r"\brecommend(?:s|ed|ation)?\s+(?:flagging|changing|setting|excluding|reconstructing)", re.I),
]
INTENT_PATTERNS = [
    re.compile(r"(?<!not )(?<!does not )(?<!do not )(?<!cannot )(?<!never )\b(?:establish(?:es|ed)?|prove[sd]?|confirm(?:s|ed)?)\s+(?:the\s+)?malicious intent\b", re.I),
    re.compile(r"\bverified malicious intent\b", re.I),
]
ORACLE_PATTERNS = [
    re.compile(r"\b(?:achiev(?:e|ed)|restor(?:e|ed)|confirm(?:s|ed)?)\s+(?:an?\s+|the\s+)?(?:exact\s+)?oracle clean(?:\s+recovery|\s+state)?\b", re.I),
    re.compile(r"\bexact oracle clean recovery (?:was|is) (?:achieved|verified|confirmed)\b", re.I),
]
TERMINOLOGY_PATTERNS = [
    re.compile(r"\bDDoS(?:\s+to\s+|[-–]to[-–])Benign attribution\b", re.I),
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def report_path(case_id: str, repetition: int) -> Path:
    return C2_ROOT / "reports" / case_id / f"repeat_{repetition}.json"


def resolved_review_content_path(item: Any) -> Path:
    frozen_path = str(item.content_path)
    if "cic_iot_diad_task61_c2_llm_reports_v423/" in frozen_path:
        match = re.search(r"repeat_([123])\.json$", frozen_path)
        if match is None:
            raise RuntimeError(f"Cannot resolve frozen LLM review path: {frozen_path}")
        return report_path(str(item.case_id), int(match.group(1)))
    return ROOT / frozen_path


def narrative_parts(report: dict[str, Any], include_refusals: bool = True) -> list[str]:
    parts = [report.get("report_title", ""), report.get("executive_summary", "")]
    parts.extend(str(item.get("statement", "")) for item in report.get("facts", []))
    parts.extend(str(item.get("statement", "")) for item in report.get("interpretations", []))
    parts.extend(str(item) for item in report.get("uncertainties", []))
    if include_refusals:
        parts.extend(str(item) for item in report.get("refusals", []))
    return [part for part in parts if part]


def decision_text(report: dict[str, Any]) -> str:
    parts = [report.get("executive_summary", "")]
    parts.extend(str(item.get("statement", "")) for item in report.get("facts", []))
    parts.extend(str(item.get("statement", "")) for item in report.get("interpretations", []))
    return "\n".join(parts)


def numeric_values(value: Any) -> list[float]:
    found: list[float] = []
    if isinstance(value, bool) or value is None:
        return found
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            found.append(float(value))
    elif isinstance(value, str):
        for token in NUMBER_RE.findall(value):
            try:
                found.append(float(token))
            except ValueError:
                pass
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in {"record_id", "case_id", "sha256", "path", "schema_version"}:
                found.extend(numeric_values(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(numeric_values(item))
    return found


def narrative_numbers(report: dict[str, Any]) -> list[float]:
    values: list[float] = []
    parts = [report.get("executive_summary", "")]
    parts.extend(str(item.get("statement", "")) for item in report.get("facts", []))
    parts.extend(str(item.get("statement", "")) for item in report.get("interpretations", []))
    parts.extend(str(item) for item in report.get("uncertainties", []))
    parts.extend(str(item) for item in report.get("refusals", []))
    for part in parts:
        values.extend(float(token) for token in NUMBER_RE.findall(part))
    return values


def grounded_number(value: float, evidence_values: list[float]) -> bool:
    for evidence in evidence_values:
        for digits in range(0, 7):
            if math.isclose(value, round(evidence, digits), rel_tol=1e-9, abs_tol=5e-7):
                return True
    return False


def required_fact_coverage(report: dict[str, Any], record: dict[str, Any]) -> tuple[int, int]:
    covered = 0
    expected = record["observed_facts"]
    for fact in expected:
        required_numbers = numeric_values(fact["statement"])
        required_refs = set(fact["evidence_refs"])
        matched = False
        for actual in report.get("facts", []):
            actual_numbers = numeric_values(actual.get("statement", ""))
            refs_match = bool(required_refs & set(actual.get("evidence_refs", [])))
            numbers_match = all(any(math.isclose(x, y, rel_tol=1e-9, abs_tol=5e-7) for y in actual_numbers) for x in required_numbers)
            if refs_match and numbers_match:
                matched = True
                break
        covered += int(matched)
    return covered, len(expected)


def used_citations(report: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for section in ("facts", "interpretations"):
        for item in report.get(section, []):
            refs.update(str(ref) for ref in item.get("evidence_refs", []))
    return refs


def syllables(word: str) -> int:
    lower = word.lower()
    groups = re.findall(r"[aeiouy]+", lower)
    count = len(groups)
    if lower.endswith("e") and count > 1 and not lower.endswith(("le", "ye")):
        count -= 1
    return max(count, 1)


def readability(text: str) -> tuple[int, int, float]:
    words = WORD_RE.findall(text)
    sentences = [piece for piece in re.split(r"[.!?]+", text) if piece.strip()]
    if not words or not sentences:
        return len(words), len(sentences), float("nan")
    score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (sum(syllables(word) for word in words) / len(words))
    return len(words), len(sentences), round(score, 6)


def count_patterns(text: str, patterns: list[re.Pattern[str]]) -> int:
    return sum(len(pattern.findall(text)) for pattern in patterns)


def usage_cost(metadata: dict[str, Any], config: dict[str, Any]) -> float:
    usage = metadata.get("usage") or {}
    details = usage.get("input_tokens_details") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cached = int(details.get("cached_tokens") or 0)
    cache_write = int(details.get("cache_write_tokens") or 0)
    ordinary = max(input_tokens - cached - cache_write, 0)
    prices = config["cost_control"]
    cost = ordinary * prices["input_usd_per_million_tokens"] / 1_000_000
    cost += cached * 0.20 / 1_000_000
    cost += cache_write * prices["input_usd_per_million_tokens"] * prices["cache_write_multiplier"] / 1_000_000
    cost += output_tokens * prices["output_usd_per_million_tokens"] / 1_000_000
    return cost


def render_review_item(item_id: str, content: dict[str, Any]) -> str:
    lines = [f"TASK 61 BLINDED REVIEW ITEM {item_id}", ""]
    if "facts" in content and "executive_summary" in content:
        lines.extend(["SUMMARY", str(content["executive_summary"]), "", "FACTS"])
        for fact in content.get("facts", []):
            lines.append(f"* {fact['statement']}")
        lines.extend(["", "INTERPRETATIONS"])
        for item in content.get("interpretations", []):
            lines.append(f"* {item['statement']} [confidence: {item.get('confidence', 'not stated')}]")
        lines.extend(["", "UNCERTAINTIES"])
        lines.extend(f"* {item}" for item in content.get("uncertainties", []))
        lines.extend(["", "REFUSALS"])
        lines.extend(f"* {item}" for item in content.get("refusals", []))
    else:
        lines.extend(["SUMMARY", "Structured verified evidence for one frozen LFighter case.", "", "FACTS"])
        lines.extend(f"* {item['statement']}" for item in content.get("observed_facts", []))
        lines.extend(["", "INTERPRETATIONS"])
        for item in content.get("interpretations", []):
            lines.append(f"* {item['statement']} [confidence: {item.get('confidence', 'not stated')}]")
        lines.extend(["", "UNCERTAINTIES"])
        lines.extend(f"* {item}" for item in content.get("uncertainty", {}).get("known_limitations", []))
        lines.extend(["", "REFUSALS"])
        lines.extend(f"* Not supported by this evidence: {item}." for item in content.get("uncertainty", {}).get("unsupported_claims", []))
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    config = load_json(CONFIG_PATH)
    amendment = load_json(AMENDMENT_PATH)
    schema = load_json(SCHEMA_PATH)
    plan = pd.read_csv(C1_ROOT / "tables" / "task61c1_llm_call_plan.csv").sort_values("call_index")
    manifest = pd.read_csv(C1_ROOT / "tables" / "task61c1_case_manifest.csv")
    records = {str(row.case_id): load_json(ROOT / str(row.evidence_path)) for row in manifest.itertuples()}
    rows: list[dict[str, Any]] = []
    for call in plan.itertuples():
        output_path = report_path(str(call.case_id), int(call.repetition))
        metadata_path = output_path.with_name(output_path.stem + "_metadata.json")
        report = load_json(output_path)
        metadata = load_json(metadata_path)
        record = records[str(call.case_id)]
        schema_valid = not list(jsonschema.Draft202012Validator(schema).iter_errors(report))
        valid_refs = {item["source_id"] for item in record["provenance"]["source_artifacts"]}
        citations = used_citations(report)
        report_numbers = narrative_numbers(report)
        evidence_numbers = numeric_values(record)
        grounded_count = sum(grounded_number(value, evidence_numbers) for value in report_numbers)
        covered, expected = required_fact_coverage(report, record)
        text = " ".join(narrative_parts(report))
        words, sentences, reading = readability(text)
        rows.append(
            {
                "call_index": int(call.call_index),
                "case_id": str(call.case_id),
                "seed": int(call.seed),
                "family_id": str(call.family_id),
                "repetition": int(call.repetition),
                "schema_valid": schema_valid,
                "citation_valid": citations <= valid_refs and bool(citations),
                "citation_count": len(citations),
                "numeric_values_reported": len(report_numbers),
                "numeric_values_grounded": grounded_count,
                "numeric_grounding_rate": grounded_count / len(report_numbers) if report_numbers else 1.0,
                "required_facts_covered": covered,
                "required_facts_expected": expected,
                "required_fact_coverage_rate": covered / expected,
                "authority_violation_count": count_patterns(decision_text(report), AUTHORITY_PATTERNS),
                "unsupported_intent_claim_count": count_patterns(decision_text(report), INTENT_PATTERNS),
                "unsupported_oracle_claim_count": count_patterns(decision_text(report), ORACLE_PATTERNS),
                "terminology_error_count_report_only": count_patterns(text, TERMINOLOGY_PATTERNS),
                "word_count": words,
                "sentence_count": sentences,
                "flesch_reading_ease": reading,
                "elapsed_seconds": float(metadata["elapsed_seconds"]),
                "input_tokens": int((metadata.get("usage") or {}).get("input_tokens") or 0),
                "output_tokens": int((metadata.get("usage") or {}).get("output_tokens") or 0),
                "total_tokens": int((metadata.get("usage") or {}).get("total_tokens") or 0),
                "estimated_cost_usd": usage_cost(metadata, config),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("call_index")
    tables = SUMMARY_ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "task61c2_report_metrics.csv", index=False)

    consistency_rows = []
    for case_id, group in plan.groupby("case_id", sort=True):
        numeric_sets = []
        citation_sets = []
        for call in group.sort_values("repetition").itertuples():
            report = load_json(report_path(str(call.case_id), int(call.repetition)))
            numeric_sets.append({round(value, 9) for value in narrative_numbers(report)})
            citation_sets.append(used_citations(report))
        numeric_scores = []
        citation_scores = []
        for left, right in itertools.combinations(range(3), 2):
            n_union = numeric_sets[left] | numeric_sets[right]
            c_union = citation_sets[left] | citation_sets[right]
            numeric_scores.append(len(numeric_sets[left] & numeric_sets[right]) / len(n_union) if n_union else 1.0)
            citation_scores.append(len(citation_sets[left] & citation_sets[right]) / len(c_union) if c_union else 1.0)
        case_metrics = metrics[metrics.case_id == case_id]
        manifest_row = manifest[manifest.case_id == case_id].iloc[0]
        consistency_rows.append(
            {
                "case_id": case_id,
                "seed": int(manifest_row.seed),
                "family_id": manifest_row.family_id,
                "repeat_numeric_consistency_rate": float(np.mean(numeric_scores)),
                "repeat_citation_consistency_rate_report_only": float(np.mean(citation_scores)),
                "all_three_schema_valid": bool(case_metrics.schema_valid.all()),
                "all_three_citations_valid": bool(case_metrics.citation_valid.all()),
                "all_three_numeric_grounded": bool((case_metrics.numeric_grounding_rate == 1.0).all()),
            }
        )
    consistency = pd.DataFrame(consistency_rows)
    consistency.to_csv(tables / "task61c2_case_repeat_consistency.csv", index=False)

    gates = config["automated_metrics"]
    decisions = [
        ("schema_conformance_rate", float(metrics.schema_valid.mean()), gates["schema_conformance_rate_required"], ">="),
        ("citation_validity_rate", float(metrics.citation_valid.mean()), gates["citation_validity_rate_required"], ">="),
        ("numeric_grounding_rate", float(metrics.numeric_values_grounded.sum() / max(metrics.numeric_values_reported.sum(), 1)), gates["numeric_grounding_rate_required"], ">="),
        ("authority_violation_count", int(metrics.authority_violation_count.sum()), gates["authority_violation_count_required"], "=="),
        ("unsupported_intent_claim_count", int(metrics.unsupported_intent_claim_count.sum()), gates["unsupported_intent_claim_count_required"], "=="),
        ("unsupported_oracle_claim_count", int(metrics.unsupported_oracle_claim_count.sum()), gates["unsupported_oracle_claim_count_required"], "=="),
        ("required_fact_coverage_rate", float(metrics.required_facts_covered.sum() / metrics.required_facts_expected.sum()), gates["required_fact_coverage_rate_minimum"], ">="),
        ("repeat_numeric_consistency_rate", float(consistency.repeat_numeric_consistency_rate.mean()), gates["repeat_numeric_consistency_rate_minimum"], ">="),
    ]
    gate_rows = []
    for metric, observed, threshold, operator in decisions:
        passed = observed >= threshold if operator == ">=" else observed == threshold
        gate_rows.append({"metric": metric, "observed": observed, "operator": operator, "threshold": threshold, "passed": passed})
    gate_table = pd.DataFrame(gate_rows)
    gate_table.to_csv(tables / "task61c2_automated_gate_decisions.csv", index=False)

    original_metadata = sorted(ORIGINAL_C2_ROOT.glob("reports/*/repeat_[123]_metadata.json"))
    original_known_cost = sum(usage_cost(load_json(path), config) for path in original_metadata)
    confirmatory_cost = float(metrics.estimated_cost_usd.sum())
    combined_known_cost = confirmatory_cost + original_known_cost
    amended_ceiling = float(amendment["combined_accounting"]["amended_cost_ceiling_usd"])
    cost_row = {
        "confirmatory_api_calls": 36,
        "original_completed_api_calls": len(original_metadata),
        "original_incomplete_api_calls": 1,
        "combined_api_attempts": 36 + len(original_metadata) + 1,
        "input_tokens": int(metrics.input_tokens.sum()),
        "output_tokens": int(metrics.output_tokens.sum()),
        "total_tokens": int(metrics.total_tokens.sum()),
        "confirmatory_estimated_cost_usd": confirmatory_cost,
        "original_known_estimated_cost_usd": original_known_cost,
        "combined_known_estimated_cost_usd": combined_known_cost,
        "failed_attempt_cost_observed": False,
        "combined_total_cost_fully_observed": False,
        "amended_cost_ceiling_usd": amended_ceiling,
        "combined_known_cost_within_amended_ceiling": combined_known_cost <= amended_ceiling,
        "pricing_retrieved_date": config["cost_control"]["pricing_retrieved_date"],
    }
    pd.DataFrame([cost_row]).to_csv(tables / "task61c2_cost_summary.csv", index=False)

    blind = pd.read_csv(C1_ROOT / "tables" / "task61c1_blinded_review_manifest.csv").sort_values("display_order")
    review_root = SUMMARY_ROOT / "human_review_package"
    item_root = review_root / "blinded_items"
    item_root.mkdir(parents=True, exist_ok=True)
    public_rows = []
    for item in blind.itertuples():
        content = load_json(resolved_review_content_path(item))
        item_path = item_root / f"{item.blinded_item_id}.txt"
        item_path.write_text(render_review_item(str(item.blinded_item_id), content), encoding="utf-8")
        public_rows.append(
            {
                "display_order": int(item.display_order),
                "blinded_item_id": str(item.blinded_item_id),
                "review_item_path": item_path.relative_to(ROOT).as_posix(),
            }
        )
    pd.DataFrame(public_rows).to_csv(review_root / "task61c2_blinded_review_public_manifest.csv", index=False)
    shutil.copy2(C1_ROOT / "tables" / "task61c1_reviewer_score_template.csv", review_root / "task61c2_reviewer_score_template.csv")
    shutil.copy2(C1_ROOT / "tables" / "task61c1_blinded_review_manifest.csv", tables / "task61c2_private_blinding_key.csv")
    shutil.copy2(C1_ROOT / "tables" / "task61c1_analyst_answer_key.csv", tables / "task61c2_private_analyst_answer_key.csv")
    instructions = """# Task 61 blinded human review\n\nTwo independent reviewers are required. Review items in display order and do not open the private blinding key or private analyst answer key until both score sheets are complete.\n\nFor every item, score factual accuracy, clarity, usefulness, trustworthiness, and uncertainty quality from 1 to 5. Record the answers you infer for the five analyst questions: plain attack rate, paired clean rate, recovery fraction, whether malicious intent may be claimed, and whether reserved test arrays were materialized. A coordinator compares those answers with the private key and records the number correct from 0 to 5.\n\nDo not discuss scores with the other reviewer until both files are frozen. Human review completion and statistical comparison occur in Task 61 C3.\n"""
    (review_root / "TASK61_C2_HUMAN_REVIEW_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")

    all_gates_passed = bool(gate_table.passed.all())
    decision = {
        "protocol_id": amendment["protocol_id"],
        "parent_protocol_id": config["protocol_id"],
        "original_attempt_disposition": amendment["original_attempt"]["disposition"],
        "original_api_attempts": 25,
        "confirmatory_api_calls": 36,
        "combined_api_attempts": 61,
        "original_attempt_outputs_reused": False,
        "automated_gate_result": "PASS" if all_gates_passed else "FAIL",
        "automated_gates_passed": int(gate_table.passed.sum()),
        "automated_gates_total": int(len(gate_table)),
        "reports_evaluated": 36,
        "cases_evaluated": 12,
        "repeats_per_case": 3,
        "terminology_error_count_report_only": int(metrics.terminology_error_count_report_only.sum()),
        "confirmatory_estimated_cost_usd": cost_row["confirmatory_estimated_cost_usd"],
        "original_known_estimated_cost_usd": cost_row["original_known_estimated_cost_usd"],
        "combined_known_estimated_cost_usd": cost_row["combined_known_estimated_cost_usd"],
        "failed_attempt_cost_observed": False,
        "combined_total_cost_fully_observed": False,
        "combined_known_cost_within_amended_ceiling": cost_row["combined_known_cost_within_amended_ceiling"],
        "human_review_complete": False,
        "minimum_independent_reviewers_required": config["human_evaluation"]["minimum_independent_reviewers"],
        "final_task61_conclusion": "PENDING_HUMAN_REVIEW",
        "llm_superiority_claim_permitted": False,
        "all_negative_outputs_retained": True,
        "best_output_selection_permitted": False,
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
        "ready_for_independent_human_review": all_gates_passed,
    }
    write_json(SUMMARY_ROOT / "task61c2_automated_evaluation_decision.json", decision)
    closeout = f"""# Task 61 C2 recovery closeout\n\nThe original run was aborted after 24 complete reports and one incomplete API attempt caused by the frozen 700 token cap. Those outputs were retained for audit but were not reused in the confirmatory comparison.\n\nA fresh 36 report matrix was generated under amendment {amendment['protocol_id']} with a 1100 token cap. Combined accounting is 61 API attempts.\n\nAutomated gate result: {decision['automated_gate_result']}\n\nReports evaluated: 36 across 12 cases and three retained repeats per case.\n\nConfirmatory estimated API cost: USD {decision['confirmatory_estimated_cost_usd']:.6f}. Combined known cost: USD {decision['combined_known_estimated_cost_usd']:.6f}. The failed attempt cost was not exposed by the SDK, so exact combined total cost is not claimed.\n\nFinal Task 61 conclusion: PENDING HUMAN REVIEW. No LLM superiority claim is permitted until Task 61 C3 receives at least two completed independent blinded reviews.\n"""
    (SUMMARY_ROOT / "TASK61_C2_CLOSEOUT.md").write_text(closeout, encoding="utf-8")
    print("===== TASK 61 C2 AUTOMATED EVALUATION =====")
    print("AUTOMATED GATE RESULT:", decision["automated_gate_result"])
    print("REPORTS EVALUATED: 36")
    print("CASES EVALUATED: 12")
    print("HUMAN REVIEW COMPLETE: False")
    print("FINAL TASK 61 CONCLUSION: PENDING_HUMAN_REVIEW")
    print("LLM SUPERIORITY CLAIM PERMITTED: False")
    print("CONFIRMATORY API CALLS: 36")
    print("ORIGINAL API ATTEMPTS RETAINED: 25")
    print("COMBINED API ATTEMPTS: 61")
    print("ORIGINAL REPORTS REUSED: False")
    print("CONFIRMATORY ESTIMATED API COST USD:", f"{decision['confirmatory_estimated_cost_usd']:.6f}")
    print("COMBINED KNOWN API COST USD:", f"{decision['combined_known_estimated_cost_usd']:.6f}")
    print("EXACT COMBINED COST OBSERVED: False")
    print("READY FOR INDEPENDENT HUMAN REVIEW:", decision["ready_for_independent_human_review"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

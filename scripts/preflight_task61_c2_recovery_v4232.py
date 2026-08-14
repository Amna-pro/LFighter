from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_PATH = ROOT / "configs" / "task61_c2_recovery_amendment_v4232.json"
ORIGINAL_ROOT = ROOT / "results" / "cic_iot_diad_task61_c2_llm_reports_v423"
FRESH_ROOT = ROOT / "results" / "cic_iot_diad_task61_c2_llm_reports_v4232"
RECOVERY_ROOT = ROOT / "results" / "cic_iot_diad_task61_c2_recovery_amendment_v4232"
C1_ROOT = ROOT / "results" / "cic_iot_diad_task61_c1_evidence_matrix_v423"
C1_AUDIT = ROOT / "results" / "cic_iot_diad_task61_c1_audit_v423" / "task61c1_evidence_matrix_audit_decision.json"
CONSOLE_LOG = ROOT / "evidence" / "task61_c2_aborted_console_log.txt"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def frozen_tag_ok(tag: str) -> bool:
    resolve = subprocess.run(["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"], cwd=ROOT, capture_output=True)
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", tag, "HEAD"], cwd=ROOT, capture_output=True)
    return resolve.returncode == 0 and ancestor.returncode == 0


def verified_completed_indices(root: Path, plan: pd.DataFrame) -> tuple[list[int], bool]:
    completed: list[int] = []
    valid = True
    for call in plan.itertuples():
        base = root / "reports" / str(call.case_id)
        report = base / f"repeat_{int(call.repetition)}.json"
        metadata = base / f"repeat_{int(call.repetition)}_metadata.json"
        marker = base / f"repeat_{int(call.repetition)}_complete.json"
        attempt = base / f"repeat_{int(call.repetition)}_attempt.json"
        present = [report.is_file(), metadata.is_file(), marker.is_file()]
        if any(present) and not all(present):
            valid = False
            continue
        if attempt.is_file() and not all(present):
            valid = False
            continue
        if all(present):
            marker_data = load_json(marker)
            metadata_data = load_json(metadata)
            valid &= marker_data.get("complete") is True
            valid &= marker_data.get("report_sha256") == sha256(report) == metadata_data.get("report_sha256")
            valid &= metadata_data.get("api_calls") == 1
            if root == FRESH_ROOT:
                valid &= attempt.is_file()
                if attempt.is_file():
                    valid &= load_json(attempt).get("status") in {"started", "completed"}
            completed.append(int(call.call_index))
    return completed, valid


def main() -> int:
    amendment = load_json(AMENDMENT_PATH)
    audit = load_json(C1_AUDIT)
    plan = pd.read_csv(C1_ROOT / "tables" / "task61c1_llm_call_plan.csv").sort_values("call_index")
    completed, valid_original = verified_completed_indices(ORIGINAL_ROOT, plan)
    fresh_completed, valid_fresh = verified_completed_indices(FRESH_ROOT, plan)
    fresh_is_prefix = fresh_completed == list(range(1, len(fresh_completed) + 1))
    log_text = CONSOLE_LOG.read_text(encoding="utf-8", errors="replace")
    checks = [
        ("frozen C1 tag verified", frozen_tag_ok(amendment["parent_tag"])),
        ("C1 audit passed", audit.get("all_checks_passed") is True),
        ("C1 authorized C2", audit.get("ready_for_task61_c2_multireport_run") is True),
        ("call plan remains 36 rows", len(plan) == 36),
        ("original completed indices are exactly 1 through 24", completed == list(range(1, 25))),
        ("original completed files pass hashes", valid_original),
        ("report 25 is not complete", 25 not in completed),
        ("no later report is complete", not any(index > 25 for index in completed)),
        ("console log identifies call 25", "CALL 25/36" in log_text),
        ("console log identifies JSON EOF", "Invalid JSON: EOF while parsing a string" in log_text),
        ("console log identifies 700 token cap", "MAXIMUM OUTPUT TOKENS: 700" in log_text),
        ("automated summary was not run", not (ROOT / "results" / "cic_iot_diad_task61_c2_summary_v423").exists()),
        ("fresh restart contains only verified complete outputs", valid_fresh),
        ("fresh restart completed indices form a prefix", fresh_is_prefix),
        ("entire matrix restart required", amendment["confirmatory_restart"]["restart_entire_matrix"] is True),
        ("original reports cannot be reused", amendment["confirmatory_restart"]["reuse_original_completed_reports"] is False),
        ("best output selection prohibited", amendment["confirmatory_restart"]["best_output_selection_permitted"] is False),
    ]
    failures = [name for name, passed in checks if not passed]
    RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
    with (RECOVERY_ROOT / "task61c2_recovery_preflight_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check_id", "passed"])
        writer.writerows(checks)
    incident = {
        "protocol_id": amendment["protocol_id"],
        "original_attempt_disposition": amendment["original_attempt"]["disposition"],
        "completed_reports_verified": len(completed),
        "incomplete_api_attempts_observed": 1,
        "total_original_api_attempts": 25,
        "failure_call_index": 25,
        "failure_type": "structured_json_truncated_at_max_output_tokens",
        "response_id_available": False,
        "failed_attempt_usage_available": False,
        "automated_semantic_outcomes_evaluated_before_amendment": False,
        "console_log_sha256": sha256(CONSOLE_LOG),
        "original_reports_retained": True,
        "original_reports_reused": False,
        "confirmatory_restart_planned_calls": 36,
        "combined_api_attempts_after_successful_restart": 61,
        "preflight_passed": not failures,
    }
    write_json(RECOVERY_ROOT / "task61c2_aborted_attempt_incident.json", incident)
    print("===== TASK 61 C2 RECOVERY PREFLIGHT =====")
    print(f"Checks passed: {len(checks) - len(failures)}/{len(checks)}")
    print("ORIGINAL COMPLETED REPORTS VERIFIED:", len(completed))
    print("ORIGINAL INCOMPLETE ATTEMPTS RECORDED: 1")
    print("ORIGINAL ATTEMPT RETAINED: True")
    print("ORIGINAL REPORTS REUSED: False")
    print("FRESH CONFIRMATORY REPORTS PLANNED: 36")
    print("FRESH REPORTS ALREADY COMPLETE:", len(fresh_completed))
    print("AMENDED MAXIMUM OUTPUT TOKENS: 1100")
    print("READY FOR FULL RESTART:", not failures)
    if failures:
        raise RuntimeError("Recovery preflight failures: " + "; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

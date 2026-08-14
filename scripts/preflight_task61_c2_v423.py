from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "task61_preregistration_v4230.json"
C1_AUDIT = ROOT / "results" / "cic_iot_diad_task61_c1_audit_v423" / "task61c1_evidence_matrix_audit_decision.json"
PLAN = ROOT / "results" / "cic_iot_diad_task61_c1_evidence_matrix_v423" / "tables" / "task61c1_llm_call_plan.csv"
TAG = "task61-c1-llm-evaluation-preflight-frozen-v4231"


def check_tag() -> bool:
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{TAG}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if resolved.returncode != 0:
        return False
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", TAG, "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return ancestor.returncode == 0


def main() -> int:
    checks: list[tuple[str, bool]] = []
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    audit = json.loads(C1_AUDIT.read_text(encoding="utf-8"))
    plan = pd.read_csv(PLAN).sort_values("call_index")
    checks.extend(
        [
            ("frozen C1 tag resolves and is an ancestor", check_tag()),
            ("C1 audit passed", bool(audit.get("all_checks_passed"))),
            ("C1 audit authorizes C2", bool(audit.get("ready_for_task61_c2_multireport_run"))),
            ("call plan has 36 rows", len(plan) == 36),
            ("call indices are exactly 1 through 36", plan["call_index"].tolist() == list(range(1, 37))),
            ("twelve cases are represented", plan["case_id"].nunique() == 12),
            ("three repeats per case", bool((plan.groupby("case_id").size() == 3).all())),
            ("model is frozen", set(plan["model"]) == {config["model"]["model_id"]}),
            ("reasoning effort is frozen", set(plan["reasoning_effort"]) == {config["model"]["reasoning_effort"]}),
            ("output cap is frozen", set(plan["max_output_tokens"]) == {config["model"]["max_output_tokens"]}),
            ("best output selection is prohibited", config["cohort"]["best_output_selection_permitted"] is False),
            ("training is prohibited", config["data_boundaries"]["training_permitted"] is False),
            ("new SHAP is prohibited", config["data_boundaries"]["new_shap_evaluations_permitted"] is False),
            ("reserved test arrays remain closed", config["data_boundaries"]["reserved_test_arrays_materialized"] is False),
        ]
    )
    failures = [name for name, passed in checks if not passed]
    print("===== TASK 61 C2 PREFLIGHT =====")
    print(f"Checks passed: {len(checks) - len(failures)}/{len(checks)}")
    print("FROZEN C1 TAG VERIFIED:", not any("tag" in name for name in failures))
    print("PLANNED CASES: 12")
    print("PLANNED LLM REPORTS: 36")
    print("BEST OUTPUT SELECTION PERMITTED: False")
    print("TRAINING PERMITTED: False")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR C2 MULTIREPORT RUN:", not failures)
    if failures:
        raise RuntimeError("Preflight failures: " + "; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

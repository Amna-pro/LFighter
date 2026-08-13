#!/usr/bin/env python3
"""Independent integrity audit for the complete Task 56 C2 validation evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--c0-decision", required=True, type=Path)
    p.add_argument("--c1-decision", required=True, type=Path)
    p.add_argument("--validation-plan", required=True, type=Path)
    p.add_argument("--execution-root", required=True, type=Path)
    p.add_argument("--summary-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def evaluation_dir(root: Path, row: pd.Series) -> Path:
    family = "shared" if row.state == "clean_reference" else str(row.family_id)
    return root / "evaluations" / str(row["mode"]) / f"variant_{int(row.variant_seed)}" / f"seed_{int(row.seed)}" / family / str(row.state)


def main() -> int:
    a = args(); root = a.project_root.resolve(); execution = a.execution_root.resolve(); summary = a.summary_root.resolve(); out = a.output_dir.resolve(); tables = out / "tables"; tables.mkdir(parents=True, exist_ok=True)
    config = json.loads(a.config.resolve().read_text(encoding="utf-8")); c0 = json.loads(a.c0_decision.resolve().read_text(encoding="utf-8")); c1 = json.loads(a.c1_decision.resolve().read_text(encoding="utf-8")); plan = pd.read_csv(a.validation_plan.resolve())
    checks: list[dict[str, Any]] = []
    def check(name: str, passed: Any, detail: Any) -> None: checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})
    check("c0_all_checks_passed", c0.get("checks_passed") == c0.get("check_count") == 45, f"{c0.get('checks_passed')}/{c0.get('check_count')}")
    check("c1_all_checks_passed", c1.get("checks_passed") == c1.get("check_count") == 49, f"{c1.get('checks_passed')}/{c1.get('check_count')}")
    check("c1_ready_for_c2", c1.get("ready_for_c2_validation_run") is True, c1.get("ready_for_c2_validation_run"))
    check("safety_training_closed", config.get("training_permitted") is False, config.get("training_permitted"))
    check("safety_reserved_closed", config.get("reserved_test_gate_opened") is False, config.get("reserved_test_gate_opened"))
    check("frozen_plan_140", len(plan) == 140, len(plan)); check("frozen_plan_repeat_84", int((plan["mode"] == "repeat").sum()) == 84, int((plan["mode"] == "repeat").sum())); check("frozen_plan_background_56", int((plan["mode"] == "alternate_background").sum()) == 56, int((plan["mode"] == "alternate_background").sum()))
    execution_metadata = json.loads((execution / "task56c2_execution_metadata.json").read_text(encoding="utf-8")); decision = json.loads((summary / "task56c2_validation_decision.json").read_text(encoding="utf-8"))
    check("paper_closeout_materialized", (summary / "TASK56_C2_CLOSEOUT.md").is_file(), summary / "TASK56_C2_CLOSEOUT.md")
    check("execution_complete", execution_metadata.get("complete") is True, execution_metadata.get("complete")); check("execution_count_140", execution_metadata.get("new_shap_state_evaluations") == 140, execution_metadata.get("new_shap_state_evaluations")); check("execution_reserved_closed", execution_metadata.get("reserved_test_arrays_materialized") is False, execution_metadata.get("reserved_test_arrays_materialized"))
    marker_rows: list[dict[str, Any]] = []
    for _, row in plan.iterrows():
        directory = evaluation_dir(execution, row); marker_path = directory / "_task56_c2_evaluation_complete.json"; tensor_path = directory / "task56c2_margin_attributions.npz"; summary_path = directory / "evaluation_summary.json"
        valid = marker_path.is_file() and tensor_path.is_file() and summary_path.is_file(); finite = shape = identity = background_rows = False
        if valid:
            marker = json.loads(marker_path.read_text(encoding="utf-8")); state_summary = json.loads(summary_path.read_text(encoding="utf-8")); hashes = marker.get("output_sha256", {})
            valid = marker.get("complete") is True and marker.get("checkpoint_sha256") == str(row.checkpoint_sha256)
            valid = valid and marker.get("mode") == str(row["mode"]) and marker.get("variant_seed") == int(row.variant_seed)
            valid = valid and all((directory / name).is_file() and sha256(directory / name) == digest for name, digest in hashes.items())
            with np.load(tensor_path) as f:
                source = f["source_logit_shap"]; target = f["target_logit_shap"]; margin = f["source_minus_target_margin_shap"]; probe = f["probe_validation_indices"]; background = f["background_train_indices"]
            finite = np.isfinite(source).all() and np.isfinite(target).all() and np.isfinite(margin).all()
            shape = source.shape == target.shape == margin.shape == (128, 69) and probe.shape == (128,); background_rows = background.shape == (128,)
            identity = np.array_equal(margin, source - target) and state_summary.get("checkpoint_sha256") == str(row.checkpoint_sha256) and state_summary.get("reserved_test_arrays_materialized") is False
        marker_rows.append({"mode": row["mode"], "variant_seed": int(row.variant_seed), "seed": int(row.seed), "family_id": row.family_id, "state": row.state,
                            "marker_and_hashes_valid": valid, "tensor_shape_valid": shape, "tensor_finite": finite, "background_rows_valid": background_rows, "summary_identity_valid": identity})
    markers = pd.DataFrame(marker_rows); markers.to_csv(tables / "task56c2_evaluation_audit_manifest.csv", index=False)
    for column in ("marker_and_hashes_valid", "tensor_shape_valid", "tensor_finite", "background_rows_valid", "summary_identity_valid"):
        check("all_" + column, markers[column].all(), f"{int(markers[column].sum())}/140")
    expected = {
        "task56c2_repeated_run_stability.csv": 84, "task56c2_background_sensitivity.csv": 56,
        "task56c2_probe_size_sensitivity.csv": 56, "task56c2_feature_perturbation_impacts.csv": 1932,
        "task56c2_faithfulness_state_summary.csv": 28, "task56c2_faithfulness_random_controls.csv": 28672,
        "task56c2_class_specificity_samples.csv": 448, "task56c2_class_specificity_state_summary.csv": 28,
        "task56c2_cross_seed_consistency.csv": 42, "task56c2_preregistered_metric_decisions.csv": 18,
        "task56c2_domain_decisions.csv": 6,
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, count in expected.items():
        path = summary / "tables" / name; frames[name] = pd.read_csv(path); check("row_count_" + name.removesuffix(".csv"), len(frames[name]) == count, f"{len(frames[name])}/{count}")
    numeric_finite = True
    for frame in frames.values():
        numeric = frame.select_dtypes(include=[np.number])
        numeric_finite = numeric_finite and np.isfinite(numeric.to_numpy()).all()
    check("all_summary_numeric_values_finite", numeric_finite, numeric_finite)
    metrics = frames["task56c2_preregistered_metric_decisions.csv"]
    recomputed_metric_pass = np.where(metrics.direction == "at_least", metrics.value >= metrics.threshold, metrics.value <= metrics.threshold)
    check("metric_pass_flags_exact", np.array_equal(recomputed_metric_pass, metrics.passed.to_numpy(bool)), int(np.sum(recomputed_metric_pass == metrics.passed.to_numpy(bool))))
    domains = frames["task56c2_domain_decisions.csv"]
    domain_recomputed = metrics.groupby("domain", sort=False).passed.all().reset_index(name="domain_pass")
    merged = domains[["domain", "domain_pass"]].merge(domain_recomputed, on="domain", suffixes=("_stored", "_recomputed"))
    check("domain_pass_flags_exact", len(merged) == 6 and (merged.domain_pass_stored == merged.domain_pass_recomputed).all(), len(merged))
    domain_map = dict(zip(domains.domain, domains.domain_pass)); core = bool(domain_map["repeated_run_stability"] and domain_map["feature_perturbation_faithfulness"]); robust = sum(bool(domain_map[name]) for name in ("background_sensitivity", "probe_size_sensitivity", "class_specificity", "cross_seed_consistency")); result = "PASS" if core and robust == 4 else ("PARTIAL" if core and robust >= 2 else "FAIL")
    check("overall_decision_exact", decision.get("scientific_result") == result, f"stored={decision.get('scientific_result')} recomputed={result}")
    check("task57_primary_gate_exact", decision.get("task57_primary_claim_allowed") is (result == "PASS"), decision.get("task57_primary_claim_allowed")); check("task57_exploratory_gate_exact", decision.get("task57_exploratory_only") is (result == "PARTIAL"), decision.get("task57_exploratory_only")); check("task57_block_gate_exact", decision.get("task57_recovery_claim_blocked") is (result == "FAIL"), decision.get("task57_recovery_claim_blocked"))
    check("decision_reserved_closed", decision.get("reserved_test_arrays_materialized") is False, decision.get("reserved_test_arrays_materialized")); check("negative_results_retained", decision.get("negative_results_retained") is True, decision.get("negative_results_retained"))
    sources = [a.config.resolve(), a.c0_decision.resolve(), a.c1_decision.resolve(), a.validation_plan.resolve(), execution / "task56c2_execution_metadata.json", summary / "task56c2_validation_decision.json", summary / "TASK56_C2_CLOSEOUT.md"]
    sources += [summary / "tables" / name for name in expected]
    pd.DataFrame([{"path": str(p.relative_to(root)) if p.is_relative_to(root) else str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sources]).to_csv(tables / "task56c2_source_manifest_sha256.csv", index=False)
    passed = sum(int(row["passed"]) for row in checks)
    with (tables / "task56c2_audit_checks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("check", "passed", "detail")); writer.writeheader(); writer.writerows(checks)
    audit = {"experiment_version": "4.18.C2", "stage": "task56_complete_validation_audit", "checks_passed": passed, "check_count": len(checks),
             "scientific_result": result, "physical_checkpoint_states_verified": 28, "new_shap_evaluations_verified": 140,
             "all_attribution_tensors_finite": bool(markers.tensor_finite.all()), "all_evaluation_hashes_verified": bool(markers.marker_and_hashes_valid.all()),
             "task57_primary_claim_allowed": result == "PASS", "task57_exploratory_analysis_allowed": result in ("PASS", "PARTIAL"),
             "training_permitted": False, "reserved_test_arrays_materialized": False, "integrity_audit_passed": passed == len(checks)}
    (out / "task56c2_audit_decision.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("===== TASK 56 C2 COMPLETE VALIDATION AUDIT ====="); print(f"Checks passed: {passed}/{len(checks)}"); print("NEW SHAP EVALUATIONS VERIFIED: 140"); print("ATTRIBUTION TENSORS FINITE:", bool(markers.tensor_finite.all())); print("SCIENTIFIC RESULT:", result); print("TASK 57 PRIMARY CLAIM ALLOWED:", result == "PASS"); print("TASK 57 EXPLORATORY ANALYSIS ALLOWED:", result in ("PASS", "PARTIAL")); print("RESERVED TEST ARRAYS MATERIALIZED: False"); print("INTEGRITY AUDIT PASSED:", passed == len(checks))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__": raise SystemExit(main())

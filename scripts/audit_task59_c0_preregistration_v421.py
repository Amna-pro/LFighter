from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

PARENT_TAG = "task58-c1-xai-publication-frozen-v4201"
PARENT_COMMIT_SHORT = "2d13ee9"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="results/cic_iot_diad_task59_c0_preregistration_v421")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = (root / args.output_root).resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    cfg_path = root / "configs/task59_preregistration_v4210.json"
    md_path = root / "configs/TASK59_PREREGISTRATION_V421.md"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append((name, bool(passed), str(detail)))

    add("config_exists", cfg_path.is_file(), cfg_path)
    add("narrative_exists", md_path.is_file(), md_path)
    add("task", cfg.get("task") == 59, cfg.get("task"))
    add("version", cfg.get("experiment_version") == "4.21.0", cfg.get("experiment_version"))
    add("parent_tag", cfg.get("parent_tag") == PARENT_TAG, cfg.get("parent_tag"))
    add("parent_commit", cfg.get("parent_commit_short") == PARENT_COMMIT_SHORT, cfg.get("parent_commit_short"))
    add("schema_standard", cfg.get("schema_standard") == "https://json-schema.org/draft/2020-12/schema", cfg.get("schema_standard"))
    add("schema_version", cfg.get("schema_version") == "1.0.0", cfg.get("schema_version"))
    add("no_llm_calls", cfg.get("llm_calls_permitted") is False, cfg.get("llm_calls_permitted"))
    add("no_training", cfg.get("training_permitted") is False, cfg.get("training_permitted"))
    add("no_new_shap", cfg.get("new_shap_evaluations_permitted") is False, cfg.get("new_shap_evaluations_permitted"))
    add("reserved_closed", cfg.get("reserved_test_arrays_permitted") is False, cfg.get("reserved_test_arrays_permitted"))
    required_domains = cfg.get("required_domains", [])
    expected_domains = {"case_selection", "experiment", "attack", "detection", "reconstruction", "behavior", "xai", "observed_facts", "interpretations", "uncertainty", "authority", "provenance"}
    add("required_domains", set(required_domains) == expected_domains, required_domains)
    authority = cfg.get("authority_contract", {})
    add("report_only", authority.get("llm_role") == "report_only", authority)
    add("deterministic_authority", authority.get("decision_authority") == "deterministic_pipeline", authority)
    for field in ["can_flag_clients", "can_change_thresholds", "can_change_trust_weights", "can_reconstruct_updates", "can_control_aggregation", "can_claim_malicious_intent"]:
        add(f"authority_{field}", authority.get(field) is False, authority.get(field))
    grounding = cfg.get("grounding_rules", {})
    for field in ["all_observed_facts_require_evidence_references", "all_interpretations_require_basis_references", "facts_and_interpretations_are_separate", "missing_values_are_explicit", "unknown_threshold_values_must_not_be_invented", "source_files_require_sha256", "additional_properties_rejected", "control_characters_rejected"]:
        add(f"grounding_{field}", grounding.get(field) is True, grounding.get(field))
    case = cfg.get("expected_case", {})
    add("case_seed", case.get("seed") == 7, case)
    add("case_family", case.get("family_id") == "B_hash_ranked", case)
    add("case_size", case.get("coalition_size") == 10, case)
    add("case_pair", case.get("source_class") == "DDoS" and case.get("target_class") == "Benign", case)
    negative = cfg.get("required_negative_tests", [])
    add("negative_tests", set(negative) == {"authority_escalation", "unknown_property", "out_of_range_rate", "missing_provenance", "control_character_injection"}, negative)
    try:
        commit = subprocess.check_output(["git", "rev-list", "-n", "1", PARENT_TAG], cwd=root, text=True).strip()
        add("parent_tag_resolves", commit.startswith(PARENT_COMMIT_SHORT), commit)
    except Exception as exc:
        add("parent_tag_resolves", False, exc)
    with (tables / "task59c0_preregistration_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check", "passed", "detail"])
        writer.writerows(checks)
    with (tables / "task59c0_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        for path in [cfg_path, md_path]:
            writer.writerow([path.relative_to(root).as_posix(), path.stat().st_size, digest(path)])
    passed = sum(x[1] for x in checks)
    decision = {"experiment_version": "4.21.0", "checks_passed": passed, "checks_total": len(checks), "all_checks_passed": passed == len(checks), "llm_calls": 0, "training_permitted": False, "new_shap_evaluations": 0, "reserved_test_arrays_materialized": False, "ready_for_c1_schema": passed == len(checks)}
    (output / "task59c0_preregistration_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print("===== TASK 59 C0 SCHEMA PREREGISTRATION AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("LLM CALLS: 0")
    print("TRAINING PERMITTED: False")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR C1 SCHEMA:", passed == len(checks))
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

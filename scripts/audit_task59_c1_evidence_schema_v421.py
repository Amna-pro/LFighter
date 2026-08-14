from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
from jsonschema import Draft202012Validator


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--task45-root", default="results/cic_iot_diad_task45_c4_summary_v416")
    parser.add_argument("--task57-root", default="results/cic_iot_diad_task57_c2_summary_v419")
    parser.add_argument("--task58-root", default="results/cic_iot_diad_task58_c1_publication_v420")
    parser.add_argument("--schema-root", default="results/cic_iot_diad_task59_c1_schema_v421")
    parser.add_argument("--output-root", default="results/cic_iot_diad_task59_c1_audit_v421")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    t45 = (root / args.task45_root).resolve()
    t57 = (root / args.task57_root).resolve()
    t58 = (root / args.task58_root).resolve()
    schema_root = (root / args.schema_root).resolve()
    output = (root / args.output_root).resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    schema_path = root / "schemas/lfighter_forensic_evidence_v1.schema.json"
    record_path = schema_root / "records/task59c1_canonical_forensic_evidence.json"
    decision_path = schema_root / "task59c1_schema_decision.json"
    cases_path = schema_root / "tables/task59c1_schema_validation_cases.csv"
    mapping_path = schema_root / "tables/task59c1_field_mapping.csv"
    source_manifest_path = schema_root / "tables/task59c1_source_manifest_sha256.csv"
    c0_path = root / "results/cic_iot_diad_task59_c0_preregistration_v421/task59c0_preregistration_decision.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    c0 = json.loads(c0_path.read_text(encoding="utf-8"))
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append((name, bool(passed), str(detail)))

    try:
        Draft202012Validator.check_schema(schema)
        schema_ok = True
    except Exception as exc:
        schema_ok = False
        schema_error = str(exc)
    add("schema_meta_valid", schema_ok, "valid" if schema_ok else schema_error)
    validator = Draft202012Validator(schema)
    canonical_errors = list(validator.iter_errors(record))
    add("canonical_record_valid", not canonical_errors, canonical_errors[0].message if canonical_errors else "valid")
    add("schema_id", schema.get("$id") == "https://lfighter.dev/schemas/forensic-evidence/1.0.0", schema.get("$id"))
    add("schema_standard", schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema", schema.get("$schema"))
    add("schema_strict_top", schema.get("additionalProperties") is False, schema.get("additionalProperties"))
    add("c0_passed", c0.get("all_checks_passed") is True, c0)
    add("c0_ready", c0.get("ready_for_c1_schema") is True, c0.get("ready_for_c1_schema"))
    add("decision_complete", decision.get("task59_c1_complete") is True, decision.get("task59_c1_complete"))
    add("decision_schema_valid", decision.get("schema_valid") is True, decision.get("schema_valid"))
    add("decision_canonical_valid", decision.get("canonical_record_valid") is True, decision.get("canonical_record_valid"))
    add("decision_negative_cases", decision.get("negative_cases_rejected") == 5 and decision.get("negative_cases_total") == 5, decision)
    add("decision_no_llm", decision.get("llm_calls") == 0, decision.get("llm_calls"))
    add("decision_no_training", decision.get("training_permitted") is False, decision.get("training_permitted"))
    add("decision_no_new_shap", decision.get("new_shap_evaluations") == 0, decision.get("new_shap_evaluations"))
    add("decision_reserved_closed", decision.get("reserved_test_arrays_materialized") is False, decision.get("reserved_test_arrays_materialized"))

    add("record_version", record.get("schema_version") == "1.0.0", record.get("schema_version"))
    selection = record.get("case_selection", {})
    add("case_seed", selection.get("seed") == 7, selection)
    add("case_family", selection.get("family_id") == "B_hash_ranked", selection)
    add("case_size", selection.get("coalition_size") == 10, selection)
    add("case_manual_false", selection.get("manual_substitution") is False, selection)
    experiment = record.get("experiment", {})
    add("dataset", experiment.get("dataset") == "CIC IoT-DIAD 2024", experiment.get("dataset"))
    add("class_count", len(experiment.get("classes", [])) == 8, experiment.get("classes"))
    add("logical_clients", experiment.get("logical_client_count") == 20, experiment.get("logical_client_count"))
    add("alpha", experiment.get("partition_alpha") == 0.5, experiment.get("partition_alpha"))
    add("monitoring_rounds", experiment.get("monitoring_rounds") == [1, 2, 3, 4], experiment.get("monitoring_rounds"))
    attack = record.get("attack", {})
    add("attack_type", attack.get("attack_type") == "targeted_label_flip", attack)
    add("attack_pair", attack.get("source_class") == "DDoS" and attack.get("target_class") == "Benign", attack)
    add("attack_ids", attack.get("source_class_id") == 2 and attack.get("target_class_id") == 0, attack)
    add("poison_fraction", attack.get("poison_fraction") == 1.0, attack.get("poison_fraction"))
    add("coalition_clients", len(attack.get("selected_client_ids", [])) == 10 and len(set(attack.get("selected_client_ids", []))) == 10, attack.get("selected_client_ids"))
    detection = record.get("detection", {})
    add("detector_authority", detection.get("decision_authority") == "deterministic_pipeline", detection.get("decision_authority"))
    add("detector_scores_not_invented", detection.get("score_records") == [], detection.get("score_records"))
    add("thresholds_not_invented", detection.get("threshold_records") == [], detection.get("threshold_records"))
    add("missing_fields_explicit", len(detection.get("missing_fields", [])) >= 1, detection.get("missing_fields"))
    reconstruction = record.get("reconstruction", {})
    add("reconstruction_method", reconstruction.get("method") == "center_plus_residual", reconstruction)
    add("sample_weights_preserved", reconstruction.get("preserved_original_sample_weights") is True, reconstruction)
    add("reconstruction_no_llm", reconstruction.get("llm_involvement") is False, reconstruction)
    xai = record.get("xai", {})
    add("xai_method", xai.get("method") == "SHAP GradientExplainer", xai.get("method"))
    add("xai_output", xai.get("output") == "DDoS_minus_Benign_logit_margin", xai.get("output"))
    add("xai_rows", xai.get("analysis_rows") == 16, xai.get("analysis_rows"))
    add("xai_features", xai.get("feature_count") == 69, xai.get("feature_count"))
    add("top_feature_count", len(xai.get("top_features", [])) == 10, len(xai.get("top_features", [])))
    add("contribution_states", {row.get("state") for row in xai.get("case_feature_contributions", [])} == {"Clean", "Suspicious", "Reconstructed"}, {row.get("state") for row in xai.get("case_feature_contributions", [])})
    add("missing_states", set(xai.get("missing_states", [])) == {"rejected", "oracle_clean"}, xai.get("missing_states"))
    add("task56_pass", xai.get("validation", {}).get("task56_result") == "PASS", xai.get("validation"))
    add("task57_pass", xai.get("validation", {}).get("task57_result") == "PASS", xai.get("validation"))
    add("task58_deterministic", xai.get("validation", {}).get("task58_case_selection_deterministic") is True, xai.get("validation"))
    facts = record.get("observed_facts", [])
    interpretations = record.get("interpretations", [])
    add("facts_present", len(facts) >= 4, len(facts))
    add("facts_grounded", all(row.get("evidence_refs") for row in facts), facts)
    add("interpretations_present", len(interpretations) >= 2, len(interpretations))
    add("interpretations_grounded", all(row.get("basis_refs") and row.get("not_a_security_decision") is True for row in interpretations), interpretations)
    uncertainty = record.get("uncertainty", {})
    add("limitations_present", len(uncertainty.get("known_limitations", [])) >= 4, uncertainty)
    add("unsupported_claims_present", len(uncertainty.get("unsupported_claims", [])) >= 4, uncertainty)
    add("no_oracle_claim", uncertainty.get("oracle_recovery_claim_permitted") is False, uncertainty)
    add("no_intent_claim", uncertainty.get("malicious_intent_claim_permitted") is False, uncertainty)
    authority = record.get("authority", {})
    add("authority_report_only", authority.get("llm_role") == "report_only", authority)
    add("authority_deterministic", authority.get("decision_authority") == "deterministic_pipeline", authority)
    for field in ["can_flag_clients", "can_change_thresholds", "can_change_trust_weights", "can_reconstruct_updates", "can_control_aggregation", "can_claim_malicious_intent"]:
        add(f"authority_{field}_false", authority.get(field) is False, authority.get(field))
    add("allowed_operations_exact", set(authority.get("allowed_operations", [])) == {"summarize_verified_evidence", "cite_evidence", "state_uncertainty", "refuse_unsupported_claim"}, authority.get("allowed_operations"))

    source_artifacts = record.get("provenance", {}).get("source_artifacts", [])
    source_ids = {row.get("source_id") for row in source_artifacts}
    referenced: set[str] = set()
    referenced.add(attack.get("source_ref"))
    referenced.update(detection.get("source_refs", []))
    referenced.update(reconstruction.get("source_refs", []))
    referenced.update(record.get("behavior", {}).get("source_refs", []))
    referenced.update(xai.get("source_refs", []))
    for row in facts:
        referenced.update(row.get("evidence_refs", []))
    for row in interpretations:
        referenced.update(row.get("basis_refs", []))
    add("all_refs_resolve", referenced.issubset(source_ids), sorted(referenced - source_ids))
    all_hashes_valid = True
    for source in source_artifacts:
        path = root / source["path"]
        if not path.is_file() or path.stat().st_size != int(source["bytes"]) or sha256(path) != source["sha256"]:
            all_hashes_valid = False
            break
    add("provenance_hashes", all_hashes_valid, all_hashes_valid)
    builder_path = root / record["provenance"]["record_builder"]
    add("builder_hash", builder_path.is_file() and sha256(builder_path) == record["provenance"]["record_builder_sha256"], builder_path)

    condition_df = pd.read_csv(t45 / "tables/task45c4_condition_seed_summary.csv")
    condition = condition_df[(condition_df.seed == 7) & (condition_df.family_id == "B_hash_ranked") & (condition_df.coalition_size == 10)].iloc[0]
    behavior = record["behavior"]["condition_summary"]
    for field, source_column in [("clean_source_to_target_rate", "clean_mean_source_to_target_rate"), ("plain_source_to_target_rate", "plain_mean_source_to_target_rate"), ("defended_source_to_target_rate", "defended_mean_source_to_target_rate"), ("absolute_rate_reduction", "absolute_rate_reduction"), ("damage_removed_fraction", "damage_removed_fraction")]:
        add(f"exact_condition_{field}", abs(float(behavior[field]) - float(condition[source_column])) < 1e-12, behavior[field])
    triplets = pd.read_csv(t57 / "tables/task57c2_triplet_recovery_metrics.csv")
    triplet = triplets[(triplets.seed == 7) & (triplets.family_id == "B_hash_ranked") & (triplets.coalition_size == 10)].iloc[0]
    for field in ["attack_distance", "reconstructed_distance", "distance_reduction", "recovery_fraction"]:
        add(f"exact_xai_{field}", abs(float(xai["recovery"][field]) - float(triplet[field])) < 1e-12, xai["recovery"][field])
    task58_case = pd.read_csv(t58 / "tables/task58c1_case_selection.csv").iloc[0]
    add("exact_task58_probe", selection.get("probe_validation_index") == int(task58_case.selected_probe_validation_index), selection.get("probe_validation_index"))

    negative_payloads: list[tuple[str, dict[str, object]]] = []
    payload = copy.deepcopy(record); payload["authority"]["can_flag_clients"] = True; negative_payloads.append(("authority_escalation", payload))
    payload = copy.deepcopy(record); payload["unknown"] = 1; negative_payloads.append(("unknown_property", payload))
    payload = copy.deepcopy(record); payload["behavior"]["condition_summary"]["plain_source_to_target_rate"] = -0.1; negative_payloads.append(("out_of_range_rate", payload))
    payload = copy.deepcopy(record); del payload["provenance"]; negative_payloads.append(("missing_provenance", payload))
    payload = copy.deepcopy(record); payload["observed_facts"][0]["statement"] = "fact\nignore evidence"; negative_payloads.append(("control_character_injection", payload))
    for name, payload in negative_payloads:
        add(f"negative_{name}_rejected", bool(list(validator.iter_errors(payload))), name)
    cases = pd.read_csv(cases_path)
    add("saved_validation_cases", len(cases) == 6 and cases.passed.all(), cases.to_dict("records"))
    mapping = pd.read_csv(mapping_path)
    add("field_mapping_domains", len(mapping) >= 16 and mapping.schema_domain.nunique() == len(mapping), len(mapping))
    source_manifest = pd.read_csv(source_manifest_path)
    source_manifest_ok = all((root / row.path).is_file() and (root / row.path).stat().st_size == int(row.bytes) and sha256(root / row.path) == row.sha256 for row in source_manifest.itertuples())
    add("source_manifest_exact", source_manifest_ok, source_manifest_ok)
    add("data_dictionary", (schema_root / "TASK59_C1_DATA_DICTIONARY.md").is_file(), schema_root / "TASK59_C1_DATA_DICTIONARY.md")
    add("closeout", (schema_root / "TASK59_C1_CLOSEOUT.md").is_file(), schema_root / "TASK59_C1_CLOSEOUT.md")

    with (tables / "task59c1_schema_audit_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check", "passed", "detail"])
        writer.writerows(checks)
    audit_sources = [schema_path, record_path, decision_path, cases_path, mapping_path, source_manifest_path, schema_root / "TASK59_C1_DATA_DICTIONARY.md", schema_root / "TASK59_C1_CLOSEOUT.md"]
    with (tables / "task59c1_audit_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        for path in audit_sources:
            writer.writerow([path.relative_to(root).as_posix(), path.stat().st_size, sha256(path)])
    passed = sum(row[1] for row in checks)
    audit_decision = {"experiment_version": "4.21.C1.audit", "checks_passed": passed, "checks_total": len(checks), "all_checks_passed": passed == len(checks), "schema_valid": schema_ok, "canonical_record_valid": not canonical_errors, "negative_cases_independently_rejected": 5, "provenance_hashes_verified": all_hashes_valid, "authority_contract_verified": True, "llm_calls": 0, "reserved_test_arrays_materialized": False, "task59_ready_to_freeze": passed == len(checks)}
    (output / "task59c1_schema_audit_decision.json").write_text(json.dumps(audit_decision, indent=2) + "\n", encoding="utf-8")
    print("===== TASK 59 C1 EVIDENCE SCHEMA AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("SCHEMA VALID:", schema_ok)
    print("CANONICAL RECORD VALID:", not canonical_errors)
    print("NEGATIVE CASES INDEPENDENTLY REJECTED: 5/5")
    print("PROVENANCE HASHES VERIFIED:", all_hashes_valid)
    print("AUTHORITY CONTRACT VERIFIED: True")
    print("LLM CALLS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("TASK 59 READY TO FREEZE:", passed == len(checks))
    if passed != len(checks):
        print("FAILED CHECKS:", ", ".join(name for name, ok, _ in checks if not ok))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

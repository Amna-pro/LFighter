from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "task61_preregistration_v4230.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def as_int_list(value: Any) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", str(value))]


def f6(value: Any) -> str:
    return f"{float(value):.6f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# {report['report_title']}", "", report["executive_summary"], "", "## Observed facts", ""]
    for item in report["facts"]:
        lines.append(f"* {item['statement']} [{', '.join(item['evidence_refs'])}]")
    lines.extend(["", "## Interpretation", ""])
    for item in report["interpretations"]:
        lines.append(f"* {item['statement']} Confidence: {item['confidence']}. [{', '.join(item['evidence_refs'])}]")
    lines.extend(["", "## Uncertainty", ""])
    lines.extend(f"* {item}" for item in report["uncertainties"])
    lines.extend(["", "## Unsupported claims refused", ""])
    lines.extend(f"* {item}" for item in report["refusals"])
    lines.extend(["", "## Authority", "", report["authority_statement"], ""])
    return "\n".join(lines)


def main() -> int:
    c = load_json(CONFIG_PATH)
    cohort = c["cohort"]
    out = ROOT / c["outputs"]["c1_root"]
    evidence_root = out / "evidence"
    template_root = out / "deterministic_reports"
    tables_root = out / "tables"
    evidence_root.mkdir(parents=True, exist_ok=True)
    template_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    conditions = pd.read_csv(ROOT / c["inputs"]["task45_conditions"])
    rounds = pd.read_csv(ROOT / c["inputs"]["task45_rounds"])
    triplets = pd.read_csv(ROOT / c["inputs"]["task57_triplets"])
    features = pd.read_csv(ROOT / c["inputs"]["task57_features"])
    task56 = load_json(ROOT / c["inputs"]["task56_validation"])
    task57 = load_json(ROOT / c["inputs"]["task57_decision"])

    conditions = conditions[
        conditions["seed"].isin(cohort["seeds"])
        & conditions["family_id"].isin(cohort["families"])
        & (conditions["coalition_size"] == cohort["coalition_size"])
    ].copy()
    rounds = rounds[
        rounds["seed"].isin(cohort["seeds"])
        & rounds["family_id"].isin(cohort["families"])
        & (rounds["coalition_size"] == cohort["coalition_size"])
    ].copy()
    triplets = triplets[
        triplets["seed"].isin(cohort["seeds"])
        & triplets["family_id"].isin(cohort["families"])
        & (triplets["coalition_size"] == cohort["coalition_size"])
    ].copy()
    conditions = conditions.sort_values(["seed", "family_id"]).reset_index(drop=True)
    triplets = triplets.sort_values(["seed", "family_id"]).reset_index(drop=True)

    source_specs = [
        ("src:task45_condition", c["inputs"]["task45_conditions"]),
        ("src:task45_rounds", c["inputs"]["task45_rounds"]),
        ("src:task56_validation", c["inputs"]["task56_validation"]),
        ("src:task57_decision", c["inputs"]["task57_decision"]),
        ("src:task57_triplets", c["inputs"]["task57_triplets"]),
        ("src:task57_features", c["inputs"]["task57_features"]),
        ("src:task60_audit", c["inputs"]["task60_audit"]),
    ]
    sources = [
        {"source_id": source_id, "path": path, "sha256": sha256(ROOT / path)}
        for source_id, path in source_specs
    ]
    top_features = []
    for _, row in features.head(5).iterrows():
        top_features.append(
            {
                "rank": int(row["rank"]),
                "feature": str(row["feature"]),
                "majority_direction": str(row["majority_direction"]),
                "same_direction_fraction": float(row["same_direction_fraction"]),
                "median_absolute_attack_shift": float(row["median_absolute_attack_shift"]),
                "median_absolute_reconstructed_shift": float(row["median_absolute_reconstructed_shift"]),
                "recovery_fraction": float(row["feature_recovery_fraction"]),
                "source_ref": "src:task57_features",
            }
        )

    case_manifest: list[dict[str, Any]] = []
    call_plan: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for _, trow in triplets.iterrows():
        seed = int(trow["seed"])
        family = str(trow["family_id"])
        matched = conditions[(conditions["seed"] == seed) & (conditions["family_id"] == family)]
        if len(matched) != 1:
            raise RuntimeError(f"Expected one Task 45 condition for seed {seed} and family {family}, found {len(matched)}")
        crow = matched.iloc[0]
        matched_rounds = rounds[(rounds["seed"] == seed) & (rounds["family_id"] == family)].sort_values("monitoring_round")
        if len(matched_rounds) != 4:
            raise RuntimeError(f"Expected four rounds for seed {seed} and family {family}")
        case_id = f"seed_{seed}__{family}__size_{cohort['coalition_size']}"
        record_id = f"lfighter:task61:seed{seed}:{family}:size{cohort['coalition_size']}"
        round_records = []
        for _, row in matched_rounds.iterrows():
            round_records.append(
                {
                    "round": int(row["monitoring_round"]),
                    "clean_source_to_target_rate": float(row["clean_source_to_target_rate"]),
                    "plain_source_to_target_rate": float(row["plain_source_to_target_rate"]),
                    "defended_source_to_target_rate": float(row["defended_source_to_target_rate"]),
                    "malicious_recall": float(row["malicious_recall"]),
                    "benign_fpr": float(row["benign_fpr"]),
                    "defense_improved_vs_plain": as_bool(row["defense_improved_vs_plain"]),
                    "source_ref": "src:task45_rounds",
                }
            )
        facts = [
            {
                "fact_id": "fact.behavior_rate",
                "statement": (
                    f"The plain attack mean DDoS to Benign prediction rate was {f6(crow['plain_mean_source_to_target_rate'])} "
                    f"versus {f6(crow['clean_mean_source_to_target_rate'])} in the paired clean branch."
                ),
                "evidence_refs": ["src:task45_condition"],
            },
            {
                "fact_id": "fact.detector_performance",
                "statement": (
                    f"Mean malicious recall was {f6(crow['mean_malicious_recall'])}; minimum malicious recall was "
                    f"{f6(crow['minimum_malicious_recall'])}; maximum benign false positive rate was {f6(crow['maximum_benign_fpr'])}."
                ),
                "evidence_refs": ["src:task45_condition"],
            },
            {
                "fact_id": "fact.attribution_recovery",
                "statement": (
                    f"Normalized attribution distance decreased from {f6(trow['attack_distance'])} to "
                    f"{f6(trow['reconstructed_distance'])}, with recovery fraction {f6(trow['recovery_fraction'])}."
                ),
                "evidence_refs": ["src:task57_triplets"],
            },
            {
                "fact_id": "fact.defense_effect",
                "statement": (
                    f"The defended mean DDoS to Benign prediction rate was {f6(crow['defended_mean_source_to_target_rate'])}, "
                    f"an absolute reduction of {f6(crow['absolute_rate_reduction'])} from the plain attack branch."
                ),
                "evidence_refs": ["src:task45_condition"],
            },
        ]
        interpretations = [
            {
                "interpretation_id": "interpretation.targeted_suppression",
                "statement": "The observed prediction behavior is consistent with targeted suppression of DDoS predictions toward the Benign class, but it does not establish malicious intent.",
                "confidence": "moderate",
                "basis_refs": ["src:task45_condition", "src:task45_rounds"],
                "not_a_security_decision": True,
            },
            {
                "interpretation_id": "interpretation.reference_recovery",
                "statement": "Reconstruction moved the attribution profile toward its paired preattack clean reference; this is not evidence of oracle clean restoration.",
                "confidence": "high",
                "basis_refs": ["src:task57_triplets", "src:task57_decision"],
                "not_a_security_decision": True,
            },
        ]
        record = {
            "schema_version": "1.0.0",
            "record_id": record_id,
            "case_id": case_id,
            "cohort_selection": {
                "rule": "complete Task 57 triplet cohort at coalition size 10",
                "seed": seed,
                "family_id": family,
                "coalition_size": int(cohort["coalition_size"]),
                "manual_substitution": False,
                "case_exclusion_permitted": False,
            },
            "experiment": {
                "dataset": "CIC IoT-DIAD 2024",
                "seed": seed,
                "development_and_validation_only": True,
            },
            "attack": {
                "attack_type": "targeted_label_flip",
                "source_class": "DDoS",
                "target_class": "Benign",
                "coalition_family": family,
                "coalition_size": int(cohort["coalition_size"]),
                "selected_client_ids": as_int_list(crow["selected_clients"]),
                "exact_poison_pair": as_bool(crow["exact_poison_pair"]),
                "source_ref": "src:task45_condition",
            },
            "detection": {
                "decision_authority": "deterministic_pipeline",
                "summary": {
                    "mean_malicious_recall": float(crow["mean_malicious_recall"]),
                    "minimum_malicious_recall": float(crow["minimum_malicious_recall"]),
                    "mean_benign_fpr": float(crow["mean_benign_fpr"]),
                    "maximum_benign_fpr": float(crow["maximum_benign_fpr"]),
                },
                "missing_fields": ["Per client detector scores and numeric detector thresholds were not materialized and are not invented."],
                "source_refs": ["src:task45_condition", "src:task45_rounds"],
            },
            "reconstruction": {
                "applied": True,
                "method": "center_plus_residual",
                "llm_involvement": False,
                "source_refs": ["src:task45_condition"],
            },
            "behavior": {
                "condition_summary": {
                    "clean_source_to_target_rate": float(crow["clean_mean_source_to_target_rate"]),
                    "plain_source_to_target_rate": float(crow["plain_mean_source_to_target_rate"]),
                    "defended_source_to_target_rate": float(crow["defended_mean_source_to_target_rate"]),
                    "absolute_rate_reduction": float(crow["absolute_rate_reduction"]),
                    "damage_removed_fraction": float(crow["damage_removed_fraction"]),
                    "clean_macro_f1": float(crow["clean_mean_macro_f1"]),
                    "plain_macro_f1": float(crow["plain_mean_macro_f1"]),
                    "defended_macro_f1": float(crow["defended_mean_macro_f1"]),
                },
                "round_metrics": round_records,
                "source_refs": ["src:task45_condition", "src:task45_rounds"],
            },
            "xai": {
                "method": "SHAP GradientExplainer",
                "output": "DDoS_minus_Benign_logit_margin",
                "recovery": {
                    "attack_distance": float(trow["attack_distance"]),
                    "reconstructed_distance": float(trow["reconstructed_distance"]),
                    "distance_reduction": float(trow["distance_reduction"]),
                    "recovery_fraction": float(trow["recovery_fraction"]),
                    "attack_exceeds_repeat_noise_q95": as_bool(trow["attack_exceeds_repeat_noise_q95"]),
                },
                "prediction_context": {
                    "clean_ddos_prediction_rate": float(trow["clean_ddos_prediction_rate"]),
                    "suspicious_ddos_prediction_rate": float(trow["suspicious_ddos_prediction_rate"]),
                    "reconstructed_ddos_prediction_rate": float(trow["reconstructed_ddos_prediction_rate"]),
                },
                "top_features": top_features,
                "validation": {
                    "task56_result": task56["scientific_result"],
                    "task57_result": task57["scientific_result"],
                    "task57_triplets_analyzed": int(task57["triplets_analyzed"]),
                },
                "missing_states": ["rejected", "oracle_clean"],
                "source_refs": ["src:task56_validation", "src:task57_decision", "src:task57_triplets", "src:task57_features"],
            },
            "observed_facts": facts,
            "interpretations": interpretations,
            "uncertainty": {
                "known_limitations": [
                    "The clean attribution comparator is the paired round 4 preattack reference rather than an oracle clean state.",
                    "Rejected and oracle clean attribution states were unavailable and were not substituted.",
                    "Per client detector scores and numeric threshold values were not present in the selected summary artifacts.",
                    "The evidence is development and validation evidence; reserved test arrays remained closed.",
                ],
                "unsupported_claims": [
                    "Verified malicious intent",
                    "Exact oracle clean recovery",
                    "Generalization to unseen organizations or physical clients",
                    "Authority for an LLM to change any defense decision",
                ],
            },
            "authority": {
                "llm_role": "report_only",
                "decision_authority": "deterministic_pipeline",
                "can_flag_clients": False,
                "can_change_thresholds": False,
                "can_change_trust_weights": False,
                "can_reconstruct_updates": False,
                "can_control_aggregation": False,
                "can_claim_malicious_intent": False,
            },
            "provenance": {"source_artifacts": sources},
        }
        evidence_path = evidence_root / f"{case_id}.json"
        write_json(evidence_path, record)
        report = {
            "schema_version": "1.0.0",
            "record_id": record_id,
            "report_title": "LFighter grounded forensic evidence report",
            "executive_summary": "This deterministic report summarizes the frozen development and validation evidence. Facts, interpretations, uncertainty, and unsupported claims remain separate.",
            "facts": [{"statement": x["statement"], "evidence_refs": x["evidence_refs"]} for x in facts],
            "interpretations": [
                {"statement": x["statement"], "confidence": x["confidence"], "evidence_refs": x["basis_refs"]}
                for x in interpretations
            ],
            "uncertainties": record["uncertainty"]["known_limitations"],
            "refusals": [f"Not supported by this evidence: {x}." for x in record["uncertainty"]["unsupported_claims"]],
            "authority_statement": "Informational report only; the deterministic LFighter pipeline retains all security decision authority.",
        }
        report_path = template_root / f"{case_id}.json"
        write_json(report_path, report)
        (template_root / f"{case_id}.md").write_text(render_markdown(report), encoding="utf-8")
        case_manifest.append(
            {
                "case_id": case_id,
                "record_id": record_id,
                "seed": seed,
                "family_id": family,
                "coalition_size": int(cohort["coalition_size"]),
                "evidence_path": evidence_path.relative_to(ROOT).as_posix(),
                "evidence_sha256": sha256(evidence_path),
                "deterministic_report_path": report_path.relative_to(ROOT).as_posix(),
                "deterministic_report_sha256": sha256(report_path),
            }
        )
        for repetition in range(1, int(cohort["llm_repetitions_per_case"]) + 1):
            call_plan.append(
                {
                    "call_index": len(call_plan) + 1,
                    "case_id": case_id,
                    "seed": seed,
                    "family_id": family,
                    "repetition": repetition,
                    "model": c["model"]["model_id"],
                    "reasoning_effort": c["model"]["reasoning_effort"],
                    "max_output_tokens": c["model"]["max_output_tokens"],
                    "planned_output_path": f"results/cic_iot_diad_task61_c2_llm_reports_v423/reports/{case_id}/repeat_{repetition}.json",
                }
            )
        answer_key.extend(
            [
                {"case_id": case_id, "question_id": "plain_rate", "answer": f6(crow["plain_mean_source_to_target_rate"]), "source_ref": "src:task45_condition"},
                {"case_id": case_id, "question_id": "clean_rate", "answer": f6(crow["clean_mean_source_to_target_rate"]), "source_ref": "src:task45_condition"},
                {"case_id": case_id, "question_id": "recovery_fraction", "answer": f6(trow["recovery_fraction"]), "source_ref": "src:task57_triplets"},
                {"case_id": case_id, "question_id": "malicious_intent_permitted", "answer": "false", "source_ref": "authority"},
                {"case_id": case_id, "question_id": "reserved_test_arrays_materialized", "answer": "false", "source_ref": "data_boundary"},
            ]
        )

    with (tables_root / "task61c1_case_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_manifest[0]))
        writer.writeheader()
        writer.writerows(case_manifest)
    with (tables_root / "task61c1_llm_call_plan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(call_plan[0]))
        writer.writeheader()
        writer.writerows(call_plan)
    with (tables_root / "task61c1_analyst_answer_key.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(answer_key[0]))
        writer.writeheader()
        writer.writerows(answer_key)

    rng = random.Random(int(c["human_evaluation"]["randomization_seed"]))
    review_items = []
    for case in case_manifest:
        for comparator, path, available in (
            ("raw_compact_evidence", case["evidence_path"], True),
            ("deterministic_template", case["deterministic_report_path"], True),
            ("llm_report_repeat_1", f"results/cic_iot_diad_task61_c2_llm_reports_v423/reports/{case['case_id']}/repeat_1.json", False),
        ):
            review_items.append({"case_id": case["case_id"], "comparator": comparator, "content_path": path, "available_at_c1": available})
    rng.shuffle(review_items)
    blinded_manifest = []
    reviewer_rows = []
    for index, item in enumerate(review_items, start=1):
        blind_id = f"item_{index:03d}"
        blinded_manifest.append({"display_order": index, "blinded_item_id": blind_id, **item})
        reviewer_rows.append(
            {
                "reviewer_id": "",
                "blinded_item_id": blind_id,
                "factual_accuracy_1_to_5": "",
                "clarity_1_to_5": "",
                "usefulness_1_to_5": "",
                "trustworthiness_1_to_5": "",
                "uncertainty_quality_1_to_5": "",
                "analyst_answers_correct_0_to_5": "",
                "completion_seconds": "",
                "unsupported_claim_count": "",
                "notes": "",
            }
        )
    with (tables_root / "task61c1_blinded_review_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blinded_manifest[0]))
        writer.writeheader()
        writer.writerows(blinded_manifest)
    with (tables_root / "task61c1_reviewer_score_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reviewer_rows[0]))
        writer.writeheader()
        writer.writerows(reviewer_rows)

    metadata = {
        "protocol_id": c["protocol_id"],
        "cases_materialized": len(case_manifest),
        "deterministic_reports_materialized": len(case_manifest),
        "planned_llm_reports": len(call_plan),
        "llm_calls": 0,
        "review_items": len(blinded_manifest),
        "minimum_independent_reviewers": c["human_evaluation"]["minimum_independent_reviewers"],
        "best_output_selection_permitted": False,
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
        "complete": True,
    }
    write_json(out / "task61c1_evidence_matrix_metadata.json", metadata)
    print("===== TASK 61 C1 MULTICASE EVIDENCE MATRIX =====")
    print("CASES MATERIALIZED:", len(case_manifest))
    print("DETERMINISTIC REPORTS:", len(case_manifest))
    print("PLANNED LLM REPORTS:", len(call_plan))
    print("LLM CALLS: 0")
    print("BLINDED REVIEW ITEMS:", len(blinded_manifest))
    print("TRAINING PERMITTED: False")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("TASK 61 C1 COMPLETE: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Independently audit Task 57 C2 attribution recovery outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PARENT_TAG = "task57-c1-attribution-recovery-preflight-frozen-v4191"
PARENT_COMMIT_SHORT = "e9a969d"
SEEDS = (7, 99, 123, 2026)
FAMILIES = ("A_development_anchor", "B_hash_ranked", "C_hash_ranked")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--c1-root", required=True, type=Path)
    parser.add_argument("--task55-c2-root", required=True, type=Path)
    parser.add_argument("--task55-c3-root", required=True, type=Path)
    parser.add_argument("--summary-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def baseline_path(task55_c2: Path, task55_c3: Path, seed: int, family: str, state: str) -> Path:
    if seed == 7:
        run_root = task55_c2
        prefix = "task55c2"
    else:
        run_root = task55_c3 / f"seed_{seed}"
        prefix = "task55c3"
    if state == "clean_reference":
        return run_root / "common" / "clean_reference" / f"{prefix}_attributions.npz"
    return run_root / "families" / family / state / f"{prefix}_attributions.npz"


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def top_k(profile: np.ndarray, names: np.ndarray, k: int) -> set[str]:
    order = sorted(range(len(profile)), key=lambda index: (-float(profile[index]), str(names[index])))
    return {str(names[index]) for index in order[:k]}


def normalized_l1(left: np.ndarray, right: np.ndarray, denominator: np.ndarray, epsilon: float) -> float:
    return float(np.abs(left - right).sum() / (float(np.abs(denominator).sum()) + epsilon))


def exact_sign_flip_p(values: np.ndarray) -> float:
    observed = float(np.mean(values))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistics.append(float(np.mean(values * np.asarray(signs))))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def hierarchical_bootstrap(frame: pd.DataFrame, seed: int, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    grouped = {
        int(seed_value): group["recovery_fraction"].to_numpy(dtype=np.float64)
        for seed_value, group in frame.groupby("seed", sort=True)
    }
    seed_values = np.asarray(sorted(grouped), dtype=np.int64)
    result = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.choice(seed_values, size=len(seed_values), replace=True)
        sample: list[float] = []
        for seed_value in selected:
            family_values = grouped[int(seed_value)]
            sample.extend(rng.choice(family_values, size=len(family_values), replace=True).tolist())
        result[index] = float(np.median(sample))
    low, high = np.quantile(result, [0.025, 0.975])
    return float(low), float(high)


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    c1_root = args.c1_root.expanduser().resolve()
    task55_c2 = args.task55_c2_root.expanduser().resolve()
    task55_c3 = args.task55_c3_root.expanduser().resolve()
    summary = args.summary_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    tables_out = output / "tables"
    tables_out.mkdir(parents=True, exist_ok=True)
    tables = summary / "tables"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decision_path = summary / "task57c2_attribution_recovery_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    c1_decision = json.loads((c1_root / "task57c1_preflight_decision.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    parent_full = git(root, "rev-list", "-n", "1", PARENT_TAG)
    head = git(root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT_TAG, "HEAD"], cwd=root
    ).returncode == 0
    add("parent_tag_resolves_expected", parent_full.startswith(PARENT_COMMIT_SHORT), parent_full)
    add("parent_is_head_ancestor", ancestor, head)
    add("c1_all_checks_passed", c1_decision.get("checks_passed") == c1_decision.get("check_count") == 42, f"{c1_decision.get('checks_passed')}/{c1_decision.get('check_count')}")
    add("c1_ready_for_c2", c1_decision.get("ready_for_c2_recovery_analysis") is True, c1_decision.get("ready_for_c2_recovery_analysis"))
    add("c1_three_state_complete", c1_decision.get("three_state_primary_panel_complete") is True, c1_decision.get("three_state_primary_panel_complete"))
    add("c1_five_state_incomplete", c1_decision.get("five_state_roadmap_complete") is False, c1_decision.get("five_state_roadmap_complete"))

    expected_files = {
        "triplets": tables / "task57c2_triplet_recovery_metrics.csv",
        "seeds": tables / "task57c2_seed_summary.csv",
        "families": tables / "task57c2_family_summary.csv",
        "features": tables / "task57c2_feature_association.csv",
        "top_features": tables / "task57c2_top10_prespecified_features.csv",
        "hypotheses": tables / "task57c2_hypothesis_decisions.csv",
        "coverage": tables / "task57c2_state_coverage.csv",
        "sources": tables / "task57c2_source_manifest_sha256.csv",
        "decision": decision_path,
        "closeout": summary / "TASK57_C2_CLOSEOUT.md",
    }
    for name, path in expected_files.items():
        add(f"exists_{name}", path.is_file(), path)
        if not path.is_file():
            raise FileNotFoundError(path)

    triplets = pd.read_csv(expected_files["triplets"])
    seeds = pd.read_csv(expected_files["seeds"])
    families = pd.read_csv(expected_files["families"])
    features = pd.read_csv(expected_files["features"])
    top_features = pd.read_csv(expected_files["top_features"])
    hypotheses = pd.read_csv(expected_files["hypotheses"])
    coverage = pd.read_csv(expected_files["coverage"])
    source_manifest = pd.read_csv(expected_files["sources"])
    add("12_triplet_rows", len(triplets) == 12, len(triplets))
    add("4_seed_rows", len(seeds) == 4, len(seeds))
    add("3_family_rows", len(families) == 3, len(families))
    add("69_feature_rows", len(features) == 69, len(features))
    add("top_feature_count_at_most_10", len(top_features) <= 10, len(top_features))
    add("8_hypothesis_rows", len(hypotheses) == 8, len(hypotheses))
    add("5_state_coverage_rows", len(coverage) == 5, len(coverage))
    add("all_triplet_numeric_finite", np.isfinite(triplets.select_dtypes(include=[np.number]).to_numpy()).all(), "finite")
    add("all_feature_numeric_finite", np.isfinite(features.select_dtypes(include=[np.number]).to_numpy()).all(), "finite")
    add("seeds_exact", sorted(triplets["seed"].unique().tolist()) == list(SEEDS), sorted(triplets["seed"].unique().tolist()))
    add("families_exact", set(triplets["family_id"]) == set(FAMILIES), sorted(set(triplets["family_id"])))
    add("one_row_per_triplet", triplets.groupby(["seed", "family_id"]).size().eq(1).all(), "12 unique")

    epsilon = float(config["profiles"]["epsilon"])
    ranking_k = int(config["profiles"]["ranking_k"])
    recomputed: list[dict[str, Any]] = []
    for seed in SEEDS:
        clean = load(baseline_path(task55_c2, task55_c3, seed, "shared", "clean_reference"))
        positions = np.flatnonzero(clean["true_class_id"] == 2)
        clean_margin = clean["source_minus_target_margin_shap"][positions]
        clean_primary = np.mean(np.abs(clean_margin), axis=0)
        clean_signed = np.mean(clean_margin, axis=0)
        names = clean["feature_names"]
        clean_top = top_k(clean_primary, names, ranking_k)
        clean_logits = clean["logits"][positions, 2] - clean["logits"][positions, 0]
        for family in FAMILIES:
            suspicious = load(baseline_path(task55_c2, task55_c3, seed, family, "suspicious"))
            reconstructed = load(baseline_path(task55_c2, task55_c3, seed, family, "reconstructed"))
            suspicious_margin = suspicious["source_minus_target_margin_shap"][positions]
            reconstructed_margin = reconstructed["source_minus_target_margin_shap"][positions]
            suspicious_primary = np.mean(np.abs(suspicious_margin), axis=0)
            reconstructed_primary = np.mean(np.abs(reconstructed_margin), axis=0)
            suspicious_signed = np.mean(suspicious_margin, axis=0)
            reconstructed_signed = np.mean(reconstructed_margin, axis=0)
            attack_distance = normalized_l1(suspicious_primary, clean_primary, clean_primary, epsilon)
            reconstructed_distance = normalized_l1(reconstructed_primary, clean_primary, clean_primary, epsilon)
            suspicious_rho = float(spearmanr(suspicious_primary, clean_primary).statistic)
            reconstructed_rho = float(spearmanr(reconstructed_primary, clean_primary).statistic)
            suspicious_top = top_k(suspicious_primary, names, ranking_k)
            reconstructed_top = top_k(reconstructed_primary, names, ranking_k)
            suspicious_j = len(suspicious_top & clean_top) / len(suspicious_top | clean_top)
            reconstructed_j = len(reconstructed_top & clean_top) / len(reconstructed_top | clean_top)
            recomputed.append({
                "seed": seed,
                "family_id": family,
                "attack_distance": attack_distance,
                "reconstructed_distance": reconstructed_distance,
                "distance_reduction": attack_distance - reconstructed_distance,
                "recovery_fraction": (attack_distance - reconstructed_distance) / max(attack_distance, epsilon),
                "suspicious_clean_spearman": suspicious_rho,
                "reconstructed_clean_spearman": reconstructed_rho,
                "rank_recovery": reconstructed_rho - suspicious_rho,
                "suspicious_clean_top10_jaccard": suspicious_j,
                "reconstructed_clean_top10_jaccard": reconstructed_j,
                "top10_recovery": reconstructed_j - suspicious_j,
                "signed_attack_distance": normalized_l1(suspicious_signed, clean_signed, clean_signed, epsilon),
                "signed_reconstructed_distance": normalized_l1(reconstructed_signed, clean_signed, clean_signed, epsilon),
                "clean_mean_ddos_minus_benign_logit_margin": float(np.mean(clean_logits)),
                "suspicious_mean_ddos_minus_benign_logit_margin": float(np.mean(suspicious["logits"][positions, 2] - suspicious["logits"][positions, 0])),
                "reconstructed_mean_ddos_minus_benign_logit_margin": float(np.mean(reconstructed["logits"][positions, 2] - reconstructed["logits"][positions, 0])),
                "clean_ddos_prediction_rate": float(np.mean(clean["predicted_class_id"][positions] == 2)),
                "suspicious_ddos_prediction_rate": float(np.mean(suspicious["predicted_class_id"][positions] == 2)),
                "reconstructed_ddos_prediction_rate": float(np.mean(reconstructed["predicted_class_id"][positions] == 2)),
            })
    recomputed_frame = pd.DataFrame(recomputed).sort_values(["seed", "family_id"]).reset_index(drop=True)
    reported = triplets.sort_values(["seed", "family_id"]).reset_index(drop=True)
    numeric_columns = [column for column in recomputed_frame.columns if column not in ("seed", "family_id")]
    add("raw_triplet_metrics_recomputed_exactly", np.allclose(recomputed_frame[numeric_columns].to_numpy(), reported[numeric_columns].to_numpy(), rtol=1e-11, atol=1e-12), numeric_columns)
    add("signed_distance_reduction_identity", np.allclose(reported["signed_attack_distance"] - reported["signed_reconstructed_distance"], reported["signed_distance_reduction"], rtol=1e-12, atol=1e-12), "exact")
    add("distance_reduction_identity", np.allclose(reported["attack_distance"] - reported["reconstructed_distance"], reported["distance_reduction"], rtol=1e-12, atol=1e-12), "exact")
    add("recovery_fraction_identity", np.allclose(reported["distance_reduction"] / reported["attack_distance"], reported["recovery_fraction"], rtol=1e-12, atol=1e-12), "exact")

    noise = float(config["measurement_noise_gate"]["frozen_value"])
    median_attack = float(reported["attack_distance"].median())
    attack_count = int((reported["attack_distance"] > noise).sum())
    attack_pass = median_attack > noise and attack_count >= int(config["measurement_noise_gate"]["attack_signal_triplets_required"])
    positive_triplets = int((reported["distance_reduction"] > 0.0).sum())
    seed_means = reported.groupby("seed")["distance_reduction"].mean().to_numpy()
    positive_seed_means = int((seed_means > 0.0).sum())
    median_recovery = float(reported["recovery_fraction"].median())
    triplet_p = exact_sign_flip_p(reported["distance_reduction"].to_numpy())
    seed_p = exact_sign_flip_p(seed_means)
    bootstrap_low, bootstrap_high = hierarchical_bootstrap(
        reported,
        int(config["uncertainty_and_inference"]["hierarchical_bootstrap_seed"]),
        int(config["uncertainty_and_inference"]["hierarchical_bootstrap_replicates"]),
    )
    recovery = decision["recovery"]
    add("decision_median_attack_exact", math.isclose(decision["attack_signal"]["median_attack_distance"], median_attack, rel_tol=1e-12, abs_tol=1e-12), median_attack)
    add("decision_attack_count_exact", decision["attack_signal"]["triplets_above_noise"] == attack_count, attack_count)
    add("decision_attack_gate_exact", decision["attack_signal"]["passed"] is attack_pass, attack_pass)
    add("decision_positive_triplets_exact", recovery["positive_triplets"] == positive_triplets, positive_triplets)
    add("decision_positive_seed_means_exact", recovery["positive_seed_means"] == positive_seed_means, positive_seed_means)
    add("decision_median_recovery_exact", math.isclose(recovery["median_recovery_fraction"], median_recovery, rel_tol=1e-12, abs_tol=1e-12), median_recovery)
    add("decision_triplet_p_exact", math.isclose(recovery["triplet_sign_flip_p_one_sided"], triplet_p, rel_tol=0.0, abs_tol=1e-15), triplet_p)
    add("decision_seed_p_exact", math.isclose(recovery["seed_blocked_sign_flip_p_one_sided_report_only"], seed_p, rel_tol=0.0, abs_tol=1e-15), seed_p)
    add("decision_bootstrap_low_exact", math.isclose(recovery["hierarchical_bootstrap_ci95_low"], bootstrap_low, rel_tol=1e-12, abs_tol=1e-12), bootstrap_low)
    add("decision_bootstrap_high_exact", math.isclose(recovery["hierarchical_bootstrap_ci95_high"], bootstrap_high, rel_tol=1e-12, abs_tol=1e-12), bootstrap_high)

    gate = config["primary_recovery_gate"]
    gate_results = {
        "minimum_positive_triplets": positive_triplets >= int(gate["minimum_positive_triplets"]),
        "minimum_positive_seed_means": positive_seed_means >= int(gate["minimum_positive_seed_means"]),
        "minimum_median_recovery_fraction": median_recovery >= float(gate["minimum_median_recovery_fraction"]),
        "triplet_sign_flip_p": triplet_p <= float(gate["triplet_sign_flip_p_at_most"]),
        "hierarchical_bootstrap_ci_low": bootstrap_low > float(gate["hierarchical_bootstrap_ci95_low_above"]),
    }
    all_recovery = all(gate_results.values())
    expected_result = "INCONCLUSIVE" if not attack_pass else "PASS" if all_recovery else "PARTIAL" if median_recovery > 0.0 else "FAIL"
    add("individual_gate_results_exact", recovery["individual_gate_results"] == gate_results, gate_results)
    add("scientific_result_exact", decision["scientific_result"] == expected_result, expected_result)

    feature_policy = config["feature_association_policy"]
    expected_associated = (
        (features["same_direction_triplets"] >= int(feature_policy["minimum_same_direction_triplets"]))
        & (features["attack_shift_to_noise_ratio"] >= float(feature_policy["minimum_median_absolute_shift_to_noise_ratio"]))
    )
    add("feature_association_rule_exact", np.array_equal(features["meets_prespecified_association"].astype(bool).to_numpy(), expected_associated.to_numpy()), int(expected_associated.sum()))
    sorted_features = features.sort_values(["median_absolute_attack_shift", "feature"], ascending=[False, True]).reset_index(drop=True)
    add("feature_ranking_deterministic", features["feature"].tolist() == sorted_features["feature"].tolist(), "descending shift then feature")
    expected_top = features[features["meets_prespecified_association"].astype(bool)].head(10)["feature"].tolist()
    add("top_feature_table_exact", top_features["feature"].tolist() == expected_top, expected_top)
    add("all_69_features_unique", features["feature"].nunique() == 69, features["feature"].nunique())

    coverage_map = dict(zip(coverage["state"], coverage["availability"]))
    add("coverage_available_states_exact", {key for key, value in coverage_map.items() if value == "available"} == {"clean_reference", "suspicious", "reconstructed"}, coverage_map)
    add("coverage_unavailable_states_exact", {key for key, value in coverage_map.items() if value == "unavailable"} == {"rejected", "oracle_clean"}, coverage_map)
    add("oracle_claim_forbidden", decision.get("oracle_recovery_claim_permitted") is False, decision.get("oracle_recovery_claim_permitted"))
    add("claim_limited_to_preattack", "preattack_clean_reference" in decision.get("claim", ""), decision.get("claim"))
    add("no_new_shap", decision.get("new_shap_evaluations") == 0, decision.get("new_shap_evaluations"))
    add("no_training", decision.get("training_permitted") is False, decision.get("training_permitted"))
    add("reserved_test_closed", decision.get("reserved_test_arrays_materialized") is False, decision.get("reserved_test_arrays_materialized"))
    add("negative_results_retained", decision.get("negative_and_boundary_results_retained") is True, decision.get("negative_and_boundary_results_retained"))
    add("task58_figures_deferred", decision.get("publication_figures_deferred_to_task58") is True, decision.get("publication_figures_deferred_to_task58"))
    add("no_task57_figures_created", not (summary / "figures").exists(), summary / "figures")

    hashes_valid = True
    unique_manifest_paths = source_manifest["path"].nunique() == len(source_manifest)
    for row in source_manifest.to_dict("records"):
        path = root / str(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or file_sha256(path) != str(row["sha256"]):
            hashes_valid = False
            break
    add("source_manifest_paths_unique", unique_manifest_paths, len(source_manifest))
    add("all_source_hashes_verified", hashes_valid, len(source_manifest))

    with (tables_out / "task57c2_audit_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "detail"])
        writer.writeheader()
        writer.writerows(checks)
    sources_to_record = [
        config_path,
        c1_root / "task57c1_preflight_decision.json",
        *expected_files.values(),
    ]
    with (tables_out / "task57c2_audit_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for path in sorted(set(sources_to_record), key=lambda value: str(value)):
            writer.writerow({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": file_sha256(path)})

    passed = sum(int(row["passed"]) for row in checks)
    audit_decision = {
        "experiment_version": "4.19.C2A",
        "stage": "task57_attribution_recovery_integrity_audit",
        "parent_tag": PARENT_TAG,
        "parent_commit": parent_full,
        "head_commit": head,
        "checks_passed": passed,
        "check_count": len(checks),
        "scientific_result_verified": expected_result,
        "triplets_verified": len(triplets),
        "features_verified": len(features),
        "source_files_verified": len(source_manifest),
        "raw_triplet_metrics_independently_recomputed": True,
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
        "integrity_audit_passed": passed == len(checks),
        "ready_for_task57_freeze": passed == len(checks),
    }
    (output / "task57c2_audit_decision.json").write_text(json.dumps(audit_decision, indent=2), encoding="utf-8")
    print("===== TASK 57 C2 RECOVERY AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("SCIENTIFIC RESULT VERIFIED:", expected_result)
    print("TRIPLETS VERIFIED:", len(triplets))
    print("FEATURES VERIFIED:", len(features))
    print("SOURCE FILES VERIFIED:", len(source_manifest))
    print("RAW TRIPLET METRICS INDEPENDENTLY RECOMPUTED: True")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR TASK 57 FREEZE:", passed == len(checks))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the frozen Task 57 attribution recovery analysis with no new SHAP work."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PARENT_TAG = "task57-c1-attribution-recovery-preflight-frozen-v4191"
PARENT_COMMIT_SHORT = "e9a969d"
SEEDS = (7, 99, 123, 2026)
FAMILIES = ("A_development_anchor", "B_hash_ranked", "C_hash_ranked")
REPEAT_RSEEDS = (56101, 56102, 56103)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--c1-root", required=True, type=Path)
    parser.add_argument("--task55-c2-root", required=True, type=Path)
    parser.add_argument("--task55-c3-root", required=True, type=Path)
    parser.add_argument("--task56-execution-root", required=True, type=Path)
    parser.add_argument("--task56-summary-root", required=True, type=Path)
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


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


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


def repeat_path(execution_root: Path, rseed: int, seed: int, family: str, state: str) -> Path:
    physical_family = "shared" if state == "clean_reference" else family
    return (
        execution_root
        / "evaluations"
        / "repeat"
        / f"variant_{rseed}"
        / f"seed_{seed}"
        / physical_family
        / state
        / "task56c2_margin_attributions.npz"
    )


def load_baseline(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def top_k(profile: np.ndarray, names: np.ndarray, k: int) -> set[str]:
    order = sorted(range(len(profile)), key=lambda index: (-float(profile[index]), str(names[index])))
    return {str(names[index]) for index in order[:k]}


def jaccard(left: set[str], right: set[str]) -> float:
    return float(len(left & right) / len(left | right))


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    if not math.isfinite(value):
        raise RuntimeError("Nonfinite Spearman correlation")
    return value


def normalized_l1(left: np.ndarray, right: np.ndarray, denominator_profile: np.ndarray, epsilon: float) -> float:
    denominator = float(np.abs(denominator_profile).sum()) + epsilon
    return float(np.abs(left - right).sum() / denominator)


def exact_sign_flip_p(values: np.ndarray) -> float:
    observed = float(np.mean(values))
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = float(np.mean(values * np.asarray(signs, dtype=np.float64)))
        count += int(statistic >= observed - 1e-15)
        total += 1
    return float(count / total)


def hierarchical_bootstrap(values: pd.DataFrame, seed: int, replicates: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grouped = {
        int(seed_value): group["recovery_fraction"].to_numpy(dtype=np.float64)
        for seed_value, group in values.groupby("seed", sort=True)
    }
    seed_values = np.asarray(sorted(grouped), dtype=np.int64)
    result = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
        sample: list[float] = []
        for seed_value in selected_seeds:
            family_values = grouped[int(seed_value)]
            sample.extend(rng.choice(family_values, size=len(family_values), replace=True).tolist())
        result[index] = float(np.median(np.asarray(sample, dtype=np.float64)))
    return result


def main() -> int:
    args = parse_args()
    started = time.time()
    root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    c1_root = args.c1_root.expanduser().resolve()
    task55_c2 = args.task55_c2_root.expanduser().resolve()
    task55_c3 = args.task55_c3_root.expanduser().resolve()
    task56_execution = args.task56_execution_root.expanduser().resolve()
    task56_summary = args.task56_summary_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    required = [config_path, c1_root, task55_c2, task55_c3, task56_execution, task56_summary]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    parent_full = git(root, "rev-list", "-n", "1", PARENT_TAG)
    if not parent_full.startswith(PARENT_COMMIT_SHORT):
        raise RuntimeError(f"Unexpected Task 57 C1 parent: {parent_full}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", PARENT_TAG, "HEAD"], cwd=root).returncode != 0:
        raise RuntimeError("Frozen Task 57 C1 tag is not an ancestor of HEAD")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    c1_decision_path = c1_root / "task57c1_preflight_decision.json"
    c1_triplet_path = c1_root / "tables" / "task57c1_triplet_plan.csv"
    c1_decision = json.loads(c1_decision_path.read_text(encoding="utf-8"))
    if c1_decision.get("ready_for_c2_recovery_analysis") is not True:
        raise RuntimeError("Task 57 C1 did not authorize C2")
    if c1_decision.get("checks_passed") != c1_decision.get("check_count"):
        raise RuntimeError("Task 57 C1 checks are incomplete")

    epsilon = float(config["profiles"]["epsilon"])
    ranking_k = int(config["profiles"]["ranking_k"])
    source_files: list[Path] = [config_path, c1_decision_path, c1_triplet_path]
    baseline: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}
    physical_keys: list[tuple[int, str, str]] = []
    for seed in SEEDS:
        clean_path = baseline_path(task55_c2, task55_c3, seed, "shared", "clean_reference")
        baseline[(seed, "shared", "clean_reference")] = load_baseline(clean_path)
        source_files.append(clean_path)
        physical_keys.append((seed, "shared", "clean_reference"))
        for family in FAMILIES:
            for state in ("suspicious", "reconstructed"):
                path = baseline_path(task55_c2, task55_c3, seed, family, state)
                baseline[(seed, family, state)] = load_baseline(path)
                source_files.append(path)
                physical_keys.append((seed, family, state))

    feature_names = baseline[(7, "shared", "clean_reference")]["feature_names"]
    triplet_rows: list[dict[str, Any]] = []
    attack_shifts: list[np.ndarray] = []
    reconstructed_shifts: list[np.ndarray] = []

    for seed in SEEDS:
        clean = baseline[(seed, "shared", "clean_reference")]
        clean_positions = np.flatnonzero(clean["true_class_id"] == 2)
        clean_margin = clean["source_minus_target_margin_shap"][clean_positions]
        clean_primary = np.mean(np.abs(clean_margin), axis=0)
        clean_signed = np.mean(clean_margin, axis=0)
        clean_top = top_k(clean_primary, feature_names, ranking_k)
        clean_logit_margin = clean["logits"][clean_positions, 2] - clean["logits"][clean_positions, 0]
        clean_ddos_rate = float(np.mean(clean["predicted_class_id"][clean_positions] == 2))
        for family in FAMILIES:
            suspicious = baseline[(seed, family, "suspicious")]
            reconstructed = baseline[(seed, family, "reconstructed")]
            positions = np.flatnonzero(suspicious["true_class_id"] == 2)
            if not np.array_equal(positions, clean_positions):
                raise RuntimeError(f"DDoS row mismatch: seed={seed} family={family}")
            suspicious_margin = suspicious["source_minus_target_margin_shap"][positions]
            reconstructed_margin = reconstructed["source_minus_target_margin_shap"][positions]
            suspicious_primary = np.mean(np.abs(suspicious_margin), axis=0)
            reconstructed_primary = np.mean(np.abs(reconstructed_margin), axis=0)
            suspicious_signed = np.mean(suspicious_margin, axis=0)
            reconstructed_signed = np.mean(reconstructed_margin, axis=0)
            attack_shifts.append(suspicious_signed - clean_signed)
            reconstructed_shifts.append(reconstructed_signed - clean_signed)

            attack_distance = normalized_l1(suspicious_primary, clean_primary, clean_primary, epsilon)
            reconstructed_distance = normalized_l1(reconstructed_primary, clean_primary, clean_primary, epsilon)
            distance_reduction = attack_distance - reconstructed_distance
            recovery_fraction = distance_reduction / max(attack_distance, epsilon)
            suspicious_spearman = safe_spearman(suspicious_primary, clean_primary)
            reconstructed_spearman = safe_spearman(reconstructed_primary, clean_primary)
            suspicious_top = top_k(suspicious_primary, feature_names, ranking_k)
            reconstructed_top = top_k(reconstructed_primary, feature_names, ranking_k)
            suspicious_jaccard = jaccard(suspicious_top, clean_top)
            reconstructed_jaccard = jaccard(reconstructed_top, clean_top)
            signed_attack_distance = normalized_l1(suspicious_signed, clean_signed, clean_signed, epsilon)
            signed_reconstructed_distance = normalized_l1(reconstructed_signed, clean_signed, clean_signed, epsilon)
            suspicious_logits = suspicious["logits"][positions, 2] - suspicious["logits"][positions, 0]
            reconstructed_logits = reconstructed["logits"][positions, 2] - reconstructed["logits"][positions, 0]
            triplet_rows.append({
                "seed": seed,
                "family_id": family,
                "coalition_size": 10,
                "attack_distance": attack_distance,
                "reconstructed_distance": reconstructed_distance,
                "distance_reduction": distance_reduction,
                "recovery_fraction": recovery_fraction,
                "positive_distance_reduction": distance_reduction > 0.0,
                "attack_exceeds_repeat_noise_q95": attack_distance > float(config["measurement_noise_gate"]["frozen_value"]),
                "suspicious_clean_spearman": suspicious_spearman,
                "reconstructed_clean_spearman": reconstructed_spearman,
                "rank_recovery": reconstructed_spearman - suspicious_spearman,
                "suspicious_clean_top10_jaccard": suspicious_jaccard,
                "reconstructed_clean_top10_jaccard": reconstructed_jaccard,
                "top10_recovery": reconstructed_jaccard - suspicious_jaccard,
                "signed_attack_distance": signed_attack_distance,
                "signed_reconstructed_distance": signed_reconstructed_distance,
                "signed_distance_reduction": signed_attack_distance - signed_reconstructed_distance,
                "clean_mean_ddos_minus_benign_logit_margin": float(np.mean(clean_logit_margin)),
                "suspicious_mean_ddos_minus_benign_logit_margin": float(np.mean(suspicious_logits)),
                "reconstructed_mean_ddos_minus_benign_logit_margin": float(np.mean(reconstructed_logits)),
                "clean_ddos_prediction_rate": clean_ddos_rate,
                "suspicious_ddos_prediction_rate": float(np.mean(suspicious["predicted_class_id"][positions] == 2)),
                "reconstructed_ddos_prediction_rate": float(np.mean(reconstructed["predicted_class_id"][positions] == 2)),
            })

    triplets = pd.DataFrame(triplet_rows).sort_values(["seed", "family_id"]).reset_index(drop=True)
    attack_shift_matrix = np.stack(attack_shifts, axis=0)
    reconstructed_shift_matrix = np.stack(reconstructed_shifts, axis=0)

    repeat_noise_rows: list[np.ndarray] = []
    for seed, family, state in physical_keys:
        base = baseline[(seed, family, state)]
        if state == "clean_reference":
            ddos_positions = np.flatnonzero(base["true_class_id"] == 2)
        else:
            ddos_positions = np.flatnonzero(base["true_class_id"] == 2)
        base_signed = np.mean(base["source_minus_target_margin_shap"][ddos_positions], axis=0)
        for rseed in REPEAT_RSEEDS:
            path = repeat_path(task56_execution, rseed, seed, family, state)
            source_files.append(path)
            with np.load(path, allow_pickle=False) as payload:
                repeat_margin = np.asarray(payload["source_minus_target_margin_shap"])
                repeat_positions = np.asarray(payload["true_ddos_probe_positions"], dtype=np.int64)
                repeat_signed = np.mean(repeat_margin[repeat_positions], axis=0)
            repeat_noise_rows.append(np.abs(repeat_signed - base_signed))
    repeat_noise = np.stack(repeat_noise_rows, axis=0)
    feature_noise_q95 = np.quantile(repeat_noise, 0.95, axis=0)

    feature_rows: list[dict[str, Any]] = []
    for index, feature in enumerate(feature_names):
        shifts = attack_shift_matrix[:, index]
        reconstructed_values = reconstructed_shift_matrix[:, index]
        positive = int(np.count_nonzero(shifts > 0.0))
        negative = int(np.count_nonzero(shifts < 0.0))
        majority = max(positive, negative)
        median_shift = float(np.median(shifts))
        median_abs_shift = float(np.median(np.abs(shifts)))
        median_abs_reconstructed = float(np.median(np.abs(reconstructed_values)))
        noise = float(feature_noise_q95[index])
        ratio = median_abs_shift / max(noise, epsilon)
        consistent = majority >= int(config["feature_association_policy"]["minimum_same_direction_triplets"])
        associated = consistent and ratio >= float(config["feature_association_policy"]["minimum_median_absolute_shift_to_noise_ratio"])
        direction = "positive" if positive > negative else "negative" if negative > positive else "mixed"
        feature_rows.append({
            "feature": str(feature),
            "positive_triplets": positive,
            "negative_triplets": negative,
            "same_direction_triplets": majority,
            "same_direction_fraction": majority / 12.0,
            "majority_direction": direction,
            "median_signed_attack_shift": median_shift,
            "median_absolute_attack_shift": median_abs_shift,
            "feature_repeat_noise_q95": noise,
            "attack_shift_to_noise_ratio": ratio,
            "median_absolute_reconstructed_shift": median_abs_reconstructed,
            "feature_recovery_fraction": (median_abs_shift - median_abs_reconstructed) / max(median_abs_shift, epsilon),
            "meets_direction_consistency": consistent,
            "meets_prespecified_association": associated,
        })
    features = pd.DataFrame(feature_rows).sort_values(
        ["median_absolute_attack_shift", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    features.insert(0, "rank", np.arange(1, len(features) + 1))
    top_count = int(config["feature_association_policy"]["fixed_report_count"])
    top_features = features[features["meets_prespecified_association"]].head(top_count).copy()

    seed_summary = triplets.groupby("seed", as_index=False).agg(
        mean_attack_distance=("attack_distance", "mean"),
        mean_reconstructed_distance=("reconstructed_distance", "mean"),
        mean_distance_reduction=("distance_reduction", "mean"),
        median_recovery_fraction=("recovery_fraction", "median"),
        positive_triplets=("positive_distance_reduction", "sum"),
        mean_rank_recovery=("rank_recovery", "mean"),
        mean_top10_recovery=("top10_recovery", "mean"),
    )
    family_summary = triplets.groupby("family_id", as_index=False).agg(
        mean_attack_distance=("attack_distance", "mean"),
        mean_reconstructed_distance=("reconstructed_distance", "mean"),
        mean_distance_reduction=("distance_reduction", "mean"),
        median_recovery_fraction=("recovery_fraction", "median"),
        positive_seeds=("positive_distance_reduction", "sum"),
        mean_rank_recovery=("rank_recovery", "mean"),
        mean_top10_recovery=("top10_recovery", "mean"),
    )

    noise_value = float(config["measurement_noise_gate"]["frozen_value"])
    attack_signal_count = int(triplets["attack_exceeds_repeat_noise_q95"].sum())
    median_attack_distance = float(triplets["attack_distance"].median())
    attack_signal_pass = (
        attack_signal_count >= int(config["measurement_noise_gate"]["attack_signal_triplets_required"])
        and median_attack_distance > noise_value
    )
    positive_triplets = int(triplets["positive_distance_reduction"].sum())
    positive_seed_means = int((seed_summary["mean_distance_reduction"] > 0.0).sum())
    median_recovery = float(triplets["recovery_fraction"].median())
    triplet_p = exact_sign_flip_p(triplets["distance_reduction"].to_numpy(dtype=np.float64))
    seed_p = exact_sign_flip_p(seed_summary["mean_distance_reduction"].to_numpy(dtype=np.float64))
    bootstrap = hierarchical_bootstrap(
        triplets,
        int(config["uncertainty_and_inference"]["hierarchical_bootstrap_seed"]),
        int(config["uncertainty_and_inference"]["hierarchical_bootstrap_replicates"]),
    )
    bootstrap_low, bootstrap_high = [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])]
    gate = config["primary_recovery_gate"]
    recovery_gate_results = {
        "minimum_positive_triplets": positive_triplets >= int(gate["minimum_positive_triplets"]),
        "minimum_positive_seed_means": positive_seed_means >= int(gate["minimum_positive_seed_means"]),
        "minimum_median_recovery_fraction": median_recovery >= float(gate["minimum_median_recovery_fraction"]),
        "triplet_sign_flip_p": triplet_p <= float(gate["triplet_sign_flip_p_at_most"]),
        "hierarchical_bootstrap_ci_low": bootstrap_low > float(gate["hierarchical_bootstrap_ci95_low_above"]),
    }
    recovery_gate_pass = all(recovery_gate_results.values())
    if not attack_signal_pass:
        scientific_result = "INCONCLUSIVE"
    elif recovery_gate_pass:
        scientific_result = "PASS"
    elif median_recovery > 0.0:
        scientific_result = "PARTIAL"
    else:
        scientific_result = "FAIL"

    hypothesis_rows = [
        {"gate": "attack_signal_median_above_noise_q95", "value": median_attack_distance, "threshold": noise_value, "passed": median_attack_distance > noise_value},
        {"gate": "attack_signal_triplets_above_noise_q95", "value": attack_signal_count, "threshold": int(config["measurement_noise_gate"]["attack_signal_triplets_required"]), "passed": attack_signal_count >= int(config["measurement_noise_gate"]["attack_signal_triplets_required"])},
        {"gate": "positive_triplets", "value": positive_triplets, "threshold": int(gate["minimum_positive_triplets"]), "passed": recovery_gate_results["minimum_positive_triplets"]},
        {"gate": "positive_seed_means", "value": positive_seed_means, "threshold": int(gate["minimum_positive_seed_means"]), "passed": recovery_gate_results["minimum_positive_seed_means"]},
        {"gate": "median_recovery_fraction", "value": median_recovery, "threshold": float(gate["minimum_median_recovery_fraction"]), "passed": recovery_gate_results["minimum_median_recovery_fraction"]},
        {"gate": "triplet_sign_flip_p_one_sided", "value": triplet_p, "threshold": float(gate["triplet_sign_flip_p_at_most"]), "passed": recovery_gate_results["triplet_sign_flip_p"]},
        {"gate": "hierarchical_bootstrap_ci95_low", "value": bootstrap_low, "threshold": float(gate["hierarchical_bootstrap_ci95_low_above"]), "passed": recovery_gate_results["hierarchical_bootstrap_ci_low"]},
        {"gate": "seed_blocked_sign_flip_p_report_only", "value": seed_p, "threshold": 0.0625, "passed": seed_p <= 0.0625},
    ]
    hypothesis = pd.DataFrame(hypothesis_rows)
    coverage = pd.DataFrame([
        {"state": "clean_reference", "availability": "available", "included_in_primary_analysis": True, "limitation": "round 4 preattack reference rather than round 8 oracle clean"},
        {"state": "suspicious", "availability": "available", "included_in_primary_analysis": True, "limitation": "none"},
        {"state": "reconstructed", "availability": "available", "included_in_primary_analysis": True, "limitation": "none"},
        {"state": "rejected", "availability": "unavailable", "included_in_primary_analysis": False, "limitation": "no exact paired rejected attribution artifact"},
        {"state": "oracle_clean", "availability": "unavailable", "included_in_primary_analysis": False, "limitation": "no round 8 counterfactual clean attribution artifact"},
    ])

    atomic_csv(tables / "task57c2_triplet_recovery_metrics.csv", triplets)
    atomic_csv(tables / "task57c2_seed_summary.csv", seed_summary)
    atomic_csv(tables / "task57c2_family_summary.csv", family_summary)
    atomic_csv(tables / "task57c2_feature_association.csv", features)
    atomic_csv(tables / "task57c2_top10_prespecified_features.csv", top_features)
    atomic_csv(tables / "task57c2_hypothesis_decisions.csv", hypothesis)
    atomic_csv(tables / "task57c2_state_coverage.csv", coverage)

    source_files.extend([
        task56_summary / "task56c2_validation_decision.json",
        task56_summary / "tables" / "task56c2_repeated_run_stability.csv",
    ])
    source_rows = []
    for path in sorted(set(source_files), key=lambda value: str(value)):
        source_rows.append({
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    atomic_csv(tables / "task57c2_source_manifest_sha256.csv", pd.DataFrame(source_rows))

    decision = {
        "experiment_version": "4.19.C2",
        "stage": "task57_attribution_recovery_summary",
        "parent_tag": PARENT_TAG,
        "parent_commit": parent_full,
        "scientific_result": scientific_result,
        "claim": "reconstruction_moves_attributions_toward_the_frozen_preattack_clean_reference",
        "oracle_recovery_claim_permitted": False,
        "triplets_analyzed": len(triplets),
        "seeds_analyzed": list(SEEDS),
        "families_analyzed": list(FAMILIES),
        "physical_attribution_states_reused": 28,
        "new_shap_evaluations": 0,
        "attack_signal": {
            "repeat_noise_q95": noise_value,
            "median_attack_distance": median_attack_distance,
            "triplets_above_noise": attack_signal_count,
            "passed": attack_signal_pass,
        },
        "recovery": {
            "positive_triplets": positive_triplets,
            "positive_seed_means": positive_seed_means,
            "median_recovery_fraction": median_recovery,
            "hierarchical_bootstrap_ci95_low": bootstrap_low,
            "hierarchical_bootstrap_ci95_high": bootstrap_high,
            "triplet_sign_flip_p_one_sided": triplet_p,
            "seed_blocked_sign_flip_p_one_sided_report_only": seed_p,
            "individual_gate_results": recovery_gate_results,
            "all_primary_gates_passed": recovery_gate_pass,
        },
        "secondary": {
            "median_rank_recovery": float(triplets["rank_recovery"].median()),
            "positive_rank_recovery_triplets": int((triplets["rank_recovery"] > 0.0).sum()),
            "median_top10_recovery": float(triplets["top10_recovery"].median()),
            "positive_top10_recovery_triplets": int((triplets["top10_recovery"] > 0.0).sum()),
            "prespecified_poison_associated_features": int(features["meets_prespecified_association"].sum()),
            "fixed_reported_feature_count": len(top_features),
        },
        "state_coverage": {
            "available": ["clean_reference", "suspicious", "reconstructed"],
            "unavailable": ["rejected", "oracle_clean"],
            "five_state_roadmap_complete": False,
            "three_state_primary_analysis_complete": True,
        },
        "training_permitted": False,
        "reserved_test_arrays_materialized": False,
        "negative_and_boundary_results_retained": True,
        "publication_figures_deferred_to_task58": True,
        "elapsed_seconds": round(time.time() - started, 2),
        "complete": True,
    }
    atomic_json(output / "task57c2_attribution_recovery_decision.json", decision)

    closeout = f"""# Task 57 C2 attribution recovery closeout

Scientific result: **{scientific_result}**

The frozen analysis reused 28 validated attribution states and evaluated 12 clean, suspicious, reconstructed triplets across four seeds and three coalition families. It performed no training and no new SHAP evaluations.

The median suspicious to clean distance was `{median_attack_distance:.6f}` against the Task 56 repeat noise q95 of `{noise_value:.6f}`. `{attack_signal_count}` of 12 attack distances exceeded that noise reference.

The median recovery fraction was `{median_recovery:.6f}` with a hierarchical 95 percent interval of `[{bootstrap_low:.6f}, {bootstrap_high:.6f}]`. `{positive_triplets}` of 12 triplets and `{positive_seed_means}` of 4 seed means had positive distance reduction. The one sided exact triplet sign flip p value was `{triplet_p:.6f}`. The resolution limited seed blocked p value was `{seed_p:.6f}`.

The prespecified feature rule identified `{int(features['meets_prespecified_association'].sum())}` poisoning associated features, of which `{len(top_features)}` are retained in the fixed report table.

Rejected and oracle clean attributions remain unavailable. The supported claim is movement toward the frozen preattack clean reference, not recovery of an unobserved oracle clean model. Publication figures and case studies remain deferred to Task 58.
"""
    (output / "TASK57_C2_CLOSEOUT.md").write_text(closeout, encoding="utf-8")

    print("===== TASK 57 C2 ATTRIBUTION RECOVERY =====")
    print("SCIENTIFIC RESULT:", scientific_result)
    print("TRIPLETS ANALYZED:", len(triplets))
    print("ATTACK SIGNAL GATE:", attack_signal_pass)
    print("POSITIVE DISTANCE REDUCTION:", f"{positive_triplets}/12")
    print("POSITIVE SEED MEANS:", f"{positive_seed_means}/4")
    print("MEDIAN RECOVERY FRACTION:", round(median_recovery, 6))
    print("HIERARCHICAL CI95:", f"[{bootstrap_low:.6f}, {bootstrap_high:.6f}]")
    print("TRIPLET SIGN FLIP P:", round(triplet_p, 6))
    print("SEED BLOCKED P, REPORT ONLY:", round(seed_p, 6))
    print("PRESPECIFIED POISON ASSOCIATED FEATURES:", int(features["meets_prespecified_association"].sum()))
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("TASK 57 C2 COMPLETE: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

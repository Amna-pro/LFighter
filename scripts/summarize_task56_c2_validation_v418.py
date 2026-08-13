#!/usr/bin/env python3
"""Summarize preregistered Task 56 stability, faithfulness, and specificity domains."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

SEEDS = (7, 99, 123, 2026)
FAMILIES = ("A_development_anchor", "B_hash_ranked", "C_hash_ranked")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-file", required=True, type=Path)
    p.add_argument("--frozen-index-file", required=True, type=Path)
    p.add_argument("--checkpoint-readiness", required=True, type=Path)
    p.add_argument("--warmup-root", required=True, type=Path)
    p.add_argument("--task45-c2-root", required=True, type=Path)
    p.add_argument("--task45-c3-root", required=True, type=Path)
    p.add_argument("--task55-c2-root", required=True, type=Path)
    p.add_argument("--task55-c3-root", required=True, type=Path)
    p.add_argument("--execution-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--threads", type=int, default=6)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); frame.to_csv(tmp, index=False); os.replace(tmp, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, indent=2), encoding="utf-8"); os.replace(tmp, path)


def state_key(seed: int, family: str, state: str) -> str:
    return f"seed_{seed}__clean_reference" if state == "clean_reference" else f"seed_{seed}__{family}__{state}"


def rank_metric(a: np.ndarray, b: np.ndarray) -> float:
    value = float(spearmanr(a, b).statistic)
    return 0.0 if not np.isfinite(value) else value


def top(values: np.ndarray, k: int = 10) -> set[int]:
    return set(np.lexsort((np.arange(len(values)), -np.asarray(values)))[:k].tolist())


def jaccard(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    x, y = top(a, k), top(b, k)
    return len(x & y) / len(x | y)


def baseline_dir(c2: Path, c3: Path, seed: int, family: str, state: str) -> tuple[Path, str]:
    root, prefix = (c2, "task55c2") if seed == 7 else (c3 / f"seed_{seed}", "task55c3")
    directory = root / "common" / "clean_reference" if state == "clean_reference" else root / "families" / family / state
    return directory, prefix


def baseline_tensor(c2: Path, c3: Path, seed: int, family: str, state: str) -> np.ndarray:
    directory, prefix = baseline_dir(c2, c3, seed, family, state)
    with np.load(directory / f"{prefix}_attributions.npz") as f:
        return f["source_minus_target_margin_shap"].astype(np.float64)


def new_tensor(root: Path, mode: str, variant: int, seed: int, family: str, state: str) -> np.ndarray:
    fam = "shared" if state == "clean_reference" else family
    path = root / "evaluations" / mode / f"variant_{variant}" / f"seed_{seed}" / fam / state / "task56c2_margin_attributions.npz"
    with np.load(path) as f: return f["source_minus_target_margin_shap"].astype(np.float64)


def checkpoint(row: pd.Series, warmup: Path, c2: Path, c3: Path) -> Path:
    seed, family, state = int(row.seed), str(row.family_id), str(row.state)
    if state == "clean_reference": return warmup / f"seed_{seed}" / "warmup" / "checkpoints" / "common_round4_warmup_model.pt"
    root = c2 if seed == 7 else c3
    branch, name = (("plain_attack", "continuation_last_round_model.pt") if state == "suspicious" else ("trusted_reconstruction", "reconstruction_last_round_model.pt"))
    return root / family / "size_10" / f"seed_{seed}" / branch / "checkpoints" / name


def cluster_ci(frame: pd.DataFrame, cluster: str, column: str, statistic: Callable[[np.ndarray], float], rng: np.random.Generator, reps: int) -> tuple[float, float]:
    groups = [g[column].to_numpy(float) for _, g in frame.groupby(cluster, sort=False)]
    estimates = np.empty(reps)
    for i in range(reps):
        chosen = rng.integers(0, len(groups), len(groups))
        estimates[i] = statistic(np.concatenate([groups[j] for j in chosen]))
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def main() -> int:
    a = args(); root = a.project_root.resolve(); out = a.output_dir.resolve(); tables = out / "tables"; tables.mkdir(parents=True, exist_ok=True)
    config = json.loads(a.config.resolve().read_text(encoding="utf-8"))
    ready = pd.read_csv(a.checkpoint_readiness.resolve())
    if len(ready) != 28: raise RuntimeError("Expected 28 physical states")
    execution = json.loads((a.execution_root.resolve() / "task56c2_execution_metadata.json").read_text(encoding="utf-8"))
    if not execution.get("complete") or execution.get("new_shap_state_evaluations") != 140: raise RuntimeError("Task 56 C2 execution is incomplete")
    with np.load(a.frozen_index_file.resolve()) as f:
        ddos = {k: f[f"true_ddos_probe_positions_{k}"].astype(np.int64) for k in (8, 12, 16)}
        probe_indices = f["probe_validation_indices"].astype(np.int64)
        baseline_background = f["baseline_background_train_indices"].astype(np.int64)
        random_panels = {k: f[f"random_feature_panels_k{k}"].astype(np.int64) for k in (1, 5, 10, 20)}
    baseline_profiles: dict[str, np.ndarray] = {}
    baseline_tensors: dict[str, np.ndarray] = {}
    repeat_rows: list[dict[str, Any]] = []; background_rows: list[dict[str, Any]] = []; probe_rows: list[dict[str, Any]] = []
    for _, row in ready.iterrows():
        seed, family, state = int(row.seed), str(row.family_id), str(row.state); key = state_key(seed, family, state)
        tensor = baseline_tensor(a.task55_c2_root.resolve(), a.task55_c3_root.resolve(), seed, family, state)
        baseline_tensors[key] = tensor
        reference = np.mean(np.abs(tensor[ddos[16]]), axis=0); baseline_profiles[key] = reference
        for variant in (56101, 56102, 56103):
            profile = np.mean(np.abs(new_tensor(a.execution_root.resolve(), "repeat", variant, seed, family, state)[ddos[16]]), axis=0)
            repeat_rows.append({"physical_state_key": key, "seed": seed, "family_id": family, "state": state, "repeat_rseed": variant,
                                "spearman_rho": rank_metric(profile, reference), "top10_jaccard": jaccard(profile, reference),
                                "normalized_l1_drift": float(np.sum(np.abs(profile-reference)) / max(np.sum(np.abs(reference)), 1e-15))})
        for variant in (5601, 5602):
            profile = np.mean(np.abs(new_tensor(a.execution_root.resolve(), "alternate_background", variant, seed, family, state)[ddos[16]]), axis=0)
            background_rows.append({"physical_state_key": key, "seed": seed, "family_id": family, "state": state, "background_seed": variant,
                                    "spearman_rho": rank_metric(profile, reference), "top10_jaccard": jaccard(profile, reference)})
        profile16 = np.mean(np.abs(tensor[ddos[16]]), axis=0)
        for size in (8, 12):
            profile = np.mean(np.abs(tensor[ddos[size]]), axis=0)
            probe_rows.append({"physical_state_key": key, "seed": seed, "family_id": family, "state": state, "probe_size": size,
                               "reference_probe_size": 16, "spearman_rho": rank_metric(profile, profile16), "top10_jaccard": jaccard(profile, profile16)})
    repeat = pd.DataFrame(repeat_rows); background = pd.DataFrame(background_rows); probe = pd.DataFrame(probe_rows)
    atomic_csv(tables / "task56c2_repeated_run_stability.csv", repeat)
    atomic_csv(tables / "task56c2_background_sensitivity.csv", background)
    atomic_csv(tables / "task56c2_probe_size_sensitivity.csv", probe)

    loaded: list[str] = []
    with np.load(a.data_file.resolve()) as d:
        X_train = d["X_train"].astype(np.float32, copy=False); loaded.append("X_train")
        y_train = d["y_train"].astype(np.int64, copy=False); loaded.append("y_train")
        X_val = d["X_val"].astype(np.float32, copy=False); loaded.append("X_val")
        y_val = d["y_val"].astype(np.int64, copy=False); loaded.append("y_val")
    if loaded != ["X_train", "y_train", "X_val", "y_val"]: raise RuntimeError("Unexpected array access")
    replacement = X_train[baseline_background].mean(axis=0)
    ddos_values = X_val[probe_indices[ddos[16]]]
    src = root / "src"; sys.path.insert(0, str(src)) if str(src) not in sys.path else None
    from neural_models_v24 import build_model
    torch.set_num_threads(a.threads)
    feature_rows: list[dict[str, Any]] = []; faith_rows: list[dict[str, Any]] = []; random_rows: list[dict[str, Any]] = []
    for i, row in ready.iterrows():
        seed, family, state = int(row.seed), str(row.family_id), str(row.state); key = state_key(seed, family, state)
        model = build_model("resmlp", 69, 8); payload = torch.load(checkpoint(row, a.warmup_root.resolve(), a.task45_c2_root.resolve(), a.task45_c3_root.resolve()), map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"]); model.eval()
        def margins(values: np.ndarray) -> np.ndarray:
            outputs: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(values), 1024):
                    z = model(torch.from_numpy(values[start:start+1024])).numpy()
                    outputs.append((z[:, 2] - z[:, 0]).astype(np.float64))
            return np.concatenate(outputs)
        base = margins(ddos_values)
        individual_batch = np.repeat(ddos_values[None, :, :], 69, axis=0)
        for feature in range(69): individual_batch[feature, :, feature] = replacement[feature]
        individual_margins = margins(individual_batch.reshape(-1, 69)).reshape(69, 16)
        impacts = np.mean(np.abs(individual_margins - base[None, :]), axis=1)
        for feature in range(69):
            feature_rows.append({"physical_state_key": key, "seed": seed, "family_id": family, "state": state,
                                 "feature_index": feature, "mean_absolute_margin_shap": baseline_profiles[key][feature],
                                 "mean_absolute_perturbation_impact": impacts[feature]})
        rho = rank_metric(baseline_profiles[key], impacts); observed: dict[int, float] = {}; percentiles: dict[int, float] = {}
        for k in (1, 5, 10, 20):
            selected = np.asarray(sorted(top(baseline_profiles[key], k)), dtype=np.int64)
            changed = ddos_values.copy(); changed[:, selected] = replacement[selected]
            observed[k] = float(np.mean(np.abs(margins(changed)-base)))
            control_batch = np.repeat(ddos_values[None, :, :], 256, axis=0)
            for p, panel in enumerate(random_panels[k]): control_batch[p][:, panel] = replacement[panel]
            control_margins = margins(control_batch.reshape(-1, 69)).reshape(256, 16)
            controls = np.mean(np.abs(control_margins - base[None, :]), axis=1)
            for p in range(256):
                random_rows.append({"physical_state_key": key, "k": k, "panel_index": p, "random_panel_impact": controls[p]})
            percentiles[k] = float(np.mean(controls <= observed[k]))
        faith_rows.append({"physical_state_key": key, "seed": seed, "family_id": family, "state": state,
                           "feature_impact_spearman_rho": rho,
                           **{f"top{k}_observed_impact": observed[k] for k in (1,5,10,20)},
                           **{f"top{k}_random_percentile": percentiles[k] for k in (1,5,10,20)},
                           "top10_above_random_median": percentiles[10] >= 0.5})
        print(f"FAITHFULNESS [{i+1}/28]:", key)
    feature_frame, faith, random_frame = pd.DataFrame(feature_rows), pd.DataFrame(faith_rows), pd.DataFrame(random_rows)
    atomic_csv(tables / "task56c2_feature_perturbation_impacts.csv", feature_frame)
    atomic_csv(tables / "task56c2_faithfulness_state_summary.csv", faith)
    atomic_csv(tables / "task56c2_faithfulness_random_controls.csv", random_frame)

    sample_rows: list[dict[str, Any]] = []; specificity_rows: list[dict[str, Any]] = []
    ddos_set = set(ddos[16].tolist()); non_ddos = np.asarray([i for i in range(128) if i not in ddos_set], dtype=np.int64)
    for _, row in ready.iterrows():
        seed, family, state = int(row.seed), str(row.family_id), str(row.state); key = state_key(seed, family, state)
        profiles = np.abs(baseline_tensors[key]); dd = profiles[ddos[16]]; other = profiles[non_ddos].mean(axis=0)
        differences = []
        for position in range(16):
            within = np.delete(dd, position, axis=0).mean(axis=0)
            within_rho = rank_metric(dd[position], within); between_rho = rank_metric(dd[position], other); difference = within_rho-between_rho
            differences.append(difference); sample_rows.append({"physical_state_key": key, "ddos_sample_position": position,
                                                                 "within_spearman_rho": within_rho, "between_spearman_rho": between_rho,
                                                                 "within_minus_between": difference})
        specificity_rows.append({"physical_state_key": key, "seed": seed, "family_id": family, "state": state,
                                 "median_within_minus_between": float(np.median(differences)),
                                 "fraction_positive_samples": float(np.mean(np.asarray(differences) > 0))})
    specificity_samples, specificity = pd.DataFrame(sample_rows), pd.DataFrame(specificity_rows)
    atomic_csv(tables / "task56c2_class_specificity_samples.csv", specificity_samples)
    atomic_csv(tables / "task56c2_class_specificity_state_summary.csv", specificity)

    cross_rows: list[dict[str, Any]] = []
    conditions = [("clean_reference", "shared", "clean_reference")] + [(f"{f}_{s}", f, s) for f in FAMILIES for s in ("suspicious", "reconstructed")]
    for condition, family, state in conditions:
        for left, right in combinations(SEEDS, 2):
            lkey, rkey = state_key(left, family, state), state_key(right, family, state)
            cross_rows.append({"condition": condition, "seed_left": left, "seed_right": right,
                               "left_physical_state_key": lkey, "right_physical_state_key": rkey,
                               "spearman_rho": rank_metric(baseline_profiles[lkey], baseline_profiles[rkey])})
    cross = pd.DataFrame(cross_rows); atomic_csv(tables / "task56c2_cross_seed_consistency.csv", cross)

    rng = np.random.default_rng(int(config["multiplicity_and_uncertainty"]["bootstrap_seed"])); reps = int(config["multiplicity_and_uncertainty"]["bootstrap_replicates"])
    metric_rows: list[dict[str, Any]] = []
    def metric(domain: str, name: str, frame: pd.DataFrame, column: str, stat: Callable[[np.ndarray], float], threshold: float, direction: str) -> None:
        value = stat(frame[column].to_numpy(float)); low, high = cluster_ci(frame, "physical_state_key", column, stat, rng, reps)
        passed = value >= threshold if direction == "at_least" else value <= threshold
        metric_rows.append({"domain": domain, "metric": name, "value": value, "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                            "bootstrap_replicates": reps, "bootstrap_unit": "physical_checkpoint_state", "threshold": threshold,
                            "direction": direction, "passed": bool(passed)})
    median = lambda x: float(np.median(x)); q05 = lambda x: float(np.quantile(x, .05)); positive = lambda x: float(np.mean(x > 0)); above03 = lambda x: float(np.mean(x >= .30)); true_fraction = lambda x: float(np.mean(x >= .5))
    metric("repeated_run_stability", "median_spearman", repeat, "spearman_rho", median, .90, "at_least")
    metric("repeated_run_stability", "fifth_percentile_spearman", repeat, "spearman_rho", q05, .75, "at_least")
    metric("repeated_run_stability", "median_top10_jaccard", repeat, "top10_jaccard", median, .67, "at_least")
    metric("repeated_run_stability", "median_normalized_l1_drift", repeat, "normalized_l1_drift", median, .25, "at_most")
    metric("background_sensitivity", "median_spearman", background, "spearman_rho", median, .80, "at_least")
    metric("background_sensitivity", "fifth_percentile_spearman", background, "spearman_rho", q05, .60, "at_least")
    metric("background_sensitivity", "median_top10_jaccard", background, "top10_jaccard", median, .54, "at_least")
    metric("probe_size_sensitivity", "median_spearman", probe, "spearman_rho", median, .80, "at_least")
    metric("probe_size_sensitivity", "fifth_percentile_spearman", probe, "spearman_rho", q05, .60, "at_least")
    metric("probe_size_sensitivity", "median_top10_jaccard", probe, "top10_jaccard", median, .54, "at_least")
    metric("feature_perturbation_faithfulness", "median_state_spearman", faith, "feature_impact_spearman_rho", median, .50, "at_least")
    metric("feature_perturbation_faithfulness", "fraction_states_positive_spearman", faith, "feature_impact_spearman_rho", positive, .75, "at_least")
    metric("feature_perturbation_faithfulness", "median_top10_random_percentile", faith, "top10_random_percentile", median, .90, "at_least")
    metric("feature_perturbation_faithfulness", "fraction_states_top10_above_random_median", faith, "top10_random_percentile", true_fraction, .75, "at_least")
    metric("class_specificity", "median_state_difference", specificity, "median_within_minus_between", median, .10, "at_least")
    metric("class_specificity", "fraction_states_positive", specificity, "median_within_minus_between", positive, .75, "at_least")
    # Cross-seed uncertainty uses one average value per physical checkpoint state; point gates use all 42 frozen comparisons.
    cross_state_rows = []
    for _, row in ready.iterrows():
        key = state_key(int(row.seed), str(row.family_id), str(row.state))
        values = cross.loc[(cross.left_physical_state_key == key) | (cross.right_physical_state_key == key), "spearman_rho"]
        cross_state_rows.append({"physical_state_key": key, "state_mean_spearman": float(values.mean())})
    cross_state = pd.DataFrame(cross_state_rows)
    for name, stat, threshold in (("median_spearman", median, .50), ("fraction_comparisons_above_0p3", above03, .75)):
        point = median(cross.spearman_rho.to_numpy()) if name == "median_spearman" else above03(cross.spearman_rho.to_numpy())
        low, high = cluster_ci(cross_state, "physical_state_key", "state_mean_spearman", stat, rng, reps)
        metric_rows.append({"domain": "cross_seed_consistency", "metric": name, "value": point, "bootstrap_ci95_low": low,
                            "bootstrap_ci95_high": high, "bootstrap_replicates": reps, "bootstrap_unit": "physical_checkpoint_state",
                            "threshold": threshold, "direction": "at_least", "passed": point >= threshold})
    metrics = pd.DataFrame(metric_rows); atomic_csv(tables / "task56c2_preregistered_metric_decisions.csv", metrics)
    domains = metrics.groupby("domain", sort=False).agg(metrics_passed=("passed", "sum"), metric_count=("passed", "size")).reset_index()
    domains["domain_pass"] = domains.metrics_passed == domains.metric_count
    atomic_csv(tables / "task56c2_domain_decisions.csv", domains)
    d = dict(zip(domains.domain, domains.domain_pass)); core = d["repeated_run_stability"] and d["feature_perturbation_faithfulness"]
    robust = sum(d[name] for name in ("background_sensitivity", "probe_size_sensitivity", "class_specificity", "cross_seed_consistency"))
    result = "PASS" if core and robust == 4 else ("PARTIAL" if core and robust >= 2 else "FAIL")
    runtime = pd.read_csv(a.execution_root.resolve() / "tables" / "task56c2_evaluation_runtime.csv")
    runtime_summary = {"evaluation_count": len(runtime), "median_wall_seconds": float(runtime.wall_seconds.median()),
                       "total_wall_seconds": float(runtime.wall_seconds.sum()), "maximum_peak_rss_bytes": int(runtime.process_peak_rss_bytes.max())}
    decision = {"experiment_version": "4.18.C2", "stage": "task56_xai_validation_summary", "scientific_result": result,
                "core_domains_passed": bool(core), "robustness_domains_passed": int(robust), "domain_results": {k: bool(v) for k,v in d.items()},
                "physical_checkpoint_states": 28, "new_shap_state_evaluations": 140, "repeat_comparisons": 84,
                "background_comparisons": 56, "probe_size_comparisons": 56, "faithfulness_states": 28,
                "class_specificity_states": 28, "cross_seed_comparisons": 42, "runtime_and_memory": runtime_summary,
                "task57_primary_claim_allowed": result == "PASS", "task57_exploratory_only": result == "PARTIAL",
                "task57_recovery_claim_blocked": result == "FAIL", "negative_results_retained": True,
                "arrays_materialized": loaded, "training_permitted": False, "reserved_test_arrays_materialized": False, "complete": True}
    atomic_json(out / "task56c2_validation_decision.json", decision)
    report = ["# Task 56 C2 XAI validation closeout", "", f"Overall preregistered result: **{result}**", "",
              "No model training or reserved test-array access occurred. All negative results are retained.", "", "## Domain decisions", ""]
    for _, domain_row in domains.iterrows():
        report.append(f"- {domain_row['domain']}: **{'PASS' if domain_row['domain_pass'] else 'FAIL'}** ({int(domain_row['metrics_passed'])}/{int(domain_row['metric_count'])} gates)")
    report += ["", "## Preregistered metrics", "", "| Domain | Metric | Value | Threshold | Pass |", "|---|---|---:|---:|:---:|"]
    for _, metric_row in metrics.iterrows():
        comparator = ">=" if metric_row["direction"] == "at_least" else "<="
        report.append(f"| {metric_row['domain']} | {metric_row['metric']} | {metric_row['value']:.4f} | {comparator} {metric_row['threshold']:.4f} | {'Yes' if metric_row['passed'] else 'No'} |")
    report += ["", "## Task 57 gate", "", f"- Primary recovery claim allowed: **{result == 'PASS'}**",
               f"- Exploratory-only status: **{result == 'PARTIAL'}**", f"- Recovery claim blocked: **{result == 'FAIL'}**", ""]
    (out / "TASK56_C2_CLOSEOUT.md").write_text("\n".join(report), encoding="utf-8")
    print("===== TASK 56 C2 XAI VALIDATION DECISION =====")
    for name, passed in d.items(): print(name.upper() + ":", "PASS" if passed else "FAIL")
    print("OVERALL PREREGISTERED RESULT:", result)
    print("TASK 57 PRIMARY CLAIM ALLOWED:", result == "PASS")
    print("TASK 57 EXPLORATORY ONLY:", result == "PARTIAL")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    return 0


if __name__ == "__main__": raise SystemExit(main())

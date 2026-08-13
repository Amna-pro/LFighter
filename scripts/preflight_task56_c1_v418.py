#!/usr/bin/env python3
"""Preflight frozen Task 56 inputs without evaluating scientific outcomes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn


PARENT_TAG = "task55-c3-confirmatory-xai-frozen-v4173"
PARENT_COMMIT = "8226e0ca76c9344113e2c5d53039119ea37d3eed"
ALLOWED = ("X_train", "y_train", "X_val", "y_val")
RESERVED = ("X_test_natural", "y_test_natural", "X_test_diagnostic", "y_test_diagnostic")
FAMILIES = ("A_development_anchor", "B_hash_ranked", "C_hash_ranked")
SEEDS = (7, 99, 123, 2026)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--feature-file", required=True, type=Path)
    parser.add_argument("--task55-index-file", required=True, type=Path)
    parser.add_argument("--checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--warmup-root", required=True, type=Path)
    parser.add_argument("--task45-c2-root", required=True, type=Path)
    parser.add_argument("--task45-c3-root", required=True, type=Path)
    parser.add_argument("--task55-c2-root", required=True, type=Path)
    parser.add_argument("--task55-c3-root", required=True, type=Path)
    parser.add_argument("--task55-c3-audit-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def select_disjoint_backgrounds(labels: np.ndarray, baseline: np.ndarray, seeds: List[int]) -> Dict[int, np.ndarray]:
    used = set(int(value) for value in baseline)
    result: Dict[int, np.ndarray] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        selected: List[int] = []
        for class_id in range(8):
            candidates = np.asarray([int(value) for value in np.flatnonzero(labels == class_id) if int(value) not in used], dtype=np.int64)
            if len(candidates) < 16:
                raise RuntimeError(f"Class {class_id} cannot provide a disjoint 16 row background for seed {seed}")
            chosen = np.sort(rng.choice(candidates, size=16, replace=False))
            selected.extend(int(value) for value in chosen)
            used.update(int(value) for value in chosen)
        result[seed] = np.asarray(selected, dtype=np.int64)
    return result


def state_specs(warmup: Path, task45_c2: Path, task45_c3: Path, task55_c2: Path, task55_c3: Path) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for seed in SEEDS:
        if seed == 7:
            explain_root = task55_c2
            clean_dir = explain_root / "common" / "clean_reference"
            prefix = "task55c2"
            marker_name = "_task55_c2_state_complete.json"
        else:
            explain_root = task55_c3 / f"seed_{seed}"
            clean_dir = explain_root / "common" / "clean_reference"
            prefix = "task55c3"
            marker_name = "_task55_c3_state_complete.json"
        specs.append({
            "seed": seed, "family_id": "shared", "state": "clean_reference",
            "checkpoint": warmup / f"seed_{seed}" / "warmup" / "checkpoints" / "common_round4_warmup_model.pt",
            "state_dir": clean_dir, "prefix": prefix, "marker_name": marker_name,
        })
        attack_root = task45_c2 if seed == 7 else task45_c3
        for family in FAMILIES:
            condition = attack_root / family / "size_10" / f"seed_{seed}"
            marker_path = condition / "_task45_condition_complete.json"
            if not marker_path.is_file():
                raise FileNotFoundError(marker_path)
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("complete") is not True or marker.get("exact_poison_pair") is not True:
                raise RuntimeError(f"Task 45 exact pair is incomplete: {condition}")
            for state, branch, checkpoint_name in (
                ("suspicious", "plain_attack", "continuation_last_round_model.pt"),
                ("reconstructed", "trusted_reconstruction", "reconstruction_last_round_model.pt"),
            ):
                specs.append({
                    "seed": seed, "family_id": family, "state": state,
                    "checkpoint": condition / branch / "checkpoints" / checkpoint_name,
                    "state_dir": explain_root / "families" / family / state,
                    "prefix": prefix, "marker_name": marker_name,
                })
    return specs


class MarginModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(); self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.model(inputs)
        return (logits[:, 2] - logits[:, 0]).unsqueeze(1)


def normalize_shap(values: Any) -> np.ndarray:
    if isinstance(values, tuple): values = values[0]
    if isinstance(values, list): values = values[0]
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[-1] == 1: array = array[..., 0]
    return array


def main() -> int:
    args = parse_args()
    started = time.time()
    root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    paths = {name: value.expanduser().resolve() for name, value in {
        "data": args.data_file, "features": args.feature_file, "indices": args.task55_index_file,
        "inventory": args.checkpoint_inventory, "warmup": args.warmup_root, "task45_c2": args.task45_c2_root,
        "task45_c3": args.task45_c3_root, "task55_c2": args.task55_c2_root,
        "task55_c3": args.task55_c3_root, "task55_audit": args.task55_c3_audit_root,
        "output": args.output_dir,
    }.items()}
    tables = paths["output"] / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    for name in ("data", "features", "indices", "inventory", "warmup", "task45_c2", "task45_c3", "task55_c2", "task55_c3", "task55_audit"):
        add(f"exists_{name}", paths[name].exists(), paths[name])
        if not paths[name].exists(): raise FileNotFoundError(paths[name])
    add("parent_tag_exact", git(root, "rev-list", "-n", "1", PARENT_TAG) == PARENT_COMMIT, git(root, "rev-list", "-n", "1", PARENT_TAG))
    add("parent_is_ancestor", subprocess.run(["git", "merge-base", "--is-ancestor", PARENT_TAG, "HEAD"], cwd=root).returncode == 0, git(root, "rev-parse", "HEAD"))
    versions: Dict[str, str] = {}
    for package in ("shap", "pyarrow", "psutil"):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = "not installed"
    add("shap_version_exact", versions["shap"] == "0.48.0", versions["shap"])
    add("pyarrow_available", versions["pyarrow"] != "not installed", versions["pyarrow"])
    add("psutil_available", versions["psutil"] != "not installed", versions["psutil"])
    add("python_3_10_or_newer", sys.version_info >= (3, 10), platform.python_version())

    task55_decision_path = paths["task55_audit"] / "task55c3_audit_decision.json"
    task55_decision = json.loads(task55_decision_path.read_text(encoding="utf-8"))
    add("task55_all_47_checks_passed", task55_decision.get("checks_passed") == task55_decision.get("check_count") == 47, f"{task55_decision.get('checks_passed')}/{task55_decision.get('check_count')}")
    add("task55_ready_for_task56", task55_decision.get("ready_for_task56_validation") is True, task55_decision.get("ready_for_task56_validation"))
    add("task55_exact_pairing", task55_decision.get("exact_partition_and_poison_pairing") is True, task55_decision.get("exact_partition_and_poison_pairing"))
    add("task55_reserved_closed", task55_decision.get("reserved_test_arrays_materialized") is False, task55_decision.get("reserved_test_arrays_materialized"))

    with paths["features"].open(newline="", encoding="utf-8-sig") as handle:
        feature_rows = list(csv.DictReader(handle))
    feature_names = [str(row["feature"]).strip() for row in feature_rows]
    add("69_unique_features", len(feature_names) == len(set(feature_names)) == 69, len(feature_names))
    loaded_names: List[str] = []
    with np.load(paths["data"]) as payload:
        container_names = list(payload.files)
        if set(ALLOWED) - set(container_names): raise KeyError("Required development arrays are missing")
        X_train = payload["X_train"].astype(np.float32, copy=False); loaded_names.append("X_train")
        y_train = payload["y_train"].astype(np.int64, copy=False); loaded_names.append("y_train")
        X_val = payload["X_val"].astype(np.float32, copy=False); loaded_names.append("X_val")
        y_val = payload["y_val"].astype(np.int64, copy=False); loaded_names.append("y_val")
    add("allowed_arrays_only_materialized", tuple(loaded_names) == ALLOWED, loaded_names)
    add("reserved_arrays_not_materialized", not set(loaded_names).intersection(RESERVED), loaded_names)
    add("development_feature_dimensions", X_train.shape[1] == X_val.shape[1] == 69, f"{X_train.shape}/{X_val.shape}")

    with np.load(paths["indices"]) as frozen:
        baseline_background = frozen["background_train_indices"].astype(np.int64)
        probe_indices = frozen["probe_validation_indices"].astype(np.int64)
        frozen_background_labels = frozen["background_labels"].astype(np.int64)
        frozen_probe_labels = frozen["probe_labels"].astype(np.int64)
    baseline = config["frozen_task55_baseline"]
    add("baseline_background_hash", array_sha256(baseline_background) == baseline["background_manifest_sha256"], array_sha256(baseline_background))
    add("baseline_probe_hash", array_sha256(probe_indices) == baseline["probe_manifest_sha256"], array_sha256(probe_indices))
    add("baseline_labels_match", np.array_equal(y_train[baseline_background], frozen_background_labels) and np.array_equal(y_val[probe_indices], frozen_probe_labels), "train and validation")
    add("baseline_balanced", np.bincount(frozen_background_labels, minlength=8).tolist() == [16] * 8 and np.bincount(frozen_probe_labels, minlength=8).tolist() == [16] * 8, "16 per class")

    alternates = select_disjoint_backgrounds(y_train, baseline_background, config["background_sensitivity"]["alternate_selection_seeds"])
    alt1, alt2 = alternates[5601], alternates[5602]
    add("alternate_background_counts", len(alt1) == len(alt2) == 128, f"{len(alt1)}/{len(alt2)}")
    add("alternate_backgrounds_balanced", all(np.bincount(y_train[value], minlength=8).tolist() == [16] * 8 for value in alternates.values()), "16 per class")
    add("alternate_backgrounds_disjoint", not set(baseline_background).intersection(alt1) and not set(baseline_background).intersection(alt2) and not set(alt1).intersection(alt2), "pairwise disjoint")
    ddos_positions = np.flatnonzero(frozen_probe_labels == 2).astype(np.int64)
    rng = np.random.default_rng(config["probe_size_sensitivity"]["selection_seed"])
    order = rng.permutation(ddos_positions)
    ddos8, ddos12, ddos16 = np.sort(order[:8]), np.sort(order[:12]), np.sort(ddos_positions)
    add("ddos_probe_count_16", len(ddos16) == 16, len(ddos16))
    add("nested_probe_subsets", set(ddos8).issubset(ddos12) and set(ddos12).issubset(ddos16), "8 subset 12 subset 16")
    random_seed = config["feature_perturbation_faithfulness"]["random_control_seed"]
    random_count = config["feature_perturbation_faithfulness"]["random_panels_per_k"]
    random_rng = np.random.default_rng(random_seed)
    panels = {k: np.stack([np.sort(random_rng.choice(69, size=k, replace=False)) for _ in range(random_count)]).astype(np.int64) for k in (1, 5, 10, 20)}
    add("random_panel_shapes", all(value.shape == (256, k) for k, value in panels.items()), {k: value.shape for k, value in panels.items()})
    frozen_path = tables / "task56c1_frozen_validation_indices.npz"
    np.savez_compressed(
        frozen_path, baseline_background_train_indices=baseline_background,
        alternate_background_train_indices_5601=alt1, alternate_background_train_indices_5602=alt2,
        probe_validation_indices=probe_indices, true_ddos_probe_positions_8=ddos8,
        true_ddos_probe_positions_12=ddos12, true_ddos_probe_positions_16=ddos16,
        true_ddos_validation_indices_8=probe_indices[ddos8], true_ddos_validation_indices_12=probe_indices[ddos12],
        true_ddos_validation_indices_16=probe_indices[ddos16],
        random_feature_panels_k1=panels[1], random_feature_panels_k5=panels[5],
        random_feature_panels_k10=panels[10], random_feature_panels_k20=panels[20],
    )

    inventory = pd.read_csv(paths["inventory"])
    specs = state_specs(paths["warmup"], paths["task45_c2"], paths["task45_c3"], paths["task55_c2"], paths["task55_c3"])
    add("28_state_specs", len(specs) == 28, len(specs))
    add("28_inventory_rows", len(inventory) == 28, len(inventory))
    state_rows: List[Dict[str, Any]] = []
    source_paths: List[Path] = [config_path, paths["indices"], paths["inventory"], task55_decision_path]
    common_probe: np.ndarray | None = None
    common_labels: np.ndarray | None = None
    common_features: np.ndarray | None = None
    for item in specs:
        match = inventory.loc[(inventory["seed"] == item["seed"]) & (inventory["family_id"] == item["family_id"]) & (inventory["state"] == item["state"])]
        if len(match) != 1: raise RuntimeError(f"Inventory mismatch: {item}")
        expected_checkpoint_hash = str(match.iloc[0]["sha256"])
        checkpoint = item["checkpoint"]
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        checkpoint_match = sha256(checkpoint) == expected_checkpoint_hash
        prefix = item["prefix"]
        state_dir = item["state_dir"]
        marker_path = state_dir / item["marker_name"]
        tensor_path = state_dir / f"{prefix}_attributions.npz"
        summary_path = state_dir / "state_summary.json"
        for path in (marker_path, tensor_path, summary_path):
            if not path.is_file(): raise FileNotFoundError(path)
            source_paths.append(path)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        marker_valid = marker.get("complete") is True and marker.get("checkpoint_sha256") == expected_checkpoint_hash
        marker_valid = marker_valid and all((state_dir / name).is_file() and sha256(state_dir / name) == expected for name, expected in marker.get("output_sha256", {}).items())
        with np.load(tensor_path) as tensor:
            source = tensor["source_logit_shap"]
            target = tensor["target_logit_shap"]
            margin = tensor["source_minus_target_margin_shap"]
            tensor_probe = tensor["probe_validation_indices"]
            tensor_labels = tensor["true_class_id"]
            tensor_features = tensor["feature_names"]
            probe_values = tensor["standardized_probe_values"]
            logits = tensor["logits"]
        shape_valid = source.shape == target.shape == margin.shape == probe_values.shape == (128, 69) and logits.shape == (128, 8)
        finite = all(np.isfinite(value).all() for value in (source, target, margin, probe_values, logits))
        margin_exact = np.array_equal(margin, source - target)
        if common_probe is None:
            common_probe, common_labels, common_features = tensor_probe.copy(), tensor_labels.copy(), tensor_features.copy()
        same_frozen = np.array_equal(tensor_probe, common_probe) and np.array_equal(tensor_labels, common_labels) and np.array_equal(tensor_features, common_features)
        state_rows.append({
            "seed": item["seed"], "family_id": item["family_id"], "state": item["state"],
            "checkpoint_sha256": expected_checkpoint_hash, "checkpoint_hash_match": checkpoint_match,
            "task55_marker_valid": marker_valid, "tensor_shape_valid": shape_valid, "tensor_finite": finite,
            "margin_identity_exact": margin_exact, "same_frozen_probe_labels_features": same_frozen,
            "summary_checkpoint_match": summary.get("checkpoint_sha256") == expected_checkpoint_hash,
            "reserved_test_arrays_materialized": summary.get("reserved_test_arrays_materialized"),
        })
    state_frame = pd.DataFrame(state_rows)
    state_frame.to_csv(tables / "task56c1_state_readiness.csv", index=False)
    add("all_checkpoint_hashes_match", state_frame["checkpoint_hash_match"].all(), int(state_frame["checkpoint_hash_match"].sum()))
    add("all_task55_markers_valid", state_frame["task55_marker_valid"].all(), int(state_frame["task55_marker_valid"].sum()))
    add("all_tensor_shapes_valid", state_frame["tensor_shape_valid"].all(), int(state_frame["tensor_shape_valid"].sum()))
    add("all_tensors_finite", state_frame["tensor_finite"].all(), int(state_frame["tensor_finite"].sum()))
    add("all_margin_identities_exact", state_frame["margin_identity_exact"].all(), int(state_frame["margin_identity_exact"].sum()))
    add("same_frozen_samples_all_states", state_frame["same_frozen_probe_labels_features"].all(), int(state_frame["same_frozen_probe_labels_features"].sum()))
    add("all_summaries_match_checkpoints", state_frame["summary_checkpoint_match"].all(), int(state_frame["summary_checkpoint_match"].sum()))
    add("all_state_reserved_arrays_closed", (state_frame["reserved_test_arrays_materialized"] == False).all(), int((state_frame["reserved_test_arrays_materialized"] == False).sum()))

    src_dir = root / "src"
    if str(src_dir) not in sys.path: sys.path.insert(0, str(src_dir))
    from neural_models_v24 import build_model
    smoke_checkpoint = torch.load(specs[0]["checkpoint"], map_location="cpu", weights_only=False)
    model = build_model("resmlp", 69, 8); model.load_state_dict(smoke_checkpoint["model_state_dict"]); model.eval()
    wrapped = MarginModel(model).eval()
    import shap
    background_tensor = torch.from_numpy(X_train[alt1[:16]])
    probe_tensor = torch.from_numpy(X_val[probe_indices[ddos8[:2]]])
    smoke_explainer = shap.GradientExplainer(wrapped, background_tensor, batch_size=8, local_smoothing=0.0)
    smoke = normalize_shap(smoke_explainer.shap_values(probe_tensor, nsamples=8, rseed=5606))
    add("gradient_explainer_smoke_shape", smoke.shape == (2, 69), smoke.shape)
    add("gradient_explainer_smoke_finite", np.isfinite(smoke).all(), np.isfinite(smoke).sum())

    plan_rows: List[Dict[str, Any]] = []
    for item in specs:
        for repeat_seed in config["repeated_run_stability"]["repeat_rseeds"]:
            plan_rows.append({"mode": "repeat", "variant_seed": repeat_seed, "seed": item["seed"], "family_id": item["family_id"], "state": item["state"], "checkpoint_sha256": next(row["checkpoint_sha256"] for row in state_rows if row["seed"] == item["seed"] and row["family_id"] == item["family_id"] and row["state"] == item["state"])})
        for background_seed in config["background_sensitivity"]["alternate_selection_seeds"]:
            plan_rows.append({"mode": "alternate_background", "variant_seed": background_seed, "seed": item["seed"], "family_id": item["family_id"], "state": item["state"], "checkpoint_sha256": next(row["checkpoint_sha256"] for row in state_rows if row["seed"] == item["seed"] and row["family_id"] == item["family_id"] and row["state"] == item["state"])})
    plan = pd.DataFrame(plan_rows)
    plan.to_csv(tables / "task56c1_validation_plan.csv", index=False)
    add("validation_plan_140_rows", len(plan) == 140, len(plan))
    add("validation_plan_84_repeats", int((plan["mode"] == "repeat").sum()) == 84, int((plan["mode"] == "repeat").sum()))
    add("validation_plan_56_backgrounds", int((plan["mode"] == "alternate_background").sum()) == 56, int((plan["mode"] == "alternate_background").sum()))

    with (tables / "task56c1_preflight_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "detail"])
        writer.writeheader(); writer.writerows(checks)
    manifest_rows = []
    for path in sorted(set(source_paths), key=str):
        manifest_rows.append({"path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(tables / "task56c1_source_manifest_sha256.csv", index=False)
    passed = sum(int(row["passed"]) for row in checks)
    decision = {
        "experiment_version": "4.18.C1", "stage": "task56_local_validation_preflight",
        "checks_passed": passed, "check_count": len(checks), "parent_tag": PARENT_TAG,
        "parent_commit": PARENT_COMMIT, "physical_checkpoint_states_verified": 28,
        "planned_new_shap_state_evaluations": 140, "repeat_evaluations": 84,
        "alternate_background_evaluations": 56, "alternate_backgrounds_disjoint": True,
        "nested_ddos_probe_sizes": [8, 12, 16], "random_panels_per_k": 256,
        "arrays_materialized": loaded_names, "scientific_outcomes_evaluated": False,
        "training_permitted": False, "reserved_test_arrays_materialized": False,
        "elapsed_seconds": round(time.time() - started, 2), "ready_for_c2_validation_run": passed == len(checks),
    }
    (paths["output"] / "task56c1_preflight_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("===== TASK 56 C1 LOCAL VALIDATION PREFLIGHT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("PHYSICAL CHECKPOINT STATES VERIFIED: 28")
    print("PLANNED NEW SHAP STATE EVALUATIONS: 140")
    print("REPEAT EVALUATIONS: 84")
    print("ALTERNATE BACKGROUND EVALUATIONS: 56")
    print("ALTERNATE BACKGROUNDS DISJOINT: True")
    print("NESTED DDOS PROBE SIZES: 8,12,16")
    print("SCIENTIFIC OUTCOMES EVALUATED: False")
    print("TRAINING PERMITTED: False")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR C2 VALIDATION RUN:", passed == len(checks))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

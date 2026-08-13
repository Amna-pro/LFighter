#!/usr/bin/env python3
"""Run the frozen Task 55 C2 seed 7 three family SHAP pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn


C1_TAG = "task55-c1-xai-preflight-frozen-v4171"
EXPECTED_C1_COMMIT = "abc05f1426f34baf3d103a459838140230a65f4a"
ALLOWED = ("X_train", "y_train", "X_val", "y_val")
RESERVED = (
    "X_test_natural",
    "y_test_natural",
    "X_test_diagnostic",
    "y_test_diagnostic",
)
FAMILIES = ("A_development_anchor", "B_hash_ranked", "C_hash_ranked")
CLASSES = (
    "Benign",
    "BruteForce",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web-Based",
)
SOURCE_ID = 2
TARGET_ID = 0
SEED = 7
SIZE = 10
OUTPUT_NAMES = ("source_logit", "target_logit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--feature-file", required=True, type=Path)
    parser.add_argument("--sample-index-file", required=True, type=Path)
    parser.add_argument("--checkpoint-inventory", type=Path)
    parser.add_argument("--warmup-root", required=True, type=Path)
    parser.add_argument("--task45-c2-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
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


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.parquet")
    frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    os.replace(temporary, path)


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def normalize_shap_values(values: Any, samples: int, features: int) -> np.ndarray:
    if isinstance(values, tuple):
        values = values[0]
    if isinstance(values, list):
        if len(values) != 2:
            raise RuntimeError(f"Expected two SHAP outputs, received {len(values)}")
        values = np.stack([np.asarray(item) for item in values], axis=-1)
    array = np.asarray(values)
    expected = (samples, features, 2)
    if array.shape == (2, samples, features):
        array = np.moveaxis(array, 0, -1)
    if array.shape != expected:
        raise RuntimeError(f"Unexpected SHAP shape {array.shape}; expected {expected}")
    return array.astype(np.float32, copy=False)


class SourceTargetLogitModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.model(inputs)
        return torch.stack((logits[:, SOURCE_ID], logits[:, TARGET_ID]), dim=1)


def load_feature_names(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "feature" not in rows[0]:
        raise RuntimeError("Feature file must contain a feature column")
    names = [str(row["feature"]).strip() for row in rows]
    if len(names) != 69 or len(set(names)) != 69:
        raise RuntimeError(f"Expected 69 unique feature names, found {len(names)}")
    return names


def expected_checkpoint_rows(
    inventory: pd.DataFrame,
    warmup_root: Path,
    task45_root: Path,
) -> List[Dict[str, Any]]:
    seed_rows = inventory.loc[inventory["seed"] == SEED].copy()
    clean_rows = seed_rows.loc[seed_rows["state"] == "clean_reference"]
    if len(clean_rows) != 1:
        raise RuntimeError("C1 inventory must contain one seed 7 clean checkpoint")

    clean_path = warmup_root / "seed_7" / "warmup" / "checkpoints" / "common_round4_warmup_model.pt"
    result: List[Dict[str, Any]] = [{
        "state_key": "clean_reference",
        "family_id": "shared",
        "state": "clean_reference",
        "checkpoint": clean_path,
        "expected_sha256": str(clean_rows.iloc[0]["sha256"]),
        "output_dir_parts": ("common", "clean_reference"),
    }]
    for family in FAMILIES:
        condition = task45_root / family / "size_10" / "seed_7"
        marker = condition / "_task45_condition_complete.json"
        if not marker.is_file():
            raise FileNotFoundError(marker)
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        if marker_payload.get("complete") is not True or marker_payload.get("exact_poison_pair") is not True:
            raise RuntimeError(f"Task 45 condition is not complete and exactly paired: {condition}")
        for state, branch, filename in (
            ("suspicious", "plain_attack", "continuation_last_round_model.pt"),
            ("reconstructed", "trusted_reconstruction", "reconstruction_last_round_model.pt"),
        ):
            match = seed_rows.loc[
                (seed_rows["family_id"] == family) & (seed_rows["state"] == state)
            ]
            if len(match) != 1:
                raise RuntimeError(f"C1 inventory mismatch for {family} {state}")
            checkpoint = condition / branch / "checkpoints" / filename
            result.append({
                "state_key": f"{family}__{state}",
                "family_id": family,
                "state": state,
                "checkpoint": checkpoint,
                "expected_sha256": str(match.iloc[0]["sha256"]),
                "output_dir_parts": ("families", family, state),
            })
    return result


def completed_state_valid(output_dir: Path, expected: Dict[str, Any], config_hash: str) -> bool:
    marker = output_dir / "_task55_c2_state_complete.json"
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("complete") is not True:
            return False
        if payload.get("checkpoint_sha256") != expected["expected_sha256"]:
            return False
        if payload.get("config_sha256") != config_hash:
            return False
        for filename, expected_hash in payload.get("output_sha256", {}).items():
            path = output_dir / filename
            if not path.is_file() or sha256(path) != expected_hash:
                return False
        return True
    except Exception:
        return False


def state_long_frame(
    family: str,
    state: str,
    checkpoint_hash: str,
    feature_names: Sequence[str],
    probe_indices: np.ndarray,
    probe_values: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    source_shap: np.ndarray,
    target_shap: np.ndarray,
    margin_shap: np.ndarray,
    background_hash: str,
    probe_hash: str,
    shap_version: str,
) -> pd.DataFrame:
    samples, features = source_shap.shape
    repeated_samples = np.repeat(np.arange(samples), features)
    tiled_features = np.tile(np.arange(features), samples)
    return pd.DataFrame({
        "family_id": family,
        "coalition_size": SIZE,
        "seed": SEED,
        "state": state,
        "probe_global_index": np.repeat(probe_indices, features),
        "probe_position": repeated_samples,
        "true_class_id": np.repeat(true_labels, features),
        "true_class_name": np.repeat(np.asarray(CLASSES, dtype=object)[true_labels], features),
        "predicted_class_id": np.repeat(predicted_labels, features),
        "predicted_class_name": np.repeat(np.asarray(CLASSES, dtype=object)[predicted_labels], features),
        "feature_index": tiled_features,
        "feature_name": np.tile(np.asarray(feature_names, dtype=object), samples),
        "standardized_feature_value": probe_values.reshape(-1),
        "source_logit_shap": source_shap.reshape(-1),
        "target_logit_shap": target_shap.reshape(-1),
        "source_minus_target_margin_shap": margin_shap.reshape(-1),
        "checkpoint_sha256": checkpoint_hash,
        "background_manifest_sha256": background_hash,
        "probe_manifest_sha256": probe_hash,
        "explainer_version": shap_version,
    })


def feature_summary_frame(
    family: str,
    state: str,
    feature_names: Sequence[str],
    true_labels: np.ndarray,
    source_shap: np.ndarray,
    target_shap: np.ndarray,
    margin_shap: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scope, mask in (
        ("all_probe_rows", np.ones(len(true_labels), dtype=bool)),
        ("true_DDoS_rows", true_labels == SOURCE_ID),
    ):
        for feature_index, feature_name in enumerate(feature_names):
            source = source_shap[mask, feature_index]
            target = target_shap[mask, feature_index]
            margin = margin_shap[mask, feature_index]
            rows.append({
                "family_id": family,
                "coalition_size": SIZE,
                "seed": SEED,
                "state": state,
                "scope": scope,
                "row_count": int(mask.sum()),
                "feature_index": feature_index,
                "feature_name": feature_name,
                "mean_source_logit_shap": float(source.mean()),
                "mean_absolute_source_logit_shap": float(np.abs(source).mean()),
                "mean_target_logit_shap": float(target.mean()),
                "mean_absolute_target_logit_shap": float(np.abs(target).mean()),
                "mean_margin_shap": float(margin.mean()),
                "mean_absolute_margin_shap": float(np.abs(margin).mean()),
            })
    return pd.DataFrame(rows)


def run_state(
    root: Path,
    output_root: Path,
    state_spec: Dict[str, Any],
    config: Dict[str, Any],
    config_hash: str,
    feature_names: Sequence[str],
    background_values: np.ndarray,
    probe_values: np.ndarray,
    probe_indices: np.ndarray,
    true_labels: np.ndarray,
    background_hash: str,
    probe_hash: str,
) -> Dict[str, Any]:
    output_dir = output_root.joinpath(*state_spec["output_dir_parts"])
    if completed_state_valid(output_dir, state_spec, config_hash):
        print("SKIP VERIFIED COMPLETE:", state_spec["state_key"])
        return json.loads((output_dir / "state_summary.json").read_text(encoding="utf-8"))

    checkpoint_path = state_spec["checkpoint"]
    print("EXPLAIN:", state_spec["state_key"])
    started = time.time()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from neural_models_v24 import build_model

    model = build_model("resmlp", 69, 8)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    wrapped = SourceTargetLogitModel(model).eval()

    background_tensor = torch.from_numpy(background_values)
    probe_tensor = torch.from_numpy(probe_values)
    with torch.no_grad():
        full_logits = model(probe_tensor).cpu().numpy().astype(np.float32)
        background_pair = wrapped(background_tensor).cpu().numpy().astype(np.float32)
    predicted = np.argmax(full_logits, axis=1).astype(np.int64)

    import shap

    explainer_config = config["explainer"]
    random.seed(int(explainer_config["rseed"]))
    np.random.seed(int(explainer_config["rseed"]))
    torch.manual_seed(int(explainer_config["rseed"]))
    explainer = shap.GradientExplainer(
        wrapped,
        background_tensor,
        batch_size=int(explainer_config["batch_size"]),
        local_smoothing=float(explainer_config["local_smoothing"]),
    )
    values = explainer.shap_values(
        probe_tensor,
        nsamples=int(explainer_config["nsamples"]),
        rseed=int(explainer_config["rseed"]),
    )
    shap_pair = normalize_shap_values(values, len(probe_values), len(feature_names))
    source_shap = shap_pair[:, :, 0]
    target_shap = shap_pair[:, :, 1]
    margin_shap = (source_shap - target_shap).astype(np.float32)

    if not np.isfinite(shap_pair).all() or not np.isfinite(margin_shap).all():
        raise RuntimeError(f"Nonfinite SHAP values for {state_spec['state_key']}")
    margin_identity_error = float(np.max(np.abs(margin_shap - (source_shap - target_shap))))
    if margin_identity_error != 0.0:
        raise RuntimeError(f"Derived margin identity failed: {margin_identity_error}")

    base_source, base_target = background_pair.mean(axis=0)
    source_residual = source_shap.sum(axis=1) - (full_logits[:, SOURCE_ID] - base_source)
    target_residual = target_shap.sum(axis=1) - (full_logits[:, TARGET_ID] - base_target)
    margin_logits = full_logits[:, SOURCE_ID] - full_logits[:, TARGET_ID]
    ddos_mask = true_labels == SOURCE_ID
    checkpoint_hash = sha256(checkpoint_path)
    shap_version = importlib.metadata.version("shap")

    tensor_path = output_dir / "task55c2_attributions.npz"
    long_path = output_dir / "task55c2_attributions_long.parquet"
    feature_path = output_dir / "task55c2_feature_summary.csv"
    summary_path = output_dir / "state_summary.json"
    write_npz_atomic(
        tensor_path,
        source_logit_shap=source_shap,
        target_logit_shap=target_shap,
        source_minus_target_margin_shap=margin_shap,
        standardized_probe_values=probe_values,
        probe_validation_indices=probe_indices,
        true_class_id=true_labels,
        predicted_class_id=predicted,
        logits=full_logits,
        feature_names=np.asarray(feature_names),
        output_names=np.asarray(("source_logit", "target_logit", "source_minus_target_margin")),
    )
    long_frame = state_long_frame(
        state_spec["family_id"], state_spec["state"], checkpoint_hash,
        feature_names, probe_indices, probe_values, true_labels, predicted,
        source_shap, target_shap, margin_shap, background_hash, probe_hash,
        shap_version,
    )
    write_parquet_atomic(long_path, long_frame)
    feature_frame = feature_summary_frame(
        state_spec["family_id"], state_spec["state"], feature_names,
        true_labels, source_shap, target_shap, margin_shap,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_frame.to_csv(feature_path, index=False)

    summary: Dict[str, Any] = {
        "experiment_version": "4.17.C2",
        "state_key": state_spec["state_key"],
        "family_id": state_spec["family_id"],
        "coalition_size": SIZE,
        "seed": SEED,
        "state": state_spec["state"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "partition_hash": checkpoint.get("partition_hash"),
        "poison_index_hash": checkpoint.get("poison_index_hash"),
        "global_round": checkpoint.get("global_round", checkpoint.get("round")),
        "probe_rows": int(len(probe_values)),
        "ddos_probe_rows": int(ddos_mask.sum()),
        "feature_count": int(len(feature_names)),
        "source_class": "DDoS",
        "target_class": "Benign",
        "mean_source_logit_true_ddos": float(full_logits[ddos_mask, SOURCE_ID].mean()),
        "mean_target_logit_true_ddos": float(full_logits[ddos_mask, TARGET_ID].mean()),
        "mean_margin_logit_true_ddos": float(margin_logits[ddos_mask].mean()),
        "ddos_to_benign_prediction_rate": float(np.mean(predicted[ddos_mask] == TARGET_ID)),
        "ddos_correct_prediction_rate": float(np.mean(predicted[ddos_mask] == SOURCE_ID)),
        "mean_absolute_margin_shap_true_ddos": float(np.abs(margin_shap[ddos_mask]).mean()),
        "maximum_source_additivity_residual": float(np.max(np.abs(source_residual))),
        "maximum_target_additivity_residual": float(np.max(np.abs(target_residual))),
        "derived_margin_identity_max_abs": margin_identity_error,
        "attribution_shape": list(shap_pair.shape),
        "background_manifest_sha256": background_hash,
        "probe_manifest_sha256": probe_hash,
        "shap_version": shap_version,
        "torch_version": torch.__version__,
        "elapsed_seconds": round(time.time() - started, 2),
        "reserved_test_arrays_materialized": False,
    }
    write_json_atomic(summary_path, summary)
    output_hashes = {
        tensor_path.name: sha256(tensor_path),
        long_path.name: sha256(long_path),
        feature_path.name: sha256(feature_path),
        summary_path.name: sha256(summary_path),
    }
    marker = {
        "experiment_version": "4.17.C2",
        "state_key": state_spec["state_key"],
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": config_hash,
        "background_manifest_sha256": background_hash,
        "probe_manifest_sha256": probe_hash,
        "output_sha256": output_hashes,
        "reserved_test_arrays_materialized": False,
        "complete": True,
    }
    write_json_atomic(output_dir / "_task55_c2_state_complete.json", marker)
    print("COMPLETE:", state_spec["state_key"], "seconds=", summary["elapsed_seconds"])
    return summary


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    data_file = args.data_file.expanduser().resolve()
    feature_file = args.feature_file.expanduser().resolve()
    index_file = args.sample_index_file.expanduser().resolve()
    inventory_path = (
        args.checkpoint_inventory.expanduser().resolve()
        if args.checkpoint_inventory
        else root / "results" / "cic_iot_diad_task55_c1_preflight_v417" / "tables" / "task55c1_checkpoint_inventory.csv"
    )
    warmup_root = args.warmup_root.expanduser().resolve()
    task45_root = args.task45_c2_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if args.threads < 1:
        raise ValueError("threads must be positive")
    for path in (config_path, data_file, feature_file, index_file, inventory_path, warmup_root, task45_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if importlib.metadata.version("shap") != "0.48.0":
        raise RuntimeError("Task 55 C2 requires SHAP 0.48.0")
    importlib.metadata.version("pyarrow")

    tag_commit = git(root, "rev-list", "-n", "1", C1_TAG)
    if tag_commit != EXPECTED_C1_COMMIT:
        raise RuntimeError(f"Unexpected C1 tag commit: {tag_commit}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", C1_TAG, "HEAD"], cwd=root).returncode != 0:
        raise RuntimeError("Frozen C1 tag is not an ancestor of HEAD")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("training_permitted") is not False or config.get("reserved_test_gate_opened") is not False:
        raise RuntimeError("Frozen Task 55 safety configuration is invalid")
    if config["frozen_primary_panel"]["development_seed"] != SEED:
        raise RuntimeError("C2 requires frozen development seed 7")
    feature_names = load_feature_names(feature_file)
    inventory = pd.read_csv(inventory_path)
    state_specs = expected_checkpoint_rows(inventory, warmup_root, task45_root)
    if len(state_specs) != 7:
        raise RuntimeError(f"Expected seven unique checkpoint states, found {len(state_specs)}")
    for state_spec in state_specs:
        checkpoint = state_spec["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        actual_hash = sha256(checkpoint)
        if actual_hash != state_spec["expected_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")

    with np.load(data_file) as payload:
        names_in_container = list(payload.files)
    missing = sorted(set(ALLOWED) - set(names_in_container))
    if missing:
        raise KeyError(f"Missing development arrays: {missing}")

    print("===== TASK 55 C2 SEED 7 PLAN =====")
    print("UNIQUE CHECKPOINT STATES:", len(state_specs))
    print("LOGICAL STATE COMPARISONS: 9")
    print("BACKGROUND ROWS: 128")
    print("PROBE ROWS: 128")
    print("SHAP NSAMPLES:", config["explainer"]["nsamples"])
    print("TRAINING PERMITTED: False")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    if args.dry_run:
        print("DRY RUN COMPLETE: True")
        return 0

    torch.set_num_threads(args.threads)
    random.seed(5517)
    np.random.seed(5517)
    torch.manual_seed(5517)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    loaded_names: List[str] = []
    with np.load(data_file) as payload:
        X_train = payload["X_train"].astype(np.float32, copy=False)
        loaded_names.append("X_train")
        y_train = payload["y_train"].astype(np.int64, copy=False)
        loaded_names.append("y_train")
        X_val = payload["X_val"].astype(np.float32, copy=False)
        loaded_names.append("X_val")
        y_val = payload["y_val"].astype(np.int64, copy=False)
        loaded_names.append("y_val")
    if tuple(loaded_names) != ALLOWED or set(loaded_names).intersection(RESERVED):
        raise RuntimeError(f"Unexpected array access: {loaded_names}")

    with np.load(index_file) as indices:
        background_indices = indices["background_train_indices"].astype(np.int64)
        probe_indices = indices["probe_validation_indices"].astype(np.int64)
        frozen_background_labels = indices["background_labels"].astype(np.int64)
        frozen_probe_labels = indices["probe_labels"].astype(np.int64)
    if len(background_indices) != 128 or len(probe_indices) != 128:
        raise RuntimeError("Frozen sample indices must contain 128 background and probe rows")
    if not np.array_equal(y_train[background_indices], frozen_background_labels):
        raise RuntimeError("Frozen background labels do not match training data")
    if not np.array_equal(y_val[probe_indices], frozen_probe_labels):
        raise RuntimeError("Frozen probe labels do not match validation data")

    c1_decision_path = root / "results" / "cic_iot_diad_task55_c1_preflight_v417" / "task55c1_preflight_decision.json"
    c1_decision = json.loads(c1_decision_path.read_text(encoding="utf-8"))
    background_hash = array_sha256(background_indices)
    probe_hash = array_sha256(probe_indices)
    if background_hash != c1_decision["background_manifest_sha256"]:
        raise RuntimeError("Frozen background index hash mismatch")
    if probe_hash != c1_decision["probe_manifest_sha256"]:
        raise RuntimeError("Frozen probe index hash mismatch")

    background_values = X_train[background_indices]
    probe_values = X_val[probe_indices]
    true_labels = y_val[probe_indices]
    config_hash = sha256(config_path)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for state_spec in state_specs:
        summary = run_state(
            root, output_root, state_spec, config, config_hash, feature_names,
            background_values, probe_values, probe_indices, true_labels,
            background_hash, probe_hash,
        )
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(output_root / "task55c2_progress.csv", index=False)

    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    state_summary = pd.DataFrame(summaries)
    state_summary.to_csv(tables / "task55c2_state_summary.csv", index=False)

    feature_frames: List[pd.DataFrame] = []
    for state_spec in state_specs:
        output_dir = output_root.joinpath(*state_spec["output_dir_parts"])
        feature_frames.append(pd.read_csv(output_dir / "task55c2_feature_summary.csv"))
    pd.concat(feature_frames, ignore_index=True).to_csv(
        tables / "task55c2_feature_summary.csv", index=False
    )

    logical_rows: List[Dict[str, Any]] = []
    clean_summary = next(row for row in summaries if row["state"] == "clean_reference")
    for family in FAMILIES:
        for state in ("clean_reference", "suspicious", "reconstructed"):
            row = clean_summary if state == "clean_reference" else next(
                item for item in summaries if item["family_id"] == family and item["state"] == state
            )
            logical_rows.append({
                "family_id": family,
                "coalition_size": SIZE,
                "seed": SEED,
                "state": state,
                "physical_state_key": row["state_key"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "background_manifest_sha256": background_hash,
                "probe_manifest_sha256": probe_hash,
            })
    pd.DataFrame(logical_rows).to_csv(tables / "task55c2_logical_state_manifest.csv", index=False)

    metadata = {
        "experiment_version": "4.17.C2",
        "stage": "task55_seed7_three_family_shap_pilot",
        "c1_tag": C1_TAG,
        "c1_commit": EXPECTED_C1_COMMIT,
        "seed": SEED,
        "coalition_size": SIZE,
        "families": list(FAMILIES),
        "unique_checkpoint_states": len(state_specs),
        "logical_state_comparisons": len(logical_rows),
        "background_rows": len(background_indices),
        "probe_rows": len(probe_indices),
        "feature_count": len(feature_names),
        "explainer": config["explainer"],
        "background_manifest_sha256": background_hash,
        "probe_manifest_sha256": probe_hash,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "shap_version": importlib.metadata.version("shap"),
        "pyarrow_version": importlib.metadata.version("pyarrow"),
        "arrays_materialized": loaded_names,
        "training_permitted": False,
        "reserved_test_arrays_materialized": False,
        "complete": True,
    }
    write_json_atomic(output_root / "task55c2_run_metadata.json", metadata)
    print("===== TASK 55 C2 EXECUTION COMPLETE =====")
    print("UNIQUE CHECKPOINT STATES:", len(state_specs))
    print("LOGICAL STATE COMPARISONS:", len(logical_rows))
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

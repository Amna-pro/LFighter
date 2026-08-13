#!/usr/bin/env python3
"""Run frozen Task 55 C3 SHAP attributions for confirmatory seeds."""
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
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


C2_TAG = "task55-c2-seed7-xai-frozen-v4172"
EXPECTED_C2_COMMIT = "324b3bd1e72a00e507b672ea425c477d711b794f"
ALLOWED = ("X_train", "y_train", "X_val", "y_val")
RESERVED = ("X_test_natural", "y_test_natural", "X_test_diagnostic", "y_test_diagnostic")
FAMILIES = ("A_development_anchor", "B_hash_ranked", "C_hash_ranked")
SEEDS = (99, 123, 2026)
CLASSES = ("Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-Based")
SOURCE_ID = 2
TARGET_ID = 0
SIZE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--feature-file", required=True, type=Path)
    parser.add_argument("--sample-index-file", required=True, type=Path)
    parser.add_argument("--checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--warmup-root", required=True, type=Path)
    parser.add_argument("--task45-c3-root", required=True, type=Path)
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


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def normalize_shap_values(values: Any, samples: int, features: int) -> np.ndarray:
    if isinstance(values, tuple):
        values = values[0]
    if isinstance(values, list):
        if len(values) != 2:
            raise RuntimeError(f"Expected two SHAP outputs, received {len(values)}")
        values = np.stack([np.asarray(item) for item in values], axis=-1)
    array = np.asarray(values)
    if array.shape == (2, samples, features):
        array = np.moveaxis(array, 0, -1)
    expected = (samples, features, 2)
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


def expected_checkpoint_rows(inventory: pd.DataFrame, warmup_root: Path, task45_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        seed_rows = inventory.loc[inventory["seed"] == seed]
        clean = seed_rows.loc[(seed_rows["family_id"] == "shared") & (seed_rows["state"] == "clean_reference")]
        if len(clean) != 1:
            raise RuntimeError(f"C1 inventory must contain one seed {seed} clean checkpoint")
        rows.append({
            "state_key": f"seed_{seed}__clean_reference",
            "seed": seed,
            "family_id": "shared",
            "state": "clean_reference",
            "checkpoint": warmup_root / f"seed_{seed}" / "warmup" / "checkpoints" / "common_round4_warmup_model.pt",
            "expected_sha256": str(clean.iloc[0]["sha256"]),
            "output_dir_parts": (f"seed_{seed}", "common", "clean_reference"),
        })
        for family in FAMILIES:
            condition = task45_root / family / "size_10" / f"seed_{seed}"
            marker_path = condition / "_task45_condition_complete.json"
            if not marker_path.is_file():
                raise FileNotFoundError(marker_path)
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("complete") is not True or marker.get("exact_poison_pair") is not True:
                raise RuntimeError(f"Task 45 condition is not complete and exactly paired: {condition}")
            for state, branch, filename in (
                ("suspicious", "plain_attack", "continuation_last_round_model.pt"),
                ("reconstructed", "trusted_reconstruction", "reconstruction_last_round_model.pt"),
            ):
                match = seed_rows.loc[(seed_rows["family_id"] == family) & (seed_rows["state"] == state)]
                if len(match) != 1:
                    raise RuntimeError(f"C1 inventory mismatch for seed {seed}, {family}, {state}")
                rows.append({
                    "state_key": f"seed_{seed}__{family}__{state}",
                    "seed": seed,
                    "family_id": family,
                    "state": state,
                    "checkpoint": condition / branch / "checkpoints" / filename,
                    "expected_sha256": str(match.iloc[0]["sha256"]),
                    "output_dir_parts": (f"seed_{seed}", "families", family, state),
                })
    return rows


def completed_state_valid(output_dir: Path, expected: Dict[str, Any], config_hash: str) -> bool:
    marker_path = output_dir / "_task55_c3_state_complete.json"
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("complete") is not True or marker.get("checkpoint_sha256") != expected["expected_sha256"]:
            return False
        if marker.get("config_sha256") != config_hash:
            return False
        hashes = marker.get("output_sha256", {})
        if set(hashes) != {"task55c3_attributions.npz", "task55c3_attributions_long.parquet", "task55c3_feature_summary.csv", "state_summary.json"}:
            return False
        return all((output_dir / name).is_file() and sha256(output_dir / name) == value for name, value in hashes.items())
    except Exception:
        return False


def state_long_frame(spec: Dict[str, Any], checkpoint_hash: str, feature_names: Sequence[str], probe_indices: np.ndarray,
                     probe_values: np.ndarray, true_labels: np.ndarray, predicted: np.ndarray, source: np.ndarray,
                     target: np.ndarray, margin: np.ndarray, background_hash: str, probe_hash: str,
                     shap_version: str) -> pd.DataFrame:
    samples, features = source.shape
    return pd.DataFrame({
        "family_id": spec["family_id"], "coalition_size": SIZE, "seed": spec["seed"], "state": spec["state"],
        "probe_global_index": np.repeat(probe_indices, features),
        "probe_position": np.repeat(np.arange(samples), features),
        "true_class_id": np.repeat(true_labels, features),
        "true_class_name": np.repeat(np.asarray(CLASSES, dtype=object)[true_labels], features),
        "predicted_class_id": np.repeat(predicted, features),
        "predicted_class_name": np.repeat(np.asarray(CLASSES, dtype=object)[predicted], features),
        "feature_index": np.tile(np.arange(features), samples),
        "feature_name": np.tile(np.asarray(feature_names, dtype=object), samples),
        "standardized_feature_value": probe_values.reshape(-1),
        "source_logit_shap": source.reshape(-1), "target_logit_shap": target.reshape(-1),
        "source_minus_target_margin_shap": margin.reshape(-1), "checkpoint_sha256": checkpoint_hash,
        "background_manifest_sha256": background_hash, "probe_manifest_sha256": probe_hash,
        "explainer_version": shap_version,
    })


def feature_summary_frame(spec: Dict[str, Any], feature_names: Sequence[str], true_labels: np.ndarray,
                          source: np.ndarray, target: np.ndarray, margin: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scope, mask in (("all_probe_rows", np.ones(len(true_labels), dtype=bool)), ("true_DDoS_rows", true_labels == SOURCE_ID)):
        for feature_index, feature_name in enumerate(feature_names):
            src, tgt, mar = source[mask, feature_index], target[mask, feature_index], margin[mask, feature_index]
            rows.append({
                "family_id": spec["family_id"], "coalition_size": SIZE, "seed": spec["seed"], "state": spec["state"],
                "scope": scope, "row_count": int(mask.sum()), "feature_index": feature_index, "feature_name": feature_name,
                "mean_source_logit_shap": float(src.mean()), "mean_absolute_source_logit_shap": float(np.abs(src).mean()),
                "mean_target_logit_shap": float(tgt.mean()), "mean_absolute_target_logit_shap": float(np.abs(tgt).mean()),
                "mean_margin_shap": float(mar.mean()), "mean_absolute_margin_shap": float(np.abs(mar).mean()),
            })
    return pd.DataFrame(rows)


def run_state(root: Path, output_root: Path, spec: Dict[str, Any], config: Dict[str, Any], config_hash: str,
              feature_names: Sequence[str], background_values: np.ndarray, probe_values: np.ndarray,
              probe_indices: np.ndarray, true_labels: np.ndarray, background_hash: str, probe_hash: str) -> Dict[str, Any]:
    output_dir = output_root.joinpath(*spec["output_dir_parts"])
    if completed_state_valid(output_dir, spec, config_hash):
        print("SKIP VERIFIED COMPLETE:", spec["state_key"])
        return json.loads((output_dir / "state_summary.json").read_text(encoding="utf-8"))

    print("EXPLAIN:", spec["state_key"])
    started = time.time()
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
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
        logits = model(probe_tensor).cpu().numpy().astype(np.float32)
        background_pair = wrapped(background_tensor).cpu().numpy().astype(np.float32)
    predicted = np.argmax(logits, axis=1).astype(np.int64)

    import shap
    explainer_config = config["explainer"]
    explain_seed = int(explainer_config["rseed"])
    random.seed(explain_seed)
    np.random.seed(explain_seed)
    torch.manual_seed(explain_seed)
    explainer = shap.GradientExplainer(wrapped, background_tensor, batch_size=int(explainer_config["batch_size"]),
                                      local_smoothing=float(explainer_config["local_smoothing"]))
    values = explainer.shap_values(probe_tensor, nsamples=int(explainer_config["nsamples"]), rseed=explain_seed)
    pair = normalize_shap_values(values, len(probe_values), len(feature_names))
    source, target = pair[:, :, 0], pair[:, :, 1]
    margin = (source - target).astype(np.float32)
    if not np.isfinite(pair).all() or not np.isfinite(margin).all():
        raise RuntimeError(f"Nonfinite SHAP values for {spec['state_key']}")
    identity_error = float(np.max(np.abs(margin - (source - target))))
    if identity_error != 0.0:
        raise RuntimeError(f"Derived margin identity failed: {identity_error}")

    base_source, base_target = background_pair.mean(axis=0)
    source_residual = source.sum(axis=1) - (logits[:, SOURCE_ID] - base_source)
    target_residual = target.sum(axis=1) - (logits[:, TARGET_ID] - base_target)
    ddos = true_labels == SOURCE_ID
    checkpoint_hash = sha256(spec["checkpoint"])
    shap_version = importlib.metadata.version("shap")
    tensor_path = output_dir / "task55c3_attributions.npz"
    long_path = output_dir / "task55c3_attributions_long.parquet"
    feature_path = output_dir / "task55c3_feature_summary.csv"
    summary_path = output_dir / "state_summary.json"
    write_npz_atomic(tensor_path, source_logit_shap=source, target_logit_shap=target,
                     source_minus_target_margin_shap=margin, standardized_probe_values=probe_values,
                     probe_validation_indices=probe_indices, true_class_id=true_labels, predicted_class_id=predicted,
                     logits=logits, feature_names=np.asarray(feature_names),
                     output_names=np.asarray(("source_logit", "target_logit", "source_minus_target_margin")))
    write_parquet_atomic(long_path, state_long_frame(spec, checkpoint_hash, feature_names, probe_indices, probe_values,
                                                     true_labels, predicted, source, target, margin, background_hash,
                                                     probe_hash, shap_version))
    write_csv_atomic(feature_path, feature_summary_frame(spec, feature_names, true_labels, source, target, margin))
    summary: Dict[str, Any] = {
        "experiment_version": "4.17.C3", "state_key": spec["state_key"], "family_id": spec["family_id"],
        "coalition_size": SIZE, "seed": spec["seed"], "state": spec["state"],
        "checkpoint_path": str(spec["checkpoint"]), "checkpoint_sha256": checkpoint_hash,
        "partition_hash": checkpoint.get("partition_hash"), "poison_index_hash": checkpoint.get("poison_index_hash"),
        "global_round": checkpoint.get("global_round", checkpoint.get("round")), "probe_rows": len(probe_values),
        "ddos_probe_rows": int(ddos.sum()), "feature_count": len(feature_names), "source_class": "DDoS",
        "target_class": "Benign", "mean_source_logit_true_ddos": float(logits[ddos, SOURCE_ID].mean()),
        "mean_target_logit_true_ddos": float(logits[ddos, TARGET_ID].mean()),
        "mean_margin_logit_true_ddos": float((logits[ddos, SOURCE_ID] - logits[ddos, TARGET_ID]).mean()),
        "ddos_to_benign_prediction_rate": float(np.mean(predicted[ddos] == TARGET_ID)),
        "ddos_correct_prediction_rate": float(np.mean(predicted[ddos] == SOURCE_ID)),
        "mean_absolute_margin_shap_true_ddos": float(np.abs(margin[ddos]).mean()),
        "maximum_source_additivity_residual": float(np.max(np.abs(source_residual))),
        "maximum_target_additivity_residual": float(np.max(np.abs(target_residual))),
        "derived_margin_identity_max_abs": identity_error, "attribution_shape": list(pair.shape),
        "background_manifest_sha256": background_hash, "probe_manifest_sha256": probe_hash,
        "shap_version": shap_version, "torch_version": torch.__version__,
        "elapsed_seconds": round(time.time() - started, 2), "reserved_test_arrays_materialized": False,
    }
    write_json_atomic(summary_path, summary)
    hashes = {path.name: sha256(path) for path in (tensor_path, long_path, feature_path, summary_path)}
    write_json_atomic(output_dir / "_task55_c3_state_complete.json", {
        "experiment_version": "4.17.C3", "state_key": spec["state_key"], "checkpoint_sha256": checkpoint_hash,
        "config_sha256": config_hash, "background_manifest_sha256": background_hash,
        "probe_manifest_sha256": probe_hash, "output_sha256": hashes,
        "reserved_test_arrays_materialized": False, "complete": True,
    })
    print("COMPLETE:", spec["state_key"], "seconds=", summary["elapsed_seconds"])
    return summary


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    paths = {name: value.expanduser().resolve() for name, value in {
        "config": args.config, "data": args.data_file, "features": args.feature_file,
        "indices": args.sample_index_file, "inventory": args.checkpoint_inventory,
        "warmup": args.warmup_root, "task45": args.task45_c3_root, "output": args.output_root,
    }.items()}
    if args.threads < 1:
        raise ValueError("threads must be positive")
    for name in ("config", "data", "features", "indices", "inventory", "warmup", "task45"):
        if not paths[name].exists():
            raise FileNotFoundError(paths[name])
    if importlib.metadata.version("shap") != "0.48.0":
        raise RuntimeError("Task 55 C3 requires SHAP 0.48.0")
    importlib.metadata.version("pyarrow")
    if git(root, "rev-list", "-n", "1", C2_TAG) != EXPECTED_C2_COMMIT:
        raise RuntimeError("Frozen C2 tag does not resolve to the expected commit")
    if subprocess.run(["git", "merge-base", "--is-ancestor", C2_TAG, "HEAD"], cwd=root).returncode != 0:
        raise RuntimeError("Frozen C2 tag is not an ancestor of HEAD")

    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    if config.get("training_permitted") is not False or config.get("reserved_test_gate_opened") is not False:
        raise RuntimeError("Frozen Task 55 safety configuration is invalid")
    if tuple(config["frozen_primary_panel"]["confirmatory_seeds"]) != SEEDS:
        raise RuntimeError("Confirmatory seed panel differs from preregistration")
    feature_names = load_feature_names(paths["features"])
    inventory = pd.read_csv(paths["inventory"])
    specs = expected_checkpoint_rows(inventory, paths["warmup"], paths["task45"])
    if len(specs) != 21:
        raise RuntimeError(f"Expected 21 unique checkpoint states, found {len(specs)}")
    for spec in specs:
        if not spec["checkpoint"].is_file():
            raise FileNotFoundError(spec["checkpoint"])
        if sha256(spec["checkpoint"]) != spec["expected_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {spec['checkpoint']}")
    with np.load(paths["data"]) as payload:
        names_in_container = list(payload.files)
    missing = sorted(set(ALLOWED) - set(names_in_container))
    if missing:
        raise KeyError(f"Missing development arrays: {missing}")

    print("===== TASK 55 C3 CONFIRMATORY PLAN =====")
    print("SEEDS:", ",".join(map(str, SEEDS)))
    print("UNIQUE CHECKPOINT STATES:", len(specs))
    print("LOGICAL STATE COMPARISONS: 27")
    print("BACKGROUND ROWS: 128")
    print("PROBE ROWS: 128")
    print("SHAP NSAMPLES:", config["explainer"]["nsamples"])
    print("TRAINING PERMITTED: False")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    if args.dry_run:
        print("DRY RUN COMPLETE: True")
        return 0

    torch.set_num_threads(args.threads)
    random.seed(5517); np.random.seed(5517); torch.manual_seed(5517)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    loaded_names: List[str] = []
    with np.load(paths["data"]) as payload:
        X_train = payload["X_train"].astype(np.float32, copy=False); loaded_names.append("X_train")
        y_train = payload["y_train"].astype(np.int64, copy=False); loaded_names.append("y_train")
        X_val = payload["X_val"].astype(np.float32, copy=False); loaded_names.append("X_val")
        y_val = payload["y_val"].astype(np.int64, copy=False); loaded_names.append("y_val")
    if tuple(loaded_names) != ALLOWED or set(loaded_names).intersection(RESERVED):
        raise RuntimeError(f"Unexpected array access: {loaded_names}")
    with np.load(paths["indices"]) as frozen:
        background_indices = frozen["background_train_indices"].astype(np.int64)
        probe_indices = frozen["probe_validation_indices"].astype(np.int64)
        background_labels = frozen["background_labels"].astype(np.int64)
        probe_labels = frozen["probe_labels"].astype(np.int64)
    if not np.array_equal(y_train[background_indices], background_labels) or not np.array_equal(y_val[probe_indices], probe_labels):
        raise RuntimeError("Frozen sample labels do not match development data")
    c1 = json.loads((root / "results/cic_iot_diad_task55_c1_preflight_v417/task55c1_preflight_decision.json").read_text(encoding="utf-8"))
    background_hash, probe_hash = array_sha256(background_indices), array_sha256(probe_indices)
    if background_hash != c1["background_manifest_sha256"] or probe_hash != c1["probe_manifest_sha256"]:
        raise RuntimeError("Frozen sample manifest hash mismatch")
    background_values, probe_values, true_labels = X_train[background_indices], X_val[probe_indices], y_val[probe_indices]
    config_hash = sha256(paths["config"])
    paths["output"].mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []
    for spec in specs:
        summaries.append(run_state(root, paths["output"], spec, config, config_hash, feature_names,
                                   background_values, probe_values, probe_indices, true_labels,
                                   background_hash, probe_hash))
        write_csv_atomic(paths["output"] / "task55c3_progress.csv", pd.DataFrame(summaries))
    tables = paths["output"] / "tables"
    write_csv_atomic(tables / "task55c3_state_summary.csv", pd.DataFrame(summaries))
    feature_frames = [pd.read_csv(paths["output"].joinpath(*spec["output_dir_parts"]) / "task55c3_feature_summary.csv") for spec in specs]
    write_csv_atomic(tables / "task55c3_feature_summary.csv", pd.concat(feature_frames, ignore_index=True))
    logical: List[Dict[str, Any]] = []
    for seed in SEEDS:
        clean = next(row for row in summaries if row["seed"] == seed and row["state"] == "clean_reference")
        for family in FAMILIES:
            for state in ("clean_reference", "suspicious", "reconstructed"):
                row = clean if state == "clean_reference" else next(item for item in summaries if item["seed"] == seed and item["family_id"] == family and item["state"] == state)
                logical.append({"family_id": family, "coalition_size": SIZE, "seed": seed, "state": state,
                                "physical_state_key": row["state_key"], "checkpoint_sha256": row["checkpoint_sha256"],
                                "background_manifest_sha256": background_hash, "probe_manifest_sha256": probe_hash})
    write_csv_atomic(tables / "task55c3_logical_state_manifest.csv", pd.DataFrame(logical))
    write_json_atomic(paths["output"] / "task55c3_run_metadata.json", {
        "experiment_version": "4.17.C3", "stage": "task55_confirmatory_multiseed_attribution_run",
        "c2_tag": C2_TAG, "c2_commit": EXPECTED_C2_COMMIT, "seeds": list(SEEDS), "coalition_size": SIZE,
        "families": list(FAMILIES), "unique_checkpoint_states": len(specs), "logical_state_comparisons": len(logical),
        "background_rows": len(background_indices), "probe_rows": len(probe_indices), "feature_count": len(feature_names),
        "explainer": config["explainer"], "background_manifest_sha256": background_hash,
        "probe_manifest_sha256": probe_hash, "python_version": platform.python_version(),
        "torch_version": torch.__version__, "shap_version": importlib.metadata.version("shap"),
        "pyarrow_version": importlib.metadata.version("pyarrow"), "arrays_materialized": loaded_names,
        "training_permitted": False, "reserved_test_arrays_materialized": False, "scientific_result_evaluated": False,
        "complete": True,
    })
    print("===== TASK 55 C3 EXECUTION COMPLETE =====")
    print("UNIQUE CHECKPOINT STATES:", len(specs))
    print("LOGICAL STATE COMPARISONS:", len(logical))
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

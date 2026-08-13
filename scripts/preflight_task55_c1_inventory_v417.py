#!/usr/bin/env python3
"""Inventory frozen Task 55 inputs and run a tiny GradientExplainer smoke test.

Only training and validation arrays are materialized. Reserved test arrays are
listed from the NPZ directory but are never indexed or loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


ALLOWED = ("X_train", "y_train", "X_val", "y_val")
RESERVED = (
    "X_test_natural",
    "y_test_natural",
    "X_test_diagnostic",
    "y_test_diagnostic",
)
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
FAMILIES = ("A_development_anchor", "B_hash_ranked", "C_hash_ranked")
SEEDS = (7, 99, 123, 2026)
EXPECTED_SHAP = "0.48.0"
FORBIDDEN_FEATURES = {
    "flow id",
    "src ip",
    "dst ip",
    "timestamp",
    "label",
    "src port",
    "dst port",
    "protocol",
    "class_name",
    "target",
    "fine_label",
    "source_file",
    "source_row",
    "row_hash",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--feature-file", required=True, type=Path)
    parser.add_argument("--warmup-root", required=True, type=Path)
    parser.add_argument("--c2-root", required=True, type=Path)
    parser.add_argument("--c3-root", required=True, type=Path)
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


def normalize_feature(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def stratified_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for class_id in range(len(CLASSES)):
        candidates = np.flatnonzero(labels == class_id)
        if len(candidates) < per_class:
            raise RuntimeError(
                f"Class {CLASSES[class_id]} has {len(candidates)} rows; {per_class} required"
            )
        chosen = np.sort(rng.choice(candidates, size=per_class, replace=False))
        selected.extend(int(value) for value in chosen)
    return np.asarray(selected, dtype=np.int64)


def checkpoint_inventory(path: Path, state: str, family: str, seed: int) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint has no model_state_dict: {path}")
    input_weight = state_dict.get("input_projection.0.weight")
    classifier_weight = state_dict.get("classifier.weight")
    if input_weight is None or classifier_weight is None:
        raise RuntimeError(f"Checkpoint is not the frozen ResidualMLP: {path}")
    return {
        "family_id": family,
        "coalition_size": 10,
        "seed": seed,
        "state": state,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "input_dim": int(input_weight.shape[1]),
        "hidden_width": int(input_weight.shape[0]),
        "num_classes": int(classifier_weight.shape[0]),
        "global_round": payload.get("global_round", payload.get("round")),
        "partition_hash": payload.get("partition_hash"),
        "poison_index_hash": payload.get("poison_index_hash"),
    }


class MarginModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.model(inputs)
        return (logits[:, 2] - logits[:, 0]).unsqueeze(1)


def normalize_shap_values(values: Any) -> np.ndarray:
    if isinstance(values, tuple):
        values = values[0]
    if isinstance(values, list):
        if len(values) != 1:
            raise RuntimeError(f"Unexpected SHAP output list length: {len(values)}")
        values = values[0]
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def main() -> int:
    args = parse_args()
    started = time.time()
    root = args.project_root.expanduser().resolve()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    data_file = args.data_file.expanduser().resolve()
    feature_file = args.feature_file.expanduser().resolve()
    warmup_root = args.warmup_root.expanduser().resolve()
    c2_root = args.c2_root.expanduser().resolve()
    c3_root = args.c3_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    for path in (data_file, feature_file, warmup_root, c2_root, c3_root):
        add(f"exists_{path.name}", path.exists(), path)
        if not path.exists():
            raise FileNotFoundError(path)

    shap_version = None
    try:
        shap_version = importlib.metadata.version("shap")
    except importlib.metadata.PackageNotFoundError:
        pass
    add("shap_version_exact", shap_version == EXPECTED_SHAP, shap_version or "not installed")
    add("python_3_10_or_newer", sys.version_info >= (3, 10), platform.python_version())

    with feature_file.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "feature" not in rows[0]:
        raise RuntimeError("Feature file must contain a feature column")
    features = [str(row["feature"]).strip() for row in rows]
    normalized_features = {normalize_feature(value) for value in features}
    disallowed = sorted(normalized_features.intersection(FORBIDDEN_FEATURES))
    add("feature_count_69", len(features) == 69, len(features))
    add("feature_names_unique", len(set(features)) == len(features), len(set(features)))
    add("no_forbidden_features", not disallowed, disallowed)

    loaded_names: List[str] = []
    with np.load(data_file) as payload:
        names_in_container = list(payload.files)
        missing = sorted(set(ALLOWED) - set(names_in_container))
        if missing:
            raise KeyError(f"Missing development arrays: {missing}")
        X_train = payload["X_train"].astype(np.float32, copy=False)
        loaded_names.append("X_train")
        y_train = payload["y_train"].astype(np.int64, copy=False)
        loaded_names.append("y_train")
        X_val = payload["X_val"].astype(np.float32, copy=False)
        loaded_names.append("X_val")
        y_val = payload["y_val"].astype(np.int64, copy=False)
        loaded_names.append("y_val")

    add("allowed_arrays_only_materialized", loaded_names == list(ALLOWED), loaded_names)
    add("reserved_arrays_not_materialized", not set(loaded_names).intersection(RESERVED), loaded_names)
    add("train_feature_dimension_69", X_train.ndim == 2 and X_train.shape[1] == 69, X_train.shape)
    add("validation_feature_dimension_69", X_val.ndim == 2 and X_val.shape[1] == 69, X_val.shape)
    add("train_rows_align", len(X_train) == len(y_train), f"{len(X_train)}:{len(y_train)}")
    add("validation_rows_align", len(X_val) == len(y_val), f"{len(X_val)}:{len(y_val)}")
    add("all_train_classes_present", set(np.unique(y_train)) == set(range(8)), np.unique(y_train).tolist())
    add("all_validation_classes_present", set(np.unique(y_val)) == set(range(8)), np.unique(y_val).tolist())

    background_indices = stratified_indices(y_train, 16, 5501)
    probe_indices = stratified_indices(y_val, 16, 5502)
    index_path = tables / "task55c1_frozen_sample_indices.npz"
    np.savez_compressed(
        index_path,
        background_train_indices=background_indices,
        probe_validation_indices=probe_indices,
        background_labels=y_train[background_indices],
        probe_labels=y_val[probe_indices],
    )
    add("background_count_128", len(background_indices) == 128, len(background_indices))
    add("probe_count_128", len(probe_indices) == 128, len(probe_indices))
    add("background_balanced", np.bincount(y_train[background_indices], minlength=8).tolist() == [16] * 8, np.bincount(y_train[background_indices], minlength=8).tolist())
    add("probe_balanced", np.bincount(y_val[probe_indices], minlength=8).tolist() == [16] * 8, np.bincount(y_val[probe_indices], minlength=8).tolist())

    c3_decision_path = root / "results" / "cic_iot_diad_task45_c3_audit_v416" / "task45c3_integrity_decision.json"
    c4_decision_path = root / "results" / "cic_iot_diad_task45_c4_summary_v416" / "task45c4_coalition_size_decision.json"
    c3_decision = json.loads(c3_decision_path.read_text(encoding="utf-8"))
    c4_decision = json.loads(c4_decision_path.read_text(encoding="utf-8"))
    add("task45_c3_72_conditions_verified", c3_decision.get("conditions_verified") == 72, c3_decision.get("conditions_verified"))
    add("task45_c3_exact_pairing", c3_decision.get("exact_poison_pairing_all") is True, c3_decision.get("exact_poison_pairing_all"))
    add("task45_c3_reserved_arrays_closed", c3_decision.get("reserved_test_arrays_materialized") is False, c3_decision.get("reserved_test_arrays_materialized"))
    add("task45_c4_closeout_complete", c4_decision.get("task45_development_and_validation_closeout_complete") is True, c4_decision.get("task45_development_and_validation_closeout_complete"))
    add("task45_c4_partial_retained", c4_decision.get("overall_preregistered_hypothesis_result") == "PARTIAL", c4_decision.get("overall_preregistered_hypothesis_result"))

    checkpoint_rows: List[Dict[str, Any]] = []
    unique_clean: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        warmup_path = warmup_root / f"seed_{seed}" / "warmup" / "checkpoints" / "common_round4_warmup_model.pt"
        if not warmup_path.is_file():
            raise FileNotFoundError(warmup_path)
        unique_clean[seed] = checkpoint_inventory(warmup_path, "clean_reference", "shared", seed)
        source_root = c2_root if seed == 7 else c3_root
        for family in FAMILIES:
            condition = source_root / family / "size_10" / f"seed_{seed}"
            marker = condition / "_task45_condition_complete.json"
            plain = condition / "plain_attack" / "checkpoints" / "continuation_last_round_model.pt"
            defended = condition / "trusted_reconstruction" / "checkpoints" / "reconstruction_last_round_model.pt"
            plain_adapter = condition / "plain_attack" / "task45_dev_only_adapter_manifest.json"
            defended_adapter = condition / "trusted_reconstruction" / "task45_dev_only_adapter_manifest.json"
            for path in (marker, plain, defended, plain_adapter, defended_adapter):
                if not path.is_file():
                    raise FileNotFoundError(path)
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
            plain_adapter_payload = json.loads(plain_adapter.read_text(encoding="utf-8"))
            defended_adapter_payload = json.loads(defended_adapter.read_text(encoding="utf-8"))
            add(f"complete_{family}_{seed}", marker_payload.get("complete") is True and marker_payload.get("exact_poison_pair") is True, marker)
            add(f"plain_reserved_closed_{family}_{seed}", plain_adapter_payload.get("reserved_arrays_materialized") is False, plain_adapter)
            add(f"defense_reserved_closed_{family}_{seed}", defended_adapter_payload.get("reserved_arrays_materialized") is False, defended_adapter)
            plain_row = checkpoint_inventory(plain, "suspicious", family, seed)
            defense_row = checkpoint_inventory(defended, "reconstructed", family, seed)
            add(f"paired_poison_hash_{family}_{seed}", plain_row["poison_index_hash"] == defense_row["poison_index_hash"], plain_row["poison_index_hash"])
            checkpoint_rows.extend([plain_row, defense_row])

    checkpoint_rows = [*unique_clean.values(), *checkpoint_rows]
    add("checkpoint_file_count_28", len(checkpoint_rows) == 28, len(checkpoint_rows))
    add("all_checkpoint_input_dims_69", all(row["input_dim"] == 69 for row in checkpoint_rows), sorted({row["input_dim"] for row in checkpoint_rows}))
    add("all_checkpoint_class_counts_8", all(row["num_classes"] == 8 for row in checkpoint_rows), sorted({row["num_classes"] for row in checkpoint_rows}))

    checkpoint_table = tables / "task55c1_checkpoint_inventory.csv"
    with checkpoint_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checkpoint_rows[0]))
        writer.writeheader()
        writer.writerows(checkpoint_rows)

    smoke_shape: Sequence[int] | None = None
    smoke_repeat_max_abs = None
    smoke_pass = False
    if shap_version == EXPECTED_SHAP:
        import shap

        src_dir = root / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from neural_models_v24 import build_model

        checkpoint = torch.load(Path(unique_clean[7]["path"]), map_location="cpu", weights_only=False)
        model = build_model("resmlp", 69, 8)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        wrapped = MarginModel(model).eval()
        background = torch.from_numpy(X_train[background_indices[:16]])
        probes = torch.from_numpy(X_val[probe_indices[:2]])
        explainer = shap.GradientExplainer(wrapped, background, batch_size=8, local_smoothing=0.0)
        np.random.seed(5517)
        torch.manual_seed(5517)
        first = normalize_shap_values(explainer.shap_values(probes, nsamples=8, rseed=5517))
        np.random.seed(5517)
        torch.manual_seed(5517)
        second = normalize_shap_values(explainer.shap_values(probes, nsamples=8, rseed=5517))
        smoke_shape = list(first.shape)
        smoke_repeat_max_abs = float(np.max(np.abs(first - second)))
        smoke_pass = first.shape == (2, 69) and np.isfinite(first).all() and smoke_repeat_max_abs <= 1e-6
    add("gradient_explainer_smoke", smoke_pass, f"shape={smoke_shape}, repeat_max_abs={smoke_repeat_max_abs}")

    with (tables / "task55c1_preflight_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    source_rows = []
    for path in (args.config.expanduser().resolve(), data_file, feature_file, c3_decision_path, c4_decision_path, index_path):
        source_rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    with (tables / "task55c1_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(source_rows)

    passed = sum(int(row["passed"]) for row in checks)
    decision = {
        "experiment_version": "4.17.C1",
        "stage": "task55_local_artifact_inventory_and_gradient_explainer_smoke",
        "checks_passed": passed,
        "check_count": len(checks),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "shap_version": shap_version,
        "feature_count": len(features),
        "arrays_present_in_container": names_in_container,
        "arrays_materialized": loaded_names,
        "reserved_test_arrays_materialized": False,
        "background_manifest_sha256": array_sha256(background_indices),
        "probe_manifest_sha256": array_sha256(probe_indices),
        "checkpoint_files_verified": len(checkpoint_rows),
        "gradient_explainer_smoke_shape": smoke_shape,
        "gradient_explainer_repeat_max_abs": smoke_repeat_max_abs,
        "elapsed_seconds": round(time.time() - started, 2),
        "ready_for_c2_seed7_pilot": passed == len(checks),
    }
    (output / "task55c1_preflight_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print("===== TASK 55 C1 LOCAL INVENTORY =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("FEATURES VERIFIED:", len(features))
    print("CHECKPOINT FILES VERIFIED:", len(checkpoint_rows))
    print("SHAP VERSION:", shap_version)
    print("GRADIENT EXPLAINER SMOKE:", smoke_pass)
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR C2 SEED 7 PILOT:", passed == len(checks))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

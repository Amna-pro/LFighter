#!/usr/bin/env python3
"""Run the preregistered Task 56 C2 SHAP validation evaluations."""
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
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn

FROZEN_TAG = "task56-c1-xai-validation-preflight-frozen-v4181"
FROZEN_COMMIT = "14d82bbb27e3f1423e0c6c842f4e4368a71fc164"
ALLOWED = ("X_train", "y_train", "X_val", "y_val")
RESERVED = ("X_test_natural", "y_test_natural", "X_test_diagnostic", "y_test_diagnostic")
FAMILIES = ("A_development_anchor", "B_hash_ranked", "C_hash_ranked")
SEEDS = (7, 99, 123, 2026)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-file", required=True, type=Path)
    p.add_argument("--feature-file", required=True, type=Path)
    p.add_argument("--frozen-index-file", required=True, type=Path)
    p.add_argument("--validation-plan", required=True, type=Path)
    p.add_argument("--checkpoint-readiness", required=True, type=Path)
    p.add_argument("--warmup-root", required=True, type=Path)
    p.add_argument("--task45-c2-root", required=True, type=Path)
    p.add_argument("--task45-c3-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **values)
    os.replace(tmp, path)


def git(root: Path, *items: str) -> str:
    return subprocess.run(["git", *items], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


class SourceTargetLogitModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return torch.stack((logits[:, 2], logits[:, 0]), dim=1)


def normalize(values: Any, rows: int) -> np.ndarray:
    if isinstance(values, tuple):
        values = values[0]
    if isinstance(values, list):
        if len(values) != 2: raise RuntimeError(f"Expected two SHAP outputs, received {len(values)}")
        values = np.stack([np.asarray(item) for item in values], axis=-1)
    a = np.asarray(values)
    if a.shape == (2, rows, 69): a = np.moveaxis(a, 0, -1)
    if a.shape != (rows, 69, 2):
        raise RuntimeError(f"Unexpected SHAP shape {a.shape}; expected {(rows, 69, 2)}")
    return a.astype(np.float32, copy=False)


def state_key(seed: int, family: str, state: str) -> str:
    return f"seed_{seed}__clean_reference" if state == "clean_reference" else f"seed_{seed}__{family}__{state}"


def checkpoint_path(row: pd.Series, warmup: Path, c2: Path, c3: Path) -> Path:
    seed, family, state = int(row.seed), str(row.family_id), str(row.state)
    if state == "clean_reference":
        return warmup / f"seed_{seed}" / "warmup" / "checkpoints" / "common_round4_warmup_model.pt"
    root = c2 if seed == 7 else c3
    branch, filename = (("plain_attack", "continuation_last_round_model.pt") if state == "suspicious"
                        else ("trusted_reconstruction", "reconstruction_last_round_model.pt"))
    return root / family / "size_10" / f"seed_{seed}" / branch / "checkpoints" / filename


def eval_dir(root: Path, row: pd.Series) -> Path:
    family = "shared" if row.state == "clean_reference" else str(row.family_id)
    return root / "evaluations" / str(row["mode"]) / f"variant_{int(row.variant_seed)}" / f"seed_{int(row.seed)}" / family / str(row.state)


def verified_complete(directory: Path, row: pd.Series, config_hash: str) -> bool:
    marker_path = directory / "_task56_c2_evaluation_complete.json"
    if not marker_path.is_file():
        return False
    try:
        m = json.loads(marker_path.read_text(encoding="utf-8"))
        if not m.get("complete") or m.get("config_sha256") != config_hash:
            return False
        if m.get("checkpoint_sha256") != str(row.checkpoint_sha256):
            return False
        if m.get("mode") != str(row["mode"]) or m.get("variant_seed") != int(row.variant_seed):
            return False
        return all((directory / name).is_file() and sha256(directory / name) == digest
                   for name, digest in m.get("output_sha256", {}).items())
    except Exception:
        return False


def main() -> int:
    a = args()
    root = a.project_root.resolve()
    paths = {k: v.resolve() for k, v in {
        "config": a.config, "data": a.data_file, "features": a.feature_file,
        "indices": a.frozen_index_file, "plan": a.validation_plan,
        "readiness": a.checkpoint_readiness, "warmup": a.warmup_root,
        "c2": a.task45_c2_root, "c3": a.task45_c3_root, "output": a.output_root,
    }.items()}
    for name in ("config", "data", "features", "indices", "plan", "readiness", "warmup", "c2", "c3"):
        if not paths[name].exists():
            raise FileNotFoundError(paths[name])
    if git(root, "rev-list", "-n", "1", FROZEN_TAG) != FROZEN_COMMIT:
        raise RuntimeError("Frozen Task 56 C1 tag does not resolve to the expected commit")
    if subprocess.run(["git", "merge-base", "--is-ancestor", FROZEN_TAG, "HEAD"], cwd=root).returncode:
        raise RuntimeError("Frozen Task 56 C1 tag is not an ancestor of HEAD")
    if importlib.metadata.version("shap") != "0.48.0":
        raise RuntimeError("Task 56 requires SHAP 0.48.0")
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    if config.get("training_permitted") is not False or config.get("reserved_test_gate_opened") is not False:
        raise RuntimeError("Frozen safety gate is invalid")
    plan = pd.read_csv(paths["plan"])
    ready = pd.read_csv(paths["readiness"])
    if len(plan) != 140 or (plan["mode"] == "repeat").sum() != 84 or (plan["mode"] == "alternate_background").sum() != 56:
        raise RuntimeError("Frozen 140-evaluation plan is invalid")
    if len(ready) != 28 or not ready[["checkpoint_hash_match", "task55_marker_valid", "tensor_shape_valid", "tensor_finite", "margin_identity_exact"]].all().all():
        raise RuntimeError("Frozen checkpoint readiness table is invalid")
    with paths["features"].open(newline="", encoding="utf-8-sig") as f:
        names = [str(row["feature"]).strip() for row in csv.DictReader(f)]
    if len(names) != 69 or len(set(names)) != 69:
        raise RuntimeError("Expected 69 unique features")
    for _, row in ready.iterrows():
        checkpoint = checkpoint_path(row, paths["warmup"], paths["c2"], paths["c3"])
        if not checkpoint.is_file() or sha256(checkpoint) != str(row.checkpoint_sha256):
            raise RuntimeError(f"Checkpoint mismatch: {checkpoint}")

    print("===== TASK 56 C2 VALIDATION PLAN =====")
    print("PHYSICAL CHECKPOINT STATES: 28")
    print("NEW SHAP STATE EVALUATIONS: 140")
    print("REPEAT EVALUATIONS: 84")
    print("ALTERNATE BACKGROUND EVALUATIONS: 56")
    print("CHECKPOINT RESUME MARKERS: ENABLED")
    print("TRAINING PERMITTED: False")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    if a.dry_run:
        print("DRY RUN COMPLETE: True")
        return 0

    loaded: list[str] = []
    with np.load(paths["data"]) as d:
        X_train = d["X_train"].astype(np.float32, copy=False); loaded.append("X_train")
        y_train = d["y_train"].astype(np.int64, copy=False); loaded.append("y_train")
        X_val = d["X_val"].astype(np.float32, copy=False); loaded.append("X_val")
        y_val = d["y_val"].astype(np.int64, copy=False); loaded.append("y_val")
    if tuple(loaded) != ALLOWED or set(loaded).intersection(RESERVED):
        raise RuntimeError(f"Unexpected array materialization: {loaded}")
    with np.load(paths["indices"]) as f:
        baseline = f["baseline_background_train_indices"].astype(np.int64)
        alternatives = {seed: f[f"alternate_background_train_indices_{seed}"].astype(np.int64) for seed in (5601, 5602)}
        probe_all = f["probe_validation_indices"].astype(np.int64)
        ddos_positions = f["true_ddos_probe_positions_16"].astype(np.int64)
    probe_indices = probe_all
    if len(probe_indices) != 128 or len(ddos_positions) != 16 or not np.all(y_val[probe_indices[ddos_positions]] == 2):
        raise RuntimeError("Frozen true-DDoS probe is invalid")
    baseline_hash = array_hash(baseline)
    if baseline_hash != config["frozen_task55_baseline"]["background_manifest_sha256"]:
        raise RuntimeError("Frozen baseline background hash mismatch")
    if array_hash(probe_indices) != config["frozen_task55_baseline"]["probe_manifest_sha256"]:
        raise RuntimeError("Frozen baseline probe hash mismatch")
    if any(len(value) != 128 or np.bincount(y_train[value], minlength=8).tolist() != [16] * 8 for value in alternatives.values()):
        raise RuntimeError("Alternate backgrounds are not balanced 128-row panels")
    if set(baseline).intersection(alternatives[5601]) or set(baseline).intersection(alternatives[5602]) or set(alternatives[5601]).intersection(alternatives[5602]):
        raise RuntimeError("Alternate backgrounds are not pairwise disjoint")
    config_hash = sha256(paths["config"])
    paths["output"].mkdir(parents=True, exist_ok=True)
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from neural_models_v24 import build_model
    import shap

    torch.set_num_threads(a.threads)
    summaries: list[dict[str, Any]] = []
    process = psutil.Process(os.getpid())
    for index, row in plan.iterrows():
        directory = eval_dir(paths["output"], row)
        if verified_complete(directory, row, config_hash):
            print(f"SKIP VERIFIED COMPLETE [{index + 1}/140]:", state_key(int(row.seed), str(row.family_id), str(row.state)), row["mode"], int(row.variant_seed))
            summaries.append(json.loads((directory / "evaluation_summary.json").read_text(encoding="utf-8")))
            continue
        checkpoint = checkpoint_path(row, paths["warmup"], paths["c2"], paths["c3"])
        background_indices = baseline if row["mode"] == "repeat" else alternatives[int(row.variant_seed)]
        explain_seed = int(row.variant_seed) if row["mode"] == "repeat" else 5517
        random.seed(explain_seed); np.random.seed(explain_seed); torch.manual_seed(explain_seed)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model("resmlp", 69, 8)
        model.load_state_dict(payload["model_state_dict"]); model.eval()
        wrapped = SourceTargetLogitModel(model).eval()
        background_tensor = torch.from_numpy(X_train[background_indices])
        probe_tensor = torch.from_numpy(X_val[probe_indices])
        peak = [process.memory_info().rss]
        stop = threading.Event()
        def sample_memory() -> None:
            while not stop.wait(0.05):
                try: peak[0] = max(peak[0], process.memory_info().rss)
                except psutil.Error: return
        monitor = threading.Thread(target=sample_memory, daemon=True); monitor.start()
        started = time.perf_counter()
        print(f"EXPLAIN [{index + 1}/140]:", state_key(int(row.seed), str(row.family_id), str(row.state)), row["mode"], int(row.variant_seed))
        explainer = shap.GradientExplainer(wrapped, background_tensor, batch_size=64, local_smoothing=0.0)
        pair = normalize(explainer.shap_values(probe_tensor, nsamples=128, rseed=explain_seed), 128)
        source, target = pair[:, :, 0], pair[:, :, 1]
        values = (source - target).astype(np.float32)
        wall = time.perf_counter() - started
        stop.set(); monitor.join()
        if not np.isfinite(values).all():
            raise RuntimeError("Nonfinite Task 56 SHAP tensor")
        tensor_path = directory / "task56c2_margin_attributions.npz"
        atomic_npz(tensor_path, source_logit_shap=source, target_logit_shap=target,
                   source_minus_target_margin_shap=values,
                   feature_names=np.asarray(names), probe_validation_indices=probe_indices,
                   true_ddos_probe_positions=ddos_positions, background_train_indices=background_indices)
        summary = {
            "experiment_version": "4.18.C2", "evaluation_index": int(index), "mode": str(row["mode"]),
            "variant_seed": int(row.variant_seed), "explainer_rseed": explain_seed,
            "seed": int(row.seed), "family_id": str(row.family_id), "state": str(row.state),
            "physical_state_key": state_key(int(row.seed), str(row.family_id), str(row.state)),
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "config_sha256": config_hash, "background_manifest_sha256": array_hash(background_indices),
            "probe_manifest_sha256": array_hash(probe_indices), "probe_rows": 128, "ddos_probe_rows": 16, "feature_count": 69,
            "wall_seconds": round(wall, 4), "process_peak_rss_bytes": int(peak[0]),
            "shap_version": importlib.metadata.version("shap"), "torch_version": torch.__version__,
            "training_permitted": False, "reserved_test_arrays_materialized": False,
        }
        summary_path = directory / "evaluation_summary.json"
        atomic_json(summary_path, summary)
        output_hashes = {p.name: sha256(p) for p in (tensor_path, summary_path)}
        atomic_json(directory / "_task56_c2_evaluation_complete.json", {
            "experiment_version": "4.18.C2", "complete": True, "mode": str(row["mode"]),
            "variant_seed": int(row.variant_seed), "checkpoint_sha256": str(row.checkpoint_sha256),
            "config_sha256": config_hash, "output_sha256": output_hashes,
            "training_permitted": False, "reserved_test_arrays_materialized": False,
        })
        summaries.append(summary)
        atomic_csv(paths["output"] / "task56c2_progress.csv", pd.DataFrame(summaries))
        print("COMPLETE: seconds=", round(wall, 2))
        del model, wrapped, explainer, pair, source, target, values, payload
    atomic_csv(paths["output"] / "tables" / "task56c2_evaluation_runtime.csv", pd.DataFrame(summaries))
    atomic_json(paths["output"] / "task56c2_execution_metadata.json", {
        "experiment_version": "4.18.C2", "stage": "task56_validation_shap_execution",
        "frozen_tag": FROZEN_TAG, "frozen_commit": FROZEN_COMMIT,
        "physical_checkpoint_states": 28, "new_shap_state_evaluations": len(summaries),
        "repeat_evaluations": sum(s["mode"] == "repeat" for s in summaries),
        "alternate_background_evaluations": sum(s["mode"] == "alternate_background" for s in summaries),
        "python_version": platform.python_version(), "arrays_materialized": loaded,
        "training_permitted": False, "reserved_test_arrays_materialized": False,
        "complete": len(summaries) == 140,
    })
    print("===== TASK 56 C2 SHAP EXECUTION COMPLETE =====")
    print("NEW SHAP STATE EVALUATIONS:", len(summaries))
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

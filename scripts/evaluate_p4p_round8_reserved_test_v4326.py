#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from federated_iot_v26 import CLASS_NAMES, NUM_CLASSES, evaluation_artifacts  # noqa: E402
from neural_models_v24 import build_model  # noqa: E402

SEEDS = [1379954285, 1886033230, 480705558, 1377035733, 1707771978]
ATTACKS = [
    "all_to_one_benign",
    "cyclic_shift",
    "multiclass_partial_cycle",
    "pairwise_swap",
    "random_flip",
]
DATA_FILE = (
    ROOT / "data" / "processed" / "cic_iot_diad_2024_v2_1_recovery_check"
    / "arrays" / "behavioral_only.npz"
)
PREFLIGHT = (
    ROOT / "results" / "reviewer_p4p_attacked_preflight_v4325"
    / "all_to_one_benign" / "seed_1379954285"
)
GRID = ROOT / "results" / "reviewer_p4p_matched_grid_v4325"
MANIFEST_EVID = ROOT / "reviewer_revision"

OUTPUT = ROOT / "results" / "reviewer_p4p_round8_test_evaluation_v4326"
TABLES = OUTPUT / "tables"
ACCESS = OUTPUT / "P4P_TEST_ACCESS_STARTED_v4326.json"
COMPLETE = OUTPUT / "P4P_ROUND8_TEST_EVALUATION_COMPLETE_v4326.json"

EXPECTED_PARTITION = "5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"
EXPECTED_TASK65_EVALUATOR_SHA = "805b39b24ee9ca20fa2ea484472928fad97a5b560f26fdad1787e3bdee84559a"
EXPECTED_P4P_RUNNER_SHA = "fa904d89f391eb41a2d5920e174abc89915a4040e6dee8e92d401e782c95a09f"
PRIMARY_ROUND = 8
EVAL_BATCH = 4096
THREADS = 6


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def condition_dir(seed: int, attack: str) -> Path:
    if attack == "all_to_one_benign" and seed == 1379954285:
        return PREFLIGHT
    return GRID / attack / f"seed_{seed}"


def checkpoint_path(seed: int, attack: str) -> Path:
    return (
        condition_dir(seed, attack)
        / "checkpoints" / "reviewer_round_checkpoints"
        / "global_round_08_model.pt"
    )


def completion_path(seed: int, attack: str) -> Path:
    return condition_dir(seed, attack) / "REVIEWER_P4P_MATCHED_COMPLETE.json"


def manifest_evidence_path(seed: int, attack: str) -> Path:
    return (
        MANIFEST_EVID
        / f"TASK65_ATTACK_MANIFEST_RECOVERY_SEED_{seed}_{attack.upper()}_v4324.json"
    )


def load_expected_poison(seed: int, attack: str) -> str:
    p = manifest_evidence_path(seed, attack)
    if not p.exists():
        raise FileNotFoundError(p)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("status") != "PASS":
        raise RuntimeError(f"Manifest evidence is not PASS: {p}")
    cond = obj.get("condition", {})
    if int(cond.get("model_seed", -1)) != seed:
        raise RuntimeError(f"Manifest seed mismatch: {p}")
    if cond.get("attack_type") != attack:
        raise RuntimeError(f"Manifest attack mismatch: {p}")
    return str(obj["poison_index_hash_sha256"])


def verify_sources() -> dict:
    task65_eval = ROOT / "scripts" / "evaluate_task65_c1_one_shot_v4282.py"
    p4p_runner = ROOT / "scripts" / "run_reviewer_p4p_matched_v4322.py"
    actual_task65 = sha256_file(task65_eval)
    actual_p4p = sha256_file(p4p_runner)
    if actual_task65 != EXPECTED_TASK65_EVALUATOR_SHA:
        raise RuntimeError(f"Task65 evaluator source hash mismatch: {actual_task65}")
    if actual_p4p != EXPECTED_P4P_RUNNER_SHA:
        raise RuntimeError(f"P4P runner source hash mismatch: {actual_p4p}")
    return {
        "task65_evaluator_sha256": actual_task65,
        "p4p_runner_sha256": actual_p4p,
        "federated_iot_v26_sha256": sha256_file(ROOT / "src" / "federated_iot_v26.py"),
        "neural_models_v24_sha256": sha256_file(ROOT / "src" / "neural_models_v24.py"),
    }


def verify_all_checkpoints() -> list[dict]:
    records = []
    for attack in ATTACKS:
        for seed in SEEDS:
            cdir = condition_dir(seed, attack)
            complete = completion_path(seed, attack)
            ckpt = checkpoint_path(seed, attack)
            if not cdir.exists():
                raise FileNotFoundError(cdir)
            if not complete.exists():
                raise FileNotFoundError(complete)
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)

            meta = json.loads(complete.read_text(encoding="utf-8"))
            expected_poison = load_expected_poison(seed, attack)
            checks = {
                "mode": meta.get("mode") == "strong_attack",
                "attack": meta.get("attack_type") == attack,
                "seed": int(meta.get("model_seed", -1)) == seed,
                "partition": meta.get("partition_hash_sha256") == EXPECTED_PARTITION,
                "poison": meta.get("poison_index_hash_sha256") == expected_poison,
                "test_boundary": meta.get("test_sets_accessed") is False,
                "retuning": meta.get("attack_specific_retuning") is False,
            }
            failed = [k for k, v in checks.items() if not v]
            if failed:
                raise RuntimeError(
                    f"Frozen P4P completion validation failed attack={attack} seed={seed}: {failed}"
                )

            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            if payload.get("arm") != "p4p_matched":
                raise RuntimeError(f"Checkpoint arm mismatch: {ckpt}")
            if int(payload.get("global_round", -1)) != PRIMARY_ROUND:
                raise RuntimeError(f"Checkpoint round mismatch: {ckpt}")
            if int(payload.get("model_seed", -1)) != seed:
                raise RuntimeError(f"Checkpoint seed mismatch: {ckpt}")
            if payload.get("partition_hash") != EXPECTED_PARTITION:
                raise RuntimeError(f"Checkpoint partition mismatch: {ckpt}")
            if payload.get("poison_index_hash") != expected_poison:
                raise RuntimeError(f"Checkpoint poison mismatch: {ckpt}")
            state = payload.get("model_state_dict")
            if not isinstance(state, dict):
                raise RuntimeError(f"model_state_dict missing: {ckpt}")
            first = state.get("input_projection.0.weight")
            if first is None or tuple(first.shape) != (256, 69):
                raise RuntimeError(f"Unexpected model input layer shape: {ckpt}")

            records.append({
                "attack_type": attack,
                "model_seed": seed,
                "checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_sha256": sha256_file(ckpt),
                "poison_index_hash_sha256": expected_poison,
                "source": (
                    "audited_preflight_reused"
                    if attack == "all_to_one_benign" and seed == 1379954285
                    else "full_grid"
                ),
            })
    if len(records) != 25:
        raise RuntimeError(f"Expected 25 checkpoints, found {len(records)}")
    return records


def confusion_long(matrix: np.ndarray, *, dataset: str, attack: str, seed: int, ckpt_sha: str):
    rows = []
    for true_id, true_name in enumerate(CLASS_NAMES):
        for pred_id, pred_name in enumerate(CLASS_NAMES):
            rows.append({
                "dataset": dataset,
                "attack_type": attack,
                "model_seed": seed,
                "global_round": PRIMARY_ROUND,
                "true_class_id": true_id,
                "true_class_name": true_name,
                "predicted_class_id": pred_id,
                "predicted_class_name": pred_name,
                "count": int(matrix[true_id, pred_id]),
                "checkpoint_sha256": ckpt_sha,
            })
    return rows


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(
            f"One-shot P4P reserved-test output already exists; refusing rerun: {OUTPUT}"
        )

    source_hashes = verify_sources()
    checkpoint_records = verify_all_checkpoints()

    OUTPUT.mkdir(parents=True, exist_ok=False)
    TABLES.mkdir(parents=True, exist_ok=False)

    data_sha = sha256_file(DATA_FILE)
    access = {
        "protocol": "reviewer_v4326_p4p_round8_reserved_test_evaluation",
        "time_unix": time.time(),
        "git_head": git_head(),
        "data_file": str(DATA_FILE.relative_to(ROOT)).replace("\\", "/"),
        "data_file_sha256": data_sha,
        "global_round_fixed": PRIMARY_ROUND,
        "conditions_fixed": 25,
        "prior_structural_test_array_shape_access": True,
        "prior_p4p_test_outcome_evaluation": False,
        "checkpoint_selection_used_test_outcomes": False,
        "post_outcome_retuning_permitted": False,
        "rerun_permitted": False,
        "source_hashes": source_hashes,
    }
    ACCESS.write_text(json.dumps(access, indent=2) + "\n", encoding="utf-8")

    # First P4P reserved-test outcome materialization occurs here.
    with np.load(DATA_FILE, allow_pickle=False) as data:
        required = [
            "X_test_natural",
            "y_test_natural",
            "X_test_diagnostic",
            "y_test_diagnostic",
        ]
        missing = [k for k in required if k not in data.files]
        if missing:
            raise KeyError(f"Missing reserved-test arrays: {missing}")
        test_sets = {
            "diagnostic": (
                np.asarray(data["X_test_diagnostic"], dtype=np.float32),
                np.asarray(data["y_test_diagnostic"], dtype=np.int64),
            ),
            "natural": (
                np.asarray(data["X_test_natural"], dtype=np.float32),
                np.asarray(data["y_test_natural"], dtype=np.int64),
            ),
        }

    if test_sets["diagnostic"][0].shape != (113210, 69):
        raise RuntimeError(f"Unexpected diagnostic X shape: {test_sets['diagnostic'][0].shape}")
    if test_sets["natural"][0].shape != (199990, 69):
        raise RuntimeError(f"Unexpected natural X shape: {test_sets['natural'][0].shape}")

    torch.set_num_threads(THREADS)
    metric_rows = []
    class_rows = []
    confusion_rows = []

    by_key = {(r["attack_type"], r["model_seed"]): r for r in checkpoint_records}
    started = time.perf_counter()

    for attack in ATTACKS:
        for seed in SEEDS:
            rec = by_key[(attack, seed)]
            ckpt = ROOT / rec["checkpoint"]
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)

            model = build_model("resmlp", 69, NUM_CLASSES)
            model.load_state_dict(payload["model_state_dict"])
            model.eval()

            for dataset in ("diagnostic", "natural"):
                X, y = test_sets[dataset]
                metrics, per_class, matrix, _prediction_table = evaluation_artifacts(
                    model, X, y, EVAL_BATCH, dataset
                )

                recalls = per_class["recall"].to_numpy(dtype=float)
                worst_recall = float(np.min(recalls))
                worst_idx = int(np.argmin(recalls))
                metric_rows.append({
                    "dataset": dataset,
                    "attack_type": attack,
                    "model_seed": seed,
                    "global_round": PRIMARY_ROUND,
                    "arm": "p4p_matched",
                    "accuracy": float(metrics["accuracy"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "weighted_f1": float(metrics["weighted_f1"]),
                    "mcc": float(metrics["mcc"]),
                    "log_loss": float(metrics["log_loss"]),
                    "ece_15bin": float(metrics["ece_15bin"]),
                    "worst_class_recall": worst_recall,
                    "worst_class_id": worst_idx,
                    "worst_class_name": CLASS_NAMES[worst_idx],
                    "checkpoint_sha256": rec["checkpoint_sha256"],
                    "poison_index_hash_sha256": rec["poison_index_hash_sha256"],
                    "checkpoint_source": rec["source"],
                })

                per_class = per_class.copy()
                per_class.insert(0, "dataset", dataset)
                per_class.insert(1, "attack_type", attack)
                per_class.insert(2, "model_seed", seed)
                per_class.insert(3, "global_round", PRIMARY_ROUND)
                per_class["checkpoint_sha256"] = rec["checkpoint_sha256"]
                class_rows.extend(per_class.to_dict(orient="records"))

                confusion_rows.extend(
                    confusion_long(
                        matrix,
                        dataset=dataset,
                        attack=attack,
                        seed=seed,
                        ckpt_sha=rec["checkpoint_sha256"],
                    )
                )

            del model

    metrics_df = pd.DataFrame(metric_rows)
    class_df = pd.DataFrame(class_rows)
    confusion_df = pd.DataFrame(confusion_rows)

    if len(metrics_df) != 50:
        raise RuntimeError(f"Expected 50 metric rows, got {len(metrics_df)}")
    if len(class_df) != 400:
        raise RuntimeError(f"Expected 400 class rows, got {len(class_df)}")
    if len(confusion_df) != 3200:
        raise RuntimeError(f"Expected 3200 confusion rows, got {len(confusion_df)}")

    metrics_path = TABLES / "p4p_round8_reserved_test_metrics.csv"
    class_path = TABLES / "p4p_round8_reserved_test_per_class.csv"
    confusion_path = TABLES / "p4p_round8_reserved_test_confusion_long.csv"

    metrics_df.to_csv(metrics_path, index=False)
    class_df.to_csv(class_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)

    output_hashes = {
        metrics_path.name: sha256_file(metrics_path),
        class_path.name: sha256_file(class_path),
        confusion_path.name: sha256_file(confusion_path),
        ACCESS.name: sha256_file(ACCESS),
    }

    completion = {
        "protocol": "reviewer_v4326_p4p_round8_reserved_test_evaluation",
        "status": "PASS",
        "time_unix": time.time(),
        "git_head": git_head(),
        "data_file_sha256": data_sha,
        "reserved_test_arrays_materialized_for_outcome_evaluation": True,
        "prior_structural_test_array_shape_access": True,
        "first_p4p_reserved_test_outcome_evaluation": True,
        "global_round": PRIMARY_ROUND,
        "checkpoint_selection_rule": "fixed_global_round_8_for_all_conditions",
        "best_round_selection_used": False,
        "post_outcome_retuning_used": False,
        "model_training_run": False,
        "p4p_federated_branch_rerun": False,
        "formal_cross_method_statistics_performed": False,
        "conditions_evaluated": 25,
        "datasets_evaluated": ["diagnostic", "natural"],
        "metric_rows": len(metrics_df),
        "per_class_rows": len(class_df),
        "confusion_rows": len(confusion_df),
        "evaluation_batch_size": EVAL_BATCH,
        "threads": THREADS,
        "elapsed_seconds": float(time.perf_counter() - started),
        "source_hashes": source_hashes,
        "output_sha256": output_hashes,
    }
    COMPLETE.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")

    print("P4P ROUND8 RESERVED TEST EVALUATION = PASS")
    print("CONDITIONS EVALUATED: 25")
    print("DATASETS EVALUATED: diagnostic,natural")
    print("METRIC ROWS:", len(metrics_df))
    print("PER CLASS ROWS:", len(class_df))
    print("CONFUSION ROWS:", len(confusion_df))
    print("GLOBAL ROUND FIXED: 8")
    print("BEST ROUND SELECTION USED: False")
    print("POST OUTCOME RETUNING USED: False")
    print("MODEL TRAINING RUN: False")
    print("P4P FEDERATED BRANCH RERUN: False")
    print("FORMAL CROSS METHOD STATISTICS PERFORMED: False")
    print("OUTPUT:", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

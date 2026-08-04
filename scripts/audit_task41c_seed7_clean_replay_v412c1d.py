#!/usr/bin/env python3
"""Task 41C.1d seed-7 clean replay reproducibility audit.

Runs two exact, attack-free replays of the frozen V3.10 clean warmup chronology
using development arrays only. Compares each replay with:
- the frozen common round-4 checkpoint,
- frozen per-round client transition signatures,
- frozen local-training metrics,
- frozen round-level validation metrics,
- the other current-runtime replay.

No D1/D2/D3 calibration is accepted by this script.
No reserved natural or diagnostic test array is loaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for search_path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import federated_iot_v26
import neural_models_v24
import run_true_warmup_v310
import transition_signature_features_v38

from federated_iot_v26 import (
    CLASS_NAMES,
    NUM_CLASSES,
    set_seed,
    sqrt_class_weights,
    train_local_model,
)
from neural_models_v24 import build_model
from run_targeted_label_flip_v292 import (
    load_fixed_partitions,
    read_clean_partition_hash,
)
from transition_signature_features_v38 import (
    balanced_probe_indices,
    class_conditional_probability_means,
    evaluate_validation,
    predict_probabilities,
)
from independent_anchor_v310 import weighted_average_states
from trusted_update_reconstruction_v312 import (
    checkpoint_sha256,
    state_max_abs_difference,
)

EXPERIMENT_VERSION = "4.12C.1d"
ALLOWED_ARRAYS = ("X_train", "y_train", "X_val", "y_val")
RESERVED_ARRAYS = (
    "X_test_natural",
    "y_test_natural",
    "X_test_diagnostic",
    "y_test_diagnostic",
)
SEED = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit exact clean warmup replay reproducibility for seed 7."
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--partition-file", type=Path, required=True)
    parser.add_argument("--clean-seed-dir", type=Path, required=True)
    parser.add_argument("--warmup-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--warmup-rounds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--probe-per-class", type=int, default=48)
    parser.add_argument("--probe-seed", type=int, default=3701)
    parser.add_argument("--source-class", default="DDoS")
    parser.add_argument("--target-class", default="Benign")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_development_arrays(path: Path) -> Dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    arrays: Dict[str, np.ndarray] = {}
    with np.load(resolved, allow_pickle=False) as archive:
        missing = [key for key in ALLOWED_ARRAYS if key not in archive.files]
        if missing:
            raise KeyError(f"Missing development arrays: {missing}")
        for key in ALLOWED_ARRAYS:
            arrays[key] = np.asarray(archive[key])
    return arrays


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def save_figure(figure: plt.Figure, base: Path) -> None:
    figure.tight_layout()
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def state_relative_l2_difference(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    if set(left) != set(right):
        return float("inf")
    numerator = 0.0
    denominator = 0.0
    for key in left:
        left_tensor = left[key].detach().cpu()
        right_tensor = right[key].detach().cpu()
        if torch.is_floating_point(left_tensor):
            difference = (
                left_tensor.to(torch.float64)
                - right_tensor.to(torch.float64)
            )
            numerator += float(torch.sum(difference * difference))
            denominator += float(
                torch.sum(
                    right_tensor.to(torch.float64)
                    * right_tensor.to(torch.float64)
                )
            )
        elif not torch.equal(left_tensor, right_tensor):
            return float("inf")
    return float(
        math.sqrt(max(numerator, 0.0))
        / max(math.sqrt(max(denominator, 0.0)), 1e-12)
    )


def dataframe_numeric_differences(
    replay: pd.DataFrame,
    frozen: pd.DataFrame,
    key_columns: Sequence[str],
    metric_columns: Sequence[str],
    comparison: str,
) -> pd.DataFrame:
    left = replay.copy()
    right = frozen.copy()
    for key in key_columns:
        left[key] = pd.to_numeric(left[key], errors="raise").astype(int)
        right[key] = pd.to_numeric(right[key], errors="raise").astype(int)
    merged = left.merge(
        right,
        on=list(key_columns),
        how="outer",
        suffixes=("_replay", "_frozen"),
        indicator=True,
        validate="one_to_one",
    )
    rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        base = {
            "comparison": comparison,
            **{key: row[key] for key in key_columns},
            "merge_status": row["_merge"],
        }
        for metric in metric_columns:
            replay_value = pd.to_numeric(
                pd.Series([row.get(f"{metric}_replay")]),
                errors="coerce",
            ).iloc[0]
            frozen_value = pd.to_numeric(
                pd.Series([row.get(f"{metric}_frozen")]),
                errors="coerce",
            ).iloc[0]
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "replay_value": replay_value,
                    "frozen_value": frozen_value,
                    "absolute_difference": (
                        abs(float(replay_value) - float(frozen_value))
                        if pd.notna(replay_value) and pd.notna(frozen_value)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def load_frozen_signatures(path: Path) -> Dict[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            int(key.split("_")[-1]): np.asarray(archive[key])
            for key in archive.files
        }


def run_replay(
    replicate: int,
    arrays: Dict[str, np.ndarray],
    client_indices: Mapping[int, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    X_train = arrays["X_train"].astype(np.float32, copy=False)
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    X_val = arrays["X_val"].astype(np.float32, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)

    set_seed(SEED)
    source_id = CLASS_NAMES.index(args.source_class)
    target_id = CLASS_NAMES.index(args.target_class)
    probe_indices = balanced_probe_indices(
        y_val, args.probe_per_class, args.probe_seed
    )
    X_probe = X_val[probe_indices]
    y_probe = y_val[probe_indices]
    class_weights = sqrt_class_weights(
        y_train, args.max_class_weight
    )
    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)

    round_rows: List[Dict[str, Any]] = []
    local_rows: List[Dict[str, Any]] = []
    matrices_by_round: Dict[int, np.ndarray] = {}
    started = time.time()

    print()
    print(f"Exact clean replay replicate {replicate}")

    for round_id in range(1, args.warmup_rounds + 1):
        round_started = time.time()
        reference_state = copy.deepcopy(model.state_dict())
        local_states: List[Mapping[str, torch.Tensor]] = []
        sample_counts: List[int] = []
        local_matrices: List[np.ndarray] = []
        current_training: List[Dict[str, Any]] = []

        for client_id in range(args.num_clients):
            indices = client_indices[client_id]
            local_model = build_model(
                "resmlp", X_train.shape[1], NUM_CLASSES
            )
            local_model.load_state_dict(reference_state)
            state, metrics = train_local_model(
                model=local_model,
                X=X_train[indices],
                y=y_train[indices],
                class_weights=class_weights,
                local_epochs=1,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                seed=SEED + round_id * 1000 + client_id,
            )
            probabilities = predict_probabilities(
                local_model,
                X_probe,
                args.evaluation_batch_size,
            )
            local_means = class_conditional_probability_means(
                probabilities, y_probe
            )
            local_states.append(state)
            sample_counts.append(int(len(indices)))
            local_matrices.append(local_means)
            row = {
                "warmup_round": round_id,
                "client_id": client_id,
                "client_samples": int(len(indices)),
                **metrics,
            }
            local_rows.append(row)
            current_training.append(row)
            del local_model

        model.load_state_dict(
            weighted_average_states(
                local_states,
                sample_counts,
                reference_state,
            )
        )
        matrices_by_round[round_id] = np.stack(
            local_matrices, axis=0
        )
        validation_metrics, validation_pair = evaluate_validation(
            model,
            X_val,
            y_val,
            source_id,
            target_id,
            args.evaluation_batch_size,
        )
        round_rows.append(
            {
                "warmup_round": round_id,
                "participating_samples": int(sum(sample_counts)),
                "mean_local_train_loss": float(
                    np.mean(
                        [row["local_train_loss"] for row in current_training]
                    )
                ),
                "mean_local_train_accuracy": float(
                    np.mean(
                        [
                            row["local_train_accuracy"]
                            for row in current_training
                        ]
                    )
                ),
                "round_seconds": float(time.time() - round_started),
                **{
                    f"val_{key}": value
                    for key, value in validation_metrics.items()
                },
                **{
                    f"val_{key}": value
                    for key, value in validation_pair.items()
                },
            }
        )
        print(
            f"replicate {replicate}, round {round_id:02d}, "
            f"macro F1={validation_metrics['macro_f1']:.8f}, "
            f"seconds={round_rows[-1]['round_seconds']:.1f}"
        )

    return {
        "replicate": replicate,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "round_rows": pd.DataFrame(round_rows),
        "local_rows": pd.DataFrame(local_rows),
        "matrices_by_round": matrices_by_round,
        "probe_indices": probe_indices,
        "total_seconds": float(time.time() - started),
    }


def main() -> int:
    args = parse_args()
    if args.replicates != 2:
        raise ValueError("Task 41C.1d is frozen to exactly two replays.")
    if args.num_clients != 20 or args.warmup_rounds != 4:
        raise ValueError("Task 41C.1d requires 20 clients and 4 rounds.")
    if args.probe_per_class != 48 or args.probe_seed != 3701:
        raise ValueError("Frozen probe configuration mismatch.")
    if args.source_class != "DDoS" or args.target_class != "Benign":
        raise ValueError("Frozen validation pair mismatch.")

    torch.set_num_threads(max(1, args.threads))

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}"
            )
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_development_arrays(args.data_file)
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)

    partition_file = args.partition_file.expanduser().resolve()
    clean_seed_dir = args.clean_seed_dir.expanduser().resolve()
    warmup_dir = args.warmup_dir.expanduser().resolve()

    client_indices, partition_hash = load_fixed_partitions(
        partition_file,
        expected_clients=args.num_clients,
        train_rows=len(y_train),
    )
    clean_hash = read_clean_partition_hash(clean_seed_dir)
    if clean_hash != partition_hash:
        raise RuntimeError("Clean partition hash mismatch.")

    frozen_checkpoint_path = (
        warmup_dir / "checkpoints" / "common_round4_warmup_model.pt"
    )
    frozen_signatures_path = (
        warmup_dir / "tables" / "warmup_local_signatures.npz"
    )
    frozen_local_path = (
        warmup_dir / "tables" / "warmup_local_training.csv"
    )
    frozen_round_path = (
        warmup_dir / "tables" / "warmup_round_metrics.csv"
    )
    frozen_metadata_path = (
        warmup_dir / "true_warmup_v310_metadata.json"
    )
    for path in (
        frozen_checkpoint_path,
        frozen_signatures_path,
        frozen_local_path,
        frozen_round_path,
        frozen_metadata_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    frozen_checkpoint = torch.load(
        frozen_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    frozen_signatures = load_frozen_signatures(
        frozen_signatures_path
    )
    frozen_local = pd.read_csv(frozen_local_path)
    frozen_round = pd.read_csv(frozen_round_path)
    frozen_metadata = json.loads(
        frozen_metadata_path.read_text(encoding="utf-8")
    )

    probe_indices = balanced_probe_indices(
        y_val, args.probe_per_class, args.probe_seed
    )
    probe_hash = sha256_array(probe_indices)
    if probe_hash != frozen_metadata["probe_hash_sha256"]:
        raise RuntimeError("Probe hash mismatch.")
    if partition_hash != frozen_metadata["partition_hash_sha256"]:
        raise RuntimeError("Frozen metadata partition mismatch.")

    source_rows = []
    source_objects = {
        "set_seed": federated_iot_v26.set_seed,
        "sqrt_class_weights": federated_iot_v26.sqrt_class_weights,
        "train_local_model": federated_iot_v26.train_local_model,
        "build_model": neural_models_v24.build_model,
        "balanced_probe_indices": (
            transition_signature_features_v38.balanced_probe_indices
        ),
        "predict_probabilities": (
            transition_signature_features_v38.predict_probabilities
        ),
        "class_conditional_probability_means": (
            transition_signature_features_v38
            .class_conditional_probability_means
        ),
        "evaluate_validation": (
            transition_signature_features_v38.evaluate_validation
        ),
        "frozen_warmup_main": run_true_warmup_v310.main,
    }
    for name, function in source_objects.items():
        source = inspect.getsource(function)
        source_rows.append(
            {
                "symbol": name,
                "module": function.__module__,
                "source_sha256": sha256_text(source),
                "source_line_count": len(source.splitlines()),
            }
        )

    replays = [
        run_replay(
            replicate=index,
            arrays=arrays,
            client_indices=client_indices,
            args=args,
        )
        for index in range(1, args.replicates + 1)
    ]

    state_rows = []
    frozen_state = frozen_checkpoint["model_state_dict"]
    for replay in replays:
        state_rows.append(
            {
                "comparison": (
                    f"replicate_{replay['replicate']}_vs_frozen"
                ),
                "max_abs_state_difference": state_max_abs_difference(
                    replay["model_state_dict"], frozen_state
                ),
                "relative_state_l2_difference": (
                    state_relative_l2_difference(
                        replay["model_state_dict"], frozen_state
                    )
                ),
            }
        )
    state_rows.append(
        {
            "comparison": "replicate_1_vs_replicate_2",
            "max_abs_state_difference": state_max_abs_difference(
                replays[0]["model_state_dict"],
                replays[1]["model_state_dict"],
            ),
            "relative_state_l2_difference": (
                state_relative_l2_difference(
                    replays[0]["model_state_dict"],
                    replays[1]["model_state_dict"],
                )
            ),
        }
    )
    state_frame = pd.DataFrame(state_rows)

    signature_rows = []
    for replay in replays:
        for round_id in range(1, args.warmup_rounds + 1):
            replay_matrix = replay["matrices_by_round"][round_id]
            frozen_matrix = frozen_signatures[round_id]
            difference = (
                replay_matrix.astype(np.float64)
                - frozen_matrix.astype(np.float64)
            )
            signature_rows.append(
                {
                    "comparison": (
                        f"replicate_{replay['replicate']}_vs_frozen"
                    ),
                    "warmup_round": round_id,
                    "maximum_absolute_difference": float(
                        np.max(np.abs(difference))
                    ),
                    "mean_absolute_difference": float(
                        np.mean(np.abs(difference))
                    ),
                    "rmse": float(
                        np.sqrt(np.mean(difference * difference))
                    ),
                }
            )
    for round_id in range(1, args.warmup_rounds + 1):
        difference = (
            replays[0]["matrices_by_round"][round_id].astype(np.float64)
            - replays[1]["matrices_by_round"][round_id].astype(np.float64)
        )
        signature_rows.append(
            {
                "comparison": "replicate_1_vs_replicate_2",
                "warmup_round": round_id,
                "maximum_absolute_difference": float(
                    np.max(np.abs(difference))
                ),
                "mean_absolute_difference": float(
                    np.mean(np.abs(difference))
                ),
                "rmse": float(
                    np.sqrt(np.mean(difference * difference))
                ),
            }
        )
    signature_frame = pd.DataFrame(signature_rows)

    local_metrics = [
        column
        for column in (
            "client_samples",
            "local_train_loss",
            "local_train_accuracy",
        )
        if column in frozen_local.columns
    ]
    local_difference_frames = []
    for replay in replays:
        local_difference_frames.append(
            dataframe_numeric_differences(
                replay["local_rows"],
                frozen_local,
                ("warmup_round", "client_id"),
                local_metrics,
                f"replicate_{replay['replicate']}_vs_frozen",
            )
        )
    local_difference_frame = pd.concat(
        local_difference_frames, ignore_index=True
    )

    round_metrics = [
        column
        for column in frozen_round.columns
        if column.startswith("val_")
        or column
        in (
            "participating_samples",
            "mean_local_train_loss",
            "mean_local_train_accuracy",
        )
    ]
    round_difference_frames = []
    for replay in replays:
        round_difference_frames.append(
            dataframe_numeric_differences(
                replay["round_rows"],
                frozen_round,
                ("warmup_round",),
                round_metrics,
                f"replicate_{replay['replicate']}_vs_frozen",
            )
        )
    round_difference_frame = pd.concat(
        round_difference_frames, ignore_index=True
    )

    replicate_state_row = state_frame.loc[
        state_frame["comparison"] == "replicate_1_vs_replicate_2"
    ].iloc[0]
    deterministic_current_runtime = bool(
        float(replicate_state_row["max_abs_state_difference"]) <= 1e-8
        and float(
            replicate_state_row["relative_state_l2_difference"]
        )
        <= 1e-8
    )
    frozen_exact = bool(
        (
            state_frame.loc[
                state_frame["comparison"].str.contains("_vs_frozen"),
                "max_abs_state_difference",
            ]
            <= 1e-8
        ).all()
        and (
            state_frame.loc[
                state_frame["comparison"].str.contains("_vs_frozen"),
                "relative_state_l2_difference",
            ]
            <= 1e-8
        ).all()
    )

    if frozen_exact and deterministic_current_runtime:
        classification = "EXACT_FROZEN_REPLAY"
    elif deterministic_current_runtime:
        classification = "DETERMINISTIC_CURRENT_RUNTIME_WITH_FROZEN_DRIFT"
    else:
        classification = "CURRENT_RUNTIME_NONDETERMINISM"

    source_frame = pd.DataFrame(source_rows)
    runtime = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "mkl_available": bool(torch.backends.mkl.is_available()),
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
    }

    state_frame.to_csv(
        tables_dir / "task41c1d_state_replay_differences.csv",
        index=False,
    )
    signature_frame.to_csv(
        tables_dir / "task41c1d_signature_replay_differences.csv",
        index=False,
    )
    local_difference_frame.to_csv(
        tables_dir / "task41c1d_local_metric_differences.csv",
        index=False,
    )
    round_difference_frame.to_csv(
        tables_dir / "task41c1d_round_metric_differences.csv",
        index=False,
    )
    source_frame.to_csv(
        tables_dir / "task41c1d_runtime_source_hashes.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "replicate": replay["replicate"],
                "total_seconds": replay["total_seconds"],
            }
            for replay in replays
        ]
    ).to_csv(
        tables_dir / "task41c1d_replay_runtime.csv",
        index=False,
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(
        state_frame["comparison"],
        state_frame["relative_state_l2_difference"],
    )
    axis.set_ylabel("Relative state L2 difference")
    axis.set_title("Task 41C seed-7 clean replay state reproducibility")
    axis.tick_params(axis="x", rotation=25)
    axis.set_yscale("symlog", linthresh=1e-12)
    axis.grid(axis="y", alpha=0.25)
    save_figure(
        figure,
        figures_dir / "task41c1d_state_reproducibility",
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    for comparison, group in signature_frame.groupby("comparison"):
        axis.plot(
            group["warmup_round"],
            group["maximum_absolute_difference"],
            marker="o",
            label=comparison,
        )
    axis.set_xlabel("Warmup round")
    axis.set_ylabel("Maximum signature absolute difference")
    axis.set_title("Task 41C seed-7 transition-signature reproducibility")
    axis.set_yscale("symlog", linthresh=1e-12)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(
        figure,
        figures_dir / "task41c1d_signature_reproducibility",
    )

    decision = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": "seed7_clean_replay_reproducibility_audit",
        "classification": classification,
        "frozen_exact_replay": frozen_exact,
        "deterministic_current_runtime": deterministic_current_runtime,
        "replicate_count": args.replicates,
        "seed": SEED,
        "partition_hash_sha256": partition_hash,
        "probe_hash_sha256": probe_hash,
        "frozen_checkpoint_sha256": checkpoint_sha256(
            frozen_checkpoint_path
        ),
        "maximum_replicate_to_frozen_state_difference": float(
            state_frame.loc[
                state_frame["comparison"].str.contains("_vs_frozen"),
                "max_abs_state_difference",
            ].max()
        ),
        "maximum_between_replicate_state_difference": float(
            replicate_state_row["max_abs_state_difference"]
        ),
        "maximum_signature_difference_to_frozen": float(
            signature_frame.loc[
                signature_frame["comparison"].str.contains("_vs_frozen"),
                "maximum_absolute_difference",
            ].max()
        ),
        "maximum_signature_difference_between_replicates": float(
            signature_frame.loc[
                signature_frame["comparison"]
                == "replicate_1_vs_replicate_2",
                "maximum_absolute_difference",
            ].max()
        ),
        "development_arrays_loaded": list(ALLOWED_ARRAYS),
        "reserved_arrays_loaded": [],
        "attack_execution_performed": False,
        "calibration_result_accepted": False,
        "task41b_reopened": False,
        "reserved_test_accessed": False,
        "runtime": runtime,
        "next_action": (
            "Use this classification to determine whether C1 calibration "
            "must use exact frozen replay, a deterministic current-runtime "
            "clean recalibration branch, or a single-thread deterministic "
            "replay audit. Do not relax equivalence gates without this evidence."
        ),
    }
    (output_dir / "task41c1d_replay_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print()
    print("===== TASK 41C.1D SEED-7 CLEAN REPLAY AUDIT =====")
    print("Classification:", classification)
    print("Frozen exact replay:", frozen_exact)
    print(
        "Deterministic current runtime:",
        deterministic_current_runtime,
    )
    print(
        "Maximum replicate-to-frozen state difference:",
        f"{decision['maximum_replicate_to_frozen_state_difference']:.10g}",
    )
    print(
        "Maximum between-replicate state difference:",
        f"{decision['maximum_between_replicate_state_difference']:.10g}",
    )
    print(
        "Maximum signature difference to frozen:",
        f"{decision['maximum_signature_difference_to_frozen']:.10g}",
    )
    print(
        "Maximum signature difference between replicates:",
        f"{decision['maximum_signature_difference_between_replicates']:.10g}",
    )
    print("CALIBRATION RESULT ACCEPTED: False")
    print("ATTACK EXECUTION PERFORMED: False")
    print("TASK 41B REOPENED: False")
    print("TEST SETS ACCESSED: False")
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Task 41C.1c clean-only calibration for D1, D2, and D3.

D1, trigger-response shift
    Client-specific excess Benign-target probability shifts on two frozen,
    class-conditional triggered server probes, relative to the current global
    reference model and a leave-one-round-out historical client profile.

D2, trigger-gradient alignment
    Client-update cosine alignment with negative target-class gradients on the
    same frozen triggered probes, relative to a leave-one-round-out historical
    client profile.

D3, equal-rank fusion
    Equal-weight empirical-rank fusion of the frozen Task 40 instantaneous
    score, D1, and D2. No component weights are tuned.

Calibration uses four clean warmup rounds only. It loads X_train, y_train,
X_val, and y_val, replays the exact clean warmup chronology, verifies the final
state against the frozen branch checkpoint, and never loads reserved natural or
diagnostic test arrays. No attack is executed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for search_path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

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
    predict_probabilities,
)
from independent_anchor_v310 import (
    CANDIDATE as D0_INSTANT,
    SCORE_COLUMN as D0_EMA,
    file_sha256,
    quantile_higher,
    weighted_average_states,
)
from trusted_update_reconstruction_v312 import (
    checkpoint_sha256,
    state_max_abs_difference,
)

EXPERIMENT_VERSION = "4.12C.1c"
ALLOWED_ARRAYS = ("X_train", "y_train", "X_val", "y_val")
RESERVED_ARRAYS = (
    "X_test_natural",
    "y_test_natural",
    "X_test_diagnostic",
    "y_test_diagnostic",
)
FROZEN_SEEDS = (7, 99, 123, 2026)
FROZEN_TRIGGER_SLOTS = {
    "flow_iat_exact": "multi_flow_iat_bundle__exact_template__donor_00",
    "active_idle_exact": "multi_active_idle_bundle__exact_template__donor_18",
}
FROZEN_MALICIOUS_CLIENTS = (1, 7, 8, 10, 14, 15, 17, 18)
EMA_DECAY = 0.65
INSTANT_QUANTILE = 0.95
EMA_QUANTILE = 0.99
MAX_Z = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate Task 41C D1, D2, and D3 from clean warmup only."
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--partition-file", type=Path, required=True)
    parser.add_argument("--clean-seed-root", type=Path, required=True)
    parser.add_argument("--warmup-root", type=Path, required=True)
    parser.add_argument("--trigger-spec-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", default="7,99,123,2026")
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_seeds(text: str) -> List[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if values != list(FROZEN_SEEDS):
        raise ValueError(
            "Task 41C.1c seeds are frozen to exactly 7,99,123,2026."
        )
    return values


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
        raise ValueError("State keys differ.")
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


def robust_scale(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    center = float(np.median(array))
    scale = 1.4826 * float(np.median(np.abs(array - center)))
    if scale < 1e-9:
        scale = float(np.std(array))
    return float(scale if scale >= 1e-9 else 1.0)


def positive_z(
    value: np.ndarray,
    profile: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return np.clip(
        np.maximum(
            (np.asarray(value, dtype=np.float64)
             - np.asarray(profile, dtype=np.float64))
            / np.maximum(np.asarray(scale, dtype=np.float64), 1e-12),
            0.0,
        ),
        0.0,
        MAX_Z,
    )


def add_ema(
    table: pd.DataFrame,
    instant_column: str,
    ema_column: str,
    decay: float,
) -> pd.DataFrame:
    result = table.sort_values(["client_id", "monitoring_round"]).copy()
    output = pd.Series(index=result.index, dtype=float)
    for client_id, group in result.groupby("client_id", sort=False):
        previous = 0.0
        for row_index in group.index:
            previous = (
                float(decay) * previous
                + (1.0 - float(decay))
                * float(result.at[row_index, instant_column])
            )
            output.at[row_index] = previous
    result[ema_column] = output
    return result.sort_values(
        ["monitoring_round", "client_id"]
    ).reset_index(drop=True)


def empirical_percentiles(values: Sequence[float]) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )


def parameter_names(model: torch.nn.Module) -> List[str]:
    return [name for name, _ in model.named_parameters()]


def update_vector(
    local_state: Mapping[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor],
    names: Sequence[str],
) -> torch.Tensor:
    parts = []
    for name in names:
        local = local_state[name].detach().cpu().to(torch.float64)
        reference = reference_state[name].detach().cpu().to(torch.float64)
        parts.append((local - reference).reshape(-1))
    if not parts:
        raise RuntimeError("No model parameters were available.")
    return torch.cat(parts)


def gradient_direction(
    reference_state: Mapping[str, torch.Tensor],
    input_dim: int,
    X: np.ndarray,
    target_id: int,
) -> torch.Tensor:
    model = build_model("resmlp", input_dim, NUM_CLASSES)
    model.load_state_dict(reference_state)
    model.train()
    inputs = torch.from_numpy(np.asarray(X, dtype=np.float32))
    targets = torch.full(
        (len(inputs),),
        int(target_id),
        dtype=torch.long,
    )
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    parameters = [parameter for _, parameter in model.named_parameters()]
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    direction = torch.cat(
        [(-gradient.detach().cpu().to(torch.float64)).reshape(-1)
         for gradient in gradients]
    )
    norm = float(torch.linalg.vector_norm(direction))
    if norm <= 1e-12:
        raise RuntimeError("Trigger-gradient direction has zero norm.")
    return direction / norm


def cosine(vector: torch.Tensor, direction: torch.Tensor) -> float:
    vector = vector.detach().cpu().to(torch.float64)
    direction = direction.detach().cpu().to(torch.float64)
    denominator = (
        float(torch.linalg.vector_norm(vector))
        * float(torch.linalg.vector_norm(direction))
    )
    if denominator <= 1e-12:
        return 0.0
    return float(torch.dot(vector, direction) / denominator)


def load_trigger_panel(path: Path) -> tuple[List[Dict[str, Any]], str]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if bool(payload.get("test_arrays_loaded", False)):
        raise RuntimeError("Frozen trigger specification reports test access.")
    selected = payload.get("selected_specs", {})
    panel: List[Dict[str, Any]] = []
    for slot, candidate_id in FROZEN_TRIGGER_SLOTS.items():
        if candidate_id not in selected:
            raise KeyError(f"Frozen trigger candidate is missing: {candidate_id}")
        spec = dict(selected[candidate_id])
        indices = np.asarray(spec["feature_indices"], dtype=np.int64)
        template = np.asarray(
            spec["template_standardized"], dtype=np.float32
        )
        names = list(spec["feature_names"])
        if len(indices) == 0 or len(indices) != len(template):
            raise RuntimeError(f"Invalid trigger specification: {candidate_id}")
        if len(names) != len(indices):
            raise RuntimeError(f"Trigger feature names mismatch: {candidate_id}")
        if int(spec["target_class_id"]) != CLASS_NAMES.index("Benign"):
            raise RuntimeError("Task 41C C1 trigger target must be Benign.")
        panel.append(
            {
                "slot": slot,
                "candidate_id": candidate_id,
                "feature_indices": indices,
                "template": template,
                "feature_names": names,
            }
        )
    return panel, file_sha256(resolved)


def triggered_probe(
    X_probe: np.ndarray,
    feature_indices: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    output = np.asarray(X_probe, dtype=np.float32).copy()
    output[:, feature_indices] = template.reshape(1, -1)
    return output


def response_vector(
    model: torch.nn.Module,
    X_probe: np.ndarray,
    y_probe: np.ndarray,
    trigger_panel: Sequence[Dict[str, Any]],
    target_id: int,
    batch_size: int,
) -> np.ndarray:
    clean_probabilities = predict_probabilities(
        model, X_probe, batch_size
    )
    values: List[float] = []
    source_ids = [
        class_id
        for class_id in range(NUM_CLASSES)
        if class_id != target_id
    ]
    for trigger in trigger_panel:
        X_triggered = triggered_probe(
            X_probe,
            trigger["feature_indices"],
            trigger["template"],
        )
        trigger_probabilities = predict_probabilities(
            model, X_triggered, batch_size
        )
        for source_id in source_ids:
            mask = y_probe == source_id
            if not np.any(mask):
                raise RuntimeError(
                    f"Probe lacks source class {source_id}."
                )
            shift = float(
                trigger_probabilities[mask, target_id].mean()
                - clean_probabilities[mask, target_id].mean()
            )
            values.append(shift)
    return np.asarray(values, dtype=np.float64)


def trigger_gradient_panel(
    reference_state: Mapping[str, torch.Tensor],
    input_dim: int,
    X_probe: np.ndarray,
    y_probe: np.ndarray,
    trigger_panel: Sequence[Dict[str, Any]],
    target_id: int,
) -> List[torch.Tensor]:
    directions: List[torch.Tensor] = []
    source_ids = [
        class_id
        for class_id in range(NUM_CLASSES)
        if class_id != target_id
    ]
    for trigger in trigger_panel:
        X_triggered = triggered_probe(
            X_probe,
            trigger["feature_indices"],
            trigger["template"],
        )
        for source_id in source_ids:
            mask = y_probe == source_id
            directions.append(
                gradient_direction(
                    reference_state,
                    input_dim,
                    X_triggered[mask],
                    target_id,
                )
            )
    return directions


def feature_labels(
    trigger_panel: Sequence[Dict[str, Any]],
    target_id: int,
) -> List[Dict[str, Any]]:
    rows = []
    feature_id = 0
    for trigger in trigger_panel:
        for source_id, source_name in enumerate(CLASS_NAMES):
            if source_id == target_id:
                continue
            rows.append(
                {
                    "feature_id": feature_id,
                    "trigger_slot": trigger["slot"],
                    "trigger_candidate_id": trigger["candidate_id"],
                    "source_class_id": source_id,
                    "source_class_name": source_name,
                    "target_class_id": target_id,
                    "target_class_name": CLASS_NAMES[target_id],
                }
            )
            feature_id += 1
    return rows


def calibrate_seed(
    *,
    seed: int,
    arrays: Dict[str, np.ndarray],
    partition_file: Path,
    clean_seed_dir: Path,
    warmup_dir: Path,
    trigger_panel: Sequence[Dict[str, Any]],
    trigger_spec_sha256: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    started = time.time()
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    calibration = output_dir / "calibration"
    for path in (tables, figures, calibration):
        path.mkdir(parents=True, exist_ok=True)

    X_train = arrays["X_train"].astype(np.float32, copy=False)
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    X_val = arrays["X_val"].astype(np.float32, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)

    set_seed(seed)
    client_indices, partition_hash = load_fixed_partitions(
        partition_file,
        expected_clients=args.num_clients,
        train_rows=len(y_train),
    )
    clean_hash = read_clean_partition_hash(clean_seed_dir)
    if clean_hash != partition_hash:
        raise RuntimeError(
            f"Seed {seed}: clean partition hash mismatch."
        )

    metadata_path = warmup_dir / "true_warmup_v310_metadata.json"
    checkpoint_path = (
        warmup_dir / "checkpoints" / "common_round4_warmup_model.pt"
    )
    d0_scores_path = (
        warmup_dir
        / "calibration"
        / "clean_leave_one_round_out_scores.csv"
    )
    for path in (metadata_path, checkpoint_path, d0_scores_path):
        if not path.exists():
            raise FileNotFoundError(path)

    warmup_metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    if int(warmup_metadata["model_seed"]) != seed:
        raise RuntimeError(f"Seed {seed}: warmup seed mismatch.")
    if warmup_metadata["partition_hash_sha256"] != partition_hash:
        raise RuntimeError(f"Seed {seed}: warmup partition mismatch.")
    if int(warmup_metadata["warmup_rounds"]) != args.warmup_rounds:
        raise RuntimeError(f"Seed {seed}: warmup round mismatch.")
    if not bool(
        warmup_metadata.get("monitoring_ema_reset_after_warmup", False)
    ):
        raise RuntimeError(
            f"Seed {seed}: warmup EMA-reset contract is missing."
        )

    probe_indices = balanced_probe_indices(
        y_val, args.probe_per_class, args.probe_seed
    )
    probe_hash = sha256_array(probe_indices)
    if probe_hash != warmup_metadata["probe_hash_sha256"]:
        raise RuntimeError(f"Seed {seed}: probe hash mismatch.")
    X_probe = X_val[probe_indices]
    y_probe = y_val[probe_indices]

    d0_scores = pd.read_csv(d0_scores_path)
    required_d0 = {
        "monitoring_round",
        "client_id",
        D0_INSTANT,
        D0_EMA,
    }
    missing_d0 = required_d0.difference(d0_scores.columns)
    if missing_d0:
        raise RuntimeError(
            f"Seed {seed}: D0 score table lacks {sorted(missing_d0)}."
        )
    d0_scores = d0_scores.sort_values(
        ["monitoring_round", "client_id"]
    ).reset_index(drop=True)
    if len(d0_scores) != args.warmup_rounds * args.num_clients:
        raise RuntimeError(f"Seed {seed}: D0 score row count mismatch.")

    class_weights = sqrt_class_weights(
        y_train, args.max_class_weight
    )
    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    names = parameter_names(model)
    target_id = CLASS_NAMES.index("Benign")
    vector_count = len(trigger_panel) * (NUM_CLASSES - 1)

    d1_vectors = np.zeros(
        (args.warmup_rounds, args.num_clients, vector_count),
        dtype=np.float64,
    )
    d2_vectors = np.zeros_like(d1_vectors)
    training_rows: List[Dict[str, Any]] = []
    round_rows: List[Dict[str, Any]] = []

    print()
    print(f"Task 41C.1c seed {seed}")
    print("Replaying four clean warmup rounds")

    for round_id in range(1, args.warmup_rounds + 1):
        round_started = time.time()
        reference_state = copy.deepcopy(model.state_dict())
        reference_model = build_model(
            "resmlp", X_train.shape[1], NUM_CLASSES
        )
        reference_model.load_state_dict(reference_state)
        reference_response = response_vector(
            reference_model,
            X_probe,
            y_probe,
            trigger_panel,
            target_id,
            args.evaluation_batch_size,
        )
        directions = trigger_gradient_panel(
            reference_state,
            X_train.shape[1],
            X_probe,
            y_probe,
            trigger_panel,
            target_id,
        )
        if len(directions) != vector_count:
            raise RuntimeError("Gradient direction count mismatch.")

        local_states: List[Mapping[str, torch.Tensor]] = []
        sample_counts: List[int] = []

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
                seed=seed + round_id * 1000 + client_id,
            )
            local_response = response_vector(
                local_model,
                X_probe,
                y_probe,
                trigger_panel,
                target_id,
                args.evaluation_batch_size,
            )
            d1_vectors[round_id - 1, client_id, :] = (
                local_response - reference_response
            )
            update = update_vector(
                state, reference_state, names
            )
            d2_vectors[round_id - 1, client_id, :] = np.asarray(
                [cosine(update, direction) for direction in directions],
                dtype=np.float64,
            )
            local_states.append(state)
            sample_counts.append(int(len(indices)))
            training_rows.append(
                {
                    "monitoring_round": round_id,
                    "client_id": client_id,
                    "client_samples": int(len(indices)),
                    **metrics,
                }
            )
            del local_model

        model.load_state_dict(
            weighted_average_states(
                local_states,
                sample_counts,
                reference_state,
            )
        )
        round_rows.append(
            {
                "monitoring_round": round_id,
                "participating_samples": int(sum(sample_counts)),
                "round_seconds": float(time.time() - round_started),
                "mean_d1_excess_response": float(
                    d1_vectors[round_id - 1].mean()
                ),
                "maximum_d1_excess_response": float(
                    d1_vectors[round_id - 1].max()
                ),
                "mean_d2_alignment": float(
                    d2_vectors[round_id - 1].mean()
                ),
                "maximum_d2_alignment": float(
                    d2_vectors[round_id - 1].max()
                ),
            }
        )
        print(
            f"clean replay round {round_id:02d}, "
            f"seconds={round_rows[-1]['round_seconds']:.1f}"
        )

    frozen_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if frozen_checkpoint.get("partition_hash") != partition_hash:
        raise RuntimeError(f"Seed {seed}: checkpoint partition mismatch.")
    replay_max_abs = state_max_abs_difference(
        model.state_dict(),
        frozen_checkpoint["model_state_dict"],
    )
    replay_relative_l2 = state_relative_l2_difference(
        model.state_dict(),
        frozen_checkpoint["model_state_dict"],
    )
    if replay_max_abs > 0.005 or replay_relative_l2 > 0.005:
        raise RuntimeError(
            f"Seed {seed}: clean replay equivalence failed, "
            f"max_abs={replay_max_abs}, relative_l2={replay_relative_l2}."
        )

    d1_final_profiles = np.median(d1_vectors, axis=0)
    d2_final_profiles = np.median(d2_vectors, axis=0)
    d1_loo_residuals = np.zeros_like(d1_vectors)
    d2_loo_residuals = np.zeros_like(d2_vectors)

    for heldout in range(args.warmup_rounds):
        others = [
            round_index
            for round_index in range(args.warmup_rounds)
            if round_index != heldout
        ]
        d1_profile = np.median(d1_vectors[others], axis=0)
        d2_profile = np.median(d2_vectors[others], axis=0)
        d1_loo_residuals[heldout] = (
            d1_vectors[heldout] - d1_profile
        )
        d2_loo_residuals[heldout] = (
            d2_vectors[heldout] - d2_profile
        )

    d1_scales = np.asarray(
        [
            robust_scale(d1_loo_residuals[:, :, feature].reshape(-1))
            for feature in range(vector_count)
        ],
        dtype=np.float64,
    )
    d2_scales = np.asarray(
        [
            robust_scale(d2_loo_residuals[:, :, feature].reshape(-1))
            for feature in range(vector_count)
        ],
        dtype=np.float64,
    )

    score_rows: List[Dict[str, Any]] = []
    feature_long_rows: List[Dict[str, Any]] = []
    labels = feature_labels(trigger_panel, target_id)
    d0_lookup = d0_scores.set_index(
        ["monitoring_round", "client_id"]
    )

    for heldout in range(args.warmup_rounds):
        others = [
            round_index
            for round_index in range(args.warmup_rounds)
            if round_index != heldout
        ]
        d1_profile = np.median(d1_vectors[others], axis=0)
        d2_profile = np.median(d2_vectors[others], axis=0)
        monitoring_round = heldout + 1
        for client_id in range(args.num_clients):
            d1_z = positive_z(
                d1_vectors[heldout, client_id],
                d1_profile[client_id],
                d1_scales,
            )
            d2_z = positive_z(
                d2_vectors[heldout, client_id],
                d2_profile[client_id],
                d2_scales,
            )
            d1_instant = float(np.max(d1_z))
            d2_instant = float(np.max(d2_z))
            d0_row = d0_lookup.loc[(monitoring_round, client_id)]
            score_rows.append(
                {
                    "monitoring_round": monitoring_round,
                    "client_id": client_id,
                    "actual_malicious": False,
                    "d0_instant": float(d0_row[D0_INSTANT]),
                    "d0_ema": float(d0_row[D0_EMA]),
                    "d1_trigger_response_shift": d1_instant,
                    "d2_trigger_gradient_alignment": d2_instant,
                }
            )
            for label in labels:
                feature_id = int(label["feature_id"])
                feature_long_rows.append(
                    {
                        "monitoring_round": monitoring_round,
                        "client_id": client_id,
                        **label,
                        "d1_observed_excess_response": float(
                            d1_vectors[heldout, client_id, feature_id]
                        ),
                        "d1_loo_profile": float(
                            d1_profile[client_id, feature_id]
                        ),
                        "d1_clean_scale": float(d1_scales[feature_id]),
                        "d1_positive_z": float(d1_z[feature_id]),
                        "d2_observed_alignment": float(
                            d2_vectors[heldout, client_id, feature_id]
                        ),
                        "d2_loo_profile": float(
                            d2_profile[client_id, feature_id]
                        ),
                        "d2_clean_scale": float(d2_scales[feature_id]),
                        "d2_positive_z": float(d2_z[feature_id]),
                    }
                )

    scores = pd.DataFrame(score_rows)
    scores = add_ema(
        scores,
        "d1_trigger_response_shift",
        "d1_trigger_response_shift_ema",
        EMA_DECAY,
    )
    scores = add_ema(
        scores,
        "d2_trigger_gradient_alignment",
        "d2_trigger_gradient_alignment_ema",
        EMA_DECAY,
    )

    scores["d0_empirical_rank"] = empirical_percentiles(
        scores["d0_instant"]
    )
    scores["d1_empirical_rank"] = empirical_percentiles(
        scores["d1_trigger_response_shift"]
    )
    scores["d2_empirical_rank"] = empirical_percentiles(
        scores["d2_trigger_gradient_alignment"]
    )
    scores["d3_equal_rank_fusion"] = scores[
        [
            "d0_empirical_rank",
            "d1_empirical_rank",
            "d2_empirical_rank",
        ]
    ].mean(axis=1)
    scores = add_ema(
        scores,
        "d3_equal_rank_fusion",
        "d3_equal_rank_fusion_ema",
        EMA_DECAY,
    )

    candidate_columns = {
        "D0_frozen_task40_lfighter": (
            "d0_instant",
            "d0_ema",
        ),
        "D1_trigger_response_shift": (
            "d1_trigger_response_shift",
            "d1_trigger_response_shift_ema",
        ),
        "D2_trigger_gradient_alignment": (
            "d2_trigger_gradient_alignment",
            "d2_trigger_gradient_alignment_ema",
        ),
        "D3_equal_rank_fusion": (
            "d3_equal_rank_fusion",
            "d3_equal_rank_fusion_ema",
        ),
    }
    threshold_rows = []
    for candidate, (instant_column, ema_column) in candidate_columns.items():
        instant_threshold = quantile_higher(
            scores[instant_column].to_numpy(dtype=np.float64),
            INSTANT_QUANTILE,
        )
        ema_threshold = quantile_higher(
            scores[ema_column].to_numpy(dtype=np.float64),
            EMA_QUANTILE,
        )
        instant_flags = (
            scores[instant_column].to_numpy(dtype=np.float64)
            > instant_threshold
        )
        ema_flags = (
            scores[ema_column].to_numpy(dtype=np.float64)
            > ema_threshold
        )
        policy_flags = instant_flags | ema_flags
        scores[f"{candidate}_instant_threshold"] = instant_threshold
        scores[f"{candidate}_ema_threshold"] = ema_threshold
        scores[f"{candidate}_clean_flag"] = policy_flags
        threshold_rows.append(
            {
                "candidate": candidate,
                "instant_column": instant_column,
                "ema_column": ema_column,
                "instant_quantile": INSTANT_QUANTILE,
                "ema_quantile": EMA_QUANTILE,
                "instant_threshold": float(instant_threshold),
                "ema_threshold": float(ema_threshold),
                "clean_instant_fpr": float(instant_flags.mean()),
                "clean_ema_fpr": float(ema_flags.mean()),
                "clean_policy_fpr": float(policy_flags.mean()),
                "attack_labels_used": False,
                "reserved_test_accessed": False,
            }
        )

    thresholds = pd.DataFrame(threshold_rows)
    feature_table = pd.DataFrame(feature_long_rows)
    replay_table = pd.DataFrame(
        [
            {
                "seed": seed,
                "partition_hash_sha256": partition_hash,
                "probe_hash_sha256": probe_hash,
                "common_warmup_checkpoint_sha256": checkpoint_sha256(
                    checkpoint_path
                ),
                "replay_max_abs_state_difference": replay_max_abs,
                "replay_relative_state_l2_difference": replay_relative_l2,
                "replay_equivalence_passed": True,
                "development_arrays_loaded": "|".join(ALLOWED_ARRAYS),
                "reserved_arrays_loaded": "",
                "attack_execution_performed": False,
                "reserved_test_accessed": False,
            }
        ]
    )

    scores.to_csv(
        tables / "task41c1c_clean_candidate_scores.csv",
        index=False,
    )
    thresholds.to_csv(
        calibration / "task41c1c_candidate_thresholds.csv",
        index=False,
    )
    feature_table.to_csv(
        tables / "task41c1c_clean_feature_vectors_long.csv",
        index=False,
    )
    pd.DataFrame(training_rows).to_csv(
        tables / "task41c1c_clean_replay_local_training.csv",
        index=False,
    )
    pd.DataFrame(round_rows).to_csv(
        tables / "task41c1c_clean_replay_rounds.csv",
        index=False,
    )
    replay_table.to_csv(
        tables / "task41c1c_replay_equivalence.csv",
        index=False,
    )
    pd.DataFrame(labels).to_csv(
        calibration / "task41c1c_feature_panel.csv",
        index=False,
    )

    np.savez_compressed(
        calibration / "task41c1c_backdoor_detector_profiles.npz",
        d1_client_profiles=d1_final_profiles.astype(np.float64),
        d1_feature_scales=d1_scales.astype(np.float64),
        d2_client_profiles=d2_final_profiles.astype(np.float64),
        d2_feature_scales=d2_scales.astype(np.float64),
        d0_empirical_reference=np.sort(
            scores["d0_instant"].to_numpy(dtype=np.float64)
        ),
        d1_empirical_reference=np.sort(
            scores["d1_trigger_response_shift"].to_numpy(
                dtype=np.float64
            )
        ),
        d2_empirical_reference=np.sort(
            scores["d2_trigger_gradient_alignment"].to_numpy(
                dtype=np.float64
            )
        ),
    )

    profile_path = (
        calibration / "task41c1c_backdoor_detector_profiles.npz"
    )
    threshold_path = (
        calibration / "task41c1c_candidate_thresholds.csv"
    )
    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": "task41c_clean_only_detector_calibration",
        "seed": seed,
        "trigger_slots": list(FROZEN_TRIGGER_SLOTS),
        "trigger_candidates": list(FROZEN_TRIGGER_SLOTS.values()),
        "trigger_spec_sha256": trigger_spec_sha256,
        "feature_count": vector_count,
        "num_clients": args.num_clients,
        "warmup_rounds": args.warmup_rounds,
        "ema_decay": EMA_DECAY,
        "instant_quantile": INSTANT_QUANTILE,
        "ema_quantile": EMA_QUANTILE,
        "d1_definition": (
            "maximum positive robust-z deviation of client excess "
            "Benign-target response shift from its historical profile"
        ),
        "d2_definition": (
            "maximum positive robust-z deviation of client-update cosine "
            "alignment with negative triggered target gradients"
        ),
        "d3_definition": (
            "equal mean of empirical clean ranks of D0, D1, and D2"
        ),
        "d3_weights": [1.0, 1.0, 1.0],
        "weight_tuning_used": False,
        "attack_specific_thresholds": False,
        "trigger_specific_thresholds": False,
        "profile_sha256": file_sha256(profile_path),
        "threshold_sha256": file_sha256(threshold_path),
        "partition_hash_sha256": partition_hash,
        "probe_hash_sha256": probe_hash,
        "common_warmup_checkpoint_sha256": checkpoint_sha256(
            checkpoint_path
        ),
        "replay_max_abs_state_difference": replay_max_abs,
        "replay_relative_state_l2_difference": replay_relative_l2,
        "attack_labels_used": False,
        "attack_execution_performed": False,
        "reserved_test_accessed": False,
        "total_seconds": float(time.time() - started),
    }
    metadata_path_out = (
        output_dir / "task41c1c_clean_calibration_metadata.json"
    )
    metadata_path_out.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(10.5, 6.0))
    for candidate, (instant_column, _) in candidate_columns.items():
        axis.hist(
            scores[instant_column],
            bins=20,
            alpha=0.45,
            label=candidate,
        )
    axis.set_xlabel("Clean instantaneous score")
    axis.set_ylabel("Client-round count")
    axis.set_title(f"Task 41C clean score distributions, seed {seed}")
    axis.legend(fontsize=8)
    save_figure(
        figure,
        figures / "task41c1c_clean_instant_score_distributions",
    )

    figure, axis = plt.subplots(figsize=(10.5, 6.0))
    plot_thresholds = thresholds.set_index("candidate")
    axis.bar(
        plot_thresholds.index,
        plot_thresholds["clean_policy_fpr"],
    )
    axis.axhline(0.05, linestyle="--", label="Preregistered FPR ceiling")
    axis.set_ylabel("Clean client-round FPR")
    axis.set_title(f"Task 41C clean detector FPR, seed {seed}")
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    save_figure(
        figure,
        figures / "task41c1c_clean_candidate_fpr",
    )

    figure, axis = plt.subplots(figsize=(10.5, 6.0))
    axis.plot(
        pd.DataFrame(round_rows)["monitoring_round"],
        pd.DataFrame(round_rows)["maximum_d1_excess_response"],
        marker="o",
        label="Maximum D1 excess response",
    )
    axis.plot(
        pd.DataFrame(round_rows)["monitoring_round"],
        pd.DataFrame(round_rows)["maximum_d2_alignment"],
        marker="o",
        label="Maximum D2 alignment",
    )
    axis.set_xlabel("Clean warmup round")
    axis.set_ylabel("Raw clean maximum")
    axis.set_title(f"Task 41C clean feature chronology, seed {seed}")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(
        figure,
        figures / "task41c1c_clean_feature_chronology",
    )

    print(
        f"Seed {seed} calibration complete, "
        f"replay max abs={replay_max_abs:.8g}, "
        f"relative L2={replay_relative_l2:.8g}"
    )
    return {
        "seed": seed,
        "output_dir": str(output_dir),
        "partition_hash_sha256": partition_hash,
        "probe_hash_sha256": probe_hash,
        "trigger_spec_sha256": trigger_spec_sha256,
        "profile_sha256": metadata["profile_sha256"],
        "threshold_sha256": metadata["threshold_sha256"],
        "replay_max_abs_state_difference": replay_max_abs,
        "replay_relative_state_l2_difference": replay_relative_l2,
        "replay_equivalence_passed": True,
        "maximum_clean_policy_fpr": float(
            thresholds["clean_policy_fpr"].max()
        ),
        "attack_labels_used": False,
        "attack_execution_performed": False,
        "reserved_test_accessed": False,
        "total_seconds": metadata["total_seconds"],
    }


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if args.num_clients != 20:
        raise ValueError("Task 41C is frozen to 20 clients.")
    if args.warmup_rounds != 4:
        raise ValueError("Task 41C C1 is frozen to four warmup rounds.")
    if args.probe_per_class != 48 or args.probe_seed != 3701:
        raise ValueError("Task 41C reuses the frozen balanced probe.")
    if args.threads < 1:
        raise ValueError("threads must be positive.")

    torch.set_num_threads(args.threads)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"Output root is not empty: {output_root}"
            )
    output_root.mkdir(parents=True, exist_ok=True)

    protocol_path = (
        PROJECT_ROOT / "configs" / "task41c_preregistration_v412c0.json"
    )
    c1a_decision_path = (
        PROJECT_ROOT
        / "results"
        / "cic_iot_diad_task41c_clean_calibration_v412c1a"
        / "preflight"
        / "task41c1_clean_preflight_decision.json"
    )
    c1b_contract_path = (
        PROJECT_ROOT
        / "results"
        / "cic_iot_diad_task41c_clean_calibration_v412c1b"
        / "interface_audit"
        / "task41c1b_interface_contract.json"
    )
    for path in (
        protocol_path,
        c1a_decision_path,
        c1b_contract_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    c1a_decision = json.loads(
        c1a_decision_path.read_text(encoding="utf-8")
    )
    c1b_contract = json.loads(
        c1b_contract_path.read_text(encoding="utf-8")
    )
    if protocol.get("frozen_seeds") != list(FROZEN_SEEDS):
        raise RuntimeError("Frozen seed panel mismatch.")
    if not bool(c1a_decision.get("c1_clean_calibration_allowed", False)):
        raise RuntimeError("C1a did not authorize clean calibration.")
    if bool(c1a_decision.get("attack_execution_allowed", True)):
        raise RuntimeError("C1a attack-execution gate is invalid.")
    if not bool(c1b_contract.get("c1_implementation_allowed", False)):
        raise RuntimeError("C1b did not authorize implementation.")
    if bool(c1b_contract.get("reserved_test_accessed", True)):
        raise RuntimeError("C1b reports reserved test access.")

    arrays = load_development_arrays(args.data_file)
    trigger_panel, trigger_spec_sha256 = load_trigger_panel(
        args.trigger_spec_file
    )

    summary_rows = []
    for seed in seeds:
        seed_output = output_root / f"seed_{seed}"
        summary_rows.append(
            calibrate_seed(
                seed=seed,
                arrays=arrays,
                partition_file=args.partition_file.expanduser().resolve(),
                clean_seed_dir=(
                    args.clean_seed_root.expanduser().resolve()
                    / f"seed_{seed}"
                ),
                warmup_dir=(
                    args.warmup_root.expanduser().resolve()
                    / f"seed_{seed}"
                    / "warmup"
                ),
                trigger_panel=trigger_panel,
                trigger_spec_sha256=trigger_spec_sha256,
                output_dir=seed_output,
                args=args,
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary_tables = output_root / "summary" / "tables"
    summary_figures = output_root / "summary" / "figures"
    summary_tables.mkdir(parents=True, exist_ok=True)
    summary_figures.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        summary_tables / "task41c1c_multiseed_calibration_summary.csv",
        index=False,
    )

    threshold_frames = []
    for seed in seeds:
        path = (
            output_root
            / f"seed_{seed}"
            / "calibration"
            / "task41c1c_candidate_thresholds.csv"
        )
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        threshold_frames.append(frame)
    all_thresholds = pd.concat(threshold_frames, ignore_index=True)
    all_thresholds.to_csv(
        summary_tables / "task41c1c_multiseed_thresholds.csv",
        index=False,
    )

    decision = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": "task41c_clean_only_detector_calibration",
        "seed_count": len(seeds),
        "ready_seed_count": int(
            summary["replay_equivalence_passed"].sum()
        ),
        "candidate_count": 4,
        "calibrated_candidates": [
            "D0_frozen_task40_lfighter",
            "D1_trigger_response_shift",
            "D2_trigger_gradient_alignment",
            "D3_equal_rank_fusion",
        ],
        "d3_weights": [1.0, 1.0, 1.0],
        "weight_tuning_used": False,
        "attack_specific_thresholds": False,
        "trigger_specific_thresholds": False,
        "maximum_clean_policy_fpr": float(
            all_thresholds["clean_policy_fpr"].max()
        ),
        "clean_fpr_ceiling": 0.05,
        "clean_fpr_gate_passed": bool(
            all_thresholds["clean_policy_fpr"].max() <= 0.05
        ),
        "replay_equivalence_passed_all_seeds": bool(
            summary["replay_equivalence_passed"].all()
        ),
        "attack_labels_used": False,
        "attack_execution_performed": False,
        "task41b_reopened": False,
        "reserved_test_accessed": False,
        "c2_seed7_screen_allowed": bool(
            all_thresholds["clean_policy_fpr"].max() <= 0.05
            and summary["replay_equivalence_passed"].all()
        ),
        "next_stage": (
            "Freeze C1c calibration evidence, then implement the exact "
            "seed-7 C2 candidate screen. Do not run confirmatory multiseed "
            "attacks before C2 selects one candidate."
        ),
    }
    (
        output_root
        / "summary"
        / "task41c1c_clean_calibration_decision.json"
    ).write_text(json.dumps(decision, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(11, 6))
    for candidate, group in all_thresholds.groupby("candidate"):
        axis.plot(
            group["seed"].astype(str),
            group["clean_policy_fpr"],
            marker="o",
            label=candidate,
        )
    axis.axhline(0.05, linestyle="--", label="FPR ceiling")
    axis.set_xlabel("Seed")
    axis.set_ylabel("Clean client-round FPR")
    axis.set_ylim(0, max(0.055, float(all_thresholds["clean_policy_fpr"].max()) + 0.01))
    axis.set_title("Task 41C clean calibration FPR across seeds")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(
        figure,
        summary_figures / "task41c1c_multiseed_clean_fpr",
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(
        summary["seed"].astype(str),
        summary["total_seconds"],
    )
    axis.set_xlabel("Seed")
    axis.set_ylabel("Calibration seconds")
    axis.set_title("Task 41C clean calibration runtime")
    axis.grid(axis="y", alpha=0.25)
    save_figure(
        figure,
        summary_figures / "task41c1c_multiseed_runtime",
    )

    print()
    print("===== TASK 41C.1C CLEAN-ONLY DETECTOR CALIBRATION =====")
    print("Seeds calibrated:", f"{len(seeds)}/{len(FROZEN_SEEDS)}")
    print("Candidates calibrated: 4")
    print(
        "Maximum clean policy FPR:",
        f"{decision['maximum_clean_policy_fpr']:.6f}",
    )
    print(
        "Replay equivalence all seeds:",
        decision["replay_equivalence_passed_all_seeds"],
    )
    print("C2 SEED-7 SCREEN ALLOWED:", decision["c2_seed7_screen_allowed"])
    print("ATTACK LABELS USED: False")
    print("ATTACK EXECUTION PERFORMED: False")
    print("TASK 41B REOPENED: False")
    print("TEST SETS ACCESSED: False")
    print("Tables:", summary_tables)
    print("PNG and PDF figures:", summary_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Task 43 C2: seed-7 development screen, Byzantine attacks + baselines.

Scope
-----
- Seed 7 only. Clean reference plus 7 attack conditions: constant (1),
  extreme-value (k in {2,5,10}), coordinated/omniscient split-replacement
  (boost in {3,5,10}).
- Five arms: plain_fedavg, D0 (frozen Task 40/41 detector, unmodified,
  detect+reconstruct), coordinate_median, trimmed_mean, multi_krum (the
  last three are pure robust-aggregation rules -- no detection step, no
  per-client flagging, no reconstruction; they aggregate all 20 submitted
  updates directly using the verified functions from
  verify_task43_c1_baselines_v414.py).
- Four post-warmup monitoring rounds, identical warmup/partition/coalition
  to every prior stage.

Attack mechanics
-----------------
- constant: each malicious client submits a FIXED vector, unrelated to
  local training entirely, reused identically every round. Magnitude
  fixed at the 95th-percentile clean update norm (2.37, from Task 42's
  C1b pooled distribution). Direction is drawn once per malicious client
  from a fixed seed (deterministic, differs per client, frozen across
  rounds).
- extreme_value: each malicious client submits a vector with every
  coordinate at +/- k*sigma_coordinate, where sigma_coordinate is that
  parameter position's own standard deviation across the captured clean
  seed-7 update matrices (reusing Task 42's C1 capture infrastructure).
  Sign is chosen per-coordinate to oppose that round's honest-client
  coordinate-median (an omniscient-attacker assumption: Byzantine threat
  models classically assume strong attacker knowledge, unlike the
  defense side which must remain causal).
- coordinated_split: the malicious coalition shares ONE target
  perturbation (Task 42's collapse-to-Benign direction, boosted) and
  SPLITS it evenly across all 8 malicious clients (each submits
  target/8), so no single client's update looks unusually large --
  this is the qualitative difference from Task 42's model_replacement,
  which had each malicious client submit the full-strength attack
  independently.

D0 uses the same frozen thresholds as Task 42 (reused from Task 41C's
calibration). coordinate_median/trimmed_mean/multi_krum never classify
individual clients, so malicious_recall/benign_fpr are not applicable
for those arms (recorded as NaN, matching how plain_fedavg is already
handled).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

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

from federated_iot_v26 import (
    CLASS_NAMES,
    NUM_CLASSES,
    set_seed,
    sqrt_class_weights,
    train_local_model,
)
from neural_models_v24 import build_model
from run_targeted_label_flip_v292 import load_fixed_partitions, read_clean_partition_hash
from transition_signature_features_v38 import (
    balanced_probe_indices,
    class_conditional_probability_means,
    evaluate_validation,
    predict_probabilities,
)
from independent_anchor_v310 import (
    CANDIDATE as D0_INSTANT,
    SCORE_COLUMN as D0_EMA,
    add_candidate_and_ema,
    apply_feature_calibration,
    file_sha256,
    load_profiles_npz,
    normalize_rows,
    raw_features,
    weighted_average_states,
)
from trusted_update_reconstruction_v312 import (
    checkpoint_sha256,
    coordinate_median,
    floating_update,
    reconstruct_update,
    state_from_update,
    update_norm,
)
from verify_task43_c1_baselines_v414 import trimmed_mean, multi_krum

EXPERIMENT_VERSION = "4.14.C2a"
MODEL_SEED = 7
NUM_CLIENTS = 20
CONTINUATION_ROUNDS = 4
EMA_DECAY = 0.65
MALICIOUS_CLIENTS = (1, 7, 8, 10, 14, 15, 17, 18)
PLAIN_ARM = "plain_fedavg"
DETECTION_ARM = "D0_frozen_task40_lfighter"
BASELINE_ARMS = ("coordinate_median", "trimmed_mean", "multi_krum")
ALL_ARMS = (PLAIN_ARM, DETECTION_ARM, *BASELINE_ARMS)
EXTREME_VALUE_K = (2.0, 5.0, 10.0)
COORDINATED_BOOSTS = (3.0, 5.0, 10.0)
CONSTANT_MAGNITUDE = 2.37  # 95th-pct clean update norm, Task 42 C1b
TRIMMED_MEAN_FRACTION = 0.2  # symmetric trim: 4 of 20 clients per side
MULTI_KRUM_F = 8  # assumed Byzantine count, matches true coalition size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Task 43 C2 seed-7 Byzantine + baseline screen."
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--partition-file", type=Path, required=True)
    parser.add_argument("--clean-seed-dir", type=Path, required=True)
    parser.add_argument("--warmup-dir", type=Path, required=True)
    parser.add_argument("--reconstruction-calibration-dir", type=Path, required=True)
    parser.add_argument("--d0-threshold-file", type=Path, required=True)
    parser.add_argument(
        "--clean-capture-dir",
        type=Path,
        required=True,
        help=(
            "results\\cic_iot_diad_full_update_capture_v318b\\captures\\"
            "clean_seed_7 (for extreme_value's per-coordinate sigma)"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--probe-per-class", type=int, default=48)
    parser.add_argument("--probe-seed", type=int, default=3701)
    parser.add_argument("--constant-direction-seed", type=int, default=430901)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def save_figure(figure: plt.Figure, base: Path) -> None:
    figure.tight_layout()
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_output_manifest(output_root: Path) -> Path:
    manifest_path = output_root / "summary" / "tables" / "task43c2_summary_manifest_sha256.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(
        candidate
        for candidate in output_root.rglob("*")
        if candidate.is_file() and candidate.resolve() != manifest_path.resolve()
    ):
        rows.append(
            {
                "relative_path": str(path.relative_to(output_root)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


class Condition:
    def __init__(self, condition_id: str, attack_type: str, strength: float | None):
        self.condition_id = condition_id
        self.attack_type = attack_type
        self.strength = strength

    @property
    def is_attack(self) -> bool:
        return self.attack_type != "clean"


def frozen_conditions() -> List[Condition]:
    conditions = [Condition("clean_reference", "clean", None)]
    conditions.append(Condition("A1_constant", "constant", None))
    for k in EXTREME_VALUE_K:
        conditions.append(Condition(f"A1_extreme_value_{k:g}x", "extreme_value", k))
    for boost in COORDINATED_BOOSTS:
        conditions.append(
            Condition(f"A1_coordinated_split_{boost:g}x", "coordinated_split", boost)
        )
    return conditions


def build_target_state(
    reference_state: Mapping[str, torch.Tensor], benign_index: int
) -> Dict[str, torch.Tensor]:
    """Collapse-to-Benign target, identical to Task 42."""
    target = {key: value.clone() for key, value in reference_state.items()}
    target["classifier.weight"] = torch.zeros_like(target["classifier.weight"])
    bias = torch.zeros_like(target["classifier.bias"])
    bias[benign_index] = 10.0
    target["classifier.bias"] = bias
    return target


def build_constant_vectors(
    parameter_names: List[str],
    reference_state: Mapping[str, torch.Tensor],
    magnitude: float,
    seed: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """One fixed direction per malicious client, drawn once, reused every round."""
    vectors: Dict[int, Dict[str, torch.Tensor]] = {}
    for client_id in MALICIOUS_CLIENTS:
        generator = torch.Generator().manual_seed(seed + client_id)
        pieces = {}
        raw_norm_sq = 0.0
        for name in parameter_names:
            shape = reference_state[name].shape
            draw = torch.randn(shape, generator=generator)
            pieces[name] = draw
            raw_norm_sq += float(torch.sum(draw.to(torch.float64) ** 2))
        raw_norm = float(np.sqrt(raw_norm_sq)) if raw_norm_sq > 0 else 1.0
        scale = magnitude / raw_norm
        vectors[client_id] = {name: scale * pieces[name] for name in parameter_names}
    return vectors


def compute_coordinate_sigma(
    clean_capture_dir: Path,
    parameter_names: List[str],
    reference_state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Per-coordinate std of update values across the captured clean seed-7
    continuation rounds (Task 42's C1 capture infrastructure, reused).

    Shapes are taken from reference_state (the guaranteed-correct source
    of each parameter's real shape) rather than parsed from the layout
    CSV, since the CSV's own shape representation is not confirmed and a
    silent flat-vs-shaped mismatch here would corrupt every extreme_value
    branch without necessarily raising an obvious error.
    """
    index = pd.read_csv(clean_capture_dir / "tables" / "v318b_update_capture_index.csv")
    layout = pd.read_csv(clean_capture_dir / "update_artifacts" / "parameter_layout.csv")
    included = layout[layout["included_in_update_vector"].astype(bool)].sort_values(
        "start_index"
    )
    layout_names = included["parameter_name"].astype(str).tolist()
    if layout_names != parameter_names:
        raise RuntimeError(
            "Captured update matrix parameter order does not match this "
            "run's parameter_names -- cannot safely compute per-coordinate "
            "sigma without risking silent misalignment."
        )
    all_rows = []
    for _, row in index.iterrows():
        matrix_path = clean_capture_dir / str(row["update_matrix_file"])
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        all_rows.append(np.asarray(matrix, dtype=np.float64))
    stacked = np.concatenate(all_rows, axis=0)  # (rounds*clients, parameter_count)
    sigma_flat = stacked.std(axis=0)
    sigma_flat = np.maximum(sigma_flat, 1e-8)  # avoid zero-sigma coordinates

    sigma: Dict[str, torch.Tensor] = {}
    offset = 0
    for _, row in included.iterrows():
        name = str(row["parameter_name"])
        numel = int(row["numel"])
        segment = sigma_flat[offset : offset + numel]
        expected_shape = tuple(reference_state[name].shape)
        if int(np.prod(expected_shape)) != numel:
            raise RuntimeError(
                f"Parameter {name!r}: layout numel={numel} does not match "
                f"reference_state shape {expected_shape} "
                f"(product={int(np.prod(expected_shape))})."
            )
        sigma[name] = torch.from_numpy(
            segment.astype(np.float32)
        ).reshape(expected_shape)
        offset += numel
    return sigma


def apply_attack_transform(
    *,
    condition: Condition,
    client_id: int,
    honest_update: Dict[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor] | None,
    constant_vectors: Dict[int, Dict[str, torch.Tensor]] | None,
    coordinate_sigma: Dict[str, torch.Tensor] | None,
    honest_median_direction: Dict[str, torch.Tensor] | None,
) -> Dict[str, torch.Tensor]:
    if condition.attack_type == "constant":
        return constant_vectors[client_id]

    if condition.attack_type == "extreme_value":
        k = float(condition.strength)
        result = {}
        for name, sigma_tensor in coordinate_sigma.items():
            opposing_sign = -torch.sign(honest_median_direction[name])
            opposing_sign = torch.where(
                opposing_sign == 0, torch.ones_like(opposing_sign), opposing_sign
            )
            result[name] = opposing_sign * k * sigma_tensor
        return result

    if condition.attack_type == "coordinated_split":
        boost = float(condition.strength)
        return {
            key: (boost / len(MALICIOUS_CLIENTS)) * (target_state[key] - reference_state[key])
            for key in reference_state
        }

    raise ValueError(f"Unknown attack type: {condition.attack_type}")


class FrozenContext:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def load_frozen_context(args: argparse.Namespace) -> FrozenContext:
    with np.load(args.data_file.expanduser().resolve(), allow_pickle=False) as archive:
        X_train = np.asarray(archive["X_train"], dtype=np.float32)
        y_train = np.asarray(archive["y_train"], dtype=np.int64)
        X_val = np.asarray(archive["X_val"], dtype=np.float32)
        y_val = np.asarray(archive["y_val"], dtype=np.int64)

    client_indices, partition_hash = load_fixed_partitions(
        args.partition_file.expanduser().resolve(),
        expected_clients=NUM_CLIENTS,
        train_rows=len(y_train),
    )
    clean_hash = read_clean_partition_hash(args.clean_seed_dir.expanduser().resolve())
    if partition_hash != clean_hash:
        raise RuntimeError("Clean seed partition hash mismatch.")

    warmup_dir = args.warmup_dir.expanduser().resolve()
    warmup_metadata = json.loads(
        (warmup_dir / "true_warmup_v310_metadata.json").read_text(encoding="utf-8")
    )
    warmup_checkpoint_path = warmup_dir / "checkpoints" / "common_round4_warmup_model.pt"
    warmup_checkpoint = torch.load(
        warmup_checkpoint_path, map_location="cpu", weights_only=False
    )
    if warmup_checkpoint.get("partition_hash") != partition_hash:
        raise RuntimeError("Warmup checkpoint partition mismatch.")

    probe_indices = balanced_probe_indices(y_val, args.probe_per_class, args.probe_seed)
    probe_hash = hashlib.sha256(np.ascontiguousarray(probe_indices).tobytes()).hexdigest()
    if probe_hash != warmup_metadata["probe_hash_sha256"]:
        raise RuntimeError("Warmup probe hash mismatch.")
    X_probe = X_val[probe_indices]
    y_probe = y_val[probe_indices]

    task40_profiles = load_profiles_npz(
        warmup_dir / "calibration" / "trusted_client_profiles.npz", NUM_CLIENTS
    )
    task40_feature_calibration = pd.read_csv(
        warmup_dir / "calibration" / "feature_calibration.csv"
    )

    d0_threshold_table = pd.read_csv(args.d0_threshold_file.expanduser().resolve())
    d0_row = d0_threshold_table[d0_threshold_table["candidate"] == DETECTION_ARM]
    if len(d0_row) != 1:
        raise RuntimeError(
            f"Expected exactly one {DETECTION_ARM} row in d0-threshold-file, "
            f"found {len(d0_row)}."
        )
    d0_instant_threshold = float(d0_row.iloc[0]["instant_threshold"])
    d0_ema_threshold = float(d0_row.iloc[0]["ema_threshold"])

    reconstruction_dir = args.reconstruction_calibration_dir.expanduser().resolve()
    reconstruction_bundle = torch.load(
        reconstruction_dir / "calibration" / "trusted_update_reconstruction_profiles.pt",
        map_location="cpu",
        weights_only=False,
    )
    if int(reconstruction_bundle["model_seed"]) != MODEL_SEED:
        raise RuntimeError("Reconstruction bundle seed mismatch.")
    if reconstruction_bundle["partition_hash"] != partition_hash:
        raise RuntimeError("Reconstruction partition mismatch.")
    selected_policy = str(reconstruction_bundle["selected_policy"])
    if selected_policy != "center_plus_residual":
        raise RuntimeError("Task 43 requires center_plus_residual reconstruction for D0.")
    reconstruction_profiles = {
        int(client_id): {
            key: value.detach().cpu().to(torch.float32)
            for key, value in profile.items()
        }
        for client_id, profile in reconstruction_bundle["client_residual_profiles"].items()
    }

    probe_model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    parameter_names = [name for name, _ in probe_model.state_dict().items()]
    del probe_model

    benign_index = CLASS_NAMES.index("Benign")
    target_state = build_target_state(warmup_checkpoint["model_state_dict"], benign_index)

    constant_vectors = build_constant_vectors(
        parameter_names,
        warmup_checkpoint["model_state_dict"],
        CONSTANT_MAGNITUDE,
        args.constant_direction_seed,
    )

    coordinate_sigma = compute_coordinate_sigma(
        args.clean_capture_dir.expanduser().resolve(),
        parameter_names,
        warmup_checkpoint["model_state_dict"],
    )

    return FrozenContext(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        client_indices=client_indices,
        partition_hash=partition_hash,
        X_probe=X_probe,
        y_probe=y_probe,
        probe_hash=probe_hash,
        warmup_checkpoint=warmup_checkpoint,
        task40_profiles=task40_profiles,
        task40_feature_calibration=task40_feature_calibration,
        reconstruction_profiles=reconstruction_profiles,
        selected_reconstruction_policy=selected_policy,
        warmup_update_scale=float(reconstruction_bundle["warmup_update_scale"]),
        scale_lower=float(reconstruction_bundle["scale_lower"]),
        scale_upper=float(reconstruction_bundle["scale_upper"]),
        norm_clip_multiplier=float(reconstruction_bundle["norm_clip_multiplier"]),
        parameter_names=parameter_names,
        d0_instant_threshold=d0_instant_threshold,
        d0_ema_threshold=d0_ema_threshold,
        target_state=target_state,
        constant_vectors=constant_vectors,
        coordinate_sigma=coordinate_sigma,
        source_id=CLASS_NAMES.index("DDoS"),
        target_id=benign_index,
    )


def d0_score_table(
    *,
    local_matrices,
    sample_counts,
    context: FrozenContext,
    monitoring_round: int,
    actual_malicious,
    ema_memory: Dict[int, float],
):
    matrix_stack = np.stack(local_matrices, axis=0)
    consensus = normalize_rows(np.median(matrix_stack, axis=0))
    raw_rows: List[Dict[str, Any]] = []
    for client_id, local_matrix_raw in enumerate(local_matrices):
        local_matrix = normalize_rows(local_matrix_raw)
        raw_rows.append(
            {
                "monitoring_round": monitoring_round,
                "client_id": client_id,
                "actual_malicious": bool(actual_malicious[client_id]),
                **raw_features(
                    local_matrix,
                    context.task40_profiles[client_id],
                    consensus,
                    context.source_id,
                    context.target_id,
                ),
            }
        )
    scored = apply_feature_calibration(pd.DataFrame(raw_rows), context.task40_feature_calibration)
    scored, updated_ema = add_candidate_and_ema(scored, EMA_DECAY, initial_ema=ema_memory)
    return scored.sort_values("client_id").reset_index(drop=True), updated_ema


def run_branch(
    *, condition: Condition, arm: str, context: FrozenContext, output_root: Path, args
) -> Dict[str, Any]:
    branch_dir = output_root / "branches" / condition.condition_id / arm
    complete_path = branch_dir / "branch_complete.json"
    if complete_path.exists():
        metadata = json.loads(complete_path.read_text(encoding="utf-8"))
        if metadata.get("complete") is True:
            print(f"SKIP COMPLETE: {condition.condition_id} | {arm}")
            return metadata
    if branch_dir.exists() and any(branch_dir.iterdir()):
        if args.resume:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            shutil.move(str(branch_dir), str(branch_dir) + f"_partial_{stamp}")
        else:
            raise FileExistsError(f"Partial branch exists without --resume: {branch_dir}")

    tables_dir = branch_dir / "tables"
    figures_dir = branch_dir / "figures"
    for path in (tables_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    set_seed(MODEL_SEED)
    model = build_model("resmlp", context.X_train.shape[1], NUM_CLASSES)
    model.load_state_dict(context.warmup_checkpoint["model_state_dict"])
    class_weights = sqrt_class_weights(context.y_train, args.max_class_weight)

    actual_malicious = [
        condition.is_attack and client_id in MALICIOUS_CLIENTS
        for client_id in range(NUM_CLIENTS)
    ]

    d0_ema_memory: Dict[int, float] = {}
    round_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    branch_started = time.time()

    print()
    print(f"RUN: {condition.condition_id} | {arm}")

    for monitoring_round in range(1, CONTINUATION_ROUNDS + 1):
        global_round = 4 + monitoring_round
        round_started = time.time()
        reference_state = copy.deepcopy(model.state_dict())
        local_states: List[Mapping[str, torch.Tensor]] = []
        honest_states: List[Mapping[str, torch.Tensor]] = []
        sample_counts: List[int] = []
        local_matrices: List[np.ndarray] = []

        for client_id in range(NUM_CLIENTS):
            indices = context.client_indices[client_id]
            local_model = build_model("resmlp", context.X_train.shape[1], NUM_CLASSES)
            local_model.load_state_dict(reference_state)
            honest_state, _ = train_local_model(
                model=local_model,
                X=context.X_train[indices],
                y=context.y_train[indices],
                class_weights=class_weights,
                local_epochs=1,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                seed=MODEL_SEED + global_round * 1000 + client_id,
            )
            del local_model
            honest_states.append(honest_state)
            sample_counts.append(int(len(indices)))

        # extreme_value needs this round's honest coordinate-median as an
        # omniscient-attacker proxy for "trusted center" -- computed AFTER
        # all honest local training this round, before any attack is applied.
        honest_median_direction = None
        if condition.attack_type == "extreme_value":
            honest_updates_all = [
                {key: honest_states[c][key] - reference_state[key] for key in reference_state}
                for c in range(NUM_CLIENTS)
            ]
            honest_median_direction = coordinate_median(honest_updates_all)

        for client_id in range(NUM_CLIENTS):
            if actual_malicious[client_id]:
                honest_update = {
                    key: honest_states[client_id][key] - reference_state[key]
                    for key in reference_state
                }
                malicious_update = apply_attack_transform(
                    condition=condition,
                    client_id=client_id,
                    honest_update=honest_update,
                    reference_state=reference_state,
                    target_state=context.target_state,
                    constant_vectors=context.constant_vectors,
                    coordinate_sigma=context.coordinate_sigma,
                    honest_median_direction=honest_median_direction,
                )
                submitted_state = {
                    key: reference_state[key] + malicious_update[key]
                    for key in reference_state
                }
            else:
                submitted_state = honest_states[client_id]

            probe_model = build_model("resmlp", context.X_train.shape[1], NUM_CLASSES)
            probe_model.load_state_dict(submitted_state)
            probabilities = predict_probabilities(
                probe_model, context.X_probe, args.evaluation_batch_size
            )
            local_means = class_conditional_probability_means(probabilities, context.y_probe)
            del probe_model

            local_states.append(submitted_state)
            local_matrices.append(local_means)

        updates_all = [
            floating_update(state, reference_state) for state in local_states
        ]

        flags = np.zeros(NUM_CLIENTS, dtype=bool)
        replaced_clients = 0

        if arm == PLAIN_ARM:
            aggregated_state = weighted_average_states(local_states, sample_counts, reference_state)

        elif arm == DETECTION_ARM:
            d0_table, d0_ema_memory = d0_score_table(
                local_matrices=local_matrices,
                sample_counts=sample_counts,
                context=context,
                monitoring_round=monitoring_round,
                actual_malicious=actual_malicious,
                ema_memory=d0_ema_memory,
            )
            instant = d0_table[D0_INSTANT].to_numpy(dtype=float)
            ema = d0_table[D0_EMA].to_numpy(dtype=float)
            flags = (instant > context.d0_instant_threshold) | (ema > context.d0_ema_threshold)
            d0_table["flagged"] = flags
            d0_table["monitoring_round"] = monitoring_round
            score_rows.extend(d0_table.to_dict(orient="records"))

            trusted_ids = [c for c in range(NUM_CLIENTS) if not flags[c]]
            if len(trusted_ids) == 0:
                aggregated_state = {key: value.clone() for key, value in reference_state.items()}
            else:
                replacement_states = list(local_states)
                trusted_updates = [updates_all[c] for c in trusted_ids]
                trusted_norms = [update_norm(u) for u in trusted_updates]
                current_center = coordinate_median(trusted_updates)
                current_update_scale = float(np.median(trusted_norms))
                for client_id in range(NUM_CLIENTS):
                    if flags[client_id]:
                        reconstructed_update, _ = reconstruct_update(
                            current_center=current_center,
                            historical_residual=context.reconstruction_profiles[client_id],
                            selected_policy=context.selected_reconstruction_policy,
                            current_update_scale=current_update_scale,
                            warmup_update_scale=context.warmup_update_scale,
                            trusted_norms=trusted_norms,
                            scale_lower=context.scale_lower,
                            scale_upper=context.scale_upper,
                            norm_clip_multiplier=context.norm_clip_multiplier,
                        )
                        replacement_states[client_id] = state_from_update(
                            reference_state, reconstructed_update
                        )
                        replaced_clients += 1
                aggregated_state = weighted_average_states(
                    replacement_states, sample_counts, reference_state
                )

        elif arm == "coordinate_median":
            aggregated_update = coordinate_median(updates_all)
            aggregated_state = state_from_update(reference_state, aggregated_update)

        elif arm == "trimmed_mean":
            aggregated_update = trimmed_mean(updates_all, trim_fraction=TRIMMED_MEAN_FRACTION)
            aggregated_state = state_from_update(reference_state, aggregated_update)

        elif arm == "multi_krum":
            aggregated_update, selected = multi_krum(updates_all, assumed_byzantine_count=MULTI_KRUM_F)
            aggregated_state = state_from_update(reference_state, aggregated_update)
        else:
            raise ValueError(f"Unknown arm: {arm}")

        model.load_state_dict(aggregated_state)
        val_metrics, val_pair = evaluate_validation(
            model, context.X_val, context.y_val, context.source_id, context.target_id,
            args.evaluation_batch_size,
        )

        labels = np.asarray(actual_malicious, dtype=bool)
        benign = ~labels
        if arm == DETECTION_ARM and condition.is_attack:
            malicious_recall = float(np.mean(flags[labels])) if labels.any() else float("nan")
            benign_fpr = float(np.mean(flags[benign]))
        elif arm == DETECTION_ARM:
            malicious_recall = float("nan")
            benign_fpr = float(np.mean(flags[benign]))
        else:
            malicious_recall = float("nan")
            benign_fpr = float("nan")

        row = {
            "condition_id": condition.condition_id,
            "attack_type": condition.attack_type,
            "attack_strength": condition.strength,
            "arm": arm,
            "monitoring_round": monitoring_round,
            "global_round": global_round,
            "flagged_clients": int(flags.sum()),
            "replaced_clients": int(replaced_clients),
            "malicious_recall": malicious_recall,
            "benign_false_positive_rate": benign_fpr,
            "round_seconds": float(time.time() - round_started),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"val_{k}": v for k, v in val_pair.items()},
        }
        round_rows.append(row)
        print(
            f"  round {monitoring_round}: macroF1={row['val_macro_f1']:.4f}, "
            f"recall={malicious_recall}, FPR={benign_fpr}, "
            f"replaced={replaced_clients}, seconds={row['round_seconds']:.1f}"
        )

    round_table = pd.DataFrame(round_rows)
    round_table.to_csv(tables_dir / "branch_round_metrics.csv", index=False)
    if score_rows:
        pd.DataFrame(score_rows).to_csv(tables_dir / "branch_client_scores.csv", index=False)

    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    axis.plot(round_table["monitoring_round"], round_table["val_macro_f1"], marker="o")
    axis.set_xlabel("Monitoring round")
    axis.set_ylabel("Clean validation macro F1")
    axis.set_title(f"{condition.condition_id} | {arm}")
    axis.grid(alpha=0.25)
    save_figure(figure, figures_dir / "branch_macro_f1")

    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "complete": True,
        "condition_id": condition.condition_id,
        "arm": arm,
        "model_seed": MODEL_SEED,
        "total_seconds": float(time.time() - branch_started),
        "mean_macro_f1": float(round_table["val_macro_f1"].mean()),
    }
    (branch_dir / "branch_complete.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined.")
    torch.set_num_threads(args.threads)

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_root)
        elif not args.resume:
            raise FileExistsError(f"Output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    context = load_frozen_context(args)
    conditions = frozen_conditions()

    if args.preflight_only:
        print("Preflight OK.")
        print("D0 instant threshold:", context.d0_instant_threshold)
        print("D0 EMA threshold:", context.d0_ema_threshold)
        print("Constant vector magnitude check (client 1):", float(
            torch.sqrt(sum(torch.sum(v.to(torch.float64) ** 2) for v in context.constant_vectors[1].values()))
        ))
        print("Conditions:", [c.condition_id for c in conditions])
        print("Arms:", ALL_ARMS)
        return 0

    for condition in conditions:
        for arm in ALL_ARMS:
            run_branch(condition=condition, arm=arm, context=context, output_root=output_root, args=args)

    write_output_manifest(output_root)
    print("Task 43 C2 screen complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

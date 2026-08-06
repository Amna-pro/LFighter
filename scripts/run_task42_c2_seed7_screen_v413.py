#!/usr/bin/env python3
"""Task 42 C2: seed-7 development screen for model-update poisoning.

Scope
-----
- Seed 7 only (development screen; confirmatory multiseed is a later stage,
  matching the C2->C3 pattern established in Task 41C).
- Clean reference plus 12 attack conditions: 4 attack types x 3 strengths.
- Five arms: plain_fedavg, D0 (frozen Task 40 LFighter, mandatory
  reference), D1 (update-norm deviation), D2 (update-direction deviation),
  D3 (equal-weight fusion of D0/D1/D2).
- Four post-warmup monitoring rounds, identical warmup/partition/coalition
  to every prior stage.
- Development arrays only (X_train, y_train, X_val, y_val). No reserved
  test access.

Attack mechanics (all four apply to the SUBMITTED UPDATE only; local
training is always honest and identical for every client, malicious or
not -- these attacks bypass local training entirely, which is the point):

- sign_flip: malicious_update = -honest_update
- scale: malicious_update = scale_factor * honest_update
  (scale_factor in {2, 5, 10}, per the preregistration's grounding in the
  observed clean update-norm distribution)
- additive_noise: malicious_update = honest_update + noise, where noise is
  a random Gaussian direction rescaled so its L2 norm equals
  noise_fraction * ||honest_update|| (noise_fraction in {0.5, 1.0, 2.0})
- model_replacement: malicious_update = boost * (target_state -
  reference_state), REPLACING the honest update entirely. target_state is
  the warmup reference state with classifier.weight zeroed and
  classifier.bias set to strongly favor the Benign class -- a classifier
  that always predicts Benign regardless of input. boost in {3, 5, 10},
  grounded in the exact sample-weighted malicious fraction f=0.3368
  computed from this seed's frozen partition (1/f ~= 2.97 ~= 3x textbook
  boost).

D1/D2 thresholds are instant-only (q99), matching the C1b calibration
decision: these attacks corrupt a single submitted update directly, with
no gradual accumulation to wait for, so EMA's purpose (catching drift
that instant thresholds alone would miss) does not apply the way it does
to label-poisoning attacks.

D3 fusion here is a straightforward equal-weight average of each round's
D0/D1/D2 scores after independently rank-normalizing each to [0,1] within
that round's 20 clients (empirical percentile rank). This is a simpler
fusion than Task 41C's stored-empirical-reference approach, documented
explicitly here rather than silently matching a more complex prior
pattern this task has no equivalent calibration artifact for yet.

Primary evaluation metric is clean validation macro-F1 damage/recovery,
matching Task 40's convention -- these attacks are untargeted in
update-space (except model_replacement's Benign-collapse target, which is
still evaluated primarily via macro-F1 and secondarily via Benign
recall/precision shift).
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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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

EXPERIMENT_VERSION = "4.13.C2a"
MODEL_SEED = 7
NUM_CLIENTS = 20
CONTINUATION_ROUNDS = 4
EMA_DECAY = 0.65
MALICIOUS_CLIENTS = (1, 7, 8, 10, 14, 15, 17, 18)
PLAIN_ARM = "plain_fedavg"
CANDIDATES = (
    "D0_frozen_task40_lfighter",
    "D1_update_norm_deviation",
    "D2_update_direction_deviation",
    "D3_equal_weight_fusion",
)
ALL_ARMS = (PLAIN_ARM, *CANDIDATES)
SCALE_FACTORS = (2.0, 5.0, 10.0)
NOISE_FRACTIONS = (0.5, 1.0, 2.0)
REPLACEMENT_BOOSTS = (3.0, 5.0, 10.0)
D1_THRESHOLD_QUANTILE = 0.99
D2_THRESHOLD_QUANTILE = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Task 42 C2 seed-7 model-update-poisoning screen."
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--partition-file", type=Path, required=True)
    parser.add_argument("--clean-seed-dir", type=Path, required=True)
    parser.add_argument("--warmup-dir", type=Path, required=True)
    parser.add_argument("--reconstruction-calibration-dir", type=Path, required=True)
    parser.add_argument(
        "--d0-threshold-file",
        type=Path,
        required=True,
        help=(
            "Task 41C's task41c1c_candidate_thresholds.csv (seed 7), "
            "carrying D0's real frozen instant_threshold/ema_threshold"
        ),
    )
    parser.add_argument(
        "--c1b-threshold-file",
        type=Path,
        required=True,
        help="task42c1b_pooled_threshold_candidates.csv",
    )
    parser.add_argument(
        "--c1b-scores-file",
        type=Path,
        required=True,
        help="task42c1b_all_clean_scores.csv (for D1/D2 per-client baselines)",
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
    parser.add_argument("--noise-seed", type=int, default=421301)
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
    manifest_path = output_root / "summary" / "tables" / "task42c2_summary_manifest_sha256.csv"
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


def stable_text_hash(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


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
    for factor in SCALE_FACTORS:
        conditions.append(
            Condition(f"A1_scale_{factor:g}x", "scale", factor)
        )
    for fraction in NOISE_FRACTIONS:
        conditions.append(
            Condition(
                f"A1_additive_noise_{fraction:g}x".replace(".", "p"),
                "additive_noise",
                fraction,
            )
        )
    for boost in REPLACEMENT_BOOSTS:
        conditions.append(
            Condition(
                f"A1_model_replacement_{boost:g}x".replace(".", "p"),
                "model_replacement",
                boost,
            )
        )
    conditions.append(Condition("A1_sign_flip", "sign_flip", None))
    return conditions


def build_target_state(
    reference_state: Mapping[str, torch.Tensor], benign_index: int
) -> Dict[str, torch.Tensor]:
    """Collapse-to-Benign target for the model-replacement attack.

    classifier.weight -> zero (removes all input dependence)
    classifier.bias   -> large positive for Benign, zero elsewhere
    Every other parameter is left identical to the reference state (the
    attacker only needs to corrupt the decision layer to achieve a
    constant-Benign classifier; corrupting earlier layers would just add
    unnecessary norm without changing the attack's actual effect).
    """
    target = {key: value.clone() for key, value in reference_state.items()}
    if "classifier.weight" not in target or "classifier.bias" not in target:
        raise RuntimeError(
            "Expected 'classifier.weight' and 'classifier.bias' in the "
            "reference state_dict -- architecture assumption violated."
        )
    target["classifier.weight"] = torch.zeros_like(target["classifier.weight"])
    bias = torch.zeros_like(target["classifier.bias"])
    bias[benign_index] = 10.0
    target["classifier.bias"] = bias
    return target


def apply_attack_transform(
    *,
    condition: Condition,
    honest_update: Dict[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor] | None,
    rng: np.random.Generator,
) -> Dict[str, torch.Tensor]:
    if condition.attack_type == "sign_flip":
        return {key: -value for key, value in honest_update.items()}

    if condition.attack_type == "scale":
        factor = float(condition.strength)
        return {key: factor * value for key, value in honest_update.items()}

    if condition.attack_type == "additive_noise":
        fraction = float(condition.strength)
        honest_norm = float(
            torch.sqrt(
                sum(
                    torch.sum(value.to(torch.float64) ** 2)
                    for value in honest_update.values()
                )
            )
        )
        target_noise_norm = fraction * honest_norm
        noise: Dict[str, torch.Tensor] = {}
        raw_norm_sq = 0.0
        for key, value in honest_update.items():
            draw = rng.standard_normal(size=tuple(value.shape)).astype(np.float32)
            tensor = torch.from_numpy(draw)
            noise[key] = tensor
            raw_norm_sq += float(np.sum(draw.astype(np.float64) ** 2))
        raw_norm = float(np.sqrt(raw_norm_sq)) if raw_norm_sq > 0 else 1.0
        scale = target_noise_norm / raw_norm
        return {
            key: honest_update[key] + scale * noise[key] for key in honest_update
        }

    if condition.attack_type == "model_replacement":
        if target_state is None:
            raise RuntimeError("model_replacement requires a target_state.")
        boost = float(condition.strength)
        return {
            key: boost * (target_state[key] - reference_state[key])
            for key in reference_state
        }

    raise ValueError(f"Unknown attack type: {condition.attack_type}")


def load_c1_thresholds(
    threshold_file: Path, scores_file: Path, model_seed: int
) -> Tuple[float, float, pd.DataFrame]:
    thresholds = pd.read_csv(threshold_file)
    row = thresholds[
        np.isclose(thresholds["quantile"], D1_THRESHOLD_QUANTILE)
    ]
    if len(row) != 1:
        raise RuntimeError(
            f"Expected exactly one C1b threshold row at quantile "
            f"{D1_THRESHOLD_QUANTILE}, found {len(row)}."
        )
    d1_threshold = float(row.iloc[0]["d1_instant_threshold"])
    d2_threshold = float(row.iloc[0]["d2_instant_threshold"])

    scores = pd.read_csv(scores_file)
    # task42c1b_all_clean_scores.csv stores a PER-SEED D1 baseline for
    # each client (each seed's warmup rounds produce that seed's own
    # median/scale) -- must filter to this run's model_seed before
    # deduplicating, or client baselines from all four seeds collide.
    seed_scores = scores[scores["model_seed"] == model_seed]
    if len(seed_scores) == 0:
        raise RuntimeError(
            f"No rows found for model_seed={model_seed} in {scores_file}."
        )
    d1_baseline = (
        seed_scores[["client_id", "log_norm_median", "log_norm_robust_scale"]]
        .drop_duplicates()
        .sort_values("client_id")
        .reset_index(drop=True)
    )
    if len(d1_baseline) != NUM_CLIENTS:
        raise RuntimeError(
            f"Expected {NUM_CLIENTS} unique client D1 baselines for "
            f"model_seed={model_seed} in {scores_file}, found "
            f"{len(d1_baseline)}."
        )
    return d1_threshold, d2_threshold, d1_baseline


def d0_score_table(
    *,
    local_matrices: Sequence[np.ndarray],
    sample_counts: Sequence[int],
    context: "FrozenContext",
    monitoring_round: int,
    actual_malicious: Sequence[bool],
    ema_memory: Dict[int, float],
) -> Tuple[pd.DataFrame, Dict[int, float]]:
    matrix_stack = np.stack(local_matrices, axis=0)
    consensus = normalize_rows(np.median(matrix_stack, axis=0))
    raw_weights = np.asarray(sample_counts, dtype=np.float64) / float(
        np.sum(sample_counts)
    )
    raw_rows: List[Dict[str, Any]] = []
    for client_id, local_matrix_raw in enumerate(local_matrices):
        local_matrix = normalize_rows(local_matrix_raw)
        raw_rows.append(
            {
                "monitoring_round": monitoring_round,
                "client_id": client_id,
                "actual_malicious": bool(actual_malicious[client_id]),
                "client_samples": int(sample_counts[client_id]),
                "aggregation_weight": float(raw_weights[client_id]),
                **raw_features(
                    local_matrix,
                    context.task40_profiles[client_id],
                    consensus,
                    context.source_id,
                    context.target_id,
                ),
            }
        )
    scored = apply_feature_calibration(
        pd.DataFrame(raw_rows), context.task40_feature_calibration
    )
    scored, updated_ema = add_candidate_and_ema(
        scored, EMA_DECAY, initial_ema=ema_memory
    )
    return scored.sort_values("client_id").reset_index(drop=True), updated_ema


def d1_d2_scores(
    *,
    reference_state: Mapping[str, torch.Tensor],
    local_states: Sequence[Mapping[str, torch.Tensor]],
    d1_baseline: pd.DataFrame,
    d2_reference_directions: np.ndarray,
    parameter_names: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    baseline_by_client = d1_baseline.set_index("client_id")
    for client_id, local_state in enumerate(local_states):
        update = torch.cat(
            [
                (local_state[name] - reference_state[name])
                .detach()
                .cpu()
                .reshape(-1)
                .to(torch.float64)
                for name in parameter_names
            ]
        ).numpy()
        norm = float(np.linalg.norm(update))
        log_z = (
            np.log1p(norm) - float(baseline_by_client.loc[client_id, "log_norm_median"])
        ) / float(baseline_by_client.loc[client_id, "log_norm_robust_scale"])
        if norm > 1e-12:
            cosine = float(
                np.dot(update / norm, d2_reference_directions[client_id])
            )
        else:
            cosine = float("nan")
        deviation = 1.0 - cosine
        rows.append(
            {
                "client_id": client_id,
                "d1_raw_norm": norm,
                "d1_log_robust_z": log_z,
                "d2_cosine_similarity": cosine,
                "d2_deviation": deviation,
            }
        )
    return pd.DataFrame(rows).sort_values("client_id").reset_index(drop=True)


class FrozenContext:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def load_frozen_context(args: argparse.Namespace) -> FrozenContext:
    arrays_path = args.data_file.expanduser().resolve()
    with np.load(arrays_path, allow_pickle=False) as archive:
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
    probe_hash = hashlib.sha256(
        np.ascontiguousarray(probe_indices).tobytes()
    ).hexdigest()
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
    # D0's real frozen dual thresholds (EMA q99 + instant q95, from
    # V3.10.2/V3.10.3) do NOT live in the warmup module's
    # calibration_summary.csv -- that file only has a single
    # 'clean_ema_threshold' from an earlier, simpler calibration stage.
    # The actual frozen values are already stored, per-seed, in Task 41C's
    # own C1c3 calibration output (task41c1c_candidate_thresholds.csv),
    # which itself required and carried a D0_frozen_task40_lfighter row
    # with both columns. Reusing that existing frozen artifact directly
    # rather than re-deriving anything.
    d0_threshold_path = args.d0_threshold_file.expanduser().resolve()
    if not d0_threshold_path.exists():
        raise FileNotFoundError(d0_threshold_path)
    d0_threshold_table = pd.read_csv(d0_threshold_path)
    d0_row = d0_threshold_table[
        d0_threshold_table["candidate"] == "D0_frozen_task40_lfighter"
    ]
    if len(d0_row) != 1:
        raise RuntimeError(
            f"Expected exactly one D0_frozen_task40_lfighter row in "
            f"{d0_threshold_path}, found {len(d0_row)}. Columns present: "
            f"{list(d0_threshold_table.columns)}."
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
        raise RuntimeError("Task 42 requires center_plus_residual reconstruction.")
    reconstruction_profiles = {
        int(client_id): {
            key: value.detach().cpu().to(torch.float32)
            for key, value in profile.items()
        }
        for client_id, profile in reconstruction_bundle[
            "client_residual_profiles"
        ].items()
    }

    probe_model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    parameter_names = [name for name, _ in probe_model.state_dict().items()]
    del probe_model

    d1_threshold, d2_threshold, d1_baseline = load_c1_thresholds(
        args.c1b_threshold_file, args.c1b_scores_file, MODEL_SEED
    )

    # D2 reference directions, flattened in the SAME order as
    # parameter_names above (state_dict insertion order) -- must match
    # exactly how update vectors are flattened during scoring, or cosine
    # similarity is meaningless.
    d2_reference_directions = np.empty(
        (NUM_CLIENTS, sum(v.numel() for v in reconstruction_profiles[0].values())),
        dtype=np.float64,
    )
    for client_id in range(NUM_CLIENTS):
        profile = reconstruction_profiles[client_id]
        flat = torch.cat(
            [profile[name].detach().cpu().reshape(-1).to(torch.float64) for name in parameter_names]
        ).numpy()
        norm = float(np.linalg.norm(flat))
        if norm <= 1e-12:
            raise RuntimeError(f"Client {client_id}: degenerate D2 reference direction.")
        d2_reference_directions[client_id] = flat / norm

    benign_index = CLASS_NAMES.index("Benign")
    target_state = build_target_state(
        warmup_checkpoint["model_state_dict"], benign_index
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
        warmup_checkpoint_path=warmup_checkpoint_path,
        task40_profiles=task40_profiles,
        task40_feature_calibration=task40_feature_calibration,
        reconstruction_profiles=reconstruction_profiles,
        selected_reconstruction_policy=selected_policy,
        warmup_update_scale=float(reconstruction_bundle["warmup_update_scale"]),
        scale_lower=float(reconstruction_bundle["scale_lower"]),
        scale_upper=float(reconstruction_bundle["scale_upper"]),
        norm_clip_multiplier=float(reconstruction_bundle["norm_clip_multiplier"]),
        parameter_names=parameter_names,
        d1_threshold=d1_threshold,
        d2_threshold=d2_threshold,
        d1_baseline=d1_baseline,
        d2_reference_directions=d2_reference_directions,
        d0_ema_threshold=d0_ema_threshold,
        d0_instant_threshold=d0_instant_threshold,
        target_state=target_state,
        source_id=CLASS_NAMES.index("DDoS"),
        target_id=benign_index,
    )


def run_branch(
    *,
    condition: Condition,
    arm: str,
    context: FrozenContext,
    output_root: Path,
    args: argparse.Namespace,
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

    rng = np.random.default_rng(args.noise_seed + hash(condition.condition_id) % 1_000_000)

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

            if actual_malicious[client_id]:
                honest_update = {
                    key: honest_state[key] - reference_state[key]
                    for key in reference_state
                }
                malicious_update = apply_attack_transform(
                    condition=condition,
                    honest_update=honest_update,
                    reference_state=reference_state,
                    target_state=context.target_state,
                    rng=rng,
                )
                submitted_state = {
                    key: reference_state[key] + malicious_update[key]
                    for key in reference_state
                }
            else:
                submitted_state = honest_state

            probe_model = build_model("resmlp", context.X_train.shape[1], NUM_CLASSES)
            probe_model.load_state_dict(submitted_state)
            probabilities = predict_probabilities(
                probe_model, context.X_probe, args.evaluation_batch_size
            )
            local_means = class_conditional_probability_means(
                probabilities, context.y_probe
            )
            del probe_model

            local_states.append(submitted_state)
            sample_counts.append(int(len(indices)))
            local_matrices.append(local_means)

        d0_table, d0_ema_memory = d0_score_table(
            local_matrices=local_matrices,
            sample_counts=sample_counts,
            context=context,
            monitoring_round=monitoring_round,
            actual_malicious=actual_malicious,
            ema_memory=d0_ema_memory,
        )
        d1_d2_table = d1_d2_scores(
            reference_state=reference_state,
            local_states=local_states,
            d1_baseline=context.d1_baseline,
            d2_reference_directions=context.d2_reference_directions,
            parameter_names=context.parameter_names,
        )
        merged = d0_table.merge(d1_d2_table, on="client_id", how="inner", validate="one_to_one")

        flags = np.zeros(NUM_CLIENTS, dtype=bool)
        replacement_states = list(local_states)
        replaced_clients = 0

        if arm != PLAIN_ARM:
            if arm == "D0_frozen_task40_lfighter":
                instant = merged[D0_INSTANT].to_numpy(dtype=float)
                ema = merged[D0_EMA].to_numpy(dtype=float)
                flags = (instant > context.d0_instant_threshold) | (
                    ema > context.d0_ema_threshold
                )
            elif arm == "D1_update_norm_deviation":
                flags = merged["d1_log_robust_z"].to_numpy(dtype=float) > context.d1_threshold
            elif arm == "D2_update_direction_deviation":
                flags = merged["d2_deviation"].to_numpy(dtype=float) > context.d2_threshold
            elif arm == "D3_equal_weight_fusion":
                # OR-fusion: flag if ANY component detector independently
                # exceeds its OWN already-calibrated threshold. This
                # replaces an earlier averaged-rank fusion that diluted a
                # strong individual signal whenever another component was
                # weak in that specific condition -- the whole point of a
                # fusion detector is "any one catching it is enough,"
                # which averaging does not preserve. Note the expected
                # trade-off: OR-fusion's benign FPR will be at least as
                # high as its worst single component's FPR in any given
                # condition (it inherits D2's scaling-attack collapse,
                # for instance) in exchange for never underperforming its
                # best component's recall.
                d0_instant_val = merged[D0_INSTANT].to_numpy(dtype=float)
                d0_ema_val = merged[D0_EMA].to_numpy(dtype=float)
                d0_flagged = (d0_instant_val > context.d0_instant_threshold) | (
                    d0_ema_val > context.d0_ema_threshold
                )
                d1_flagged = (
                    merged["d1_log_robust_z"].to_numpy(dtype=float)
                    > context.d1_threshold
                )
                d2_flagged = (
                    merged["d2_deviation"].to_numpy(dtype=float)
                    > context.d2_threshold
                )
                merged["d0_component_flagged"] = d0_flagged
                merged["d1_component_flagged"] = d1_flagged
                merged["d2_component_flagged"] = d2_flagged
                flags = d0_flagged | d1_flagged | d2_flagged

            updates = [floating_update(state, reference_state) for state in local_states]
            trusted_ids = [c for c in range(NUM_CLIENTS) if not flags[c]]
            detector_collapsed = len(trusted_ids) == 0
            if detector_collapsed:
                # Every client flagged -- no basis exists to compute a
                # trusted center. Record this honestly rather than crash
                # or fabricate a center from nothing: freeze the model
                # this round (skip aggregation) and continue, so the
                # grid captures the full collapse trajectory instead of
                # losing all remaining conditions.
                aggregated_state = {
                    key: value.clone() for key, value in reference_state.items()
                }
            else:
                trusted_updates = [updates[c] for c in trusted_ids]
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
        else:
            aggregated_state = weighted_average_states(
                replacement_states, sample_counts, reference_state
            )

        merged["monitoring_round"] = monitoring_round
        merged["global_round"] = global_round
        merged["flagged"] = flags
        merged["detector_collapsed"] = detector_collapsed if arm != PLAIN_ARM else False
        score_rows.extend(merged.to_dict(orient="records"))

        model.load_state_dict(aggregated_state)
        val_metrics, val_pair = evaluate_validation(
            model,
            context.X_val,
            context.y_val,
            context.source_id,
            context.target_id,
            args.evaluation_batch_size,
        )

        labels = np.asarray(actual_malicious, dtype=bool)
        benign = ~labels
        malicious_recall = (
            float(np.mean(flags[labels])) if condition.is_attack and arm != PLAIN_ARM else float("nan")
        )
        benign_fpr = float(np.mean(flags[benign]))

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
            f"recall={malicious_recall}, FPR={benign_fpr:.3f}, "
            f"replaced={replaced_clients}, seconds={row['round_seconds']:.1f}"
        )

    round_table = pd.DataFrame(round_rows)
    score_table = pd.DataFrame(score_rows)
    round_table.to_csv(tables_dir / "branch_round_metrics.csv", index=False)
    score_table.to_csv(tables_dir / "branch_client_scores.csv", index=False)

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
        "attack_type": condition.attack_type,
        "attack_strength": condition.strength,
        "arm": arm,
        "model_seed": MODEL_SEED,
        "total_seconds": float(time.time() - branch_started),
        "mean_macro_f1": float(round_table["val_macro_f1"].mean()),
        "maximum_benign_fpr": float(round_table["benign_false_positive_rate"].max()),
        "mean_malicious_recall": (
            None if not condition.is_attack or arm == PLAIN_ARM
            else float(round_table["malicious_recall"].mean())
        ),
    }
    (branch_dir / "branch_complete.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
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
        print("D0 EMA threshold:", context.d0_ema_threshold)
        print("D0 instant threshold:", context.d0_instant_threshold)
        print("D1 threshold:", context.d1_threshold)
        print("D2 threshold:", context.d2_threshold)
        print("Conditions:", [c.condition_id for c in conditions])
        return 0

    for condition in conditions:
        for arm in ALL_ARMS:
            run_branch(condition=condition, arm=arm, context=context, output_root=output_root, args=args)

    write_output_manifest(output_root)
    print("Task 42 C2 screen complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

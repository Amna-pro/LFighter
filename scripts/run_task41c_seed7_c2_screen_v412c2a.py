#!/usr/bin/env python3
"""Task 41C.2 seed-7 preregistered development screen.

Scope
-----
- Seed 7 only.
- Clean continuation reference plus all six preregistered A1 dirty-label
  conditions: two frozen trigger families and poison fractions 0.005, 0.01,
  and 0.02.
- Exact paired plain FedAvg and four frozen detector/reconstruction candidates:
  D0, D1, D2, and D3.
- Four post-warmup monitoring rounds.
- Development arrays only: X_train, y_train, X_val, y_val.
- No reserved natural or diagnostic test arrays.
- No A2, A3, or A4 execution.
- No threshold, profile, weight, seed, client, trigger, or fraction tuning.

Selection follows the preregistered lexicographic rule:
1. Reject if maximum benign FPR exceeds 0.05.
2. Reject if mean clean macro-F1 loss exceeds 0.01 or any round loss exceeds
   0.02.
3. Maximize mean malicious-client recall across all A1 seed-7 condition-rounds.
4. Break ties by mean paired relative macro triggered-ASR reduction.
5. Break remaining ties by lower runtime overhead.
6. If no candidate survives, record failure and do not create a new candidate.

The runner is branch-resumable. A branch is skipped only when its completion
metadata and required tables exist and match the frozen condition identifiers.
A partial branch is archived and rerun from the common frozen warmup point.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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
from run_targeted_label_flip_v292 import (
    load_fixed_partitions,
    read_clean_partition_hash,
    save_poisoned_indices,
)
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
from run_task41b_backdoor_smoke_v411b1 import (
    BACKDOOR_ATTACK_TYPE,
    build_triggered_validation,
    evaluate_triggered_asr,
    load_trigger_spec,
    prepare_backdoor_attack,
    save_poisoned_labels,
    verify_trigger_application,
)
from run_task41c_clean_detector_calibration_v412c1c3 import (
    cosine,
    load_trigger_panel,
    parameter_names,
    positive_z,
    response_vector,
    trigger_gradient_panel,
    update_vector,
)

EXPERIMENT_VERSION = "4.12C.2a"
MODEL_SEED = 7
ATTACK_SEED = 42
NUM_CLIENTS = 20
CONTINUATION_ROUNDS = 4
EMA_DECAY = 0.65
FPR_CEILING = 0.05
MEAN_CLEAN_F1_LOSS_CEILING = 0.01
MAX_ROUND_CLEAN_F1_LOSS_CEILING = 0.02
ALLOWED_ARRAYS = ("X_train", "y_train", "X_val", "y_val")
RESERVED_ARRAYS = (
    "X_test_natural",
    "y_test_natural",
    "X_test_diagnostic",
    "y_test_diagnostic",
)
MALICIOUS_CLIENTS = (1, 7, 8, 10, 14, 15, 17, 18)
TRIGGER_SLOTS = {
    "flow_iat_exact": "multi_flow_iat_bundle__exact_template__donor_00",
    "active_idle_exact": "multi_active_idle_bundle__exact_template__donor_18",
}
POISON_FRACTIONS = (0.005, 0.01, 0.02)
PLAIN_ARM = "plain_fedavg"
CANDIDATES = (
    "D0_frozen_task40_lfighter",
    "D1_trigger_response_shift",
    "D2_trigger_gradient_alignment",
    "D3_equal_rank_fusion",
)
ALL_ARMS = (PLAIN_ARM, *CANDIDATES)
REQUIRED_C1_VERSION = "4.12C.1c3"
REQUIRED_C1_RESULTS_TAG = "task41c-c1c3-results-v4.12c1c3"
REQUIRED_C0_TAG = "task41c-c0-preregistered-v4.12c0"


@dataclass(frozen=True)
class Condition:
    condition_id: str
    mode: str
    trigger_slot: str
    trigger_candidate_id: str
    poison_fraction: float

    @property
    def is_attack(self) -> bool:
        return self.mode == "A1_dirty_label_low_rate"


@dataclass
class FrozenContext:
    arrays: Dict[str, np.ndarray]
    client_indices: List[np.ndarray]
    partition_hash: str
    X_probe: np.ndarray
    y_probe: np.ndarray
    probe_hash: str
    warmup_metadata: Dict[str, Any]
    warmup_checkpoint_path: Path
    warmup_checkpoint: Dict[str, Any]
    task40_profiles: Dict[int, np.ndarray]
    task40_feature_calibration: pd.DataFrame
    reconstruction_profiles: Dict[int, Dict[str, torch.Tensor]]
    selected_reconstruction_policy: str
    warmup_update_scale: float
    scale_lower: float
    scale_upper: float
    norm_clip_multiplier: float
    trigger_panel: List[Dict[str, Any]]
    trigger_panel_sha256: str
    trigger_specs: Dict[str, Dict[str, Any]]
    trigger_spec_file_sha256: str
    thresholds: pd.DataFrame
    detector_profiles: Dict[str, np.ndarray]
    detector_profile_sha256: str
    detector_threshold_sha256: str
    parameter_names: List[str]
    source_id: int
    target_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Task 41C C2 seed-7 candidate screen."
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--partition-file", type=Path, required=True)
    parser.add_argument("--clean-seed-dir", type=Path, required=True)
    parser.add_argument("--warmup-dir", type=Path, required=True)
    parser.add_argument(
        "--reconstruction-calibration-dir", type=Path, required=True
    )
    parser.add_argument("--trigger-spec-file", type=Path, required=True)
    parser.add_argument("--c1-calibration-dir", type=Path, required=True)
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
    manifest_path = (
        output_root
        / "summary"
        / "tables"
        / "task41c2_summary_manifest_sha256.csv"
    )
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


def sha256_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def stable_text_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_frozen_git_inputs() -> Dict[str, str]:
    c0_commit = git_output("rev-list", "-n", "1", REQUIRED_C0_TAG)
    c1_commit = git_output("rev-list", "-n", "1", REQUIRED_C1_RESULTS_TAG)
    protocol_diff = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            REQUIRED_C0_TAG,
            "--",
            "configs/task41c_preregistration_v412c0.json",
        ],
        cwd=str(PROJECT_ROOT),
    )
    if protocol_diff.returncode != 0:
        raise RuntimeError(
            "Frozen Task 41C preregistration differs from the C0 tag."
        )
    return {
        "c0_tag": REQUIRED_C0_TAG,
        "c0_commit": c0_commit,
        "c1_results_tag": REQUIRED_C1_RESULTS_TAG,
        "c1_results_commit": c1_commit,
        "current_head": git_output("rev-parse", "HEAD"),
    }


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


def empirical_cdf_rank(sorted_reference: np.ndarray, value: float) -> float:
    """Frozen empirical midrank matching C1 average-rank calibration.

    Existing tied values receive the same average percentile used by
    pandas.rank(method="average", pct=True). Values falling between observed
    clean scores receive the midpoint of the adjacent empirical interval.
    """
    reference = np.asarray(sorted_reference, dtype=np.float64)
    if reference.ndim != 1 or len(reference) == 0:
        raise ValueError("Empirical reference must be a nonempty vector.")
    left = int(np.searchsorted(reference, float(value), side="left"))
    right = int(np.searchsorted(reference, float(value), side="right"))
    n = float(len(reference))
    if right > left:
        return float((left + 1 + right) / (2.0 * n))
    return float((left + 0.5) / n)


def threshold_lookup(table: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    required = {
        "candidate",
        "instant_threshold",
        "ema_threshold",
        "clean_policy_fpr",
        "attack_labels_used",
        "reserved_test_accessed",
    }
    missing = required.difference(table.columns)
    if missing:
        raise RuntimeError(
            f"C1 threshold table lacks columns: {sorted(missing)}"
        )
    mapping: Dict[str, Dict[str, float]] = {}
    for row in table.to_dict(orient="records"):
        candidate = str(row["candidate"])
        mapping[candidate] = {
            "instant_threshold": float(row["instant_threshold"]),
            "ema_threshold": float(row["ema_threshold"]),
            "clean_policy_fpr": float(row["clean_policy_fpr"]),
        }
        if bool(row["attack_labels_used"]):
            raise RuntimeError("C1 threshold table reports attack-label use.")
        if bool(row["reserved_test_accessed"]):
            raise RuntimeError("C1 threshold table reports test access.")
    if set(mapping) != set(CANDIDATES):
        raise RuntimeError(
            f"Frozen candidate threshold set mismatch: {sorted(mapping)}"
        )
    return mapping


def load_frozen_context(args: argparse.Namespace) -> Tuple[FrozenContext, Dict[str, str]]:
    git_info = verify_frozen_git_inputs()

    protocol_path = (
        PROJECT_ROOT / "configs" / "task41c_preregistration_v412c0.json"
    )
    c1_decision_path = (
        args.c1_calibration_dir.expanduser().resolve()
        / "summary"
        / "task41c1c_clean_calibration_decision.json"
    )
    seed7_dir = (
        args.c1_calibration_dir.expanduser().resolve() / "seed_7"
    )
    threshold_path = (
        seed7_dir
        / "calibration"
        / "task41c1c_candidate_thresholds.csv"
    )
    detector_profile_path = (
        seed7_dir
        / "calibration"
        / "task41c1c_backdoor_detector_profiles.npz"
    )
    c1_metadata_path = (
        seed7_dir / "task41c1c_clean_calibration_metadata.json"
    )
    for path in (
        protocol_path,
        c1_decision_path,
        threshold_path,
        detector_profile_path,
        c1_metadata_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    c1_decision = json.loads(
        c1_decision_path.read_text(encoding="utf-8")
    )
    c1_metadata = json.loads(
        c1_metadata_path.read_text(encoding="utf-8")
    )

    if protocol.get("experiment_version") != "4.12C.0":
        raise RuntimeError("Task 41C protocol version mismatch.")
    if protocol.get("frozen_seeds") != [7, 99, 123, 2026]:
        raise RuntimeError("Task 41C frozen seed panel mismatch.")
    if protocol["frozen_clients"]["malicious_clients"] != list(
        MALICIOUS_CLIENTS
    ):
        raise RuntimeError("Task 41C malicious coalition mismatch.")
    if c1_decision.get("experiment_version") != REQUIRED_C1_VERSION:
        raise RuntimeError("C1 calibration version mismatch.")
    if not bool(c1_decision.get("c2_seed7_screen_allowed", False)):
        raise RuntimeError("C1 did not authorize the seed-7 C2 screen.")
    if not bool(
        c1_decision.get("replay_equivalence_passed_all_seeds", False)
    ):
        raise RuntimeError("C1 replay-equivalence gate is false.")
    if bool(c1_decision.get("attack_execution_performed", True)):
        raise RuntimeError("C1 decision reports attack execution.")
    if bool(c1_decision.get("reserved_test_accessed", True)):
        raise RuntimeError("C1 decision reports reserved-test access.")
    if c1_metadata.get("seed") != MODEL_SEED:
        raise RuntimeError("C1 seed-7 metadata mismatch.")
    if c1_metadata.get("d3_weights") != [1.0, 1.0, 1.0]:
        raise RuntimeError("Frozen D3 weights mismatch.")
    if bool(c1_metadata.get("weight_tuning_used", True)):
        raise RuntimeError("C1 metadata reports D3 weight tuning.")
    if bool(c1_metadata.get("attack_specific_thresholds", True)):
        raise RuntimeError("C1 metadata reports attack-specific thresholds.")
    if bool(c1_metadata.get("trigger_specific_thresholds", True)):
        raise RuntimeError("C1 metadata reports trigger-specific thresholds.")
    if file_sha256(detector_profile_path) != c1_metadata["profile_sha256"]:
        raise RuntimeError("C1 detector profile hash mismatch.")
    if file_sha256(threshold_path) != c1_metadata["threshold_sha256"]:
        raise RuntimeError("C1 detector threshold hash mismatch.")

    thresholds = pd.read_csv(threshold_path)
    threshold_lookup(thresholds)

    detector_profiles: Dict[str, np.ndarray] = {}
    with np.load(detector_profile_path, allow_pickle=False) as archive:
        required_keys = {
            "d1_client_profiles",
            "d1_feature_scales",
            "d2_client_profiles",
            "d2_feature_scales",
            "d0_empirical_reference",
            "d1_empirical_reference",
            "d2_empirical_reference",
        }
        missing = required_keys.difference(archive.files)
        if missing:
            raise RuntimeError(
                f"C1 profile NPZ lacks arrays: {sorted(missing)}"
            )
        for key in required_keys:
            detector_profiles[key] = np.asarray(archive[key])
    if detector_profiles["d1_client_profiles"].shape[0] != NUM_CLIENTS:
        raise RuntimeError("D1 profile client count mismatch.")
    if detector_profiles["d2_client_profiles"].shape[0] != NUM_CLIENTS:
        raise RuntimeError("D2 profile client count mismatch.")

    arrays = load_development_arrays(args.data_file)
    X_train = arrays["X_train"].astype(np.float32, copy=False)
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    X_val = arrays["X_val"].astype(np.float32, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)

    client_indices, partition_hash = load_fixed_partitions(
        args.partition_file.expanduser().resolve(),
        expected_clients=NUM_CLIENTS,
        train_rows=len(y_train),
    )
    clean_hash = read_clean_partition_hash(
        args.clean_seed_dir.expanduser().resolve()
    )
    if partition_hash != clean_hash:
        raise RuntimeError("Clean seed partition hash mismatch.")

    warmup_dir = args.warmup_dir.expanduser().resolve()
    warmup_metadata_path = warmup_dir / "true_warmup_v310_metadata.json"
    warmup_checkpoint_path = (
        warmup_dir
        / "checkpoints"
        / "common_round4_warmup_model.pt"
    )
    task40_profiles_path = (
        warmup_dir / "calibration" / "trusted_client_profiles.npz"
    )
    task40_feature_path = (
        warmup_dir / "calibration" / "feature_calibration.csv"
    )
    for path in (
        warmup_metadata_path,
        warmup_checkpoint_path,
        task40_profiles_path,
        task40_feature_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    warmup_metadata = json.loads(
        warmup_metadata_path.read_text(encoding="utf-8")
    )
    if int(warmup_metadata["model_seed"]) != MODEL_SEED:
        raise RuntimeError("Warmup seed mismatch.")
    if warmup_metadata["partition_hash_sha256"] != partition_hash:
        raise RuntimeError("Warmup partition hash mismatch.")
    if not bool(
        warmup_metadata.get("monitoring_ema_reset_after_warmup", False)
    ):
        raise RuntimeError("Warmup EMA-reset contract is missing.")
    if file_sha256(task40_profiles_path) != warmup_metadata["profile_sha256"]:
        raise RuntimeError("Task 40 trusted-profile hash mismatch.")
    if (
        file_sha256(task40_feature_path)
        != warmup_metadata["feature_calibration_sha256"]
    ):
        raise RuntimeError("Task 40 feature-calibration hash mismatch.")

    warmup_checkpoint = torch.load(
        warmup_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if warmup_checkpoint.get("partition_hash") != partition_hash:
        raise RuntimeError("Warmup checkpoint partition mismatch.")

    probe_indices = balanced_probe_indices(
        y_val, args.probe_per_class, args.probe_seed
    )
    probe_hash = sha256_array(probe_indices)
    if probe_hash != warmup_metadata["probe_hash_sha256"]:
        raise RuntimeError("Warmup probe hash mismatch.")
    X_probe = X_val[probe_indices]
    y_probe = y_val[probe_indices]

    task40_profiles = load_profiles_npz(
        task40_profiles_path, NUM_CLIENTS
    )
    task40_feature_calibration = pd.read_csv(task40_feature_path)

    trigger_panel, trigger_panel_sha256 = load_trigger_panel(
        args.trigger_spec_file
    )
    trigger_specs: Dict[str, Dict[str, Any]] = {}
    trigger_spec_file_sha256 = file_sha256(
        args.trigger_spec_file.expanduser().resolve()
    )
    for slot, candidate_id in TRIGGER_SLOTS.items():
        spec, spec_hash = load_trigger_spec(
            args.trigger_spec_file, candidate_id
        )
        if spec_hash != trigger_spec_file_sha256:
            raise RuntimeError("Trigger specification file hash changed.")
        trigger_specs[slot] = spec
    if c1_metadata["trigger_spec_sha256"] != trigger_spec_file_sha256:
        raise RuntimeError("C1 trigger-specification hash mismatch.")
    if c1_metadata["partition_hash_sha256"] != partition_hash:
        raise RuntimeError("C1 partition hash mismatch.")
    if c1_metadata["probe_hash_sha256"] != probe_hash:
        raise RuntimeError("C1 probe hash mismatch.")
    if (
        c1_metadata["common_warmup_checkpoint_sha256"]
        != checkpoint_sha256(warmup_checkpoint_path)
    ):
        raise RuntimeError("C1 warmup-checkpoint hash mismatch.")

    reconstruction_dir = (
        args.reconstruction_calibration_dir.expanduser().resolve()
    )
    reconstruction_checkpoint_path = (
        reconstruction_dir
        / "calibration"
        / "trusted_update_reconstruction_profiles.pt"
    )
    reconstruction_summary_path = (
        reconstruction_dir
        / "calibration"
        / "reconstruction_calibration_summary.csv"
    )
    for path in (
        reconstruction_checkpoint_path,
        reconstruction_summary_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    reconstruction_bundle = torch.load(
        reconstruction_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if int(reconstruction_bundle["model_seed"]) != MODEL_SEED:
        raise RuntimeError("Reconstruction seed mismatch.")
    if reconstruction_bundle["partition_hash"] != partition_hash:
        raise RuntimeError("Reconstruction partition mismatch.")
    if (
        reconstruction_bundle["common_warmup_checkpoint_sha256"]
        != checkpoint_sha256(warmup_checkpoint_path)
    ):
        raise RuntimeError("Reconstruction warmup-checkpoint mismatch.")
    selected_reconstruction_policy = str(
        reconstruction_bundle["selected_policy"]
    )
    if selected_reconstruction_policy != "center_plus_residual":
        raise RuntimeError(
            "Task 41C requires center_plus_residual reconstruction."
        )
    reconstruction_profiles = {
        int(client_id): {
            key: value.detach().cpu().to(torch.float32)
            for key, value in profile.items()
        }
        for client_id, profile in reconstruction_bundle[
            "client_residual_profiles"
        ].items()
    }
    if set(reconstruction_profiles) != set(range(NUM_CLIENTS)):
        raise RuntimeError("Reconstruction client-profile set mismatch.")

    probe_model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    names = parameter_names(probe_model)
    del probe_model

    context = FrozenContext(
        arrays=arrays,
        client_indices=client_indices,
        partition_hash=partition_hash,
        X_probe=X_probe,
        y_probe=y_probe,
        probe_hash=probe_hash,
        warmup_metadata=warmup_metadata,
        warmup_checkpoint_path=warmup_checkpoint_path,
        warmup_checkpoint=warmup_checkpoint,
        task40_profiles=task40_profiles,
        task40_feature_calibration=task40_feature_calibration,
        reconstruction_profiles=reconstruction_profiles,
        selected_reconstruction_policy=selected_reconstruction_policy,
        warmup_update_scale=float(
            reconstruction_bundle["warmup_update_scale"]
        ),
        scale_lower=float(reconstruction_bundle["scale_lower"]),
        scale_upper=float(reconstruction_bundle["scale_upper"]),
        norm_clip_multiplier=float(
            reconstruction_bundle["norm_clip_multiplier"]
        ),
        trigger_panel=trigger_panel,
        trigger_panel_sha256=trigger_panel_sha256,
        trigger_specs=trigger_specs,
        trigger_spec_file_sha256=trigger_spec_file_sha256,
        thresholds=thresholds,
        detector_profiles=detector_profiles,
        detector_profile_sha256=file_sha256(detector_profile_path),
        detector_threshold_sha256=file_sha256(threshold_path),
        parameter_names=names,
        source_id=CLASS_NAMES.index("DDoS"),
        target_id=CLASS_NAMES.index("Benign"),
    )
    return context, git_info


def frozen_conditions() -> List[Condition]:
    conditions = [
        Condition(
            condition_id="clean_reference",
            mode="clean",
            trigger_slot="flow_iat_exact",
            trigger_candidate_id=TRIGGER_SLOTS["flow_iat_exact"],
            poison_fraction=0.0,
        )
    ]
    for trigger_slot, candidate_id in TRIGGER_SLOTS.items():
        for fraction in POISON_FRACTIONS:
            fraction_token = f"{int(round(fraction * 10000)):04d}"
            conditions.append(
                Condition(
                    condition_id=(
                        f"A1_{trigger_slot}_poison_{fraction_token}"
                    ),
                    mode="A1_dirty_label_low_rate",
                    trigger_slot=trigger_slot,
                    trigger_candidate_id=candidate_id,
                    poison_fraction=float(fraction),
                )
            )
    return conditions


def prepare_condition_poison_plan(
    condition: Condition,
    context: FrozenContext,
    output_root: Path,
) -> Tuple[
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
    pd.DataFrame,
    pd.DataFrame,
    str,
]:
    plan_dir = output_root / "attack_plans" / condition.condition_id
    plan_dir.mkdir(parents=True, exist_ok=True)

    if not condition.is_attack:
        positions = {
            client_id: np.empty(0, dtype=np.int64)
            for client_id in range(NUM_CLIENTS)
        }
        labels = {
            client_id: np.empty(0, dtype=np.int64)
            for client_id in range(NUM_CLIENTS)
        }
        manifest = pd.DataFrame(
            [
                {
                    "client_id": client_id,
                    "is_malicious": False,
                    "client_rows": len(context.client_indices[client_id]),
                    "eligible_rows": 0,
                    "poisoned_rows": 0,
                    "changed_rows_verified": 0,
                    "client_poison_rate_over_eligible": 0.0,
                    "client_poison_rate_over_all_rows": 0.0,
                    "attack_type": "none",
                    "trigger_candidate_id": condition.trigger_candidate_id,
                    "label_policy": "none",
                    "deployment_policy": "none",
                    "target_class_id": context.target_id,
                    "target_class_name": CLASS_NAMES[context.target_id],
                }
                for client_id in range(NUM_CLIENTS)
            ]
        )
        source_manifest = pd.DataFrame()
        poison_hash = stable_text_hash(
            ["clean", condition.condition_id, context.partition_hash]
        )
    else:
        (
            positions,
            labels,
            manifest,
            source_manifest,
            poison_hash,
        ) = prepare_backdoor_attack(
            client_indices=context.client_indices,
            y_train=context.arrays["y_train"].astype(
                np.int64, copy=False
            ),
            malicious_clients=list(MALICIOUS_CLIENTS),
            poison_fraction=condition.poison_fraction,
            attack_seed=ATTACK_SEED,
            candidate_id=condition.trigger_candidate_id,
            label_policy="dirty_label_all_nonbenign_to_benign",
            deployment_policy="centralized_full_trigger",
            target_id=context.target_id,
            trigger_spec_sha256=context.trigger_spec_file_sha256,
        )
        trigger_spec = context.trigger_specs[condition.trigger_slot]
        verified = verify_trigger_application(
            X_train=context.arrays["X_train"].astype(
                np.float32, copy=False
            ),
            client_indices=context.client_indices,
            poisoned_positions=positions,
            feature_indices=np.asarray(
                trigger_spec["feature_indices"], dtype=np.int64
            ),
            template=np.asarray(
                trigger_spec["template_standardized"],
                dtype=np.float32,
            ),
        )
        expected = {
            client_id: int(len(values))
            for client_id, values in positions.items()
        }
        if verified != expected:
            raise RuntimeError(
                f"Trigger application verification failed: "
                f"{condition.condition_id}"
            )

    manifest.to_csv(
        plan_dir / "malicious_client_poison_manifest.csv",
        index=False,
    )
    source_manifest.to_csv(
        plan_dir / "poisoned_source_manifest.csv",
        index=False,
    )
    save_poisoned_indices(
        context.client_indices,
        positions,
        plan_dir / "poisoned_indices.npz",
    )
    save_poisoned_labels(
        labels,
        plan_dir / "poisoned_labels.npz",
    )
    plan_metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "condition_id": condition.condition_id,
        "mode": condition.mode,
        "model_seed": MODEL_SEED,
        "attack_seed": ATTACK_SEED,
        "trigger_slot": condition.trigger_slot,
        "trigger_candidate_id": condition.trigger_candidate_id,
        "poison_fraction": condition.poison_fraction,
        "malicious_clients": (
            list(MALICIOUS_CLIENTS) if condition.is_attack else []
        ),
        "partition_hash_sha256": context.partition_hash,
        "trigger_spec_sha256": context.trigger_spec_file_sha256,
        "poison_plan_hash_sha256": poison_hash,
        "development_arrays_loaded": list(ALLOWED_ARRAYS),
        "reserved_arrays_loaded": [],
        "reserved_test_accessed": False,
    }
    (plan_dir / "attack_plan_metadata.json").write_text(
        json.dumps(plan_metadata, indent=2), encoding="utf-8"
    )
    return positions, labels, manifest, source_manifest, poison_hash


def branch_complete(
    branch_dir: Path,
    condition: Condition,
    arm: str,
    poison_hash: str,
) -> bool:
    metadata_path = branch_dir / "branch_complete.json"
    round_path = branch_dir / "tables" / "branch_round_metrics.csv"
    score_path = branch_dir / "tables" / "branch_client_scores.csv"
    if not all(path.exists() for path in (metadata_path, round_path, score_path)):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return bool(
        metadata.get("complete") is True
        and metadata.get("condition_id") == condition.condition_id
        and metadata.get("arm") == arm
        and metadata.get("poison_plan_hash_sha256") == poison_hash
        and metadata.get("reserved_test_accessed") is False
    )


def archive_partial_branch(branch_dir: Path) -> None:
    if not branch_dir.exists() or not any(branch_dir.iterdir()):
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archive = branch_dir.with_name(
        f"{branch_dir.name}_partial_{stamp}"
    )
    shutil.move(str(branch_dir), str(archive))


def d0_score_table(
    *,
    local_matrices: Sequence[np.ndarray],
    sample_counts: Sequence[int],
    context: FrozenContext,
    monitoring_round: int,
    actual_malicious: Sequence[bool],
    poisoned_rows: Sequence[int],
    ema_memory: Dict[int, float],
) -> Tuple[pd.DataFrame, Dict[int, float]]:
    matrix_stack = np.stack(local_matrices, axis=0)
    consensus = normalize_rows(np.median(matrix_stack, axis=0))
    raw_weights = np.asarray(
        sample_counts, dtype=np.float64
    ) / float(np.sum(sample_counts))
    raw_rows: List[Dict[str, Any]] = []
    for client_id, local_matrix_raw in enumerate(local_matrices):
        local_matrix = normalize_rows(local_matrix_raw)
        raw_rows.append(
            {
                "monitoring_round": monitoring_round,
                "client_id": client_id,
                "actual_malicious": bool(
                    actual_malicious[client_id]
                ),
                "client_samples": int(sample_counts[client_id]),
                "aggregation_weight": float(raw_weights[client_id]),
                "poisoned_rows": int(poisoned_rows[client_id]),
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
        pd.DataFrame(raw_rows),
        context.task40_feature_calibration,
    )
    scored, updated_ema = add_candidate_and_ema(
        scored,
        EMA_DECAY,
        initial_ema=ema_memory,
    )
    return (
        scored.sort_values("client_id").reset_index(drop=True),
        updated_ema,
    )


def candidate_scores(
    *,
    arm: str,
    reference_state: Mapping[str, torch.Tensor],
    local_states: Sequence[Mapping[str, torch.Tensor]],
    d0_table: pd.DataFrame,
    context: FrozenContext,
    d1_ema: Dict[int, float],
    d2_ema: Dict[int, float],
    d3_ema: Dict[int, float],
    batch_size: int,
) -> Tuple[pd.DataFrame, Dict[int, float], Dict[int, float], Dict[int, float]]:
    profiles = context.detector_profiles
    rows: List[Dict[str, Any]] = []

    needs_d1 = arm in (
        "D1_trigger_response_shift",
        "D3_equal_rank_fusion",
    )
    needs_d2 = arm in (
        "D2_trigger_gradient_alignment",
        "D3_equal_rank_fusion",
    )

    reference_response: np.ndarray | None = None
    directions: List[torch.Tensor] | None = None
    if needs_d1:
        reference_model = build_model(
            "resmlp",
            context.arrays["X_train"].shape[1],
            NUM_CLASSES,
        )
        reference_model.load_state_dict(reference_state)
        reference_response = response_vector(
            reference_model,
            context.X_probe,
            context.y_probe,
            context.trigger_panel,
            context.target_id,
            batch_size,
        )
        del reference_model
    if needs_d2:
        directions = trigger_gradient_panel(
            reference_state,
            context.arrays["X_train"].shape[1],
            context.X_probe,
            context.y_probe,
            context.trigger_panel,
            context.target_id,
        )

    for client_id, local_state in enumerate(local_states):
        d0_instant = float(
            d0_table.loc[
                d0_table["client_id"] == client_id,
                D0_INSTANT,
            ].iloc[0]
        )
        d0_ema_value = float(
            d0_table.loc[
                d0_table["client_id"] == client_id,
                D0_EMA,
            ].iloc[0]
        )

        d1_instant = float("nan")
        d2_instant = float("nan")
        d3_instant = float("nan")

        if needs_d1:
            local_model = build_model(
                "resmlp",
                context.arrays["X_train"].shape[1],
                NUM_CLASSES,
            )
            local_model.load_state_dict(local_state)
            local_response = response_vector(
                local_model,
                context.X_probe,
                context.y_probe,
                context.trigger_panel,
                context.target_id,
                batch_size,
            )
            del local_model
            d1_vector = local_response - reference_response
            d1_z = positive_z(
                d1_vector,
                profiles["d1_client_profiles"][client_id],
                profiles["d1_feature_scales"],
            )
            d1_instant = float(np.max(d1_z))
            d1_ema[client_id] = (
                EMA_DECAY * float(d1_ema.get(client_id, 0.0))
                + (1.0 - EMA_DECAY) * d1_instant
            )

        if needs_d2:
            update = update_vector(
                local_state,
                reference_state,
                context.parameter_names,
            )
            d2_vector = np.asarray(
                [cosine(update, direction) for direction in directions],
                dtype=np.float64,
            )
            d2_z = positive_z(
                d2_vector,
                profiles["d2_client_profiles"][client_id],
                profiles["d2_feature_scales"],
            )
            d2_instant = float(np.max(d2_z))
            d2_ema[client_id] = (
                EMA_DECAY * float(d2_ema.get(client_id, 0.0))
                + (1.0 - EMA_DECAY) * d2_instant
            )

        if arm == "D3_equal_rank_fusion":
            d0_rank = empirical_cdf_rank(
                profiles["d0_empirical_reference"], d0_instant
            )
            d1_rank = empirical_cdf_rank(
                profiles["d1_empirical_reference"], d1_instant
            )
            d2_rank = empirical_cdf_rank(
                profiles["d2_empirical_reference"], d2_instant
            )
            d3_instant = float(
                (d0_rank + d1_rank + d2_rank) / 3.0
            )
            d3_ema[client_id] = (
                EMA_DECAY * float(d3_ema.get(client_id, 0.0))
                + (1.0 - EMA_DECAY) * d3_instant
            )
        else:
            d0_rank = float("nan")
            d1_rank = float("nan")
            d2_rank = float("nan")

        if arm == "D0_frozen_task40_lfighter":
            instant = d0_instant
            ema = d0_ema_value
        elif arm == "D1_trigger_response_shift":
            instant = d1_instant
            ema = float(d1_ema[client_id])
        elif arm == "D2_trigger_gradient_alignment":
            instant = d2_instant
            ema = float(d2_ema[client_id])
        elif arm == "D3_equal_rank_fusion":
            instant = d3_instant
            ema = float(d3_ema[client_id])
        else:
            raise ValueError(f"Unknown detector arm: {arm}")

        rows.append(
            {
                "client_id": client_id,
                "d0_instant": d0_instant,
                "d0_ema": d0_ema_value,
                "d1_instant": d1_instant,
                "d1_ema": (
                    float(d1_ema.get(client_id, float("nan")))
                ),
                "d2_instant": d2_instant,
                "d2_ema": (
                    float(d2_ema.get(client_id, float("nan")))
                ),
                "d0_rank": d0_rank,
                "d1_rank": d1_rank,
                "d2_rank": d2_rank,
                "d3_instant": d3_instant,
                "d3_ema": (
                    float(d3_ema.get(client_id, float("nan")))
                ),
                "candidate_instant": instant,
                "candidate_ema": ema,
            }
        )
    return pd.DataFrame(rows), d1_ema, d2_ema, d3_ema


def exact_pair_hash(
    *,
    condition: Condition,
    monitoring_round: int,
    poison_hash: str,
    context: FrozenContext,
) -> str:
    return stable_text_hash(
        [
            EXPERIMENT_VERSION,
            condition.condition_id,
            condition.trigger_slot,
            condition.trigger_candidate_id,
            f"{condition.poison_fraction:.12f}",
            str(MODEL_SEED),
            str(ATTACK_SEED),
            context.partition_hash,
            checkpoint_sha256(context.warmup_checkpoint_path),
            context.trigger_spec_file_sha256,
            poison_hash,
            str(monitoring_round),
            str(4 + monitoring_round),
        ]
    )


def run_branch(
    *,
    condition: Condition,
    arm: str,
    context: FrozenContext,
    poisoned_positions: Dict[int, np.ndarray],
    poisoned_labels: Dict[int, np.ndarray],
    poison_hash: str,
    output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    branch_dir = (
        output_root
        / "branches"
        / condition.condition_id
        / arm
    )
    if branch_complete(branch_dir, condition, arm, poison_hash):
        print(
            f"SKIP COMPLETE: {condition.condition_id} | {arm}"
        )
        return json.loads(
            (branch_dir / "branch_complete.json").read_text(
                encoding="utf-8"
            )
        )
    if branch_dir.exists() and any(branch_dir.iterdir()):
        if args.resume:
            archive_partial_branch(branch_dir)
        else:
            raise FileExistsError(
                f"Partial branch exists without --resume: {branch_dir}"
            )

    tables_dir = branch_dir / "tables"
    figures_dir = branch_dir / "figures"
    checkpoints_dir = branch_dir / "checkpoints"
    for path in (tables_dir, figures_dir, checkpoints_dir):
        path.mkdir(parents=True, exist_ok=True)

    X_train = context.arrays["X_train"].astype(
        np.float32, copy=False
    )
    y_train = context.arrays["y_train"].astype(
        np.int64, copy=False
    )
    X_val = context.arrays["X_val"].astype(
        np.float32, copy=False
    )
    y_val = context.arrays["y_val"].astype(
        np.int64, copy=False
    )

    trigger_spec = context.trigger_specs[condition.trigger_slot]
    trigger_indices = np.asarray(
        trigger_spec["feature_indices"], dtype=np.int64
    )
    trigger_template = np.asarray(
        trigger_spec["template_standardized"], dtype=np.float32
    )
    X_triggered, y_triggered_source = build_triggered_validation(
        X_val,
        y_val,
        context.target_id,
        trigger_indices,
        trigger_template,
    )

    set_seed(MODEL_SEED)
    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    model.load_state_dict(
        context.warmup_checkpoint["model_state_dict"]
    )
    class_weights = sqrt_class_weights(
        y_train, args.max_class_weight
    )

    actual_malicious = [
        condition.is_attack and client_id in MALICIOUS_CLIENTS
        for client_id in range(NUM_CLIENTS)
    ]
    poisoned_counts = [
        int(len(poisoned_positions[client_id]))
        for client_id in range(NUM_CLIENTS)
    ]

    threshold_map = threshold_lookup(context.thresholds)
    candidate_threshold = (
        threshold_map.get(arm) if arm != PLAIN_ARM else None
    )

    d0_ema_memory: Dict[int, float] = {}
    d1_ema_memory: Dict[int, float] = {}
    d2_ema_memory: Dict[int, float] = {}
    d3_ema_memory: Dict[int, float] = {}

    round_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    local_rows: List[Dict[str, Any]] = []
    reconstruction_rows: List[Dict[str, Any]] = []
    asr_rows: List[Dict[str, Any]] = []
    branch_started = time.time()

    print()
    print(
        f"RUN: {condition.condition_id} | {arm} | "
        f"poison_hash={poison_hash[:12]}"
    )

    for monitoring_round in range(1, CONTINUATION_ROUNDS + 1):
        global_round = 4 + monitoring_round
        round_started = time.time()
        reference_state = copy.deepcopy(model.state_dict())
        local_states: List[Mapping[str, torch.Tensor]] = []
        sample_counts: List[int] = []
        local_matrices: List[np.ndarray] = []
        current_training: List[Dict[str, Any]] = []

        for client_id in range(NUM_CLIENTS):
            indices = context.client_indices[client_id]
            local_X = X_train[indices].astype(np.float32, copy=True)
            local_y = y_train[indices].astype(np.int64, copy=True)
            positions = poisoned_positions[client_id]
            if len(positions):
                replacements = poisoned_labels[client_id]
                if len(replacements) != len(positions):
                    raise RuntimeError(
                        f"Poison label mismatch for client {client_id}."
                    )
                local_X[np.ix_(positions, trigger_indices)] = (
                    trigger_template.reshape(1, -1)
                )
                local_y[positions] = replacements

            local_model = build_model(
                "resmlp", X_train.shape[1], NUM_CLASSES
            )
            local_model.load_state_dict(reference_state)
            state, metrics = train_local_model(
                model=local_model,
                X=local_X,
                y=local_y,
                class_weights=class_weights,
                local_epochs=1,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                seed=MODEL_SEED + global_round * 1000 + client_id,
            )
            probabilities = predict_probabilities(
                local_model,
                context.X_probe,
                args.evaluation_batch_size,
            )
            local_means = class_conditional_probability_means(
                probabilities, context.y_probe
            )
            local_states.append(state)
            sample_counts.append(int(len(indices)))
            local_matrices.append(local_means)
            row = {
                "condition_id": condition.condition_id,
                "arm": arm,
                "monitoring_round": monitoring_round,
                "global_round": global_round,
                "client_id": client_id,
                "actual_malicious": bool(
                    actual_malicious[client_id]
                ),
                "client_samples": int(len(indices)),
                "poisoned_rows": int(len(positions)),
                **metrics,
            }
            local_rows.append(row)
            current_training.append(row)
            del local_model

        next_round_torch_rng_state = torch.random.get_rng_state().clone()

        d0_table, d0_ema_memory = d0_score_table(
            local_matrices=local_matrices,
            sample_counts=sample_counts,
            context=context,
            monitoring_round=monitoring_round,
            actual_malicious=actual_malicious,
            poisoned_rows=poisoned_counts,
            ema_memory=d0_ema_memory,
        )

        replacement_states = list(local_states)
        flags = np.zeros(NUM_CLIENTS, dtype=bool)
        current_scores = pd.DataFrame(
            {
                "client_id": np.arange(NUM_CLIENTS),
                "candidate_instant": np.nan,
                "candidate_ema": np.nan,
            }
        )
        replaced_clients = 0

        if arm != PLAIN_ARM:
            (
                current_scores,
                d1_ema_memory,
                d2_ema_memory,
                d3_ema_memory,
            ) = candidate_scores(
                arm=arm,
                reference_state=reference_state,
                local_states=local_states,
                d0_table=d0_table,
                context=context,
                d1_ema=d1_ema_memory,
                d2_ema=d2_ema_memory,
                d3_ema=d3_ema_memory,
                batch_size=args.evaluation_batch_size,
            )
            instant_threshold = float(
                candidate_threshold["instant_threshold"]
            )
            ema_threshold = float(
                candidate_threshold["ema_threshold"]
            )
            flags = (
                (
                    current_scores["candidate_instant"].to_numpy(
                        dtype=np.float64
                    )
                    > instant_threshold
                )
                | (
                    current_scores["candidate_ema"].to_numpy(
                        dtype=np.float64
                    )
                    > ema_threshold
                )
            )
            updates = [
                floating_update(state, reference_state)
                for state in local_states
            ]
            trusted_ids = [
                client_id
                for client_id in range(NUM_CLIENTS)
                if not flags[client_id]
            ]
            if len(trusted_ids) < 8:
                raise RuntimeError(
                    f"{condition.condition_id} {arm}: only "
                    f"{len(trusted_ids)} trusted clients remain."
                )
            trusted_updates = [
                updates[client_id] for client_id in trusted_ids
            ]
            trusted_norms = [
                update_norm(update) for update in trusted_updates
            ]
            current_center = coordinate_median(trusted_updates)
            current_update_scale = float(np.median(trusted_norms))

            for client_id in range(NUM_CLIENTS):
                replaced = bool(flags[client_id])
                reconstruction_meta: Dict[str, Any] = {
                    "residual_scale": float("nan"),
                    "norm_clip_factor": float("nan"),
                    "norm_clip_upper": float("nan"),
                    "reconstructed_update_norm": float("nan"),
                }
                if replaced:
                    (
                        reconstructed_update,
                        reconstruction_meta,
                    ) = reconstruct_update(
                        current_center=current_center,
                        historical_residual=(
                            context.reconstruction_profiles[client_id]
                        ),
                        selected_policy=(
                            context.selected_reconstruction_policy
                        ),
                        current_update_scale=current_update_scale,
                        warmup_update_scale=(
                            context.warmup_update_scale
                        ),
                        trusted_norms=trusted_norms,
                        scale_lower=context.scale_lower,
                        scale_upper=context.scale_upper,
                        norm_clip_multiplier=(
                            context.norm_clip_multiplier
                        ),
                    )
                    replacement_states[client_id] = state_from_update(
                        reference_state, reconstructed_update
                    )
                    replaced_clients += 1
                reconstruction_rows.append(
                    {
                        "condition_id": condition.condition_id,
                        "arm": arm,
                        "monitoring_round": monitoring_round,
                        "global_round": global_round,
                        "client_id": client_id,
                        "actual_malicious": bool(
                            actual_malicious[client_id]
                        ),
                        "flagged": replaced,
                        "update_replaced": replaced,
                        "actual_update_norm": update_norm(
                            updates[client_id]
                        ),
                        **reconstruction_meta,
                    }
                )
        else:
            instant_threshold = float("nan")
            ema_threshold = float("nan")

        current_scores = current_scores.merge(
            d0_table[
                [
                    "client_id",
                    D0_INSTANT,
                    D0_EMA,
                    "aggregation_weight",
                ]
            ],
            on="client_id",
            how="left",
            suffixes=("", "_task40"),
        )
        current_scores["condition_id"] = condition.condition_id
        current_scores["arm"] = arm
        current_scores["monitoring_round"] = monitoring_round
        current_scores["global_round"] = global_round
        current_scores["actual_malicious"] = actual_malicious
        current_scores["poisoned_rows"] = poisoned_counts
        current_scores["instant_threshold"] = instant_threshold
        current_scores["ema_threshold"] = ema_threshold
        current_scores["flagged"] = flags
        score_rows.extend(
            current_scores.to_dict(orient="records")
        )

        model.load_state_dict(
            weighted_average_states(
                replacement_states,
                sample_counts,
                reference_state,
            )
        )
        val_metrics, val_pair = evaluate_validation(
            model,
            X_val,
            y_val,
            context.source_id,
            context.target_id,
            args.evaluation_batch_size,
        )
        triggered_summary, current_asr_rows = evaluate_triggered_asr(
            model=model,
            X_triggered=X_triggered,
            y_source=y_triggered_source,
            target_id=context.target_id,
            batch_size=args.evaluation_batch_size,
            monitoring_round=monitoring_round,
            global_round=global_round,
            mode=condition.mode,
            replacement_policy=arm,
            candidate_id=condition.trigger_candidate_id,
        )
        for row in current_asr_rows:
            row["condition_id"] = condition.condition_id
            row["arm"] = arm
            row["poison_fraction"] = condition.poison_fraction
            row["trigger_slot"] = condition.trigger_slot
        asr_rows.extend(current_asr_rows)

        labels = np.asarray(actual_malicious, dtype=bool)
        benign = ~labels
        if condition.is_attack:
            malicious_recall = float(np.mean(flags[labels]))
            detection_precision = float(
                np.sum(flags & labels) / max(int(np.sum(flags)), 1)
            )
        else:
            malicious_recall = float("nan")
            detection_precision = float("nan")
        benign_fpr = float(np.mean(flags[benign]))

        pair_hash = exact_pair_hash(
            condition=condition,
            monitoring_round=monitoring_round,
            poison_hash=poison_hash,
            context=context,
        )
        row = {
            "condition_id": condition.condition_id,
            "mode": condition.mode,
            "trigger_slot": condition.trigger_slot,
            "trigger_candidate_id": condition.trigger_candidate_id,
            "poison_fraction": condition.poison_fraction,
            "arm": arm,
            "monitoring_round": monitoring_round,
            "global_round": global_round,
            "poison_plan_hash_sha256": poison_hash,
            "exact_pair_hash_sha256": pair_hash,
            "participating_samples": int(sum(sample_counts)),
            "flagged_clients": int(flags.sum()),
            "replaced_clients": int(replaced_clients),
            "malicious_recall": malicious_recall,
            "benign_false_positive_rate": benign_fpr,
            "detection_precision": detection_precision,
            "mean_local_train_loss": float(
                np.mean(
                    [
                        value["local_train_loss"]
                        for value in current_training
                    ]
                )
            ),
            "mean_local_train_accuracy": float(
                np.mean(
                    [
                        value["local_train_accuracy"]
                        for value in current_training
                    ]
                )
            ),
            "round_seconds": float(time.time() - round_started),
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"val_{key}": value for key, value in val_pair.items()},
            **triggered_summary,
        }
        round_rows.append(row)

        if arm != PLAIN_ARM:
            torch.random.set_rng_state(next_round_torch_rng_state)

        pd.DataFrame(round_rows).to_csv(
            tables_dir / "branch_round_metrics_partial.csv",
            index=False,
        )
        pd.DataFrame(score_rows).to_csv(
            tables_dir / "branch_client_scores_partial.csv",
            index=False,
        )
        print(
            f"  round {monitoring_round}: "
            f"macroF1={row['val_macro_f1']:.4f}, "
            f"macroASR={row['triggered_asr_macro_source']:.4f}, "
            f"recall={row['malicious_recall']}, "
            f"FPR={row['benign_false_positive_rate']:.3f}, "
            f"replaced={replaced_clients}, "
            f"seconds={row['round_seconds']:.1f}"
        )

    round_table = pd.DataFrame(round_rows)
    score_table = pd.DataFrame(score_rows)
    local_table = pd.DataFrame(local_rows)
    reconstruction_table = pd.DataFrame(reconstruction_rows)
    asr_table = pd.DataFrame(asr_rows)

    round_table.to_csv(
        tables_dir / "branch_round_metrics.csv", index=False
    )
    score_table.to_csv(
        tables_dir / "branch_client_scores.csv", index=False
    )
    local_table.to_csv(
        tables_dir / "branch_local_training.csv", index=False
    )
    reconstruction_table.to_csv(
        tables_dir / "branch_reconstruction_rows.csv",
        index=False,
    )
    asr_table.to_csv(
        tables_dir / "branch_triggered_asr_long.csv",
        index=False,
    )

    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    axis.plot(
        round_table["monitoring_round"],
        round_table["val_macro_f1"],
        marker="o",
        label="Clean validation macro F1",
    )
    axis.plot(
        round_table["monitoring_round"],
        round_table["triggered_asr_macro_source"],
        marker="s",
        label="Macro triggered ASR",
    )
    axis.set_xlabel("Monitoring round")
    axis.set_ylabel("Rate")
    axis.set_ylim(0, 1)
    axis.set_title(f"{condition.condition_id} | {arm}")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(
        figure,
        figures_dir / "branch_clean_utility_and_asr",
    )

    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    axis.plot(
        round_table["monitoring_round"],
        round_table["benign_false_positive_rate"],
        marker="o",
        label="Benign FPR",
    )
    if condition.is_attack and arm != PLAIN_ARM:
        axis.plot(
            round_table["monitoring_round"],
            round_table["malicious_recall"],
            marker="s",
            label="Malicious recall",
        )
    axis.axhline(FPR_CEILING, linestyle="--", label="FPR ceiling")
    axis.set_xlabel("Monitoring round")
    axis.set_ylabel("Rate")
    axis.set_ylim(0, 1)
    axis.set_title(f"Detection: {condition.condition_id} | {arm}")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(
        figure,
        figures_dir / "branch_detection_dynamics",
    )

    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": "task41c_c2_seed7_development_screen_branch",
        "complete": True,
        "condition_id": condition.condition_id,
        "mode": condition.mode,
        "arm": arm,
        "model_seed": MODEL_SEED,
        "attack_seed": ATTACK_SEED,
        "trigger_slot": condition.trigger_slot,
        "trigger_candidate_id": condition.trigger_candidate_id,
        "poison_fraction": condition.poison_fraction,
        "poison_plan_hash_sha256": poison_hash,
        "exact_pair_hash_scheme": (
            "version|condition|trigger|fraction|model_seed|attack_seed|"
            "partition|warmup_checkpoint|trigger_spec|poison_plan|round"
        ),
        "partition_hash_sha256": context.partition_hash,
        "probe_hash_sha256": context.probe_hash,
        "warmup_checkpoint_sha256": checkpoint_sha256(
            context.warmup_checkpoint_path
        ),
        "detector_profile_sha256": (
            context.detector_profile_sha256
        ),
        "detector_threshold_sha256": (
            context.detector_threshold_sha256
        ),
        "thresholds_frozen_without_attack_labels": True,
        "d3_weights": [1.0, 1.0, 1.0],
        "weight_tuning_used": False,
        "attack_specific_thresholds": False,
        "trigger_specific_thresholds": False,
        "continuation_rounds": CONTINUATION_ROUNDS,
        "development_arrays_loaded": list(ALLOWED_ARRAYS),
        "reserved_arrays_loaded": [],
        "reserved_test_accessed": False,
        "a2_executed": False,
        "adaptive_attack_executed": False,
        "task41b_reopened": False,
        "total_seconds": float(time.time() - branch_started),
        "maximum_benign_fpr": float(
            round_table["benign_false_positive_rate"].max()
        ),
        "mean_malicious_recall": (
            None
            if not condition.is_attack or arm == PLAIN_ARM
            else float(round_table["malicious_recall"].mean())
        ),
        "mean_macro_triggered_asr": float(
            round_table["triggered_asr_macro_source"].mean()
        ),
        "mean_clean_validation_macro_f1": float(
            round_table["val_macro_f1"].mean()
        ),
    }
    (branch_dir / "branch_complete.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def load_all_branch_tables(
    output_root: Path,
    conditions: Sequence[Condition],
) -> pd.DataFrame:
    frames = []
    for condition in conditions:
        for arm in ALL_ARMS:
            path = (
                output_root
                / "branches"
                / condition.condition_id
                / arm
                / "tables"
                / "branch_round_metrics.csv"
            )
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_screen(
    output_root: Path,
    conditions: Sequence[Condition],
    git_info: Dict[str, str],
) -> Dict[str, Any]:
    summary_dir = output_root / "summary"
    tables_dir = summary_dir / "tables"
    figures_dir = summary_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    round_table = load_all_branch_tables(output_root, conditions)
    expected_rows = len(conditions) * len(ALL_ARMS) * CONTINUATION_ROUNDS
    if len(round_table) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} branch-round rows, "
            f"found {len(round_table)}."
        )
    pair_counts = (
        round_table.groupby(
            ["condition_id", "monitoring_round"]
        )["exact_pair_hash_sha256"]
        .nunique()
    )
    if int(pair_counts.max()) != 1:
        raise RuntimeError("Exact-pair hashes differ between arms.")

    round_table.to_csv(
        tables_dir / "task41c2_all_branch_round_metrics.csv",
        index=False,
    )

    clean = round_table[
        round_table["condition_id"] == "clean_reference"
    ].copy()
    attacks = round_table[
        round_table["mode"] == "A1_dirty_label_low_rate"
    ].copy()
    plain_clean = clean[clean["arm"] == PLAIN_ARM][
        ["monitoring_round", "val_macro_f1", "round_seconds"]
    ].rename(
        columns={
            "val_macro_f1": "plain_clean_macro_f1",
            "round_seconds": "plain_clean_round_seconds",
        }
    )

    paired_rows: List[Dict[str, Any]] = []
    for condition_id, group in attacks.groupby("condition_id"):
        plain = group[group["arm"] == PLAIN_ARM][
            [
                "monitoring_round",
                "triggered_asr_macro_source",
                "round_seconds",
            ]
        ].rename(
            columns={
                "triggered_asr_macro_source": "plain_macro_asr",
                "round_seconds": "plain_round_seconds",
            }
        )
        for candidate in CANDIDATES:
            defended = group[group["arm"] == candidate].merge(
                plain,
                on="monitoring_round",
                how="inner",
                validate="one_to_one",
            )
            for row in defended.to_dict(orient="records"):
                plain_asr = float(row["plain_macro_asr"])
                defended_asr = float(
                    row["triggered_asr_macro_source"]
                )
                paired_rows.append(
                    {
                        "condition_id": condition_id,
                        "trigger_slot": row["trigger_slot"],
                        "poison_fraction": row["poison_fraction"],
                        "monitoring_round": row["monitoring_round"],
                        "candidate": candidate,
                        "plain_macro_asr": plain_asr,
                        "defended_macro_asr": defended_asr,
                        "absolute_asr_reduction": (
                            plain_asr - defended_asr
                        ),
                        "relative_asr_reduction": (
                            (plain_asr - defended_asr)
                            / max(plain_asr, 1e-12)
                        ),
                        "malicious_recall": row["malicious_recall"],
                        "benign_fpr": (
                            row["benign_false_positive_rate"]
                        ),
                        "candidate_round_seconds": row["round_seconds"],
                        "plain_round_seconds": row["plain_round_seconds"],
                        "runtime_overhead_seconds": (
                            row["round_seconds"]
                            - row["plain_round_seconds"]
                        ),
                    }
                )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(
        tables_dir / "task41c2_paired_attack_rounds.csv",
        index=False,
    )

    condition_summary = (
        paired.groupby(
            [
                "condition_id",
                "trigger_slot",
                "poison_fraction",
                "candidate",
            ],
            as_index=False,
        )
        .agg(
            plain_macro_asr=("plain_macro_asr", "mean"),
            defended_macro_asr=("defended_macro_asr", "mean"),
            mean_absolute_asr_reduction=(
                "absolute_asr_reduction",
                "mean",
            ),
            mean_relative_asr_reduction=(
                "relative_asr_reduction",
                "mean",
            ),
            mean_malicious_recall=("malicious_recall", "mean"),
            maximum_benign_fpr=("benign_fpr", "max"),
            mean_runtime_overhead_seconds=(
                "runtime_overhead_seconds",
                "mean",
            ),
        )
    )
    condition_summary.to_csv(
        tables_dir / "task41c2_attack_condition_summary.csv",
        index=False,
    )

    candidate_rows: List[Dict[str, Any]] = []
    for candidate in CANDIDATES:
        clean_candidate = clean[clean["arm"] == candidate].merge(
            plain_clean,
            on="monitoring_round",
            how="inner",
            validate="one_to_one",
        )
        clean_candidate["clean_macro_f1_loss"] = (
            clean_candidate["plain_clean_macro_f1"]
            - clean_candidate["val_macro_f1"]
        )
        candidate_attack = paired[
            paired["candidate"] == candidate
        ]
        candidate_all = round_table[
            round_table["arm"] == candidate
        ]
        mean_clean_loss = float(
            clean_candidate["clean_macro_f1_loss"].mean()
        )
        maximum_clean_round_loss = float(
            clean_candidate["clean_macro_f1_loss"].max()
        )
        maximum_benign_fpr = float(
            candidate_all["benign_false_positive_rate"].max()
        )
        mean_recall = float(
            candidate_attack["malicious_recall"].mean()
        )
        mean_relative_reduction = float(
            candidate_attack["relative_asr_reduction"].mean()
        )
        mean_absolute_reduction = float(
            candidate_attack["absolute_asr_reduction"].mean()
        )
        mean_runtime_overhead = float(
            candidate_attack["runtime_overhead_seconds"].mean()
        )
        survives_fpr = maximum_benign_fpr <= FPR_CEILING
        survives_clean = (
            mean_clean_loss <= MEAN_CLEAN_F1_LOSS_CEILING
            and maximum_clean_round_loss
            <= MAX_ROUND_CLEAN_F1_LOSS_CEILING
        )
        candidate_rows.append(
            {
                "candidate": candidate,
                "maximum_benign_fpr": maximum_benign_fpr,
                "fpr_ceiling": FPR_CEILING,
                "survives_fpr_gate": survives_fpr,
                "mean_clean_macro_f1_loss": mean_clean_loss,
                "mean_clean_loss_ceiling": (
                    MEAN_CLEAN_F1_LOSS_CEILING
                ),
                "maximum_clean_round_macro_f1_loss": (
                    maximum_clean_round_loss
                ),
                "maximum_round_loss_ceiling": (
                    MAX_ROUND_CLEAN_F1_LOSS_CEILING
                ),
                "survives_clean_utility_gate": survives_clean,
                "survives_hard_gates": (
                    survives_fpr and survives_clean
                ),
                "mean_malicious_recall_all_A1_rounds": mean_recall,
                "mean_paired_relative_asr_reduction": (
                    mean_relative_reduction
                ),
                "mean_paired_absolute_asr_reduction": (
                    mean_absolute_reduction
                ),
                "mean_runtime_overhead_seconds": (
                    mean_runtime_overhead
                ),
                "attack_condition_count": int(
                    candidate_attack["condition_id"].nunique()
                ),
                "attack_round_pair_count": int(len(candidate_attack)),
                "clean_round_pair_count": int(len(clean_candidate)),
            }
        )
    selection = pd.DataFrame(candidate_rows)

    survivors = selection[
        selection["survives_hard_gates"].astype(bool)
    ].copy()
    if survivors.empty:
        selected_candidate = None
        selection_status = "NO_CANDIDATE_SURVIVED"
    else:
        survivors = survivors.sort_values(
            by=[
                "mean_malicious_recall_all_A1_rounds",
                "mean_paired_relative_asr_reduction",
                "mean_runtime_overhead_seconds",
                "candidate",
            ],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        selected_candidate = str(
            survivors.iloc[0]["candidate"]
        )
        selection_status = "ONE_CANDIDATE_SELECTED"
    selection["selected"] = (
        selection["candidate"] == selected_candidate
        if selected_candidate is not None
        else False
    )
    selection["selection_rank_among_survivors"] = np.nan
    if not survivors.empty:
        for rank, candidate in enumerate(
            survivors["candidate"].tolist(), start=1
        ):
            selection.loc[
                selection["candidate"] == candidate,
                "selection_rank_among_survivors",
            ] = rank

    selection.to_csv(
        tables_dir / "task41c2_candidate_selection.csv",
        index=False,
    )

    figure, axis = plt.subplots(figsize=(9.5, 6.0))
    for row in selection.to_dict(orient="records"):
        axis.scatter(
            row["maximum_benign_fpr"],
            row["mean_malicious_recall_all_A1_rounds"],
            s=80,
            label=row["candidate"],
        )
    axis.axvline(FPR_CEILING, linestyle="--", label="FPR ceiling")
    axis.set_xlabel("Maximum benign FPR")
    axis.set_ylabel("Mean malicious-client recall")
    axis.set_title("Task 41C C2 detector selection plane")
    axis.set_xlim(left=0)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(
        figure,
        figures_dir / "task41c2_detection_selection_plane",
    )

    figure, axis = plt.subplots(figsize=(11, 6))
    pivot = condition_summary.pivot(
        index="condition_id",
        columns="candidate",
        values="mean_relative_asr_reduction",
    )
    pivot.plot(kind="bar", ax=axis)
    axis.axhline(0.0, linestyle="--")
    axis.set_ylabel("Mean paired relative ASR reduction")
    axis.set_title("Task 41C C2 mitigation by A1 condition")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    save_figure(
        figure,
        figures_dir / "task41c2_paired_asr_reduction",
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(
        selection["candidate"],
        selection["mean_clean_macro_f1_loss"],
    )
    axis.axhline(
        MEAN_CLEAN_F1_LOSS_CEILING,
        linestyle="--",
        label="Mean loss ceiling",
    )
    axis.set_ylabel("Mean clean macro-F1 loss")
    axis.set_title("Task 41C C2 clean-utility gate")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save_figure(
        figure,
        figures_dir / "task41c2_clean_utility_gate",
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(
        selection["candidate"],
        selection["mean_runtime_overhead_seconds"],
    )
    axis.set_ylabel("Mean runtime overhead per attack round, seconds")
    axis.set_title("Task 41C C2 runtime tie-break evidence")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    save_figure(
        figure,
        figures_dir / "task41c2_runtime_overhead",
    )

    decision = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": "task41c_c2_seed7_development_screen",
        "selection_status": selection_status,
        "selected_candidate": selected_candidate,
        "candidate_count": len(CANDIDATES),
        "candidate_survivor_count": int(len(survivors)),
        "A1_condition_count": 6,
        "attack_round_pair_count_per_candidate": 24,
        "clean_round_pair_count_per_candidate": 4,
        "selection_rule": [
            "Reject maximum benign FPR above 0.05.",
            "Reject mean clean macro-F1 loss above 0.01 or any clean round loss above 0.02.",
            "Maximize mean malicious recall across all A1 seed-7 condition-rounds.",
            "Tie-break by mean paired relative ASR reduction.",
            "Tie-break by lower runtime overhead.",
        ],
        "c3_confirmatory_multiseed_allowed_after_c2_freeze": (
            selected_candidate is not None
        ),
        "a2_executed": False,
        "adaptive_attack_executed": False,
        "reserved_test_accessed": False,
        "task41b_reopened": False,
        "thresholds_retuned": False,
        "profiles_retuned": False,
        "weights_retuned": False,
        "git_integrity": git_info,
        "next_stage": (
            "Freeze C2 evidence and selected candidate. Then run C3 "
            "confirmatory multiseed A1 and A2 only with no retuning."
            if selected_candidate is not None
            else "Record Task 41C candidate-development failure and do not invent a new candidate."
        ),
    }
    (summary_dir / "task41c2_selection_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    return decision


def write_preflight(
    output_root: Path,
    context: FrozenContext,
    conditions: Sequence[Condition],
    git_info: Dict[str, str],
) -> None:
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    condition_rows = [
        {
            "condition_id": condition.condition_id,
            "mode": condition.mode,
            "trigger_slot": condition.trigger_slot,
            "trigger_candidate_id": condition.trigger_candidate_id,
            "poison_fraction": condition.poison_fraction,
            "attack_execution_performed": False,
        }
        for condition in conditions
    ]
    pd.DataFrame(condition_rows).to_csv(
        tables_dir / "task41c2_preflight_conditions.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "arm": arm,
                "role": (
                    "paired_plain_reference"
                    if arm == PLAIN_ARM
                    else "frozen_candidate"
                ),
                "thresholds_frozen": arm != PLAIN_ARM,
                "profiles_frozen": arm != PLAIN_ARM,
            }
            for arm in ALL_ARMS
        ]
    ).to_csv(
        tables_dir / "task41c2_preflight_arms.csv",
        index=False,
    )
    threshold_table = context.thresholds.copy()
    threshold_table.to_csv(
        tables_dir / "task41c2_preflight_seed7_thresholds.csv",
        index=False,
    )

    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    attack_conditions = [
        condition for condition in conditions if condition.is_attack
    ]
    counts = pd.DataFrame(
        [
            {
                "trigger_slot": slot,
                "condition_count": sum(
                    condition.trigger_slot == slot
                    for condition in attack_conditions
                ),
            }
            for slot in TRIGGER_SLOTS
        ]
    )
    axis.bar(counts["trigger_slot"], counts["condition_count"])
    axis.set_ylabel("A1 poison-fraction conditions")
    axis.set_title("Task 41C C2 frozen seed-7 attack panel")
    axis.grid(axis="y", alpha=0.25)
    save_figure(
        figure,
        figures_dir / "task41c2_preflight_attack_panel",
    )

    decision = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": "task41c_c2_seed7_preflight",
        "preflight_passed": True,
        "model_seed": MODEL_SEED,
        "condition_count": len(conditions),
        "A1_condition_count": len(attack_conditions),
        "arm_count": len(ALL_ARMS),
        "candidate_count": len(CANDIDATES),
        "partition_hash_sha256": context.partition_hash,
        "probe_hash_sha256": context.probe_hash,
        "warmup_checkpoint_sha256": checkpoint_sha256(
            context.warmup_checkpoint_path
        ),
        "detector_profile_sha256": context.detector_profile_sha256,
        "detector_threshold_sha256": context.detector_threshold_sha256,
        "trigger_spec_sha256": context.trigger_spec_file_sha256,
        "development_arrays_loaded": list(ALLOWED_ARRAYS),
        "reserved_arrays_loaded": [],
        "attack_execution_performed": False,
        "a2_executed": False,
        "adaptive_attack_executed": False,
        "thresholds_retuned": False,
        "profiles_retuned": False,
        "weights_retuned": False,
        "task41b_reopened": False,
        "reserved_test_accessed": False,
        "git_integrity": git_info,
        "c2_execution_allowed": True,
    }
    (output_root / "task41c2_preflight_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    write_output_manifest(output_root)
    print("===== TASK 41C.2 SEED-7 PREFLIGHT =====")
    print("Preflight passed: True")
    print("Conditions:", len(conditions))
    print("A1 conditions:", len(attack_conditions))
    print("Arms:", len(ALL_ARMS))
    print("Candidates:", len(CANDIDATES))
    print("C2 EXECUTION ALLOWED: True")
    print("ATTACK EXECUTION PERFORMED: False")
    print("A2 EXECUTED: False")
    print("ADAPTIVE ATTACK EXECUTED: False")
    print("THRESHOLDS RETUNED: False")
    print("PROFILES RETUNED: False")
    print("WEIGHTS RETUNED: False")
    print("TASK 41B REOPENED: False")
    print("TEST SETS ACCESSED: False")
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)


def main() -> int:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined.")
    if args.probe_per_class != 48 or args.probe_seed != 3701:
        raise ValueError("Frozen probe configuration mismatch.")
    if args.threads < 1:
        raise ValueError("threads must be positive.")

    torch.set_num_threads(args.threads)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_root)
        elif not args.resume:
            raise FileExistsError(
                f"Output root is not empty: {output_root}"
            )
    output_root.mkdir(parents=True, exist_ok=True)

    context, git_info = load_frozen_context(args)
    conditions = frozen_conditions()

    protocol_snapshot = (
        PROJECT_ROOT / "configs" / "task41c_preregistration_v412c0.json"
    )
    shutil.copy2(
        protocol_snapshot,
        output_root / "task41c_preregistration_v412c0_snapshot.json",
    )

    if args.preflight_only:
        write_preflight(output_root, context, conditions, git_info)
        return 0

    screen_started = time.time()
    for condition_index, condition in enumerate(conditions, start=1):
        print()
        print("=" * 100)
        print(
            f"CONDITION {condition_index}/{len(conditions)}: "
            f"{condition.condition_id}"
        )
        print("=" * 100)
        (
            poisoned_positions,
            poisoned_labels,
            _,
            _,
            poison_hash,
        ) = prepare_condition_poison_plan(
            condition, context, output_root
        )
        for arm_index, arm in enumerate(ALL_ARMS, start=1):
            print(
                f"ARM {arm_index}/{len(ALL_ARMS)}: {arm}"
            )
            run_branch(
                condition=condition,
                arm=arm,
                context=context,
                poisoned_positions=poisoned_positions,
                poisoned_labels=poisoned_labels,
                poison_hash=poison_hash,
                output_root=output_root,
                args=args,
            )

    decision = summarize_screen(
        output_root, conditions, git_info
    )
    decision["total_screen_seconds"] = float(
        time.time() - screen_started
    )
    decision_path = (
        output_root
        / "summary"
        / "task41c2_selection_decision.json"
    )
    decision_path.write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    write_output_manifest(output_root)

    print()
    print("=" * 100)
    print("TASK 41C.2 SEED-7 DEVELOPMENT SCREEN")
    print("=" * 100)
    print("Selection status:", decision["selection_status"])
    print("Selected candidate:", decision["selected_candidate"])
    print(
        "Candidate survivors:",
        f"{decision['candidate_survivor_count']}/"
        f"{decision['candidate_count']}",
    )
    print(
        "C3 CONFIRMATORY MULTISEED ALLOWED AFTER C2 FREEZE:",
        decision[
            "c3_confirmatory_multiseed_allowed_after_c2_freeze"
        ],
    )
    print("A2 EXECUTED: False")
    print("ADAPTIVE ATTACK EXECUTED: False")
    print("THRESHOLDS RETUNED: False")
    print("PROFILES RETUNED: False")
    print("WEIGHTS RETUNED: False")
    print("TASK 41B REOPENED: False")
    print("TEST SETS ACCESSED: False")
    print(
        "Selection table:",
        output_root
        / "summary"
        / "tables"
        / "task41c2_candidate_selection.csv",
    )
    print(
        "PNG and PDF figures:",
        output_root / "summary" / "figures",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

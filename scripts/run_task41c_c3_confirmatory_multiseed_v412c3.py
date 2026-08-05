#!/usr/bin/env python3
"""Task 41C.3 confirmatory multiseed screen (A1 only).

Scope
-----
- Seeds 7, 99, 123, 2026 (the frozen Task 41C seed panel).
- Clean continuation reference plus all six preregistered A1 dirty-label
  conditions per seed: two frozen trigger families and poison fractions
  0.005, 0.01, and 0.02.
- Exact paired plain FedAvg, D0 (frozen Task 40 LFighter, mandatory
  reference), and D2 (trigger_gradient_alignment, the C2-selected
  candidate). D1 and D3 are NOT re-run here: both failed the C2 hard gates
  (benign FPR ceiling) and the preregistration prohibits inventing or
  reviving candidates after the C2 freeze.
- Four post-warmup monitoring rounds per seed, exactly as in C2.
- Development arrays only. No reserved natural/diagnostic test access.
- No A2, A3, or A4 execution in this script (see module docstring note
  below). No threshold, profile, weight, seed, client, trigger, or
  fraction tuning of any kind versus C1c3/C2.

A2 (clean-label source-preserving attacks) is NOT implemented in this
script. C2's poisoning logic (imported from run_task41b_backdoor_smoke_v411b1)
only supports the dirty-label attack type used in A1. Building A2 requires
a separate, explicitly preregistered clean-label poison-generation function
and must not be improvised inside this confirmatory stage.

Per-seed frozen inputs are supplied via --seed-manifest, a JSON file of the
form:

{
  "7":    {"clean_seed_dir": "...", "warmup_dir": "...",
           "reconstruction_calibration_dir": "...",
           "c1_calibration_dir": "..."},
  "99":   {...},
  "123":  {...},
  "2026": {...}
}

c1_calibration_dir may be the same multiseed root for every seed (C1c3
already stores seed_7/seed_99/seed_123/seed_2026 subfolders under one
root) -- the loader appends "seed_<N>" itself, exactly as in C2.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

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

# Reuse every frozen building block from the C2 seed-7 screen verbatim.
# This is deliberate: C3 must be the *same* implementation as C2, only
# generalized over seeds, or the confirmatory claim is void.
#
# EXCEPTION: load_frozen_context is NOT imported directly. C2's version
# hardcodes the C1 calibration subdirectory as the literal "seed_7"
# (harmless in C2, which is seed-7-only). Reusing it verbatim here would
# silently load seed 7's calibration for every seed. load_frozen_context_
# for_seed() below is a faithful copy of C2's function with only that one
# line parameterized by model_seed -- everything else, including every
# hash-verification check, is unchanged.
from run_task41c_seed7_c2_screen_v412c2a import (  # noqa: E402
    EXPERIMENT_VERSION as C2_EXPERIMENT_VERSION,
    NUM_CLIENTS,
    NUM_CLASSES,
    CONTINUATION_ROUNDS,
    FPR_CEILING,
    MEAN_CLEAN_F1_LOSS_CEILING,
    MAX_ROUND_CLEAN_F1_LOSS_CEILING,
    PLAIN_ARM,
    CANDIDATES,
    ALLOWED_ARRAYS,
    MALICIOUS_CLIENTS,
    TRIGGER_SLOTS,
    POISON_FRACTIONS,
    REQUIRED_C0_TAG,
    REQUIRED_C1_RESULTS_TAG,
    REQUIRED_C1_VERSION,
    CLASS_NAMES,
    Condition,
    FrozenContext,
    frozen_conditions,
    prepare_condition_poison_plan,
    run_branch,
    save_figure,
    write_output_manifest,
    file_sha256,
    stable_text_hash,
)

# Additional low-level imports needed only to reconstruct a seed-parameterized
# load_frozen_context. These are the exact same modules C2 imports.
from federated_iot_v26 import set_seed  # noqa: E402
from neural_models_v24 import build_model  # noqa: E402
from run_targeted_label_flip_v292 import (  # noqa: E402
    load_fixed_partitions,
    read_clean_partition_hash,
)
from transition_signature_features_v38 import balanced_probe_indices  # noqa: E402
from independent_anchor_v310 import load_profiles_npz  # noqa: E402
from trusted_update_reconstruction_v312 import checkpoint_sha256  # noqa: E402
from run_task41b_backdoor_smoke_v411b1 import load_trigger_spec  # noqa: E402
import run_task41c_seed7_c2_screen_v412c2a as _c2_module  # noqa: E402

EXPERIMENT_VERSION = "4.12C.3"
ATTACK_SEED = 42  # Held fixed across model seeds, matching C2 and the
                   # Task 40 V3.20B.1 convention (attack construction is
                   # keyed by ATTACK_SEED, not by MODEL_SEED).
CONFIRMATORY_SEEDS = (7, 99, 123, 2026)
CONFIRMATORY_ARMS = (PLAIN_ARM, "D0_frozen_task40_lfighter", "D2_trigger_gradient_alignment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Task 41C C3 confirmatory multiseed screen "
            "(A1 only; D0 and D2 arms)."
        )
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--partition-file", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--trigger-spec-file", type=Path, required=True)
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
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate all four seeds' frozen inputs and hashes "
            "(context load only) without running any training or "
            "attack branches."
        ),
    )
    return parser.parse_args()


def load_seed_manifest(path: Path) -> Dict[int, Dict[str, Path]]:
    raw = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    manifest: Dict[int, Dict[str, Path]] = {}
    required_keys = {
        "clean_seed_dir",
        "warmup_dir",
        "reconstruction_calibration_dir",
        "c1_calibration_dir",
    }
    for seed in CONFIRMATORY_SEEDS:
        entry = raw.get(str(seed))
        if entry is None:
            raise KeyError(f"Seed manifest missing entry for seed {seed}.")
        missing = required_keys.difference(entry)
        if missing:
            raise KeyError(f"Seed {seed} manifest missing keys: {sorted(missing)}")
        manifest[seed] = {key: Path(entry[key]) for key in required_keys}
    return manifest


def load_frozen_context_for_seed(
    model_seed: int, args: argparse.Namespace
) -> Any:
    """Seed-parameterized reconstruction of C2's load_frozen_context.

    Identical to run_task41c_seed7_c2_screen_v412c2a.load_frozen_context in
    every check and every hash comparison. The only change: the C1
    calibration subdirectory is f"seed_{model_seed}" instead of C2's
    hardcoded literal "seed_7". Every internal helper (git verification,
    array loading, threshold parsing, profile loading, trigger-spec
    loading, reconstruction-bundle loading) is called via the C2 module
    object so the underlying logic is byte-identical to C2's frozen
    implementation.
    """
    c2 = _c2_module
    NUM_CLIENTS_ = NUM_CLIENTS

    git_info = c2.verify_frozen_git_inputs()

    protocol_path = (
        PROJECT_ROOT / "configs" / "task41c_preregistration_v412c0.json"
    )
    seed_dir = (
        args.c1_calibration_dir.expanduser().resolve() / f"seed_{model_seed}"
    )
    c1_decision_path = (
        args.c1_calibration_dir.expanduser().resolve()
        / "summary"
        / "task41c1c_clean_calibration_decision.json"
    )
    threshold_path = (
        seed_dir / "calibration" / "task41c1c_candidate_thresholds.csv"
    )
    detector_profile_path = (
        seed_dir / "calibration" / "task41c1c_backdoor_detector_profiles.npz"
    )
    c1_metadata_path = seed_dir / "task41c1c_clean_calibration_metadata.json"
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
    c1_decision = json.loads(c1_decision_path.read_text(encoding="utf-8"))
    c1_metadata = json.loads(c1_metadata_path.read_text(encoding="utf-8"))

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
    if not bool(c1_decision.get("replay_equivalence_passed_all_seeds", False)):
        raise RuntimeError("C1 replay-equivalence gate is false.")
    if bool(c1_decision.get("attack_execution_performed", True)):
        raise RuntimeError("C1 decision reports attack execution.")
    if bool(c1_decision.get("reserved_test_accessed", True)):
        raise RuntimeError("C1 decision reports reserved-test access.")
    if c1_metadata.get("seed") != model_seed:
        raise RuntimeError(
            f"C1 calibration metadata seed mismatch: expected {model_seed}, "
            f"found {c1_metadata.get('seed')} at {c1_metadata_path}."
        )
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
    c2.threshold_lookup(thresholds)

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
            raise RuntimeError(f"C1 profile NPZ lacks arrays: {sorted(missing)}")
        for key in required_keys:
            detector_profiles[key] = np.asarray(archive[key])
    if detector_profiles["d1_client_profiles"].shape[0] != NUM_CLIENTS_:
        raise RuntimeError("D1 profile client count mismatch.")
    if detector_profiles["d2_client_profiles"].shape[0] != NUM_CLIENTS_:
        raise RuntimeError("D2 profile client count mismatch.")

    arrays = c2.load_development_arrays(args.data_file)
    X_train = arrays["X_train"].astype(np.float32, copy=False)
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    X_val = arrays["X_val"].astype(np.float32, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)

    client_indices, partition_hash = load_fixed_partitions(
        args.partition_file.expanduser().resolve(),
        expected_clients=NUM_CLIENTS_,
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
        warmup_dir / "checkpoints" / "common_round4_warmup_model.pt"
    )
    task40_profiles_path = (
        warmup_dir / "calibration" / "trusted_client_profiles.npz"
    )
    task40_feature_path = warmup_dir / "calibration" / "feature_calibration.csv"
    for path in (
        warmup_metadata_path,
        warmup_checkpoint_path,
        task40_profiles_path,
        task40_feature_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    warmup_metadata = json.loads(warmup_metadata_path.read_text(encoding="utf-8"))
    if int(warmup_metadata["model_seed"]) != model_seed:
        raise RuntimeError("Warmup seed mismatch.")
    if warmup_metadata["partition_hash_sha256"] != partition_hash:
        raise RuntimeError("Warmup partition hash mismatch.")
    if not bool(warmup_metadata.get("monitoring_ema_reset_after_warmup", False)):
        raise RuntimeError("Warmup EMA-reset contract is missing.")
    if file_sha256(task40_profiles_path) != warmup_metadata["profile_sha256"]:
        raise RuntimeError("Task 40 trusted-profile hash mismatch.")
    if (
        file_sha256(task40_feature_path)
        != warmup_metadata["feature_calibration_sha256"]
    ):
        raise RuntimeError("Task 40 feature-calibration hash mismatch.")

    warmup_checkpoint = torch.load(
        warmup_checkpoint_path, map_location="cpu", weights_only=False
    )
    if warmup_checkpoint.get("partition_hash") != partition_hash:
        raise RuntimeError("Warmup checkpoint partition mismatch.")

    probe_indices = balanced_probe_indices(
        y_val, args.probe_per_class, args.probe_seed
    )
    probe_hash = c2.sha256_array(probe_indices)
    if probe_hash != warmup_metadata["probe_hash_sha256"]:
        raise RuntimeError("Warmup probe hash mismatch.")
    X_probe = X_val[probe_indices]
    y_probe = y_val[probe_indices]

    task40_profiles = load_profiles_npz(task40_profiles_path, NUM_CLIENTS_)
    task40_feature_calibration = pd.read_csv(task40_feature_path)

    trigger_panel, trigger_panel_sha256 = c2.load_trigger_panel(
        args.trigger_spec_file
    )
    trigger_specs: Dict[str, Any] = {}
    trigger_spec_file_sha256 = file_sha256(
        args.trigger_spec_file.expanduser().resolve()
    )
    for slot, candidate_id in TRIGGER_SLOTS.items():
        spec, spec_hash = load_trigger_spec(args.trigger_spec_file, candidate_id)
        if spec_hash != trigger_spec_file_sha256:
            raise RuntimeError("Trigger specification file hash changed.")
        trigger_specs[slot] = spec
    if c1_metadata["trigger_spec_sha256"] != trigger_spec_file_sha256:
        raise RuntimeError("C1 trigger-specification hash mismatch.")
    if c1_metadata["partition_hash_sha256"] != partition_hash:
        raise RuntimeError("C1 partition hash mismatch.")
    if c1_metadata["probe_hash_sha256"] != probe_hash:
        raise RuntimeError("C1 probe hash mismatch.")
    if c1_metadata["common_warmup_checkpoint_sha256"] != checkpoint_sha256(
        warmup_checkpoint_path
    ):
        raise RuntimeError("C1 warmup-checkpoint hash mismatch.")

    reconstruction_dir = args.reconstruction_calibration_dir.expanduser().resolve()
    reconstruction_checkpoint_path = (
        reconstruction_dir
        / "calibration"
        / "trusted_update_reconstruction_profiles.pt"
    )
    reconstruction_summary_path = (
        reconstruction_dir / "calibration" / "reconstruction_calibration_summary.csv"
    )
    for path in (reconstruction_checkpoint_path, reconstruction_summary_path):
        if not path.exists():
            raise FileNotFoundError(path)
    reconstruction_bundle = torch.load(
        reconstruction_checkpoint_path, map_location="cpu", weights_only=False
    )
    if int(reconstruction_bundle["model_seed"]) != model_seed:
        raise RuntimeError("Reconstruction seed mismatch.")
    if reconstruction_bundle["partition_hash"] != partition_hash:
        raise RuntimeError("Reconstruction partition mismatch.")
    if reconstruction_bundle[
        "common_warmup_checkpoint_sha256"
    ] != checkpoint_sha256(warmup_checkpoint_path):
        raise RuntimeError("Reconstruction warmup-checkpoint mismatch.")
    selected_reconstruction_policy = str(reconstruction_bundle["selected_policy"])
    if selected_reconstruction_policy != "center_plus_residual":
        raise RuntimeError("Task 41C requires center_plus_residual reconstruction.")
    reconstruction_profiles = {
        int(client_id): {
            key: value.detach().cpu().to(torch.float32)
            for key, value in profile.items()
        }
        for client_id, profile in reconstruction_bundle[
            "client_residual_profiles"
        ].items()
    }
    if set(reconstruction_profiles) != set(range(NUM_CLIENTS_)):
        raise RuntimeError("Reconstruction client-profile set mismatch.")

    probe_model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    names = c2.parameter_names(probe_model)
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
        warmup_update_scale=float(reconstruction_bundle["warmup_update_scale"]),
        scale_lower=float(reconstruction_bundle["scale_lower"]),
        scale_upper=float(reconstruction_bundle["scale_upper"]),
        norm_clip_multiplier=float(reconstruction_bundle["norm_clip_multiplier"]),
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


def seed_args_for(
    model_seed: int, seed_paths: Dict[str, Path], args: argparse.Namespace
) -> argparse.Namespace:
    seed_args = argparse.Namespace(**vars(args))
    seed_args.clean_seed_dir = seed_paths["clean_seed_dir"]
    seed_args.warmup_dir = seed_paths["warmup_dir"]
    seed_args.reconstruction_calibration_dir = seed_paths[
        "reconstruction_calibration_dir"
    ]
    seed_args.c1_calibration_dir = seed_paths["c1_calibration_dir"]
    return seed_args


def preflight_seed(
    *,
    model_seed: int,
    seed_paths: Dict[str, Path],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Load and hash-verify one seed's frozen inputs without training.

    Mirrors load_frozen_context's own internal checks (partition hash,
    warmup checkpoint hash, C1c3 calibration hash, trigger-spec hash,
    reconstruction-bundle hash) -- if any of those fail, this raises
    exactly the same RuntimeError/FileNotFoundError the real run would,
    just before spending any compute.
    """
    import run_task41c_seed7_c2_screen_v412c2a as c2

    original_model_seed = c2.MODEL_SEED
    c2.MODEL_SEED = model_seed
    try:
        seed_args = seed_args_for(model_seed, seed_paths, args)
        context, git_info = load_frozen_context_for_seed(model_seed, seed_args)
        conditions = frozen_conditions()
        return {
            "model_seed": model_seed,
            "preflight_passed": True,
            "partition_hash_sha256": context.partition_hash,
            "probe_hash_sha256": context.probe_hash,
            "warmup_checkpoint_sha256": file_sha256(
                context.warmup_checkpoint_path
            ),
            "detector_profile_sha256": context.detector_profile_sha256,
            "detector_threshold_sha256": context.detector_threshold_sha256,
            "trigger_spec_sha256": context.trigger_spec_file_sha256,
            "condition_count": len(conditions),
            "arm_count": len(CONFIRMATORY_ARMS),
            "git_integrity": git_info,
        }
    finally:
        c2.MODEL_SEED = original_model_seed


def run_seed(
    *,
    model_seed: int,
    seed_paths: Dict[str, Path],
    args: argparse.Namespace,
    output_root: Path,
) -> pd.DataFrame:
    """Run all A1 conditions x {plain, D0, D2} for one confirmatory seed.

    Mirrors run_task41c_seed7_c2_screen_v412c2a.main()'s per-condition loop
    exactly, but restricted to CONFIRMATORY_ARMS and parameterized over
    model_seed via a monkey-patched context (MODEL_SEED is read from the
    frozen module at import time, so we pass model_seed through the args
    namespace and rely on load_frozen_context's own seed-consistency checks
    to catch any seed/input mismatch immediately).
    """
    import run_task41c_seed7_c2_screen_v412c2a as c2

    original_model_seed = c2.MODEL_SEED
    c2.MODEL_SEED = model_seed
    try:
        seed_args = seed_args_for(model_seed, seed_paths, args)
        context, git_info = load_frozen_context_for_seed(model_seed, seed_args)
        conditions = frozen_conditions()

        seed_output_root = output_root / f"seed_{model_seed}"
        seed_output_root.mkdir(parents=True, exist_ok=True)

        round_frames: List[pd.DataFrame] = []
        for condition_index, condition in enumerate(conditions, start=1):
            print()
            print("=" * 100)
            print(
                f"SEED {model_seed} | CONDITION {condition_index}/"
                f"{len(conditions)}: {condition.condition_id}"
            )
            print("=" * 100)
            (
                poisoned_positions,
                poisoned_labels,
                _,
                _,
                poison_hash,
            ) = prepare_condition_poison_plan(
                condition, context, seed_output_root
            )
            for arm_index, arm in enumerate(CONFIRMATORY_ARMS, start=1):
                print(f"ARM {arm_index}/{len(CONFIRMATORY_ARMS)}: {arm}")
                metadata = run_branch(
                    condition=condition,
                    arm=arm,
                    context=context,
                    poisoned_positions=poisoned_positions,
                    poisoned_labels=poisoned_labels,
                    poison_hash=poison_hash,
                    output_root=seed_output_root,
                    args=seed_args,
                )
                round_path = (
                    seed_output_root
                    / "branches"
                    / condition.condition_id
                    / arm
                    / "tables"
                    / "branch_round_metrics.csv"
                )
                frame = pd.read_csv(round_path)
                frame["model_seed"] = model_seed
                round_frames.append(frame)

        write_output_manifest(seed_output_root)
        return pd.concat(round_frames, ignore_index=True)
    finally:
        c2.MODEL_SEED = original_model_seed


def summarize_confirmatory(
    output_root: Path,
    all_rounds: pd.DataFrame,
) -> Dict[str, Any]:
    summary_dir = output_root / "summary"
    tables_dir = summary_dir / "tables"
    figures_dir = summary_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_rounds.to_csv(
        tables_dir / "task41c3_all_seed_round_metrics.csv", index=False
    )

    clean = all_rounds[all_rounds["condition_id"] == "clean_reference"].copy()
    attacks = all_rounds[
        all_rounds["mode"] == "A1_dirty_label_low_rate"
    ].copy()

    plain_clean = clean[clean["arm"] == PLAIN_ARM][
        ["model_seed", "monitoring_round", "val_macro_f1"]
    ].rename(columns={"val_macro_f1": "plain_clean_macro_f1"})

    paired_rows: List[Dict[str, Any]] = []
    for (model_seed, condition_id), group in attacks.groupby(
        ["model_seed", "condition_id"]
    ):
        plain = group[group["arm"] == PLAIN_ARM][
            ["monitoring_round", "triggered_asr_macro_source"]
        ].rename(columns={"triggered_asr_macro_source": "plain_macro_asr"})
        for candidate in ("D0_frozen_task40_lfighter", "D2_trigger_gradient_alignment"):
            defended = group[group["arm"] == candidate].merge(
                plain, on="monitoring_round", how="inner", validate="one_to_one"
            )
            for row in defended.to_dict(orient="records"):
                plain_asr = float(row["plain_macro_asr"])
                defended_asr = float(row["triggered_asr_macro_source"])
                paired_rows.append(
                    {
                        "model_seed": model_seed,
                        "condition_id": condition_id,
                        "candidate": candidate,
                        "monitoring_round": row["monitoring_round"],
                        "absolute_asr_reduction": plain_asr - defended_asr,
                        "relative_asr_reduction": (
                            (plain_asr - defended_asr) / max(plain_asr, 1e-12)
                        ),
                        "malicious_recall": row["malicious_recall"],
                        "benign_fpr": row["benign_false_positive_rate"],
                    }
                )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(tables_dir / "task41c3_paired_attack_rounds.csv", index=False)

    candidate_rows: List[Dict[str, Any]] = []
    for candidate in ("D0_frozen_task40_lfighter", "D2_trigger_gradient_alignment"):
        clean_candidate = clean[clean["arm"] == candidate].merge(
            plain_clean, on=["model_seed", "monitoring_round"], how="inner"
        )
        clean_candidate["clean_macro_f1_loss"] = (
            clean_candidate["plain_clean_macro_f1"] - clean_candidate["val_macro_f1"]
        )
        candidate_attack = paired[paired["candidate"] == candidate]
        candidate_all = all_rounds[all_rounds["arm"] == candidate]

        seed_positive_recall = (
            candidate_attack.groupby("model_seed")["malicious_recall"]
            .mean()
            .gt(0)
        )
        candidate_rows.append(
            {
                "candidate": candidate,
                "maximum_benign_fpr": float(
                    candidate_all["benign_false_positive_rate"].max()
                ),
                "fpr_ceiling": FPR_CEILING,
                "mean_clean_macro_f1_loss": float(
                    clean_candidate["clean_macro_f1_loss"].mean()
                ),
                "maximum_clean_round_macro_f1_loss": float(
                    clean_candidate["clean_macro_f1_loss"].max()
                ),
                "mean_malicious_recall_all_seeds": float(
                    candidate_attack["malicious_recall"].mean()
                ),
                "minimum_malicious_recall_all_seeds": float(
                    candidate_attack["malicious_recall"].min()
                ),
                "mean_relative_asr_reduction": float(
                    candidate_attack["relative_asr_reduction"].mean()
                ),
                "seeds_with_positive_mean_recall": int(seed_positive_recall.sum()),
                "seed_count": len(CONFIRMATORY_SEEDS),
            }
        )
    candidate_summary = pd.DataFrame(candidate_rows)
    candidate_summary.to_csv(
        tables_dir / "task41c3_candidate_confirmatory_summary.csv", index=False
    )

    # Preregistered H1/H2/H3 checks against the C0 hypotheses, evaluated
    # only for the selected candidate D2. D0 is reported alongside as the
    # mandatory reference, not judged against these acceptance criteria.
    d2_row = candidate_summary[
        candidate_summary["candidate"] == "D2_trigger_gradient_alignment"
    ].iloc[0]
    h1_pass = bool(
        d2_row["mean_malicious_recall_all_seeds"] >= 0.50
        and d2_row["maximum_benign_fpr"] <= 0.05
    )
    h3_pass = bool(
        d2_row["mean_clean_macro_f1_loss"] <= MEAN_CLEAN_F1_LOSS_CEILING
        and d2_row["maximum_clean_round_macro_f1_loss"]
        <= MAX_ROUND_CLEAN_F1_LOSS_CEILING
    )
    # H2 requires per-trigger-family checks; computed at the condition level.
    condition_level = (
        paired[paired["candidate"] == "D2_trigger_gradient_alignment"]
        .groupby("condition_id")
        .agg(
            mean_relative_asr_reduction=("relative_asr_reduction", "mean"),
        )
    )
    trigger_family_pass = {}
    for slot in TRIGGER_SLOTS:
        slot_conditions = [
            cid for cid in condition_level.index if slot in cid
        ]
        if not slot_conditions:
            continue
        trigger_family_pass[slot] = bool(
            (condition_level.loc[slot_conditions, "mean_relative_asr_reduction"] >= 0.25).all()
        )
    h2_pass = bool(trigger_family_pass) and all(trigger_family_pass.values())

    decision = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": "task41c_c3_confirmatory_multiseed_A1",
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "candidate_evaluated": "D2_trigger_gradient_alignment",
        "mandatory_reference": "D0_frozen_task40_lfighter",
        "a2_executed": False,
        "adaptive_attack_executed": False,
        "reserved_test_accessed": False,
        "h1_detection_pass": h1_pass,
        "h2_mitigation_pass_by_trigger_family": trigger_family_pass,
        "h2_mitigation_pass": h2_pass,
        "h3_clean_utility_pass": h3_pass,
        "confirmatory_status": (
            "PROCEED_TO_C4_ADAPTIVE" if (h1_pass and h2_pass and h3_pass)
            else "CONFIRMATORY_FAILURE_RECORD_AND_STOP"
        ),
        "next_stage": (
            "Freeze C3 evidence. Proceed to C4 adaptive multiseed evaluation "
            "(A3/A4) for D2, per preregistration."
            if (h1_pass and h2_pass and h3_pass)
            else "Record Task 41C confirmatory failure for D2. Do not retune "
            "or select a replacement candidate inside this preregistered task."
        ),
    }
    (summary_dir / "task41c3_confirmatory_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    for candidate, group in paired.groupby("candidate"):
        per_seed = group.groupby("model_seed")["relative_asr_reduction"].mean()
        axis.plot(
            per_seed.index.astype(str), per_seed.values, marker="o", label=candidate
        )
    axis.axhline(0.0, linestyle="--")
    axis.set_xlabel("Model seed")
    axis.set_ylabel("Mean relative ASR reduction")
    axis.set_title("Task 41C C3 confirmatory ASR reduction by seed")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, figures_dir / "task41c3_asr_reduction_by_seed")

    write_output_manifest(output_root)
    return decision


def main() -> int:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined.")
    if args.probe_per_class != 48 or args.probe_seed != 3701:
        raise ValueError("Frozen probe configuration mismatch.")

    torch.set_num_threads(args.threads)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_root)
        elif not args.resume:
            raise FileExistsError(f"Output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_seed_manifest(args.seed_manifest)

    if args.preflight_only:
        results = []
        for model_seed in CONFIRMATORY_SEEDS:
            print(f"PREFLIGHT seed {model_seed}...")
            result = preflight_seed(
                model_seed=model_seed,
                seed_paths=manifest[model_seed],
                args=args,
            )
            results.append(result)
            print(f"  OK: partition={result['partition_hash_sha256'][:12]} "
                  f"warmup={result['warmup_checkpoint_sha256'][:12]} "
                  f"detector={result['detector_profile_sha256'][:12]}")
        decision_path = output_root / "task41c3_preflight_decision.json"
        decision_path.write_text(
            json.dumps(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "stage": "task41c_c3_preflight",
                    "preflight_passed": True,
                    "seeds": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print()
        print("ALL FOUR SEEDS PREFLIGHT PASSED.")
        print("Decision file:", decision_path)
        return 0

    started = time.time()
    all_frames: List[pd.DataFrame] = []
    for model_seed in CONFIRMATORY_SEEDS:
        frame = run_seed(
            model_seed=model_seed,
            seed_paths=manifest[model_seed],
            args=args,
            output_root=output_root,
        )
        all_frames.append(frame)

    all_rounds = pd.concat(all_frames, ignore_index=True)
    decision = summarize_confirmatory(output_root, all_rounds)
    decision["total_seconds"] = float(time.time() - started)

    print()
    print("=" * 100)
    print("TASK 41C.3 CONFIRMATORY MULTISEED SCREEN (A1 ONLY)")
    print("=" * 100)
    print("Confirmatory status:", decision["confirmatory_status"])
    print("H1 detection pass:", decision["h1_detection_pass"])
    print("H2 mitigation pass:", decision["h2_mitigation_pass"])
    print("H3 clean utility pass:", decision["h3_clean_utility_pass"])
    print("Next stage:", decision["next_stage"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

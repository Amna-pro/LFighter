# v4.34.0e PRE-OUTCOME SETUP CORRECTION\n# Parent runner: v4.34.0c SHA256 D2EB3C8CD2803A2807705C317528830897DBEAC5901747C70356DB4D54A55D7B\n# Scientific settings unchanged; only inherited calibration-dir CLI guard removed.\n# REVIEWER-DERIVED FILE v4.34.0c
# Derived from frozen V3.20B.1 SHA256 5F3852FC13959301B31ADF47F57DF3B65456FB028A647978B355F976E5AEE951
# No reconstruction-calibration bundle is used because neither ablation requires historical residuals.
#!/usr/bin/env python3
"""Reviewer reconstruction mitigation ablation runner V4.34.1."""
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
from typing import Dict, List, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from federated_iot_v26 import (
    CLASS_NAMES,
    NUM_CLASSES,
    load_protocol_arrays,
    set_seed,
    sqrt_class_weights,
    train_local_model,
)
from neural_models_v24 import build_model
from run_targeted_label_flip_v292 import (
    load_fixed_partitions,
    prepare_static_attack,
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
    CANDIDATE,
    SCORE_COLUMN,
    add_candidate_and_ema,
    apply_feature_calibration,
    file_sha256,
    load_profiles_npz,
    normalize_rows,
    quantile_higher,
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

DEFAULT_MALICIOUS_CLIENTS = "1,7,8,10,14,15,17,18"
FROZEN_EMA_QUANTILE = 0.99
FROZEN_INSTANT_QUANTILE = 0.95
FROZEN_TRUST_GAMMA = 3.0
FROZEN_MINIMUM_TRUST = 0.05

ATTACK_TYPES = (
    "random_flip",
    "cyclic_shift",
    "pairwise_swap",
    "all_to_one_benign",
    "multiclass_partial_cycle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reviewer reconstruction mitigation ablation V4.34.1.")
    parser.add_argument("--mode", choices=["clean", "strong_attack"], required=True)
    parser.add_argument(
        "--replacement-policy",
        choices=["plain_fedavg", "trusted_reconstruction"],
        required=True,
    )
    parser.add_argument(
        "--ablation-arm",
        choices=['center_plus_residual', 'center_only', 'hard_rejection', 'down_weighting'],
        required=True,
    )
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-seed-dir", required=True, type=Path)
    parser.add_argument("--warmup-dir", required=True, type=Path)
    parser.add_argument(
        "--reconstruction-calibration-dir",
        type=Path,
    )
    parser.add_argument(
        "--plain-branch-dir",
        type=Path,
        help="Exact V3.20A.3 plain branch supplying the frozen poison plan.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--continuation-rounds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--probe-per-class", type=int, default=48)
    parser.add_argument("--probe-seed", type=int, default=3701)
    parser.add_argument(
        "--attack-type",
        choices=ATTACK_TYPES,
        required=True,
    )
    # Fixed diagnostic pair for the frozen detector feature computation.
    # Untargeted utility is measured with macro-F1, balanced accuracy,
    # per-class recall, and the full confusion matrix.
    parser.add_argument("--source-class", default="DDoS")
    parser.add_argument("--target-class", default="Benign")
    parser.add_argument("--ema-decay", type=float, default=0.65)
    parser.add_argument("--malicious-clients", default=DEFAULT_MALICIOUS_CLIENTS)
    parser.add_argument("--poison-fraction", type=float, default=1.0)
    parser.add_argument("--min-source-samples", type=int, default=1000)
    parser.add_argument("--attack-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_client_ids(text: str) -> List[int]:
    return sorted({int(value.strip()) for value in text.split(",") if value.strip()})



def stable_attack_seed(
    base_seed: int,
    client_id: int,
    attack_type: str,
) -> int:
    digest = hashlib.sha256(
        f"{base_seed}|{client_id}|{attack_type}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def mapped_labels(
    labels: np.ndarray,
    attack_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    result = labels.copy()

    if attack_type == "random_flip":
        offsets = rng.integers(1, NUM_CLASSES, size=len(labels))
        result = (labels + offsets) % NUM_CLASSES
    elif attack_type == "cyclic_shift":
        result = (labels + 1) % NUM_CLASSES
    elif attack_type == "pairwise_swap":
        result = labels ^ 1
    elif attack_type == "all_to_one_benign":
        result[:] = CLASS_NAMES.index("Benign")
    elif attack_type == "multiclass_partial_cycle":
        mapping = {
            CLASS_NAMES.index("BruteForce"):
                CLASS_NAMES.index("DDoS"),
            CLASS_NAMES.index("DDoS"):
                CLASS_NAMES.index("Mirai"),
            CLASS_NAMES.index("Mirai"):
                CLASS_NAMES.index("Web-Based"),
            CLASS_NAMES.index("Web-Based"):
                CLASS_NAMES.index("BruteForce"),
        }
        for source_id, target_id in mapping.items():
            result[labels == source_id] = target_id
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    return result.astype(np.int64, copy=False)


def prepare_untargeted_attack(
    *,
    client_indices: List[np.ndarray],
    y_train: np.ndarray,
    malicious_clients: List[int],
    poison_fraction: float,
    attack_seed: int,
    attack_type: str,
) -> tuple[
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
    pd.DataFrame,
    str,
]:
    if not (0.0 < float(poison_fraction) <= 1.0):
        raise ValueError("poison_fraction must be in (0, 1]")

    malicious_set = set(int(x) for x in malicious_clients)
    poisoned_positions: Dict[int, np.ndarray] = {}
    poisoned_labels: Dict[int, np.ndarray] = {}
    rows: List[Dict[str, object]] = []
    digest = hashlib.sha256()

    for client_id, indices in enumerate(client_indices):
        clean_labels = y_train[indices].astype(np.int64, copy=False)
        rng = np.random.default_rng(
            stable_attack_seed(
                attack_seed,
                client_id,
                attack_type,
            )
        )
        mapped = mapped_labels(clean_labels, attack_type, rng)
        eligible = np.flatnonzero(
            mapped != clean_labels
        ).astype(np.int64)

        if client_id in malicious_set and len(eligible):
            count = int(np.ceil(poison_fraction * len(eligible)))
            count = min(max(count, 1), len(eligible))
            if count == len(eligible):
                selected = eligible.copy()
            else:
                selected = np.sort(
                    rng.choice(
                        eligible,
                        size=count,
                        replace=False,
                    )
                ).astype(np.int64)
            replacements = mapped[selected].astype(
                np.int64,
                copy=True,
            )
        else:
            selected = np.empty(0, dtype=np.int64)
            replacements = np.empty(0, dtype=np.int64)

        poisoned_positions[client_id] = selected
        poisoned_labels[client_id] = replacements
        digest.update(f"client_{client_id:03d}".encode("utf-8"))
        digest.update(selected.tobytes())
        digest.update(replacements.tobytes())

        changed = (
            int(
                np.sum(
                    clean_labels[selected] != replacements
                )
            )
            if len(selected)
            else 0
        )
        rows.append({
            "client_id": client_id,
            "is_malicious": bool(client_id in malicious_set),
            "client_rows": int(len(indices)),
            "eligible_rows": int(len(eligible)),
            "poisoned_rows": int(len(selected)),
            "changed_rows_verified": changed,
            "client_poison_rate_over_eligible": (
                float(len(selected) / len(eligible))
                if len(eligible)
                else 0.0
            ),
            "client_poison_rate_over_all_rows": float(
                len(selected) / max(len(indices), 1)
            ),
            "attack_type": attack_type,
        })

    manifest = pd.DataFrame(rows)
    malicious = manifest[
        manifest["is_malicious"].astype(bool)
    ]
    if int(malicious["poisoned_rows"].sum()) <= 0:
        raise RuntimeError(
            "Untargeted attack changed no malicious-client rows"
        )
    if not np.array_equal(
        manifest["poisoned_rows"].to_numpy(dtype=int),
        manifest["changed_rows_verified"].to_numpy(dtype=int),
    ):
        raise RuntimeError(
            "At least one selected label did not change"
        )

    return (
        poisoned_positions,
        poisoned_labels,
        manifest,
        digest.hexdigest(),
    )


def save_poisoned_labels(
    poisoned_labels: Dict[int, np.ndarray],
    path: Path,
) -> None:
    np.savez_compressed(
        path,
        **{
            f"client_{client_id:03d}":
                labels.astype(np.int64)
            for client_id, labels
            in poisoned_labels.items()
        },
    )



def _npz_client_array(
    archive: np.lib.npyio.NpzFile,
    client_id: int,
    suffix: str = "",
) -> np.ndarray:
    keys = []
    if suffix:
        keys.extend(
            (
                f"client_{client_id:03d}_{suffix}",
                f"client_{client_id}_{suffix}",
            )
        )
    keys.extend(
        (
            f"client_{client_id:03d}",
            f"client_{client_id}",
            str(client_id),
        )
    )
    for key in keys:
        if key in archive.files:
            return np.asarray(archive[key])
    raise KeyError(
        f"No NPZ array for client {client_id}"
        f" with suffix {suffix!r}; keys={archive.files}"
    )

def load_exact_v320a3_poison_plan(
    *,
    plain_branch_dir: Path,
    client_indices: List[np.ndarray],
    attack_type: str,
    model_seed: int,
    expected_malicious_clients: List[int],
) -> tuple[
    List[int],
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
    pd.DataFrame,
    str,
]:
    branch = plain_branch_dir.expanduser().resolve()
    metadata_path = (
        branch / "exact_untargeted_plain_v320a3_metadata.json"
    )
    manifest_dir = branch / "attack_manifest"
    manifest_path = (
        manifest_dir / "malicious_client_poison_manifest.csv"
    )
    indices_path = manifest_dir / "poisoned_indices.npz"
    labels_path = manifest_dir / "poisoned_labels.npz"

    for path in (
        metadata_path,
        manifest_path,
        indices_path,
        labels_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if metadata.get("attack_type") != attack_type:
        raise RuntimeError(
            f"Attack mismatch in exact plain branch: "
            f"{metadata.get('attack_type')} != {attack_type}"
        )
    if int(metadata.get("model_seed")) != int(model_seed):
        raise RuntimeError(
            f"Seed mismatch in exact plain branch: "
            f"{metadata.get('model_seed')} != {model_seed}"
        )
    if metadata.get("mode") != "strong_attack":
        raise RuntimeError("Supplied exact plain branch is not strong_attack")
    if bool(metadata.get("test_sets_accessed")):
        raise RuntimeError("Supplied exact plain branch reports test access")

    malicious_clients = [
        int(value) for value in metadata.get("malicious_clients", [])
    ]
    if malicious_clients != expected_malicious_clients:
        raise RuntimeError(
            f"Coalition mismatch: {malicious_clients} "
            f"!= {expected_malicious_clients}"
        )

    positions: Dict[int, np.ndarray] = {}
    labels: Dict[int, np.ndarray] = {}
    with np.load(indices_path, allow_pickle=False) as index_npz, np.load(
        labels_path, allow_pickle=False
    ) as label_npz:
        for client_id, partition in enumerate(client_indices):
            global_indices = _npz_client_array(
                index_npz, client_id, "global_indices"
            ).astype(np.int64, copy=False)
            replacements = _npz_client_array(
                label_npz, client_id
            ).astype(np.int64, copy=False)
            if len(global_indices) != len(replacements):
                raise RuntimeError(
                    f"Poison plan length mismatch for client {client_id}"
                )
            lookup = {
                int(global_index): int(local_position)
                for local_position, global_index in enumerate(partition)
            }
            try:
                local_positions = np.asarray(
                    [lookup[int(index)] for index in global_indices],
                    dtype=np.int64,
                )
            except KeyError as exc:
                raise RuntimeError(
                    f"Poisoned global row is outside client {client_id}"
                ) from exc
            if len(local_positions) and not np.array_equal(
                np.asarray(partition)[local_positions],
                global_indices,
            ):
                raise RuntimeError(
                    f"Global-to-local poison mapping failed for client {client_id}"
                )
            positions[client_id] = local_positions
            labels[client_id] = replacements.copy()

    manifest = pd.read_csv(manifest_path)
    malicious_manifest = manifest[
        manifest["is_malicious"].astype(bool)
    ]
    if len(malicious_manifest) != 8:
        raise RuntimeError("Expected eight malicious manifest rows")
    if int(malicious_manifest["changed_rows_verified"].sum()) <= 0:
        raise RuntimeError("Exact poisoning plan changes no labels")
    if not (
        malicious_manifest["poisoned_rows"].astype(int)
        == malicious_manifest["changed_rows_verified"].astype(int)
    ).all():
        raise RuntimeError("Exact poisoning manifest is inconsistent")

    return (
        malicious_clients,
        positions,
        labels,
        manifest,
        str(metadata["poison_index_hash_sha256"]),
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def save_partial(
    tables_dir: Path,
    round_rows: List[Dict[str, object]],
    local_rows: List[Dict[str, object]],
    score_rows: List[Dict[str, object]],
    signature_rows: List[Dict[str, object]],
) -> None:
    pd.DataFrame(round_rows).to_csv(tables_dir / "continuation_round_metrics_partial.csv", index=False)
    pd.DataFrame(local_rows).to_csv(tables_dir / "continuation_local_training_partial.csv", index=False)
    pd.DataFrame(score_rows).to_csv(tables_dir / "continuation_client_anchor_scores_partial.csv", index=False)
    pd.DataFrame(signature_rows).to_csv(tables_dir / "continuation_transition_signature_long_partial.csv", index=False)


def main() -> int:
    args = parse_args()
    if args.continuation_rounds != 4:
        raise ValueError("V3.10 development chronology is frozen to four continuation rounds")
    if args.ema_decay != 0.65:
        raise ValueError("V3.10 is frozen to EMA=0.65")
    if args.source_class not in CLASS_NAMES or args.target_class not in CLASS_NAMES:
        raise ValueError("Unknown source or target class")
    if args.attack_type not in ATTACK_TYPES:
        raise ValueError(
            f"Unknown untargeted attack type: {args.attack_type}"
        )
    if args.source_class == args.target_class:
        raise ValueError("Source and target classes must differ")
    if args.mode == "clean" and args.replacement_policy != "plain_fedavg":
        raise ValueError("Clean adapter must use plain_fedavg")
    if args.mode == "strong_attack" and args.plain_branch_dir is None:
        raise ValueError(
            "--plain-branch-dir is required for strong_attack"
        )
    if (
        args.replacement_policy == 'trusted_reconstruction'
        and args.ablation_arm == 'center_plus_residual'
        and args.reconstruction_calibration_dir is None
    ):
        raise ValueError('--reconstruction-calibration-dir is required for center_plus_residual')
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")

    torch.set_num_threads(max(1, args.threads))
    set_seed(args.model_seed)
    source_id = CLASS_NAMES.index(args.source_class)
    target_id = CLASS_NAMES.index(args.target_class)

    warmup_dir = args.warmup_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"
    attack_dir = output_dir / "attack_manifest"
    for path in (tables_dir, figures_dir, checkpoints_dir, attack_dir):
        path.mkdir(parents=True, exist_ok=True)

    warmup_metadata_path = warmup_dir / "true_warmup_v310_metadata.json"
    branch_checkpoint_path = warmup_dir / "checkpoints" / "common_round4_warmup_model.pt"
    profiles_path = warmup_dir / "calibration" / "trusted_client_profiles.npz"
    feature_path = warmup_dir / "calibration" / "feature_calibration.csv"
    summary_path = warmup_dir / "calibration" / "calibration_summary.csv"
    for path in (warmup_metadata_path, branch_checkpoint_path, profiles_path, feature_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"Warmup artifact is missing: {path}")

    with warmup_metadata_path.open("r", encoding="utf-8") as handle:
        warmup_metadata = json.load(handle)
    calibration_summary = pd.read_csv(summary_path).iloc[0]
    if int(warmup_metadata["warmup_rounds"]) != 4:
        raise RuntimeError("Warmup metadata is not a four-round V3.10 branch point")
    if int(warmup_metadata["model_seed"]) != int(args.model_seed):
        raise RuntimeError("Model seed does not match warmup branch point")
    if not bool(warmup_metadata.get("monitoring_ema_reset_after_warmup", False)):
        raise RuntimeError("Warmup metadata does not freeze the monitoring EMA reset")

    arrays = load_protocol_arrays(args.data_file.expanduser().resolve())
    X_train = arrays["X_train"].astype(np.float32, copy=False)
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    X_val = arrays["X_val"].astype(np.float32, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)

    client_indices, partition_hash = load_fixed_partitions(
        args.partition_file.expanduser().resolve(),
        expected_clients=args.num_clients,
        train_rows=len(y_train),
    )
    clean_hash = read_clean_partition_hash(args.clean_seed_dir.expanduser().resolve())
    if partition_hash != clean_hash or partition_hash != warmup_metadata["partition_hash_sha256"]:
        raise RuntimeError("Partition hash does not match warmup branch point")

    probe_indices = balanced_probe_indices(y_val, args.probe_per_class, args.probe_seed)
    probe_hash = hashlib.sha256(np.ascontiguousarray(probe_indices).tobytes()).hexdigest()
    if probe_hash != warmup_metadata["probe_hash_sha256"]:
        raise RuntimeError("Probe hash does not match warmup branch point")
    X_probe = X_val[probe_indices]
    y_probe = y_val[probe_indices]

    if file_sha256(profiles_path) != warmup_metadata["profile_sha256"]:
        raise RuntimeError("Trusted profile hash mismatch")
    if file_sha256(feature_path) != warmup_metadata["feature_calibration_sha256"]:
        raise RuntimeError("Feature calibration hash mismatch")
    profiles = load_profiles_npz(profiles_path, args.num_clients)
    feature_calibration = pd.read_csv(feature_path)
    threshold = float(calibration_summary["clean_ema_threshold"])

    checkpoint = torch.load(branch_checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("partition_hash") != partition_hash:
        raise RuntimeError("Warmup checkpoint partition mismatch")
    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.mode == "strong_attack":
        requested_clients = parse_client_ids(
            args.malicious_clients
        )
        if len(requested_clients) != 8:
            raise ValueError(
                "V4.34.1 reviewer ablation is frozen to eight malicious clients"
            )
        (
            malicious_clients,
            poisoned_positions,
            poisoned_labels,
            poison_manifest,
            poison_hash,
        ) = load_exact_v320a3_poison_plan(
            plain_branch_dir=args.plain_branch_dir,
            client_indices=client_indices,
            attack_type=args.attack_type,
            model_seed=args.model_seed,
            expected_malicious_clients=requested_clients,
        )
    else:
        malicious_clients = []
        poisoned_positions = {
            client_id: np.empty(0, dtype=np.int64)
            for client_id in range(args.num_clients)
        }
        poisoned_labels = {
            client_id: np.empty(0, dtype=np.int64)
            for client_id in range(args.num_clients)
        }
        poison_manifest = pd.DataFrame(
            [
                {
                    "client_id": client_id,
                    "is_malicious": False,
                    "client_rows": int(len(indices)),
                    "eligible_rows": 0,
                    "poisoned_rows": 0,
                    "changed_rows_verified": 0,
                    "client_poison_rate_over_eligible": 0.0,
                    "client_poison_rate_over_all_rows": 0.0,
                    "attack_type": args.attack_type,
                }
                for client_id, indices
                in enumerate(client_indices)
            ]
        )
        digest = hashlib.sha256()
        for client_id in range(args.num_clients):
            digest.update(
                f"client_{client_id:03d}".encode("utf-8")
            )
        poison_hash = digest.hexdigest()

    poison_manifest.to_csv(
        attack_dir / "malicious_client_poison_manifest.csv",
        index=False,
    )
    if args.mode == "strong_attack":
        source_manifest = (
            args.plain_branch_dir.expanduser().resolve()
            / "attack_manifest"
        )
        shutil.copy2(
            source_manifest / "poisoned_indices.npz",
            attack_dir / "poisoned_indices.npz",
        )
        shutil.copy2(
            source_manifest / "poisoned_labels.npz",
            attack_dir / "poisoned_labels.npz",
        )
    else:
        save_poisoned_indices(
            client_indices,
            poisoned_positions,
            attack_dir / "poisoned_indices.npz",
        )
        save_poisoned_labels(
            poisoned_labels,
            attack_dir / "poisoned_labels.npz",
        )
    pd.DataFrame(
        [
            {
                "mode": args.mode,
            "replacement_policy": args.replacement_policy,
                "attack_type": args.attack_type,
                "diagnostic_source_class": args.source_class,
                "diagnostic_target_class": args.target_class,
                "malicious_clients": "|".join(map(str, malicious_clients)),
                "partition_hash_sha256": partition_hash,
                "poison_index_hash_sha256": poison_hash,
            }
        ]
    ).to_csv(attack_dir / "attack_summary.csv", index=False)

    ema_threshold_q99 = float(threshold)
    instant_threshold_q95 = float("nan")
    selected_reconstruction_policy = ""
    residual_profiles: Dict[int, Dict[str, torch.Tensor]] = {}
    warmup_update_scale = float("nan")
    scale_lower = float("nan")
    scale_upper = float("nan")
    norm_clip_multiplier = float("nan")
    reconstruction_checkpoint_path: Path | None = None

    mitigation_action = "plain_fedavg_no_mitigation"

    if args.replacement_policy == "trusted_reconstruction":
        warmup_scores_path = (
            warmup_dir / "calibration"
            / "clean_leave_one_round_out_scores.csv"
        )
        if not warmup_scores_path.exists():
            raise FileNotFoundError(warmup_scores_path)
        warmup_scores = pd.read_csv(warmup_scores_path)
        ema_threshold_q99 = quantile_higher(
            warmup_scores[SCORE_COLUMN].to_numpy(dtype=np.float64),
            FROZEN_EMA_QUANTILE,
        )
        instant_threshold_q95 = quantile_higher(
            warmup_scores[CANDIDATE].to_numpy(dtype=np.float64),
            FROZEN_INSTANT_QUANTILE,
        )

        if args.ablation_arm == 'center_plus_residual':
            calibration_dir = (
                args.reconstruction_calibration_dir.expanduser().resolve()
            )
            reconstruction_checkpoint_path = (
                calibration_dir / "calibration"
                / "trusted_update_reconstruction_profiles.pt"
            )
            reconstruction_summary_path = (
                calibration_dir / "calibration"
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
            if int(reconstruction_bundle["model_seed"]) != int(
                args.model_seed
            ):
                raise RuntimeError(
                    "Reconstruction calibration model seed mismatch"
                )
            if reconstruction_bundle["partition_hash"] != partition_hash:
                raise RuntimeError(
                    "Reconstruction calibration partition mismatch"
                )
            if (
                reconstruction_bundle[
                    "common_warmup_checkpoint_sha256"
                ]
                != checkpoint_sha256(branch_checkpoint_path)
            ):
                raise RuntimeError(
                    "Reconstruction calibration checkpoint mismatch"
                )
    
            residual_profiles = {
                int(client_id): {
                    key: value.detach().cpu().to(torch.float32)
                    for key, value in profile.items()
                }
                for client_id, profile in reconstruction_bundle[
                    "client_residual_profiles"
                ].items()
            }
            selected_reconstruction_policy = str(
                reconstruction_bundle["selected_policy"]
            )
            if selected_reconstruction_policy != "center_plus_residual":
                raise RuntimeError(
                    "V4.34.1 center_plus_residual arm requires reconstruction calibration"
                )
            warmup_update_scale = float(
                reconstruction_bundle["warmup_update_scale"]
            )
            scale_lower = float(reconstruction_bundle["scale_lower"])
            scale_upper = float(reconstruction_bundle["scale_upper"])
            norm_clip_multiplier = float(
                reconstruction_bundle["norm_clip_multiplier"]
            )
    
            calibration_summary = pd.read_csv(reconstruction_summary_path)
            if len(calibration_summary) != 1:
                raise RuntimeError('Expected exactly one reconstruction calibration summary row')
            if str(calibration_summary.iloc[0]['test_sets_accessed']).strip().lower() != 'false':
                raise RuntimeError('Reconstruction calibration reports test-set access')
            if str(calibration_summary.iloc[0]['malicious_labels_used']).strip().lower() != 'false':
                raise RuntimeError('Reconstruction calibration used malicious labels')
        elif args.ablation_arm == 'center_only':
            selected_reconstruction_policy = 'center_only'
            residual_profiles = {client_id: {} for client_id in range(args.num_clients)}
        elif args.ablation_arm == 'hard_rejection':
            selected_reconstruction_policy = 'hard_rejection'
            residual_profiles = {client_id: {} for client_id in range(args.num_clients)}
        elif args.ablation_arm == 'down_weighting':
            selected_reconstruction_policy = 'down_weighting'
            residual_profiles = {client_id: {} for client_id in range(args.num_clients)}
        else:
            raise RuntimeError(f'Unsupported ablation arm: {args.ablation_arm}')

        mitigation_action = ('plain_fedavg_no_mitigation' if args.replacement_policy == 'plain_fedavg' else {'center_plus_residual': 'coordinate_median_center_plus_historical_residual_reconstruction', 'center_only': 'coordinate_median_center_reconstruction', 'hard_rejection': 'hard_rejection_and_weight_renormalization', 'down_weighting': 'raw_sample_count_times_frozen_soft_trust'}[args.ablation_arm])

    class_weights = sqrt_class_weights(y_train, args.max_class_weight)
    round_rows: List[Dict[str, object]] = []
    local_rows: List[Dict[str, object]] = []
    score_rows: List[Dict[str, object]] = []
    reconstruction_rows: List[Dict[str, object]] = []
    signature_rows: List[Dict[str, object]] = []
    class_metric_rows: List[Dict[str, object]] = []
    confusion_rows: List[Dict[str, object]] = []
    ema_memory: Dict[int, float] = {}
    start_round = 1
    continuation_checkpoint = checkpoints_dir / "v4341_last_round_model.pt"

    if args.resume:
        if not continuation_checkpoint.exists():
            raise FileNotFoundError("Resume requested but continuation checkpoint is missing")
        saved = torch.load(continuation_checkpoint, map_location="cpu", weights_only=False)
        for key, expected in {
            "mode": args.mode,
            "replacement_policy": args.replacement_policy,
            "ablation_arm": args.ablation_arm,
            "partition_hash": partition_hash,
            "poison_index_hash": poison_hash,
            "warmup_profile_hash": warmup_metadata["profile_sha256"],
            "model_seed": int(args.model_seed),
        }.items():
            if saved.get(key) != expected:
                raise RuntimeError(f"Continuation resume mismatch for {key}")
        model.load_state_dict(saved["model_state_dict"])
        saved_rng = saved.get("next_round_torch_rng_state")
        if saved_rng is not None:
            torch.random.set_rng_state(saved_rng)
        ema_memory = {
            int(k): float(v)
            for k, v in saved.get("ema_memory", {}).items()
        }
        start_round = int(saved["monitoring_round"]) + 1
        round_rows = load_rows(tables_dir / "continuation_round_metrics_partial.csv")
        local_rows = load_rows(tables_dir / "continuation_local_training_partial.csv")
        score_rows = load_rows(tables_dir / "continuation_client_anchor_scores_partial.csv")
        signature_rows = load_rows(tables_dir / "continuation_transition_signature_long_partial.csv")
        print(f"Resuming {args.mode} continuation from monitoring round {start_round}")

    started = time.time()
    print("Reviewer Reconstruction Ablation V4.34.1")
    print("Mode:", args.mode)
    print("Replacement policy:", args.replacement_policy)
    print("Attack type:", args.attack_type)
    print("Common warmup checkpoint:", branch_checkpoint_path)
    print("Frozen EMA threshold:", f"{ema_threshold_q99:.6f}")
    if args.replacement_policy == "trusted_reconstruction":
        print(
            "Frozen instant threshold:",
            f"{instant_threshold_q95:.6f}",
        )
        print(
            "Reconstruction policy:",
            selected_reconstruction_policy,
        )
    print("EMA reset at monitoring start: True")
    print("Malicious clients:", malicious_clients)

    for monitoring_round in range(start_round, args.continuation_rounds + 1):
        global_round = 4 + monitoring_round
        round_started = time.time()
        reference_state = copy.deepcopy(model.state_dict())
        local_states: List[Mapping[str, torch.Tensor]] = []
        sample_counts: List[int] = []
        local_matrices: List[np.ndarray] = []
        current_training = []

        for client_id in range(args.num_clients):
            indices = client_indices[client_id]
            local_y = y_train[indices].copy()
            positions = poisoned_positions[client_id]
            if len(positions) > 0:
                replacements = poisoned_labels[client_id]
                if len(replacements) != len(positions):
                    raise RuntimeError(
                        f"Poison plan mismatch for client {client_id}"
                    )
                local_y[positions] = replacements
            local_model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
            local_model.load_state_dict(reference_state)
            state, metrics = train_local_model(
                model=local_model,
                X=X_train[indices],
                y=local_y,
                class_weights=class_weights,
                local_epochs=1,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                seed=args.model_seed + global_round * 1000 + client_id,
            )
            probabilities = predict_probabilities(local_model, X_probe, args.evaluation_batch_size)
            local_means = class_conditional_probability_means(probabilities, y_probe)
            local_states.append(state)
            sample_counts.append(int(len(indices)))
            local_matrices.append(local_means)
            row = {
                "monitoring_round": int(monitoring_round),
                "global_round": int(global_round),
                "client_id": int(client_id),
                "actual_malicious": bool(client_id in malicious_clients),
                "client_samples": int(len(indices)),
                "poisoned_rows": int(len(positions)),
                **metrics,
            }
            local_rows.append(row)
            current_training.append(row)
            del local_model

        next_round_torch_rng_state = (
            torch.random.get_rng_state().clone()
        )

        matrix_stack = np.stack(local_matrices, axis=0)
        consensus = normalize_rows(np.median(matrix_stack, axis=0))
        current_raw = []
        raw_weights = np.asarray(sample_counts, dtype=np.float64) / float(np.sum(sample_counts))
        for client_id, local_matrix in enumerate(local_matrices):
            local_matrix = normalize_rows(local_matrix)
            for source_index in range(NUM_CLASSES):
                for target_index in range(NUM_CLASSES):
                    signature_rows.append({
                        "monitoring_round": int(monitoring_round),
                        "global_round": int(global_round),
                        "client_id": int(client_id),
                        "actual_malicious": bool(client_id in malicious_clients),
                        "source_id": int(source_index),
                        "source_name": CLASS_NAMES[source_index],
                        "target_id": int(target_index),
                        "target_name": CLASS_NAMES[target_index],
                        "local_probability": float(local_matrix[source_index, target_index]),
                        "round_consensus_probability": float(consensus[source_index, target_index]),
                        "frozen_profile_probability": float(profiles[client_id][source_index, target_index]),
                    })
            current_raw.append(
                {
                    "monitoring_round": int(monitoring_round),
                    "global_round": int(global_round),
                    "client_id": int(client_id),
                    "actual_malicious": bool(client_id in malicious_clients),
                    "client_samples": int(sample_counts[client_id]),
                    "aggregation_weight": float(raw_weights[client_id]),
                    "poisoned_rows": int(len(poisoned_positions[client_id])),
                    **raw_features(
                        local_matrix,
                        profiles[client_id],
                        consensus,
                        source_id,
                        target_id,
                    ),
                }
            )

        current_scored = apply_feature_calibration(pd.DataFrame(current_raw), feature_calibration)
        current_scored, ema_memory = add_candidate_and_ema(
            current_scored, args.ema_decay, initial_ema=ema_memory
        )
        replacement_states: List[Mapping[str, torch.Tensor]] = list(
            local_states
        )
        replaced_clients = 0

        if args.replacement_policy == "plain_fedavg":
            current_scored["clean_ema_threshold"] = threshold
            current_scored["instant_threshold"] = float("nan")
            current_scored["ema_ratio"] = (
                current_scored[SCORE_COLUMN].to_numpy(dtype=np.float64)
                / max(float(threshold), 1e-12)
            )
            current_scored["instant_ratio"] = float("nan")
            current_scored["policy_ratio"] = current_scored[
                "ema_ratio"
            ]
            current_scored["flagged"] = (
                current_scored[SCORE_COLUMN] > threshold
            )
        else:
            current_scored["clean_ema_threshold"] = (
                ema_threshold_q99
            )
            current_scored["instant_threshold"] = (
                instant_threshold_q95
            )
            current_scored["ema_ratio"] = (
                current_scored[SCORE_COLUMN].to_numpy(dtype=np.float64)
                / max(ema_threshold_q99, 1e-12)
            )
            current_scored["instant_ratio"] = (
                current_scored[CANDIDATE].to_numpy(dtype=np.float64)
                / max(instant_threshold_q95, 1e-12)
            )
            current_scored["policy_ratio"] = np.maximum(
                current_scored["ema_ratio"].to_numpy(
                    dtype=np.float64
                ),
                current_scored["instant_ratio"].to_numpy(
                    dtype=np.float64
                ),
            )
            current_scored["flagged"] = (
                current_scored["policy_ratio"] > 1.0
            )

            flags_for_reconstruction = current_scored[
                "flagged"
            ].astype(bool).to_numpy()
            updates = [
                floating_update(state, reference_state)
                for state in local_states
            ]
            trusted_ids = [
                client_id
                for client_id in range(args.num_clients)
                if not flags_for_reconstruction[client_id]
            ]
            if len(trusted_ids) < 8:
                raise RuntimeError(
                    f"Only {len(trusted_ids)} trusted clients remain"
                )
            trusted_updates = [
                updates[client_id] for client_id in trusted_ids
            ]
            trusted_norms = [
                update_norm(update) for update in trusted_updates
            ]
            current_center = coordinate_median(trusted_updates)
            current_update_scale = float(np.median(trusted_norms))

            for client_id in range(args.num_clients):
                flagged_for_mitigation = bool(
                    flags_for_reconstruction[client_id]
                )
                replaced = bool(
                    flagged_for_mitigation
                    and args.ablation_arm in {'center_plus_residual', 'center_only'}
                )
                rejected = bool(
                    flagged_for_mitigation
                    and args.ablation_arm == "hard_rejection"
                )
                reconstruction_meta = {
                    "residual_scale": float("nan"),
                    "norm_clip_factor": float("nan"),
                    "norm_clip_upper": float("nan"),
                    "reconstructed_update_norm": float("nan"),
                }
                if replaced:
                    reconstructed_update, reconstruction_meta = (
                        reconstruct_update(
                            current_center=current_center,
                            historical_residual=residual_profiles[
                                client_id
                            ],
                            selected_policy=(
                                selected_reconstruction_policy
                            ),
                            current_update_scale=(
                                current_update_scale
                            ),
                            warmup_update_scale=(
                                warmup_update_scale
                            ),
                            trusted_norms=trusted_norms,
                            scale_lower=scale_lower,
                            scale_upper=scale_upper,
                            norm_clip_multiplier=(
                                norm_clip_multiplier
                            ),
                        )
                    )
                    replacement_states[client_id] = (
                        state_from_update(
                            reference_state,
                            reconstructed_update,
                        )
                    )
                    replaced_clients += 1

                reconstruction_rows.append({
                    "monitoring_round": int(monitoring_round),
                    "global_round": int(global_round),
                    "client_id": int(client_id),
                    "actual_malicious": bool(
                        client_id in malicious_clients
                    ),
                    "flagged": bool(
                        flags_for_reconstruction[client_id]
                    ),
                    "update_replaced": replaced,
                    "update_rejected": rejected,
                    "ablation_arm": args.ablation_arm,
                    "selected_reconstruction_policy": (
                        selected_reconstruction_policy
                    ),
                    "actual_update_norm": update_norm(
                        updates[client_id]
                    ),
                    **reconstruction_meta,
                })

        current_scored["policy_exceedance"] = np.maximum(
            current_scored["policy_ratio"].to_numpy(dtype=np.float64) - 1.0,
            0.0,
        )
        current_scored["flagged"] = current_scored["policy_ratio"] > 1.0
        current_scored["trust_factor"] = np.clip(
            np.exp(
                -FROZEN_TRUST_GAMMA
                * current_scored["policy_exceedance"].to_numpy(dtype=np.float64)
            ),
            FROZEN_MINIMUM_TRUST,
            1.0,
        )

        raw_aggregation_weights = current_scored[
            "aggregation_weight"
        ].to_numpy(dtype=np.float64)
        unnormalized_trust_weights = (
            raw_aggregation_weights
            * current_scored["trust_factor"].to_numpy(dtype=np.float64)
        )
        trust_adjusted_weights = (
            unnormalized_trust_weights
            / max(float(unnormalized_trust_weights.sum()), 1e-12)
        )
        current_scored["raw_aggregation_weight"] = raw_aggregation_weights
        current_scored["trust_adjusted_weight"] = trust_adjusted_weights
        if args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm == 'hard_rejection':
            effective_aggregation_weight = np.zeros(args.num_clients, dtype=np.float64)
            retained_counts = np.asarray([sample_counts[client_id] for client_id in trusted_ids], dtype=np.float64)
            effective_aggregation_weight[np.asarray(trusted_ids, dtype=np.int64)] = retained_counts / max(float(retained_counts.sum()), 1e-12)
        elif args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm == 'down_weighting':
            effective_aggregation_weight = current_scored['trust_adjusted_weight'].to_numpy(dtype=np.float64)
        else:
            effective_aggregation_weight = raw_weights.copy()
        current_scored['effective_aggregation_weight'] = effective_aggregation_weight
        current_scored['effective_harm_weight'] = effective_aggregation_weight * current_scored['source_target_growth_from_profile'].to_numpy(dtype=np.float64)

        current_scored["harm_weight"] = (
            current_scored["aggregation_weight"]
            * current_scored[
                "source_target_growth_from_profile"
            ]
        )
        score_rows.extend(
            current_scored.to_dict(orient="records")
        )

        if args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm == 'hard_rejection':
            aggregation_states = [replacement_states[client_id] for client_id in trusted_ids]
            aggregation_sample_counts = [sample_counts[client_id] for client_id in trusted_ids]
        elif args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm == 'down_weighting':
            aggregation_states = replacement_states
            aggregation_sample_counts = (np.asarray(sample_counts, dtype=np.float64) * current_scored['trust_factor'].to_numpy(dtype=np.float64)).tolist()
        else:
            aggregation_states = replacement_states
            aggregation_sample_counts = sample_counts

        model.load_state_dict(
            weighted_average_states(
                aggregation_states,
                aggregation_sample_counts,
                reference_state,
            )
        )
        val_metrics, val_pair = evaluate_validation(
            model, X_val, y_val, source_id, target_id, args.evaluation_batch_size
        )
        val_probabilities = predict_probabilities(
            model,
            X_val,
            args.evaluation_batch_size,
        )
        val_predictions = val_probabilities.argmax(axis=1)
        val_matrix = np.zeros(
            (NUM_CLASSES, NUM_CLASSES),
            dtype=np.int64,
        )
        np.add.at(
            val_matrix,
            (y_val, val_predictions),
            1,
        )
        for class_id, class_name in enumerate(CLASS_NAMES):
            support = int(val_matrix[class_id, :].sum())
            predicted = int(val_matrix[:, class_id].sum())
            true_positive = int(val_matrix[class_id, class_id])
            recall = (
                float(true_positive / support)
                if support
                else float("nan")
            )
            precision = (
                float(true_positive / predicted)
                if predicted
                else float("nan")
            )
            f1 = (
                float(
                    2.0 * precision * recall
                    / (precision + recall)
                )
                if np.isfinite(precision)
                and np.isfinite(recall)
                and (precision + recall) > 0
                else 0.0
            )
            class_metric_rows.append({
                "monitoring_round": monitoring_round,
                "global_round": global_round,
                "mode": args.mode,
                "attack_type": args.attack_type,
                "class_id": class_id,
                "class_name": class_name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })
            for predicted_id, predicted_name in enumerate(
                CLASS_NAMES
            ):
                confusion_rows.append({
                    "monitoring_round": monitoring_round,
                    "global_round": global_round,
                    "mode": args.mode,
                    "attack_type": args.attack_type,
                    "true_id": class_id,
                    "true_class": class_name,
                    "predicted_id": predicted_id,
                    "predicted_class": predicted_name,
                    "count": int(
                        val_matrix[class_id, predicted_id]
                    ),
                })

        labels = current_scored["actual_malicious"].astype(bool).to_numpy()
        flags = current_scored["flagged"].astype(bool).to_numpy()
        benign = ~labels
        if labels.any():
            malicious_recall = float(np.mean(flags[labels]))
            precision = float(np.sum(flags & labels) / max(np.sum(flags), 1))
            mean_malicious_score = float(current_scored.loc[labels, SCORE_COLUMN].mean())
            malicious_harm = float(current_scored.loc[labels, "harm_weight"].sum())
            malicious_effective_harm = float(current_scored.loc[labels, "effective_harm_weight"].sum())
        else:
            malicious_recall = float("nan")
            precision = float("nan")
            mean_malicious_score = float("nan")
            malicious_harm = 0.0
            malicious_effective_harm = 0.0
        benign_fpr = float(np.mean(flags[benign])) if benign.any() else float("nan")
        mean_benign_score = float(current_scored.loc[benign, SCORE_COLUMN].mean()) if benign.any() else float("nan")
        total_harm = float(current_scored["harm_weight"].sum())
        malicious_harm_share = malicious_harm / max(total_harm, 1e-12) if labels.any() else float("nan")
        total_effective_harm = float(current_scored["effective_harm_weight"].sum())
        malicious_effective_harm_share = malicious_effective_harm / max(total_effective_harm, 1e-12) if labels.any() else float("nan")

        row = {
            "monitoring_round": int(monitoring_round),
            "global_round": int(global_round),
            "mode": args.mode,
            "replacement_policy": args.replacement_policy,
            "ablation_arm": args.ablation_arm,
            'mitigation_action': mitigation_action,
            "participating_samples": int(sum(sample_counts)),
            "replaced_clients": int(replaced_clients),
            "rejected_clients": int(
                flags.sum()
                if args.replacement_policy == "trusted_reconstruction"
                and args.ablation_arm == "hard_rejection"
                else 0
            ),
            "aggregation_retained_clients": int(
                len(trusted_ids)
                if args.replacement_policy == "trusted_reconstruction"
                and args.ablation_arm == "hard_rejection"
                else args.num_clients
            ),
            "mean_local_train_loss": float(np.mean([x["local_train_loss"] for x in current_training])),
            "mean_local_train_accuracy": float(np.mean([x["local_train_accuracy"] for x in current_training])),
            "flagged_clients": int(flags.sum()),
            "malicious_recall": malicious_recall,
            "benign_false_positive_rate": benign_fpr,
            "detection_precision": precision,
            "mean_benign_anchor_ema": mean_benign_score,
            "mean_malicious_anchor_ema": mean_malicious_score,
            "total_harm_weight": total_harm,
            "malicious_harm_share": malicious_harm_share,
            "total_effective_harm_weight": total_effective_harm,
            "malicious_effective_harm_share": malicious_effective_harm_share,
            "round_seconds": float(time.time() - round_started),
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"val_{key}": value for key, value in val_pair.items()},
        }
        round_rows.append(row)

        if args.replacement_policy == "trusted_reconstruction":
            torch.random.set_rng_state(
                next_round_torch_rng_state
            )

        torch.save(
            {
                "experiment_version": "4.34.1-RECONSTRUCTION-ABLATION-EXPANSION",
                "phase": "reviewer_reconstruction_mitigation_ablation",
                "attack_type": args.attack_type,
                "mode": args.mode,
                "replacement_policy": args.replacement_policy,
                "ablation_arm": args.ablation_arm,
                "monitoring_round": int(monitoring_round),
                "global_round": int(global_round),
                "model_state_dict": model.state_dict(),
                "ema_memory": ema_memory,
                "partition_hash": partition_hash,
                "poison_index_hash": poison_hash,
                "warmup_profile_hash": warmup_metadata["profile_sha256"],
                "model_seed": int(args.model_seed),
                "next_round_torch_rng_state":
                    next_round_torch_rng_state,
            },
            continuation_checkpoint,
        )
        save_partial(tables_dir, round_rows, local_rows, score_rows, signature_rows)
        recall_text = "NA" if not labels.any() else f"{malicious_recall:.3f}"
        print(
            f"monitoring round {monitoring_round:02d}, global round {global_round:02d}, "
            f"val macro F1={val_metrics['macro_f1']:.4f}, "
            f"{args.source_class}->{args.target_class}={val_pair['source_to_target_rate']:.4%}, "
            f"recall={recall_text}, benign FPR={benign_fpr:.3f}, "
            f"replaced={replaced_clients}, "
            f"seconds={row['round_seconds']:.1f}"
        )

    round_table = pd.DataFrame(round_rows)
    local_table = pd.DataFrame(local_rows)
    score_table = pd.DataFrame(score_rows)
    signature_table = pd.DataFrame(signature_rows)
    round_table.to_csv(tables_dir / "continuation_round_metrics.csv", index=False)
    local_table.to_csv(tables_dir / "continuation_local_training.csv", index=False)
    score_table.to_csv(tables_dir / "continuation_client_anchor_scores.csv", index=False)
    signature_table.to_csv(
        tables_dir / "continuation_transition_signature_long.csv",
        index=False,
    )
    pd.DataFrame(reconstruction_rows).to_csv(
        tables_dir / "reconstruction_client_rows.csv",
        index=False,
    )
    pd.DataFrame(class_metric_rows).to_csv(
        tables_dir / "validation_class_metrics_long.csv",
        index=False,
    )
    pd.DataFrame(confusion_rows).to_csv(
        tables_dir / "validation_confusion_matrix_long.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(round_table["monitoring_round"], round_table["val_macro_f1"], marker="o", label="Macro F1")
    ax.plot(round_table["monitoring_round"], round_table["val_source_to_target_rate"], marker="s", label=f"{args.source_class}->{args.target_class}")
    ax.set_xlabel("Post-warmup monitoring round")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title(f"V3.10 {args.mode} continuation from common warmup checkpoint")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "validation_and_attack_dynamics")

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(round_table["monitoring_round"], round_table["benign_false_positive_rate"], marker="o", label="Benign FPR")
    if args.mode == "strong_attack":
        ax.plot(round_table["monitoring_round"], round_table["malicious_recall"], marker="s", label="Malicious recall")
    ax.set_xlabel("Post-warmup monitoring round")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Frozen independent-anchor detection")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "anchor_detection_dynamics")

    metadata = {
        "experiment_version": "4.34.1-RECONSTRUCTION-ABLATION-EXPANSION",
        "phase": "reviewer_reconstruction_mitigation_ablation",
        "attack_type": args.attack_type,
        "status": "development_chronology_experiment_not_final_paper_result",
        "mode": args.mode,
        "replacement_policy": args.replacement_policy,
        "ablation_arm": args.ablation_arm,
        "mitigation_action": mitigation_action,
        'reconstruction_calibration_required': bool(args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm == 'center_plus_residual'),
        'preoutcome_design_freeze_commit': '74ecea61b89ef1b3cbd8c6cfa3889cc9c4bd1a70',
        'reconstruction_calibration_cli_guard_restored': True,
        'reconstruction_calibration_used': bool(args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm == 'center_plus_residual'),
        'historical_residual_used': bool(args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm == 'center_plus_residual'),
        'hard_rejection_used': bool(args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm == 'hard_rejection'),
        'center_only_used': bool(args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm == 'center_only'),
        'down_weighting_used': bool(args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm == 'down_weighting'),
        'center_plus_residual_used': bool(args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm == 'center_plus_residual'),
        'down_weighting_trust_gamma': FROZEN_TRUST_GAMMA if args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm == 'down_weighting' else None,
        'down_weighting_minimum_trust': FROZEN_MINIMUM_TRUST if args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm == 'down_weighting' else None,
        'historical_trust_parameter_source_commit': 'fd8410656e7810330cb3e3befa0bb4a8409efff8',
        'historical_gamma_minimum_trust_development_selection_used_malicious_labels': True,
        "detector_algorithm_changed": False,
        "detector_thresholds_changed": False,
        "model_seed": int(args.model_seed),
        "common_warmup_rounds": 4,
        "continuation_rounds": int(args.continuation_rounds),
        'aggregation_rule': ('raw_sample_count_fedavg' if args.replacement_policy == 'plain_fedavg' else {'center_plus_residual': 'raw_sample_count_fedavg', 'center_only': 'raw_sample_count_fedavg', 'hard_rejection': 'hard_rejection_raw_sample_count_renormalized', 'down_weighting': 'raw_sample_count_times_frozen_soft_trust'}[args.ablation_arm]),
        "count_cap_used": False,
        'defense_weights_applied': bool(args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm in {'hard_rejection', 'down_weighting'}),
        "selected_reconstruction_policy": (
            selected_reconstruction_policy
            if args.replacement_policy == "trusted_reconstruction"
            else None
        ),
        "frozen_ema_quantile": FROZEN_EMA_QUANTILE,
        "frozen_instant_quantile": FROZEN_INSTANT_QUANTILE,
        "frozen_ema_threshold": float(ema_threshold_q99),
        "frozen_instant_threshold": (
            None
            if not np.isfinite(instant_threshold_q95)
            else float(instant_threshold_q95)
        ),
        "candidate": CANDIDATE,
        "score_column": SCORE_COLUMN,
        "ema_decay": float(args.ema_decay),
        "monitoring_ema_reset_after_warmup": True,
        "clean_ema_threshold": float(threshold),
        "current_global_reference_used": False,
        "stable_client_identity_required": True,
        "malicious_clients": malicious_clients,
        "partition_hash_sha256": partition_hash,
        "poison_index_hash_sha256": poison_hash,
        "warmup_profile_sha256": warmup_metadata["profile_sha256"],
        "common_branch_checkpoint": str(branch_checkpoint_path),
        "exact_v320a3_plain_branch": (
            None
            if args.plain_branch_dir is None
            else str(args.plain_branch_dir.expanduser().resolve())
        ),
        "exact_v320a3_poison_plan_reused": bool(
            args.mode == "strong_attack"
        ),
        "attack_specific_retuning": False,
        "test_sets_accessed": False,
        "malicious_labels_used_for_aggregation": False,
        "malicious_labels_used_for_reporting_only": True,
        "rounds_completed": int(round_table["monitoring_round"].max()),
        "mean_validation_macro_f1": float(round_table["val_macro_f1"].mean()),
        "mean_validation_source_to_target_rate": float(round_table["val_source_to_target_rate"].mean()),
        "mean_benign_false_positive_rate": float(round_table["benign_false_positive_rate"].mean()),
        "mean_malicious_recall": (
            None if args.mode == "clean" else float(round_table["malicious_recall"].mean())
        ),
        "total_seconds": float(time.time() - started),
    }
    with (output_dir / "reviewer_reconstruction_ablation_v4341_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print('Reviewer Reconstruction Ablation V4.34.1 complete')
    print("Mode:", args.mode)
    print("Attack type:", args.attack_type)
    print("Ablation arm:", args.ablation_arm)
    print('Reconstruction calibration used:', args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm == 'center_plus_residual')
    print('Historical residual used:', args.replacement_policy == 'trusted_reconstruction' and args.ablation_arm == 'center_plus_residual')
    print("Mean validation macro F1:", f"{round_table['val_macro_f1'].mean():.6f}")
    print("Mean source-to-target rate:", f"{round_table['val_source_to_target_rate'].mean():.6f}")
    print("Mean benign FPR:", f"{round_table['benign_false_positive_rate'].mean():.6f}")
    if args.mode == "strong_attack":
        print("Mean malicious recall:", f"{round_table['malicious_recall'].mean():.6f}")
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

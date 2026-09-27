#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_exact_untargeted_plain_v320a3 import prepare_untargeted_attack  # noqa: E402
from run_frozen_untargeted_defense_v320b1 import load_exact_v320a3_poison_plan  # noqa: E402
from run_targeted_label_flip_v292 import load_fixed_partitions  # noqa: E402

EXPECTED_DATA_SHA256 = "1475d26cbddfcaf874c5f00ad83d027b1de529e06be6bf0c330d89abe26752fd"
EXPECTED_PARTITION_HASH = "5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"
EXPECTED_PLAIN_NORMALIZED_SHA256 = "c579d04815a7e7a72a934926c8092056e83714702322a38a97f2b3b4dab38b75"
EXPECTED_DEFENSE_NORMALIZED_SHA256 = "9103a41b8152034be2745cd9fbcdbc320b2d935023ceb93eb7b06d93cf7e5b33"
DEFAULT_MALICIOUS = [1, 7, 8, 10, 14, 15, 17, 18]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized_lf_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def parse_clients(text: str):
    out = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(out) != 8 or sorted(out) != sorted(DEFAULT_MALICIOUS):
        raise ValueError(f"Recovery is frozen to malicious clients {DEFAULT_MALICIOUS}; got {out}")
    return out


def save_poisoned_indices(*, client_indices, poisoned_positions: Dict[int, np.ndarray], path: Path) -> None:
    arrays = {}
    for client_id, local_positions in sorted(poisoned_positions.items()):
        local_positions = np.asarray(local_positions, dtype=np.int64)
        global_indices = np.asarray(client_indices[client_id][local_positions], dtype=np.int64)
        arrays[f"client_{client_id:03d}_local_positions"] = local_positions
        arrays[f"client_{client_id:03d}_global_indices"] = global_indices
    np.savez_compressed(path, **arrays)


def save_poisoned_labels(poisoned_labels: Dict[int, np.ndarray], path: Path) -> None:
    np.savez_compressed(
        path,
        **{
            f"client_{client_id:03d}": np.asarray(labels, dtype=np.int64)
            for client_id, labels in sorted(poisoned_labels.items())
        },
    )


def dict_arrays_equal(a, b) -> bool:
    return set(a) == set(b) and all(
        np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a
    )


def call_exact_loader(
    *,
    out: Path,
    client_indices,
    y_train,
    attack_type: str,
    model_seed: int,
    malicious,
    poison_fraction: float,
    partition_hash: str,
):
    loader = load_exact_v320a3_poison_plan
    sig = inspect.signature(loader)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in {"plain_branch_dir", "plain_branch", "branch", "branch_dir"}:
            kwargs[name] = out
        elif name in {"client_indices", "partitions"}:
            kwargs[name] = client_indices
        elif name == "y_train":
            kwargs[name] = y_train
        elif name == "attack_type":
            kwargs[name] = attack_type
        elif name == "model_seed":
            kwargs[name] = model_seed
        elif name in {"expected_malicious_clients", "malicious_clients"}:
            kwargs[name] = malicious
        elif name == "num_clients":
            kwargs[name] = len(client_indices)
        elif name == "poison_fraction":
            kwargs[name] = poison_fraction
        elif name in {"partition_hash", "expected_partition_hash"}:
            kwargs[name] = partition_hash
        elif param.default is not inspect.Parameter.empty:
            continue
        else:
            raise RuntimeError(
                f"Unrecognized required exact-loader parameter {name!r}; signature={sig}"
            )

    print("EXACT LOADER SIGNATURE:", sig)
    result = loader(**kwargs)
    if not isinstance(result, tuple) or len(result) != 5:
        raise RuntimeError(
            f"Unexpected exact-loader return shape: type={type(result)!r}, "
            f"length={len(result) if isinstance(result, tuple) else 'n/a'}"
        )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Recover frozen Task65 attack manifest without training")
    ap.add_argument("--data-file", required=True, type=Path)
    ap.add_argument("--partition-file", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--evidence-file", required=True, type=Path)
    ap.add_argument("--model-seed", required=True, type=int)
    ap.add_argument("--attack-seed", required=True, type=int)
    ap.add_argument(
        "--attack-type",
        required=True,
        choices=[
            "all_to_one_benign",
            "cyclic_shift",
            "multiclass_partial_cycle",
            "pairwise_swap",
            "random_flip",
        ],
    )
    ap.add_argument("--num-clients", type=int, default=20)
    ap.add_argument("--malicious-clients", default="1,7,8,10,14,15,17,18")
    ap.add_argument("--poison-fraction", type=float, default=1.0)
    args = ap.parse_args()

    if args.model_seed != args.attack_seed:
        raise RuntimeError("Task65 recovery requires attack_seed == model_seed")
    if args.num_clients != 20:
        raise RuntimeError("Task65 recovery is frozen to 20 clients")
    if float(args.poison_fraction) != 1.0:
        raise RuntimeError("Task65 recovery is frozen to poison fraction 1.0")
    malicious = parse_clients(args.malicious_clients)

    data_file = args.data_file.resolve()
    partition_file = args.partition_file.resolve()
    out = args.output_dir.resolve()
    evidence_file = args.evidence_file.resolve()

    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Recovery output already exists and is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    attack_dir = out / "attack_manifest"
    attack_dir.mkdir(parents=True, exist_ok=True)
    evidence_file.parent.mkdir(parents=True, exist_ok=True)

    data_sha = sha256_file(data_file)
    if data_sha != EXPECTED_DATA_SHA256:
        raise RuntimeError(f"Data SHA256 mismatch: {data_sha}")

    plain_source = PROJECT_ROOT / "scripts" / "run_exact_untargeted_plain_v320a3.py"
    defense_source = PROJECT_ROOT / "scripts" / "run_frozen_untargeted_defense_v320b1.py"
    plain_raw_sha = sha256_file(plain_source)
    defense_raw_sha = sha256_file(defense_source)
    plain_norm_sha = normalized_lf_sha256(plain_source)
    defense_norm_sha = normalized_lf_sha256(defense_source)

    if plain_norm_sha != EXPECTED_PLAIN_NORMALIZED_SHA256:
        raise RuntimeError(f"Normalized frozen plain source mismatch: {plain_norm_sha}")
    if defense_norm_sha != EXPECTED_DEFENSE_NORMALIZED_SHA256:
        raise RuntimeError(f"Normalized frozen defense source mismatch: {defense_norm_sha}")

    # Deliberately materialize only y_train.
    with np.load(data_file, allow_pickle=False) as payload:
        if "y_train" not in payload.files:
            raise KeyError("y_train missing from protocol NPZ")
        y_train = np.asarray(payload["y_train"], dtype=np.int64)

    client_indices, partition_hash = load_fixed_partitions(
        partition_file,
        expected_clients=args.num_clients,
        train_rows=len(y_train),
    )
    if partition_hash != EXPECTED_PARTITION_HASH:
        raise RuntimeError(f"Logical partition hash mismatch: {partition_hash}")

    p1, l1, m1, h1 = prepare_untargeted_attack(
        client_indices=client_indices,
        y_train=y_train,
        malicious_clients=malicious,
        poison_fraction=args.poison_fraction,
        attack_seed=args.attack_seed,
        attack_type=args.attack_type,
    )
    p2, l2, m2, h2 = prepare_untargeted_attack(
        client_indices=client_indices,
        y_train=y_train,
        malicious_clients=malicious,
        poison_fraction=args.poison_fraction,
        attack_seed=args.attack_seed,
        attack_type=args.attack_type,
    )

    if h1 != h2:
        raise RuntimeError("Independent attack-constructor poison digests differ")
    if not dict_arrays_equal(p1, p2):
        raise RuntimeError("Independent attack-constructor selected positions differ")
    if not dict_arrays_equal(l1, l2):
        raise RuntimeError("Independent attack-constructor replacement labels differ")
    pd.testing.assert_frame_equal(
        m1.reset_index(drop=True), m2.reset_index(drop=True), check_exact=True
    )

    total_poisoned = int(m1["poisoned_rows"].sum())
    malicious_poisoned = int(
        m1.loc[m1["is_malicious"].astype(bool), "poisoned_rows"].sum()
    )
    if total_poisoned <= 0 or malicious_poisoned <= 0:
        raise RuntimeError("Recovered attack manifest contains no poisoning")

    manifest_csv = attack_dir / "malicious_client_poison_manifest.csv"
    indices_npz = attack_dir / "poisoned_indices.npz"
    labels_npz = attack_dir / "poisoned_labels.npz"
    m1.to_csv(manifest_csv, index=False)
    save_poisoned_indices(
        client_indices=client_indices,
        poisoned_positions=p1,
        path=indices_npz,
    )
    save_poisoned_labels(l1, labels_npz)

    pd.DataFrame([{
        "attack_type": args.attack_type,
        "model_seed": args.model_seed,
        "attack_seed": args.attack_seed,
        "num_clients": args.num_clients,
        "malicious_clients": ",".join(str(x) for x in malicious),
        "poison_fraction": float(args.poison_fraction),
        "total_poisoned_rows": total_poisoned,
        "malicious_poisoned_rows": malicious_poisoned,
        "poison_index_hash_sha256": h1,
        "partition_hash_sha256": partition_hash,
        "recovery_status": "deterministic_reviewer_reconstruction_from_frozen_constructor",
    }]).to_csv(attack_dir / "attack_summary.csv", index=False)

    metadata = {
        "experiment_version": "3.20A.3",
        "phase": "exact_untargeted_plain_fedavg_qualification",
        "status": "reviewer_recovered_manifest_only_from_frozen_task65_constructor",
        "mode": "strong_attack",
        "attack_type": args.attack_type,
        "model_seed": int(args.model_seed),
        "attack_seed": int(args.attack_seed),
        "common_warmup_rounds": 4,
        "continuation_rounds": 4,
        "aggregation_rule": "raw_sample_count_fedavg",
        "count_cap_used": False,
        "defense_weights_applied": False,
        "ema_decay": 0.65,
        "monitoring_ema_reset_after_warmup": True,
        "malicious_clients": malicious,
        "partition_hash_sha256": partition_hash,
        "poison_index_hash_sha256": h1,
        "test_sets_accessed": False,
        "attack_specific_retuning": False,
        "reviewer_recovery_v4324": True,
        "historical_attack_manifest_binary_recovered": False,
    }
    metadata_path = out / "exact_untargeted_plain_v320a3_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    loaded_mal, loaded_pos, loaded_labels, loaded_manifest, loaded_hash = call_exact_loader(
        out=out,
        client_indices=client_indices,
        y_train=y_train,
        attack_type=args.attack_type,
        model_seed=args.model_seed,
        malicious=malicious,
        poison_fraction=args.poison_fraction,
        partition_hash=partition_hash,
    )

    if sorted(loaded_mal) != sorted(malicious):
        raise RuntimeError("Exact loader returned wrong malicious client set")
    if loaded_hash != h1:
        raise RuntimeError("Exact loader poison hash differs from constructor hash")
    if not dict_arrays_equal(loaded_pos, p1):
        raise RuntimeError("Exact loader positions differ from constructor output")
    if not dict_arrays_equal(loaded_labels, l1):
        raise RuntimeError("Exact loader labels differ from constructor output")
    pd.testing.assert_frame_equal(
        loaded_manifest.reset_index(drop=True),
        m1.reset_index(drop=True),
        check_exact=True,
    )

    evidence = {
        "protocol": "reviewer_v4324_task65_attack_manifest_recovery",
        "task65_git_reference": "2e40296",
        "condition": {
            "model_seed": int(args.model_seed),
            "attack_seed": int(args.attack_seed),
            "attack_type": args.attack_type,
            "num_clients": int(args.num_clients),
            "malicious_clients": malicious,
            "poison_fraction": float(args.poison_fraction),
        },
        "data_sha256": data_sha,
        "partition_logical_sha256": partition_hash,
        "plain_source_raw_worktree_sha256": plain_raw_sha,
        "plain_source_lf_normalized_sha256": plain_norm_sha,
        "defense_source_raw_worktree_sha256": defense_raw_sha,
        "defense_source_lf_normalized_sha256": defense_norm_sha,
        "poison_index_hash_sha256": h1,
        "total_poisoned_rows": total_poisoned,
        "malicious_poisoned_rows": malicious_poisoned,
        "independent_constructor_repeat_exact": True,
        "existing_exact_loader_validation_passed": True,
        "historical_attack_manifest_binary_recovered": False,
        "test_arrays_materialized": False,
        "local_model_training_run": False,
        "p4p_attack_outcome_run": False,
        "batr_attack_outcome_run": False,
        "manifest_files": {
            "malicious_client_poison_manifest_csv_sha256": sha256_file(manifest_csv),
            "poisoned_indices_npz_sha256": sha256_file(indices_npz),
            "poisoned_labels_npz_sha256": sha256_file(labels_npz),
            "metadata_json_sha256": sha256_file(metadata_path),
        },
        "status": "PASS",
    }
    evidence_file.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    print("TASK65 ATTACK MANIFEST DETERMINISTIC RECOVERY = PASS")
    print("MODEL SEED:", args.model_seed)
    print("ATTACK SEED:", args.attack_seed)
    print("ATTACK TYPE:", args.attack_type)
    print("PLAIN SOURCE LF NORMALIZED HASH:", plain_norm_sha)
    print("DEFENSE SOURCE LF NORMALIZED HASH:", defense_norm_sha)
    print("PARTITION HASH:", partition_hash)
    print("POISON INDEX HASH:", h1)
    print("TOTAL POISONED ROWS:", total_poisoned)
    print("MALICIOUS POISONED ROWS:", malicious_poisoned)
    print("INDEPENDENT CONSTRUCTOR REPEAT EXACT: True")
    print("EXISTING EXACT LOADER VALIDATION PASSED: True")
    print("HISTORICAL ATTACK MANIFEST BINARY RECOVERED: False")
    print("TEST ARRAYS MATERIALIZED: False")
    print("LOCAL MODEL TRAINING RUN: False")
    print("P4P ATTACK OUTCOME RUN: False")
    print("OUTPUT:", out)
    print("EVIDENCE:", evidence_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

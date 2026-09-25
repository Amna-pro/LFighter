#!/usr/bin/env python3
"""Pre-outcome deterministic reconstruction of W3 for matched P4P.

Replays the trusted clean warmup through round 4, verifies equivalence to the
historical frozen W4 branch checkpoint, and stores only the recovered W3 state
plus audit metadata needed to initialize the first P4P probe.  No final-test
arrays are materialized.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from federated_iot_v26 import NUM_CLASSES, set_seed, sqrt_class_weights, train_local_model  # noqa: E402
from independent_anchor_v310 import weighted_average_states  # noqa: E402
from neural_models_v24 import build_model  # noqa: E402
from run_targeted_label_flip_v292 import load_fixed_partitions, read_clean_partition_hash  # noqa: E402
from trusted_update_reconstruction_v312 import checkpoint_sha256, state_max_abs_difference  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build verified P4P W3/W4 probe bootstrap")
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--partition-file", type=Path, required=True)
    p.add_argument("--clean-seed-dir", type=Path, required=True)
    p.add_argument("--warmup-dir", type=Path, required=True)
    p.add_argument("--output-file", type=Path, required=True)
    p.add_argument("--model-seed", type=int, required=True)
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-class-weight", type=float, default=4.0)
    p.add_argument("--gradient-clip-norm", type=float, default=5.0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--equivalence-tolerance", type=float, default=1e-7)
    return p.parse_args()


def load_train_only(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as payload:
        required = {"X_train", "y_train"}
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"Missing train arrays: {sorted(missing)}")
        X_train = np.asarray(payload["X_train"], dtype=np.float32)
        y_train = np.asarray(payload["y_train"], dtype=np.int64)
    return X_train, y_train


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    set_seed(args.model_seed)

    warmup_dir = args.warmup_dir.expanduser().resolve()
    metadata_path = warmup_dir / "true_warmup_v310_metadata.json"
    frozen_round4_path = warmup_dir / "checkpoints" / "common_round4_warmup_model.pt"
    if not metadata_path.exists() or not frozen_round4_path.exists():
        raise FileNotFoundError("Frozen warmup metadata/checkpoint is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata["model_seed"]) != int(args.model_seed):
        raise RuntimeError("Warmup model seed mismatch")
    if int(metadata["warmup_rounds"]) != 4:
        raise RuntimeError("Expected four-round trusted warmup")

    X_train, y_train = load_train_only(args.data_file)
    client_indices, partition_hash = load_fixed_partitions(
        args.partition_file.expanduser().resolve(),
        expected_clients=args.num_clients,
        train_rows=len(y_train),
    )
    clean_hash = read_clean_partition_hash(args.clean_seed_dir.expanduser().resolve())
    if partition_hash != clean_hash or partition_hash != metadata["partition_hash_sha256"]:
        raise RuntimeError("Partition hash mismatch during P4P bootstrap replay")

    class_weights = sqrt_class_weights(y_train, args.max_class_weight)
    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    round3_state: Dict[str, torch.Tensor] | None = None

    for round_id in range(1, 5):
        reference_state = copy.deepcopy(model.state_dict())
        local_states: List[Mapping[str, torch.Tensor]] = []
        sample_counts: List[int] = []
        for client_id in range(args.num_clients):
            indices = client_indices[client_id]
            local_model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
            local_model.load_state_dict(reference_state)
            state, _ = train_local_model(
                model=local_model,
                X=X_train[indices],
                y=y_train[indices],
                class_weights=class_weights,
                local_epochs=1,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                seed=args.model_seed + round_id * 1000 + client_id,
            )
            local_states.append(state)
            sample_counts.append(int(len(indices)))
            del local_model
        model.load_state_dict(weighted_average_states(local_states, sample_counts, reference_state))
        if round_id == 3:
            round3_state = copy.deepcopy(model.state_dict())

    if round3_state is None:
        raise RuntimeError("Failed to capture replayed W3")

    frozen = torch.load(frozen_round4_path, map_location="cpu", weights_only=False)
    frozen_state = frozen["model_state_dict"]
    replay_state = model.state_dict()
    if set(frozen_state) != set(replay_state):
        raise RuntimeError("Replayed/frozen W4 state keys differ")
    max_abs = state_max_abs_difference(replay_state, frozen_state)
    if max_abs > args.equivalence_tolerance:
        raise RuntimeError(
            f"P4P warmup replay equivalence failed: max_abs={max_abs:.12g} > "
            f"tol={args.equivalence_tolerance:.12g}"
        )

    output = args.output_file.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "protocol": "reviewer_v4322_p4p_probe_bootstrap",
        "model_seed": int(args.model_seed),
        "partition_hash_sha256": partition_hash,
        "frozen_round4_checkpoint": str(frozen_round4_path),
        "frozen_round4_checkpoint_sha256": checkpoint_sha256(frozen_round4_path),
        "round4_replay_max_abs_difference": float(max_abs),
        "equivalence_tolerance": float(args.equivalence_tolerance),
        "test_arrays_materialized": False,
        "round3_model_state_dict": round3_state,
    }
    torch.save(bundle, output)
    marker = output.with_suffix(".json")
    marker.write_text(
        json.dumps(
            {k: v for k, v in bundle.items() if k != "round3_model_state_dict"},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print("P4P PROBE BOOTSTRAP VERIFIED")
    print("MODEL SEED:", args.model_seed)
    print("ROUND4 MAX ABS DIFFERENCE:", f"{max_abs:.12g}")
    print("TEST ARRAYS MATERIALIZED: False")
    print("BOOTSTRAP:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

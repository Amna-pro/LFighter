#!/usr/bin/env python3
"""Select one clean-only non-Benign target for every attack source class.

Selection is made before attack execution from the frozen common round-4
warmup checkpoints for development seeds 7, 99, 123, and 2026.

For each non-Benign source class, select the non-Benign, non-self target with
the highest mean clean hard-prediction transition rate. Ties are broken by the
fixed class ID. This is a difficult but operationally plausible confusion
target and does not use attack outcomes or test data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from federated_iot_v26 import CLASS_NAMES, NUM_CLASSES, load_protocol_arrays
from neural_models_v24 import build_model
from transition_signature_features_v38 import predict_probabilities


FROZEN_SEEDS = [7, 99, 123, 2026]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True, type=Path)
    p.add_argument("--v3101-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seeds", default="7,99,123,2026")
    p.add_argument("--evaluation-batch-size", type=int, default=4096)
    p.add_argument("--threads", type=int, default=6)
    return p.parse_args()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    a = parse_args()
    seeds = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]
    if seeds != FROZEN_SEEDS:
        raise ValueError(
            f"V3.13.4 pair selection is frozen to {FROZEN_SEEDS}, got {seeds}"
        )

    torch.set_num_threads(max(1, int(a.threads)))
    output = a.output_dir.expanduser().resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    arrays = load_protocol_arrays(a.data_file.expanduser().resolve())
    X_val = arrays["X_val"].astype(np.float32, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)
    input_dim = int(X_val.shape[1])

    nonbenign_ids = [
        class_id
        for class_id, name in enumerate(CLASS_NAMES)
        if name != "Benign"
    ]
    rows: List[Dict[str, object]] = []
    checkpoint_rows: List[Dict[str, object]] = []

    for seed in seeds:
        checkpoint_path = (
            a.v3101_root.expanduser().resolve()
            / f"seed_{seed}" / "warmup" / "checkpoints"
            / "common_round4_warmup_model.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if int(checkpoint.get("model_seed", seed)) != seed:
            raise RuntimeError(
                f"Warmup checkpoint seed mismatch for seed {seed}"
            )

        model = build_model("resmlp", input_dim, NUM_CLASSES)
        model.load_state_dict(checkpoint["model_state_dict"])
        probabilities = predict_probabilities(
            model, X_val, int(a.evaluation_batch_size)
        )
        predictions = np.argmax(probabilities, axis=1).astype(np.int64)

        checkpoint_rows.append({
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "validation_rows": int(len(y_val)),
        })

        for source_id in nonbenign_ids:
            source_mask = y_val == source_id
            support = int(source_mask.sum())
            if support <= 0:
                raise RuntimeError(
                    f"Validation source support is zero for {CLASS_NAMES[source_id]}"
                )
            for target_id in nonbenign_ids:
                if target_id == source_id:
                    continue
                count = int(np.sum(predictions[source_mask] == target_id))
                rows.append({
                    "seed": seed,
                    "source_id": source_id,
                    "source_class": CLASS_NAMES[source_id],
                    "target_id": target_id,
                    "target_class": CLASS_NAMES[target_id],
                    "source_support": support,
                    "clean_source_to_target_count": count,
                    "clean_source_to_target_rate": float(count / support),
                    "selection_data": "validation_only",
                    "selection_checkpoint": "common_round4_warmup",
                    "attack_outcomes_used": False,
                    "test_sets_accessed": False,
                })

    long_table = pd.DataFrame(rows)
    long_table.to_csv(
        tables / "v3134_clean_nonbenign_pair_candidates_long.csv",
        index=False,
    )
    pd.DataFrame(checkpoint_rows).to_csv(
        tables / "v3134_pair_selection_checkpoints.csv",
        index=False,
    )

    aggregate = (
        long_table.groupby(
            ["source_id", "source_class", "target_id", "target_class"],
            as_index=False,
        )
        .agg(
            mean_clean_source_to_target_rate=(
                "clean_source_to_target_rate", "mean"
            ),
            minimum_seed_clean_source_to_target_rate=(
                "clean_source_to_target_rate", "min"
            ),
            maximum_seed_clean_source_to_target_rate=(
                "clean_source_to_target_rate", "max"
            ),
            mean_source_support=("source_support", "mean"),
        )
    )
    aggregate["selected"] = False
    selected_rows = []
    for source_id in nonbenign_ids:
        candidates = aggregate[
            aggregate["source_id"].astype(int) == source_id
        ].sort_values(
            [
                "mean_clean_source_to_target_rate",
                "target_id",
            ],
            ascending=[False, True],
        )
        if len(candidates) == 0:
            raise RuntimeError(
                f"No non-Benign target candidates for {CLASS_NAMES[source_id]}"
            )
        selected = candidates.iloc[0]
        mask = (
            aggregate["source_id"].astype(int).eq(source_id)
            & aggregate["target_id"].astype(int).eq(
                int(selected["target_id"])
            )
        )
        aggregate.loc[mask, "selected"] = True
        selected_rows.append({
            "pair_id": len(selected_rows) + 1,
            "source_id": source_id,
            "source_class": CLASS_NAMES[source_id],
            "target_id": int(selected["target_id"]),
            "target_class": str(selected["target_class"]),
            "pair_name": (
                f"{CLASS_NAMES[source_id]}_to_"
                f"{selected['target_class']}"
            ),
            "mean_clean_source_to_target_rate": float(
                selected["mean_clean_source_to_target_rate"]
            ),
            "minimum_seed_clean_source_to_target_rate": float(
                selected["minimum_seed_clean_source_to_target_rate"]
            ),
            "maximum_seed_clean_source_to_target_rate": float(
                selected["maximum_seed_clean_source_to_target_rate"]
            ),
            "selection_rule": (
                "highest_mean_clean_round4_warmup_hard_transition_"
                "excluding_benign_and_self_then_target_id"
            ),
            "selection_uses_attack_outcomes": False,
            "selection_uses_test_sets": False,
        })

    aggregate.to_csv(
        tables / "v3134_clean_nonbenign_pair_candidates_aggregate.csv",
        index=False,
    )
    selected_table = pd.DataFrame(selected_rows)
    selected_table.to_csv(
        tables / "v3134_selected_nonbenign_pair_manifest.csv",
        index=False,
    )

    metadata = {
        "experiment_version": "3.13.4",
        "stage": "clean_only_nonbenign_target_pair_selection",
        "seeds": seeds,
        "source_classes": [
            CLASS_NAMES[class_id] for class_id in nonbenign_ids
        ],
        "selected_pair_count": int(len(selected_table)),
        "selection_rule": (
            "highest_mean_clean_round4_warmup_hard_transition_"
            "excluding_benign_and_self_then_target_id"
        ),
        "attack_outcomes_used": False,
        "natural_test_accessed": False,
        "diagnostic_test_accessed": False,
    }
    with (output / "v3134_pair_selection_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print("V3.13.4 clean-only non-Benign pair selection complete")
    print(selected_table.to_string(index=False))
    print("Tables:", tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

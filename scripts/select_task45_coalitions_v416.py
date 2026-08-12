#!/usr/bin/env python3
"""Freeze Task 45 nested coalition identities from training data only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASS_NAMES = [
    "Benign",
    "BruteForce",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web-Based",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--protocol-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ranked(ids: Sequence[int], key_template: str) -> List[int]:
    def key(client_id: int):
        value = key_template.format(client_id_3digit=f"{client_id:03d}")
        return hashlib.sha256(value.encode("utf-8")).hexdigest(), client_id

    return sorted((int(value) for value in ids), key=key)


def load_fixed_partitions(
    path: Path,
    expected_clients: int,
    train_rows: int,
) -> Tuple[List[np.ndarray], str]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    with np.load(resolved) as payload:
        keys = sorted(payload.files)
        if len(keys) != expected_clients:
            raise ValueError(
                f"Expected {expected_clients} clients, partition has {len(keys)}"
            )
        expected_keys = [f"client_{client_id:03d}" for client_id in range(expected_clients)]
        if keys != expected_keys:
            raise ValueError("Partition keys do not match stable zero-padded client IDs")
        partitions = [np.asarray(payload[key], dtype=np.int64) for key in keys]

    all_indices = np.concatenate(partitions)
    if len(all_indices) != train_rows:
        raise ValueError("Partition row count does not match y_train")
    unique = np.unique(all_indices)
    if (
        len(unique) != train_rows
        or int(unique.min()) != 0
        or int(unique.max()) != train_rows - 1
    ):
        raise ValueError("Partition does not cover the training rows exactly once")

    digest = hashlib.sha256()
    for client_id, indices in enumerate(partitions):
        digest.update(f"client_{client_id:03d}".encode("utf-8"))
        digest.update(indices.tobytes())
    return partitions, digest.hexdigest()


def coalition_hash(
    protocol_hash: str,
    partition_hash: str,
    family_id: str,
    size: int,
    selected_ids: Sequence[int],
) -> str:
    digest = hashlib.sha256()
    digest.update(protocol_hash.encode("utf-8"))
    digest.update(partition_hash.encode("utf-8"))
    digest.update(family_id.encode("utf-8"))
    digest.update(np.asarray([size], dtype=np.int64).tobytes())
    digest.update(np.asarray(sorted(selected_ids), dtype=np.int64).tobytes())
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol_file.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("experiment_version") != "4.16.0":
        raise ValueError("Task 45 protocol version mismatch")

    scope = protocol["scope"]
    if scope["source_class"] not in CLASS_NAMES:
        raise ValueError("Frozen source class is unknown")
    source_id = CLASS_NAMES.index(scope["source_class"])

    data_path = args.data_file.expanduser().resolve()
    with np.load(data_path) as payload:
        if "y_train" not in payload.files:
            raise KeyError("Prepared NPZ has no y_train")
        y_train = payload["y_train"].astype(np.int64, copy=False)

    partitions, partition_hash = load_fixed_partitions(
        args.partition_file.expanduser().resolve(),
        expected_clients=int(scope["num_clients"]),
        train_rows=len(y_train),
    )
    counts = [
        int(np.sum(y_train[indices] == source_id)) for indices in partitions
    ]
    minimum = int(scope["min_source_samples"])
    eligible = [client_id for client_id, count in enumerate(counts) if count >= minimum]
    expected = protocol["eligibility"]["expected_eligible_client_ids"]
    if eligible != expected:
        raise RuntimeError(
            f"Eligible clients differ from preregistration: {eligible} != {expected}"
        )

    families = protocol["coalition_design"]["families"]
    sizes = protocol["coalition_design"]["sizes"]
    original = {1, 7, 8, 10, 14, 15, 17, 18}
    derived_orders: Dict[str, List[int]] = {}
    for family in families:
        family_id = family["family_id"]
        keys = family["ranking_keys"]
        if family_id == "A_development_anchor":
            order = ranked(original, keys["anchor_pool"])
            order += ranked(set(eligible) - original, keys["extension_pool"])
        else:
            order = ranked(eligible, keys["eligible_pool"])
        if order != family["ordered_client_ids"]:
            raise RuntimeError(f"Frozen ranking mismatch for {family_id}")
        derived_orders[family_id] = order

    protocol_hash = file_sha256(protocol_path)
    count_rows = []
    for client_id, count in enumerate(counts):
        count_rows.append(
            {
                "source_class": scope["source_class"],
                "source_id": source_id,
                "client_id": client_id,
                "clean_source_rows": count,
                "eligible_ge_1000": client_id in eligible,
            }
        )

    manifest_rows = []
    identity_sets: Dict[int, set] = {int(size): set() for size in sizes}
    for family_id, order in derived_orders.items():
        previous: set = set()
        for size in sizes:
            selected = sorted(order[: int(size)])
            selected_set = set(selected)
            if not previous.issubset(selected_set):
                raise RuntimeError(f"Family {family_id} is not nested at size {size}")
            if len(selected_set) != int(size):
                raise RuntimeError(f"Family {family_id} has duplicate IDs at size {size}")
            identity_key = tuple(selected)
            if identity_key in identity_sets[int(size)]:
                raise RuntimeError(f"Duplicate coalition identity at size {size}")
            identity_sets[int(size)].add(identity_key)
            if any(counts[client_id] < minimum for client_id in selected):
                raise RuntimeError("Ineligible client entered a frozen coalition")
            if family_id == "A_development_anchor" and int(size) == 8:
                if selected_set != original:
                    raise RuntimeError("Family A size 8 does not preserve the anchor")

            selected_rows = int(sum(counts[client_id] for client_id in selected))
            global_rows = int(sum(counts))
            manifest_rows.append(
                {
                    "family_id": family_id,
                    "coalition_size": int(size),
                    "selected_clients": "|".join(map(str, selected)),
                    "selected_source_rows": selected_rows,
                    "global_source_rows": global_rows,
                    "global_source_exposure_fraction": selected_rows / max(global_rows, 1),
                    "minimum_selected_client_source_rows": min(counts[i] for i in selected),
                    "maximum_selected_client_source_rows": max(counts[i] for i in selected),
                    "coalition_hash_sha256": coalition_hash(
                        protocol_hash,
                        partition_hash,
                        family_id,
                        int(size),
                        selected,
                    ),
                }
            )
            previous = selected_set

    output = args.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    count_frame = pd.DataFrame(count_rows)
    manifest_frame = pd.DataFrame(manifest_rows)
    count_frame.to_csv(tables / "task45_source_client_counts.csv", index=False)
    manifest_frame.to_csv(tables / "task45_coalition_manifest.csv", index=False)

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for family_id, group in manifest_frame.groupby("family_id", sort=False):
        axis.plot(
            group["coalition_size"],
            group["global_source_exposure_fraction"],
            marker="o",
            label=family_id,
        )
    axis.set_xlabel("Malicious coalition size")
    axis.set_ylabel("Fraction of global clean DDoS rows in coalition")
    axis.set_title("Task 45 preregistered coalition exposure")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "task45_coalition_exposure.png", dpi=300)
    figure.savefig(figures / "task45_coalition_exposure.pdf")
    plt.close(figure)

    metadata = {
        "experiment_version": "4.16.C1",
        "stage": "task45_coalition_manifest_freeze",
        "protocol_sha256": protocol_hash,
        "partition_hash_sha256": partition_hash,
        "source_class": scope["source_class"],
        "target_class": scope["target_class"],
        "min_source_samples": minimum,
        "eligible_client_ids": eligible,
        "coalition_sizes": sizes,
        "coalition_family_count": len(families),
        "coalition_condition_count": len(manifest_rows),
        "nested_within_family": True,
        "distinct_identity_at_each_size": True,
        "selection_uses_attack_outcomes": False,
        "selection_uses_validation_metrics": False,
        "reserved_test_arrays_materialized": False,
        "training_started": False,
    }
    (output / "task45_coalition_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Task 45 coalition manifest created")
    print(manifest_frame.to_string(index=False))
    print("TRAINING STARTED: False")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Select frozen source-capable stress coalitions for V3.13."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from federated_iot_v26 import CLASS_NAMES, load_protocol_arrays
from run_targeted_label_flip_v292 import load_fixed_partitions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True, type=Path)
    p.add_argument("--partition-file", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--source-classes",
        default="BruteForce,DDoS,DoS,Mirai,Recon,Spoofing,Web-Based",
    )
    p.add_argument("--target-class", default="Benign")
    p.add_argument("--coalition-size", type=int, default=8)
    p.add_argument("--num-clients", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sources = [x.strip() for x in args.source_classes.split(",") if x.strip()]
    if args.target_class not in CLASS_NAMES:
        raise ValueError(f"Unknown target class: {args.target_class}")
    if any(source not in CLASS_NAMES for source in sources):
        raise ValueError("One or more source classes are unknown")
    if args.target_class in sources:
        raise ValueError("Target class cannot also be a source class")
    if not 1 <= args.coalition_size < args.num_clients:
        raise ValueError("Invalid coalition size")

    out = args.output_dir.expanduser().resolve()
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    arrays = load_protocol_arrays(args.data_file.expanduser().resolve())
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    partitions, partition_hash = load_fixed_partitions(
        args.partition_file.expanduser().resolve(),
        expected_clients=args.num_clients,
        train_rows=len(y_train),
    )

    count_rows = []
    coalition_rows = []
    manifest = {}
    for source in sources:
        source_id = CLASS_NAMES.index(source)
        counts = []
        for client_id, indices in enumerate(partitions):
            count = int(np.sum(y_train[indices] == source_id))
            counts.append((client_id, count))
            count_rows.append({
                "source_class": source,
                "source_id": source_id,
                "client_id": client_id,
                "clean_source_rows": count,
            })

        eligible = [(client_id, count) for client_id, count in counts if count > 0]
        if len(eligible) < args.coalition_size:
            raise RuntimeError(
                f"{source} has only {len(eligible)} source-capable clients"
            )

        # Frozen, attack-outcome-independent selection rule:
        # descending clean source-row count, then ascending client ID.
        if (
            source == "DDoS"
            and args.target_class == "Benign"
            and args.coalition_size == 8
        ):
            # Preserve the original development coalition as a positive
            # control. Unseen source classes use the source-capable rule.
            development_ids = [1, 7, 8, 10, 14, 15, 17, 18]
            count_map = dict(counts)
            if any(count_map[client_id] <= 0 for client_id in development_ids):
                raise RuntimeError(
                    "Original DDoS development coalition is not source-capable"
                )
            selected = [
                (client_id, count_map[client_id])
                for client_id in development_ids
            ]
            selected_ids = development_ids
            selection_rule = "original_ddos_development_coalition"
        else:
            selected = sorted(
                eligible,
                key=lambda item: (-item[1], item[0]),
            )[: args.coalition_size]
            selected_ids = sorted(client_id for client_id, _ in selected)
            selection_rule = (
                "top_k_clean_source_row_count_then_client_id"
            )
        global_rows = int(sum(count for _, count in counts))
        selected_rows = int(sum(count for _, count in selected))
        exposure = float(selected_rows / max(global_rows, 1))

        digest = hashlib.sha256()
        digest.update(source.encode("utf-8"))
        digest.update(args.target_class.encode("utf-8"))
        digest.update(np.asarray(selected_ids, dtype=np.int64).tobytes())
        coalition_hash = digest.hexdigest()

        row = {
            "source_class": source,
            "source_id": source_id,
            "target_class": args.target_class,
            "target_id": CLASS_NAMES.index(args.target_class),
            "coalition_size": args.coalition_size,
            "selected_clients": "|".join(map(str, selected_ids)),
            "global_source_rows": global_rows,
            "selected_source_rows": selected_rows,
            "global_source_exposure_fraction": exposure,
            "minimum_selected_client_source_rows": int(
                min(count for _, count in selected)
            ),
            "maximum_selected_client_source_rows": int(
                max(count for _, count in selected)
            ),
            "coalition_hash_sha256": coalition_hash,
            "selection_rule": selection_rule,
            "is_development_pair": bool(
                source == "DDoS" and args.target_class == "Benign"
            ),
        }
        coalition_rows.append(row)
        manifest[source] = {
            "selected_clients": selected_ids,
            "global_source_rows": global_rows,
            "selected_source_rows": selected_rows,
            "global_source_exposure_fraction": exposure,
            "coalition_hash_sha256": coalition_hash,
        }

    count_table = pd.DataFrame(count_rows)
    coalition_table = pd.DataFrame(coalition_rows)
    count_table["selected_for_source"] = False
    for row in coalition_rows:
        ids = {int(x) for x in row["selected_clients"].split("|")}
        mask = (
            count_table["source_class"].eq(row["source_class"])
            & count_table["client_id"].isin(ids)
        )
        count_table.loc[mask, "selected_for_source"] = True

    count_table.to_csv(tables / "v313_source_client_counts.csv", index=False)
    coalition_table.to_csv(
        tables / "v313_frozen_coalition_manifest.csv", index=False
    )
    with (out / "v313_coalition_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "experiment_version": "3.13",
            "partition_hash_sha256": partition_hash,
            "target_class": args.target_class,
            "coalition_size": args.coalition_size,
            "selection_rule": selection_rule,
            "selection_uses_attack_outcomes": False,
            "selection_uses_test_sets": False,
            "coalitions": manifest,
        }, handle, indent=2)

    print("V3.13 frozen source-capable coalitions created")
    print(coalition_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

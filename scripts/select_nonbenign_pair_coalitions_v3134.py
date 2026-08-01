#!/usr/bin/env python3
"""Create frozen source-capable coalitions for selected V3.13.4 pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

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
    p.add_argument("--pair-manifest", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--coalition-size", type=int, default=8)
    p.add_argument("--num-clients", type=int, default=20)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    if int(a.coalition_size) != 8 or int(a.num_clients) != 20:
        raise ValueError("V3.13.4 is frozen to 8 of 20 malicious clients")

    output = a.output_dir.expanduser().resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(a.pair_manifest.expanduser().resolve())
    if len(pairs) != 7:
        raise RuntimeError(
            f"Expected seven selected non-Benign pairs, found {len(pairs)}"
        )
    if (pairs["target_class"] == "Benign").any():
        raise RuntimeError("Non-Benign pair manifest contains Benign target")

    arrays = load_protocol_arrays(a.data_file.expanduser().resolve())
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    partitions, partition_hash = load_fixed_partitions(
        a.partition_file.expanduser().resolve(),
        expected_clients=int(a.num_clients),
        train_rows=len(y_train),
    )

    count_rows: List[Dict[str, object]] = []
    coalition_rows: List[Dict[str, object]] = []
    for _, pair in pairs.sort_values("pair_id").iterrows():
        source = str(pair["source_class"])
        target = str(pair["target_class"])
        source_id = CLASS_NAMES.index(source)
        target_id = CLASS_NAMES.index(target)
        counts = []
        for client_id, indices in enumerate(partitions):
            count = int(np.sum(y_train[indices] == source_id))
            counts.append((client_id, count))
            count_rows.append({
                "pair_name": str(pair["pair_name"]),
                "source_class": source,
                "target_class": target,
                "client_id": client_id,
                "clean_source_rows": count,
            })

        eligible = [
            (client_id, count)
            for client_id, count in counts
            if count > 0
        ]
        if len(eligible) < int(a.coalition_size):
            raise RuntimeError(
                f"{source} has only {len(eligible)} source-capable clients"
            )
        selected = sorted(
            eligible, key=lambda item: (-item[1], item[0])
        )[: int(a.coalition_size)]
        selected_ids = sorted(client_id for client_id, _ in selected)
        global_rows = int(sum(count for _, count in counts))
        selected_rows = int(sum(count for _, count in selected))
        exposure = float(selected_rows / max(global_rows, 1))

        h = hashlib.sha256()
        h.update(source.encode("utf-8"))
        h.update(target.encode("utf-8"))
        h.update(np.asarray(selected_ids, dtype=np.int64).tobytes())

        coalition_rows.append({
            "pair_id": int(pair["pair_id"]),
            "pair_name": str(pair["pair_name"]),
            "source_class": source,
            "source_id": source_id,
            "target_class": target,
            "target_id": target_id,
            "coalition_size": int(a.coalition_size),
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
            "coalition_hash_sha256": h.hexdigest(),
            "selection_rule": (
                "top_k_clean_source_row_count_then_client_id"
            ),
            "selection_uses_attack_outcomes": False,
            "selection_uses_test_sets": False,
            "is_development_pair": False,
        })

    coalition_table = pd.DataFrame(coalition_rows)
    count_table = pd.DataFrame(count_rows)
    count_table["selected_for_pair"] = False
    for _, row in coalition_table.iterrows():
        selected_ids = {
            int(x) for x in str(row["selected_clients"]).split("|")
        }
        mask = (
            count_table["pair_name"].eq(row["pair_name"])
            & count_table["client_id"].isin(selected_ids)
        )
        count_table.loc[mask, "selected_for_pair"] = True

    coalition_table.to_csv(
        tables / "v3134_frozen_pair_coalition_manifest.csv",
        index=False,
    )
    count_table.to_csv(
        tables / "v3134_pair_client_source_counts.csv",
        index=False,
    )
    with (output / "v3134_coalition_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "experiment_version": "3.13.4",
            "partition_hash_sha256": partition_hash,
            "coalition_size": int(a.coalition_size),
            "selection_rule":
                "top_k_clean_source_row_count_then_client_id",
            "selection_uses_attack_outcomes": False,
            "selection_uses_test_sets": False,
        }, handle, indent=2)

    print("V3.13.4 frozen pair coalitions created")
    print(coalition_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

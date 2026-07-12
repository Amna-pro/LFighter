#!/usr/bin/env python3
"""Validate CIC IoT-DIAD Protocol V2.1 outputs before benchmarking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPECTED_CLASSES = [
    "Benign",
    "BruteForce",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web-Based",
]
EXPECTED_SPLITS = ["train", "val", "test_diagnostic", "test_natural"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-root",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    root = args.protocol_root.expanduser().resolve()
    tables = root / "tables"
    metadata_path = root / "metadata_v2.json"
    distribution_path = tables / "prepared_class_distribution.csv"
    overlap_path = tables / "row_hash_overlap_audit.csv"

    for path in (metadata_path, distribution_path, overlap_path):
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    distribution = pd.read_csv(distribution_path)
    overlap = pd.read_csv(overlap_path)

    checks = []
    checks.append(
        {
            "check": "protocol_version_is_2_1",
            "passed": str(metadata.get("protocol_version")) == "2.1",
            "detail": str(metadata.get("protocol_version")),
        }
    )

    disallowed = overlap.loc[~overlap["allowed_overlap"].astype(bool)]
    max_disallowed = int(disallowed["overlap_rows"].max()) if len(disallowed) else 0
    checks.append(
        {
            "check": "no_disallowed_row_hash_overlap",
            "passed": max_disallowed == 0,
            "detail": str(max_disallowed),
        }
    )

    for split in EXPECTED_SPLITS:
        present = set(
            distribution.loc[
                (distribution["split"] == split) & (distribution["rows"] > 0),
                "class_name",
            ].astype(str)
        )
        missing = sorted(set(EXPECTED_CLASSES) - present)
        checks.append(
            {
                "check": f"all_classes_present_{split}",
                "passed": not missing,
                "detail": "none" if not missing else ",".join(missing),
            }
        )

    allowed_pair = overlap[
        overlap.apply(
            lambda row: {str(row["split_a"]), str(row["split_b"])}
            == {"test_diagnostic", "test_natural"},
            axis=1,
        )
    ]
    allowed_count = int(allowed_pair["overlap_rows"].iloc[0]) if len(allowed_pair) else -1
    checks.append(
        {
            "check": "alternative_test_overlap_explicitly_reported",
            "passed": allowed_count >= 0,
            "detail": str(allowed_count),
        }
    )

    result = pd.DataFrame(checks)
    result.to_csv(tables / "protocol_v21_validation.csv", index=False)
    summary = {
        "protocol_root": str(root),
        "passed": bool(result["passed"].all()),
        "checks": result.to_dict(orient="records"),
    }
    (root / "protocol_v21_validation.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(result.to_string(index=False))
    print()
    print(f"Overall validation passed: {summary['passed']}")
    print(f"CSV: {tables / 'protocol_v21_validation.csv'}")
    print(f"JSON: {root / 'protocol_v21_validation.json'}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

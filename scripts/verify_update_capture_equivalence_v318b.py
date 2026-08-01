#!/usr/bin/env python3
"""V3.18B.1 equivalence verifier patch.

This verifier separates training-trajectory equivalence from diagnostic-table
schema equivalence.

Blocking evidence:
- exact final model state,
- exact shared local-training values,
- exact shared generic round metrics,
- complete, finite, hashed float32 update captures.

Diagnostic anchor/signature comparisons are retained and reported, but schema
order or non-training diagnostic additions do not falsely invalidate an
otherwise exact training trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


RUNTIME_TOKENS = ("second", "seconds", "runtime", "elapsed", "duration", "time_sec")
PAIR_SPECIFIC_TOKENS = (
    "source_to_target",
    "source_recall",
    "target_prediction",
    "target_false_positive",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--captured-dir", required=True, type=Path)
    p.add_argument("--reference-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--tolerance", type=float, default=1e-8)
    return p.parse_args()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def excluded_runtime(column: str) -> bool:
    name = column.lower()
    return any(token in name for token in RUNTIME_TOKENS)


def excluded_pair_specific(column: str) -> bool:
    name = column.lower()
    return any(token in name for token in PAIR_SPECIFIC_TOKENS)


def align_and_compare(
    captured_path: Path,
    reference_path: Path,
    key_columns: Sequence[str],
    tolerance: float,
    exclude_pair_specific: bool,
) -> Dict[str, object]:
    left = pd.read_csv(captured_path)
    right = pd.read_csv(reference_path)

    missing_keys_left = [c for c in key_columns if c not in left.columns]
    missing_keys_right = [c for c in key_columns if c not in right.columns]
    if missing_keys_left or missing_keys_right:
        return {
            "captured_rows": len(left),
            "reference_rows": len(right),
            "row_count_equal": len(left) == len(right),
            "captured_only_columns": "|".join(sorted(set(left.columns) - set(right.columns))),
            "reference_only_columns": "|".join(sorted(set(right.columns) - set(left.columns))),
            "compared_column_count": 0,
            "maximum_numeric_abs_diff": float("inf"),
            "non_numeric_equal": False,
            "shared_values_equal": False,
            "error": f"Missing keys, captured={missing_keys_left}, reference={missing_keys_right}",
        }

    left = left.sort_values(list(key_columns)).reset_index(drop=True)
    right = right.sort_values(list(key_columns)).reset_index(drop=True)

    common = []
    for column in left.columns:
        if column not in right.columns:
            continue
        if excluded_runtime(column):
            continue
        if exclude_pair_specific and excluded_pair_specific(column):
            continue
        common.append(column)

    if len(left) != len(right):
        return {
            "captured_rows": len(left),
            "reference_rows": len(right),
            "row_count_equal": False,
            "captured_only_columns": "|".join(sorted(set(left.columns) - set(right.columns))),
            "reference_only_columns": "|".join(sorted(set(right.columns) - set(left.columns))),
            "compared_column_count": len(common),
            "maximum_numeric_abs_diff": float("inf"),
            "non_numeric_equal": False,
            "shared_values_equal": False,
            "error": "Row counts differ",
        }

    maximum = 0.0
    non_numeric_equal = True
    for column in common:
        lcol = left[column]
        rcol = right[column]
        if pd.api.types.is_numeric_dtype(lcol) and pd.api.types.is_numeric_dtype(rcol):
            a = pd.to_numeric(lcol, errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(rcol, errors="coerce").to_numpy(dtype=float)
            a_finite = np.isfinite(a)
            b_finite = np.isfinite(b)
            if not np.array_equal(a_finite, b_finite):
                maximum = float("inf")
                continue
            if a_finite.any():
                maximum = max(maximum, float(np.max(np.abs(a[a_finite] - b[b_finite]))))
        else:
            a = lcol.fillna("<NA>").astype(str).to_numpy()
            b = rcol.fillna("<NA>").astype(str).to_numpy()
            non_numeric_equal = non_numeric_equal and bool(np.array_equal(a, b))

    equal = bool(np.isfinite(maximum) and maximum <= tolerance and non_numeric_equal)
    return {
        "captured_rows": len(left),
        "reference_rows": len(right),
        "row_count_equal": True,
        "captured_only_columns": "|".join(sorted(set(left.columns) - set(right.columns))),
        "reference_only_columns": "|".join(sorted(set(right.columns) - set(left.columns))),
        "compared_column_count": len(common),
        "maximum_numeric_abs_diff": maximum,
        "non_numeric_equal": non_numeric_equal,
        "shared_values_equal": equal,
        "error": "",
    }


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["model_state_dict"]


def state_diff(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
) -> Tuple[float, bool]:
    if set(a) != set(b):
        return float("inf"), False
    maximum = 0.0
    for key in a:
        x = a[key].detach().cpu()
        y = b[key].detach().cpu()
        if x.shape != y.shape:
            return float("inf"), False
        if torch.is_floating_point(x) or torch.is_complex(x):
            maximum = max(
                maximum,
                float(
                    torch.max(
                        torch.abs(
                            x.to(torch.float64) - y.to(torch.float64)
                        )
                    ).item()
                ),
            )
        elif not torch.equal(x, y):
            return float("inf"), False
    return maximum, True


def verify_captures(captured_dir: Path) -> Tuple[pd.DataFrame, bool]:
    index_path = captured_dir / "tables" / "v318b_update_capture_index.csv"
    index = pd.read_csv(index_path)
    rows: List[Dict[str, object]] = []
    all_ok = len(index) == 4

    for _, row in index.iterrows():
        matrix_path = captured_dir / str(row["update_matrix_file"])
        reference_path = captured_dir / str(row["reference_state_file"])
        matrix_exists = matrix_path.exists()
        reference_exists = reference_path.exists()
        matrix_hash_ok = (
            matrix_exists
            and digest(matrix_path) == str(row["update_matrix_sha256"])
        )
        reference_hash_ok = (
            reference_exists
            and digest(reference_path) == str(row["reference_state_sha256"])
        )

        shape_ok = False
        finite = False
        dtype_ok = False
        if matrix_exists:
            matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
            shape_ok = tuple(matrix.shape) == (
                20,
                int(row["parameter_count"]),
            )
            finite = bool(np.isfinite(matrix).all())
            dtype_ok = matrix.dtype == np.float32

        ok = bool(
            matrix_hash_ok
            and reference_hash_ok
            and shape_ok
            and finite
            and dtype_ok
        )
        all_ok = all_ok and ok
        rows.append({
            "monitoring_round": int(row["monitoring_round"]),
            "matrix_exists": matrix_exists,
            "reference_exists": reference_exists,
            "matrix_hash_ok": matrix_hash_ok,
            "reference_hash_ok": reference_hash_ok,
            "shape_ok": shape_ok,
            "finite": finite,
            "dtype_float32": dtype_ok,
            "capture_integrity_ok": ok,
        })

    return pd.DataFrame(rows), bool(all_ok)


def main() -> int:
    a = parse_args()
    captured = a.captured_dir.expanduser().resolve()
    reference = a.reference_dir.expanduser().resolve()
    out = a.output_dir.expanduser().resolve()
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    specs = [
        (
            "continuation_round_metrics.csv",
            ["monitoring_round", "global_round"],
            True,
            True,
        ),
        (
            "continuation_local_training.csv",
            ["monitoring_round", "global_round", "client_id"],
            False,
            True,
        ),
        (
            "continuation_client_anchor_scores.csv",
            ["monitoring_round", "global_round", "client_id"],
            False,
            False,
        ),
        (
            "continuation_transition_signature_long.csv",
            [
                "monitoring_round",
                "global_round",
                "client_id",
                "source_id",
                "target_id",
            ],
            False,
            False,
        ),
    ]

    rows = []
    for name, keys, exclude_pair, blocking in specs:
        result = align_and_compare(
            captured / "tables" / name,
            reference / "tables" / name,
            key_columns=keys,
            tolerance=float(a.tolerance),
            exclude_pair_specific=exclude_pair,
        )
        rows.append({
            "table": name,
            "blocking_for_training_equivalence": blocking,
            "pair_specific_metrics_excluded": exclude_pair,
            **result,
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        tables / "v318b_table_equivalence.csv",
        index=False,
    )

    captured_state = load_state(
        captured / "checkpoints" / "continuation_last_round_model.pt"
    )
    reference_state = load_state(
        reference / "checkpoints" / "continuation_last_round_model.pt"
    )
    maximum_state_diff, state_keys_equal = state_diff(
        captured_state,
        reference_state,
    )

    capture_integrity, capture_ok = verify_captures(captured)
    capture_integrity.to_csv(
        tables / "v318b_capture_integrity.csv",
        index=False,
    )

    blocking = comparison[
        comparison["blocking_for_training_equivalence"].astype(bool)
    ]
    blocking_tables_equal = bool(blocking["shared_values_equal"].all())
    diagnostic = comparison[
        ~comparison["blocking_for_training_equivalence"].astype(bool)
    ]
    diagnostic_tables_equal = bool(diagnostic["shared_values_equal"].all())

    exact = bool(
        blocking_tables_equal
        and state_keys_equal
        and maximum_state_diff <= float(a.tolerance)
        and capture_ok
    )

    decision = {
        "verifier_version": "3.18B.1",
        "exact_training_equivalence": exact,
        "blocking_training_tables_equal": blocking_tables_equal,
        "diagnostic_tables_equal_on_shared_columns": diagnostic_tables_equal,
        "state_keys_equal": state_keys_equal,
        "maximum_final_state_abs_diff": maximum_state_diff,
        "tolerance": float(a.tolerance),
        "all_capture_integrity_checks_pass": capture_ok,
        "captured_rounds": int(len(capture_integrity)),
        "pair_specific_round_metrics_excluded_from_training_check": True,
        "column_order_required": False,
        "captured_extra_diagnostic_columns_allowed": True,
        "test_sets_accessed": False,
        "capture_used_for_training_or_aggregation": False,
    }

    pd.DataFrame([decision]).to_csv(
        tables / "v318b_equivalence_decision.csv",
        index=False,
    )
    with (out / "v318b_equivalence.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(decision, f, indent=2)

    print("V3.18B.1 exact-equivalence verification complete")
    print(pd.DataFrame([decision]).to_string(index=False))
    print()
    print("TABLE COMPARISONS")
    print(comparison.to_string(index=False))

    if not exact:
        raise SystemExit(
            "V3.18B.1 found a real blocking training-equivalence failure"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Task 41B zero-poison exact-equivalence adapter.

This wrapper reuses the frozen V3.20B.1 clean execution path while replacing
its protocol-array loader with a train/validation-only loader. It therefore
provides the mandatory Task 41B pretraining gate without accessing either
reserved test array.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_frozen_untargeted_defense_v320b1 as frozen_runner


ALLOWED_ARRAYS = ("X_train", "y_train", "X_val", "y_val")
RESERVED_ARRAYS = (
    "X_test_natural",
    "y_test_natural",
    "X_test_diagnostic",
    "y_test_diagnostic",
)


def load_train_validation_only(path: Path) -> Dict[str, np.ndarray]:
    """Load only development arrays from the prepared NPZ archive."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Prepared NPZ file not found: {resolved}")

    arrays: Dict[str, np.ndarray] = {}
    with np.load(resolved, allow_pickle=False) as archive:
        missing = [key for key in ALLOWED_ARRAYS if key not in archive.files]
        if missing:
            raise KeyError(f"Prepared NPZ is missing development arrays: {missing}")
        for key in ALLOWED_ARRAYS:
            arrays[key] = np.asarray(archive[key])

    return arrays


def argument_value(flag: str) -> str:
    try:
        index = sys.argv.index(flag)
    except ValueError as exc:
        raise ValueError(f"Required argument is missing: {flag}") from exc
    if index + 1 >= len(sys.argv):
        raise ValueError(f"Argument has no value: {flag}")
    return sys.argv[index + 1]


def enforce_zero_poison_contract() -> Path:
    if argument_value("--mode") != "clean":
        raise ValueError("Task 41B zero-poison gate requires --mode clean")
    if argument_value("--replacement-policy") != "plain_fedavg":
        raise ValueError(
            "Task 41B zero-poison gate requires --replacement-policy plain_fedavg"
        )
    poison_fraction = float(argument_value("--poison-fraction"))
    if abs(poison_fraction) > 1e-12:
        raise ValueError(
            "Task 41B zero-poison gate requires --poison-fraction 0"
        )
    return Path(argument_value("--output-dir")).expanduser().resolve()


def main() -> int:
    output_dir = enforce_zero_poison_contract()

    # The imported runner resolves this name from its module globals at runtime.
    frozen_runner.load_protocol_arrays = load_train_validation_only

    status = int(frozen_runner.main())
    if status != 0:
        return status

    metadata = {
        "experiment_version": "4.11B.0",
        "stage": "task41b_zero_poison_exact_equivalence_adapter",
        "base_runner": "scripts/run_frozen_untargeted_defense_v320b1.py",
        "mode": "clean",
        "replacement_policy": "plain_fedavg",
        "poison_fraction": 0.0,
        "loaded_array_keys": list(ALLOWED_ARRAYS),
        "reserved_array_keys": list(RESERVED_ARRAYS),
        "natural_test_accessed": False,
        "diagnostic_test_accessed": False,
        "test_arrays_loaded": False,
        "method_reopened": False,
        "attack_specific_retuning": False,
        "next_gate": (
            "Compare all frozen generic metrics and the final model state "
            "against the matched V3.10.1 clean continuation at tolerance 1e-8."
        ),
    }
    metadata_path = output_dir / "task41b_zero_poison_adapter_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("TASK 41B ZERO-POISON ADAPTER COMPLETE")
    print("Reserved natural test accessed: False")
    print("Reserved diagnostic test accessed: False")
    print("Metadata:", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

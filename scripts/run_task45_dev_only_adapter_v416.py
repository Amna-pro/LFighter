#!/usr/bin/env python3
"""Run approved frozen Task 45 runners without materializing reserved tests.

The inherited ``load_protocol_arrays`` helper eagerly loads every NPZ array.
Task 45 development permits only train and validation arrays. This adapter
patches the runner module's imported loader with a strict development-only
loader. It does not modify training, attack, detector, aggregation, or
reconstruction logic.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


ALLOWED_ARRAYS = ("X_train", "y_train", "X_val", "y_val")
RESERVED_ARRAYS = (
    "X_test_natural",
    "y_test_natural",
    "X_test_diagnostic",
    "y_test_diagnostic",
)
APPROVED_RUNNERS = {
    "run_exact_plain_fedavg_v3132.py",
    "run_frozen_reconstruction_v3123.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 45 development-only array isolation adapter."
    )
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments for the approved runner, optionally preceded by --.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def output_dir_from_args(arguments: List[str]) -> Path:
    if "--output-dir" not in arguments:
        raise ValueError("Approved runner invocation is missing --output-dir")
    index = arguments.index("--output-dir")
    if index + 1 >= len(arguments):
        raise ValueError("--output-dir has no value")
    return Path(arguments[index + 1]).expanduser().resolve()


def main() -> int:
    args = parse_args()
    runner_path = args.runner.expanduser().resolve()
    if runner_path.name not in APPROVED_RUNNERS:
        raise ValueError(
            f"Runner {runner_path.name!r} is not approved for Task 45"
        )
    if not runner_path.exists():
        raise FileNotFoundError(runner_path)

    runner_args = list(args.runner_args)
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    output_dir = output_dir_from_args(runner_args)

    loaded_arrays: Dict[str, Dict[str, object]] = {}
    data_file_holder: Dict[str, str] = {}

    def development_only_loader(path: Path) -> Dict[str, np.ndarray]:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        result: Dict[str, np.ndarray] = {}
        with np.load(resolved) as payload:
            available = set(payload.files)
            missing = sorted(set(ALLOWED_ARRAYS) - available)
            if missing:
                raise KeyError(f"Development NPZ is missing arrays: {missing}")
            for key in ALLOWED_ARRAYS:
                value = payload[key]
                result[key] = value
                loaded_arrays[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": array_sha256(value),
                }
            data_file_holder["path"] = str(resolved)
            data_file_holder["reserved_names_present_in_container"] = str(
                bool(set(RESERVED_ARRAYS) & available)
            )
        return result

    module_name = f"task45_approved_{runner_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import approved runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "load_protocol_arrays"):
        raise RuntimeError("Approved runner has no patchable load_protocol_arrays")
    module.load_protocol_arrays = development_only_loader

    federated_module = sys.modules.get("federated_iot_v26")
    if federated_module is not None:
        setattr(federated_module, "load_protocol_arrays", development_only_loader)

    previous_argv = sys.argv
    sys.argv = [str(runner_path), *runner_args]
    try:
        return_code = int(module.main())
    finally:
        sys.argv = previous_argv

    if set(loaded_arrays) != set(ALLOWED_ARRAYS):
        raise RuntimeError(
            f"Unexpected development array access: {sorted(loaded_arrays)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_version": "4.16.0",
        "stage": "task45_development_only_loader_adapter",
        "approved_runner": str(runner_path),
        "approved_runner_sha256": file_sha256(runner_path),
        "data_file": data_file_holder.get("path"),
        "allowed_arrays_materialized": loaded_arrays,
        "reserved_array_names": list(RESERVED_ARRAYS),
        "reserved_arrays_materialized": False,
        "runner_return_code": return_code,
    }
    (output_dir / "task45_dev_only_adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

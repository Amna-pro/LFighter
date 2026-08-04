#!/usr/bin/env python3
"""
Task 41C.1b clean-calibration interface audit.

Purpose:
- validate the frozen C0 protocol and valid C1a preflight,
- inspect the exact source interfaces and clean calibration artifact schemas
  needed to implement D1, D2, and D3,
- inspect CSV headers, JSON keys, and NPZ headers only,
- exclude every path containing reserved natural/diagnostic test names before
  opening, hashing, or stat-based schema inspection,
- produce CSV/JSON tables plus PNG/PDF figures.

This script performs no training, no attack execution, and loads no dataset
arrays.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib import format as npformat


EXPECTED_BRANCH = "feature/cic-iot-diad-task41c-backdoor-extension-v1"
EXPECTED_C0_TAG = "task41c-c0-preregistered-v4.12c0"
EXPECTED_C0_COMMIT_PREFIX = "a605541"
EXPECTED_SEEDS = [7, 99, 123, 2026]
RESERVED_TOKENS = ("test_natural", "test_diagnostic")

SOURCE_FILES = (
    "scripts/run_task41b_backdoor_smoke_v411b1.py",
    "scripts/build_frozen_reconstruction_calibration_v3123.py",
    "scripts/run_frozen_reconstruction_v3123.py",
    "scripts/run_true_warmup_v310.py",
    "scripts/run_post_warmup_capture_v310.py",
    "scripts/run_update_capture_v318b.py",
    "scripts/audit_robust_centers_v319a.py",
    "src/independent_anchor_v310.py",
    "src/trusted_update_reconstruction_v312.py",
)

KEYWORDS = (
    "client_id",
    "monitoring_round",
    "probe",
    "anchor",
    "threshold",
    "score",
    "signature",
    "update",
    "gradient",
    "center",
    "residual",
    "malicious",
    "benign",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Task 41C clean calibration implementation interfaces."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--clean-seed-root", type=Path, required=True)
    parser.add_argument("--warmup-root", type=Path, required=True)
    parser.add_argument("--reconstruction-root", type=Path, required=True)
    parser.add_argument("--trigger-spec-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_reserved_name(path: Path) -> bool:
    lower = str(path).replace("\\", "/").lower()
    return any(token in lower for token in RESERVED_TOKENS)


def save_figure(figure: plt.Figure, base: Path) -> None:
    figure.tight_layout()
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_source(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text, filename=str(path))

    definitions: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []
    keywords: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.append(
                {
                    "name": node.name,
                    "definition_type": type(node).__name__,
                    "line": int(getattr(node, "lineno", -1)),
                }
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {
                        "module": alias.name,
                        "import_type": "import",
                        "line": int(node.lineno),
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            imports.append(
                {
                    "module": node.module or "",
                    "import_type": "from",
                    "line": int(node.lineno),
                }
            )

    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        lower = line.lower()
        for keyword in KEYWORDS:
            if keyword in lower:
                keywords.append(
                    {
                        "keyword": keyword,
                        "line": line_number,
                        "text": line.strip()[:300],
                    }
                )

    return definitions, imports, keywords


def read_npy_header(handle) -> tuple[tuple[int, ...], bool, str]:
    version = npformat.read_magic(handle)
    if version == (1, 0):
        shape, fortran_order, dtype = npformat.read_array_header_1_0(handle)
    else:
        shape, fortran_order, dtype = npformat.read_array_header_2_0(handle)
    return tuple(int(value) for value in shape), bool(fortran_order), str(dtype)


def npz_header_inventory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path, "r") as archive:
        for member in sorted(archive.namelist()):
            if not member.endswith(".npy"):
                continue
            with archive.open(member, "r") as handle:
                shape, fortran_order, dtype = read_npy_header(handle)
            rows.append(
                {
                    "array_key": member[:-4],
                    "shape": "x".join(str(value) for value in shape),
                    "dtype": dtype,
                    "fortran_order": fortran_order,
                    "header_only": True,
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    root = resolve(args.project_root)
    clean_root = resolve(args.clean_seed_root)
    warmup_root = resolve(args.warmup_root)
    reconstruction_root = resolve(args.reconstruction_root)
    trigger_spec_file = resolve(args.trigger_spec_file)
    output = resolve(args.output_dir)

    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    protocol_path = root / "configs" / "task41c_preregistration_v412c0.json"
    c0_decision_path = (
        root
        / "results"
        / "cic_iot_diad_task41c_preregistration_v412c0"
        / "task41c0_preregistration_decision.json"
    )
    c1a_decision_path = (
        root
        / "results"
        / "cic_iot_diad_task41c_clean_calibration_v412c1a"
        / "preflight"
        / "task41c1_clean_preflight_decision.json"
    )

    required_paths = [
        protocol_path,
        c0_decision_path,
        c1a_decision_path,
        clean_root,
        warmup_root,
        reconstruction_root,
        trigger_spec_file,
    ]
    required_paths.extend(root / relative for relative in SOURCE_FILES)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    c0_decision = json.loads(c0_decision_path.read_text(encoding="utf-8"))
    c1a_decision = json.loads(c1a_decision_path.read_text(encoding="utf-8"))

    branch = git_output(root, "branch", "--show-current")
    head = git_output(root, "rev-parse", "HEAD")
    c0_tag_commit = git_output(root, "rev-list", "-n", "1", EXPECTED_C0_TAG)

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append(
            {"check_name": name, "passed": bool(passed), "detail": str(detail)}
        )
        if not passed:
            raise RuntimeError(f"{name} failed: {detail}")

    check("branch", branch == EXPECTED_BRANCH, branch)
    check(
        "c0_tag_commit",
        c0_tag_commit.startswith(EXPECTED_C0_COMMIT_PREFIX),
        c0_tag_commit,
    )
    check(
        "protocol_version",
        protocol.get("experiment_version") == "4.12C.0",
        protocol.get("experiment_version"),
    )
    check(
        "c0_frozen",
        c0_decision.get("preregistration_frozen") is True,
        c0_decision.get("preregistration_frozen"),
    )
    check(
        "c1a_allowed",
        c1a_decision.get("c1_clean_calibration_allowed") is True,
        c1a_decision.get("c1_clean_calibration_allowed"),
    )
    check(
        "attack_execution_blocked",
        c1a_decision.get("attack_execution_allowed") is False,
        c1a_decision.get("attack_execution_allowed"),
    )
    check(
        "c1a_ready_seeds",
        int(c1a_decision.get("ready_seed_count", -1)) == 4,
        c1a_decision.get("ready_seed_count"),
    )
    check(
        "task41b_not_reopened",
        c1a_decision.get("task41b_reopened") is False,
        c1a_decision.get("task41b_reopened"),
    )
    check(
        "reserved_test_not_accessed",
        c1a_decision.get("reserved_test_accessed") is False,
        c1a_decision.get("reserved_test_accessed"),
    )
    check(
        "frozen_seeds",
        protocol.get("frozen_seeds") == EXPECTED_SEEDS,
        protocol.get("frozen_seeds"),
    )

    source_summary_rows: list[dict[str, Any]] = []
    definition_rows: list[dict[str, Any]] = []
    import_rows: list[dict[str, Any]] = []
    keyword_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for relative in SOURCE_FILES:
        path = root / relative
        definitions, imports, keyword_locations = parse_source(path)
        source_summary_rows.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "top_level_definition_count": len(definitions),
                "import_count": len(imports),
                "keyword_location_count": len(keyword_locations),
            }
        )
        for row in definitions:
            definition_rows.append({"relative_path": relative, **row})
        for row in imports:
            import_rows.append({"relative_path": relative, **row})
        for row in keyword_locations:
            keyword_rows.append({"relative_path": relative, **row})
        manifest_rows.append(
            {
                "role": "research_source",
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    schema_rows: list[dict[str, Any]] = []
    npz_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    artifact_roots: list[tuple[int, str, Path]] = []
    for seed in EXPECTED_SEEDS:
        artifact_roots.extend(
            [
                (seed, "clean_seed", clean_root / f"seed_{seed}"),
                (seed, "warmup", warmup_root / f"seed_{seed}" / "warmup"),
                (
                    seed,
                    "reconstruction_calibration",
                    reconstruction_root / f"seed_{seed}" / "calibration",
                ),
            ]
        )

    for seed, category, artifact_root in artifact_roots:
        for path in sorted(
            candidate for candidate in artifact_root.rglob("*") if candidate.is_file()
        ):
            if is_reserved_name(path):
                excluded_rows.append(
                    {
                        "seed": seed,
                        "category": category,
                        "relative_path": str(path.relative_to(root)),
                        "reason": "reserved_name_reference_excluded_before_open_stat_hash",
                        "opened": False,
                        "hashed": False,
                    }
                )
                continue

            suffix = path.suffix.lower()
            relative_path = str(path.relative_to(root))
            manifest_rows.append(
                {
                    "role": category,
                    "relative_path": relative_path,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

            if suffix == ".csv":
                header = pd.read_csv(path, nrows=0)
                schema_rows.append(
                    {
                        "seed": seed,
                        "category": category,
                        "relative_path": relative_path,
                        "file_type": "csv",
                        "column_count": len(header.columns),
                        "columns_or_keys": "|".join(str(value) for value in header.columns),
                        "schema_only": True,
                    }
                )
            elif suffix == ".json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        keys = sorted(str(value) for value in payload.keys())
                    else:
                        keys = [f"root_type:{type(payload).__name__}"]
                except Exception as exc:
                    keys = [f"json_read_error:{type(exc).__name__}"]
                schema_rows.append(
                    {
                        "seed": seed,
                        "category": category,
                        "relative_path": relative_path,
                        "file_type": "json",
                        "column_count": len(keys),
                        "columns_or_keys": "|".join(keys),
                        "schema_only": True,
                    }
                )
            elif suffix == ".npz":
                for row in npz_header_inventory(path):
                    npz_rows.append(
                        {
                            "seed": seed,
                            "category": category,
                            "relative_path": relative_path,
                            **row,
                        }
                    )
            elif suffix == ".pt":
                schema_rows.append(
                    {
                        "seed": seed,
                        "category": category,
                        "relative_path": relative_path,
                        "file_type": "pt",
                        "column_count": 0,
                        "columns_or_keys": "not_opened",
                        "schema_only": True,
                    }
                )

    trigger_payload = json.loads(trigger_spec_file.read_text(encoding="utf-8"))
    trigger_keys = sorted(str(value) for value in trigger_payload.keys())
    trigger_summary = {
        "relative_path": str(trigger_spec_file.relative_to(root)),
        "root_keys": "|".join(trigger_keys),
        "bytes": trigger_spec_file.stat().st_size,
        "sha256": sha256_file(trigger_spec_file),
        "reserved_test_accessed": False,
    }
    manifest_rows.append(
        {
            "role": "frozen_trigger_spec",
            "relative_path": str(trigger_spec_file.relative_to(root)),
            "bytes": trigger_spec_file.stat().st_size,
            "sha256": sha256_file(trigger_spec_file),
        }
    )

    schema_frame = pd.DataFrame(schema_rows)
    npz_frame = pd.DataFrame(npz_rows)
    excluded_frame = pd.DataFrame(excluded_rows)

    check(
        "source_files_audited",
        len(source_summary_rows) == len(SOURCE_FILES),
        len(source_summary_rows),
    )
    check(
        "clean_schema_inventory_nonempty",
        len(schema_frame) > 0,
        len(schema_frame),
    )
    check(
        "npz_headers_inspected",
        len(npz_frame) > 0,
        len(npz_frame),
    )
    check(
        "reserved_paths_never_opened",
        (
            excluded_frame.empty
            or (
                (~excluded_frame["opened"].astype(bool)).all()
                and (~excluded_frame["hashed"].astype(bool)).all()
            )
        ),
        len(excluded_frame),
    )

    pd.DataFrame(checks).to_csv(
        tables / "task41c1b_interface_checks.csv", index=False
    )
    pd.DataFrame(source_summary_rows).to_csv(
        tables / "task41c1b_source_summary.csv", index=False
    )
    pd.DataFrame(definition_rows).to_csv(
        tables / "task41c1b_source_definitions.csv", index=False
    )
    pd.DataFrame(import_rows).to_csv(
        tables / "task41c1b_import_inventory.csv", index=False
    )
    pd.DataFrame(keyword_rows).to_csv(
        tables / "task41c1b_keyword_locations.csv", index=False
    )
    schema_frame.to_csv(
        tables / "task41c1b_clean_artifact_schemas.csv", index=False
    )
    npz_frame.to_csv(
        tables / "task41c1b_npz_header_inventory.csv", index=False
    )
    excluded_frame.to_csv(
        tables / "task41c1b_excluded_reserved_name_references.csv",
        index=False,
    )
    pd.DataFrame([trigger_summary]).to_csv(
        tables / "task41c1b_trigger_spec_summary.csv", index=False
    )
    pd.DataFrame(manifest_rows).to_csv(
        tables / "task41c1b_input_manifest_sha256.csv", index=False
    )

    source_frame = pd.DataFrame(source_summary_rows)
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(
        source_frame["relative_path"].str.replace("scripts/", "", regex=False).str.replace(
            "src/", "", regex=False
        ),
        source_frame["keyword_location_count"],
    )
    axis.set_ylabel("Relevant keyword locations")
    axis.set_title("Task 41C clean-calibration source interface coverage")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, figures / "task41c1b_source_interface_coverage")

    if not schema_frame.empty:
        category_counts = (
            schema_frame.groupby(["category", "file_type"]).size().unstack(fill_value=0)
        )
        figure, axis = plt.subplots(figsize=(10, 6))
        category_counts.plot(kind="bar", ax=axis)
        axis.set_ylabel("Schema files")
        axis.set_title("Task 41C clean calibration artifact schemas")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
        save_figure(figure, figures / "task41c1b_artifact_schema_inventory")

    contract = {
        "experiment_version": "4.12C.1b",
        "stage": "task41c_clean_calibration_interface_audit",
        "git_branch": branch,
        "git_head": head,
        "source_file_count": len(source_summary_rows),
        "schema_file_count": len(schema_frame),
        "npz_header_count": len(npz_frame),
        "excluded_reserved_name_reference_count": len(excluded_frame),
        "reserved_name_references_opened": False,
        "reserved_name_references_hashed": False,
        "dataset_arrays_loaded": False,
        "attack_execution_performed": False,
        "task41b_reopened": False,
        "reserved_test_accessed": False,
        "c1_implementation_allowed": True,
        "next_stage": (
            "Use the audited schemas to implement D1 trigger-response shift, "
            "D2 trigger-gradient alignment, and D3 equal-rank fusion. "
            "Calibration must use clean warmup and independent-anchor evidence only."
        ),
    }
    (output / "task41c1b_interface_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )

    print("===== TASK 41C.1B CLEAN-CALIBRATION INTERFACE AUDIT =====")
    print("Integrity checks passed:", f"{sum(row['passed'] for row in checks)}/{len(checks)}")
    print("Source files audited:", len(source_summary_rows))
    print("Clean schema files:", len(schema_frame))
    print("NPZ headers inspected:", len(npz_frame))
    print("Excluded reserved-name references:", len(excluded_frame))
    print("RESERVED-NAME REFERENCES OPENED: False")
    print("RESERVED-NAME REFERENCES HASHED: False")
    print("DATASET ARRAYS LOADED: False")
    print("ATTACK EXECUTION PERFORMED: False")
    print("TASK 41B REOPENED: False")
    print("TEST SETS ACCESSED: False")
    print("C1 IMPLEMENTATION ALLOWED: True")
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Static audit for Task 41C.2a seed-7 screen implementation.

This audit parses source and contract files only. It performs no training,
attack execution, dataset-array loading, threshold fitting, or test access.
It emits CSV, JSON, PNG, and PDF evidence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_VERSION = "4.12C.2a"
EXPECTED_SEED = 7
EXPECTED_ATTACK_SEED = 42
EXPECTED_CLIENTS = 20
EXPECTED_MALICIOUS = [1, 7, 8, 10, 14, 15, 17, 18]
EXPECTED_ROUNDS = 4
EXPECTED_FRACTIONS = [0.005, 0.01, 0.02]
EXPECTED_ARMS = [
    "plain_fedavg",
    "D0_frozen_task40_lfighter",
    "D1_trigger_response_shift",
    "D2_trigger_gradient_alignment",
    "D3_equal_rank_fusion",
]
EXPECTED_RESERVED = {
    "X_test_natural",
    "y_test_natural",
    "X_test_diagnostic",
    "y_test_diagnostic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--script-file", type=Path, required=True)
    parser.add_argument("--contract-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(figure: plt.Figure, base: Path) -> None:
    figure.tight_layout()
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def literal_assignments(tree: ast.AST) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return values


def function_names(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def forbidden_literal_subscripts(tree: ast.AST) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.slice
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            if value.value in EXPECTED_RESERVED:
                rows.append(
                    {
                        "line": int(getattr(node, "lineno", -1)),
                        "array_name": value.value,
                    }
                )
    return rows


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    script_path = args.script_file.expanduser().resolve()
    contract_path = args.contract_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for path in (project_root, script_path, contract_path):
        if not path.exists():
            raise FileNotFoundError(path)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    assignments = literal_assignments(tree)
    functions = function_names(tree)
    imports = imported_names(tree)
    forbidden_subscripts = forbidden_literal_subscripts(tree)

    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append(
            {
                "check_name": name,
                "passed": bool(passed),
                "detail": str(detail),
            }
        )
        if not passed:
            raise RuntimeError(f"{name} failed: {detail}")

    check(
        "contract_version",
        contract.get("experiment_version") == EXPECTED_VERSION,
        contract.get("experiment_version"),
    )
    check(
        "script_version",
        assignments.get("EXPERIMENT_VERSION") == EXPECTED_VERSION,
        assignments.get("EXPERIMENT_VERSION"),
    )
    check(
        "script_sha256",
        sha256(script_path) == contract.get("script_sha256"),
        sha256(script_path),
    )
    check(
        "model_seed",
        assignments.get("MODEL_SEED") == EXPECTED_SEED,
        assignments.get("MODEL_SEED"),
    )
    check(
        "attack_seed",
        assignments.get("ATTACK_SEED") == EXPECTED_ATTACK_SEED,
        assignments.get("ATTACK_SEED"),
    )
    check(
        "num_clients",
        assignments.get("NUM_CLIENTS") == EXPECTED_CLIENTS,
        assignments.get("NUM_CLIENTS"),
    )
    check(
        "malicious_clients",
        list(assignments.get("MALICIOUS_CLIENTS", []))
        == EXPECTED_MALICIOUS,
        assignments.get("MALICIOUS_CLIENTS"),
    )
    check(
        "continuation_rounds",
        assignments.get("CONTINUATION_ROUNDS") == EXPECTED_ROUNDS,
        assignments.get("CONTINUATION_ROUNDS"),
    )
    check(
        "poison_fractions",
        list(assignments.get("POISON_FRACTIONS", []))
        == EXPECTED_FRACTIONS,
        assignments.get("POISON_FRACTIONS"),
    )
    resolved_arms = [
        assignments.get("PLAIN_ARM"),
        *list(assignments.get("CANDIDATES", [])),
    ]
    check(
        "candidate_arms",
        resolved_arms == EXPECTED_ARMS,
        resolved_arms,
    )
    check(
        "d3_equal_weights",
        contract.get("D3_weights") == [1.0, 1.0, 1.0],
        contract.get("D3_weights"),
    )
    check(
        "frozen_threshold_source",
        contract.get("threshold_source")
        == "seed_7_task41c1c3_clean_calibration",
        contract.get("threshold_source"),
    )
    check(
        "selection_rule_count",
        len(contract.get("selection_rule", [])) == 6,
        len(contract.get("selection_rule", [])),
    )
    check(
        "preflight_supported",
        "--preflight-only" in source
        and "write_preflight" in functions,
        "--preflight-only",
    )
    check(
        "resume_supported",
        "--resume" in source
        and "branch_complete" in functions
        and "archive_partial_branch" in functions,
        "branch resume functions",
    )
    check(
        "exact_pair_hash_supported",
        "exact_pair_hash" in functions
        and "exact_pair_hash_sha256" in source,
        "exact_pair_hash",
    )
    check(
        "manifest_supported",
        "write_output_manifest" in functions
        and "task41c2_summary_manifest_sha256.csv" in source,
        "write_output_manifest",
    )
    check(
        "required_detector_helpers_imported",
        {
            "raw_features",
            "apply_feature_calibration",
            "response_vector",
            "trigger_gradient_panel",
            "positive_z",
            "reconstruct_update",
        }.issubset(imports),
        sorted(imports),
    )
    check(
        "reserved_literal_subscripts_absent",
        len(forbidden_subscripts) == 0,
        forbidden_subscripts,
    )
    check(
        "development_array_loader_restricted",
        'for key in ALLOWED_ARRAYS' in source
        and 'arrays[key] = np.asarray(archive[key])' in source,
        "ALLOWED_ARRAYS-only loader",
    )
    check(
        "A2_not_in_condition_panel",
        "A2_clean_label_source_specific" not in source,
        "A2 absent",
    )
    check(
        "adaptive_attacks_not_in_condition_panel",
        "A3_detector_aware_objective" not in source
        and "A4_reconstruction_aware_scaled_update" not in source,
        "A3/A4 absent",
    )
    check(
        "retuning_prohibited",
        contract.get("threshold_retuning_allowed") is False
        and contract.get("profile_retuning_allowed") is False
        and contract.get("weight_retuning_allowed") is False,
        "retuning flags",
    )
    check(
        "task41b_reopening_prohibited",
        contract.get("task41b_reopening_allowed") is False,
        contract.get("task41b_reopening_allowed"),
    )
    check(
        "final_claims_blocked",
        contract.get("final_claims_allowed") is False,
        contract.get("final_claims_allowed"),
    )

    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(
        tables_dir / "task41c2_implementation_checks.csv",
        index=False,
    )
    pd.DataFrame(forbidden_subscripts).to_csv(
        tables_dir / "task41c2_forbidden_subscript_scan.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "relative_path": str(script_path.relative_to(project_root)),
                "bytes": script_path.stat().st_size,
                "sha256": sha256(script_path),
                "role": "c2_runner",
            },
            {
                "relative_path": str(contract_path.relative_to(project_root)),
                "bytes": contract_path.stat().st_size,
                "sha256": sha256(contract_path),
                "role": "c2_contract",
            },
            {
                "relative_path": str(
                    Path(__file__).resolve().relative_to(project_root)
                ),
                "bytes": Path(__file__).resolve().stat().st_size,
                "sha256": sha256(Path(__file__).resolve()),
                "role": "c2_static_audit",
            },
        ]
    ).to_csv(
        tables_dir / "task41c2_implementation_manifest_sha256.csv",
        index=False,
    )

    figure, axis = plt.subplots(figsize=(10, 5.8))
    axis.bar(
        checks_frame["check_name"],
        checks_frame["passed"].astype(int),
    )
    axis.set_ylim(0, 1.1)
    axis.set_ylabel("Pass")
    axis.set_title("Task 41C C2 implementation integrity")
    axis.tick_params(axis="x", rotation=70)
    axis.grid(axis="y", alpha=0.25)
    save_figure(
        figure,
        figures_dir / "task41c2_implementation_integrity",
    )

    decision = {
        "experiment_version": EXPECTED_VERSION,
        "stage": "task41c_c2_static_implementation_audit",
        "integrity_check_count": len(checks_frame),
        "integrity_checks_passed": int(checks_frame["passed"].sum()),
        "implementation_frozen_ready": True,
        "attack_execution_performed": False,
        "dataset_arrays_loaded": False,
        "reserved_test_accessed": False,
        "A2_executed": False,
        "adaptive_attack_executed": False,
        "thresholds_retuned": False,
        "profiles_retuned": False,
        "weights_retuned": False,
        "task41b_reopened": False,
        "c2_local_preflight_allowed": True,
    }
    (output_dir / "task41c2_implementation_audit_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print("===== TASK 41C.2 STATIC IMPLEMENTATION AUDIT =====")
    print(
        "Integrity checks passed:",
        f"{decision['integrity_checks_passed']}/"
        f"{decision['integrity_check_count']}",
    )
    print("IMPLEMENTATION FROZEN READY: True")
    print("C2 LOCAL PREFLIGHT ALLOWED: True")
    print("ATTACK EXECUTION PERFORMED: False")
    print("DATASET ARRAYS LOADED: False")
    print("A2 EXECUTED: False")
    print("ADAPTIVE ATTACK EXECUTED: False")
    print("THRESHOLDS RETUNED: False")
    print("PROFILES RETUNED: False")
    print("WEIGHTS RETUNED: False")
    print("TASK 41B REOPENED: False")
    print("TEST SETS ACCESSED: False")
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

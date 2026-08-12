#!/usr/bin/env python3
"""Audit Task 45 C0 without importing or executing training code."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PARENT_COMMIT = "cc84bc343508ad355064b03e89645fa165a6daad"
PARENT_TAG = "task44-c2-strength-curve-decided-v415"
SIZES = [1, 2, 4, 6, 8, 10]
ELIGIBLE = [1, 3, 5, 7, 8, 10, 13, 14, 15, 16, 17, 18, 19]
ALLOWED = ["X_train", "y_train", "X_val", "y_val"]
RESERVED = [
    "X_test_natural",
    "y_test_natural",
    "X_test_diagnostic",
    "y_test_diagnostic",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    markdown_path = root / "configs" / "TASK45_PREREGISTRATION_V416.md"
    protocol_path = root / "configs" / "task45_preregistration_v4160.json"
    selector_path = root / "scripts" / "select_task45_coalitions_v416.py"
    adapter_path = root / "scripts" / "run_task45_dev_only_adapter_v416.py"
    audit_path = root / "scripts" / "audit_task45_c0_preregistration_v416.py"
    plain_path = root / "scripts" / "run_exact_plain_fedavg_v3132.py"
    defense_path = root / "scripts" / "run_frozen_reconstruction_v3123.py"
    shared_loader_path = root / "src" / "federated_iot_v26.py"
    task44_path = root / "configs" / "TASK44_PREREGISTRATION_V415.md"

    required = [
        markdown_path,
        protocol_path,
        selector_path,
        adapter_path,
        audit_path,
        plain_path,
        defense_path,
        shared_loader_path,
        task44_path,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    protocol: Dict[str, Any] = json.loads(
        protocol_path.read_text(encoding="utf-8")
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    selector_source = selector_path.read_text(encoding="utf-8")
    adapter_source = adapter_path.read_text(encoding="utf-8")
    plain_source = plain_path.read_text(encoding="utf-8")
    defense_source = defense_path.read_text(encoding="utf-8")
    shared_loader_source = shared_loader_path.read_text(encoding="utf-8")
    task44_source = task44_path.read_text(encoding="utf-8")

    checks: List[Dict[str, object]] = []

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append(
            {"check_name": name, "passed": bool(passed), "detail": str(detail)}
        )
        if not passed:
            raise RuntimeError(f"{name} failed: {detail}")

    head = git(root, "rev-parse", "HEAD")
    tags = git(root, "tag", "--points-at", "HEAD").splitlines()
    add("parent_commit", head == PARENT_COMMIT, head)
    add("parent_tag", PARENT_TAG in tags, tags)
    add("version", protocol.get("experiment_version") == "4.16.0", protocol.get("experiment_version"))
    add(
        "status",
        protocol.get("status") == "preregistered_before_any_task45_experiment",
        protocol.get("status"),
    )
    add(
        "protocol_parent_commit",
        protocol["parent_frozen_task"]["commit"] == PARENT_COMMIT,
        protocol["parent_frozen_task"]["commit"],
    )
    add(
        "protocol_parent_tag",
        protocol["parent_frozen_task"]["tag"] == PARENT_TAG,
        protocol["parent_frozen_task"]["tag"],
    )
    add("task44_not_reopened", protocol["parent_frozen_task"]["method_reopened"] is False, protocol["parent_frozen_task"])
    add("seeds", protocol["frozen_seeds"] == [7, 99, 123, 2026], protocol["frozen_seeds"])
    add("sizes", protocol["coalition_design"]["sizes"] == SIZES, protocol["coalition_design"]["sizes"])
    add("family_count", protocol["coalition_design"]["family_count"] == 3, protocol["coalition_design"]["family_count"])
    add("eligible_ids", protocol["eligibility"]["expected_eligible_client_ids"] == ELIGIBLE, protocol["eligibility"]["expected_eligible_client_ids"])
    add("eligible_count", protocol["eligibility"]["expected_eligible_count"] == 13, protocol["eligibility"]["expected_eligible_count"])
    add("source", protocol["scope"]["source_class"] == "DDoS", protocol["scope"]["source_class"])
    add("target", protocol["scope"]["target_class"] == "Benign", protocol["scope"]["target_class"])
    add("poison_fraction", protocol["scope"]["poison_fraction"] == 1.0, protocol["scope"]["poison_fraction"])
    add("minimum_source_rows", protocol["scope"]["min_source_samples"] == 1000, protocol["scope"]["min_source_samples"])
    add("safe_adapter_required", protocol["data_policy"]["safe_loader_adapter_required"] is True, protocol["data_policy"]["safe_loader_adapter_required"])
    add("allowed_arrays", protocol["data_policy"]["development_arrays_allowed"] == ALLOWED, protocol["data_policy"]["development_arrays_allowed"])
    add("reserved_arrays", protocol["data_policy"]["reserved_arrays_forbidden"] == RESERVED, protocol["data_policy"]["reserved_arrays_forbidden"])
    add("reserved_access_false", protocol["data_policy"]["reserved_test_access_before_final_freeze"] is False, protocol["data_policy"]["reserved_test_access_before_final_freeze"])

    families = protocol["coalition_design"]["families"]
    identities_by_size: Dict[int, set] = {size: set() for size in SIZES}
    for family in families:
        order = family["ordered_client_ids"]
        add(f"{family['family_id']}_eligible_order", set(order) == set(ELIGIBLE), order)
        previous = set()
        for size in SIZES:
            current = set(order[:size])
            add(f"{family['family_id']}_size_{size}", len(current) == size, sorted(current))
            add(f"{family['family_id']}_nested_{size}", previous.issubset(current), sorted(current))
            identity = tuple(sorted(current))
            add(f"{family['family_id']}_distinct_{size}", identity not in identities_by_size[size], identity)
            identities_by_size[size].add(identity)
            previous = current
    family_a = next(f for f in families if f["family_id"] == "A_development_anchor")
    add(
        "development_anchor_size8",
        set(family_a["ordered_client_ids"][:8]) == {1, 7, 8, 10, 14, 15, 17, 18},
        family_a["ordered_client_ids"][:8],
    )

    for name in ALLOWED:
        add(f"adapter_allows_{name}", repr(name) in adapter_source or f'"{name}"' in adapter_source, name)
    for name in RESERVED:
        add(f"adapter_blocks_{name}", name in adapter_source, name)
        add(f"plain_does_not_reference_{name}", name not in plain_source, name)
        add(f"defense_does_not_reference_{name}", name not in defense_source, name)
    add("inherited_loader_eager_materialization_confirmed", "arrays = {key: data[key] for key in data.files}" in shared_loader_source, "shared loader eagerly materializes all NPZ arrays")
    add("selector_avoids_inherited_loader", "load_protocol_arrays" not in selector_source, "selector reads y_train directly")
    add("selector_reads_only_y_train", 'payload["y_train"]' in selector_source, "y_train")
    add("markdown_records_loader_issue", "materializes every array" in markdown, "documented")
    add("task44_draft_caveat_is_real", "Status:** DRAFT" in task44_source, "inherited Task 44 Markdown status")
    add("markdown_records_task44_caveat", "Inherited Task 44 metadata caveat" in markdown, "documented")
    add("stage_count", len(protocol["staged_execution"]) == 6, len(protocol["staged_execution"]))
    add("condition_count", len(families) * len(SIZES) == 18, len(families) * len(SIZES))
    add("negative_results_required", protocol["final_claim_policy"]["negative_boundary_results_must_be_reported"] is True, protocol["final_claim_policy"])
    add("final_claims_blocked", protocol["final_claim_policy"]["final_paper_claims_allowed_before_C5"] is False, protocol["final_claim_policy"])

    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(tables / "task45c0_preregistration_checks.csv", index=False)

    stages = pd.DataFrame(protocol["staged_execution"])
    stages.to_csv(tables / "task45c0_execution_stages.csv", index=False)

    manifest_rows = []
    for path in required:
        manifest_rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        tables / "task45c0_source_manifest_sha256.csv", index=False
    )

    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    axis.bar(stages["stage"], range(1, len(stages) + 1))
    axis.set_ylabel("Protocol sequence")
    axis.set_title("Task 45 preregistered execution stages")
    axis.tick_params(axis="x", rotation=28)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "task45c0_execution_stages.png", dpi=300)
    figure.savefig(figures / "task45c0_execution_stages.pdf")
    plt.close(figure)

    decision = {
        "experiment_version": "4.16.0",
        "stage": "task45_c0_preregistration_integrity",
        "integrity_check_count": len(checks_frame),
        "integrity_checks_passed": int(checks_frame["passed"].sum()),
        "ready_for_c0_freeze_commit": True,
        "preregistration_frozen_by_commit": False,
        "task45_training_started": False,
        "reserved_test_arrays_materialized": False,
        "task44_reopened": False,
        "next_stage": "Commit C0 files and audit outputs, tag the C0 freeze, then begin C1 read-only coalition and interface preflight.",
    }
    (output / "task45c0_preregistration_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print("===== TASK 45 C0 PREREGISTRATION AUDIT =====")
    print(
        "Integrity checks passed:",
        f"{decision['integrity_checks_passed']}/{decision['integrity_check_count']}",
    )
    print("READY FOR C0 FREEZE COMMIT: True")
    print("PREREGISTRATION FROZEN BY COMMIT: False")
    print("TASK 45 TRAINING STARTED: False")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("TASK 44 REOPENED: False")
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
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


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    markdown_path = root / "TASK41C_PREREGISTRATION_V412C0.md"
    protocol_path = root / "configs" / "task41c_preregistration_v412c0.json"

    if not markdown_path.exists():
        raise FileNotFoundError(markdown_path)
    if not protocol_path.exists():
        raise FileNotFoundError(protocol_path)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    checks = []

    def add(name: str, passed: bool, detail) -> None:
        checks.append(
            {"check_name": name, "passed": bool(passed), "detail": str(detail)}
        )
        if not passed:
            raise RuntimeError(f"{name} failed: {detail}")

    add("version", protocol.get("experiment_version") == "4.12C.0", protocol.get("experiment_version"))
    add("status", protocol.get("status") == "preregistered_before_any_task41c_experiment", protocol.get("status"))
    add("parent_tag", protocol["parent_frozen_task"]["tag"] == "task41b-frozen-v4.11b9", protocol["parent_frozen_task"]["tag"])
    add("parent_commit", protocol["parent_frozen_task"]["commit"] == "f29b026", protocol["parent_frozen_task"]["commit"])
    add("task41b_not_reopened", protocol["parent_frozen_task"]["method_reopened"] is False, protocol["parent_frozen_task"]["method_reopened"])
    add("seeds", protocol.get("frozen_seeds") == [7, 99, 123, 2026], protocol.get("frozen_seeds"))
    add("client_count", protocol["frozen_clients"]["num_clients"] == 20, protocol["frozen_clients"]["num_clients"])
    add("malicious_clients", protocol["frozen_clients"]["malicious_clients"] == [1, 7, 8, 10, 14, 15, 17, 18], protocol["frozen_clients"]["malicious_clients"])
    add("reserved_test_forbidden", protocol["data_policy"]["reserved_test_access_before_final_freeze"] is False, protocol["data_policy"]["reserved_test_access_before_final_freeze"])
    add("final_test_gate_required", protocol["data_policy"]["final_test_access_requires_separate_gate"] is True, protocol["data_policy"]["final_test_access_requires_separate_gate"])
    add("attack_family_count", len(protocol["attack_panel"]) == 4, len(protocol["attack_panel"]))
    add("defense_candidate_count", len(protocol["defense_candidates"]) == 4, len(protocol["defense_candidates"]))
    add("comparison_arm_count", len(protocol["comparison_arms"]) == 10, len(protocol["comparison_arms"]))
    add("execution_stage_count", len(protocol["staged_execution"]) == 6, len(protocol["staged_execution"]))
    add("selection_rule", protocol["selection_rule"]["type"] == "lexicographic", protocol["selection_rule"]["type"])
    add("weight_tuning_disabled", protocol["defense_candidates"][3]["weight_tuning_allowed"] is False, protocol["defense_candidates"][3]["weight_tuning_allowed"])
    add("artifact_csv", "CSV" in protocol["artifact_requirements"]["every_stage"], protocol["artifact_requirements"]["every_stage"])
    add("artifact_json", "JSON" in protocol["artifact_requirements"]["every_stage"], protocol["artifact_requirements"]["every_stage"])
    add("artifact_png", "PNG" in protocol["artifact_requirements"]["every_stage"], protocol["artifact_requirements"]["every_stage"])
    add("artifact_pdf", "PDF" in protocol["artifact_requirements"]["every_stage"], protocol["artifact_requirements"]["every_stage"])
    add("manifest_required", protocol["artifact_requirements"]["source_hash_manifest"] is True, protocol["artifact_requirements"]["source_hash_manifest"])
    add("final_claims_blocked", protocol["final_claim_policy"]["final_paper_claims_allowed_before_C5"] is False, protocol["final_claim_policy"]["final_paper_claims_allowed_before_C5"])
    add("negative_results_required", protocol["final_claim_policy"]["negative_boundary_results_must_be_reported"] is True, protocol["final_claim_policy"]["negative_boundary_results_must_be_reported"])

    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(
        tables / "task41c0_preregistration_checks.csv", index=False
    )

    attacks = pd.DataFrame(protocol["attack_panel"])
    attacks["poison_fractions"] = attacks["poison_fractions"].apply(
        lambda values: ",".join(str(value) for value in values)
    )
    attacks["trigger_slots"] = attacks["trigger_slots"].apply(
        lambda values: ",".join(values)
    )
    attacks.to_csv(tables / "task41c0_attack_panel.csv", index=False)

    defenses = pd.DataFrame(protocol["defense_candidates"])
    defenses.to_csv(tables / "task41c0_defense_candidates.csv", index=False)

    stages = pd.DataFrame(protocol["staged_execution"])
    stages.to_csv(tables / "task41c0_execution_stages.csv", index=False)

    manifest = pd.DataFrame(
        [
            {
                "relative_path": str(markdown_path.relative_to(root)),
                "bytes": markdown_path.stat().st_size,
                "sha256": sha256(markdown_path),
            },
            {
                "relative_path": str(protocol_path.relative_to(root)),
                "bytes": protocol_path.stat().st_size,
                "sha256": sha256(protocol_path),
            },
        ]
    )
    manifest.to_csv(
        tables / "task41c0_source_manifest_sha256.csv", index=False
    )

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(stages["stage"], range(1, len(stages) + 1))
    axis.set_ylabel("Protocol sequence")
    axis.set_title("Task 41C preregistered execution stages")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        figures / "task41c0_execution_stages.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        figures / "task41c0_execution_stages.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    decision = {
        "experiment_version": "4.12C.0",
        "stage": "task41c_preregistration_integrity",
        "integrity_check_count": len(checks_frame),
        "integrity_checks_passed": int(checks_frame["passed"].sum()),
        "preregistration_frozen": True,
        "task41c_experiments_started": False,
        "task41b_reopened": False,
        "reserved_test_accessed": False,
        "next_stage": "Commit C0 preregistration, then begin C1 clean-only calibration.",
    }
    (output / "task41c0_preregistration_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print("===== TASK 41C.0 PREREGISTRATION AUDIT =====")
    print(
        "Integrity checks passed:",
        f"{decision['integrity_checks_passed']}/{decision['integrity_check_count']}",
    )
    print("PREREGISTRATION FROZEN: True")
    print("TASK 41C EXPERIMENTS STARTED: False")
    print("TASK 41B REOPENED: False")
    print("TEST SETS ACCESSED: False")
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

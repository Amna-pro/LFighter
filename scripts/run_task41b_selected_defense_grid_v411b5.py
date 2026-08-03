#!/usr/bin/env python3
"""
Task 41B.5 selected matched frozen-defense grid.

Runs trusted reconstruction only for candidates selected by the frozen
Task 41B.4 qualification audit:
- every primary qualifier,
- the protocol-mandated breadth negative control.

The script validates exact pairing against each plain branch, preserves
completed valid branches, forbids attack-specific retuning, and does not
load reserved test arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

MATCH_FIELDS = (
    "model_seed",
    "attack_seed",
    "trigger_candidate_id",
    "trigger_spec_sha256",
    "label_policy",
    "deployment_policy",
    "poison_fraction",
    "partition_hash_sha256",
    "poison_index_hash_sha256",
    "warmup_profile_sha256",
    "malicious_clients",
)

FORBIDDEN_TEST_FLAGS = (
    "test_arrays_loaded",
    "natural_test_accessed",
    "diagnostic_test_accessed",
    "test_sets_accessed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched trusted-reconstruction branches for Task 41B selections."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--partition-file", type=Path, required=True)
    parser.add_argument("--clean-seed-dir", type=Path, required=True)
    parser.add_argument("--warmup-dir", type=Path, required=True)
    parser.add_argument("--reconstruction-calibration-dir", type=Path, required=True)
    parser.add_argument("--trigger-spec-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qualification-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_access_clear(metadata: dict[str, Any]) -> bool:
    return not any(bool(metadata.get(field, False)) for field in FORBIDDEN_TEST_FLAGS)


def exact_pair_match(
    plain_metadata: dict[str, Any],
    defended_metadata: dict[str, Any],
) -> tuple[bool, list[str]]:
    mismatches = []
    for field in MATCH_FIELDS:
        if plain_metadata.get(field) != defended_metadata.get(field):
            mismatches.append(field)
    return not mismatches, mismatches


def branch_valid(
    plain_dir: Path,
    defended_dir: Path,
    candidate_id: str,
) -> tuple[bool, str]:
    plain_metadata_path = plain_dir / "task41b_backdoor_metadata.json"
    defended_metadata_path = defended_dir / "task41b_backdoor_metadata.json"
    required = (
        plain_metadata_path,
        defended_metadata_path,
        defended_dir / "tables" / "continuation_round_metrics.csv",
        defended_dir / "tables" / "triggered_validation_asr_long.csv",
        defended_dir / "figures" / "clean_utility_and_triggered_asr.png",
        defended_dir / "figures" / "clean_utility_and_triggered_asr.pdf",
    )
    if not all(path.exists() for path in required):
        return False, "required_artifact_missing"

    try:
        plain_meta = load_json(plain_metadata_path)
        defended_meta = load_json(defended_metadata_path)
    except Exception as exc:
        return False, f"metadata_read_failed:{exc}"

    if plain_meta.get("replacement_policy") != "plain_fedavg":
        return False, "plain_policy_mismatch"
    if defended_meta.get("replacement_policy") != "trusted_reconstruction":
        return False, "defended_policy_mismatch"
    if plain_meta.get("mode") != "strong_attack":
        return False, "plain_mode_mismatch"
    if defended_meta.get("mode") != "strong_attack":
        return False, "defended_mode_mismatch"
    if plain_meta.get("trigger_candidate_id") != candidate_id:
        return False, "plain_candidate_mismatch"
    if defended_meta.get("trigger_candidate_id") != candidate_id:
        return False, "defended_candidate_mismatch"
    if not test_access_clear(plain_meta) or not test_access_clear(defended_meta):
        return False, "reserved_test_access_reported"

    exact, mismatches = exact_pair_match(plain_meta, defended_meta)
    if not exact:
        return False, "pair_metadata_mismatch:" + ",".join(mismatches)

    return True, "complete_exact_pair"


def main() -> int:
    args = parse_args()
    root = resolved(args.project_root)
    output_root = resolved(args.output_root)
    qualification_dir = resolved(args.qualification_dir)

    python_exe = root / ".venv" / "Scripts" / "python.exe"
    runner = root / "scripts" / "run_task41b_backdoor_smoke_v411b1.py"
    qualification_csv = (
        qualification_dir / "tables" / "task41b3_plain_breadth_qualification.csv"
    )
    decision_json = qualification_dir / "task41b3_plain_breadth_decision.json"

    required_paths = (
        python_exe,
        runner,
        resolved(args.data_file),
        resolved(args.partition_file),
        resolved(args.clean_seed_dir),
        resolved(args.warmup_dir),
        resolved(args.reconstruction_calibration_dir),
        resolved(args.trigger_spec_file),
        qualification_csv,
        decision_json,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    import pandas as pd

    qualification = pd.read_csv(qualification_csv)
    decision = load_json(decision_json)

    selected = qualification[
        qualification["proceed_to_paired_defense"].astype(bool)
    ].copy()

    if len(selected) != 3:
        raise RuntimeError(
            f"Expected exactly 3 frozen selections, found {len(selected)}."
        )
    if int(selected["passes_primary_qualification"].astype(bool).sum()) != 2:
        raise RuntimeError("Expected exactly 2 primary qualifiers.")
    if int((selected["selection_label"] == "BREADTH_NEGATIVE_CONTROL").sum()) != 1:
        raise RuntimeError("Expected exactly 1 breadth negative control.")
    if bool(decision.get("multiseed_selection_allowed_now", True)):
        raise RuntimeError("Qualification decision unexpectedly allows multiseed selection.")

    protocol_dir = output_root / "selected_defense_protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "experiment_version": "4.11B.5",
        "stage": "task41b_selected_matched_frozen_defense_grid",
        "status": "development_paired_defense_not_final_paper_result",
        "selection_source_csv": str(qualification_csv),
        "selection_source_csv_sha256": sha256_file(qualification_csv),
        "selection_decision_json": str(decision_json),
        "selection_decision_json_sha256": sha256_file(decision_json),
        "selected_slots": selected["slot"].tolist(),
        "selected_candidates": selected["candidate_id"].tolist(),
        "selection_labels": selected["selection_label"].tolist(),
        "model_seed": 7,
        "attack_seed": 7,
        "poison_fraction": 0.01,
        "replacement_policy": "trusted_reconstruction",
        "reconstruction_policy": "center_plus_residual",
        "frozen_task40_detector_reused": True,
        "attack_specific_retuning": False,
        "threshold_retuning": False,
        "source_class": "DDoS",
        "target_class": "Benign",
        "label_policy": "dirty_label_all_nonbenign_to_benign",
        "deployment_policy": "centralized_full_trigger",
        "malicious_clients": [1, 7, 8, 10, 14, 15, 17, 18],
        "continuation_rounds": 4,
        "test_sets_accessed": False,
        "next_stage_after_completion": (
            "Run exact paired plain-versus-defense audits for all three selected "
            "branches. Do not select multiseed candidates before those audits."
        ),
    }
    protocol_path = protocol_dir / "task41b5_selected_defense_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    selected.to_csv(
        protocol_dir / "task41b5_selected_candidate_panel.csv", index=False
    )

    print("===== TASK 41B.5 SELECTED MATCHED FROZEN-DEFENSE GRID =====")
    print("Primary qualifiers:", selected.loc[
        selected["passes_primary_qualification"].astype(bool), "slot"
    ].tolist())
    print("Breadth negative control:", selected.loc[
        selected["selection_label"] == "BREADTH_NEGATIVE_CONTROL", "slot"
    ].tolist())
    print("Attack-specific retuning: False")
    print("Reserved test access allowed: False")

    completion_rows: list[dict[str, Any]] = []

    for row in selected.itertuples(index=False):
        slot = str(row.slot)
        candidate_id = str(row.candidate_id)
        selection_label = str(row.selection_label)

        plain_dir = output_root / "runs" / slot / "seed_7" / "plain_fedavg"
        defended_dir = (
            output_root / "runs" / slot / "seed_7" / "trusted_reconstruction"
        )
        plain_metadata_path = plain_dir / "task41b_backdoor_metadata.json"
        if not plain_metadata_path.exists():
            raise FileNotFoundError(plain_metadata_path)

        plain_meta = load_json(plain_metadata_path)
        if plain_meta.get("replacement_policy") != "plain_fedavg":
            raise RuntimeError(f"{slot} plain branch policy mismatch.")
        if plain_meta.get("trigger_candidate_id") != candidate_id:
            raise RuntimeError(f"{slot} plain candidate mismatch.")
        if not test_access_clear(plain_meta):
            raise RuntimeError(f"{slot} plain branch reports reserved test access.")

        valid, reason = branch_valid(plain_dir, defended_dir, candidate_id)
        if valid:
            print(f"SKIPPING COMPLETE EXACT PAIRED BRANCH: {defended_dir}")
        else:
            if defended_dir.exists():
                raise RuntimeError(
                    "Existing defended directory is incomplete or mismatched "
                    f"({reason}). Inspect or remove only this branch: {defended_dir}"
                )

            command = [
                str(python_exe),
                "-u",
                str(runner),
                "--mode",
                "strong_attack",
                "--replacement-policy",
                "trusted_reconstruction",
                "--data-file",
                str(resolved(args.data_file)),
                "--partition-file",
                str(resolved(args.partition_file)),
                "--clean-seed-dir",
                str(resolved(args.clean_seed_dir)),
                "--warmup-dir",
                str(resolved(args.warmup_dir)),
                "--reconstruction-calibration-dir",
                str(resolved(args.reconstruction_calibration_dir)),
                "--plain-branch-dir",
                str(plain_dir),
                "--trigger-spec-file",
                str(resolved(args.trigger_spec_file)),
                "--trigger-candidate-id",
                candidate_id,
                "--label-policy",
                "dirty_label_all_nonbenign_to_benign",
                "--deployment-policy",
                "centralized_full_trigger",
                "--output-dir",
                str(defended_dir),
                "--model-seed",
                "7",
                "--attack-seed",
                "7",
                "--num-clients",
                "20",
                "--continuation-rounds",
                "4",
                "--batch-size",
                str(args.batch_size),
                "--evaluation-batch-size",
                str(args.evaluation_batch_size),
                "--learning-rate",
                "0.0003",
                "--weight-decay",
                "0.0001",
                "--threads",
                str(args.threads),
                "--probe-per-class",
                "48",
                "--probe-seed",
                "3701",
                "--source-class",
                "DDoS",
                "--target-class",
                "Benign",
                "--ema-decay",
                "0.65",
                "--malicious-clients",
                "1,7,8,10,14,15,17,18",
                "--poison-fraction",
                "0.01",
                "--min-source-samples",
                "1000",
            ]

            print()
            print("RUNNING SELECTED DEFENSE BRANCH")
            print("Slot:", slot)
            print("Selection label:", selection_label)
            print("Candidate:", candidate_id)
            completed = subprocess.run(command, cwd=str(root), check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Defended branch failed with exit code "
                    f"{completed.returncode}: {slot}"
                )

            valid, reason = branch_valid(plain_dir, defended_dir, candidate_id)
            if not valid:
                raise RuntimeError(
                    f"Completion exact-pair validation failed for {slot}: {reason}"
                )

        plain_meta = load_json(plain_dir / "task41b_backdoor_metadata.json")
        defended_meta = load_json(
            defended_dir / "task41b_backdoor_metadata.json"
        )
        exact, mismatches = exact_pair_match(plain_meta, defended_meta)

        completion_rows.append(
            {
                "slot": slot,
                "candidate_id": candidate_id,
                "selection_label": selection_label,
                "passes_primary_qualification": bool(
                    row.passes_primary_qualification
                ),
                "plain_dir": str(plain_dir),
                "defended_dir": str(defended_dir),
                "complete_exact_pair": bool(exact and not mismatches),
                "metadata_mismatches": ",".join(mismatches),
                "poison_plan_hash_exact_match": (
                    plain_meta.get("poison_index_hash_sha256")
                    == defended_meta.get("poison_index_hash_sha256")
                ),
                "trigger_spec_hash_exact_match": (
                    plain_meta.get("trigger_spec_sha256")
                    == defended_meta.get("trigger_spec_sha256")
                ),
                "partition_hash_exact_match": (
                    plain_meta.get("partition_hash_sha256")
                    == defended_meta.get("partition_hash_sha256")
                ),
                "test_sets_accessed": False,
            }
        )

    completion_path = (
        protocol_dir / "task41b5_selected_defense_completion.csv"
    )
    with completion_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(completion_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(completion_rows)

    if not all(row["complete_exact_pair"] for row in completion_rows):
        raise RuntimeError("One or more selected branches failed exact-pair validation.")

    print()
    print("TASK 41B.5 SELECTED DEFENSE GRID COMPLETE")
    print(f"Completed exact pairs: {len(completion_rows)}/{len(completion_rows)}")
    print("Protocol artifacts:", protocol_dir)
    print("MULTISEED SELECTION ALLOWED NOW: False")
    print("FINAL TEST ARRAYS LOADED: False")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print()
        print(f"TASK 41B.5 GRID FAILED: {exc}", file=sys.stderr)
        raise

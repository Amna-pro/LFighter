#!/usr/bin/env python3
"""
Task 41B.7 exact matched multiseed backdoor grid.

Runs every frozen primary qualifier selected before defense evaluation across
seeds 7, 99, 123, and 2026. For each seed and trigger, the script runs or
validates:
1. plain FedAvg,
2. trusted reconstruction,
3. exact metadata and poison-plan pairing.

The script preserves valid completed branches, rejects incomplete or mismatched
directories, performs no attack-specific retuning, and does not access reserved
natural or diagnostic test arrays.
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

REQUIRED_BRANCH_ARTIFACTS = (
    Path("task41b_backdoor_metadata.json"),
    Path("tables") / "continuation_round_metrics.csv",
    Path("tables") / "triggered_validation_asr_long.csv",
    Path("figures") / "clean_utility_and_triggered_asr.png",
    Path("figures") / "clean_utility_and_triggered_asr.pdf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact matched Task 41B primary-trigger multiseed evaluation."
        )
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--partition-file", type=Path, required=True)
    parser.add_argument("--clean-seed-root", type=Path, required=True)
    parser.add_argument("--warmup-root", type=Path, required=True)
    parser.add_argument("--reconstruction-root", type=Path, required=True)
    parser.add_argument("--trigger-spec-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qualification-dir", type=Path, required=True)
    parser.add_argument("--paired-audit-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="7,99,123,2026")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def parse_seeds(text: str) -> list[int]:
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    if seeds != [7, 99, 123, 2026]:
        raise ValueError(
            "Task 41B.7 seeds are frozen to exactly 7,99,123,2026."
        )
    return seeds


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
    mismatches: list[str] = []
    for field in MATCH_FIELDS:
        if plain_metadata.get(field) != defended_metadata.get(field):
            mismatches.append(field)
    return not mismatches, mismatches


def validate_branch(
    branch_dir: Path,
    *,
    policy: str,
    candidate_id: str,
    seed: int,
) -> tuple[bool, str, dict[str, Any] | None]:
    required = tuple(branch_dir / relative for relative in REQUIRED_BRANCH_ARTIFACTS)
    if not all(path.exists() for path in required):
        return False, "required_artifact_missing", None

    metadata_path = branch_dir / "task41b_backdoor_metadata.json"
    try:
        metadata = load_json(metadata_path)
    except Exception as exc:
        return False, f"metadata_read_failed:{exc}", None

    checks = (
        (metadata.get("replacement_policy") == policy, "policy_mismatch"),
        (metadata.get("mode") == "strong_attack", "mode_mismatch"),
        (
            metadata.get("attack_type") == "dirty_label_trigger_to_benign",
            "attack_type_mismatch",
        ),
        (
            metadata.get("trigger_candidate_id") == candidate_id,
            "candidate_mismatch",
        ),
        (int(metadata.get("model_seed", -1)) == seed, "model_seed_mismatch"),
        (int(metadata.get("attack_seed", -1)) == seed, "attack_seed_mismatch"),
        (
            abs(float(metadata.get("poison_fraction", -1.0)) - 0.01) <= 1e-12,
            "poison_fraction_mismatch",
        ),
        (
            metadata.get("label_policy")
            == "dirty_label_all_nonbenign_to_benign",
            "label_policy_mismatch",
        ),
        (
            metadata.get("deployment_policy") == "centralized_full_trigger",
            "deployment_policy_mismatch",
        ),
        (
            not bool(metadata.get("attack_specific_retuning", False)),
            "attack_specific_retuning_reported",
        ),
        (test_access_clear(metadata), "reserved_test_access_reported"),
    )
    for passed, reason in checks:
        if not passed:
            return False, reason, metadata

    return True, "complete_valid_branch", metadata


def validate_exact_pair(
    plain_dir: Path,
    defended_dir: Path,
    *,
    candidate_id: str,
    seed: int,
) -> tuple[bool, str, dict[str, Any] | None, dict[str, Any] | None]:
    plain_valid, plain_reason, plain_meta = validate_branch(
        plain_dir,
        policy="plain_fedavg",
        candidate_id=candidate_id,
        seed=seed,
    )
    if not plain_valid:
        return False, f"plain:{plain_reason}", plain_meta, None

    defended_valid, defended_reason, defended_meta = validate_branch(
        defended_dir,
        policy="trusted_reconstruction",
        candidate_id=candidate_id,
        seed=seed,
    )
    if not defended_valid:
        return False, f"defended:{defended_reason}", plain_meta, defended_meta

    assert plain_meta is not None
    assert defended_meta is not None
    exact, mismatches = exact_pair_match(plain_meta, defended_meta)
    if not exact:
        return (
            False,
            "pair_metadata_mismatch:" + ",".join(mismatches),
            plain_meta,
            defended_meta,
        )

    return True, "complete_exact_pair", plain_meta, defended_meta


def runner_command(
    *,
    python_exe: Path,
    runner: Path,
    policy: str,
    data_file: Path,
    partition_file: Path,
    clean_seed_dir: Path,
    warmup_dir: Path,
    reconstruction_dir: Path | None,
    plain_dir: Path | None,
    trigger_spec_file: Path,
    candidate_id: str,
    output_dir: Path,
    seed: int,
    threads: int,
    batch_size: int,
    evaluation_batch_size: int,
) -> list[str]:
    command = [
        str(python_exe),
        "-u",
        str(runner),
        "--mode",
        "strong_attack",
        "--replacement-policy",
        policy,
        "--data-file",
        str(data_file),
        "--partition-file",
        str(partition_file),
        "--clean-seed-dir",
        str(clean_seed_dir),
        "--warmup-dir",
        str(warmup_dir),
        "--trigger-spec-file",
        str(trigger_spec_file),
        "--trigger-candidate-id",
        candidate_id,
        "--label-policy",
        "dirty_label_all_nonbenign_to_benign",
        "--deployment-policy",
        "centralized_full_trigger",
        "--output-dir",
        str(output_dir),
        "--model-seed",
        str(seed),
        "--attack-seed",
        str(seed),
        "--num-clients",
        "20",
        "--continuation-rounds",
        "4",
        "--batch-size",
        str(batch_size),
        "--evaluation-batch-size",
        str(evaluation_batch_size),
        "--learning-rate",
        "0.0003",
        "--weight-decay",
        "0.0001",
        "--threads",
        str(threads),
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

    if policy == "trusted_reconstruction":
        if reconstruction_dir is None or plain_dir is None:
            raise ValueError(
                "Defended command requires reconstruction and plain directories."
            )
        command.extend(
            [
                "--reconstruction-calibration-dir",
                str(reconstruction_dir),
                "--plain-branch-dir",
                str(plain_dir),
            ]
        )

    return command


def main() -> int:
    args = parse_args()
    root = resolved(args.project_root)
    data_file = resolved(args.data_file)
    partition_file = resolved(args.partition_file)
    clean_seed_root = resolved(args.clean_seed_root)
    warmup_root = resolved(args.warmup_root)
    reconstruction_root = resolved(args.reconstruction_root)
    trigger_spec_file = resolved(args.trigger_spec_file)
    output_root = resolved(args.output_root)
    qualification_dir = resolved(args.qualification_dir)
    paired_audit_dir = resolved(args.paired_audit_dir)
    seeds = parse_seeds(args.seeds)

    python_exe = root / ".venv" / "Scripts" / "python.exe"
    runner = root / "scripts" / "run_task41b_backdoor_smoke_v411b1.py"
    qualification_csv = (
        qualification_dir
        / "tables"
        / "task41b3_plain_breadth_qualification.csv"
    )
    paired_decision_path = (
        paired_audit_dir / "task41b6_paired_defense_decision.json"
    )
    paired_summary_path = (
        paired_audit_dir / "tables" / "task41b6_exact_pair_summary.csv"
    )

    required_paths = (
        python_exe,
        runner,
        data_file,
        partition_file,
        clean_seed_root,
        warmup_root,
        reconstruction_root,
        trigger_spec_file,
        qualification_csv,
        paired_decision_path,
        paired_summary_path,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    import pandas as pd

    qualification = pd.read_csv(qualification_csv)
    paired_decision = load_json(paired_decision_path)
    paired_summary = pd.read_csv(paired_summary_path)

    if not bool(paired_decision.get("multiseed_selection_allowed_now", False)):
        raise RuntimeError(
            "Task 41B.6 did not authorize multiseed confirmation."
        )
    if bool(paired_decision.get("reserved_test_accessed", True)):
        raise RuntimeError(
            "Task 41B.6 reports reserved test access."
        )
    if bool(paired_decision.get("attack_specific_retuning", True)):
        raise RuntimeError(
            "Task 41B.6 reports attack-specific retuning."
        )

    selected_slots = list(paired_decision.get("multiseed_selected_slots", []))
    if selected_slots != ["flow_iat_exact", "active_idle_exact"]:
        raise RuntimeError(
            "Frozen multiseed slots must be flow_iat_exact and active_idle_exact."
        )

    primary = qualification[
        qualification["passes_primary_qualification"].astype(bool)
    ].copy()
    primary = primary[primary["slot"].isin(selected_slots)].copy()
    primary["slot_order"] = primary["slot"].map(
        {"flow_iat_exact": 0, "active_idle_exact": 1}
    )
    primary = primary.sort_values("slot_order").reset_index(drop=True)

    if len(primary) != 2:
        raise RuntimeError("Expected exactly two frozen primary qualifiers.")
    if primary["slot"].tolist() != selected_slots:
        raise RuntimeError("Qualification and paired-audit slot order mismatch.")
    if set(paired_summary["slot"]) < set(selected_slots):
        raise RuntimeError("Task 41B.6 summary lacks a selected primary qualifier.")

    protocol_dir = output_root / "multiseed_protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "experiment_version": "4.11B.7",
        "stage": "task41b_exact_matched_multiseed_backdoor_grid",
        "status": "confirmatory_development_multiseed_not_final_test_result",
        "selection_frozen_before_multiseed_runs": True,
        "selection_source": str(paired_decision_path),
        "selection_source_sha256": sha256_file(paired_decision_path),
        "qualification_source": str(qualification_csv),
        "qualification_source_sha256": sha256_file(qualification_csv),
        "paired_summary_source": str(paired_summary_path),
        "paired_summary_source_sha256": sha256_file(paired_summary_path),
        "selected_slots": selected_slots,
        "selected_candidates": primary["candidate_id"].tolist(),
        "seeds": seeds,
        "poison_fraction": 0.01,
        "label_policy": "dirty_label_all_nonbenign_to_benign",
        "deployment_policy": "centralized_full_trigger",
        "source_class": "DDoS",
        "target_class": "Benign",
        "malicious_clients": [1, 7, 8, 10, 14, 15, 17, 18],
        "continuation_rounds": 4,
        "plain_policy": "plain_fedavg",
        "defended_policy": "trusted_reconstruction",
        "reconstruction_policy": "center_plus_residual",
        "frozen_task40_detector_reused": True,
        "attack_specific_retuning": False,
        "threshold_retuning": False,
        "negative_control_promoted": False,
        "reserved_test_access_allowed": False,
        "test_sets_accessed": False,
        "next_stage_after_completion": (
            "Summarize exact paired multiseed efficacy, detector recall, clean "
            "utility, seed consistency, and per-source triggered ASR. Do not "
            "access reserved test arrays before the multiseed development "
            "decision is frozen."
        ),
    }
    protocol_path = (
        protocol_dir / "task41b7_exact_multiseed_protocol.json"
    )
    protocol_path.write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    primary.to_csv(
        protocol_dir / "task41b7_frozen_primary_trigger_panel.csv",
        index=False,
    )

    print("===== TASK 41B.7 EXACT MATCHED MULTISEED GRID =====")
    print("Frozen primary qualifiers:", selected_slots)
    print("Seeds:", seeds)
    print("Negative control promoted: False")
    print("Attack-specific retuning: False")
    print("Reserved test access allowed: False")

    completion_rows: list[dict[str, Any]] = []
    executed_branch_count = 0
    skipped_branch_count = 0

    for seed in seeds:
        clean_seed_dir = clean_seed_root / f"seed_{seed}"
        warmup_dir = warmup_root / f"seed_{seed}" / "warmup"
        reconstruction_dir = (
            reconstruction_root / f"seed_{seed}" / "calibration"
        )
        for path in (
            clean_seed_dir,
            warmup_dir,
            reconstruction_dir,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        for row in primary.itertuples(index=False):
            slot = str(row.slot)
            candidate_id = str(row.candidate_id)
            plain_dir = (
                output_root / "runs" / slot / f"seed_{seed}" / "plain_fedavg"
            )
            defended_dir = (
                output_root
                / "runs"
                / slot
                / f"seed_{seed}"
                / "trusted_reconstruction"
            )

            plain_valid, plain_reason, plain_meta = validate_branch(
                plain_dir,
                policy="plain_fedavg",
                candidate_id=candidate_id,
                seed=seed,
            )
            if plain_valid:
                print(
                    f"SKIPPING COMPLETE VALID PLAIN BRANCH: {plain_dir}"
                )
                skipped_branch_count += 1
            else:
                if plain_dir.exists():
                    raise RuntimeError(
                        "Existing plain directory is incomplete or mismatched "
                        f"({plain_reason}). Inspect or quarantine only this "
                        f"branch: {plain_dir}"
                    )

                print()
                print("RUNNING MULTISEED PLAIN BRANCH")
                print("Slot:", slot)
                print("Seed:", seed)
                print("Candidate:", candidate_id)
                command = runner_command(
                    python_exe=python_exe,
                    runner=runner,
                    policy="plain_fedavg",
                    data_file=data_file,
                    partition_file=partition_file,
                    clean_seed_dir=clean_seed_dir,
                    warmup_dir=warmup_dir,
                    reconstruction_dir=None,
                    plain_dir=None,
                    trigger_spec_file=trigger_spec_file,
                    candidate_id=candidate_id,
                    output_dir=plain_dir,
                    seed=seed,
                    threads=args.threads,
                    batch_size=args.batch_size,
                    evaluation_batch_size=args.evaluation_batch_size,
                )
                completed = subprocess.run(
                    command, cwd=str(root), check=False
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Plain branch failed, slot={slot}, seed={seed}, "
                        f"exit_code={completed.returncode}."
                    )
                plain_valid, plain_reason, plain_meta = validate_branch(
                    plain_dir,
                    policy="plain_fedavg",
                    candidate_id=candidate_id,
                    seed=seed,
                )
                if not plain_valid:
                    raise RuntimeError(
                        f"Plain completion validation failed for {slot}, "
                        f"seed {seed}: {plain_reason}"
                    )
                executed_branch_count += 1

            pair_valid, pair_reason, _, _ = validate_exact_pair(
                plain_dir,
                defended_dir,
                candidate_id=candidate_id,
                seed=seed,
            )
            if pair_valid:
                print(
                    f"SKIPPING COMPLETE EXACT DEFENDED PAIR: {defended_dir}"
                )
                skipped_branch_count += 1
            else:
                if defended_dir.exists():
                    raise RuntimeError(
                        "Existing defended directory is incomplete or mismatched "
                        f"({pair_reason}). Inspect or quarantine only this "
                        f"branch: {defended_dir}"
                    )

                print()
                print("RUNNING MULTISEED DEFENDED BRANCH")
                print("Slot:", slot)
                print("Seed:", seed)
                print("Candidate:", candidate_id)
                command = runner_command(
                    python_exe=python_exe,
                    runner=runner,
                    policy="trusted_reconstruction",
                    data_file=data_file,
                    partition_file=partition_file,
                    clean_seed_dir=clean_seed_dir,
                    warmup_dir=warmup_dir,
                    reconstruction_dir=reconstruction_dir,
                    plain_dir=plain_dir,
                    trigger_spec_file=trigger_spec_file,
                    candidate_id=candidate_id,
                    output_dir=defended_dir,
                    seed=seed,
                    threads=args.threads,
                    batch_size=args.batch_size,
                    evaluation_batch_size=args.evaluation_batch_size,
                )
                completed = subprocess.run(
                    command, cwd=str(root), check=False
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Defended branch failed, slot={slot}, seed={seed}, "
                        f"exit_code={completed.returncode}."
                    )
                executed_branch_count += 1

            pair_valid, pair_reason, plain_meta, defended_meta = (
                validate_exact_pair(
                    plain_dir,
                    defended_dir,
                    candidate_id=candidate_id,
                    seed=seed,
                )
            )
            if not pair_valid:
                raise RuntimeError(
                    f"Exact-pair validation failed for {slot}, seed {seed}: "
                    f"{pair_reason}"
                )
            assert plain_meta is not None
            assert defended_meta is not None
            exact, mismatches = exact_pair_match(plain_meta, defended_meta)

            completion_rows.append(
                {
                    "slot": slot,
                    "candidate_id": candidate_id,
                    "seed": seed,
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
                    "warmup_profile_hash_exact_match": (
                        plain_meta.get("warmup_profile_sha256")
                        == defended_meta.get("warmup_profile_sha256")
                    ),
                    "plain_test_sets_accessed": not test_access_clear(plain_meta),
                    "defended_test_sets_accessed": not test_access_clear(
                        defended_meta
                    ),
                    "attack_specific_retuning": False,
                }
            )

    expected_pair_count = len(seeds) * len(primary)
    if len(completion_rows) != expected_pair_count:
        raise RuntimeError(
            f"Expected {expected_pair_count} exact pairs, "
            f"found {len(completion_rows)}."
        )
    if not all(row["complete_exact_pair"] for row in completion_rows):
        raise RuntimeError("At least one multiseed pair is not exact.")
    if not all(
        row["poison_plan_hash_exact_match"]
        and row["trigger_spec_hash_exact_match"]
        and row["partition_hash_exact_match"]
        and row["warmup_profile_hash_exact_match"]
        and not row["plain_test_sets_accessed"]
        and not row["defended_test_sets_accessed"]
        for row in completion_rows
    ):
        raise RuntimeError(
            "At least one multiseed completion integrity check failed."
        )

    completion_path = (
        protocol_dir / "task41b7_exact_multiseed_completion.csv"
    )
    with completion_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(completion_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(completion_rows)

    completion_json = {
        "experiment_version": "4.11B.7",
        "completed_exact_pair_count": len(completion_rows),
        "expected_exact_pair_count": expected_pair_count,
        "executed_branch_count": executed_branch_count,
        "skipped_valid_branch_count": skipped_branch_count,
        "all_exact_pairs_complete": True,
        "attack_specific_retuning": False,
        "reserved_test_accessed": False,
        "final_paper_claim_allowed": False,
        "next_stage": (
            "Task 41B.8 exact paired multiseed summary and decision."
        ),
    }
    (
        protocol_dir / "task41b7_exact_multiseed_completion.json"
    ).write_text(
        json.dumps(completion_json, indent=2), encoding="utf-8"
    )

    print()
    print("TASK 41B.7 EXACT MULTISEED GRID COMPLETE")
    print(
        f"Completed exact pairs: "
        f"{len(completion_rows)}/{expected_pair_count}"
    )
    print("Newly executed branches:", executed_branch_count)
    print("Skipped valid branches:", skipped_branch_count)
    print("Protocol artifacts:", protocol_dir)
    print("FINAL PAPER CLAIM ALLOWED: False")
    print("FINAL TEST ARRAYS LOADED: False")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print()
        print(f"TASK 41B.7 GRID FAILED: {exc}", file=sys.stderr)
        raise

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PANEL = [
    ("flow_iat_exact", "multi_flow_iat_bundle__exact_template__donor_00", "flow_iat", "exact_template"),
    ("flow_iat_low", "multi_flow_iat_bundle__low_intensity_template__donor_00", "flow_iat", "low_intensity"),
    ("init_windows_exact", "multi_init_windows__exact_template__donor_01", "initial_window", "exact_template"),
    ("fwd_packet_length_low", "multi_fwd_packet_length_bundle__low_intensity_template__donor_09", "forward_packet_length", "low_intensity"),
    ("bwd_packet_length_low", "multi_bwd_packet_length_bundle__low_intensity_template__donor_12", "backward_packet_length", "low_intensity"),
    ("active_idle_exact", "multi_active_idle_bundle__exact_template__donor_18", "active_idle", "exact_template"),
]

PROTOCOL = {
    "protocol_version": "4.11B.3",
    "stage": "task41b_frozen_trigger_breadth_plain_screen",
    "status": "development_screen_not_final_paper_result",
    "decision_frozen_before_new_panel_results": True,
    "primary_qualification_criteria": {
        "mean_macro_triggered_asr_minimum": 0.20,
        "final_round_macro_triggered_asr_minimum": 0.40,
        "minimum_source_classes_with_final_asr_at_least_0_20": 4,
        "maximum_absolute_clean_macro_f1_drop_vs_zero_poison": 0.03,
        "required_round_count": 4,
        "reserved_test_access_allowed": False,
    },
    "retention_rule": (
        "All primary qualifiers proceed to matched frozen-defense evaluation. "
        "Retain at least one exact-template and one low-intensity candidate as "
        "threat-model breadth controls, even if one is a documented nonqualifier."
    ),
    "multiseed_rule": (
        "Do not select by seed-7 ASR rank alone. Use effectiveness, persistence, "
        "source-class breadth, trigger-family diversity, intensity diversity, "
        "and clean-utility stealth after matched plain-versus-defense screening."
    ),
    "main_defense_policy": (
        "Use the frozen Task 40 detector and reconstruction without trigger-specific "
        "threshold tuning or attack-specific retuning."
    ),
    "test_policy": (
        "Validation data may be used for development screening. Natural and diagnostic "
        "test arrays remain reserved and must not be loaded."
    ),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--partition-file", type=Path, required=True)
    p.add_argument("--clean-seed-dir", type=Path, required=True)
    p.add_argument("--warmup-dir", type=Path, required=True)
    p.add_argument("--trigger-spec-file", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--evaluation-batch-size", type=int, default=4096)
    return p.parse_args()


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def branch_valid(output_dir: Path, candidate_id: str) -> bool:
    metadata_path = output_dir / "task41b_backdoor_metadata.json"
    required = [
        metadata_path,
        output_dir / "tables" / "continuation_round_metrics.csv",
        output_dir / "tables" / "triggered_validation_asr_long.csv",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        metadata.get("mode") == "strong_attack"
        and metadata.get("replacement_policy") == "plain_fedavg"
        and metadata.get("trigger_candidate_id") == candidate_id
        and int(metadata.get("model_seed", -1)) == 7
        and int(metadata.get("attack_seed", -1)) == 7
        and abs(float(metadata.get("poison_fraction", -1)) - 0.01) < 1e-12
        and not any(
            bool(metadata.get(key, False))
            for key in (
                "test_arrays_loaded",
                "natural_test_accessed",
                "diagnostic_test_accessed",
                "test_sets_accessed",
            )
        )
    )


def main() -> int:
    args = parse_args()
    root = resolved(args.project_root)
    python_exe = root / ".venv" / "Scripts" / "python.exe"
    runner = root / "scripts" / "run_task41b_backdoor_smoke_v411b1.py"
    output_root = resolved(args.output_root)
    protocol_dir = output_root / "frozen_breadth_protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    for path in (
        python_exe,
        runner,
        resolved(args.data_file),
        resolved(args.partition_file),
        resolved(args.clean_seed_dir),
        resolved(args.warmup_dir),
        resolved(args.trigger_spec_file),
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    frozen = dict(PROTOCOL)
    frozen["fixed_run_parameters"] = {
        "model_seed": 7,
        "attack_seed": 7,
        "poison_fraction": 0.01,
        "label_policy": "dirty_label_all_nonbenign_to_benign",
        "deployment_policy": "centralized_full_trigger",
        "source_class": "DDoS",
        "target_class": "Benign",
        "malicious_clients": [1, 7, 8, 10, 14, 15, 17, 18],
        "continuation_rounds": 4,
        "probe_per_class": 48,
        "probe_seed": 3701,
        "ema_decay": 0.65,
        "minimum_source_samples": 1000,
        "test_sets_accessed": False,
    }
    frozen["panel"] = [
        {"slot": slot, "candidate_id": candidate, "family": family, "intensity": intensity}
        for slot, candidate, family, intensity in PANEL
    ]
    (protocol_dir / "task41b3_frozen_breadth_protocol.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )
    with (protocol_dir / "task41b3_frozen_candidate_panel.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["slot", "candidate_id", "family", "intensity"])
        writer.writerows(PANEL)

    print("===== TASK 41B.3 FROZEN PLAIN BREADTH GRID =====")
    print("Protocol frozen before new runs: True")
    print("Candidate count:", len(PANEL))
    print("Reserved test access allowed: False")

    completion = []
    for slot, candidate, family, intensity in PANEL:
        output_dir = output_root / "runs" / slot / "seed_7" / "plain_fedavg"
        if branch_valid(output_dir, candidate):
            print("SKIPPING COMPLETE VALID BRANCH:", output_dir)
        else:
            if output_dir.exists():
                raise RuntimeError(
                    "Existing branch is incomplete or mismatched. Inspect only this path: "
                    + str(output_dir)
                )
            cmd = [
                str(python_exe), "-u", str(runner),
                "--mode", "strong_attack",
                "--replacement-policy", "plain_fedavg",
                "--data-file", str(resolved(args.data_file)),
                "--partition-file", str(resolved(args.partition_file)),
                "--clean-seed-dir", str(resolved(args.clean_seed_dir)),
                "--warmup-dir", str(resolved(args.warmup_dir)),
                "--trigger-spec-file", str(resolved(args.trigger_spec_file)),
                "--trigger-candidate-id", candidate,
                "--label-policy", "dirty_label_all_nonbenign_to_benign",
                "--deployment-policy", "centralized_full_trigger",
                "--output-dir", str(output_dir),
                "--model-seed", "7",
                "--attack-seed", "7",
                "--num-clients", "20",
                "--continuation-rounds", "4",
                "--batch-size", str(args.batch_size),
                "--evaluation-batch-size", str(args.evaluation_batch_size),
                "--learning-rate", "0.0003",
                "--weight-decay", "0.0001",
                "--threads", str(args.threads),
                "--probe-per-class", "48",
                "--probe-seed", "3701",
                "--source-class", "DDoS",
                "--target-class", "Benign",
                "--ema-decay", "0.65",
                "--malicious-clients", "1,7,8,10,14,15,17,18",
                "--poison-fraction", "0.01",
                "--min-source-samples", "1000",
            ]
            print()
            print("RUNNING:", slot)
            print("CANDIDATE:", candidate)
            completed = subprocess.run(cmd, cwd=str(root), check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Branch failed with exit code {completed.returncode}: {candidate}"
                )
            if not branch_valid(output_dir, candidate):
                raise RuntimeError("Completion validation failed: " + str(output_dir))

        completion.append(
            {
                "slot": slot,
                "candidate_id": candidate,
                "family": family,
                "intensity": intensity,
                "output_dir": str(output_dir),
                "complete_valid": branch_valid(output_dir, candidate),
                "test_sets_accessed": False,
            }
        )

    with (protocol_dir / "task41b3_plain_breadth_completion.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(completion[0]))
        writer.writeheader()
        writer.writerows(completion)

    if not all(row["complete_valid"] for row in completion):
        raise RuntimeError("One or more branches are incomplete.")

    print()
    print("TASK 41B.3 PLAIN BREADTH GRID COMPLETE")
    print(f"Completed valid branches: {len(completion)}/{len(completion)}")
    print("Protocol artifacts:", protocol_dir)
    print("FINAL TEST ARRAYS LOADED: False")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("TASK 41B.3 GRID FAILED:", exc, file=sys.stderr)
        raise

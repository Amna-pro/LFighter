#!/usr/bin/env python3
"""
Task 41B.9 final integrity audit and evidence freeze.

This script verifies the completed Task 41B development evidence, rechecks the
frozen decision, confirms exact pairing and zero reserved-test access, creates a
SHA-256 manifest for all Task 41B evidence artifacts, and writes final closure
tables plus PNG/PDF figures.

It never loads dataset arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_SLOTS = ["flow_iat_exact", "active_idle_exact"]
EXPECTED_SEEDS = [7, 99, 123, 2026]
EXPECTED_STATUS = (
    "COMPLETED_ATTACK_SUPPORTED_DEFENSE_NOT_CONSISTENTLY_EFFECTIVE"
)
EXPECTED_ATTACK_EVIDENCE = (
    "MATERIAL_BUT_SEED_HETEROGENEOUS_BACKDOOR_EFFECT"
)
EXPECTED_DEFENSE_EVIDENCE = "NO_CONSISTENT_MULTISEED_EFFICACY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze and audit final Task 41B evidence."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--freeze-output-dir", type=Path, required=True)
    return parser.parse_args()


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


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )


def main() -> int:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    freeze_dir = args.freeze_output_dir.expanduser().resolve()
    tables_dir = freeze_dir / "tables"
    figures_dir = freeze_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    protocol_dir = root / "multiseed_protocol"
    summary_dir = root / "exact_multiseed_summary"

    protocol_json_path = (
        protocol_dir / "task41b7_exact_multiseed_protocol.json"
    )
    completion_csv_path = (
        protocol_dir / "task41b7_exact_multiseed_completion.csv"
    )
    completion_json_path = (
        protocol_dir / "task41b7_exact_multiseed_completion.json"
    )
    decision_json_path = summary_dir / "task41b8_multiseed_decision.json"
    pair_csv_path = (
        summary_dir / "tables" / "task41b8_exact_pair_seed_results.csv"
    )
    trigger_csv_path = (
        summary_dir / "tables" / "task41b8_trigger_multiseed_summary.csv"
    )
    seed_csv_path = (
        summary_dir / "tables" / "task41b8_seed_aggregated_summary.csv"
    )
    round_csv_path = (
        summary_dir / "tables" / "task41b8_roundwise_exact_pair_results.csv"
    )
    source_csv_path = (
        summary_dir
        / "tables"
        / "task41b8_final_per_source_exact_pair_results.csv"
    )
    test_audit_path = (
        summary_dir / "tables" / "task41b8_test_access_audit.csv"
    )

    required_paths = [
        protocol_json_path,
        completion_csv_path,
        completion_json_path,
        decision_json_path,
        pair_csv_path,
        trigger_csv_path,
        seed_csv_path,
        round_csv_path,
        source_csv_path,
        test_audit_path,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    protocol = load_json(protocol_json_path)
    completion_json = load_json(completion_json_path)
    decision = load_json(decision_json_path)
    completion = pd.read_csv(completion_csv_path)
    pairs = pd.read_csv(pair_csv_path)
    triggers = pd.read_csv(trigger_csv_path)
    seeds = pd.read_csv(seed_csv_path)
    rounds = pd.read_csv(round_csv_path)
    sources = pd.read_csv(source_csv_path)
    test_audit = pd.read_csv(test_audit_path)

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {"check_name": name, "passed": bool(passed), "detail": detail}
        )
        if not passed:
            raise RuntimeError(f"{name} failed: {detail}")

    record(
        "protocol_slots",
        protocol.get("selected_slots") == EXPECTED_SLOTS,
        str(protocol.get("selected_slots")),
    )
    record(
        "protocol_seeds",
        protocol.get("seeds") == EXPECTED_SEEDS,
        str(protocol.get("seeds")),
    )
    record(
        "protocol_no_attack_specific_retuning",
        not bool(protocol.get("attack_specific_retuning", True)),
        str(protocol.get("attack_specific_retuning")),
    )
    record(
        "protocol_no_threshold_retuning",
        not bool(protocol.get("threshold_retuning", True)),
        str(protocol.get("threshold_retuning")),
    )
    record(
        "protocol_no_test_access",
        not bool(protocol.get("test_sets_accessed", True)),
        str(protocol.get("test_sets_accessed")),
    )
    record(
        "completion_all_exact",
        bool(completion_json.get("all_exact_pairs_complete", False)),
        str(completion_json.get("all_exact_pairs_complete")),
    )
    record(
        "completion_pair_count",
        int(completion_json.get("completed_exact_pair_count", -1)) == 8,
        str(completion_json.get("completed_exact_pair_count")),
    )
    record(
        "completion_rows",
        len(completion) == 8,
        str(len(completion)),
    )

    for column in (
        "complete_exact_pair",
        "poison_plan_hash_exact_match",
        "trigger_spec_hash_exact_match",
        "partition_hash_exact_match",
        "warmup_profile_hash_exact_match",
    ):
        record(
            f"completion_{column}",
            column in completion.columns
            and bool_series(completion[column]).all(),
            column,
        )

    record(
        "completion_plain_no_test_access",
        not bool_series(completion["plain_test_sets_accessed"]).any(),
        str(bool_series(completion["plain_test_sets_accessed"]).sum()),
    )
    record(
        "completion_defended_no_test_access",
        not bool_series(completion["defended_test_sets_accessed"]).any(),
        str(bool_series(completion["defended_test_sets_accessed"]).sum()),
    )
    record(
        "decision_status",
        decision.get("status") == EXPECTED_STATUS,
        str(decision.get("status")),
    )
    record(
        "decision_attack_evidence",
        decision.get("attack_evidence") == EXPECTED_ATTACK_EVIDENCE,
        str(decision.get("attack_evidence")),
    )
    record(
        "decision_defense_evidence",
        decision.get("frozen_defense_evidence")
        == EXPECTED_DEFENSE_EVIDENCE,
        str(decision.get("frozen_defense_evidence")),
    )
    record(
        "decision_method_not_reopened",
        not bool(decision.get("method_reopened", True)),
        str(decision.get("method_reopened")),
    )
    record(
        "decision_no_final_claim",
        not bool(decision.get("final_paper_claim_allowed", True)),
        str(decision.get("final_paper_claim_allowed")),
    )
    record(
        "decision_no_test_access",
        not bool(decision.get("reserved_test_accessed", True)),
        str(decision.get("reserved_test_accessed")),
    )

    record("pair_rows", len(pairs) == 8, str(len(pairs)))
    record(
        "pair_slot_panel",
        sorted(pairs["slot"].unique().tolist())
        == sorted(EXPECTED_SLOTS),
        str(sorted(pairs["slot"].unique().tolist())),
    )
    record(
        "pair_seed_panel",
        sorted(pairs["seed"].unique().tolist()) == EXPECTED_SEEDS,
        str(sorted(pairs["seed"].unique().tolist())),
    )
    record(
        "pair_exact_metadata",
        bool_series(pairs["exact_pair_metadata_match"]).all(),
        str(bool_series(pairs["exact_pair_metadata_match"]).sum()),
    )
    record(
        "pair_no_retuning",
        not bool_series(pairs["attack_specific_retuning"]).any(),
        str(bool_series(pairs["attack_specific_retuning"]).sum()),
    )
    record(
        "pair_no_test_access",
        not bool_series(pairs["test_sets_accessed"]).any(),
        str(bool_series(pairs["test_sets_accessed"]).sum()),
    )

    record("trigger_rows", len(triggers) == 2, str(len(triggers)))
    record("seed_rows", len(seeds) == 4, str(len(seeds)))
    record("round_rows", len(rounds) == 32, str(len(rounds)))
    record("source_rows", len(sources) == 56, str(len(sources)))
    record(
        "test_audit_all_false",
        not bool_series(test_audit["value"]).any(),
        str(bool_series(test_audit["value"]).sum()),
    )

    computed_plain = float(
        pairs["plain_mean_macro_triggered_asr"].mean()
    )
    computed_defended = float(
        pairs["defended_mean_macro_triggered_asr"].mean()
    )
    computed_reduction = computed_plain - computed_defended
    computed_relative = computed_reduction / computed_plain

    record(
        "decision_overall_plain_mean",
        np.isclose(
            computed_plain,
            float(decision["overall_plain_mean_macro_triggered_asr"]),
            atol=1e-12,
        ),
        f"{computed_plain:.12f}",
    )
    record(
        "decision_overall_defended_mean",
        np.isclose(
            computed_defended,
            float(decision["overall_defended_mean_macro_triggered_asr"]),
            atol=1e-12,
        ),
        f"{computed_defended:.12f}",
    )
    record(
        "decision_overall_relative_reduction",
        np.isclose(
            computed_relative,
            float(decision["overall_relative_reduction_fraction"]),
            atol=1e-12,
        ),
        f"{computed_relative:.12f}",
    )

    expected_figures = [
        summary_dir / "figures" / "task41b8_multiseed_triggered_asr.png",
        summary_dir / "figures" / "task41b8_multiseed_triggered_asr.pdf",
        summary_dir / "figures" / "task41b8_seedwise_asr_reduction.png",
        summary_dir / "figures" / "task41b8_seedwise_asr_reduction.pdf",
        summary_dir / "figures" / "task41b8_seed_direction_consistency.png",
        summary_dir / "figures" / "task41b8_seed_direction_consistency.pdf",
        summary_dir / "figures" / "task41b8_clean_utility_delta.png",
        summary_dir / "figures" / "task41b8_clean_utility_delta.pdf",
        summary_dir / "figures" / "task41b8_detector_recall.png",
        summary_dir / "figures" / "task41b8_detector_recall.pdf",
    ]
    for path in expected_figures:
        record(
            f"figure_exists_{path.name}",
            path.exists() and path.stat().st_size > 0,
            str(path),
        )

    evidence_files = sorted(
        {
            *required_paths,
            *expected_figures,
            *[
                path
                for path in root.rglob("*")
                if path.is_file()
                and (
                    "task41b" in path.name.lower()
                    or path.parent.name
                    in {
                        "tables",
                        "figures",
                        "selected_defense_protocol",
                        "multiseed_protocol",
                    }
                )
            ],
        },
        key=lambda path: str(path).lower(),
    )

    manifest_rows = []
    for path in evidence_files:
        try:
            relative_path = str(path.relative_to(root))
        except ValueError:
            relative_path = str(path)
        manifest_rows.append(
            {
                "relative_path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    checks_df = pd.DataFrame(checks)
    manifest_df = pd.DataFrame(manifest_rows)
    trigger_closure = triggers[
        [
            "slot",
            "plain_mean_macro_triggered_asr",
            "defended_mean_macro_triggered_asr",
            "mean_macro_triggered_asr_reduction",
            "relative_reduction_from_group_means",
            "paired_bootstrap_95ci_low",
            "paired_bootstrap_95ci_high",
            "exact_sign_flip_two_sided_p",
            "improved_seed_count",
            "worsened_seed_count",
            "mean_defended_malicious_recall",
            "maximum_defended_benign_fpr",
        ]
    ].copy()

    closure_row = pd.DataFrame(
        [
            {
                "experiment_version": "4.11B.9",
                "stage": "task41b_final_integrity_audit_and_evidence_freeze",
                "task41b_status": EXPECTED_STATUS,
                "attack_evidence": EXPECTED_ATTACK_EVIDENCE,
                "defense_evidence": EXPECTED_DEFENSE_EVIDENCE,
                "exact_pair_count": 8,
                "trigger_count": 2,
                "seed_count": 4,
                "round_pair_count": 32,
                "overall_plain_mean_asr": computed_plain,
                "overall_defended_mean_asr": computed_defended,
                "overall_relative_asr_reduction": computed_relative,
                "mean_defended_malicious_recall": float(
                    decision["mean_defended_malicious_recall"]
                ),
                "maximum_defended_benign_fpr": float(
                    decision["maximum_defended_benign_fpr"]
                ),
                "method_reopened": False,
                "attack_specific_retuning": False,
                "threshold_retuning": False,
                "reserved_test_accessed": False,
                "final_paper_claim_allowed": False,
                "integrity_check_count": len(checks_df),
                "integrity_checks_passed": int(checks_df["passed"].sum()),
                "manifest_file_count": len(manifest_df),
                "task41b_frozen": True,
                "next_stage": (
                    "Task 41C must be a separately preregistered extension. "
                    "Task 41B evidence and method remain unchanged."
                ),
            }
        ]
    )

    checks_df.to_csv(
        tables_dir / "task41b9_integrity_checks.csv", index=False
    )
    manifest_df.to_csv(
        tables_dir / "task41b9_evidence_manifest_sha256.csv", index=False
    )
    trigger_closure.to_csv(
        tables_dir / "task41b9_trigger_closure_summary.csv", index=False
    )
    closure_row.to_csv(
        tables_dir / "task41b9_final_closure_decision.csv", index=False
    )

    freeze_json = closure_row.iloc[0].to_dict()
    freeze_json["source_decision_sha256"] = sha256_file(decision_json_path)
    freeze_json["source_protocol_sha256"] = sha256_file(protocol_json_path)
    freeze_json["source_completion_sha256"] = sha256_file(completion_csv_path)
    (freeze_dir / "task41b9_final_closure_decision.json").write_text(
        json.dumps(freeze_json, indent=2), encoding="utf-8"
    )

    x = np.arange(len(trigger_closure))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.bar(
        x - width / 2,
        trigger_closure["plain_mean_macro_triggered_asr"],
        width,
        label="Plain FedAvg",
    )
    ax.bar(
        x + width / 2,
        trigger_closure["defended_mean_macro_triggered_asr"],
        width,
        label="Trusted reconstruction",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(trigger_closure["slot"], rotation=20, ha="right")
    ax.set_ylabel("Mean macro triggered ASR")
    ax.set_ylim(0, 1)
    ax.set_title("Task 41B frozen multiseed closure")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "task41b9_frozen_multiseed_closure")

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.bar(
        trigger_closure["slot"],
        trigger_closure["improved_seed_count"],
        label="Improved seeds",
    )
    ax.bar(
        trigger_closure["slot"],
        trigger_closure["worsened_seed_count"],
        bottom=trigger_closure["improved_seed_count"],
        label="Worsened seeds",
    )
    ax.set_ylim(0, 4)
    ax.set_ylabel("Seed count")
    ax.set_title("Task 41B defense direction across seeds")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "task41b9_seed_direction_closure")

    print("===== TASK 41B.9 FINAL INTEGRITY AUDIT AND FREEZE =====")
    print()
    print(closure_row.to_string(index=False))
    print()
    print("Integrity checks passed:", f"{checks_df['passed'].sum()}/{len(checks_df)}")
    print("Manifest files:", len(manifest_df))
    print("TASK 41B FROZEN: True")
    print("METHOD REOPENED: False")
    print("FINAL PAPER CLAIM ALLOWED: False")
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("TEST SETS ACCESSED: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

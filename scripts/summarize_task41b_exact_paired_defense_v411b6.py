#!/usr/bin/env python3
"""
Task 41B.6 exact paired plain-versus-frozen-defense audit.

This script:
- reads the frozen Task 41B.4 qualification and Task 41B.5 completion artifacts,
- validates exact pairing between plain FedAvg and trusted reconstruction,
- compares clean validation utility, triggered ASR, detector behavior, and
  per-source final-round ASR,
- preserves every primary qualifier for confirmatory multiseed evaluation,
- keeps the breadth negative control out of multiseed promotion,
- writes research-ready CSV/JSON evidence and PNG/PDF figures,
- never loads train, validation, natural-test, or diagnostic-test arrays.

This is a development-stage paired audit, not a final paper result.
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

REQUIRED_ROUND_COLUMNS = (
    "monitoring_round",
    "val_macro_f1",
    "triggered_asr_macro_source",
    "triggered_asr_worst_source",
    "malicious_recall",
    "benign_false_positive_rate",
    "replaced_clients",
)

REQUIRED_SOURCE_COLUMNS = (
    "monitoring_round",
    "source_class_id",
    "source_class_name",
    "triggered_asr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit exact paired Task 41B plain and frozen-defense branches."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qualification-dir", type=Path, required=True)
    parser.add_argument("--audit-output-dir", type=Path, required=True)
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


def safe_relative_reduction(plain: float, defended: float) -> float:
    if not np.isfinite(plain) or abs(plain) <= 1e-12:
        return float("nan")
    return float((plain - defended) / plain)


def exact_metadata_match(
    plain: dict[str, Any],
    defended: dict[str, Any],
) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for field in MATCH_FIELDS:
        if plain.get(field) != defended.get(field):
            mismatches.append(field)
    return not mismatches, mismatches


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    qualification_dir = args.qualification_dir.expanduser().resolve()
    audit_dir = args.audit_output_dir.expanduser().resolve()

    tables_dir = audit_dir / "tables"
    figures_dir = audit_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    qualification_path = (
        qualification_dir
        / "tables"
        / "task41b3_plain_breadth_qualification.csv"
    )
    qualification_decision_path = (
        qualification_dir / "task41b3_plain_breadth_decision.json"
    )
    completion_path = (
        output_root
        / "selected_defense_protocol"
        / "task41b5_selected_defense_completion.csv"
    )
    defense_protocol_path = (
        output_root
        / "selected_defense_protocol"
        / "task41b5_selected_defense_protocol.json"
    )

    for path in (
        qualification_path,
        qualification_decision_path,
        completion_path,
        defense_protocol_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    qualification = pd.read_csv(qualification_path)
    completion = pd.read_csv(completion_path)
    qualification_decision = load_json(qualification_decision_path)
    defense_protocol = load_json(defense_protocol_path)

    selected = qualification[
        bool_series(qualification["proceed_to_paired_defense"])
    ].copy()

    if len(selected) != 3:
        raise RuntimeError(
            f"Expected exactly 3 frozen paired selections, found {len(selected)}."
        )
    if int(bool_series(selected["passes_primary_qualification"]).sum()) != 2:
        raise RuntimeError("Expected exactly 2 primary qualifiers.")
    if int((selected["selection_label"] == "BREADTH_NEGATIVE_CONTROL").sum()) != 1:
        raise RuntimeError("Expected exactly 1 breadth negative control.")
    if bool(qualification_decision.get("multiseed_selection_allowed_now", True)):
        raise RuntimeError(
            "The Task 41B.4 decision unexpectedly allowed multiseed before paired audit."
        )
    if bool(defense_protocol.get("attack_specific_retuning", True)):
        raise RuntimeError("The selected defense protocol reports attack-specific retuning.")
    if bool(defense_protocol.get("threshold_retuning", True)):
        raise RuntimeError("The selected defense protocol reports threshold retuning.")
    if bool(defense_protocol.get("test_sets_accessed", True)):
        raise RuntimeError("The selected defense protocol reports reserved test access.")

    completion_selected = completion[
        completion["slot"].isin(selected["slot"])
    ].copy()
    if len(completion_selected) != 3:
        raise RuntimeError(
            "The Task 41B.5 completion table does not contain all three selections."
        )
    if not bool_series(completion_selected["complete_exact_pair"]).all():
        raise RuntimeError("At least one Task 41B.5 pair is not marked exact.")
    for field in (
        "poison_plan_hash_exact_match",
        "trigger_spec_hash_exact_match",
        "partition_hash_exact_match",
    ):
        if field not in completion_selected.columns:
            raise RuntimeError(f"Completion table lacks {field}.")
        if not bool_series(completion_selected[field]).all():
            raise RuntimeError(f"At least one exact-pair check failed: {field}.")

    summary_rows: list[dict[str, Any]] = []
    round_rows: list[pd.DataFrame] = []
    source_rows: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for role, path in (
        ("qualification_csv", qualification_path),
        ("qualification_decision_json", qualification_decision_path),
        ("selected_defense_completion_csv", completion_path),
        ("selected_defense_protocol_json", defense_protocol_path),
    ):
        manifest_rows.append(
            {
                "role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    selected = selected.sort_values("selection_rank").reset_index(drop=True)

    for row in selected.itertuples(index=False):
        slot = str(row.slot)
        candidate_id = str(row.candidate_id)
        selection_label = str(row.selection_label)
        primary = bool(row.passes_primary_qualification)

        plain_dir = output_root / "runs" / slot / "seed_7" / "plain_fedavg"
        defended_dir = (
            output_root
            / "runs"
            / slot
            / "seed_7"
            / "trusted_reconstruction"
        )

        plain_metadata_path = plain_dir / "task41b_backdoor_metadata.json"
        defended_metadata_path = defended_dir / "task41b_backdoor_metadata.json"
        plain_round_path = plain_dir / "tables" / "continuation_round_metrics.csv"
        defended_round_path = (
            defended_dir / "tables" / "continuation_round_metrics.csv"
        )
        plain_source_path = (
            plain_dir / "tables" / "triggered_validation_asr_long.csv"
        )
        defended_source_path = (
            defended_dir / "tables" / "triggered_validation_asr_long.csv"
        )

        for path in (
            plain_metadata_path,
            defended_metadata_path,
            plain_round_path,
            defended_round_path,
            plain_source_path,
            defended_source_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        plain_meta = load_json(plain_metadata_path)
        defended_meta = load_json(defended_metadata_path)

        if plain_meta.get("replacement_policy") != "plain_fedavg":
            raise RuntimeError(f"{slot} plain policy mismatch.")
        if defended_meta.get("replacement_policy") != "trusted_reconstruction":
            raise RuntimeError(f"{slot} defended policy mismatch.")
        if plain_meta.get("mode") != "strong_attack":
            raise RuntimeError(f"{slot} plain mode mismatch.")
        if defended_meta.get("mode") != "strong_attack":
            raise RuntimeError(f"{slot} defended mode mismatch.")
        if plain_meta.get("trigger_candidate_id") != candidate_id:
            raise RuntimeError(f"{slot} plain candidate mismatch.")
        if defended_meta.get("trigger_candidate_id") != candidate_id:
            raise RuntimeError(f"{slot} defended candidate mismatch.")
        if bool(plain_meta.get("attack_specific_retuning", True)):
            raise RuntimeError(f"{slot} plain metadata reports attack-specific retuning.")
        if bool(defended_meta.get("attack_specific_retuning", True)):
            raise RuntimeError(
                f"{slot} defended metadata reports attack-specific retuning."
            )

        exact, mismatches = exact_metadata_match(plain_meta, defended_meta)
        if not exact:
            raise RuntimeError(
                f"{slot} exact-pair metadata mismatch: {','.join(mismatches)}"
            )

        for branch_name, metadata in (
            ("plain_fedavg", plain_meta),
            ("trusted_reconstruction", defended_meta),
        ):
            for field in FORBIDDEN_TEST_FLAGS:
                value = bool(metadata.get(field, False))
                test_rows.append(
                    {
                        "slot": slot,
                        "branch": branch_name,
                        "field": field,
                        "value": value,
                    }
                )
                if value:
                    raise RuntimeError(
                        f"{slot} {branch_name} reports reserved test access: {field}"
                    )

        plain_round = pd.read_csv(plain_round_path).sort_values(
            "monitoring_round"
        ).reset_index(drop=True)
        defended_round = pd.read_csv(defended_round_path).sort_values(
            "monitoring_round"
        ).reset_index(drop=True)

        missing_plain = set(REQUIRED_ROUND_COLUMNS) - set(plain_round.columns)
        missing_defended = set(REQUIRED_ROUND_COLUMNS) - set(
            defended_round.columns
        )
        if missing_plain:
            raise RuntimeError(
                f"{slot} plain round table lacks {sorted(missing_plain)}."
            )
        if missing_defended:
            raise RuntimeError(
                f"{slot} defended round table lacks {sorted(missing_defended)}."
            )
        if len(plain_round) != 4 or len(defended_round) != 4:
            raise RuntimeError(
                f"{slot} must contain exactly four rounds in both branches."
            )
        if not np.array_equal(
            plain_round["monitoring_round"].to_numpy(),
            defended_round["monitoring_round"].to_numpy(),
        ):
            raise RuntimeError(f"{slot} monitoring-round alignment mismatch.")

        pair_round = pd.DataFrame(
            {
                "slot": slot,
                "candidate_id": candidate_id,
                "selection_label": selection_label,
                "passes_primary_qualification": primary,
                "monitoring_round": plain_round["monitoring_round"].astype(int),
                "plain_val_macro_f1": plain_round["val_macro_f1"],
                "defended_val_macro_f1": defended_round["val_macro_f1"],
                "clean_macro_f1_delta_defended_minus_plain": (
                    defended_round["val_macro_f1"]
                    - plain_round["val_macro_f1"]
                ),
                "plain_macro_triggered_asr": plain_round[
                    "triggered_asr_macro_source"
                ],
                "defended_macro_triggered_asr": defended_round[
                    "triggered_asr_macro_source"
                ],
                "macro_triggered_asr_reduction_plain_minus_defended": (
                    plain_round["triggered_asr_macro_source"]
                    - defended_round["triggered_asr_macro_source"]
                ),
                "plain_worst_source_triggered_asr": plain_round[
                    "triggered_asr_worst_source"
                ],
                "defended_worst_source_triggered_asr": defended_round[
                    "triggered_asr_worst_source"
                ],
                "worst_source_asr_reduction_plain_minus_defended": (
                    plain_round["triggered_asr_worst_source"]
                    - defended_round["triggered_asr_worst_source"]
                ),
                "plain_malicious_recall": plain_round["malicious_recall"],
                "defended_malicious_recall": defended_round["malicious_recall"],
                "plain_benign_fpr": plain_round[
                    "benign_false_positive_rate"
                ],
                "defended_benign_fpr": defended_round[
                    "benign_false_positive_rate"
                ],
                "plain_replaced_clients": plain_round["replaced_clients"],
                "defended_replaced_clients": defended_round[
                    "replaced_clients"
                ],
            }
        )
        pair_round[
            "macro_asr_round_improved"
        ] = pair_round[
            "macro_triggered_asr_reduction_plain_minus_defended"
        ] > 0.0
        round_rows.append(pair_round)

        plain_source = pd.read_csv(plain_source_path)
        defended_source = pd.read_csv(defended_source_path)
        missing_plain_source = set(REQUIRED_SOURCE_COLUMNS) - set(
            plain_source.columns
        )
        missing_defended_source = set(REQUIRED_SOURCE_COLUMNS) - set(
            defended_source.columns
        )
        if missing_plain_source:
            raise RuntimeError(
                f"{slot} plain source table lacks {sorted(missing_plain_source)}."
            )
        if missing_defended_source:
            raise RuntimeError(
                f"{slot} defended source table lacks "
                f"{sorted(missing_defended_source)}."
            )

        final_round = int(plain_round["monitoring_round"].max())
        plain_final_source = plain_source[
            plain_source["monitoring_round"] == final_round
        ][
            ["source_class_id", "source_class_name", "triggered_asr"]
        ].rename(columns={"triggered_asr": "plain_final_triggered_asr"})
        defended_final_source = defended_source[
            defended_source["monitoring_round"] == final_round
        ][
            ["source_class_id", "source_class_name", "triggered_asr"]
        ].rename(columns={"triggered_asr": "defended_final_triggered_asr"})

        final_source = plain_final_source.merge(
            defended_final_source,
            on=["source_class_id", "source_class_name"],
            how="inner",
            validate="one_to_one",
        )
        if len(final_source) != 7:
            raise RuntimeError(
                f"{slot} final per-source pair contains {len(final_source)} "
                "classes, expected 7."
            )
        final_source.insert(0, "slot", slot)
        final_source.insert(1, "candidate_id", candidate_id)
        final_source.insert(2, "selection_label", selection_label)
        final_source[
            "final_triggered_asr_reduction_plain_minus_defended"
        ] = (
            final_source["plain_final_triggered_asr"]
            - final_source["defended_final_triggered_asr"]
        )
        final_source["final_source_improved"] = (
            final_source[
                "final_triggered_asr_reduction_plain_minus_defended"
            ]
            > 0.0
        )
        source_rows.append(final_source)

        mean_plain_asr = float(
            plain_round["triggered_asr_macro_source"].mean()
        )
        mean_defended_asr = float(
            defended_round["triggered_asr_macro_source"].mean()
        )
        mean_asr_reduction = mean_plain_asr - mean_defended_asr
        final_plain_asr = float(
            plain_round.iloc[-1]["triggered_asr_macro_source"]
        )
        final_defended_asr = float(
            defended_round.iloc[-1]["triggered_asr_macro_source"]
        )
        final_asr_reduction = final_plain_asr - final_defended_asr

        if mean_asr_reduction > 1e-12:
            mean_effect_direction = "IMPROVED"
        elif mean_asr_reduction < -1e-12:
            mean_effect_direction = "WORSENED"
        else:
            mean_effect_direction = "UNCHANGED"

        if final_asr_reduction > 1e-12:
            final_effect_direction = "IMPROVED"
        elif final_asr_reduction < -1e-12:
            final_effect_direction = "WORSENED"
        else:
            final_effect_direction = "UNCHANGED"

        summary_rows.append(
            {
                "slot": slot,
                "candidate_id": candidate_id,
                "selection_label": selection_label,
                "passes_primary_qualification": primary,
                "round_count": 4,
                "plain_mean_clean_macro_f1": float(
                    plain_round["val_macro_f1"].mean()
                ),
                "defended_mean_clean_macro_f1": float(
                    defended_round["val_macro_f1"].mean()
                ),
                "clean_macro_f1_delta_defended_minus_plain": float(
                    defended_round["val_macro_f1"].mean()
                    - plain_round["val_macro_f1"].mean()
                ),
                "plain_mean_macro_triggered_asr": mean_plain_asr,
                "defended_mean_macro_triggered_asr": mean_defended_asr,
                "mean_macro_triggered_asr_reduction": mean_asr_reduction,
                "relative_mean_asr_reduction_fraction": safe_relative_reduction(
                    mean_plain_asr, mean_defended_asr
                ),
                "plain_final_round_macro_triggered_asr": final_plain_asr,
                "defended_final_round_macro_triggered_asr": final_defended_asr,
                "final_round_macro_triggered_asr_reduction": final_asr_reduction,
                "improved_round_count": int(
                    (
                        plain_round["triggered_asr_macro_source"]
                        > defended_round["triggered_asr_macro_source"]
                    ).sum()
                ),
                "final_source_classes_improved": int(
                    final_source["final_source_improved"].sum()
                ),
                "plain_mean_malicious_recall": float(
                    plain_round["malicious_recall"].mean()
                ),
                "defended_mean_malicious_recall": float(
                    defended_round["malicious_recall"].mean()
                ),
                "plain_mean_benign_fpr": float(
                    plain_round["benign_false_positive_rate"].mean()
                ),
                "defended_mean_benign_fpr": float(
                    defended_round[
                        "benign_false_positive_rate"
                    ].mean()
                ),
                "defended_total_replaced_client_rounds": int(
                    defended_round["replaced_clients"].sum()
                ),
                "mean_effect_direction": mean_effect_direction,
                "final_effect_direction": final_effect_direction,
                "exact_pair_metadata_match": True,
                "poison_plan_hash_exact_match": True,
                "trigger_spec_hash_exact_match": True,
                "partition_hash_exact_match": True,
                "attack_specific_retuning": False,
                "test_sets_accessed": False,
            }
        )

        for branch_name, paths in (
            (
                "plain",
                (
                    plain_metadata_path,
                    plain_round_path,
                    plain_source_path,
                ),
            ),
            (
                "defended",
                (
                    defended_metadata_path,
                    defended_round_path,
                    defended_source_path,
                ),
            ),
        ):
            for path in paths:
                manifest_rows.append(
                    {
                        "role": f"{slot}_{branch_name}_{path.name}",
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    all_rounds = pd.concat(round_rows, ignore_index=True)
    all_sources = pd.concat(source_rows, ignore_index=True)

    primary = summary[summary["passes_primary_qualification"]].copy()
    negative = summary[
        summary["selection_label"] == "BREADTH_NEGATIVE_CONTROL"
    ].copy()

    if len(primary) != 2 or len(negative) != 1:
        raise RuntimeError("Unexpected primary/control composition in paired audit.")

    primary_all_mean_improved = bool(
        (primary["mean_macro_triggered_asr_reduction"] > 0.0).all()
    )
    primary_all_final_improved = bool(
        (primary["final_round_macro_triggered_asr_reduction"] > 0.0).all()
    )
    primary_any_mean_worsened = bool(
        (primary["mean_macro_triggered_asr_reduction"] < 0.0).any()
    )
    primary_any_final_worsened = bool(
        (primary["final_round_macro_triggered_asr_reduction"] < 0.0).any()
    )

    if primary_all_mean_improved and primary_all_final_improved:
        seed7_defense_evidence = "UNIFORM_DIRECTIONAL_SUPPORT"
    elif primary_all_mean_improved:
        seed7_defense_evidence = "MEAN_SUPPORT_BUT_FINAL_ROUND_MIXED"
    elif primary_any_mean_worsened or primary_any_final_worsened:
        seed7_defense_evidence = "MIXED_NO_UNIFORM_EFFICACY"
    else:
        seed7_defense_evidence = "NO_DIRECTIONAL_CHANGE"

    multiseed_slots = primary["slot"].tolist()
    multiseed_candidates = primary["candidate_id"].tolist()

    decision = {
        "experiment_version": "4.11B.6",
        "stage": "task41b_exact_paired_plain_vs_frozen_defense_audit",
        "status": "development_paired_audit_not_final_paper_result",
        "completed_exact_pair_count": int(len(summary)),
        "primary_qualifier_count": int(len(primary)),
        "breadth_negative_control_count": int(len(negative)),
        "seed7_defense_evidence": seed7_defense_evidence,
        "positive_mean_effect_primary_count": int(
            (primary["mean_macro_triggered_asr_reduction"] > 0.0).sum()
        ),
        "positive_final_effect_primary_count": int(
            (primary["final_round_macro_triggered_asr_reduction"] > 0.0).sum()
        ),
        "attack_specific_retuning": False,
        "threshold_retuning": False,
        "reserved_test_accessed": False,
        "multiseed_selection_allowed_now": True,
        "multiseed_selection_basis": (
            "Promote every frozen primary qualifier irrespective of seed-7 "
            "defense direction. This prevents post-hoc selection based on "
            "favorable defense outcomes."
        ),
        "multiseed_selected_slots": multiseed_slots,
        "multiseed_selected_candidates": multiseed_candidates,
        "negative_control_multiseed_selected": False,
        "next_stage": (
            "Run exact matched plain-versus-trusted-reconstruction multiseed "
            "evaluation for both frozen primary qualifiers using seeds "
            "7, 99, 123, and 2026. Preserve seed 7, add only missing seeds, "
            "and keep all Task 41 settings frozen."
        ),
        "final_paper_claim_allowed": False,
        "test_sets_accessed": False,
    }

    summary.to_csv(
        tables_dir / "task41b6_exact_pair_summary.csv", index=False
    )
    all_rounds.to_csv(
        tables_dir / "task41b6_roundwise_pair_differences.csv", index=False
    )
    all_sources.to_csv(
        tables_dir / "task41b6_final_per_source_pair_differences.csv",
        index=False,
    )
    pd.DataFrame(test_rows).to_csv(
        tables_dir / "task41b6_test_access_audit.csv", index=False
    )
    pd.DataFrame(manifest_rows).to_csv(
        tables_dir / "task41b6_input_manifest_sha256.csv", index=False
    )
    (audit_dir / "task41b6_paired_defense_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    plot_order = summary.sort_values(
        ["passes_primary_qualification", "selection_rank" if "selection_rank" in summary.columns else "slot"],
        ascending=[False, True],
    ) if "selection_rank" in summary.columns else summary.copy()
    x = np.arange(len(plot_order))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.bar(
        x - width / 2,
        plot_order["plain_mean_macro_triggered_asr"],
        width,
        label="Plain FedAvg",
    )
    ax.bar(
        x + width / 2,
        plot_order["defended_mean_macro_triggered_asr"],
        width,
        label="Trusted reconstruction",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_order["slot"], rotation=25, ha="right")
    ax.set_ylabel("Mean macro triggered ASR")
    ax.set_ylim(0, 1)
    ax.set_title("Task 41B exact paired mean triggered ASR")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "task41b6_mean_triggered_asr_pair")

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.bar(
        x - width / 2,
        plot_order["plain_final_round_macro_triggered_asr"],
        width,
        label="Plain FedAvg",
    )
    ax.bar(
        x + width / 2,
        plot_order["defended_final_round_macro_triggered_asr"],
        width,
        label="Trusted reconstruction",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_order["slot"], rotation=25, ha="right")
    ax.set_ylabel("Final-round macro triggered ASR")
    ax.set_ylim(0, 1)
    ax.set_title("Task 41B exact paired final-round triggered ASR")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "task41b6_final_triggered_asr_pair")

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.bar(
        plot_order["slot"],
        plot_order["mean_macro_triggered_asr_reduction"],
    )
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel("ASR reduction, plain minus defended")
    ax.set_title("Task 41B mean ASR reduction by trigger")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "task41b6_mean_asr_reduction")

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.bar(
        plot_order["slot"],
        plot_order["clean_macro_f1_delta_defended_minus_plain"],
    )
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel("Clean macro-F1 delta, defended minus plain")
    ax.set_title("Task 41B paired clean-utility change")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "task41b6_clean_macro_f1_delta")

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.bar(
        plot_order["slot"],
        plot_order["defended_mean_malicious_recall"],
    )
    ax.set_ylabel("Mean malicious-client recall")
    ax.set_ylim(0, 1)
    ax.set_title("Task 41B frozen detector recall under backdoor attacks")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "task41b6_detector_recall")

    display_columns = [
        "slot",
        "selection_label",
        "plain_mean_macro_triggered_asr",
        "defended_mean_macro_triggered_asr",
        "mean_macro_triggered_asr_reduction",
        "relative_mean_asr_reduction_fraction",
        "plain_final_round_macro_triggered_asr",
        "defended_final_round_macro_triggered_asr",
        "improved_round_count",
        "defended_mean_malicious_recall",
        "defended_total_replaced_client_rounds",
        "mean_effect_direction",
    ]

    print("===== TASK 41B.6 EXACT PAIRED DEFENSE AUDIT =====")
    print(summary[display_columns].to_string(index=False))
    print()
    print("Seed-7 defense evidence:", seed7_defense_evidence)
    print("Primary qualifiers promoted to multiseed:", multiseed_slots)
    print("Negative control promoted to multiseed: False")
    print("MULTISEED SELECTION ALLOWED NOW: True")
    print("FINAL PAPER CLAIM ALLOWED: False")
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("TEST SETS ACCESSED: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

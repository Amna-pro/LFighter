#!/usr/bin/env python3
"""
Task 41B.4 frozen plain-breadth qualification audit.

This script:
- reads the six completed seed-7 plain backdoor branches,
- discovers the matched Task 41B zero-poison adapter,
- applies the already frozen Task 41B.3 qualification rules,
- selects only protocol-mandated breadth controls,
- writes CSV/JSON evidence plus PNG/PDF figures,
- never loads model data or reserved test arrays.

This is a development-stage selection audit, not a final paper result.
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


FORBIDDEN_TEST_FLAGS = (
    "test_arrays_loaded",
    "natural_test_accessed",
    "diagnostic_test_accessed",
    "test_sets_accessed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the frozen Task 41B.3 plain-trigger qualification rules."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def discover_zero_poison(output_root: Path) -> tuple[Path, Path, pd.DataFrame, dict[str, Any]]:
    metadata_candidates = sorted(
        output_root.rglob("task41b_zero_poison_adapter_metadata.json")
    )
    if not metadata_candidates:
        raise FileNotFoundError(
            "No Task 41B zero-poison adapter metadata was found under "
            f"{output_root}. The frozen clean-utility criterion cannot be audited."
        )

    valid: list[tuple[Path, dict[str, Any]]] = []
    for metadata_path in metadata_candidates:
        metadata = read_json(metadata_path)
        if (
            metadata.get("stage") == "task41b_zero_poison_exact_equivalence_adapter"
            and int(metadata.get("model_seed", 7)) == 7
            and abs(float(metadata.get("poison_fraction", -1.0))) < 1e-12
            and not any(bool(metadata.get(key, False)) for key in FORBIDDEN_TEST_FLAGS)
        ):
            valid.append((metadata_path, metadata))

    if len(valid) != 1:
        raise RuntimeError(
            "Expected exactly one valid seed-7 Task 41B zero-poison adapter, "
            f"found {len(valid)}."
        )

    metadata_path, metadata = valid[0]
    zero_dir = metadata_path.parent

    metric_candidates: list[tuple[Path, pd.DataFrame]] = []
    for csv_path in sorted(zero_dir.rglob("*.csv")):
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        if {"monitoring_round", "val_macro_f1"}.issubset(frame.columns):
            metric_candidates.append((csv_path, frame))

    exact_names = [
        item for item in metric_candidates
        if item[0].name == "continuation_round_metrics.csv"
    ]
    if len(exact_names) == 1:
        metric_path, metric_frame = exact_names[0]
    elif len(metric_candidates) == 1:
        metric_path, metric_frame = metric_candidates[0]
    else:
        raise RuntimeError(
            "Could not identify exactly one zero-poison continuation metric table. "
            f"Candidates: {[str(path) for path, _ in metric_candidates]}"
        )

    if len(metric_frame) != 4:
        raise RuntimeError(
            f"Zero-poison reference has {len(metric_frame)} rounds, expected 4."
        )

    return zero_dir, metric_path, metric_frame, metadata


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    audit_dir = args.audit_output_dir.expanduser().resolve()
    tables_dir = audit_dir / "tables"
    figures_dir = audit_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    protocol_path = (
        output_root
        / "frozen_breadth_protocol"
        / "task41b3_frozen_breadth_protocol.json"
    )
    panel_path = (
        output_root
        / "frozen_breadth_protocol"
        / "task41b3_frozen_candidate_panel.csv"
    )
    protocol = read_json(protocol_path)
    panel = pd.read_csv(panel_path)

    if not bool(protocol.get("decision_frozen_before_new_panel_results", False)):
        raise RuntimeError("The Task 41B.3 protocol was not frozen before results.")

    criteria = protocol["primary_qualification_criteria"]
    mean_threshold = float(criteria["mean_macro_triggered_asr_minimum"])
    final_threshold = float(criteria["final_round_macro_triggered_asr_minimum"])
    breadth_threshold = int(
        criteria["minimum_source_classes_with_final_asr_at_least_0_20"]
    )
    clean_drop_threshold = float(
        criteria["maximum_absolute_clean_macro_f1_drop_vs_zero_poison"]
    )
    required_round_count = int(criteria["required_round_count"])

    zero_dir, zero_metric_path, zero_round, zero_meta = discover_zero_poison(
        output_root
    )
    zero_mean_f1 = float(zero_round["val_macro_f1"].mean())

    rows: list[dict[str, Any]] = []
    source_rows: list[pd.DataFrame] = []
    test_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = [
        {
            "role": "frozen_protocol",
            "path": str(protocol_path),
            "bytes": protocol_path.stat().st_size,
            "sha256": sha256_file(protocol_path),
        },
        {
            "role": "frozen_panel",
            "path": str(panel_path),
            "bytes": panel_path.stat().st_size,
            "sha256": sha256_file(panel_path),
        },
        {
            "role": "zero_poison_metrics",
            "path": str(zero_metric_path),
            "bytes": zero_metric_path.stat().st_size,
            "sha256": sha256_file(zero_metric_path),
        },
    ]

    for panel_row in panel.itertuples(index=False):
        slot = str(panel_row.slot)
        candidate_id = str(panel_row.candidate_id)
        family = str(panel_row.family)
        intensity = str(panel_row.intensity)
        branch_dir = output_root / "runs" / slot / "seed_7" / "plain_fedavg"
        metadata_path = branch_dir / "task41b_backdoor_metadata.json"
        round_path = branch_dir / "tables" / "continuation_round_metrics.csv"
        source_path = branch_dir / "tables" / "triggered_validation_asr_long.csv"

        for path in (metadata_path, round_path, source_path):
            if not path.exists():
                raise FileNotFoundError(path)

        metadata = read_json(metadata_path)
        if metadata.get("mode") != "strong_attack":
            raise RuntimeError(f"{slot} is not strong_attack.")
        if metadata.get("replacement_policy") != "plain_fedavg":
            raise RuntimeError(f"{slot} is not plain_fedavg.")
        if metadata.get("trigger_candidate_id") != candidate_id:
            raise RuntimeError(f"{slot} candidate metadata mismatch.")
        if int(metadata.get("model_seed", -1)) != 7:
            raise RuntimeError(f"{slot} model seed mismatch.")
        if int(metadata.get("attack_seed", -1)) != 7:
            raise RuntimeError(f"{slot} attack seed mismatch.")
        if abs(float(metadata.get("poison_fraction", -1.0)) - 0.01) > 1e-12:
            raise RuntimeError(f"{slot} poison fraction mismatch.")

        for field in FORBIDDEN_TEST_FLAGS:
            value = bool(metadata.get(field, False))
            test_rows.append(
                {
                    "slot": slot,
                    "candidate_id": candidate_id,
                    "field": field,
                    "value": value,
                }
            )
            if value:
                raise RuntimeError(f"{slot} reports reserved test access: {field}")

        round_frame = pd.read_csv(round_path).sort_values("monitoring_round")
        source_frame = pd.read_csv(source_path).sort_values(
            ["monitoring_round", "source_class_id"]
        )

        required_round_columns = {
            "monitoring_round",
            "val_macro_f1",
            "triggered_asr_macro_source",
        }
        required_source_columns = {
            "monitoring_round",
            "source_class_id",
            "source_class_name",
            "triggered_asr",
        }
        if not required_round_columns.issubset(round_frame.columns):
            raise RuntimeError(f"{slot} round table lacks required columns.")
        if not required_source_columns.issubset(source_frame.columns):
            raise RuntimeError(f"{slot} source table lacks required columns.")

        round_count = int(len(round_frame))
        final_round_number = int(round_frame["monitoring_round"].max())
        final_source = source_frame[
            source_frame["monitoring_round"] == final_round_number
        ].copy()
        if final_source["source_class_id"].nunique() != 7:
            raise RuntimeError(
                f"{slot} final round has "
                f"{final_source['source_class_id'].nunique()} source classes, expected 7."
            )

        mean_asr = float(round_frame["triggered_asr_macro_source"].mean())
        final_asr = float(
            round_frame.iloc[-1]["triggered_asr_macro_source"]
        )
        mean_clean_f1 = float(round_frame["val_macro_f1"].mean())
        clean_f1_drop = max(0.0, zero_mean_f1 - mean_clean_f1)
        source_count_final_ge_020 = int(
            (final_source["triggered_asr"] >= 0.20).sum()
        )

        pass_mean = mean_asr >= mean_threshold
        pass_final = final_asr >= final_threshold
        pass_breadth = source_count_final_ge_020 >= breadth_threshold
        pass_clean = clean_f1_drop <= clean_drop_threshold
        pass_rounds = round_count == required_round_count
        primary_pass = bool(
            pass_mean and pass_final and pass_breadth and pass_clean and pass_rounds
        )

        rows.append(
            {
                "slot": slot,
                "candidate_id": candidate_id,
                "family": family,
                "intensity": intensity,
                "round_count": round_count,
                "zero_poison_mean_clean_macro_f1": zero_mean_f1,
                "mean_clean_macro_f1": mean_clean_f1,
                "clean_macro_f1_drop_vs_zero_poison": clean_f1_drop,
                "mean_macro_triggered_asr": mean_asr,
                "final_round_macro_triggered_asr": final_asr,
                "final_source_classes_asr_ge_0_20": source_count_final_ge_020,
                "passes_mean_asr": pass_mean,
                "passes_final_asr": pass_final,
                "passes_source_breadth": pass_breadth,
                "passes_clean_utility": pass_clean,
                "passes_round_count": pass_rounds,
                "passes_primary_qualification": primary_pass,
                "test_sets_accessed": False,
            }
        )

        annotated_source = source_frame.copy()
        annotated_source.insert(0, "slot", slot)
        annotated_source.insert(1, "candidate_id", candidate_id)
        annotated_source.insert(2, "family", family)
        annotated_source.insert(3, "intensity", intensity)
        source_rows.append(annotated_source)

        for role, path in (
            ("metadata", metadata_path),
            ("round_metrics", round_path),
            ("per_source_asr", source_path),
        ):
            manifest_rows.append(
                {
                    "role": f"{slot}_{role}",
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    summary = pd.DataFrame(rows)

    summary["selection_label"] = np.where(
        summary["passes_primary_qualification"],
        "PRIMARY_QUALIFIER",
        "NOT_SELECTED",
    )
    summary["proceed_to_paired_defense"] = summary[
        "passes_primary_qualification"
    ].astype(bool)

    # Apply the protocol-mandated intensity breadth rule without changing
    # primary qualification labels.
    for intensity in ("exact_template", "low_intensity"):
        intensity_mask = summary["intensity"] == intensity
        if not bool(
            (
                intensity_mask
                & summary["passes_primary_qualification"]
            ).any()
        ):
            candidates = summary[
                intensity_mask & ~summary["passes_primary_qualification"]
            ].copy()
            if candidates.empty:
                raise RuntimeError(
                    f"No candidate is available for required {intensity} breadth."
                )
            candidates = candidates.sort_values(
                [
                    "mean_macro_triggered_asr",
                    "final_round_macro_triggered_asr",
                    "final_source_classes_asr_ge_0_20",
                ],
                ascending=[False, False, False],
            )
            selected_index = candidates.index[0]
            summary.loc[selected_index, "selection_label"] = (
                "BREADTH_NEGATIVE_CONTROL"
            )
            summary.loc[selected_index, "proceed_to_paired_defense"] = True

    summary["selection_rank"] = (
        summary.sort_values(
            [
                "proceed_to_paired_defense",
                "passes_primary_qualification",
                "mean_macro_triggered_asr",
                "final_round_macro_triggered_asr",
            ],
            ascending=[False, False, False, False],
        )
        .reset_index()
        .reset_index()
        .set_index("index")["level_0"]
        + 1
    )

    summary = summary.sort_values("selection_rank").reset_index(drop=True)
    all_source = pd.concat(source_rows, ignore_index=True)

    summary.to_csv(
        tables_dir / "task41b3_plain_breadth_qualification.csv", index=False
    )
    all_source.to_csv(
        tables_dir / "task41b3_plain_breadth_per_source_asr_long.csv",
        index=False,
    )
    pd.DataFrame(test_rows).to_csv(
        tables_dir / "task41b3_test_access_audit.csv", index=False
    )
    pd.DataFrame(manifest_rows).to_csv(
        tables_dir / "task41b3_input_manifest_sha256.csv", index=False
    )

    decision = {
        "experiment_version": "4.11B.4",
        "stage": "task41b_frozen_plain_breadth_qualification",
        "status": "development_selection_not_final_paper_result",
        "protocol_path": str(protocol_path),
        "zero_poison_dir": str(zero_dir),
        "zero_poison_mean_clean_macro_f1": zero_mean_f1,
        "primary_qualifier_count": int(
            summary["passes_primary_qualification"].sum()
        ),
        "paired_defense_selection_count": int(
            summary["proceed_to_paired_defense"].sum()
        ),
        "selected_slots": summary.loc[
            summary["proceed_to_paired_defense"], "slot"
        ].tolist(),
        "selected_candidates": summary.loc[
            summary["proceed_to_paired_defense"], "candidate_id"
        ].tolist(),
        "multiseed_selection_allowed_now": False,
        "next_stage": (
            "Run exact matched trusted-reconstruction branches only for the "
            "primary qualifiers and protocol-mandated breadth controls, then "
            "perform paired defense audits before multiseed selection."
        ),
        "attack_specific_retuning": False,
        "test_sets_accessed": False,
    }
    (audit_dir / "task41b3_plain_breadth_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    plot_frame = summary.sort_values(
        "mean_macro_triggered_asr", ascending=False
    )
    x = np.arange(len(plot_frame))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(
        x - width / 2,
        plot_frame["mean_macro_triggered_asr"],
        width,
        label="Mean ASR",
    )
    ax.bar(
        x + width / 2,
        plot_frame["final_round_macro_triggered_asr"],
        width,
        label="Final-round ASR",
    )
    ax.axhline(mean_threshold, linestyle="--", label="Mean threshold")
    ax.axhline(final_threshold, linestyle=":", label="Final threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_frame["slot"], rotation=30, ha="right")
    ax.set_ylabel("Macro triggered ASR")
    ax.set_ylim(0, 1)
    ax.set_title("Task 41B.3 frozen plain-trigger qualification")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "task41b3_plain_mean_and_final_asr")

    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.bar(
        plot_frame["slot"],
        plot_frame["final_source_classes_asr_ge_0_20"],
    )
    ax.axhline(
        breadth_threshold,
        linestyle="--",
        label="Frozen source-breadth threshold",
    )
    ax.set_ylabel("Final-round source classes with ASR ≥ 0.20")
    ax.set_ylim(0, 7.5)
    ax.set_title("Task 41B.3 source-class attack breadth")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "task41b3_final_source_breadth")

    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.bar(
        plot_frame["slot"],
        plot_frame["clean_macro_f1_drop_vs_zero_poison"],
    )
    ax.axhline(
        clean_drop_threshold,
        linestyle="--",
        label="Frozen maximum clean-F1 drop",
    )
    ax.set_ylabel("Clean macro-F1 drop")
    ax.set_title("Task 41B.3 clean-utility stealth")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "task41b3_clean_macro_f1_drop")

    print("===== TASK 41B.4 FROZEN PLAIN-BREADTH QUALIFICATION =====")
    print(
        summary[
            [
                "slot",
                "mean_macro_triggered_asr",
                "final_round_macro_triggered_asr",
                "final_source_classes_asr_ge_0_20",
                "clean_macro_f1_drop_vs_zero_poison",
                "passes_primary_qualification",
                "selection_label",
                "proceed_to_paired_defense",
            ]
        ].to_string(index=False)
    )
    print()
    print("Primary qualifiers:", decision["primary_qualifier_count"])
    print("Selected for matched paired defense:", decision["selected_slots"])
    print("Multiseed selection allowed now: False")
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("TEST SETS ACCESSED: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Task 41B.2 paired smoke audit.

Compares the exact Task 41B plain FedAvg branch with its matched frozen
trusted-reconstruction branch. The script is read-only with respect to input
runs and writes research-ready CSV plus PNG/PDF figures.

No test arrays are loaded.
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


REQUIRED_METADATA_MATCH = (
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
        description="Audit a matched Task 41B plain/defended smoke pair."
    )
    parser.add_argument("--plain-dir", type=Path, required=True)
    parser.add_argument("--defended-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def safe_fraction_removed(plain: pd.Series, defended: pd.Series) -> pd.Series:
    plain_values = plain.astype(float).to_numpy()
    defended_values = defended.astype(float).to_numpy()
    result = np.full_like(plain_values, np.nan, dtype=float)
    valid = np.abs(plain_values) > 1e-12
    result[valid] = (plain_values[valid] - defended_values[valid]) / plain_values[valid]
    return pd.Series(result, index=plain.index)


def main() -> int:
    args = parse_args()
    plain_dir = args.plain_dir.expanduser().resolve()
    defended_dir = args.defended_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plain_metadata_path = plain_dir / "task41b_backdoor_metadata.json"
    defended_metadata_path = defended_dir / "task41b_backdoor_metadata.json"
    plain_round_path = plain_dir / "tables" / "continuation_round_metrics.csv"
    defended_round_path = defended_dir / "tables" / "continuation_round_metrics.csv"
    plain_source_path = plain_dir / "tables" / "triggered_validation_asr_long.csv"
    defended_source_path = defended_dir / "tables" / "triggered_validation_asr_long.csv"

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

    if plain_meta.get("mode") != "strong_attack":
        raise RuntimeError("Plain branch is not strong_attack.")
    if plain_meta.get("replacement_policy") != "plain_fedavg":
        raise RuntimeError("Plain branch is not plain_fedavg.")
    if defended_meta.get("mode") != "strong_attack":
        raise RuntimeError("Defended branch is not strong_attack.")
    if defended_meta.get("replacement_policy") != "trusted_reconstruction":
        raise RuntimeError("Defended branch is not trusted_reconstruction.")

    mismatch_rows: list[dict[str, Any]] = []
    for field in REQUIRED_METADATA_MATCH:
        plain_value = plain_meta.get(field)
        defended_value = defended_meta.get(field)
        exact = plain_value == defended_value
        mismatch_rows.append(
            {
                "field": field,
                "plain_value": json.dumps(plain_value, sort_keys=True),
                "defended_value": json.dumps(defended_value, sort_keys=True),
                "exact_match": bool(exact),
            }
        )
        if not exact:
            raise RuntimeError(f"Matched-pair metadata mismatch: {field}")

    test_access_rows: list[dict[str, Any]] = []
    for branch_name, metadata in (
        ("plain_fedavg", plain_meta),
        ("trusted_reconstruction", defended_meta),
    ):
        for field in FORBIDDEN_TEST_FLAGS:
            value = bool(metadata.get(field, False))
            test_access_rows.append(
                {"branch": branch_name, "field": field, "value": value}
            )
            if value:
                raise RuntimeError(
                    f"Reserved test access reported by {branch_name}: {field}"
                )

    plain_round = pd.read_csv(plain_round_path)
    defended_round = pd.read_csv(defended_round_path)

    key_columns = ["monitoring_round", "global_round"]
    for name, frame in (("plain", plain_round), ("defended", defended_round)):
        missing = [column for column in key_columns if column not in frame.columns]
        if missing:
            raise RuntimeError(f"{name} round table missing columns: {missing}")
        if frame.duplicated(key_columns).any():
            raise RuntimeError(f"{name} round table has duplicate round keys.")

    paired = plain_round.merge(
        defended_round,
        on=key_columns,
        how="outer",
        suffixes=("_plain", "_defended"),
        validate="one_to_one",
        indicator=True,
    )
    if not (paired["_merge"] == "both").all():
        raise RuntimeError("Plain and defended round sets do not match.")
    paired = paired.drop(columns=["_merge"]).sort_values(key_columns).reset_index(drop=True)

    required_round_columns = (
        "val_macro_f1_plain",
        "val_macro_f1_defended",
        "triggered_asr_all_nonbenign_plain",
        "triggered_asr_all_nonbenign_defended",
        "triggered_asr_macro_source_plain",
        "triggered_asr_macro_source_defended",
        "triggered_asr_worst_source_plain",
        "triggered_asr_worst_source_defended",
        "malicious_recall_defended",
        "benign_false_positive_rate_defended",
        "replaced_clients_defended",
    )
    missing = [column for column in required_round_columns if column not in paired.columns]
    if missing:
        raise RuntimeError(f"Paired round table missing columns: {missing}")

    paired["macro_asr_delta_defended_minus_plain"] = (
        paired["triggered_asr_macro_source_defended"]
        - paired["triggered_asr_macro_source_plain"]
    )
    paired["macro_asr_absolute_reduction"] = (
        paired["triggered_asr_macro_source_plain"]
        - paired["triggered_asr_macro_source_defended"]
    )
    paired["macro_asr_removed_fraction"] = safe_fraction_removed(
        paired["triggered_asr_macro_source_plain"],
        paired["triggered_asr_macro_source_defended"],
    )
    paired["clean_macro_f1_delta_defended_minus_plain"] = (
        paired["val_macro_f1_defended"] - paired["val_macro_f1_plain"]
    )
    paired["defense_improved_macro_asr"] = (
        paired["triggered_asr_macro_source_defended"]
        < paired["triggered_asr_macro_source_plain"]
    )

    plain_source = pd.read_csv(plain_source_path)
    defended_source = pd.read_csv(defended_source_path)
    source_keys = [
        "monitoring_round",
        "global_round",
        "source_class_id",
        "source_class_name",
    ]
    source_paired = plain_source.merge(
        defended_source,
        on=source_keys,
        how="outer",
        suffixes=("_plain", "_defended"),
        validate="one_to_one",
        indicator=True,
    )
    if not (source_paired["_merge"] == "both").all():
        raise RuntimeError("Plain and defended per-source ASR rows do not match.")
    source_paired = (
        source_paired.drop(columns=["_merge"])
        .sort_values(["source_class_id", "monitoring_round"])
        .reset_index(drop=True)
    )
    source_paired["asr_delta_defended_minus_plain"] = (
        source_paired["triggered_asr_defended"]
        - source_paired["triggered_asr_plain"]
    )
    source_paired["asr_absolute_reduction"] = (
        source_paired["triggered_asr_plain"]
        - source_paired["triggered_asr_defended"]
    )
    source_paired["asr_removed_fraction"] = safe_fraction_removed(
        source_paired["triggered_asr_plain"],
        source_paired["triggered_asr_defended"],
    )

    per_source_summary = (
        source_paired.groupby(
            ["source_class_id", "source_class_name"], as_index=False
        )
        .agg(
            mean_plain_asr=("triggered_asr_plain", "mean"),
            mean_defended_asr=("triggered_asr_defended", "mean"),
            mean_absolute_reduction=("asr_absolute_reduction", "mean"),
            mean_removed_fraction=("asr_removed_fraction", "mean"),
            final_plain_asr=("triggered_asr_plain", "last"),
            final_defended_asr=("triggered_asr_defended", "last"),
            improved_rounds=("asr_absolute_reduction", lambda values: int((values > 0).sum())),
            total_rounds=("monitoring_round", "count"),
        )
        .sort_values("source_class_id")
        .reset_index(drop=True)
    )

    final_round = paired.iloc[-1]
    source_round_stats = (
        source_paired.groupby("monitoring_round")
        .agg(
            minimum_source_asr_plain=("triggered_asr_plain", "min"),
            maximum_source_asr_plain=("triggered_asr_plain", "max"),
            minimum_source_asr_defended=("triggered_asr_defended", "min"),
            maximum_source_asr_defended=("triggered_asr_defended", "max"),
        )
        .reset_index()
    )
    paired = paired.merge(source_round_stats, on="monitoring_round", how="left")

    mean_plain_asr = float(paired["triggered_asr_macro_source_plain"].mean())
    mean_defended_asr = float(paired["triggered_asr_macro_source_defended"].mean())
    absolute_reduction = mean_plain_asr - mean_defended_asr
    removed_fraction = (
        absolute_reduction / mean_plain_asr if abs(mean_plain_asr) > 1e-12 else np.nan
    )

    summary = pd.DataFrame(
        [
            {
                "experiment_version": "4.11B.2",
                "stage": "task41b_exact_paired_backdoor_smoke_audit",
                "status": "development_smoke_not_final_paper_result",
                "model_seed": int(defended_meta["model_seed"]),
                "attack_seed": int(defended_meta["attack_seed"]),
                "trigger_candidate_id": defended_meta["trigger_candidate_id"],
                "poison_fraction": float(defended_meta["poison_fraction"]),
                "round_count": int(len(paired)),
                "mean_plain_macro_triggered_asr": mean_plain_asr,
                "mean_defended_macro_triggered_asr": mean_defended_asr,
                "mean_macro_asr_absolute_reduction": absolute_reduction,
                "mean_macro_asr_removed_fraction": removed_fraction,
                "macro_asr_improved_round_count": int(
                    paired["defense_improved_macro_asr"].sum()
                ),
                "final_plain_macro_triggered_asr": float(
                    final_round["triggered_asr_macro_source_plain"]
                ),
                "final_defended_macro_triggered_asr": float(
                    final_round["triggered_asr_macro_source_defended"]
                ),
                "final_macro_asr_absolute_reduction": float(
                    final_round["macro_asr_absolute_reduction"]
                ),
                "mean_plain_clean_macro_f1": float(
                    paired["val_macro_f1_plain"].mean()
                ),
                "mean_defended_clean_macro_f1": float(
                    paired["val_macro_f1_defended"].mean()
                ),
                "mean_clean_macro_f1_delta": float(
                    paired["clean_macro_f1_delta_defended_minus_plain"].mean()
                ),
                "mean_malicious_recall": float(
                    paired["malicious_recall_defended"].mean()
                ),
                "minimum_malicious_recall": float(
                    paired["malicious_recall_defended"].min()
                ),
                "mean_benign_fpr": float(
                    paired["benign_false_positive_rate_defended"].mean()
                ),
                "maximum_benign_fpr": float(
                    paired["benign_false_positive_rate_defended"].max()
                ),
                "total_replaced_client_rounds": int(
                    paired["replaced_clients_defended"].sum()
                ),
                "expected_malicious_client_rounds": int(
                    len(defended_meta["malicious_clients"]) * len(paired)
                ),
                "poison_plan_hash_exact_match": bool(
                    plain_meta["poison_index_hash_sha256"]
                    == defended_meta["poison_index_hash_sha256"]
                ),
                "trigger_spec_hash_exact_match": bool(
                    plain_meta["trigger_spec_sha256"]
                    == defended_meta["trigger_spec_sha256"]
                ),
                "partition_hash_exact_match": bool(
                    plain_meta["partition_hash_sha256"]
                    == defended_meta["partition_hash_sha256"]
                ),
                "test_sets_accessed": False,
            }
        ]
    )

    paired.to_csv(tables_dir / "task41b_paired_round_comparison.csv", index=False)
    source_paired.to_csv(
        tables_dir / "task41b_paired_per_source_asr_long.csv", index=False
    )
    per_source_summary.to_csv(
        tables_dir / "task41b_per_source_summary.csv", index=False
    )
    summary.to_csv(tables_dir / "task41b_paired_smoke_summary.csv", index=False)
    pd.DataFrame(mismatch_rows).to_csv(
        tables_dir / "task41b_metadata_equivalence.csv", index=False
    )
    pd.DataFrame(test_access_rows).to_csv(
        tables_dir / "task41b_test_access_audit.csv", index=False
    )

    manifest_rows = []
    for role, path in (
        ("plain_metadata", plain_metadata_path),
        ("defended_metadata", defended_metadata_path),
        ("plain_round_metrics", plain_round_path),
        ("defended_round_metrics", defended_round_path),
        ("plain_per_source_asr", plain_source_path),
        ("defended_per_source_asr", defended_source_path),
    ):
        manifest_rows.append(
            {
                "role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        tables_dir / "task41b_paired_input_manifest_sha256.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(
        paired["monitoring_round"],
        paired["triggered_asr_macro_source_plain"],
        marker="o",
        label="Plain FedAvg",
    )
    ax.plot(
        paired["monitoring_round"],
        paired["triggered_asr_macro_source_defended"],
        marker="s",
        label="Trusted reconstruction",
    )
    ax.set_xlabel("Post-warmup monitoring round")
    ax.set_ylabel("Macro per-source triggered ASR")
    ax.set_ylim(0, 1)
    ax.set_title("Task 41B paired backdoor smoke")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "task41b_plain_vs_defended_macro_asr")

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(len(per_source_summary))
    width = 0.38
    ax.bar(
        x - width / 2,
        per_source_summary["mean_plain_asr"],
        width,
        label="Plain FedAvg",
    )
    ax.bar(
        x + width / 2,
        per_source_summary["mean_defended_asr"],
        width,
        label="Trusted reconstruction",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        per_source_summary["source_class_name"], rotation=30, ha="right"
    )
    ax.set_ylabel("Mean triggered ASR")
    ax.set_ylim(0, 1)
    ax.set_title("Task 41B mean triggered ASR by source class")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "task41b_per_source_plain_vs_defended_asr")

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(
        paired["monitoring_round"],
        paired["malicious_recall_defended"],
        marker="o",
        label="Malicious recall",
    )
    ax.plot(
        paired["monitoring_round"],
        paired["benign_false_positive_rate_defended"],
        marker="s",
        label="Benign FPR",
    )
    ax.set_xlabel("Post-warmup monitoring round")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Task 41B frozen detector behavior")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "task41b_detector_behavior")

    metadata = {
        "experiment_version": "4.11B.2",
        "stage": "task41b_exact_paired_backdoor_smoke_audit",
        "status": "development_smoke_not_final_paper_result",
        "plain_dir": str(plain_dir),
        "defended_dir": str(defended_dir),
        "output_dir": str(output_dir),
        "metadata_fields_exactly_matched": list(REQUIRED_METADATA_MATCH),
        "test_access_flags_audited": list(FORBIDDEN_TEST_FLAGS),
        "test_sets_accessed": False,
        "summary": summary.iloc[0].to_dict(),
    }
    with (output_dir / "task41b_paired_smoke_audit_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print("===== TASK 41B.2 PAIRED SMOKE AUDIT =====")
    print(summary.to_string(index=False))
    print()
    print("Per-source summary:")
    print(per_source_summary.to_string(index=False))
    print()
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("TEST SETS ACCESSED: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

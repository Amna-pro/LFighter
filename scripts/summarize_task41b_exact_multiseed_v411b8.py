#!/usr/bin/env python3
"""
Task 41B.8 exact paired multiseed summary and development decision.

Inputs:
- Task 41B.7 frozen multiseed protocol and completion artifacts
- Eight exact plain-versus-trusted-reconstruction pairs
- Two frozen primary trigger candidates
- Four frozen seeds: 7, 99, 123, 2026

Outputs:
- Pair-level, trigger-level, seed-level, round-level, and per-source CSV tables
- Exact paired sign-flip tests and deterministic paired bootstrap intervals
- Publication-quality PNG and PDF figures
- A frozen development decision JSON

The script never loads training, validation, natural-test, or diagnostic-test
arrays. It reads only previously generated metadata and result tables.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = [7, 99, 123, 2026]
SLOTS = ["flow_iat_exact", "active_idle_exact"]
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
ROUND_COLUMNS = (
    "monitoring_round",
    "val_macro_f1",
    "triggered_asr_macro_source",
    "triggered_asr_worst_source",
    "malicious_recall",
    "benign_false_positive_rate",
    "replaced_clients",
)
SOURCE_COLUMNS = (
    "monitoring_round",
    "source_class_id",
    "source_class_name",
    "triggered_asr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Task 41B exact paired multiseed results."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-output-dir", type=Path, required=True)
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


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def exact_pair_match(
    plain_metadata: dict[str, Any],
    defended_metadata: dict[str, Any],
) -> tuple[bool, list[str]]:
    mismatches = [
        field
        for field in MATCH_FIELDS
        if plain_metadata.get(field) != defended_metadata.get(field)
    ]
    return not mismatches, mismatches


def test_access_clear(metadata: dict[str, Any]) -> bool:
    return not any(bool(metadata.get(field, False)) for field in FORBIDDEN_TEST_FLAGS)


def exact_sign_flip_pvalue(differences: Iterable[float]) -> float:
    values = np.asarray(list(differences), dtype=float)
    if values.ndim != 1 or values.size == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=values.size):
        permuted.append(abs(float((values * np.asarray(signs)).mean())))
    permuted_array = np.asarray(permuted)
    return float(np.mean(permuted_array >= observed - 1e-15))


def paired_bootstrap_ci(
    differences: Iterable[float],
    *,
    seed: int,
    replicates: int = 20000,
) -> tuple[float, float]:
    values = np.asarray(list(differences), dtype=float)
    if values.ndim != 1 or values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)



def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )


def validate_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label} lacks required columns: {sorted(missing)}")


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    summary_dir = args.summary_output_dir.expanduser().resolve()
    tables_dir = summary_dir / "tables"
    figures_dir = summary_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    protocol_dir = output_root / "multiseed_protocol"
    protocol_path = protocol_dir / "task41b7_exact_multiseed_protocol.json"
    completion_csv_path = protocol_dir / "task41b7_exact_multiseed_completion.csv"
    completion_json_path = protocol_dir / "task41b7_exact_multiseed_completion.json"

    protocol = load_json(protocol_path)
    completion_json = load_json(completion_json_path)
    completion = pd.read_csv(completion_csv_path)

    if protocol.get("selected_slots") != SLOTS:
        raise RuntimeError("Task 41B.7 selected-slot panel mismatch.")
    if protocol.get("seeds") != SEEDS:
        raise RuntimeError("Task 41B.7 seed panel mismatch.")
    if bool(protocol.get("attack_specific_retuning", True)):
        raise RuntimeError("Task 41B.7 reports attack-specific retuning.")
    if bool(protocol.get("threshold_retuning", True)):
        raise RuntimeError("Task 41B.7 reports threshold retuning.")
    if bool(protocol.get("test_sets_accessed", True)):
        raise RuntimeError("Task 41B.7 reports reserved test access.")
    if not bool(completion_json.get("all_exact_pairs_complete", False)):
        raise RuntimeError("Task 41B.7 did not complete every exact pair.")
    if int(completion_json.get("completed_exact_pair_count", -1)) != 8:
        raise RuntimeError("Task 41B.7 completion count is not 8.")
    if len(completion) != 8:
        raise RuntimeError(f"Expected 8 completion rows, found {len(completion)}.")

    required_completion_flags = (
        "complete_exact_pair",
        "poison_plan_hash_exact_match",
        "trigger_spec_hash_exact_match",
        "partition_hash_exact_match",
        "warmup_profile_hash_exact_match",
    )
    for field in required_completion_flags:
        if field not in completion.columns:
            raise RuntimeError(f"Completion table lacks {field}.")
        if not bool_series(completion[field]).all():
            raise RuntimeError(f"Completion integrity check failed: {field}.")
    if bool_series(completion["plain_test_sets_accessed"]).any():
        raise RuntimeError("A plain branch reports reserved test access.")
    if bool_series(completion["defended_test_sets_accessed"]).any():
        raise RuntimeError("A defended branch reports reserved test access.")

    pair_rows: list[dict[str, Any]] = []
    round_frames: list[pd.DataFrame] = []
    source_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    test_access_rows: list[dict[str, Any]] = []

    for role, path in (
        ("task41b7_protocol", protocol_path),
        ("task41b7_completion_csv", completion_csv_path),
        ("task41b7_completion_json", completion_json_path),
    ):
        manifest_rows.append(
            {
                "role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    for slot in SLOTS:
        for seed in SEEDS:
            plain_dir = output_root / "runs" / slot / f"seed_{seed}" / "plain_fedavg"
            defended_dir = (
                output_root
                / "runs"
                / slot
                / f"seed_{seed}"
                / "trusted_reconstruction"
            )
            plain_metadata_path = plain_dir / "task41b_backdoor_metadata.json"
            defended_metadata_path = defended_dir / "task41b_backdoor_metadata.json"
            plain_round_path = plain_dir / "tables" / "continuation_round_metrics.csv"
            defended_round_path = defended_dir / "tables" / "continuation_round_metrics.csv"
            plain_source_path = plain_dir / "tables" / "triggered_validation_asr_long.csv"
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
                raise RuntimeError(f"{slot}, seed {seed}, plain policy mismatch.")
            if defended_meta.get("replacement_policy") != "trusted_reconstruction":
                raise RuntimeError(f"{slot}, seed {seed}, defended policy mismatch.")
            if int(plain_meta.get("model_seed", -1)) != seed:
                raise RuntimeError(f"{slot}, seed {seed}, plain model seed mismatch.")
            if int(defended_meta.get("model_seed", -1)) != seed:
                raise RuntimeError(f"{slot}, seed {seed}, defended model seed mismatch.")
            if bool(plain_meta.get("attack_specific_retuning", True)):
                raise RuntimeError(f"{slot}, seed {seed}, plain retuning reported.")
            if bool(defended_meta.get("attack_specific_retuning", True)):
                raise RuntimeError(f"{slot}, seed {seed}, defended retuning reported.")
            if not test_access_clear(plain_meta):
                raise RuntimeError(f"{slot}, seed {seed}, plain test access reported.")
            if not test_access_clear(defended_meta):
                raise RuntimeError(f"{slot}, seed {seed}, defended test access reported.")

            exact, mismatches = exact_pair_match(plain_meta, defended_meta)
            if not exact:
                raise RuntimeError(
                    f"{slot}, seed {seed}, exact-pair mismatch: {','.join(mismatches)}"
                )

            for branch_name, metadata in (
                ("plain_fedavg", plain_meta),
                ("trusted_reconstruction", defended_meta),
            ):
                for field in FORBIDDEN_TEST_FLAGS:
                    value = bool(metadata.get(field, False))
                    test_access_rows.append(
                        {
                            "slot": slot,
                            "seed": seed,
                            "branch": branch_name,
                            "field": field,
                            "value": value,
                        }
                    )

            plain_round = pd.read_csv(plain_round_path).sort_values(
                "monitoring_round"
            ).reset_index(drop=True)
            defended_round = pd.read_csv(defended_round_path).sort_values(
                "monitoring_round"
            ).reset_index(drop=True)
            validate_columns(plain_round, ROUND_COLUMNS, f"{slot} seed {seed} plain")
            validate_columns(
                defended_round, ROUND_COLUMNS, f"{slot} seed {seed} defended"
            )
            if len(plain_round) != 4 or len(defended_round) != 4:
                raise RuntimeError(
                    f"{slot}, seed {seed}, both branches must contain four rounds."
                )
            if not np.array_equal(
                plain_round["monitoring_round"].to_numpy(),
                defended_round["monitoring_round"].to_numpy(),
            ):
                raise RuntimeError(f"{slot}, seed {seed}, round alignment mismatch.")

            paired_round = pd.DataFrame(
                {
                    "slot": slot,
                    "seed": seed,
                    "monitoring_round": plain_round["monitoring_round"].astype(int),
                    "plain_val_macro_f1": plain_round["val_macro_f1"],
                    "defended_val_macro_f1": defended_round["val_macro_f1"],
                    "clean_macro_f1_delta_defended_minus_plain": (
                        defended_round["val_macro_f1"] - plain_round["val_macro_f1"]
                    ),
                    "plain_macro_triggered_asr": plain_round[
                        "triggered_asr_macro_source"
                    ],
                    "defended_macro_triggered_asr": defended_round[
                        "triggered_asr_macro_source"
                    ],
                    "macro_asr_reduction_plain_minus_defended": (
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
                    "plain_benign_fpr": plain_round["benign_false_positive_rate"],
                    "defended_benign_fpr": defended_round[
                        "benign_false_positive_rate"
                    ],
                    "plain_replaced_clients": plain_round["replaced_clients"],
                    "defended_replaced_clients": defended_round["replaced_clients"],
                }
            )
            paired_round["macro_asr_round_improved"] = (
                paired_round["macro_asr_reduction_plain_minus_defended"] > 0.0
            )
            round_frames.append(paired_round)

            plain_source = pd.read_csv(plain_source_path)
            defended_source = pd.read_csv(defended_source_path)
            validate_columns(
                plain_source, SOURCE_COLUMNS, f"{slot} seed {seed} plain source"
            )
            validate_columns(
                defended_source, SOURCE_COLUMNS, f"{slot} seed {seed} defended source"
            )
            final_round = int(plain_round["monitoring_round"].max())
            plain_final_source = plain_source[
                plain_source["monitoring_round"] == final_round
            ][["source_class_id", "source_class_name", "triggered_asr"]].rename(
                columns={"triggered_asr": "plain_final_triggered_asr"}
            )
            defended_final_source = defended_source[
                defended_source["monitoring_round"] == final_round
            ][["source_class_id", "source_class_name", "triggered_asr"]].rename(
                columns={"triggered_asr": "defended_final_triggered_asr"}
            )
            final_source = plain_final_source.merge(
                defended_final_source,
                on=["source_class_id", "source_class_name"],
                how="inner",
                validate="one_to_one",
            )
            if len(final_source) != 7:
                raise RuntimeError(
                    f"{slot}, seed {seed}, expected 7 non-benign source classes."
                )
            final_source.insert(0, "slot", slot)
            final_source.insert(1, "seed", seed)
            final_source["final_asr_reduction_plain_minus_defended"] = (
                final_source["plain_final_triggered_asr"]
                - final_source["defended_final_triggered_asr"]
            )
            final_source["final_source_improved"] = (
                final_source["final_asr_reduction_plain_minus_defended"] > 0.0
            )
            source_frames.append(final_source)

            plain_mean_asr = float(plain_round["triggered_asr_macro_source"].mean())
            defended_mean_asr = float(
                defended_round["triggered_asr_macro_source"].mean()
            )
            plain_final_asr = float(
                plain_round.iloc[-1]["triggered_asr_macro_source"]
            )
            defended_final_asr = float(
                defended_round.iloc[-1]["triggered_asr_macro_source"]
            )
            pair_rows.append(
                {
                    "slot": slot,
                    "candidate_id": plain_meta.get("trigger_candidate_id"),
                    "seed": seed,
                    "plain_mean_macro_triggered_asr": plain_mean_asr,
                    "defended_mean_macro_triggered_asr": defended_mean_asr,
                    "mean_macro_triggered_asr_reduction": (
                        plain_mean_asr - defended_mean_asr
                    ),
                    "relative_mean_asr_reduction_fraction": (
                        (plain_mean_asr - defended_mean_asr) / plain_mean_asr
                        if abs(plain_mean_asr) > 1e-12
                        else float("nan")
                    ),
                    "plain_final_round_macro_triggered_asr": plain_final_asr,
                    "defended_final_round_macro_triggered_asr": defended_final_asr,
                    "final_round_macro_triggered_asr_reduction": (
                        plain_final_asr - defended_final_asr
                    ),
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
                    "defended_mean_malicious_recall": float(
                        defended_round["malicious_recall"].mean()
                    ),
                    "defended_max_benign_fpr": float(
                        defended_round["benign_false_positive_rate"].max()
                    ),
                    "defended_total_replaced_client_rounds": int(
                        defended_round["replaced_clients"].sum()
                    ),
                    "improved_round_count": int(
                        (
                            plain_round["triggered_asr_macro_source"]
                            > defended_round["triggered_asr_macro_source"]
                        ).sum()
                    ),
                    "final_source_classes_improved": int(
                        final_source["final_source_improved"].sum()
                    ),
                    "exact_pair_metadata_match": True,
                    "attack_specific_retuning": False,
                    "test_sets_accessed": False,
                }
            )

            for branch_name, paths in (
                (
                    "plain",
                    (plain_metadata_path, plain_round_path, plain_source_path),
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
                            "role": f"{slot}_seed_{seed}_{branch_name}_{path.name}",
                            "path": str(path),
                            "bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                    )

    pairs = pd.DataFrame(pair_rows).sort_values(["slot", "seed"]).reset_index(drop=True)
    rounds = pd.concat(round_frames, ignore_index=True)
    final_sources = pd.concat(source_frames, ignore_index=True)

    if len(pairs) != 8:
        raise RuntimeError(f"Expected 8 exact pairs, found {len(pairs)}.")

    trigger_rows: list[dict[str, Any]] = []
    for slot_index, slot in enumerate(SLOTS):
        subset = pairs[pairs["slot"] == slot].sort_values("seed")
        differences = subset["mean_macro_triggered_asr_reduction"].to_numpy()
        ci_low, ci_high = paired_bootstrap_ci(
            differences, seed=411800 + slot_index
        )
        trigger_rows.append(
            {
                "slot": slot,
                "candidate_id": subset.iloc[0]["candidate_id"],
                "seed_count": len(subset),
                "plain_mean_macro_triggered_asr": float(
                    subset["plain_mean_macro_triggered_asr"].mean()
                ),
                "plain_sd_macro_triggered_asr": float(
                    subset["plain_mean_macro_triggered_asr"].std(ddof=1)
                ),
                "defended_mean_macro_triggered_asr": float(
                    subset["defended_mean_macro_triggered_asr"].mean()
                ),
                "defended_sd_macro_triggered_asr": float(
                    subset["defended_mean_macro_triggered_asr"].std(ddof=1)
                ),
                "mean_macro_triggered_asr_reduction": float(differences.mean()),
                "relative_reduction_from_group_means": float(
                    differences.mean()
                    / subset["plain_mean_macro_triggered_asr"].mean()
                ),
                "paired_bootstrap_95ci_low": ci_low,
                "paired_bootstrap_95ci_high": ci_high,
                "exact_sign_flip_two_sided_p": exact_sign_flip_pvalue(differences),
                "improved_seed_count": int((differences > 0.0).sum()),
                "worsened_seed_count": int((differences < 0.0).sum()),
                "unchanged_seed_count": int((differences == 0.0).sum()),
                "mean_clean_macro_f1_delta": float(
                    subset["clean_macro_f1_delta_defended_minus_plain"].mean()
                ),
                "minimum_clean_macro_f1_delta": float(
                    subset["clean_macro_f1_delta_defended_minus_plain"].min()
                ),
                "mean_defended_malicious_recall": float(
                    subset["defended_mean_malicious_recall"].mean()
                ),
                "maximum_defended_benign_fpr": float(
                    subset["defended_max_benign_fpr"].max()
                ),
                "total_replaced_client_rounds": int(
                    subset["defended_total_replaced_client_rounds"].sum()
                ),
                "plain_final_asr_ge_0_20_seed_count": int(
                    (subset["plain_final_round_macro_triggered_asr"] >= 0.20).sum()
                ),
                "plain_final_asr_ge_0_50_seed_count": int(
                    (subset["plain_final_round_macro_triggered_asr"] >= 0.50).sum()
                ),
            }
        )

    trigger_summary = pd.DataFrame(trigger_rows)

    seed_summary = (
        pairs.groupby("seed", as_index=False)
        .agg(
            plain_mean_macro_triggered_asr=(
                "plain_mean_macro_triggered_asr",
                "mean",
            ),
            defended_mean_macro_triggered_asr=(
                "defended_mean_macro_triggered_asr",
                "mean",
            ),
            mean_macro_triggered_asr_reduction=(
                "mean_macro_triggered_asr_reduction",
                "mean",
            ),
            mean_clean_macro_f1_delta=(
                "clean_macro_f1_delta_defended_minus_plain",
                "mean",
            ),
            mean_defended_malicious_recall=(
                "defended_mean_malicious_recall",
                "mean",
            ),
            total_replaced_client_rounds=(
                "defended_total_replaced_client_rounds",
                "sum",
            ),
        )
        .sort_values("seed")
        .reset_index(drop=True)
    )
    seed_summary["both_triggers_improved"] = seed_summary["seed"].map(
        pairs.groupby("seed")["mean_macro_triggered_asr_reduction"]
        .apply(lambda values: bool((values > 0.0).all()))
        .to_dict()
    )

    seed_level_differences = seed_summary[
        "mean_macro_triggered_asr_reduction"
    ].to_numpy()
    overall_ci_low, overall_ci_high = paired_bootstrap_ci(
        seed_level_differences, seed=411899
    )
    overall_sign_flip_p = exact_sign_flip_pvalue(seed_level_differences)

    flow_row = trigger_summary[trigger_summary["slot"] == "flow_iat_exact"].iloc[0]
    active_row = trigger_summary[
        trigger_summary["slot"] == "active_idle_exact"
    ].iloc[0]

    uniform_trigger_direction = bool(
        (trigger_summary["mean_macro_triggered_asr_reduction"] > 0.0).all()
    )
    majority_seed_support_each_trigger = bool(
        (trigger_summary["improved_seed_count"] >= 3).all()
    )
    both_triggers_improved_seed_count = int(
        seed_summary["both_triggers_improved"].sum()
    )
    maximum_benign_fpr = float(pairs["defended_max_benign_fpr"].max())
    mean_detector_recall = float(
        pairs["defended_mean_malicious_recall"].mean()
    )
    overall_mean_plain_asr = float(
        pairs["plain_mean_macro_triggered_asr"].mean()
    )
    overall_mean_defended_asr = float(
        pairs["defended_mean_macro_triggered_asr"].mean()
    )
    overall_mean_reduction = overall_mean_plain_asr - overall_mean_defended_asr
    overall_relative_reduction = (
        overall_mean_reduction / overall_mean_plain_asr
        if abs(overall_mean_plain_asr) > 1e-12
        else float("nan")
    )

    attack_materiality_supported = bool(
        (trigger_summary["plain_final_asr_ge_0_20_seed_count"] >= 2).all()
    )
    frozen_defense_consistent_support = bool(
        uniform_trigger_direction
        and majority_seed_support_each_trigger
        and both_triggers_improved_seed_count >= 3
    )

    if frozen_defense_consistent_support:
        defense_evidence = "CONSISTENT_DIRECTIONAL_SUPPORT"
    else:
        defense_evidence = "NO_CONSISTENT_MULTISEED_EFFICACY"

    if attack_materiality_supported:
        attack_evidence = "MATERIAL_BUT_SEED_HETEROGENEOUS_BACKDOOR_EFFECT"
    else:
        attack_evidence = "INSUFFICIENT_MULTISEED_ATTACK_MATERIALITY"

    task41b_status = (
        "COMPLETED_ATTACK_SUPPORTED_DEFENSE_NOT_CONSISTENTLY_EFFECTIVE"
        if attack_materiality_supported and not frozen_defense_consistent_support
        else "COMPLETED_REVIEW_REQUIRED"
    )

    decision = {
        "experiment_version": "4.11B.8",
        "stage": "task41b_exact_paired_multiseed_summary",
        "status": task41b_status,
        "exact_pair_count": 8,
        "trigger_count": 2,
        "seed_count": 4,
        "attack_evidence": attack_evidence,
        "frozen_defense_evidence": defense_evidence,
        "overall_plain_mean_macro_triggered_asr": overall_mean_plain_asr,
        "overall_defended_mean_macro_triggered_asr": overall_mean_defended_asr,
        "overall_mean_macro_triggered_asr_reduction": overall_mean_reduction,
        "overall_relative_reduction_fraction": overall_relative_reduction,
        "overall_seed_paired_bootstrap_95ci": [
            overall_ci_low,
            overall_ci_high,
        ],
        "overall_exact_sign_flip_two_sided_p": overall_sign_flip_p,
        "flow_iat_exact_mean_reduction": float(
            flow_row["mean_macro_triggered_asr_reduction"]
        ),
        "flow_iat_exact_improved_seed_count": int(
            flow_row["improved_seed_count"]
        ),
        "active_idle_exact_mean_reduction": float(
            active_row["mean_macro_triggered_asr_reduction"]
        ),
        "active_idle_exact_improved_seed_count": int(
            active_row["improved_seed_count"]
        ),
        "both_triggers_improved_seed_count": both_triggers_improved_seed_count,
        "mean_defended_malicious_recall": mean_detector_recall,
        "maximum_defended_benign_fpr": maximum_benign_fpr,
        "uniform_trigger_direction": uniform_trigger_direction,
        "majority_seed_support_each_trigger": majority_seed_support_each_trigger,
        "attack_specific_retuning": False,
        "threshold_retuning": False,
        "method_reopened": False,
        "reserved_test_accessed": False,
        "final_paper_claim_allowed": False,
        "scientific_interpretation": (
            "The frozen Task 40 detector and reconstruction pipeline does not "
            "provide consistent multiseed mitigation against the evaluated "
            "low-rate dirty-label backdoors. The result is retained as a "
            "negative or boundary finding. No post-hoc detector or threshold "
            "retuning is permitted within this frozen experiment."
        ),
        "next_stage": (
            "Freeze Task 41B development evidence, then design Task 41C as a "
            "separate preregistered adaptive-backdoor or backdoor-defense "
            "extension. Keep Task 41B unchanged and continue withholding "
            "reserved test arrays."
        ),
    }

    pairs.to_csv(tables_dir / "task41b8_exact_pair_seed_results.csv", index=False)
    trigger_summary.to_csv(
        tables_dir / "task41b8_trigger_multiseed_summary.csv", index=False
    )
    seed_summary.to_csv(
        tables_dir / "task41b8_seed_aggregated_summary.csv", index=False
    )
    rounds.to_csv(
        tables_dir / "task41b8_roundwise_exact_pair_results.csv", index=False
    )
    final_sources.to_csv(
        tables_dir / "task41b8_final_per_source_exact_pair_results.csv",
        index=False,
    )
    pd.DataFrame(test_access_rows).to_csv(
        tables_dir / "task41b8_test_access_audit.csv", index=False
    )
    pd.DataFrame(manifest_rows).to_csv(
        tables_dir / "task41b8_input_manifest_sha256.csv", index=False
    )
    (summary_dir / "task41b8_multiseed_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    x = np.arange(len(SLOTS))
    width = 0.36
    ordered_trigger = trigger_summary.set_index("slot").loc[SLOTS].reset_index()

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.bar(
        x - width / 2,
        ordered_trigger["plain_mean_macro_triggered_asr"],
        width,
        label="Plain FedAvg",
    )
    ax.bar(
        x + width / 2,
        ordered_trigger["defended_mean_macro_triggered_asr"],
        width,
        label="Trusted reconstruction",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(SLOTS, rotation=20, ha="right")
    ax.set_ylabel("Mean macro triggered ASR")
    ax.set_ylim(0, 1)
    ax.set_title("Task 41B multiseed triggered ASR")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "task41b8_multiseed_triggered_asr")

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    for slot in SLOTS:
        subset = pairs[pairs["slot"] == slot].sort_values("seed")
        ax.plot(
            subset["seed"].astype(str),
            subset["mean_macro_triggered_asr_reduction"],
            marker="o",
            label=slot,
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Seed")
    ax.set_ylabel("ASR reduction, plain minus defended")
    ax.set_title("Task 41B seed-wise frozen-defense effect")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "task41b8_seedwise_asr_reduction")

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.bar(
        ordered_trigger["slot"],
        ordered_trigger["improved_seed_count"],
    )
    ax.axhline(3, linewidth=1, linestyle="--")
    ax.set_ylim(0, 4.2)
    ax.set_ylabel("Seeds with lower mean ASR under defense")
    ax.set_title("Task 41B directional consistency across four seeds")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "task41b8_seed_direction_consistency")

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    for slot in SLOTS:
        subset = pairs[pairs["slot"] == slot].sort_values("seed")
        ax.plot(
            subset["seed"].astype(str),
            subset["clean_macro_f1_delta_defended_minus_plain"],
            marker="o",
            label=slot,
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Clean macro-F1 delta, defended minus plain")
    ax.set_title("Task 41B clean-utility effect")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "task41b8_clean_utility_delta")

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.bar(
        ordered_trigger["slot"],
        ordered_trigger["mean_defended_malicious_recall"],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean malicious-client recall")
    ax.set_title("Task 41B frozen detector recall under backdoor attacks")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "task41b8_detector_recall")

    print("===== TASK 41B.8 EXACT PAIRED MULTISEED SUMMARY =====")
    print()
    print("TRIGGER SUMMARY")
    display_columns = [
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
    print(trigger_summary[display_columns].to_string(index=False))
    print()
    print("TASK 41B DECISION")
    print("Attack evidence:", attack_evidence)
    print("Frozen defense evidence:", defense_evidence)
    print("Task 41B status:", task41b_status)
    print(f"Overall plain mean ASR: {overall_mean_plain_asr:.6f}")
    print(f"Overall defended mean ASR: {overall_mean_defended_asr:.6f}")
    print(f"Overall relative ASR reduction: {overall_relative_reduction:.6f}")
    print(
        "Both triggers improved in seeds:",
        f"{both_triggers_improved_seed_count}/4",
    )
    print(f"Mean defended malicious recall: {mean_detector_recall:.6f}")
    print(f"Maximum defended benign FPR: {maximum_benign_fpr:.6f}")
    print("METHOD REOPENED: False")
    print("FINAL PAPER CLAIM ALLOWED: False")
    print("Tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("TEST SETS ACCESSED: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

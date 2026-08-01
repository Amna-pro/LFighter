#!/usr/bin/env python3
"""V3.13.3 targeted source-to-Benign breadth closure.

This stage consolidates existing frozen-defense evidence only. It does not
rerun training, reopen the method, inspect test sets, or change thresholds.

Inputs:
- V3.12.3 DDoS-to-Benign frozen multiseed result
- V3.13.2 DoS-to-Benign exact-baseline result
- V3.13.2 remaining source-to-Benign exact-baseline results

Outputs:
- normalized seed and round tables
- pair-level publication table
- frozen-criteria matrix
- claim-scope decision
- publication PNG/PDF figures
- LaTeX table

Roadmap interpretation:
- Task 39A, source-to-Benign breadth, is closed.
- Task 39 remains partially complete until carefully chosen non-Benign target
  pairs are evaluated with the same frozen defense.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = [7, 99, 123, 2026]
SOURCES = [
    "BruteForce",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web-Based",
]
TARGET = "Benign"

# Frozen V3.13.2 breadth criteria.
MIN_ATTACK_EXCESS = 0.02
MIN_QUALIFIED_SEEDS = 3
MIN_POSITIVE_SEEDS = 3
MIN_MEAN_EXCESS_REMOVED = 0.40
MIN_IMPROVED_ROUNDS = 10
MIN_MEAN_RECALL = 0.90
MIN_ROUND_RECALL = 0.75
MAX_MEAN_FPR = 0.05
MAX_ROUND_FPR = 0.10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ddos-root", required=True, type=Path)
    p.add_argument("--dos-root", required=True, type=Path)
    p.add_argument("--remaining-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seeds", default="7,99,123,2026")
    return p.parse_args()


def pair_slug(source: str, target: str = TARGET) -> str:
    return (
        source.lower().replace("-", "_")
        + "_to_"
        + target.lower().replace("-", "_")
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def one_row(path: Path) -> pd.Series:
    table = pd.read_csv(require_file(path))
    if len(table) != 1:
        raise RuntimeError(f"Expected one row in {path}, found {len(table)}")
    return table.iloc[0]


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return float("nan")
    return float(numerator / denominator)


def normalized_column_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def semantic_numeric_value(
    row: pd.Series,
    required_tokens: Sequence[str],
    *,
    forbidden_tokens: Sequence[str] = (
        "delta", "difference", "diff", "recovery", "loss",
        "change", "removed", "ratio",
    ),
) -> float:
    """Resolve a numeric field despite harmless historical naming changes."""
    matches: List[Tuple[int, str]] = []
    for column in row.index:
        normalized = normalized_column_name(column)
        if not all(token in normalized for token in required_tokens):
            continue
        if any(token in normalized for token in forbidden_tokens):
            continue
        score = 0
        if "mean" in normalized:
            score += 4
        if normalized.endswith("macro_f1"):
            score += 2
        if "validation" in normalized or normalized.startswith("val_"):
            score += 1
        matches.append((score, str(column)))

    if not matches:
        raise KeyError(
            "No summary column matched tokens "
            f"{list(required_tokens)}. Available columns: "
            + ", ".join(map(str, row.index))
        )

    matches.sort(key=lambda item: (-item[0], item[1]))
    value = pd.to_numeric(row[matches[0][1]], errors="coerce")
    if not np.isfinite(value):
        raise ValueError(
            f"Resolved column {matches[0][1]} is not finite: {value}"
        )
    return float(value)


def branch_macro_f1_candidates(
    root: Path,
    seed: int,
    branch: str,
) -> List[Tuple[int, Path, str]]:
    """Find branch-specific macro-F1 columns across historical CSV layouts."""
    seed_root = root / f"seed_{seed}"
    if not seed_root.exists():
        return []

    ranked: List[Tuple[int, Path, str]] = []
    for path in seed_root.rglob("*.csv"):
        try:
            if path.stat().st_size > 64 * 1024 * 1024:
                continue
            columns = pd.read_csv(path, nrows=0).columns
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue

        normalized_path = normalized_column_name(
            str(path.relative_to(seed_root))
        )
        for column in columns:
            normalized_column = normalized_column_name(column)
            if "macro" not in normalized_column or "f1" not in normalized_column:
                continue
            if any(
                token in normalized_column
                for token in (
                    "delta", "difference", "diff", "recovery",
                    "loss", "change", "removed", "ratio",
                )
            ):
                continue

            score = 0
            if normalized_column in {
                "val_macro_f1",
                "validation_macro_f1",
                "macro_f1",
            }:
                score += 4
            if "mean" in normalized_column:
                score += 2

            if branch == "plain_clean":
                if "plain" in normalized_column and "clean" in normalized_column:
                    score += 30
                if "plain_clean" in normalized_path:
                    score += 24
                if "exact_clean" in normalized_path:
                    score += 20
                if "clean" in normalized_path:
                    score += 8
                if "attack" in normalized_column:
                    score -= 40
                if "attack" in normalized_path:
                    score -= 20
                if "reconstruction" in normalized_column:
                    score -= 30
                if "trusted_attack" in normalized_path:
                    score -= 30

            elif branch == "plain_attack":
                if "plain" in normalized_column and "attack" in normalized_column:
                    score += 35
                if "plain_attack" in normalized_path:
                    score += 28
                if "exact_plain" in normalized_path:
                    score += 24
                if "attack" in normalized_path and "plain" in normalized_path:
                    score += 18
                # V3.12.3 stores both clean and attacked branch columns in
                # v3123_seed_rounds.csv, so column semantics outrank filename.
                if normalized_path.endswith("v3123_seed_rounds_csv"):
                    score += 10
                if "reconstruction" in normalized_column:
                    score -= 45
                if "trusted" in normalized_column:
                    score -= 45
                if "trusted_attack" in normalized_path:
                    score -= 35
                if (
                    "clean" in normalized_column
                    and not (
                        "plain" in normalized_column
                        and "attack" in normalized_column
                    )
                ):
                    score -= 35

            elif branch == "defended_attack":
                if (
                    "reconstruction" in normalized_column
                    and "attack" in normalized_column
                ):
                    score += 35
                if "defended" in normalized_column and "attack" in normalized_column:
                    score += 35
                if "trusted_attack" in normalized_path:
                    score += 30
                if "reconstruction_attack" in normalized_path:
                    score += 28
                if (
                    normalized_column
                    in {"val_macro_f1", "validation_macro_f1", "macro_f1"}
                    and (
                        "trusted_attack" in normalized_path
                        or "reconstruction" in normalized_path
                    )
                ):
                    score += 20
                if "plain" in normalized_column:
                    score -= 40
                if "plain_attack" in normalized_path:
                    score -= 30
                if "clean" in normalized_column:
                    score -= 25
            else:
                raise ValueError(branch)

            if score > 0:
                ranked.append((score, path, str(column)))

    ranked.sort(
        key=lambda item: (-item[0], str(item[1]), item[2])
    )
    return ranked


def mean_numeric_column(path: Path, column: str) -> float:
    table = pd.read_csv(path, usecols=[column])
    values = pd.to_numeric(
        table[column], errors="coerce"
    ).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError(
            f"Column {column} in {path} has no finite values"
        )
    return float(values.mean())


def resolve_ddos_mean_macro_f1(
    summary: pd.Series,
    root: Path,
    seed: int,
    branch: str,
) -> float:
    token_map = {
        "plain_clean": ("plain", "clean", "macro", "f1"),
        "plain_attack": ("plain", "attack", "macro", "f1"),
        "defended_attack": ("reconstruction", "attack", "macro", "f1"),
    }
    try:
        return semantic_numeric_value(
            summary,
            token_map[branch],
        )
    except (KeyError, ValueError) as summary_error:
        failures: List[str] = []
        candidates = branch_macro_f1_candidates(
            root, seed, branch
        )
        for score, path, column in candidates:
            try:
                return mean_numeric_column(path, column)
            except (
                KeyError,
                ValueError,
                OSError,
                pd.errors.ParserError,
            ) as exc:
                failures.append(
                    f"score={score}, {path}, column={column}: "
                    f"{type(exc).__name__}: {exc}"
                )

        detail = "\n".join(failures[:20])
        inspected = "\n".join(
            f"score={score}, {path}, column={column}"
            for score, path, column in candidates[:20]
        )
        raise RuntimeError(
            f"Unable to resolve DDoS {branch} mean macro-F1 for seed {seed}. "
            f"Summary resolution failed with: {summary_error}.\n"
            f"Ranked column candidates:\n"
            f"{inspected or 'No candidate macro-F1 columns found.'}\n"
            f"Read failures:\n{detail or 'None'}"
        ) from summary_error


def find_ddos_seed_summary(root: Path, seed: int) -> Path:
    direct = (
        root / f"seed_{seed}" / "summary" / "tables"
        / "v3123_seed_summary.csv"
    )
    if direct.exists():
        return direct
    candidates = [
        path for path in root.rglob("v3123_seed_summary.csv")
        if f"seed_{seed}" in str(path)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one DDoS seed summary for seed {seed} under {root}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def find_ddos_seed_rounds(root: Path, seed: int) -> Path:
    direct = (
        root / f"seed_{seed}" / "summary" / "tables"
        / "v3123_seed_rounds.csv"
    )
    if direct.exists():
        return direct
    candidates = [
        path for path in root.rglob("v3123_seed_rounds.csv")
        if f"seed_{seed}" in str(path)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one DDoS round table for seed {seed} under {root}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def ddos_attack_metrics_path(root: Path, seed: int) -> Path:
    direct = (
        root / f"seed_{seed}" / "trusted_attack" / "tables"
        / "reconstruction_round_metrics.csv"
    )
    if direct.exists():
        return direct
    candidates = [
        path for path in root.rglob("reconstruction_round_metrics.csv")
        if f"seed_{seed}" in str(path)
        and "trusted_attack" in str(path)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one DDoS trusted-attack metric table for seed {seed}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def ddos_manifest_path(root: Path, seed: int) -> Path:
    direct = (
        root / f"seed_{seed}" / "trusted_attack" / "attack_manifest"
        / "malicious_client_poison_manifest.csv"
    )
    if direct.exists():
        return direct
    candidates = [
        path for path in root.rglob("malicious_client_poison_manifest.csv")
        if f"seed_{seed}" in str(path)
        and "trusted_attack" in str(path)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one DDoS poison manifest for seed {seed}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def normalize_ddos_seed(
    root: Path,
    seed: int,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    summary = one_row(find_ddos_seed_summary(root, seed))
    rounds = pd.read_csv(find_ddos_seed_rounds(root, seed)).sort_values(
        "monitoring_round"
    ).reset_index(drop=True)
    defended_rounds = pd.read_csv(
        ddos_attack_metrics_path(root, seed)
    ).sort_values("monitoring_round").reset_index(drop=True)
    if len(rounds) != 4 or len(defended_rounds) != 4:
        raise RuntimeError("DDoS seed must contain exactly four rounds")

    manifest = pd.read_csv(ddos_manifest_path(root, seed))
    malicious = manifest[manifest["is_malicious"].astype(bool)].copy()
    selected_clients = "|".join(
        map(str, malicious["client_id"].astype(int).sort_values().tolist())
    )
    global_source_rows = int(manifest["source_rows_before_poisoning"].sum())
    selected_source_rows = int(
        malicious["source_rows_before_poisoning"].sum()
    )
    exposure = safe_ratio(selected_source_rows, global_source_rows)

    clean_rate = float(summary["plain_clean_mean_source_to_target_rate"])
    plain_rate = float(summary["plain_attack_mean_source_to_target_rate"])
    defended_rate = float(
        summary["reconstruction_attack_mean_source_to_target_rate"]
    )
    attack_excess = plain_rate - clean_rate
    residual_excess = defended_rate - clean_rate
    common_clean_removed = (
        1.0 - safe_ratio(residual_excess, attack_excess)
        if attack_excess > 1e-12
        else float("nan")
    )

    seed_row = {
        "seed": seed,
        "source_class": "DDoS",
        "target_class": TARGET,
        "pair_name": "DDoS_to_Benign",
        "is_development_pair": True,
        "selected_clients": selected_clients,
        "coalition_hash_sha256": "",
        "coalition_selection_rule": "fixed_development_coalition",
        "global_source_rows": global_source_rows,
        "selected_source_rows": selected_source_rows,
        "global_source_exposure_fraction": exposure,
        "clean_mean_macro_f1": resolve_ddos_mean_macro_f1(
            summary, root, seed, "plain_clean"
        ),
        "plain_attack_mean_macro_f1": resolve_ddos_mean_macro_f1(
            summary, root, seed, "plain_attack"
        ),
        "defended_attack_mean_macro_f1": resolve_ddos_mean_macro_f1(
            summary, root, seed, "defended_attack"
        ),
        "clean_mean_source_to_target_rate": clean_rate,
        "plain_attack_mean_source_to_target_rate": plain_rate,
        "defended_attack_mean_source_to_target_rate": defended_rate,
        "plain_attack_excess_over_clean": attack_excess,
        "defended_residual_excess_over_clean": residual_excess,
        "absolute_rate_reduction": plain_rate - defended_rate,
        "attack_excess_removed_reported": float(
            summary["reconstruction_attack_excess_removed"]
        ),
        "attack_excess_removed_common_clean_reference": common_clean_removed,
        "rounds_improved_vs_plain": int(
            summary["rounds_improved_vs_plain_attack"]
        ),
        "mean_malicious_recall": float(summary["mean_malicious_recall"]),
        "minimum_malicious_recall": float(
            summary["minimum_malicious_recall"]
        ),
        "mean_benign_fpr": float(summary["mean_benign_fpr"]),
        "maximum_benign_fpr": float(summary["maximum_benign_fpr"]),
        "plain_attack_qualified_absolute_002": bool(
            attack_excess >= MIN_ATTACK_EXCESS
        ),
        "defense_positive_reduction": bool(defended_rate < plain_rate),
        "exact_plain_baseline_used": True,
        "source_result_version": "3.12.3",
        "test_sets_accessed": False,
    }

    round_table = pd.DataFrame({
        "seed": seed,
        "source_class": "DDoS",
        "target_class": TARGET,
        "monitoring_round": rounds["monitoring_round"].astype(int),
        "clean_macro_f1": np.nan,
        "plain_attack_macro_f1": np.nan,
        "defended_attack_macro_f1": defended_rounds["val_macro_f1"],
        "clean_source_to_target_rate":
            rounds["plain_clean_source_to_target_rate"],
        "plain_attack_source_to_target_rate":
            rounds["plain_attack_source_to_target_rate"],
        "defended_attack_source_to_target_rate":
            rounds["reconstruction_attack_source_to_target_rate"],
        "malicious_recall": defended_rounds["malicious_recall"],
        "benign_false_positive_rate":
            defended_rounds["benign_false_positive_rate"],
    })
    round_table["attack_excess"] = (
        round_table["plain_attack_source_to_target_rate"]
        - round_table["clean_source_to_target_rate"]
    )
    round_table["defended_residual_excess"] = (
        round_table["defended_attack_source_to_target_rate"]
        - round_table["clean_source_to_target_rate"]
    )
    round_table["defense_improved_vs_plain"] = (
        round_table["defended_attack_source_to_target_rate"]
        < round_table["plain_attack_source_to_target_rate"]
    )
    return seed_row, round_table


def normalize_v3132_seed(
    root: Path,
    source: str,
    seed: int,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    base = (
        root / "runs" / pair_slug(source) / f"seed_{seed}"
        / "summary" / "tables"
    )
    summary = one_row(base / "v313_pair_seed_summary.csv")
    rounds = pd.read_csv(
        require_file(base / "v313_pair_seed_rounds.csv")
    ).sort_values("monitoring_round").reset_index(drop=True)
    if len(rounds) != 4:
        raise RuntimeError(
            f"Expected four rounds for {source}, seed {seed}"
        )

    if str(summary["source_class"]) != source:
        raise RuntimeError(
            f"Source mismatch in {base}: {summary['source_class']} != {source}"
        )
    if int(summary["seed"]) != seed:
        raise RuntimeError(
            f"Seed mismatch in {base}: {summary['seed']} != {seed}"
        )

    attack_excess = float(summary["plain_attack_excess_over_clean"])
    residual_excess = float(summary["defended_residual_excess_over_clean"])
    common_clean_removed = (
        1.0 - safe_ratio(residual_excess, attack_excess)
        if attack_excess > 1e-12
        else float("nan")
    )
    seed_row = {
        "seed": seed,
        "source_class": source,
        "target_class": TARGET,
        "pair_name": f"{source}_to_{TARGET}",
        "is_development_pair": bool(source == "DDoS"),
        "selected_clients": str(summary["selected_clients"]),
        "coalition_hash_sha256": str(summary["coalition_hash_sha256"]),
        "coalition_selection_rule": str(
            summary["coalition_selection_rule"]
        ),
        "global_source_rows": int(summary["global_source_rows"]),
        "selected_source_rows": int(summary["selected_source_rows"]),
        "global_source_exposure_fraction": float(
            summary["global_source_exposure_fraction"]
        ),
        "clean_mean_macro_f1": float(summary["clean_mean_macro_f1"]),
        "plain_attack_mean_macro_f1": float(
            summary["plain_attack_mean_macro_f1"]
        ),
        "defended_attack_mean_macro_f1": float(
            summary["defended_attack_mean_macro_f1"]
        ),
        "clean_mean_source_to_target_rate": float(
            summary["clean_mean_source_to_target_rate"]
        ),
        "plain_attack_mean_source_to_target_rate": float(
            summary["plain_attack_mean_source_to_target_rate"]
        ),
        "defended_attack_mean_source_to_target_rate": float(
            summary["defended_attack_mean_source_to_target_rate"]
        ),
        "plain_attack_excess_over_clean": attack_excess,
        "defended_residual_excess_over_clean": residual_excess,
        "absolute_rate_reduction": float(summary["absolute_rate_reduction"]),
        "attack_excess_removed_reported": float(
            summary["attack_excess_removed"]
        ),
        "attack_excess_removed_common_clean_reference": common_clean_removed,
        "rounds_improved_vs_plain": int(
            summary["rounds_improved_vs_plain"]
        ),
        "mean_malicious_recall": float(
            summary["mean_malicious_recall"]
        ),
        "minimum_malicious_recall": float(
            summary["minimum_malicious_recall"]
        ),
        "mean_benign_fpr": float(summary["mean_benign_fpr"]),
        "maximum_benign_fpr": float(summary["maximum_benign_fpr"]),
        "plain_attack_qualified_absolute_002": bool(
            summary["plain_attack_qualified_absolute_002"]
        ),
        "defense_positive_reduction": bool(
            summary["defense_positive_reduction"]
        ),
        "exact_plain_baseline_used": bool(
            summary["exact_v310_plain_baseline_used"]
        ),
        "source_result_version": "3.13.2",
        "test_sets_accessed": bool(summary["test_sets_accessed"]),
    }
    normalized_rounds = rounds.rename(columns={
        "benign_false_positive_rate": "benign_false_positive_rate",
    }).copy()
    normalized_rounds["source_result_version"] = "3.13.2"
    return seed_row, normalized_rounds


def classify_evidence(row: pd.Series) -> str:
    if bool(row["passes_strict_frozen_criteria"]):
        return "STRICT_PASS"
    if bool(row["passes_mitigation_only_criteria"]):
        return "MITIGATION_PASS_DETECTOR_LIMITED"
    if bool(row["defense_positive_on_3_of_4_seeds"]):
        return "CONSISTENT_PARTIAL_MITIGATION"
    return "NO_CONSISTENT_MITIGATION"


def build_pair_table(seed_table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for source in SOURCES:
        group = seed_table[
            seed_table["source_class"].eq(source)
        ].sort_values("seed")
        if len(group) != 4:
            raise RuntimeError(
                f"Expected four seeds for {source}, found {len(group)}"
            )
        finite_removed = group[
            "attack_excess_removed_reported"
        ].replace([np.inf, -np.inf], np.nan).dropna()
        qualified_seed_count = int(
            group["plain_attack_qualified_absolute_002"].sum()
        )
        positive_seed_count = int(
            group["defense_positive_reduction"].sum()
        )
        clean_macro = float(group["clean_mean_macro_f1"].mean())
        plain_macro = float(
            group["plain_attack_mean_macro_f1"].mean()
        )
        defended_macro = float(
            group["defended_attack_mean_macro_f1"].mean()
        )
        attack_macro_loss = clean_macro - plain_macro
        macro_recovery = defended_macro - plain_macro
        macro_loss_recovery_fraction = (
            safe_ratio(macro_recovery, attack_macro_loss)
            if attack_macro_loss > 1e-12
            else float("nan")
        )

        row = {
            "source_class": source,
            "target_class": TARGET,
            "pair_name": f"{source}_to_{TARGET}",
            "is_development_pair": bool(source == "DDoS"),
            "seed_count": 4,
            "selected_clients": str(group["selected_clients"].iloc[0]),
            "coalition_selection_rule": str(
                group["coalition_selection_rule"].iloc[0]
            ),
            "global_source_exposure_fraction": float(
                group["global_source_exposure_fraction"].iloc[0]
            ),
            "mean_clean_macro_f1": clean_macro,
            "mean_plain_attack_macro_f1": plain_macro,
            "mean_defended_attack_macro_f1": defended_macro,
            "mean_attack_macro_f1_recovery_vs_plain": macro_recovery,
            "mean_plain_attack_macro_f1_loss_vs_clean": attack_macro_loss,
            "macro_f1_loss_recovery_fraction":
                macro_loss_recovery_fraction,
            "mean_clean_source_to_target_rate": float(
                group["clean_mean_source_to_target_rate"].mean()
            ),
            "mean_plain_attack_source_to_target_rate": float(
                group["plain_attack_mean_source_to_target_rate"].mean()
            ),
            "mean_defended_attack_source_to_target_rate": float(
                group["defended_attack_mean_source_to_target_rate"].mean()
            ),
            "mean_plain_attack_excess_over_clean": float(
                group["plain_attack_excess_over_clean"].mean()
            ),
            "mean_absolute_rate_reduction": float(
                group["absolute_rate_reduction"].mean()
            ),
            "mean_attack_excess_removed": (
                float(finite_removed.mean())
                if len(finite_removed) else float("nan")
            ),
            "median_attack_excess_removed": (
                float(finite_removed.median())
                if len(finite_removed) else float("nan")
            ),
            "minimum_seed_attack_excess_removed": (
                float(finite_removed.min())
                if len(finite_removed) else float("nan")
            ),
            "qualified_attack_seed_count": qualified_seed_count,
            "pair_attack_qualified": bool(
                qualified_seed_count >= MIN_QUALIFIED_SEEDS
                and group[
                    "plain_attack_excess_over_clean"
                ].mean() >= MIN_ATTACK_EXCESS
            ),
            "positive_reduction_seed_count": positive_seed_count,
            "defense_positive_on_3_of_4_seeds": bool(
                positive_seed_count >= MIN_POSITIVE_SEEDS
            ),
            "total_rounds_improved": int(
                group["rounds_improved_vs_plain"].sum()
            ),
            "total_rounds": 16,
            "mean_malicious_recall": float(
                group["mean_malicious_recall"].mean()
            ),
            "minimum_malicious_recall": float(
                group["minimum_malicious_recall"].min()
            ),
            "mean_benign_fpr": float(
                group["mean_benign_fpr"].mean()
            ),
            "maximum_benign_fpr": float(
                group["maximum_benign_fpr"].max()
            ),
            "exact_plain_baseline_all_seeds": bool(
                group["exact_plain_baseline_used"].all()
            ),
            "test_sets_accessed": bool(
                group["test_sets_accessed"].any()
            ),
        }
        row["passes_attack_qualification"] = bool(
            row["pair_attack_qualified"]
        )
        row["passes_positive_seed_count"] = bool(
            row["positive_reduction_seed_count"]
            >= MIN_POSITIVE_SEEDS
        )
        row["passes_mean_excess_removed_040"] = bool(
            row["mean_attack_excess_removed"]
            >= MIN_MEAN_EXCESS_REMOVED
        )
        row["passes_improved_rounds_10_of_16"] = bool(
            row["total_rounds_improved"] >= MIN_IMPROVED_ROUNDS
        )
        row["passes_mean_recall_090"] = bool(
            row["mean_malicious_recall"] >= MIN_MEAN_RECALL
        )
        row["passes_minimum_round_recall_075"] = bool(
            row["minimum_malicious_recall"] >= MIN_ROUND_RECALL
        )
        row["passes_mean_benign_fpr_005"] = bool(
            row["mean_benign_fpr"] <= MAX_MEAN_FPR
        )
        row["passes_maximum_round_fpr_010"] = bool(
            row["maximum_benign_fpr"] <= MAX_ROUND_FPR
        )
        row["passes_mitigation_only_criteria"] = bool(
            row["passes_attack_qualification"]
            and row["passes_positive_seed_count"]
            and row["passes_mean_excess_removed_040"]
            and row["passes_improved_rounds_10_of_16"]
        )
        row["passes_detector_criteria"] = bool(
            row["passes_mean_recall_090"]
            and row["passes_minimum_round_recall_075"]
            and row["passes_mean_benign_fpr_005"]
            and row["passes_maximum_round_fpr_010"]
        )
        row["passes_strict_frozen_criteria"] = bool(
            row["passes_mitigation_only_criteria"]
            and row["passes_detector_criteria"]
        )
        rows.append(row)

    table = pd.DataFrame(rows)
    table["evidence_label"] = table.apply(classify_evidence, axis=1)
    return table


def write_claim_scope(
    path: Path,
    pair_table: pd.DataFrame,
    decision: Dict[str, object],
) -> None:
    strict = pair_table.loc[
        pair_table["passes_strict_frozen_criteria"],
        "source_class",
    ].tolist()
    mitigation = pair_table.loc[
        pair_table["passes_mitigation_only_criteria"]
        & ~pair_table["passes_strict_frozen_criteria"],
        "source_class",
    ].tolist()
    partial = pair_table.loc[
        pair_table["evidence_label"].eq(
            "CONSISTENT_PARTIAL_MITIGATION"
        ),
        "source_class",
    ].tolist()

    text = f"""# Task 39A Targeted Source-to-Benign Breadth Closure

## Frozen protocol

The temporal independent-anchor detector, EMA q99 branch, instantaneous q95
rescue branch, center-plus-residual reconstruction, original sample-count
weights, four trusted warmup rounds, stable client identities, fixed
partitions, and attack-specific top-eight source-capable coalitions were not
retuned during this consolidation.

## Evidence-supported conclusion

The frozen defense produced positive mean source-to-Benign reduction on
{decision['positive_reduction_pair_count']} of {decision['pair_count']} tested
pairs. Strict frozen criteria passed for: {', '.join(strict) if strict else 'none'}.

Pairs with mitigation-level effectiveness but detector limitations:
{', '.join(mitigation) if mitigation else 'none'}.

Pairs with consistent but sub-threshold mitigation:
{', '.join(partial) if partial else 'none'}.

The result supports a scoped claim that the frozen framework mitigates several
targeted label-flipping attacks under the stated trusted-warmup, stable-client,
20-client, alpha-0.5, full-participation assumptions. It does not support a
universal poisoning-defense claim.

## Roadmap status

Task 39A, source-to-Benign targeted breadth, is complete.

Task 39 as a whole remains partially complete because carefully selected
non-Benign target pairs have not yet been evaluated. The next experiment must
keep the defense frozen and test those pairs before moving to Task 40
untargeted label poisoning.

## Data-access and tuning restrictions

- Method reopened: no
- Attack-specific retuning: no
- Natural test accessed: no
- Diagnostic test accessed: no
- Development seeds used: 7, 99, 123, 2026
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    seeds = [
        int(x.strip()) for x in args.seeds.split(",") if x.strip()
    ]
    if seeds != SEEDS:
        raise ValueError(
            f"V3.13.3 is frozen to development seeds {SEEDS}, got {seeds}"
        )

    ddos_root = args.ddos_root.expanduser().resolve()
    dos_root = args.dos_root.expanduser().resolve()
    remaining_root = args.remaining_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    for root in (ddos_root, dos_root, remaining_root):
        if not root.exists():
            raise FileNotFoundError(root)

    seed_rows: List[Dict[str, object]] = []
    round_rows: List[pd.DataFrame] = []

    for seed in seeds:
        seed_row, rounds = normalize_ddos_seed(ddos_root, seed)
        seed_rows.append(seed_row)
        round_rows.append(rounds)

    for source in ["DoS"]:
        for seed in seeds:
            seed_row, rounds = normalize_v3132_seed(
                dos_root, source, seed
            )
            seed_rows.append(seed_row)
            round_rows.append(rounds)

    for source in [
        "BruteForce", "Mirai", "Recon", "Spoofing", "Web-Based"
    ]:
        for seed in seeds:
            seed_row, rounds = normalize_v3132_seed(
                remaining_root, source, seed
            )
            seed_rows.append(seed_row)
            round_rows.append(rounds)

    seed_table = pd.DataFrame(seed_rows).sort_values(
        ["source_class", "seed"]
    ).reset_index(drop=True)
    round_table = pd.concat(
        round_rows, ignore_index=True
    ).sort_values(
        ["source_class", "seed", "monitoring_round"]
    ).reset_index(drop=True)

    if len(seed_table) != 28:
        raise RuntimeError(
            f"Expected 28 pair-seed rows, found {len(seed_table)}"
        )
    if len(round_table) != 112:
        raise RuntimeError(
            f"Expected 112 round rows, found {len(round_table)}"
        )
    if seed_table["test_sets_accessed"].astype(bool).any():
        raise RuntimeError("A source result reports test-set access")
    if not seed_table["exact_plain_baseline_used"].astype(bool).all():
        raise RuntimeError("At least one seed lacks exact plain baseline")

    seed_table.to_csv(
        tables / "v3133_normalized_pair_seed_summary.csv",
        index=False,
    )
    round_table.to_csv(
        tables / "v3133_normalized_round_summary.csv",
        index=False,
    )

    pair_table = build_pair_table(seed_table)
    pair_table.to_csv(
        tables / "v3133_targeted_breadth_pair_summary.csv",
        index=False,
    )

    criteria_columns = [
        "source_class",
        "passes_attack_qualification",
        "passes_positive_seed_count",
        "passes_mean_excess_removed_040",
        "passes_improved_rounds_10_of_16",
        "passes_mean_recall_090",
        "passes_minimum_round_recall_075",
        "passes_mean_benign_fpr_005",
        "passes_maximum_round_fpr_010",
        "passes_mitigation_only_criteria",
        "passes_detector_criteria",
        "passes_strict_frozen_criteria",
        "evidence_label",
    ]
    criteria_table = pair_table[criteria_columns].copy()
    criteria_table.to_csv(
        tables / "v3133_frozen_criteria_matrix.csv",
        index=False,
    )

    publication_columns = [
        "source_class",
        "mean_clean_source_to_target_rate",
        "mean_plain_attack_source_to_target_rate",
        "mean_defended_attack_source_to_target_rate",
        "mean_attack_excess_removed",
        "minimum_seed_attack_excess_removed",
        "mean_clean_macro_f1",
        "mean_plain_attack_macro_f1",
        "mean_defended_attack_macro_f1",
        "mean_attack_macro_f1_recovery_vs_plain",
        "mean_malicious_recall",
        "minimum_malicious_recall",
        "mean_benign_fpr",
        "maximum_benign_fpr",
        "total_rounds_improved",
        "passes_strict_frozen_criteria",
        "evidence_label",
    ]
    publication = pair_table[publication_columns].copy()
    publication.to_csv(
        tables / "v3133_publication_table.csv",
        index=False,
    )
    try:
        latex = publication.to_latex(
            index=False,
            float_format=lambda x: f"{x:.4f}",
            escape=True,
        )
        (tables / "v3133_publication_table.tex").write_text(
            latex, encoding="utf-8"
        )
    except Exception as exc:
        (tables / "v3133_publication_table_latex_error.txt").write_text(
            f"{type(exc).__name__}: {exc}",
            encoding="utf-8",
        )

    unseen = pair_table[
        ~pair_table["is_development_pair"]
    ].copy()
    qualified_unseen = unseen[
        unseen["pair_attack_qualified"]
    ].copy()
    strict_sources = pair_table.loc[
        pair_table["passes_strict_frozen_criteria"],
        "source_class",
    ].tolist()
    mitigation_limited_sources = pair_table.loc[
        pair_table["passes_mitigation_only_criteria"]
        & ~pair_table["passes_strict_frozen_criteria"],
        "source_class",
    ].tolist()
    partial_sources = pair_table.loc[
        pair_table["evidence_label"].eq(
            "CONSISTENT_PARTIAL_MITIGATION"
        ),
        "source_class",
    ].tolist()

    decision = {
        "experiment_version": "3.13.3",
        "stage": "task39a_targeted_source_to_benign_breadth_closure",
        "pair_count": int(len(pair_table)),
        "unseen_pair_count": int(len(unseen)),
        "qualified_unseen_pair_count": int(len(qualified_unseen)),
        "positive_reduction_pair_count": int(
            pair_table[
                "defense_positive_on_3_of_4_seeds"
            ].sum()
        ),
        "strict_pass_pair_count": int(
            pair_table["passes_strict_frozen_criteria"].sum()
        ),
        "strict_pass_sources": "|".join(strict_sources),
        "mitigation_pass_detector_limited_sources":
            "|".join(mitigation_limited_sources),
        "consistent_partial_mitigation_sources":
            "|".join(partial_sources),
        "mean_qualified_unseen_attack_excess_removed": (
            float(
                qualified_unseen[
                    "mean_attack_excess_removed"
                ].mean()
            )
            if len(qualified_unseen) else float("nan")
        ),
        "minimum_qualified_unseen_attack_excess_removed": (
            float(
                qualified_unseen[
                    "mean_attack_excess_removed"
                ].min()
            )
            if len(qualified_unseen) else float("nan")
        ),
        "total_improved_rounds": int(
            round_table["defense_improved_vs_plain"].sum()
        ),
        "total_rounds": int(len(round_table)),
        "mean_malicious_recall": float(
            seed_table["mean_malicious_recall"].mean()
        ),
        "minimum_malicious_recall": float(
            seed_table["minimum_malicious_recall"].min()
        ),
        "mean_benign_fpr": float(
            seed_table["mean_benign_fpr"].mean()
        ),
        "maximum_benign_fpr": float(
            seed_table["maximum_benign_fpr"].max()
        ),
        "task39a_source_to_benign_status": "COMPLETED",
        "task39_full_status":
            "PARTIALLY_COMPLETED_NON_BENIGN_TARGETS_REMAIN",
        "frozen_detector": True,
        "frozen_reconstruction_policy": "center_plus_residual",
        "method_reopened": False,
        "attack_specific_retuning": False,
        "federated_training_rerun": False,
        "test_sets_accessed": False,
        "next_stage":
            "Task 39B frozen non-Benign target-pair breadth",
    }
    pd.DataFrame([decision]).to_csv(
        tables / "v3133_task39a_decision.csv",
        index=False,
    )
    with (output / "v3133_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({
            **decision,
            "seeds": seeds,
            "sources": SOURCES,
            "target_class": TARGET,
            "frozen_criteria": {
                "minimum_attack_excess": MIN_ATTACK_EXCESS,
                "minimum_qualified_seeds": MIN_QUALIFIED_SEEDS,
                "minimum_positive_seeds": MIN_POSITIVE_SEEDS,
                "minimum_mean_excess_removed":
                    MIN_MEAN_EXCESS_REMOVED,
                "minimum_improved_rounds": MIN_IMPROVED_ROUNDS,
                "minimum_mean_recall": MIN_MEAN_RECALL,
                "minimum_round_recall": MIN_ROUND_RECALL,
                "maximum_mean_fpr": MAX_MEAN_FPR,
                "maximum_round_fpr": MAX_ROUND_FPR,
            },
        }, handle, indent=2)

    write_claim_scope(
        output / "TASK39A_CLAIM_SCOPE.md",
        pair_table,
        decision,
    )

    x = np.arange(len(pair_table))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(
        x - width,
        pair_table["mean_clean_source_to_target_rate"],
        width,
        label="Clean",
    )
    ax.bar(
        x,
        pair_table["mean_plain_attack_source_to_target_rate"],
        width,
        label="Plain attacked FedAvg",
    )
    ax.bar(
        x + width,
        pair_table["mean_defended_attack_source_to_target_rate"],
        width,
        label="Frozen trusted reconstruction",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pair_table["source_class"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Source class redirected to Benign")
    ax.set_ylabel("Mean source-to-Benign rate")
    ax.set_title("Targeted source-to-Benign breadth")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "source_to_benign_rates")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(
        pair_table["source_class"],
        pair_table["mean_attack_excess_removed"],
    )
    ax.axhline(
        MIN_MEAN_EXCESS_REMOVED,
        linestyle="--",
        linewidth=1,
        label="Frozen 40% mitigation criterion",
    )
    ax.axhline(0, linestyle=":", linewidth=1)
    ax.set_ylabel("Mean attack-induced excess removed")
    ax.set_xlabel("Source class")
    ax.set_title("Frozen reconstruction mitigation by attack pair")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "attack_excess_removed")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(
        x - width,
        pair_table["mean_clean_macro_f1"],
        width,
        label="Clean",
    )
    ax.bar(
        x,
        pair_table["mean_plain_attack_macro_f1"],
        width,
        label="Plain attacked FedAvg",
    )
    ax.bar(
        x + width,
        pair_table["mean_defended_attack_macro_f1"],
        width,
        label="Frozen trusted reconstruction",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pair_table["source_class"])
    ax.set_ylim(0, max(0.5, pair_table[
        [
            "mean_clean_macro_f1",
            "mean_plain_attack_macro_f1",
            "mean_defended_attack_macro_f1",
        ]
    ].to_numpy().max() * 1.15))
    ax.set_xlabel("Source class")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Utility under targeted label flipping")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "macro_f1_comparison")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(
        x - width / 2,
        pair_table["mean_malicious_recall"],
        width,
        label="Mean malicious recall",
    )
    ax.bar(
        x + width / 2,
        pair_table["mean_benign_fpr"],
        width,
        label="Mean benign FPR",
    )
    ax.axhline(
        MIN_MEAN_RECALL,
        linestyle="--",
        linewidth=1,
        label="Recall criterion",
    )
    ax.axhline(
        MAX_MEAN_FPR,
        linestyle=":",
        linewidth=1,
        label="FPR criterion",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pair_table["source_class"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Source class")
    ax.set_ylabel("Rate")
    ax.set_title("Frozen detector behavior across targeted attacks")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(fig, figures / "detector_rates")

    print("V3.13.3 targeted breadth closure complete")
    print()
    print("PAIR SUMMARY")
    display = [
        "source_class",
        "mean_plain_attack_excess_over_clean",
        "mean_attack_excess_removed",
        "minimum_seed_attack_excess_removed",
        "mean_attack_macro_f1_recovery_vs_plain",
        "total_rounds_improved",
        "mean_malicious_recall",
        "minimum_malicious_recall",
        "mean_benign_fpr",
        "maximum_benign_fpr",
        "passes_mitigation_only_criteria",
        "passes_detector_criteria",
        "passes_strict_frozen_criteria",
        "evidence_label",
    ]
    print(pair_table[display].to_string(index=False))
    print()
    print("TASK 39A DECISION")
    print(pd.DataFrame([decision]).to_string(index=False))
    print()
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    print("Claim scope:", output / "TASK39A_CLAIM_SCOPE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

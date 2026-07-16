#!/usr/bin/env python3
"""Paired five-seed DDoS-to-Benign attack reliability study, V3.3.

The weak, moderate, and strong attack settings were frozen from the V3.2
validation-only calibration. This script repeats each setting over the same five
model seeds used by the clean FedAvg baseline while keeping the partition,
malicious-client IDs, and poisoned-row selection fixed.

The output includes seed-level results, aggregate mean/std/95% t intervals,
paired clean-versus-attack tests, attack-strength comparisons, CSV/JSON source
artifacts, and PNG/PDF figures.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, t as student_t, wilcoxon


SETTINGS = [
    {
        "strength": "weak",
        "malicious_client_fraction": 0.10,
        "poison_fraction": 0.25,
        "malicious_clients": [1, 10],
        "v32_run_name": "mc10_pf025",
    },
    {
        "strength": "moderate",
        "malicious_client_fraction": 0.20,
        "poison_fraction": 0.50,
        "malicious_clients": [1, 10, 14, 15],
        "v32_run_name": "mc20_pf050",
    },
    {
        "strength": "strong",
        "malicious_client_fraction": 0.40,
        "poison_fraction": 1.00,
        "malicious_clients": [1, 7, 8, 10, 14, 15, 17, 18],
        "v32_run_name": "mc40_pf100",
    },
]
STRENGTH_ORDER = ["weak", "moderate", "strong"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paired five-seed DDoS-to-Benign reliability study."
    )
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-multiseed-dir", required=True, type=Path)
    parser.add_argument("--v32-results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--runner-script",
        type=Path,
        default=Path("scripts/run_targeted_label_flip_v292.py"),
    )
    parser.add_argument("--seeds", default="42,123,2026,7,99")
    parser.add_argument("--source-class", default="DDoS")
    parser.add_argument("--target-class", default="Benign")
    parser.add_argument("--min-source-samples", type=int, default=1000)
    parser.add_argument("--attack-seed", type=int, default=42)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--participation-rate", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_seeds(text: str) -> List[int]:
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    if len(seeds) < 3:
        raise ValueError("Use at least three seeds.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")
    return seeds


def run_complete(run_dir: Path) -> bool:
    required = [
        run_dir / "targeted_label_flip_metadata.json",
        run_dir / "attack_manifest" / "attack_summary.csv",
        run_dir / "tables" / "round_metrics.csv",
        run_dir / "tables" / "clean_vs_attack_source_target.csv",
        run_dir / "tables" / "clean_vs_attack_overall.csv",
        run_dir / "tables" / "clean_vs_attack_per_class.csv",
        run_dir / "tables" / "attacked_source_target_metrics.csv",
        run_dir / "checkpoints" / "best_validation_model.pt",
    ]
    return all(path.exists() for path in required)


def validate_frozen_settings(v32_results_dir: Path) -> pd.DataFrame:
    summary_path = v32_results_dir / "tables" / "attack_strength_grid_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"V3.2 summary not found: {summary_path}")

    summary = pd.read_csv(summary_path)
    rows = []
    for setting in SETTINGS:
        selected = summary[summary["run_name"] == setting["v32_run_name"]]
        if len(selected) != 1:
            raise ValueError(
                f"Expected one V3.2 row for {setting['v32_run_name']}, "
                f"found {len(selected)}."
            )
        row = selected.iloc[0]
        observed_clients = sorted(
            int(value) for value in str(row["malicious_clients"]).split("|")
        )
        expected_clients = sorted(setting["malicious_clients"])
        if observed_clients != expected_clients:
            raise ValueError(
                f"Frozen client mismatch for {setting['strength']}: "
                f"{observed_clients} != {expected_clients}"
            )
        if not np.isclose(
            float(row["malicious_client_fraction"]),
            setting["malicious_client_fraction"],
        ):
            raise ValueError(f"Malicious fraction mismatch for {setting['strength']}.")
        if not np.isclose(
            float(row["poison_fraction"]),
            setting["poison_fraction"],
        ):
            raise ValueError(f"Poison fraction mismatch for {setting['strength']}.")

        rows.append(
            {
                "strength": setting["strength"],
                "v32_run_name": setting["v32_run_name"],
                "malicious_client_fraction": setting[
                    "malicious_client_fraction"
                ],
                "poison_fraction": setting["poison_fraction"],
                "malicious_clients": "|".join(map(str, expected_clients)),
                "global_source_exposure_fraction_v32": float(
                    row["global_source_exposure_fraction"]
                ),
                "validation_source_to_target_rate_v32": float(
                    row["validation_source_to_target_rate"]
                ),
                "natural_source_to_target_rate_v32": float(
                    row["natural_source_to_target_rate_attacked"]
                ),
            }
        )
    return pd.DataFrame(rows)


def execute_run(
    args: argparse.Namespace,
    runner_script: Path,
    clean_seed_dir: Path,
    run_dir: Path,
    setting: Dict[str, object],
    seed: int,
) -> None:
    if args.skip_existing and run_complete(run_dir):
        print(f"Reusing completed run: {setting['strength']}, seed {seed}")
        return

    if run_dir.exists():
        print(f"Removing incomplete run: {setting['strength']}, seed {seed}")
        shutil.rmtree(run_dir)

    command = [
        sys.executable,
        str(runner_script),
        "--data-file", str(args.data_file.expanduser().resolve()),
        "--partition-file", str(args.partition_file.expanduser().resolve()),
        "--clean-seed-dir", str(clean_seed_dir),
        "--output-dir", str(run_dir),
        "--source-class", args.source_class,
        "--target-class", args.target_class,
        "--malicious-client-fraction",
        str(setting["malicious_client_fraction"]),
        "--malicious-clients",
        ",".join(map(str, setting["malicious_clients"])),
        "--poison-fraction", str(setting["poison_fraction"]),
        "--min-source-samples", str(args.min_source_samples),
        "--attack-seed", str(args.attack_seed),
        "--model-seed", str(seed),
        "--num-clients", str(args.num_clients),
        "--rounds", str(args.rounds),
        "--participation-rate", str(args.participation_rate),
        "--local-epochs", str(args.local_epochs),
        "--batch-size", str(args.batch_size),
        "--evaluation-batch-size", str(args.evaluation_batch_size),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--threads", str(args.threads),
        "--early-stopping-patience", str(args.early_stopping_patience),
    ]
    print()
    print("=" * 78)
    print(
        f"Running {setting['strength']} attack, seed {seed}, "
        f"clients={setting['malicious_clients']}, "
        f"poison={setting['poison_fraction']}"
    )
    print("=" * 78)
    subprocess.run(command, check=True)


def collect_run(
    run_dir: Path,
    setting: Dict[str, object],
    seed: int,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    with (run_dir / "targeted_label_flip_metadata.json").open(
        "r", encoding="utf-8"
    ) as handle:
        metadata = json.load(handle)

    attack_summary = pd.read_csv(
        run_dir / "attack_manifest" / "attack_summary.csv"
    ).iloc[0]
    rounds = pd.read_csv(run_dir / "tables" / "round_metrics.csv")
    pair = pd.read_csv(
        run_dir / "tables" / "clean_vs_attack_source_target.csv"
    )
    overall = pd.read_csv(
        run_dir / "tables" / "clean_vs_attack_overall.csv"
    )
    per_class = pd.read_csv(
        run_dir / "tables" / "clean_vs_attack_per_class.csv"
    )

    best_round = int(metadata["best_validation_round"])
    best_row = rounds[rounds["round"] == best_round].iloc[0]
    natural_pair = pair[pair["split"] == "test_natural"].iloc[0]
    diagnostic_pair = pair[pair["split"] == "test_diagnostic"].iloc[0]
    natural_overall = overall[overall["split"] == "test_natural"].iloc[0]
    diagnostic_overall = overall[
        overall["split"] == "test_diagnostic"
    ].iloc[0]

    source_class_rows = per_class[
        (per_class["split"] == "test_natural")
        & (per_class["class_name"] == metadata["source_class"])
    ]
    if len(source_class_rows) != 1:
        raise ValueError(
            f"Expected one natural source-class row for seed {seed}, "
            f"found {len(source_class_rows)}."
        )
    source_row = source_class_rows.iloc[0]

    return (
        {
            "strength": setting["strength"],
            "seed": seed,
            "malicious_client_fraction": setting[
                "malicious_client_fraction"
            ],
            "poison_fraction": setting["poison_fraction"],
            "malicious_clients": "|".join(
                map(str, setting["malicious_clients"])
            ),
            "global_source_exposure_fraction": float(
                attack_summary["global_source_exposure_fraction"]
            ),
            "global_flipped_rows": int(attack_summary["global_flipped_rows"]),
            "best_validation_round": best_round,
            "validation_macro_f1_attacked": float(best_row["val_macro_f1"]),
            "validation_source_to_target_rate_attacked": float(
                best_row["val_source_to_target_rate"]
            ),
            "validation_source_recall_attacked": float(
                best_row["val_source_recall"]
            ),
            "natural_source_to_target_rate_clean": float(
                natural_pair["source_to_target_rate_clean"]
            ),
            "natural_source_to_target_rate_attacked": float(
                natural_pair["source_to_target_rate_attacked"]
            ),
            "natural_source_to_target_rate_delta": float(
                natural_pair[
                    "source_to_target_rate_delta_attacked_minus_clean"
                ]
            ),
            "natural_source_to_target_fold_change": float(
                natural_pair["source_to_target_fold_change"]
            ),
            "diagnostic_source_to_target_rate_clean": float(
                diagnostic_pair["source_to_target_rate_clean"]
            ),
            "diagnostic_source_to_target_rate_attacked": float(
                diagnostic_pair["source_to_target_rate_attacked"]
            ),
            "diagnostic_source_to_target_rate_delta": float(
                diagnostic_pair[
                    "source_to_target_rate_delta_attacked_minus_clean"
                ]
            ),
            "natural_macro_f1_clean": float(
                natural_overall["macro_f1_clean"]
            ),
            "natural_macro_f1_attacked": float(
                natural_overall["macro_f1_attacked"]
            ),
            "natural_macro_f1_delta": float(
                natural_overall[
                    "macro_f1_delta_attacked_minus_clean"
                ]
            ),
            "diagnostic_macro_f1_clean": float(
                diagnostic_overall["macro_f1_clean"]
            ),
            "diagnostic_macro_f1_attacked": float(
                diagnostic_overall["macro_f1_attacked"]
            ),
            "diagnostic_macro_f1_delta": float(
                diagnostic_overall[
                    "macro_f1_delta_attacked_minus_clean"
                ]
            ),
            "natural_source_recall_clean": float(source_row["recall_clean"]),
            "natural_source_recall_attacked": float(
                source_row["recall_attacked"]
            ),
            "natural_source_recall_drop": float(
                source_row["recall_clean"] - source_row["recall_attacked"]
            ),
            "rounds_completed": int(metadata["rounds_completed"]),
            "total_seconds": float(metadata["total_seconds"]),
            "partition_hash_sha256": metadata["partition_hash_sha256"],
            "poison_index_hash_sha256": metadata["poison_index_hash_sha256"],
        },
        rounds.assign(strength=setting["strength"], seed=seed),
        per_class.assign(strength=setting["strength"], seed=seed),
    )


def ci95(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        return 0.0
    critical = float(student_t.ppf(0.975, df=len(clean) - 1))
    return float(critical * clean.std(ddof=1) / math.sqrt(len(clean)))


def aggregate_table(seed_table: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "best_validation_round",
        "validation_macro_f1_attacked",
        "validation_source_to_target_rate_attacked",
        "validation_source_recall_attacked",
        "natural_source_to_target_rate_clean",
        "natural_source_to_target_rate_attacked",
        "natural_source_to_target_rate_delta",
        "natural_source_to_target_fold_change",
        "diagnostic_source_to_target_rate_clean",
        "diagnostic_source_to_target_rate_attacked",
        "diagnostic_source_to_target_rate_delta",
        "natural_macro_f1_clean",
        "natural_macro_f1_attacked",
        "natural_macro_f1_delta",
        "diagnostic_macro_f1_clean",
        "diagnostic_macro_f1_attacked",
        "diagnostic_macro_f1_delta",
        "natural_source_recall_clean",
        "natural_source_recall_attacked",
        "natural_source_recall_drop",
        "total_seconds",
    ]
    rows = []
    for strength, group in seed_table.groupby("strength", sort=False):
        for metric in metrics:
            values = group[metric].astype(float)
            rows.append(
                {
                    "strength": strength,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "median": float(values.median()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "ci95_half_width_t": ci95(values),
                }
            )
    return pd.DataFrame(rows)


def paired_cohen_dz(delta: pd.Series) -> float:
    values = delta.astype(float)
    std = float(values.std(ddof=1))
    if std == 0:
        return math.inf if float(values.mean()) != 0 else 0.0
    return float(values.mean() / std)


def safe_wilcoxon(
    clean: pd.Series,
    attacked: pd.Series,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    differences = attacked.astype(float) - clean.astype(float)
    if np.allclose(differences.to_numpy(), 0.0):
        return 0.0, 1.0
    result = wilcoxon(
        attacked.astype(float),
        clean.astype(float),
        alternative=alternative,
        zero_method="wilcox",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def paired_clean_attack_tests(seed_table: pd.DataFrame) -> pd.DataFrame:
    specs = [
        (
            "natural_source_to_target_rate",
            "natural_source_to_target_rate_clean",
            "natural_source_to_target_rate_attacked",
        ),
        (
            "natural_macro_f1",
            "natural_macro_f1_clean",
            "natural_macro_f1_attacked",
        ),
        (
            "natural_source_recall",
            "natural_source_recall_clean",
            "natural_source_recall_attacked",
        ),
        (
            "diagnostic_source_to_target_rate",
            "diagnostic_source_to_target_rate_clean",
            "diagnostic_source_to_target_rate_attacked",
        ),
        (
            "diagnostic_macro_f1",
            "diagnostic_macro_f1_clean",
            "diagnostic_macro_f1_attacked",
        ),
    ]
    rows = []
    for strength, group in seed_table.groupby("strength", sort=False):
        for name, clean_col, attacked_col in specs:
            statistic, pvalue = safe_wilcoxon(
                group[clean_col],
                group[attacked_col],
            )
            delta = group[attacked_col] - group[clean_col]
            rows.append(
                {
                    "strength": strength,
                    "metric": name,
                    "n_pairs": len(group),
                    "clean_mean": float(group[clean_col].mean()),
                    "attacked_mean": float(group[attacked_col].mean()),
                    "mean_delta_attacked_minus_clean": float(delta.mean()),
                    "std_delta": float(delta.std(ddof=1)),
                    "ci95_delta_half_width_t": ci95(delta),
                    "paired_cohen_dz": paired_cohen_dz(delta),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_two_sided": pvalue,
                }
            )
    return pd.DataFrame(rows)


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        candidate = (m - rank) * pvalues[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def strength_comparison_tests(seed_table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "natural_source_to_target_rate_attacked",
        "natural_source_to_target_rate_delta",
        "natural_macro_f1_attacked",
        "natural_source_recall_attacked",
    ]
    friedman_rows = []
    pairwise_rows = []

    for metric in metrics:
        pivot = seed_table.pivot(index="seed", columns="strength", values=metric)
        pivot = pivot[STRENGTH_ORDER]
        stat, pvalue = friedmanchisquare(
            *[pivot[column].to_numpy() for column in STRENGTH_ORDER]
        )
        friedman_rows.append(
            {
                "metric": metric,
                "n_seeds": len(pivot),
                "friedman_chi_square": float(stat),
                "friedman_p": float(pvalue),
            }
        )

        comparisons = [("weak", "moderate"), ("weak", "strong"), ("moderate", "strong")]
        local_rows = []
        local_pvalues = []
        for left, right in comparisons:
            result = wilcoxon(
                pivot[right],
                pivot[left],
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            delta = pivot[right] - pivot[left]
            local_rows.append(
                {
                    "metric": metric,
                    "left_strength": left,
                    "right_strength": right,
                    "mean_right_minus_left": float(delta.mean()),
                    "paired_cohen_dz": paired_cohen_dz(delta),
                    "wilcoxon_statistic": float(result.statistic),
                    "wilcoxon_p_raw": float(result.pvalue),
                }
            )
            local_pvalues.append(float(result.pvalue))

        adjusted = holm_adjust(local_pvalues)
        for row, adjusted_p in zip(local_rows, adjusted):
            row["wilcoxon_p_holm"] = adjusted_p
            pairwise_rows.append(row)

    return pd.DataFrame(friedman_rows), pd.DataFrame(pairwise_rows)


def plot_attack_rate(seed_table: pd.DataFrame, output: Path) -> None:
    summary = seed_table.groupby("strength", sort=False).agg(
        clean_mean=("natural_source_to_target_rate_clean", "mean"),
        attacked_mean=("natural_source_to_target_rate_attacked", "mean"),
    ).reindex(STRENGTH_ORDER)
    summary["attacked_ci"] = [
        ci95(
            seed_table.loc[
                seed_table["strength"] == strength,
                "natural_source_to_target_rate_attacked",
            ]
        )
        for strength in STRENGTH_ORDER
    ]

    x = np.arange(len(STRENGTH_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(
        x - width / 2,
        summary["clean_mean"],
        width,
        label="Clean",
    )
    ax.bar(
        x + width / 2,
        summary["attacked_mean"],
        width,
        yerr=summary["attacked_ci"],
        capsize=5,
        label="Attacked",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(STRENGTH_ORDER)
    ax.set_ylabel("DDoS predicted as Benign")
    ax.set_title("Paired Five-Seed DDoS-to-Benign Attack Reliability")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def plot_macro_f1(seed_table: pd.DataFrame, output: Path) -> None:
    summary = seed_table.groupby("strength", sort=False).agg(
        clean_mean=("natural_macro_f1_clean", "mean"),
        attacked_mean=("natural_macro_f1_attacked", "mean"),
    ).reindex(STRENGTH_ORDER)
    summary["attacked_ci"] = [
        ci95(
            seed_table.loc[
                seed_table["strength"] == strength,
                "natural_macro_f1_attacked",
            ]
        )
        for strength in STRENGTH_ORDER
    ]

    x = np.arange(len(STRENGTH_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, summary["clean_mean"], width, label="Clean")
    ax.bar(
        x + width / 2,
        summary["attacked_mean"],
        width,
        yerr=summary["attacked_ci"],
        capsize=5,
        label="Attacked",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(STRENGTH_ORDER)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Natural-test macro F1")
    ax.set_title("Clean Utility Under Frozen Attack Strengths")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def plot_source_recall(seed_table: pd.DataFrame, output: Path) -> None:
    summary = seed_table.groupby("strength", sort=False).agg(
        clean_mean=("natural_source_recall_clean", "mean"),
        attacked_mean=("natural_source_recall_attacked", "mean"),
    ).reindex(STRENGTH_ORDER)
    summary["attacked_ci"] = [
        ci95(
            seed_table.loc[
                seed_table["strength"] == strength,
                "natural_source_recall_attacked",
            ]
        )
        for strength in STRENGTH_ORDER
    ]

    x = np.arange(len(STRENGTH_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, summary["clean_mean"], width, label="Clean")
    ax.bar(
        x + width / 2,
        summary["attacked_mean"],
        width,
        yerr=summary["attacked_ci"],
        capsize=5,
        label="Attacked",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(STRENGTH_ORDER)
    ax.set_ylim(0, 1)
    ax.set_ylabel("DDoS recall")
    ax.set_title("DDoS Recall Degradation Across Attack Strengths")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def plot_seed_lines(seed_table: pd.DataFrame, output: Path) -> None:
    pivot = seed_table.pivot(
        index="seed",
        columns="strength",
        values="natural_source_to_target_rate_attacked",
    )[STRENGTH_ORDER]
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(STRENGTH_ORDER))
    for seed, row in pivot.iterrows():
        ax.plot(x, row.to_numpy(), marker="o", label=str(seed))
    ax.set_xticks(x)
    ax.set_xticklabels(STRENGTH_ORDER)
    ax.set_ylabel("Natural DDoS-to-Benign rate")
    ax.set_title("Per-Seed Attack Strength Response")
    ax.grid(alpha=0.25)
    ax.legend(title="Seed")
    save_figure(fig, output)


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    output_dir = args.output_dir.expanduser().resolve()
    runs_dir = output_dir / "runs"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for path in (runs_dir, tables_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    runner_script = args.runner_script.expanduser().resolve()
    clean_multiseed_dir = args.clean_multiseed_dir.expanduser().resolve()
    v32_results_dir = args.v32_results_dir.expanduser().resolve()
    if not runner_script.exists():
        raise FileNotFoundError(f"Targeted attack runner not found: {runner_script}")

    frozen = validate_frozen_settings(v32_results_dir)
    frozen.to_csv(tables_dir / "frozen_attack_settings.csv", index=False)

    seed_rows: List[Dict[str, object]] = []
    round_tables = []
    per_class_tables = []

    for setting in SETTINGS:
        for seed in seeds:
            clean_seed_dir = clean_multiseed_dir / "seed_runs" / f"seed_{seed}"
            if not clean_seed_dir.exists():
                raise FileNotFoundError(
                    f"Clean reference for seed {seed} not found: {clean_seed_dir}"
                )
            run_dir = (
                runs_dir
                / setting["strength"]
                / f"seed_{seed}"
            )
            execute_run(
                args=args,
                runner_script=runner_script,
                clean_seed_dir=clean_seed_dir,
                run_dir=run_dir,
                setting=setting,
                seed=seed,
            )
            row, rounds, per_class = collect_run(
                run_dir=run_dir,
                setting=setting,
                seed=seed,
            )
            seed_rows.append(row)
            round_tables.append(rounds)
            per_class_tables.append(per_class)

            pd.DataFrame(seed_rows).to_csv(
                tables_dir / "seed_level_attack_results_partial.csv",
                index=False,
            )

    seed_table = pd.DataFrame(seed_rows)
    seed_table["strength"] = pd.Categorical(
        seed_table["strength"],
        categories=STRENGTH_ORDER,
        ordered=True,
    )
    seed_table = seed_table.sort_values(["strength", "seed"])
    seed_table.to_csv(tables_dir / "seed_level_attack_results.csv", index=False)

    rounds_all = pd.concat(round_tables, ignore_index=True)
    per_class_all = pd.concat(per_class_tables, ignore_index=True)
    rounds_all.to_csv(tables_dir / "round_metrics_all_runs.csv", index=False)
    per_class_all.to_csv(
        tables_dir / "per_class_clean_vs_attack_all_runs.csv",
        index=False,
    )

    hashes = seed_table.groupby("strength")["poison_index_hash_sha256"].nunique()
    if not (hashes == 1).all():
        raise RuntimeError(
            "Poisoned-row hash changed across model seeds for at least one strength."
        )
    partition_hashes = seed_table["partition_hash_sha256"].unique()
    if len(partition_hashes) != 1:
        raise RuntimeError("Partition hash changed across runs.")

    aggregate = aggregate_table(seed_table)
    aggregate.to_csv(
        tables_dir / "aggregate_mean_std_ci.csv",
        index=False,
    )
    paired_tests = paired_clean_attack_tests(seed_table)
    paired_tests.to_csv(
        tables_dir / "paired_clean_vs_attack_tests.csv",
        index=False,
    )
    friedman, pairwise = strength_comparison_tests(seed_table)
    friedman.to_csv(
        tables_dir / "attack_strength_friedman_tests.csv",
        index=False,
    )
    pairwise.to_csv(
        tables_dir / "attack_strength_pairwise_wilcoxon_holm.csv",
        index=False,
    )

    coverage = seed_table[
        [
            "strength",
            "seed",
            "malicious_client_fraction",
            "poison_fraction",
            "malicious_clients",
            "global_source_exposure_fraction",
            "best_validation_round",
            "partition_hash_sha256",
            "poison_index_hash_sha256",
        ]
    ].copy()
    coverage["completed"] = True
    coverage.to_csv(
        tables_dir / "coverage_and_reproducibility.csv",
        index=False,
    )

    plot_attack_rate(
        seed_table,
        figures_dir / "natural_attack_rate_mean_ci",
    )
    plot_macro_f1(
        seed_table,
        figures_dir / "natural_macro_f1_mean_ci",
    )
    plot_source_recall(
        seed_table,
        figures_dir / "natural_ddos_recall_mean_ci",
    )
    plot_seed_lines(
        seed_table,
        figures_dir / "per_seed_attack_strength_response",
    )

    metadata = {
        "experiment_version": "3.3",
        "experiment_type": (
            "paired_fixed_partition_multiseed_ddos_to_benign_attack_reliability"
        ),
        "source_class": args.source_class,
        "target_class": args.target_class,
        "seeds": seeds,
        "model_seed_varied": True,
        "attack_seed": args.attack_seed,
        "attack_seed_fixed_across_model_seeds": True,
        "partition_fixed_across_runs": True,
        "partition_hash_sha256": str(partition_hashes[0]),
        "settings_frozen_from_v32_validation_only": SETTINGS,
        "weak_setting": SETTINGS[0],
        "moderate_setting": SETTINGS[1],
        "strong_setting": SETTINGS[2],
        "checkpoint_selection": (
            "Best clean-validation macro F1 independently within each paired seed."
        ),
        "tests_used_for_setting_selection": False,
        "confidence_interval": "Two-sided 95% Student t interval across five seeds.",
        "statistical_tests": [
            "Paired two-sided Wilcoxon clean versus attacked within each strength.",
            "Friedman test across weak, moderate, and strong attacked results.",
            "Pairwise two-sided Wilcoxon tests with Holm correction.",
            "Paired Cohen dz effect size.",
        ],
        "notes": [
            "The malicious-client IDs are fixed within each strength.",
            "The poisoned-row hash is verified to be identical across model seeds.",
            "This stage freezes the attack benchmark before any defense evaluation.",
            "The next stage is the original LFighter baseline under these same paired attacks.",
        ],
    }
    with (output_dir / "ddos_benign_multiseed_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("DDoS-to-Benign Multi-Seed V3.3 complete")
    print()
    print("Coverage")
    print(
        coverage[
            [
                "strength",
                "seed",
                "global_source_exposure_fraction",
                "best_validation_round",
                "completed",
            ]
        ].to_string(index=False)
    )
    print()
    print("Aggregate natural attack rates")
    print(
        aggregate[
            aggregate["metric"].isin(
                [
                    "natural_source_to_target_rate_clean",
                    "natural_source_to_target_rate_attacked",
                    "natural_source_to_target_rate_delta",
                    "natural_macro_f1_attacked",
                    "natural_source_recall_attacked",
                ]
            )
        ].to_string(index=False)
    )
    print()
    print("Paired clean-versus-attack tests")
    print(paired_tests.to_string(index=False))
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("Metadata:", output_dir / "ddos_benign_multiseed_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

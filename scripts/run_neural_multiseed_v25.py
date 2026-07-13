#!/usr/bin/env python3
"""Five-seed stability evaluation for the strongest CIC IoT-DIAD neural candidates.

This script runs the existing V2.4 neural screening script for each seed, aggregates
validation and test metrics, selects the winner using validation metrics only, and
exports CSV, JSON, PNG, and PDF research artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed stability evaluation for CIC IoT-DIAD neural candidates."
    )
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--candidates",
        default="resmlp_ce,resmlp_sqrt,resmlp_focal",
        help="Comma-separated V2.4 candidate names.",
    )
    parser.add_argument(
        "--seeds",
        default="42,123,2026,7,99",
        help="Comma-separated random seeds.",
    )
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--runner-script",
        type=Path,
        default=Path("scripts/screen_neural_iot_diad_v24.py"),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse a seed folder when its all_runs.csv already exists.",
    )
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def ci95(series: pd.Series) -> float:
    values = series.dropna().astype(float)
    if len(values) < 2:
        return 0.0
    return float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))


def run_seed(
    python_executable: str,
    runner_script: Path,
    data_file: Path,
    seed_output: Path,
    candidates: List[str],
    seed: int,
    threads: int,
    epochs: int,
    patience: int,
    batch_size: int,
    skip_existing: bool,
) -> None:
    expected = seed_output / "tables" / "all_runs.csv"
    if skip_existing and expected.exists():
        print(f"Reusing completed seed {seed}: {expected}")
        return

    seed_output.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        str(runner_script),
        "--data-file",
        str(data_file),
        "--output-dir",
        str(seed_output),
        "--candidates",
        ",".join(candidates),
        "--seed",
        str(seed),
        "--threads",
        str(threads),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--batch-size",
        str(batch_size),
    ]
    print()
    print("Running seed", seed)
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    subprocess.run(command, check=True)


def aggregate_seed_outputs(
    output_dir: Path,
    seeds: List[int],
) -> Dict[str, pd.DataFrame]:
    all_runs = []
    per_class = []
    rankings = []
    histories = []
    prior_estimates = []

    for seed in seeds:
        seed_root = output_dir / "seed_runs" / f"seed_{seed}"
        tables = seed_root / "tables"

        run_table = pd.read_csv(tables / "all_runs.csv")
        run_table["source_seed_folder"] = str(seed_root)
        all_runs.append(run_table)

        class_table = pd.read_csv(tables / "per_class_metrics.csv")
        class_table["source_seed_folder"] = str(seed_root)
        per_class.append(class_table)

        ranking_table = pd.read_csv(tables / "candidate_ranking_validation_only.csv")
        ranking_table["source_seed_folder"] = str(seed_root)
        rankings.append(ranking_table)

        history_table = pd.read_csv(tables / "training_history.csv")
        history_table["seed"] = seed
        histories.append(history_table)

        prior_path = tables / "prior_estimates.csv"
        if prior_path.exists():
            prior_table = pd.read_csv(prior_path)
            prior_table["seed"] = seed
            prior_estimates.append(prior_table)

    result = {
        "all_runs": pd.concat(all_runs, ignore_index=True),
        "per_class": pd.concat(per_class, ignore_index=True),
        "rankings": pd.concat(rankings, ignore_index=True),
        "histories": pd.concat(histories, ignore_index=True),
    }
    if prior_estimates:
        result["prior_estimates"] = pd.concat(prior_estimates, ignore_index=True)
    return result


def metric_summary(all_runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "log_loss",
        "ece_15bin",
        "train_seconds",
        "model_size_mb",
    ]
    rows = []
    for (candidate, split), group in all_runs.groupby(["candidate", "split"]):
        for metric in metrics:
            values = group[metric].astype(float)
            rows.append(
                {
                    "candidate": candidate,
                    "split": split,
                    "metric": metric,
                    "n_seeds": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(values.median()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "ci95_half_width": ci95(values),
                }
            )
    return pd.DataFrame(rows)


def validation_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    validation = summary[summary["split"] == "val"].copy()
    pivot = validation.pivot(
        index="candidate",
        columns="metric",
        values="mean",
    ).reset_index()

    required = ["macro_f1", "balanced_accuracy", "log_loss"]
    missing = [name for name in required if name not in pivot.columns]
    if missing:
        raise ValueError(f"Missing validation ranking metrics: {missing}")

    pivot = pivot.sort_values(
        ["macro_f1", "balanced_accuracy", "log_loss"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    pivot.insert(0, "validation_rank", np.arange(1, len(pivot) + 1))
    pivot = pivot.rename(
        columns={
            "macro_f1": "validation_macro_f1_mean",
            "balanced_accuracy": "validation_balanced_accuracy_mean",
            "log_loss": "validation_log_loss_mean",
        }
    )
    return pivot


def per_class_summary(per_class: pd.DataFrame) -> pd.DataFrame:
    metrics = ["precision", "recall", "f1_score"]
    rows = []
    for keys, group in per_class.groupby(
        ["candidate", "split", "class_id", "class_name"]
    ):
        candidate, split, class_id, class_name = keys
        support = int(group["support"].iloc[0])
        for metric in metrics:
            values = group[metric].astype(float)
            rows.append(
                {
                    "candidate": candidate,
                    "split": split,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "support_per_seed": support,
                    "metric": metric,
                    "n_seeds": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_half_width": ci95(values),
                }
            )
    return pd.DataFrame(rows)


def paired_tests(all_runs: pd.DataFrame, candidates: List[str]) -> pd.DataFrame:
    validation = all_runs[all_runs["split"] == "val"][
        ["candidate", "seed", "macro_f1"]
    ]
    pivot = validation.pivot(index="seed", columns="candidate", values="macro_f1")
    rows = []

    if len(candidates) >= 3 and all(candidate in pivot.columns for candidate in candidates):
        arrays = [pivot[candidate].dropna().to_numpy() for candidate in candidates]
        common_length = min(len(values) for values in arrays)
        if common_length >= 2:
            arrays = [values[:common_length] for values in arrays]
            statistic, p_value = friedmanchisquare(*arrays)
            rows.append(
                {
                    "test": "friedman_validation_macro_f1",
                    "candidate_a": "|".join(candidates),
                    "candidate_b": "",
                    "n_pairs": common_length,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "note": "Overall paired comparison across candidates.",
                }
            )

    for index, candidate_a in enumerate(candidates):
        for candidate_b in candidates[index + 1 :]:
            if candidate_a not in pivot.columns or candidate_b not in pivot.columns:
                continue
            paired = pivot[[candidate_a, candidate_b]].dropna()
            if len(paired) < 2:
                continue
            differences = paired[candidate_a] - paired[candidate_b]
            if np.allclose(differences.to_numpy(), 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                statistic, p_value = wilcoxon(
                    paired[candidate_a],
                    paired[candidate_b],
                    alternative="two-sided",
                    zero_method="wilcox",
                    mode="auto",
                )
            rows.append(
                {
                    "test": "wilcoxon_validation_macro_f1",
                    "candidate_a": candidate_a,
                    "candidate_b": candidate_b,
                    "n_pairs": len(paired),
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "mean_difference_a_minus_b": float(differences.mean()),
                    "note": "Exploratory with five seeds, report effect size and uncertainty.",
                }
            )
    return pd.DataFrame(rows)


def plot_validation_macro_f1(all_runs: pd.DataFrame, output: Path) -> None:
    data = all_runs[all_runs["split"] == "val"]
    summary = (
        data.groupby("candidate", as_index=False)["macro_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["ci95"] = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(
        summary["candidate"],
        summary["mean"],
        yerr=summary["ci95"],
        capsize=5,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation macro F1")
    ax.set_xlabel("Candidate")
    ax.set_title("Five-Seed Validation Reliability")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output)


def plot_seed_stability(all_runs: pd.DataFrame, output: Path) -> None:
    data = all_runs[all_runs["split"] == "val"].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    for candidate, group in data.groupby("candidate"):
        group = group.sort_values("seed")
        ax.plot(group["seed"].astype(str), group["macro_f1"], marker="o", label=candidate)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation macro F1")
    ax.set_xlabel("Seed")
    ax.set_title("Seed-to-Seed Stability of Neural Candidates")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def plot_selected_split_performance(
    all_runs: pd.DataFrame,
    selected_candidate: str,
    output: Path,
) -> None:
    data = all_runs[all_runs["candidate"] == selected_candidate]
    preferred_splits = ["val", "test_diagnostic", "test_natural_raw"]
    data = data[data["split"].isin(preferred_splits)]
    summary = (
        data.groupby("split", as_index=False)["macro_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["ci95"] = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"])
    order = {name: index for index, name in enumerate(preferred_splits)}
    summary["order"] = summary["split"].map(order)
    summary = summary.sort_values("order")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        summary["split"],
        summary["mean"],
        yerr=summary["ci95"],
        capsize=5,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1")
    ax.set_xlabel("Evaluation split")
    ax.set_title(f"Selected Candidate Across Leakage-Safe Splits, {selected_candidate}")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output)


def plot_selected_per_class(
    per_class_summary_table: pd.DataFrame,
    selected_candidate: str,
    split: str,
    output: Path,
) -> None:
    data = per_class_summary_table[
        (per_class_summary_table["candidate"] == selected_candidate)
        & (per_class_summary_table["split"] == split)
        & (per_class_summary_table["metric"] == "f1_score")
    ].sort_values("class_id")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        data["class_name"],
        data["mean"],
        yerr=data["ci95_half_width"],
        capsize=4,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Per-class F1")
    ax.set_xlabel("Class")
    ax.set_title(f"Five-Seed Per-Class Reliability, {selected_candidate}, {split}")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output)


def plot_runtime_reliability(all_runs: pd.DataFrame, output: Path) -> None:
    data = all_runs[all_runs["split"] == "val"]
    summary = data.groupby("candidate", as_index=False).agg(
        mean_train_seconds=("train_seconds", "mean"),
        mean_validation_macro_f1=("macro_f1", "mean"),
        std_validation_macro_f1=("macro_f1", "std"),
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(summary["mean_train_seconds"], summary["mean_validation_macro_f1"], s=80)
    for _, row in summary.iterrows():
        ax.annotate(
            str(row["candidate"]),
            (float(row["mean_train_seconds"]), float(row["mean_validation_macro_f1"])),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Mean training time, seconds")
    ax.set_ylabel("Mean validation macro F1")
    ax.set_ylim(0, 1)
    ax.set_title("Runtime and Reliability Tradeoff")
    ax.grid(alpha=0.25)
    save_figure(fig, output)


def main() -> int:
    args = parse_args()
    data_file = args.data_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    runner_script = args.runner_script.expanduser().resolve()

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not runner_script.exists():
        raise FileNotFoundError(f"V2.4 runner script not found: {runner_script}")

    candidates = [item.strip() for item in args.candidates.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if len(candidates) < 2:
        raise ValueError("At least two candidates are required.")
    if len(seeds) < 3:
        raise ValueError("At least three seeds are required.")

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        seed_output = output_dir / "seed_runs" / f"seed_{seed}"
        run_seed(
            python_executable=sys.executable,
            runner_script=runner_script,
            data_file=data_file,
            seed_output=seed_output,
            candidates=candidates,
            seed=seed,
            threads=args.threads,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing,
        )

    aggregated = aggregate_seed_outputs(output_dir, seeds)

    # Only aggregate the candidates explicitly requested for this V2.5 run.
    # A reused seed folder may contain extra candidates from an earlier screening
    # experiment, and including them would create an unequal-seed comparison.
    all_runs = aggregated["all_runs"]
    all_runs = all_runs[all_runs["candidate"].isin(candidates)].copy()

    per_class = aggregated["per_class"]
    per_class = per_class[per_class["candidate"].isin(candidates)].copy()

    rankings = aggregated["rankings"]
    rankings = rankings[rankings["candidate"].isin(candidates)].copy()

    histories = aggregated["histories"]
    histories = histories[histories["candidate"].isin(candidates)].copy()

    prior_estimates = None
    if "prior_estimates" in aggregated:
        prior_estimates = aggregated["prior_estimates"]
        if "candidate" in prior_estimates.columns:
            prior_estimates = prior_estimates[
                prior_estimates["candidate"].isin(candidates)
            ].copy()

    # Hard validation: every requested candidate must have all requested seeds.
    seed_coverage = (
        all_runs[all_runs["split"] == "val"]
        .groupby("candidate")["seed"]
        .nunique()
        .reindex(candidates, fill_value=0)
    )
    incomplete = seed_coverage[seed_coverage != len(seeds)]
    seed_coverage.reset_index(name="unique_validation_seeds").to_csv(
        tables_dir / "candidate_seed_coverage.csv",
        index=False,
    )
    if not incomplete.empty:
        details = ", ".join(
            f"{candidate}={count}/{len(seeds)}"
            for candidate, count in incomplete.items()
        )
        raise ValueError(
            "Unequal or incomplete seed coverage detected: " + details
        )

    all_runs.to_csv(tables_dir / "all_runs_all_seeds.csv", index=False)
    per_class.to_csv(tables_dir / "per_class_metrics_all_seeds.csv", index=False)
    rankings.to_csv(tables_dir / "seedwise_validation_rankings.csv", index=False)
    histories.to_csv(tables_dir / "training_history_all_seeds.csv", index=False)
    if prior_estimates is not None:
        prior_estimates.to_csv(
            tables_dir / "prior_estimates_all_seeds.csv", index=False
        )

    summary = metric_summary(all_runs)
    summary.to_csv(tables_dir / "aggregate_metrics_mean_std_ci.csv", index=False)

    ranking = validation_ranking(summary)
    ranking.to_csv(tables_dir / "final_validation_ranking.csv", index=False)
    selected_candidate = str(ranking.iloc[0]["candidate"])

    class_summary = per_class_summary(per_class)
    class_summary.to_csv(tables_dir / "per_class_mean_std_ci.csv", index=False)

    tests = paired_tests(all_runs, candidates)
    tests.to_csv(tables_dir / "paired_statistical_tests.csv", index=False)

    selection_evidence = summary[
        (summary["candidate"] == selected_candidate)
        & (summary["split"].isin(["val", "test_diagnostic", "test_natural_raw"]))
        & (summary["metric"].isin(["accuracy", "balanced_accuracy", "macro_f1", "mcc", "log_loss"]))
    ].copy()
    selection_evidence.to_csv(
        tables_dir / "selected_candidate_summary.csv",
        index=False,
    )

    plot_validation_macro_f1(all_runs, figures_dir / "validation_macro_f1_mean_ci")
    plot_seed_stability(all_runs, figures_dir / "validation_seed_stability")
    plot_selected_split_performance(
        all_runs,
        selected_candidate,
        figures_dir / "selected_candidate_split_performance",
    )
    plot_selected_per_class(
        class_summary,
        selected_candidate,
        "test_natural_raw",
        figures_dir / "selected_candidate_natural_per_class_f1",
    )
    plot_selected_per_class(
        class_summary,
        selected_candidate,
        "test_diagnostic",
        figures_dir / "selected_candidate_diagnostic_per_class_f1",
    )
    plot_runtime_reliability(all_runs, figures_dir / "runtime_reliability_tradeoff")

    metadata = {
        "evaluation_version": "2.5.1",
        "selection_rule": (
            "Highest mean validation macro F1, then mean validation balanced accuracy, "
            "then lowest mean validation log loss."
        ),
        "test_sets_used_for_model_selection": False,
        "data_file": str(data_file),
        "runner_script": str(runner_script),
        "candidates": candidates,
        "seeds": seeds,
        "selected_candidate_validation_only": selected_candidate,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "tables": str(tables_dir),
        "figures_png_pdf": str(figures_dir),
        "notes": [
            "Natural EM results are retained only as exploratory outputs from V2.4.",
            "The primary natural result is test_natural_raw.",
            "BruteForce has very low natural-test support and requires cautious interpretation.",
            "Five seeds provide uncertainty estimates but do not guarantee broad external validity.",
        ],
    }
    with (output_dir / "multiseed_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Neural Multi-Seed V2.5.1 complete")
    print()
    print("Validation-only ranking")
    print(ranking.to_string(index=False))
    print()
    print("Selected candidate:", selected_candidate)
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("Metadata:", output_dir / "multiseed_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

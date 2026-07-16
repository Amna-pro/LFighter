#!/usr/bin/env python3
"""Five-seed original-LFighter clean and strong-attack benchmark, V3.4."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, wilcoxon


DEFAULT_SEEDS = "42,123,2026,7,99"
DEFAULT_MALICIOUS_CLIENTS = "1,7,8,10,14,15,17,18"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired clean and strong-attack original LFighter over five seeds."
    )
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-multiseed-dir", required=True, type=Path)
    parser.add_argument("--fedavg-attack-v33-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--single-runner-script",
        type=Path,
        default=Path("scripts/run_lfighter_original_v34.py"),
    )
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--source-class", default="DDoS")
    parser.add_argument("--target-class", default="Benign")
    parser.add_argument("--malicious-clients", default=DEFAULT_MALICIOUS_CLIENTS)
    parser.add_argument("--poison-fraction", type=float, default=1.0)
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
    parser.add_argument("--kmeans-seed", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_seeds(text: str) -> List[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if len(values) < 3:
        raise ValueError("Use at least three model seeds.")
    if len(set(values)) != len(values):
        raise ValueError("Model seeds must be unique.")
    return values


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_complete(run_dir: Path) -> bool:
    required = [
        run_dir / "lfighter_v34_metadata.json",
        run_dir / "tables" / "round_metrics.csv",
        run_dir / "tables" / "client_decisions.csv",
        run_dir / "tables" / "class_salience_by_round.csv",
        run_dir / "tables" / "test_metrics.csv",
        run_dir / "tables" / "source_target_metrics.csv",
        run_dir / "checkpoints" / "best_validation_model.pt",
    ]
    return all(path.exists() for path in required)


def execute_run(
    args: argparse.Namespace,
    runner: Path,
    seed: int,
    mode: str,
    clean_seed_dir: Path,
    run_dir: Path,
) -> None:
    if args.skip_existing and run_complete(run_dir):
        print(f"Reusing completed V3.4 run: {mode}, seed {seed}")
        return
    if run_dir.exists():
        print(f"Removing incomplete V3.4 run: {mode}, seed {seed}")
        shutil.rmtree(run_dir)

    command = [
        sys.executable,
        str(runner),
        "--mode", mode,
        "--data-file", str(args.data_file.expanduser().resolve()),
        "--partition-file", str(args.partition_file.expanduser().resolve()),
        "--clean-seed-dir", str(clean_seed_dir),
        "--output-dir", str(run_dir),
        "--source-class", args.source_class,
        "--target-class", args.target_class,
        "--malicious-clients", args.malicious_clients,
        "--poison-fraction", str(args.poison_fraction),
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
        "--kmeans-seed", str(args.kmeans_seed),
    ]
    print("\n" + "=" * 90)
    print(f"Running original LFighter, mode={mode}, seed={seed}")
    print("=" * 90)
    subprocess.run(command, check=True)


def read_primary_tables(run_dir: Path) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, dict]:
    metrics = pd.read_csv(run_dir / "tables" / "test_metrics.csv")
    pair = pd.read_csv(run_dir / "tables" / "source_target_metrics.csv")
    rounds = pd.read_csv(run_dir / "tables" / "round_metrics.csv")
    per_class = pd.read_csv(run_dir / "tables" / "per_class_metrics.csv")
    with (run_dir / "lfighter_v34_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    natural_metrics = metrics[
        (metrics["checkpoint"] == "best_validation")
        & (metrics["split"] == "test_natural")
    ]
    natural_pair = pair[
        (pair["checkpoint"] == "best_validation")
        & (pair["split"] == "test_natural")
    ]
    if len(natural_metrics) != 1 or len(natural_pair) != 1:
        raise ValueError(f"Expected one primary natural row in {run_dir}")
    return natural_metrics.iloc[0], natural_pair.iloc[0], rounds, per_class, metadata


def ci95(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return float("nan")
    return float(student_t.ppf(0.975, df=len(arr) - 1) * np.std(arr, ddof=1) / math.sqrt(len(arr)))


def paired_cohen_dz(differences: np.ndarray) -> float:
    if len(differences) < 2:
        return float("nan")
    standard = float(np.std(differences, ddof=1))
    if standard == 0:
        if float(np.mean(differences)) == 0:
            return 0.0
        return math.copysign(float("inf"), float(np.mean(differences)))
    return float(np.mean(differences) / standard)


def safe_wilcoxon(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    differences = right - left
    if np.allclose(differences, 0.0):
        return 0.0, 1.0
    result = wilcoxon(left, right, alternative="two-sided", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue)


def aggregate_table(seed_table: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        values = seed_table[metric].astype(float).to_numpy()
        rows.append(
            {
                "metric": metric,
                "n_seeds": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "ci95_half_width_t": ci95(values),
            }
        )
    return pd.DataFrame(rows)


def paired_test_row(
    seed_table: pd.DataFrame,
    comparison: str,
    left_col: str,
    right_col: str,
) -> Dict[str, object]:
    left = seed_table[left_col].astype(float).to_numpy()
    right = seed_table[right_col].astype(float).to_numpy()
    differences = right - left
    statistic, pvalue = safe_wilcoxon(left, right)
    return {
        "comparison": comparison,
        "left_metric_column": left_col,
        "right_metric_column": right_col,
        "n_pairs": len(left),
        "left_mean": float(np.mean(left)),
        "right_mean": float(np.mean(right)),
        "mean_right_minus_left": float(np.mean(differences)),
        "std_difference": float(np.std(differences, ddof=1)),
        "ci95_difference_half_width_t": ci95(differences),
        "paired_cohen_dz": paired_cohen_dz(differences),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_two_sided": pvalue,
    }


def bar_with_ci(labels: Sequence[str], means: Sequence[float], cis: Sequence[float], ylabel: str, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=cis, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out)


def main() -> int:
    args = parse_args()
    if args.source_class != "DDoS" or args.target_class != "Benign":
        raise ValueError("V3.4 is frozen to the DDoS-to-Benign attack pair.")
    if not math.isclose(args.poison_fraction, 1.0):
        raise ValueError("V3.4 is frozen to 100% poisoning within malicious DDoS rows.")
    seeds = parse_seeds(args.seeds)
    output_dir = args.output_dir.expanduser().resolve()
    runs_dir = output_dir / "runs"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for path in (runs_dir, tables_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    runner = args.single_runner_script
    if not runner.is_absolute():
        runner = (Path.cwd() / runner).resolve()
    if not runner.exists():
        raise FileNotFoundError(f"Single-run V3.4 script not found: {runner}")

    clean_multiseed_dir = args.clean_multiseed_dir.expanduser().resolve()
    v33_dir = args.fedavg_attack_v33_dir.expanduser().resolve()
    clean_metrics_path = clean_multiseed_dir / "tables" / "final_test_metrics_all_seeds.csv"
    clean_pair_path = clean_multiseed_dir / "tables" / "attack_pair_baseline_all_seeds.csv"
    v33_seed_path = v33_dir / "tables" / "seed_level_attack_results.csv"
    for required in (clean_metrics_path, clean_pair_path, v33_seed_path):
        if not required.exists():
            raise FileNotFoundError(f"Required reference table not found: {required}")

    clean_fedavg_metrics = pd.read_csv(clean_metrics_path)
    clean_fedavg_pair = pd.read_csv(clean_pair_path)
    v33_seed = pd.read_csv(v33_seed_path)
    v33_strong = v33_seed[v33_seed["strength"] == "strong"].copy()

    rows: List[Dict[str, object]] = []
    all_rounds: List[pd.DataFrame] = []
    all_per_class: List[pd.DataFrame] = []
    coverage_rows: List[Dict[str, object]] = []

    for seed in seeds:
        clean_seed_dir = clean_multiseed_dir / "seed_runs" / f"seed_{seed}"
        if not clean_seed_dir.exists():
            raise FileNotFoundError(f"Clean FedAvg seed directory not found: {clean_seed_dir}")

        run_dirs = {
            "clean": runs_dir / "clean" / f"seed_{seed}",
            "strong_attack": runs_dir / "strong_attack" / f"seed_{seed}",
        }
        for mode, run_dir in run_dirs.items():
            execute_run(args, runner, seed, mode, clean_seed_dir, run_dir)

        clean_metric, clean_pair, clean_rounds, clean_class, clean_meta = read_primary_tables(
            run_dirs["clean"]
        )
        attack_metric, attack_pair, attack_rounds, attack_class, attack_meta = read_primary_tables(
            run_dirs["strong_attack"]
        )

        clean_rounds = clean_rounds.copy()
        clean_rounds.insert(0, "seed", seed)
        clean_rounds.insert(1, "mode", "clean")
        attack_rounds = attack_rounds.copy()
        attack_rounds.insert(0, "seed", seed)
        attack_rounds.insert(1, "mode", "strong_attack")
        all_rounds.extend([clean_rounds, attack_rounds])

        clean_class = clean_class[clean_class["checkpoint"] == "best_validation"].copy()
        attack_class = attack_class[attack_class["checkpoint"] == "best_validation"].copy()
        clean_class.insert(0, "seed", seed)
        clean_class.insert(1, "mode", "clean")
        attack_class.insert(0, "seed", seed)
        attack_class.insert(1, "mode", "strong_attack")
        all_per_class.extend([clean_class, attack_class])

        clean_fedavg_metric = clean_fedavg_metrics[
            (clean_fedavg_metrics["seed"] == seed)
            & (clean_fedavg_metrics["split"] == "test_natural")
        ]
        clean_fedavg_pair_row = clean_fedavg_pair[
            (clean_fedavg_pair["seed"] == seed)
            & (clean_fedavg_pair["split"] == "test_natural")
        ]
        fedavg_attack = v33_strong[v33_strong["seed"] == seed]
        if len(clean_fedavg_metric) != 1 or len(clean_fedavg_pair_row) != 1 or len(fedavg_attack) != 1:
            raise ValueError(f"Missing unique FedAvg reference for seed {seed}.")
        clean_fedavg_metric = clean_fedavg_metric.iloc[0]
        clean_fedavg_pair_row = clean_fedavg_pair_row.iloc[0]
        fedavg_attack = fedavg_attack.iloc[0]

        attack_best_round = int(attack_meta["best_validation_round"])
        clean_best_round = int(clean_meta["best_validation_round"])
        attack_best = attack_rounds[attack_rounds["round"] == attack_best_round].iloc[0]
        clean_best = clean_rounds[clean_rounds["round"] == clean_best_round].iloc[0]

        fedavg_attacked_rate = float(fedavg_attack["natural_source_to_target_rate_attacked"])
        lfighter_attacked_rate = float(attack_pair["source_to_target_rate"])
        absolute_reduction = fedavg_attacked_rate - lfighter_attacked_rate
        relative_reduction = absolute_reduction / max(fedavg_attacked_rate, 1e-12)

        row = {
            "seed": seed,
            "partition_hash_sha256": attack_meta["partition_hash_sha256"],
            "poison_index_hash_sha256": attack_meta["poison_index_hash_sha256"],
            "global_source_exposure_fraction": attack_meta["global_source_exposure_fraction"],
            "clean_lfighter_best_round": clean_best_round,
            "attack_lfighter_best_round": attack_best_round,
            "clean_fedavg_natural_macro_f1": float(clean_fedavg_metric["macro_f1"]),
            "clean_lfighter_natural_macro_f1": float(clean_metric["macro_f1"]),
            "clean_utility_cost_lfighter_minus_fedavg": float(clean_metric["macro_f1"] - clean_fedavg_metric["macro_f1"]),
            "attacked_fedavg_natural_macro_f1": float(fedavg_attack["natural_macro_f1_attacked"]),
            "attacked_lfighter_natural_macro_f1": float(attack_metric["macro_f1"]),
            "attacked_macro_f1_lfighter_minus_fedavg": float(attack_metric["macro_f1"] - fedavg_attack["natural_macro_f1_attacked"]),
            "clean_fedavg_natural_source_to_target_rate": float(clean_fedavg_pair_row["source_to_benign_rate"]),
            "clean_lfighter_natural_source_to_target_rate": float(clean_pair["source_to_target_rate"]),
            "attacked_fedavg_natural_source_to_target_rate": fedavg_attacked_rate,
            "attacked_lfighter_natural_source_to_target_rate": lfighter_attacked_rate,
            "absolute_attack_rate_reduction_fedavg_minus_lfighter": absolute_reduction,
            "relative_attack_rate_reduction_fedavg_minus_lfighter": relative_reduction,
            "lfighter_attack_induced_rate_delta": float(attack_pair["source_to_target_rate"] - clean_pair["source_to_target_rate"]),
            "clean_fedavg_natural_source_recall": float(fedavg_attack["natural_source_recall_clean"]),
            "attacked_fedavg_natural_source_recall": float(fedavg_attack["natural_source_recall_attacked"]),
            "clean_lfighter_natural_source_recall": float(clean_pair["source_recall"]),
            "attacked_lfighter_natural_source_recall": float(attack_pair["source_recall"]),
            "attacked_source_recall_gain_lfighter_minus_fedavg": float(attack_pair["source_recall"] - fedavg_attack["natural_source_recall_attacked"]),
            "attack_best_round_malicious_rejection_recall": float(attack_best["malicious_rejection_recall"]),
            "attack_best_round_benign_retention_rate": float(attack_best["benign_retention_rate"]),
            "attack_best_round_malicious_detection_precision": float(attack_best["malicious_detection_precision"]),
            "attack_best_round_malicious_detection_f1": float(attack_best["malicious_detection_f1"]),
            "attack_best_round_admitted_client_count": int(attack_best["admitted_client_count"]),
            "attack_best_round_pair_identified": bool(attack_best["source_target_pair_identified"]),
            "attack_mean_malicious_rejection_recall": float(attack_rounds["malicious_rejection_recall"].mean()),
            "attack_mean_benign_retention_rate": float(attack_rounds["benign_retention_rate"].mean()),
            "attack_pair_identification_rate_across_rounds": float(attack_rounds["source_target_pair_identified"].astype(float).mean()),
            "clean_mean_benign_retention_rate": float(clean_rounds["benign_retention_rate"].mean()),
            "clean_best_round_benign_retention_rate": float(clean_best["benign_retention_rate"]),
            "clean_rounds_completed": int(clean_meta["rounds_completed"]),
            "attack_rounds_completed": int(attack_meta["rounds_completed"]),
            "clean_total_seconds": float(clean_meta["total_seconds"]),
            "attack_total_seconds": float(attack_meta["total_seconds"]),
        }
        rows.append(row)
        coverage_rows.append(
            {
                "seed": seed,
                "clean_completed": run_complete(run_dirs["clean"]),
                "strong_attack_completed": run_complete(run_dirs["strong_attack"]),
                "partition_hash_sha256": attack_meta["partition_hash_sha256"],
                "poison_index_hash_sha256": attack_meta["poison_index_hash_sha256"],
                "malicious_clients": "|".join(map(str, attack_meta["malicious_clients"])),
                "global_source_exposure_fraction": attack_meta["global_source_exposure_fraction"],
            }
        )
        pd.DataFrame(rows).to_csv(tables_dir / "seed_level_defense_results_partial.csv", index=False)

    seed_table = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    seed_table.to_csv(tables_dir / "seed_level_defense_results.csv", index=False)
    pd.DataFrame(coverage_rows).sort_values("seed").to_csv(
        tables_dir / "coverage_and_reproducibility.csv", index=False
    )
    pd.concat(all_rounds, ignore_index=True).to_csv(
        tables_dir / "round_metrics_all_runs.csv", index=False
    )
    pd.concat(all_per_class, ignore_index=True).to_csv(
        tables_dir / "per_class_metrics_all_runs.csv", index=False
    )

    aggregate_metrics = [
        "clean_fedavg_natural_macro_f1",
        "clean_lfighter_natural_macro_f1",
        "clean_utility_cost_lfighter_minus_fedavg",
        "attacked_fedavg_natural_macro_f1",
        "attacked_lfighter_natural_macro_f1",
        "attacked_macro_f1_lfighter_minus_fedavg",
        "clean_fedavg_natural_source_to_target_rate",
        "clean_lfighter_natural_source_to_target_rate",
        "attacked_fedavg_natural_source_to_target_rate",
        "attacked_lfighter_natural_source_to_target_rate",
        "absolute_attack_rate_reduction_fedavg_minus_lfighter",
        "relative_attack_rate_reduction_fedavg_minus_lfighter",
        "lfighter_attack_induced_rate_delta",
        "attacked_source_recall_gain_lfighter_minus_fedavg",
        "attack_best_round_malicious_rejection_recall",
        "attack_best_round_benign_retention_rate",
        "attack_best_round_malicious_detection_precision",
        "attack_best_round_malicious_detection_f1",
        "attack_pair_identification_rate_across_rounds",
        "clean_mean_benign_retention_rate",
    ]
    aggregate = aggregate_table(seed_table, aggregate_metrics)
    aggregate.to_csv(tables_dir / "aggregate_mean_std_ci.csv", index=False)

    tests = pd.DataFrame(
        [
            paired_test_row(
                seed_table,
                "Attacked LFighter vs attacked FedAvg, natural DDoS-to-Benign rate",
                "attacked_fedavg_natural_source_to_target_rate",
                "attacked_lfighter_natural_source_to_target_rate",
            ),
            paired_test_row(
                seed_table,
                "Attacked LFighter vs attacked FedAvg, natural macro F1",
                "attacked_fedavg_natural_macro_f1",
                "attacked_lfighter_natural_macro_f1",
            ),
            paired_test_row(
                seed_table,
                "Clean LFighter vs clean FedAvg, natural macro F1",
                "clean_fedavg_natural_macro_f1",
                "clean_lfighter_natural_macro_f1",
            ),
            paired_test_row(
                seed_table,
                "Attacked vs clean LFighter, natural DDoS-to-Benign rate",
                "clean_lfighter_natural_source_to_target_rate",
                "attacked_lfighter_natural_source_to_target_rate",
            ),
            paired_test_row(
                seed_table,
                "Attacked vs clean LFighter, natural macro F1",
                "clean_lfighter_natural_macro_f1",
                "attacked_lfighter_natural_macro_f1",
            ),
        ]
    )
    tests.to_csv(tables_dir / "paired_method_tests.csv", index=False)

    def stats_for(columns: Sequence[str]) -> tuple[List[float], List[float]]:
        means, cis = [], []
        for column in columns:
            values = seed_table[column].astype(float).to_numpy()
            means.append(float(np.mean(values)))
            cis.append(ci95(values))
        return means, cis

    attack_columns = [
        "clean_fedavg_natural_source_to_target_rate",
        "attacked_fedavg_natural_source_to_target_rate",
        "clean_lfighter_natural_source_to_target_rate",
        "attacked_lfighter_natural_source_to_target_rate",
    ]
    means, cis = stats_for(attack_columns)
    bar_with_ci(
        ["Clean FedAvg", "Attacked FedAvg", "Clean LFighter", "Attacked LFighter"],
        means,
        cis,
        f"{args.source_class} predicted as {args.target_class}",
        "Original LFighter Attack-Rate Baseline",
        figures_dir / "natural_attack_rate_method_comparison",
    )

    macro_columns = [
        "clean_fedavg_natural_macro_f1",
        "attacked_fedavg_natural_macro_f1",
        "clean_lfighter_natural_macro_f1",
        "attacked_lfighter_natural_macro_f1",
    ]
    means, cis = stats_for(macro_columns)
    bar_with_ci(
        ["Clean FedAvg", "Attacked FedAvg", "Clean LFighter", "Attacked LFighter"],
        means,
        cis,
        "Natural macro F1",
        "Original LFighter Utility Comparison",
        figures_dir / "natural_macro_f1_method_comparison",
    )

    security_columns = [
        "attack_best_round_malicious_rejection_recall",
        "attack_best_round_benign_retention_rate",
        "attack_best_round_malicious_detection_precision",
        "attack_best_round_malicious_detection_f1",
        "attack_pair_identification_rate_across_rounds",
    ]
    means, cis = stats_for(security_columns)
    bar_with_ci(
        ["Malicious rejection", "Benign retention", "Detection precision", "Detection F1", "Pair identification"],
        means,
        cis,
        "Rate",
        "Original LFighter Security Diagnostics",
        figures_dir / "lfighter_security_diagnostics",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ordered = seed_table.sort_values("seed")
    ax.bar(
        ordered["seed"].astype(str),
        ordered["absolute_attack_rate_reduction_fedavg_minus_lfighter"],
    )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Model seed")
    ax.set_ylabel("FedAvg attacked rate minus LFighter attacked rate")
    ax.set_title("Per-Seed Original LFighter Attack-Rate Reduction")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "per_seed_attack_rate_reduction")

    metadata = {
        "experiment_version": "3.4",
        "experiment_type": "original_lfighter_five_seed_clean_and_strong_attack_baseline",
        "seeds": seeds,
        "source_class": args.source_class,
        "target_class": args.target_class,
        "malicious_clients": [
            int(value.strip()) for value in args.malicious_clients.split(",") if value.strip()
        ],
        "poison_fraction": args.poison_fraction,
        "attack_seed": args.attack_seed,
        "partition_file": str(args.partition_file.expanduser().resolve()),
        "clean_fedavg_reference": str(clean_multiseed_dir),
        "attacked_fedavg_reference": str(v33_dir),
        "aggregation_fidelity": (
            "Original LFighter multiclass class-pair selection, KMeans clustering, "
            "cluster dissimilarity, good-cluster selection, and equal averaging among "
            "admitted local models."
        ),
        "test_sets_used_for_checkpoint_selection": False,
        "primary_checkpoint": "best validation macro F1 independently within each run",
        "notes": [
            "Five paired model seeds are exploratory and descriptive.",
            "Exact two-sided Wilcoxon tests with five nonzero paired differences cannot attain p below 0.0625.",
            "Clean LFighter runs quantify benign false rejection and clean utility cost.",
            "Strong attacked runs use the frozen V3.3 malicious clients and poisoned row selection.",
        ],
    }
    with (output_dir / "lfighter_multiseed_v34_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nOriginal LFighter Multi-Seed V3.4 complete")
    print("\nSeed-level defense results")
    display_columns = [
        "seed",
        "attacked_fedavg_natural_source_to_target_rate",
        "attacked_lfighter_natural_source_to_target_rate",
        "absolute_attack_rate_reduction_fedavg_minus_lfighter",
        "clean_utility_cost_lfighter_minus_fedavg",
        "attack_best_round_malicious_rejection_recall",
        "attack_best_round_benign_retention_rate",
        "attack_best_round_pair_identified",
    ]
    print(seed_table[display_columns].to_string(index=False))
    print("\nCSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("Metadata:", output_dir / "lfighter_multiseed_v34_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

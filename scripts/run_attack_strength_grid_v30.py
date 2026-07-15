#!/usr/bin/env python3
"""Validation-controlled targeted label-flip strength grid for CIC IoT-DIAD V3.0.

The grid uses nested malicious-client sets and deterministic nested poisoned-row
sets. Each run delegates training and artifact generation to V2.9.1. Completed
runs can be reused with --skip-existing.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from federated_iot_v26 import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    load_protocol_arrays,
    make_loader,
    stable_softmax,
)
from neural_models_v24 import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a nested targeted label-flip strength calibration grid."
    )
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-seed-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--runner-script",
        type=Path,
        default=Path("scripts/run_targeted_label_flip_v291.py"),
    )
    parser.add_argument("--source-class", default="DDoS")
    parser.add_argument("--target-class", default="DoS")
    parser.add_argument("--malicious-fractions", default="0.10,0.20,0.40")
    parser.add_argument("--poison-fractions", default="0.50,1.00")
    parser.add_argument("--anchor-fraction", type=float, default=0.20)
    parser.add_argument("--min-source-samples", type=int, default=1000)
    parser.add_argument("--attack-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
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


def parse_fraction_list(text: str) -> List[float]:
    values = sorted({float(value.strip()) for value in text.split(",") if value.strip()})
    if not values:
        raise ValueError("At least one fraction is required.")
    if any(value <= 0 or value > 1 for value in values):
        raise ValueError("All fractions must be in (0, 1].")
    return values


def load_partitions(path: Path, expected_clients: int) -> List[np.ndarray]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Partition file not found: {path}")
    with np.load(path) as payload:
        keys = sorted(payload.files)
        if len(keys) != expected_clients:
            raise ValueError(
                f"Expected {expected_clients} clients, partition has {len(keys)}."
            )
        return [np.asarray(payload[key], dtype=np.int64) for key in keys]


def nested_client_ranking(
    partitions: Sequence[np.ndarray],
    y_train: np.ndarray,
    source_id: int,
    min_source_samples: int,
    attack_seed: int,
    anchor_fraction: float,
) -> Tuple[List[int], pd.DataFrame]:
    counts = []
    for client_id, indices in enumerate(partitions):
        counts.append((client_id, int(np.sum(y_train[indices] == source_id))))
    eligible = [client_id for client_id, count in counts if count >= min_source_samples]
    anchor_count = max(1, int(math.ceil(len(partitions) * anchor_fraction)))
    if len(eligible) < anchor_count:
        raise ValueError(
            f"Only {len(eligible)} eligible clients, anchor requires {anchor_count}."
        )

    rng = np.random.default_rng(attack_seed)
    anchor = sorted(rng.choice(eligible, size=anchor_count, replace=False).tolist())
    remaining = [value for value in eligible if value not in anchor]
    remaining_order = rng.permutation(remaining).astype(int).tolist()
    ranking = anchor + remaining_order

    count_map = dict(counts)
    rows = []
    for rank, client_id in enumerate(ranking, start=1):
        rows.append(
            {
                "nested_rank": rank,
                "client_id": client_id,
                "source_rows": count_map[client_id],
                "is_anchor_client": client_id in anchor,
            }
        )
    return ranking, pd.DataFrame(rows)


def clean_validation_baseline(
    data_file: Path,
    clean_seed_dir: Path,
    source_id: int,
    target_id: int,
    batch_size: int,
) -> Dict[str, float]:
    arrays = load_protocol_arrays(data_file)
    checkpoint_path = clean_seed_dir / "checkpoints" / "best_validation_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Clean checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(
        architecture="resmlp",
        input_dim=int(checkpoint["input_dim"]),
        num_classes=int(checkpoint["num_classes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = make_loader(
        arrays["X_val"].astype(np.float32, copy=False),
        arrays["y_val"].astype(np.int64, copy=False),
        batch_size=batch_size,
        shuffle=False,
    )
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for features, batch_labels in loader:
            logits.append(model(features).cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    probabilities = stable_softmax(np.concatenate(logits))
    y_true = np.concatenate(labels)
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y_true, predicted, labels=np.arange(NUM_CLASSES))
    source_total = int(matrix[source_id].sum())
    target_total = int(matrix[target_id].sum())
    return {
        "clean_validation_checkpoint_round": int(checkpoint["round"]),
        "clean_validation_source_recall": float(
            matrix[source_id, source_id] / max(source_total, 1)
        ),
        "clean_validation_target_recall": float(
            matrix[target_id, target_id] / max(target_total, 1)
        ),
        "clean_validation_source_to_target_rate": float(
            matrix[source_id, target_id] / max(source_total, 1)
        ),
        "clean_validation_source_to_target_count": int(matrix[source_id, target_id]),
    }


def run_complete(run_dir: Path) -> bool:
    required = [
        run_dir / "targeted_label_flip_metadata.json",
        run_dir / "attack_manifest" / "attack_summary.csv",
        run_dir / "tables" / "round_metrics.csv",
        run_dir / "tables" / "clean_vs_attack_source_target.csv",
        run_dir / "tables" / "clean_vs_attack_overall.csv",
        run_dir / "tables" / "attacked_source_target_metrics.csv",
    ]
    return all(path.exists() for path in required)


def run_name(malicious_fraction: float, poison_fraction: float) -> str:
    return f"mc{int(round(malicious_fraction * 100)):02d}_pf{int(round(poison_fraction * 100)):03d}"


def execute_run(
    args: argparse.Namespace,
    runner_script: Path,
    run_dir: Path,
    malicious_fraction: float,
    poison_fraction: float,
    malicious_clients: Sequence[int],
) -> None:
    if args.skip_existing and run_complete(run_dir):
        print(f"Reusing completed grid run: {run_dir.name}")
        return

    command = [
        sys.executable,
        str(runner_script),
        "--data-file", str(args.data_file.expanduser().resolve()),
        "--partition-file", str(args.partition_file.expanduser().resolve()),
        "--clean-seed-dir", str(args.clean_seed_dir.expanduser().resolve()),
        "--output-dir", str(run_dir),
        "--source-class", args.source_class,
        "--target-class", args.target_class,
        "--malicious-client-fraction", str(malicious_fraction),
        "--malicious-clients", ",".join(map(str, malicious_clients)),
        "--poison-fraction", str(poison_fraction),
        "--min-source-samples", str(args.min_source_samples),
        "--attack-seed", str(args.attack_seed),
        "--model-seed", str(args.model_seed),
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
    print("Running", run_dir.name)
    print("Malicious clients:", list(malicious_clients))
    subprocess.run(command, check=True)


def collect_run(
    run_dir: Path,
    malicious_fraction: float,
    poison_fraction: float,
    clean_validation: Dict[str, float],
) -> Tuple[Dict[str, object], pd.DataFrame]:
    with (run_dir / "targeted_label_flip_metadata.json").open(
        "r", encoding="utf-8"
    ) as handle:
        metadata = json.load(handle)
    attack_summary = pd.read_csv(run_dir / "attack_manifest" / "attack_summary.csv").iloc[0]
    rounds = pd.read_csv(run_dir / "tables" / "round_metrics.csv")
    best_round = int(metadata["best_validation_round"])
    best_row = rounds[rounds["round"] == best_round].iloc[0]
    primary_pair = pd.read_csv(
        run_dir / "tables" / "clean_vs_attack_source_target.csv"
    )
    primary_overall = pd.read_csv(
        run_dir / "tables" / "clean_vs_attack_overall.csv"
    )
    attacked_pair = pd.read_csv(
        run_dir / "tables" / "attacked_source_target_metrics.csv"
    )

    natural_pair = primary_pair[primary_pair["split"] == "test_natural"].iloc[0]
    diagnostic_pair = primary_pair[primary_pair["split"] == "test_diagnostic"].iloc[0]
    natural_overall = primary_overall[primary_overall["split"] == "test_natural"].iloc[0]
    diagnostic_overall = primary_overall[primary_overall["split"] == "test_diagnostic"].iloc[0]
    last_natural = attacked_pair[
        (attacked_pair["checkpoint"] == "last_round")
        & (attacked_pair["split"] == "test_natural")
    ].iloc[0]

    row: Dict[str, object] = {
        "run_name": run_dir.name,
        "malicious_client_fraction": malicious_fraction,
        "poison_fraction": poison_fraction,
        "malicious_clients": metadata["malicious_clients"],
        "malicious_client_count": len(metadata["malicious_clients"]),
        "global_flipped_rows": int(attack_summary["global_flipped_rows"]),
        "global_source_exposure_fraction": float(
            attack_summary["global_source_exposure_fraction"]
        ),
        "best_validation_round": best_round,
        "validation_macro_f1": float(best_row["val_macro_f1"]),
        "validation_source_recall": float(best_row["val_source_recall"]),
        "validation_target_recall": float(best_row["val_target_recall"]),
        "validation_source_to_target_rate": float(
            best_row["val_source_to_target_rate"]
        ),
        "validation_source_to_target_rate_clean": clean_validation[
            "clean_validation_source_to_target_rate"
        ],
        "validation_source_to_target_rate_delta": float(
            best_row["val_source_to_target_rate"]
            - clean_validation["clean_validation_source_to_target_rate"]
        ),
        "validation_source_to_target_fold_change": float(
            best_row["val_source_to_target_rate"]
            / max(clean_validation["clean_validation_source_to_target_rate"], 1e-12)
        ),
        "natural_source_to_target_rate_clean": float(
            natural_pair["source_to_target_rate_clean"]
        ),
        "natural_source_to_target_rate_attacked": float(
            natural_pair["source_to_target_rate_attacked"]
        ),
        "natural_source_to_target_rate_delta": float(
            natural_pair["source_to_target_rate_delta_attacked_minus_clean"]
        ),
        "natural_source_to_target_fold_change": float(
            natural_pair["source_to_target_fold_change"]
        ),
        "diagnostic_source_to_target_rate_attacked": float(
            diagnostic_pair["source_to_target_rate_attacked"]
        ),
        "natural_macro_f1_clean": float(natural_overall["macro_f1_clean"]),
        "natural_macro_f1_attacked": float(natural_overall["macro_f1_attacked"]),
        "natural_macro_f1_delta": float(
            natural_overall["macro_f1_delta_attacked_minus_clean"]
        ),
        "diagnostic_macro_f1_attacked": float(
            diagnostic_overall["macro_f1_attacked"]
        ),
        "last_round_natural_source_to_target_rate": float(
            last_natural["source_to_target_rate"]
        ),
        "rounds_completed": int(metadata["rounds_completed"]),
        "total_seconds": float(metadata["total_seconds"]),
        "partition_hash_sha256": metadata["partition_hash_sha256"],
        "poison_index_hash_sha256": metadata["poison_index_hash_sha256"],
    }
    return row, rounds.assign(
        run_name=run_dir.name,
        malicious_client_fraction=malicious_fraction,
        poison_fraction=poison_fraction,
    )


def matrix_from_summary(
    summary: pd.DataFrame,
    malicious_fractions: Sequence[float],
    poison_fractions: Sequence[float],
    value: str,
) -> np.ndarray:
    matrix = np.full((len(malicious_fractions), len(poison_fractions)), np.nan)
    for row_index, malicious_fraction in enumerate(malicious_fractions):
        for column_index, poison_fraction in enumerate(poison_fractions):
            selected = summary[
                np.isclose(summary["malicious_client_fraction"], malicious_fraction)
                & np.isclose(summary["poison_fraction"], poison_fraction)
            ]
            if not selected.empty:
                matrix[row_index, column_index] = float(selected.iloc[0][value])
    return matrix


def plot_heatmap(
    summary: pd.DataFrame,
    malicious_fractions: Sequence[float],
    poison_fractions: Sequence[float],
    value: str,
    title: str,
    label: str,
    output: Path,
) -> None:
    matrix = matrix_from_summary(summary, malicious_fractions, poison_fractions, value)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(poison_fractions)))
    ax.set_xticklabels([f"{value_:.0%}" for value_ in poison_fractions])
    ax.set_yticks(np.arange(len(malicious_fractions)))
    ax.set_yticklabels([f"{value_:.0%}" for value_ in malicious_fractions])
    ax.set_xlabel("Poison fraction on malicious DDoS rows")
    ax.set_ylabel("Malicious client fraction")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if np.isfinite(matrix[row, column]):
                ax.text(column, row, f"{matrix[row, column]:.4f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=label)
    save_figure(fig, output)


def main() -> int:
    args = parse_args()
    malicious_fractions = parse_fraction_list(args.malicious_fractions)
    poison_fractions = parse_fraction_list(args.poison_fractions)
    if args.anchor_fraction not in malicious_fractions:
        raise ValueError("anchor-fraction must be included in malicious-fractions.")
    if args.source_class not in CLASS_NAMES or args.target_class not in CLASS_NAMES:
        raise ValueError("Unknown source or target class.")

    torch.set_num_threads(max(1, args.threads))
    output_dir = args.output_dir.expanduser().resolve()
    runs_dir = output_dir / "runs"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for path in (runs_dir, tables_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    runner_script = args.runner_script.expanduser().resolve()
    if not runner_script.exists():
        raise FileNotFoundError(f"V2.9.1 runner not found: {runner_script}")

    arrays = load_protocol_arrays(args.data_file.expanduser().resolve())
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    source_id = CLASS_NAMES.index(args.source_class)
    target_id = CLASS_NAMES.index(args.target_class)
    partitions = load_partitions(args.partition_file, args.num_clients)

    ranking, ranking_table = nested_client_ranking(
        partitions=partitions,
        y_train=y_train,
        source_id=source_id,
        min_source_samples=args.min_source_samples,
        attack_seed=args.attack_seed,
        anchor_fraction=args.anchor_fraction,
    )
    ranking_table.to_csv(tables_dir / "nested_malicious_client_ranking.csv", index=False)

    clean_validation = clean_validation_baseline(
        data_file=args.data_file.expanduser().resolve(),
        clean_seed_dir=args.clean_seed_dir.expanduser().resolve(),
        source_id=source_id,
        target_id=target_id,
        batch_size=args.evaluation_batch_size,
    )
    pd.DataFrame([clean_validation]).to_csv(
        tables_dir / "clean_validation_attack_baseline.csv", index=False
    )

    run_rows = []
    all_rounds = []
    selection_rows = []
    for malicious_fraction in malicious_fractions:
        count = max(1, int(math.ceil(args.num_clients * malicious_fraction)))
        if count > len(ranking):
            raise ValueError(
                f"Fraction {malicious_fraction} requires {count} malicious clients, "
                f"but only {len(ranking)} are eligible."
            )
        malicious_clients = sorted(ranking[:count])
        selection_rows.append(
            {
                "malicious_client_fraction": malicious_fraction,
                "malicious_client_count": count,
                "malicious_clients": "|".join(map(str, malicious_clients)),
            }
        )
        for poison_fraction in poison_fractions:
            name = run_name(malicious_fraction, poison_fraction)
            run_dir = runs_dir / name
            execute_run(
                args=args,
                runner_script=runner_script,
                run_dir=run_dir,
                malicious_fraction=malicious_fraction,
                poison_fraction=poison_fraction,
                malicious_clients=malicious_clients,
            )
            row, rounds = collect_run(
                run_dir=run_dir,
                malicious_fraction=malicious_fraction,
                poison_fraction=poison_fraction,
                clean_validation=clean_validation,
            )
            run_rows.append(row)
            all_rounds.append(rounds)

    selection_table = pd.DataFrame(selection_rows).drop_duplicates()
    selection_table.to_csv(tables_dir / "nested_client_sets.csv", index=False)
    summary = pd.DataFrame(run_rows).sort_values(
        ["malicious_client_fraction", "poison_fraction"]
    )
    summary.to_csv(tables_dir / "attack_strength_grid_summary.csv", index=False)
    pd.concat(all_rounds, ignore_index=True).to_csv(
        tables_dir / "round_metrics_all_grid_runs.csv", index=False
    )

    validation_ranking = summary.sort_values(
        [
            "validation_source_to_target_rate_delta",
            "validation_source_to_target_rate",
            "validation_macro_f1",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    validation_ranking.insert(0, "validation_attack_rank", np.arange(1, len(validation_ranking) + 1))
    validation_ranking.to_csv(
        tables_dir / "validation_only_attack_strength_ranking.csv", index=False
    )

    plot_heatmap(
        summary, malicious_fractions, poison_fractions,
        "validation_source_to_target_rate",
        "Validation DDoS to DoS Attack Rate",
        "Rate",
        figures_dir / "validation_attack_rate_heatmap",
    )
    plot_heatmap(
        summary, malicious_fractions, poison_fractions,
        "natural_source_to_target_rate_attacked",
        "Natural-Test DDoS to DoS Attack Rate",
        "Rate",
        figures_dir / "natural_attack_rate_heatmap",
    )
    plot_heatmap(
        summary, malicious_fractions, poison_fractions,
        "natural_macro_f1_delta",
        "Natural-Test Macro F1 Change",
        "Attacked minus clean",
        figures_dir / "natural_macro_f1_delta_heatmap",
    )
    plot_heatmap(
        summary, malicious_fractions, poison_fractions,
        "global_source_exposure_fraction",
        "Global DDoS Poison Exposure",
        "Fraction",
        figures_dir / "global_exposure_heatmap",
    )

    metadata = {
        "experiment_version": "3.0",
        "experiment_type": "validation_controlled_targeted_attack_strength_grid",
        "source_class": args.source_class,
        "target_class": args.target_class,
        "malicious_fractions": malicious_fractions,
        "poison_fractions": poison_fractions,
        "anchor_fraction": args.anchor_fraction,
        "nested_malicious_client_sets": True,
        "nested_poisoned_rows_within_client": True,
        "attack_seed": args.attack_seed,
        "model_seed": args.model_seed,
        "selection_basis": (
            "Validation source-to-target attack rate. Test metrics are reported "
            "but must not be used to choose final strengths."
        ),
        "clean_validation_baseline": clean_validation,
        "runner_script": str(runner_script),
        "notes": [
            "This is a single-model-seed strength calibration grid.",
            "The next stage will repeat selected weak, moderate, and strong settings across multiple seeds.",
            "Original LFighter evaluation begins only after attack strengths are calibrated.",
        ],
    }
    with (output_dir / "attack_strength_grid_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Attack Strength Grid V3.0 complete")
    print()
    print("Nested client sets")
    print(selection_table.to_string(index=False))
    print()
    print("Validation-only attack-strength ranking")
    print(
        validation_ranking[
            [
                "validation_attack_rank",
                "run_name",
                "malicious_client_fraction",
                "poison_fraction",
                "global_source_exposure_fraction",
                "validation_source_to_target_rate",
                "validation_source_to_target_rate_delta",
                "validation_macro_f1",
                "natural_source_to_target_rate_attacked",
                "natural_macro_f1_delta",
            ]
        ].to_string(index=False)
    )
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("Metadata:", output_dir / "attack_strength_grid_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

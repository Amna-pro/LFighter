#!/usr/bin/env python3
"""Validation-controlled attack-pair screening for CIC IoT-DIAD V3.1.

This stage compares three targeted label-flipping pairs under the same strong
stress-test setting and the same malicious clients:

1. DDoS -> DoS, reused from the completed V3.0 control run
2. DDoS -> Benign
3. DoS -> Benign

Pair selection uses validation attack-success-rate increase only. Natural and
diagnostic test results are reported after the validation-controlled runs.
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


PAIR_SPECS = [
    ("DDoS", "DoS", "ddos_to_dos"),
    ("DDoS", "Benign", "ddos_to_benign"),
    ("DoS", "Benign", "dos_to_benign"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen targeted label-flipping attack pairs."
    )
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-seed-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--runner-script",
        type=Path,
        default=Path("scripts/run_targeted_label_flip_v292.py"),
    )
    parser.add_argument("--malicious-client-fraction", type=float, default=0.40)
    parser.add_argument("--poison-fraction", type=float, default=1.00)
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


def select_shared_malicious_clients(
    partitions: Sequence[np.ndarray],
    y_train: np.ndarray,
    source_ids: Sequence[int],
    requested_count: int,
    min_source_samples: int,
    attack_seed: int,
) -> Tuple[List[int], pd.DataFrame]:
    """Select one reproducible client set eligible for every source class."""
    rows = []
    eligible = []
    for client_id, indices in enumerate(partitions):
        row = {"client_id": int(client_id)}
        all_eligible = True
        for source_id in source_ids:
            class_name = CLASS_NAMES[source_id]
            count = int(np.sum(y_train[indices] == source_id))
            row[f"{class_name}_rows"] = count
            row[f"{class_name}_eligible"] = count >= min_source_samples
            all_eligible = all_eligible and count >= min_source_samples
        row["eligible_for_all_sources"] = all_eligible
        rows.append(row)
        if all_eligible:
            eligible.append(client_id)

    if len(eligible) < requested_count:
        raise ValueError(
            f"Only {len(eligible)} clients are eligible for every source class, "
            f"but {requested_count} are required."
        )

    rng = np.random.default_rng(attack_seed)
    selected = sorted(
        rng.choice(eligible, size=requested_count, replace=False).tolist()
    )
    table = pd.DataFrame(rows)
    table["selected_shared_malicious_client"] = table["client_id"].isin(selected)
    return selected, table


def validate_clients_for_pair(
    partitions: Sequence[np.ndarray],
    y_train: np.ndarray,
    malicious_clients: Sequence[int],
    source_id: int,
    min_source_samples: int,
) -> pd.DataFrame:
    rows = []
    invalid = []
    for client_id in malicious_clients:
        count = int(np.sum(y_train[partitions[client_id]] == source_id))
        rows.append(
            {
                "client_id": int(client_id),
                "source_class": CLASS_NAMES[source_id],
                "source_rows": count,
                "eligible": count >= min_source_samples,
            }
        )
        if count < min_source_samples:
            invalid.append((client_id, count))
    if invalid:
        raise ValueError(
            f"Shared malicious clients are not eligible for "
            f"{CLASS_NAMES[source_id]}: {invalid}"
        )
    return pd.DataFrame(rows)


def clean_validation_baseline(
    arrays: Dict[str, np.ndarray],
    clean_seed_dir: Path,
    source_id: int,
    target_id: int,
    batch_size: int,
) -> Dict[str, float]:
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
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y_true, predictions, labels=np.arange(NUM_CLASSES))
    source_total = int(matrix[source_id].sum())
    target_total = int(matrix[target_id].sum())
    validation_metrics = checkpoint.get("validation_metrics", {})
    return {
        "clean_validation_round": int(checkpoint["round"]),
        "clean_validation_macro_f1": float(validation_metrics.get("macro_f1", np.nan)),
        "clean_validation_source_recall": float(
            matrix[source_id, source_id] / max(source_total, 1)
        ),
        "clean_validation_target_recall": float(
            matrix[target_id, target_id] / max(target_total, 1)
        ),
        "clean_validation_source_to_target_count": int(
            matrix[source_id, target_id]
        ),
        "clean_validation_source_to_target_rate": float(
            matrix[source_id, target_id] / max(source_total, 1)
        ),
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


def execute_pair(
    args: argparse.Namespace,
    runner_script: Path,
    run_dir: Path,
    source_class: str,
    target_class: str,
    malicious_clients: Sequence[int],
) -> None:
    if args.skip_existing and run_complete(run_dir):
        print(f"Reusing completed pair run: {run_dir.name}")
        return

    command = [
        sys.executable,
        str(runner_script),
        "--data-file", str(args.data_file.expanduser().resolve()),
        "--partition-file", str(args.partition_file.expanduser().resolve()),
        "--clean-seed-dir", str(args.clean_seed_dir.expanduser().resolve()),
        "--output-dir", str(run_dir),
        "--source-class", source_class,
        "--target-class", target_class,
        "--malicious-client-fraction", str(args.malicious_client_fraction),
        "--malicious-clients", ",".join(map(str, malicious_clients)),
        "--poison-fraction", str(args.poison_fraction),
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
    print(f"Running {source_class} -> {target_class}")
    print("Malicious clients:", list(malicious_clients))
    subprocess.run(command, check=True)


def collect_pair(
    run_dir: Path,
    source_class: str,
    target_class: str,
    clean_validation: Dict[str, float],
) -> Tuple[Dict[str, object], pd.DataFrame]:
    with (run_dir / "targeted_label_flip_metadata.json").open(
        "r", encoding="utf-8"
    ) as handle:
        metadata = json.load(handle)

    attack_summary = pd.read_csv(
        run_dir / "attack_manifest" / "attack_summary.csv"
    ).iloc[0]
    rounds = pd.read_csv(run_dir / "tables" / "round_metrics.csv")
    best_round = int(metadata["best_validation_round"])
    best_row = rounds[rounds["round"] == best_round].iloc[0]

    pair_table = pd.read_csv(
        run_dir / "tables" / "clean_vs_attack_source_target.csv"
    )
    overall_table = pd.read_csv(
        run_dir / "tables" / "clean_vs_attack_overall.csv"
    )
    attacked_pair = pd.read_csv(
        run_dir / "tables" / "attacked_source_target_metrics.csv"
    )

    natural_pair = pair_table[pair_table["split"] == "test_natural"].iloc[0]
    diagnostic_pair = pair_table[
        pair_table["split"] == "test_diagnostic"
    ].iloc[0]
    natural_overall = overall_table[
        overall_table["split"] == "test_natural"
    ].iloc[0]
    diagnostic_overall = overall_table[
        overall_table["split"] == "test_diagnostic"
    ].iloc[0]
    last_natural = attacked_pair[
        (attacked_pair["checkpoint"] == "last_round")
        & (attacked_pair["split"] == "test_natural")
    ].iloc[0]

    validation_rate = float(best_row["val_source_to_target_rate"])
    validation_source_recall = float(best_row["val_source_recall"])
    validation_delta = (
        validation_rate
        - clean_validation["clean_validation_source_to_target_rate"]
    )
    exposure = float(attack_summary["global_source_exposure_fraction"])

    row: Dict[str, object] = {
        "pair_name": f"{source_class}_to_{target_class}",
        "source_class": source_class,
        "target_class": target_class,
        "malicious_clients": "|".join(map(str, metadata["malicious_clients"])),
        "malicious_client_count": len(metadata["malicious_clients"]),
        "malicious_client_fraction": float(
            metadata["malicious_client_fraction_requested"]
        ),
        "poison_fraction": float(
            metadata["poison_fraction_within_malicious_source_rows"]
        ),
        "global_source_rows": int(metadata["global_source_rows"]),
        "global_flipped_rows": int(metadata["global_flipped_rows"]),
        "global_source_exposure_fraction": exposure,
        "best_validation_round": best_round,
        "clean_validation_macro_f1": clean_validation[
            "clean_validation_macro_f1"
        ],
        "attacked_validation_macro_f1": float(best_row["val_macro_f1"]),
        "validation_macro_f1_delta": float(
            best_row["val_macro_f1"]
            - clean_validation["clean_validation_macro_f1"]
        ),
        "clean_validation_source_recall": clean_validation[
            "clean_validation_source_recall"
        ],
        "attacked_validation_source_recall": validation_source_recall,
        "validation_source_recall_drop": float(
            clean_validation["clean_validation_source_recall"]
            - validation_source_recall
        ),
        "clean_validation_source_to_target_rate": clean_validation[
            "clean_validation_source_to_target_rate"
        ],
        "attacked_validation_source_to_target_rate": validation_rate,
        "validation_source_to_target_rate_delta": validation_delta,
        "validation_source_to_target_fold_change": float(
            validation_rate
            / max(
                clean_validation["clean_validation_source_to_target_rate"],
                1e-12,
            )
        ),
        "validation_asr_gain_per_global_exposure": float(
            validation_delta / max(exposure, 1e-12)
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
        "natural_macro_f1_attacked": float(
            natural_overall["macro_f1_attacked"]
        ),
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
        pair_name=f"{source_class}_to_{target_class}",
        source_class=source_class,
        target_class=target_class,
    )


def plot_grouped_rates(
    summary: pd.DataFrame,
    clean_column: str,
    attacked_column: str,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    labels = [
        f"{row.source_class}→{row.target_class}"
        for row in summary.itertuples(index=False)
    ]
    x = np.arange(len(summary))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, summary[clean_column], width, label="Clean")
    ax.bar(x + width / 2, summary[attacked_column], width, label="Attacked")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def plot_exposure_efficiency(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        summary["global_source_exposure_fraction"],
        summary["validation_source_to_target_rate_delta"],
        s=90,
    )
    for row in summary.itertuples(index=False):
        ax.annotate(
            f"{row.source_class}→{row.target_class}",
            (
                float(row.global_source_exposure_fraction),
                float(row.validation_source_to_target_rate_delta),
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Global source-label exposure")
    ax.set_ylabel("Validation ASR increase")
    ax.set_title("Attack-Pair Effect Versus Poison Exposure")
    ax.grid(alpha=0.25)
    save_figure(fig, output)


def plot_source_recall_drop(summary: pd.DataFrame, output: Path) -> None:
    labels = [
        f"{row.source_class}→{row.target_class}"
        for row in summary.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(labels, summary["validation_source_recall_drop"])
    ax.set_ylabel("Clean recall minus attacked recall")
    ax.set_title("Validation Source-Class Recall Degradation")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output)


def main() -> int:
    args = parse_args()
    if not 0 < args.malicious_client_fraction <= 1:
        raise ValueError("malicious-client-fraction must be in (0, 1].")
    if not 0 < args.poison_fraction <= 1:
        raise ValueError("poison-fraction must be in (0, 1].")

    torch.set_num_threads(max(1, args.threads))
    output_dir = args.output_dir.expanduser().resolve()
    runs_dir = output_dir / "runs"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for path in (runs_dir, tables_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    data_file = args.data_file.expanduser().resolve()
    partition_file = args.partition_file.expanduser().resolve()
    clean_seed_dir = args.clean_seed_dir.expanduser().resolve()
    runner_script = args.runner_script.expanduser().resolve()
    if not runner_script.exists():
        raise FileNotFoundError(f"V2.9.1 runner not found: {runner_script}")

    arrays = load_protocol_arrays(data_file)
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    partitions = load_partitions(partition_file, args.num_clients)
    expected_count = max(
        1,
        int(math.ceil(args.num_clients * args.malicious_client_fraction)),
    )
    source_ids = sorted(
        {
            CLASS_NAMES.index(source_class)
            for source_class, _, _ in PAIR_SPECS
        }
    )
    malicious_clients, shared_eligibility = select_shared_malicious_clients(
        partitions=partitions,
        y_train=y_train,
        source_ids=source_ids,
        requested_count=expected_count,
        min_source_samples=args.min_source_samples,
        attack_seed=args.attack_seed,
    )
    shared_eligibility.to_csv(
        tables_dir / "shared_source_intersection_and_selection.csv",
        index=False,
    )

    run_rows = []
    round_tables = []
    eligibility_tables = []
    for source_class, target_class, run_name in PAIR_SPECS:
        source_id = CLASS_NAMES.index(source_class)
        target_id = CLASS_NAMES.index(target_class)
        eligibility = validate_clients_for_pair(
            partitions=partitions,
            y_train=y_train,
            malicious_clients=malicious_clients,
            source_id=source_id,
            min_source_samples=args.min_source_samples,
        )
        eligibility["pair_name"] = f"{source_class}_to_{target_class}"
        eligibility_tables.append(eligibility)

        clean_validation = clean_validation_baseline(
            arrays=arrays,
            clean_seed_dir=clean_seed_dir,
            source_id=source_id,
            target_id=target_id,
            batch_size=args.evaluation_batch_size,
        )

        run_dir = runs_dir / run_name
        execute_pair(
            args=args,
            runner_script=runner_script,
            run_dir=run_dir,
            source_class=source_class,
            target_class=target_class,
            malicious_clients=malicious_clients,
        )

        row, rounds = collect_pair(
            run_dir=run_dir,
            source_class=source_class,
            target_class=target_class,
            clean_validation=clean_validation,
        )
        run_rows.append(row)
        round_tables.append(rounds)

    eligibility_table = pd.concat(eligibility_tables, ignore_index=True)
    eligibility_table.to_csv(
        tables_dir / "shared_malicious_client_eligibility.csv",
        index=False,
    )
    summary = pd.DataFrame(run_rows)
    summary.to_csv(tables_dir / "attack_pair_screen_summary.csv", index=False)
    pd.concat(round_tables, ignore_index=True).to_csv(
        tables_dir / "round_metrics_all_pairs.csv",
        index=False,
    )

    ranking = summary.sort_values(
        [
            "validation_source_to_target_rate_delta",
            "validation_source_recall_drop",
            "attacked_validation_macro_f1",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranking.insert(0, "validation_pair_rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(
        tables_dir / "validation_only_attack_pair_ranking.csv",
        index=False,
    )

    plot_grouped_rates(
        summary,
        "clean_validation_source_to_target_rate",
        "attacked_validation_source_to_target_rate",
        "Validation Attack Success by Targeted Pair",
        "Source samples predicted as target",
        figures_dir / "validation_attack_success_by_pair",
    )
    plot_grouped_rates(
        summary,
        "natural_source_to_target_rate_clean",
        "natural_source_to_target_rate_attacked",
        "Natural-Test Attack Success by Targeted Pair",
        "Source samples predicted as target",
        figures_dir / "natural_attack_success_by_pair",
    )
    plot_exposure_efficiency(
        summary,
        figures_dir / "validation_attack_gain_vs_exposure",
    )
    plot_source_recall_drop(
        summary,
        figures_dir / "validation_source_recall_drop",
    )

    metadata = {
        "experiment_version": "3.1.1",
        "experiment_type": "validation_controlled_attack_pair_screening",
        "pairs": [
            {"source": source, "target": target}
            for source, target, _ in PAIR_SPECS
        ],
        "shared_malicious_clients": malicious_clients,
        "shared_malicious_clients_across_pairs": True,
        "malicious_client_fraction": args.malicious_client_fraction,
        "poison_fraction": args.poison_fraction,
        "model_seed": args.model_seed,
        "attack_seed": args.attack_seed,
                "selection_basis": (
            "Largest validation source-to-target rate increase, followed by "
            "validation source-recall degradation. Test metrics are not used "
            "for pair selection."
        ),
        "notes": [
            "All three pairs are rerun with one shared source-eligible client set.",
            "The exact same malicious client IDs are used for all three pairs.",
            "Global source exposure differs because source-class distributions differ.",
            "The winning pair must undergo a new strength calibration and multi-seed evaluation.",
        ],
    }
    with (output_dir / "attack_pair_screen_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Attack Pair Screen V3.1.1 complete")
    print()
    print("Validation-only pair ranking")
    print(
        ranking[
            [
                "validation_pair_rank",
                "source_class",
                "target_class",
                "global_source_exposure_fraction",
                "clean_validation_source_to_target_rate",
                "attacked_validation_source_to_target_rate",
                "validation_source_to_target_rate_delta",
                "validation_source_recall_drop",
                "attacked_validation_macro_f1",
                "natural_source_to_target_rate_attacked",
                "natural_macro_f1_delta",
            ]
        ].to_string(index=False)
    )
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("Metadata:", output_dir / "attack_pair_screen_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the prediction-led personalized soft-trust defense, V3.6.1."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from federated_iot_v26 import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    load_protocol_arrays,
    partition_manifest,
    set_seed,
    sqrt_class_weights,
    train_local_model,
)
from neural_models_v24 import build_model  # noqa: E402
from proposed_soft_trust_v361 import (  # noqa: E402
    PredictionSoftTrustMemory,
    aggregate_prediction_soft_trust,
)
from run_targeted_label_flip_v292 import (  # noqa: E402
    load_fixed_partitions,
    prepare_static_attack,
    read_clean_partition_hash,
    save_poisoned_indices,
)
from run_proposed_temporal_defense_v35 import (  # noqa: E402
    balanced_probe_indices,
    class_conditional_probability_means,
    evaluate_checkpoint,
    evaluate_validation,
    explanation_signature,
    local_probe_signal,
    parse_client_ids,
    predict_probabilities,
)

DEFAULT_MALICIOUS_CLIENTS = "1,7,8,10,14,15,17,18"
PREDICTION_PROFILE_SIGNAL = "max_offdiag_probability_drift"
PREDICTION_SCORE_COLUMN = "personalized_abs_z_max_offdiag_probability_drift"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V3.6.1 personalized prediction-drift soft-trust aggregation."
    )
    parser.add_argument("--mode", choices=["clean", "strong_attack"], required=True)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--partition-file", required=True, type=Path)
    parser.add_argument("--clean-seed-dir", required=True, type=Path)
    parser.add_argument(
        "--calibration-dir",
        required=True,
        type=Path,
        help="V3.5.1 personalized_scores result directory for the same model seed.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-class", default="DDoS")
    parser.add_argument("--target-class", default="Benign")
    parser.add_argument("--malicious-clients", default=DEFAULT_MALICIOUS_CLIENTS)
    parser.add_argument("--poison-fraction", type=float, default=1.0)
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
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--probe-per-class", type=int, default=48)
    parser.add_argument("--explanation-per-class", type=int, default=12)
    parser.add_argument("--probe-seed", type=int, default=3501)
    parser.add_argument("--ema-decay", type=float, default=0.65)
    parser.add_argument("--clean-ema-quantile", type=float, default=0.85)
    parser.add_argument("--trust-gamma", type=float, default=1.20)
    parser.add_argument("--minimum-trust", type=float, default=0.15)
    parser.add_argument("--count-cap-multiplier", type=float, default=3.0)
    parser.add_argument("--low-trust-cutoff", type=float, default=0.50)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def quantile_higher(values: np.ndarray, quantile: float) -> float:
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:
        return float(np.quantile(values, quantile, interpolation="higher"))


def add_client_ema(
    table: pd.DataFrame,
    score_column: str,
    decay: float,
) -> np.ndarray:
    ordered = table.sort_values(["round", "client_id"]).reset_index(drop=True)
    memory: Dict[int, float] = {}
    values = []
    for _, row in ordered.iterrows():
        client_id = int(row["client_id"])
        current = float(row[score_column])
        previous = float(memory.get(client_id, 0.0))
        ema = float(decay) * previous + (1.0 - float(decay)) * current
        memory[client_id] = ema
        values.append(ema)
    return np.asarray(values, dtype=float)


def load_calibration(
    calibration_dir: Path,
    ema_decay: float,
    clean_quantile: float,
    expected_clients: int,
) -> tuple[Dict[int, tuple[float, float]], float, pd.DataFrame]:
    tables_dir = calibration_dir / "tables"
    profile_path = tables_dir / "personalized_clean_profiles.csv"
    clean_scores_path = tables_dir / "clean_leave_one_out_scores.csv"
    for path in (profile_path, clean_scores_path):
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

    profiles = pd.read_csv(profile_path)
    profiles = profiles[profiles["signal"].eq(PREDICTION_PROFILE_SIGNAL)].copy()
    if len(profiles) != expected_clients:
        raise ValueError(
            f"Expected {expected_clients} prediction profiles, found {len(profiles)}"
        )
    profile_map = {
        int(row.client_id): (
            float(row.clean_transformed_median),
            float(row.clean_transformed_scale),
        )
        for row in profiles.itertuples()
    }

    clean_scores = pd.read_csv(clean_scores_path)
    if PREDICTION_SCORE_COLUMN not in clean_scores.columns:
        raise ValueError(
            f"Clean score table is missing {PREDICTION_SCORE_COLUMN}"
        )
    clean_ema = add_client_ema(
        clean_scores,
        score_column=PREDICTION_SCORE_COLUMN,
        decay=ema_decay,
    )
    threshold = quantile_higher(clean_ema, clean_quantile)
    calibration_summary = pd.DataFrame(
        [
            {
                "profile_signal": PREDICTION_PROFILE_SIGNAL,
                "score_column": PREDICTION_SCORE_COLUMN,
                "ema_decay": float(ema_decay),
                "clean_ema_quantile": float(clean_quantile),
                "clean_ema_threshold": float(threshold),
                "clean_rows": int(len(clean_scores)),
                "profile_clients": int(len(profile_map)),
                "threshold_source": str(clean_scores_path),
                "profile_source": str(profile_path),
            }
        ]
    )
    return profile_map, float(threshold), calibration_summary


def plot_validation(table: pd.DataFrame, source: str, target: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(table["round"], table["val_macro_f1"], marker="o", label="Validation macro F1")
    ax.plot(
        table["round"],
        table["val_source_to_target_rate"],
        marker="s",
        label=f"Validation {source} to {target}",
    )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Prediction soft-trust validation dynamics")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out)


def plot_trust(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(table["round"], table["mean_benign_trust"], marker="o", label="Mean benign trust")
    ax.plot(table["round"], table["mean_malicious_trust"], marker="s", label="Mean malicious trust")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Trust")
    ax.set_ylim(0, 1.05)
    ax.set_title("Prediction soft-trust client weighting")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out)


def plot_influence(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        table["round"],
        table["base_malicious_weight_share"],
        marker="o",
        label="Base malicious weight share",
    )
    ax.plot(
        table["round"],
        table["adjusted_malicious_weight_share"],
        marker="s",
        label="Adjusted malicious weight share",
    )
    ax.plot(
        table["round"],
        table["malicious_influence_reduction"],
        marker="^",
        label="Influence reduction",
    )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Malicious aggregation influence")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out)


def main() -> int:
    args = parse_args()
    if args.source_class not in CLASS_NAMES or args.target_class not in CLASS_NAMES:
        raise ValueError("Unknown source or target class")
    if args.source_class == args.target_class:
        raise ValueError("Source and target classes must differ")
    if not math.isclose(args.participation_rate, 1.0):
        raise ValueError("V3.6.1 is frozen to full client participation")
    if not 0.0 < args.clean_ema_quantile <= 1.0:
        raise ValueError("clean EMA quantile must be in (0, 1]")

    source_id = CLASS_NAMES.index(args.source_class)
    target_id = CLASS_NAMES.index(args.target_class)
    torch.set_num_threads(max(1, args.threads))
    set_seed(args.model_seed)

    data_file = args.data_file.expanduser().resolve()
    partition_file = args.partition_file.expanduser().resolve()
    clean_seed_dir = args.clean_seed_dir.expanduser().resolve()
    calibration_dir = args.calibration_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"
    attack_dir = output_dir / "attack_manifest"
    calibration_output_dir = output_dir / "calibration"
    for path in (
        tables_dir,
        figures_dir,
        checkpoints_dir,
        attack_dir,
        calibration_output_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    prediction_profiles, clean_ema_threshold, calibration_summary = load_calibration(
        calibration_dir=calibration_dir,
        ema_decay=args.ema_decay,
        clean_quantile=args.clean_ema_quantile,
        expected_clients=args.num_clients,
    )
    calibration_summary.to_csv(
        calibration_output_dir / "prediction_soft_trust_calibration.csv",
        index=False,
    )

    arrays = load_protocol_arrays(data_file)
    X_train = arrays["X_train"].astype(np.float32, copy=False)
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    X_val = arrays["X_val"].astype(np.float32, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)

    client_indices, partition_hash = load_fixed_partitions(
        partition_file,
        expected_clients=args.num_clients,
        train_rows=len(y_train),
    )
    clean_hash = read_clean_partition_hash(clean_seed_dir)
    if clean_hash != partition_hash:
        raise RuntimeError("Clean seed partition hash does not match supplied partition")

    client_summary, client_matrix = partition_manifest(client_indices, y_train)
    client_summary.to_csv(tables_dir / "client_partition_summary.csv", index=False)
    client_matrix.to_csv(tables_dir / "client_class_counts.csv", index=False)

    if args.mode == "strong_attack":
        malicious_clients = parse_client_ids(args.malicious_clients)
        requested_fraction = len(malicious_clients) / args.num_clients
        malicious_clients, poisoned_positions, poison_manifest, poison_hash = prepare_static_attack(
            client_indices=client_indices,
            y_train=y_train,
            source_id=source_id,
            target_id=target_id,
            malicious_client_fraction=requested_fraction,
            poison_fraction=args.poison_fraction,
            min_source_samples=args.min_source_samples,
            attack_seed=args.attack_seed,
            explicit_malicious_clients=malicious_clients,
        )
    else:
        malicious_clients = []
        poisoned_positions = {
            client_id: np.empty(0, dtype=np.int64)
            for client_id in range(args.num_clients)
        }
        poison_manifest = pd.DataFrame(
            [
                {
                    "client_id": client_id,
                    "is_malicious": False,
                    "client_rows": int(len(indices)),
                    "source_rows_before_poisoning": int(
                        np.sum(y_train[indices] == source_id)
                    ),
                    "poisoned_source_rows": 0,
                    "client_source_poison_rate": 0.0,
                    "source_rows_after_poisoning": int(
                        np.sum(y_train[indices] == source_id)
                    ),
                    "target_rows_added_by_poisoning": 0,
                }
                for client_id, indices in enumerate(client_indices)
            ]
        )
        digest = hashlib.sha256()
        for client_id in range(args.num_clients):
            digest.update(f"client_{client_id:03d}".encode("utf-8"))
        poison_hash = digest.hexdigest()

    poison_manifest.to_csv(
        attack_dir / "malicious_client_poison_manifest.csv", index=False
    )
    save_poisoned_indices(
        client_indices,
        poisoned_positions,
        attack_dir / "poisoned_indices.npz",
    )

    prediction_probe_indices = balanced_probe_indices(
        y_val, args.probe_per_class, args.probe_seed
    )
    explanation_probe_indices = balanced_probe_indices(
        y_val, args.explanation_per_class, args.probe_seed + 1
    )
    X_probe = X_val[prediction_probe_indices]
    y_probe = y_val[prediction_probe_indices]
    X_explain = X_val[explanation_probe_indices]
    y_explain = y_val[explanation_probe_indices]

    pd.DataFrame(
        {
            "validation_row_index": prediction_probe_indices,
            "class_id": y_probe,
            "class_name": [CLASS_NAMES[int(value)] for value in y_probe],
        }
    ).to_csv(tables_dir / "prediction_probe_manifest.csv", index=False)

    class_weights = sqrt_class_weights(y_train, args.max_class_weight)
    model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
    memory = PredictionSoftTrustMemory()
    rng = np.random.default_rng(args.model_seed)

    round_rows: List[Dict[str, object]] = []
    local_rows: List[Dict[str, object]] = []
    decision_tables: List[pd.DataFrame] = []
    best_macro_f1 = -math.inf
    best_round = 0
    stale_rounds = 0
    started = time.time()

    print("Prediction Soft Trust V3.6.1")
    print("Mode:", args.mode)
    print("Model seed:", args.model_seed)
    print("Partition hash:", partition_hash)
    print("Malicious clients:", malicious_clients)
    print("Clean EMA threshold:", f"{clean_ema_threshold:.6f}")

    for round_id in range(1, args.rounds + 1):
        round_started = time.time()
        selected = np.sort(
            rng.choice(args.num_clients, size=args.num_clients, replace=False)
        )
        reference_state = copy.deepcopy(model.state_dict())
        reference_probabilities = predict_probabilities(
            model, X_probe, args.evaluation_batch_size
        )
        reference_class_means = class_conditional_probability_means(
            reference_probabilities, y_probe
        )
        reference_explanation = explanation_signature(
            model, X_explain, y_explain
        )

        local_states = []
        sample_counts = []
        probe_rows = []
        current_local_rows = []

        for client_id_value in selected:
            client_id = int(client_id_value)
            indices = client_indices[client_id]
            local_y = y_train[indices].copy()
            positions = poisoned_positions[client_id]
            if len(positions) > 0:
                local_y[positions] = target_id

            local_model = build_model("resmlp", X_train.shape[1], NUM_CLASSES)
            local_model.load_state_dict(reference_state)
            state, training_metrics = train_local_model(
                model=local_model,
                X=X_train[indices],
                y=local_y,
                class_weights=class_weights,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                seed=args.model_seed + round_id * 1000 + client_id,
            )
            signal = local_probe_signal(
                model=local_model,
                X_probe=X_probe,
                y_probe=y_probe,
                X_explain=X_explain,
                y_explain=y_explain,
                reference_class_means=reference_class_means,
                reference_explanation=reference_explanation,
                batch_size=args.evaluation_batch_size,
            )
            local_states.append(state)
            sample_counts.append(int(len(indices)))
            probe_rows.append({"client_id": client_id, **signal})
            local_row = {
                "round": round_id,
                "client_id": client_id,
                "is_malicious": client_id in malicious_clients,
                "client_samples": int(len(indices)),
                "poisoned_rows": int(len(positions)),
                **training_metrics,
                **signal,
            }
            local_rows.append(local_row)
            current_local_rows.append(local_row)
            del local_model

        decision = aggregate_prediction_soft_trust(
            reference_state=reference_state,
            local_states=local_states,
            client_ids=[int(value) for value in selected],
            sample_counts=sample_counts,
            probe_signals=pd.DataFrame(probe_rows),
            prediction_profiles=prediction_profiles,
            clean_ema_threshold=clean_ema_threshold,
            memory=memory,
            round_id=round_id,
            malicious_client_ids=malicious_clients,
            ema_decay=args.ema_decay,
            trust_gamma=args.trust_gamma,
            minimum_trust=args.minimum_trust,
            count_cap_multiplier=args.count_cap_multiplier,
            low_trust_cutoff=args.low_trust_cutoff,
        )
        model.load_state_dict(decision.aggregated_state)
        decision_tables.append(decision.client_decisions.copy())

        val_metrics, val_pair = evaluate_validation(
            model,
            X_val,
            y_val,
            source_id,
            target_id,
            args.evaluation_batch_size,
        )
        round_row = {
            "round": round_id,
            "selected_clients": "|".join(map(str, selected.tolist())),
            "selected_client_count": len(selected),
            "participating_samples": int(sum(sample_counts)),
            "mean_local_train_loss": float(
                np.mean([row["local_train_loss"] for row in current_local_rows])
            ),
            "mean_local_train_accuracy": float(
                np.mean([row["local_train_accuracy"] for row in current_local_rows])
            ),
            "round_seconds": float(time.time() - round_started),
            **decision.summary,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"val_{key}": value for key, value in val_pair.items()},
        }
        round_rows.append(round_row)

        print(
            f"round {round_id:02d}, val macro F1={val_metrics['macro_f1']:.4f}, "
            f"{args.source_class}->{args.target_class}={val_pair['source_to_target_rate']:.4%}, "
            f"malicious influence reduction="
            f"{decision.summary['malicious_influence_reduction']:.3f}, "
            f"benign trust={decision.summary['mean_benign_trust']:.3f}, "
            f"malicious trust={decision.summary['mean_malicious_trust']:.3f}"
        )

        checkpoint_payload = {
            "experiment_version": "3.6.1",
            "aggregation_rule": "personalized_prediction_drift_soft_trust",
            "mode": args.mode,
            "checkpoint_type": "last_round",
            "round": round_id,
            "model_state_dict": model.state_dict(),
            "input_dim": X_train.shape[1],
            "num_classes": NUM_CLASSES,
            "validation_metrics": val_metrics,
            "validation_source_target_metrics": val_pair,
            "defense_summary": decision.summary,
            "temporal_memory": {
                "prediction_residual_ema": memory.residual_ema,
            },
            "partition_hash": partition_hash,
            "poison_index_hash": poison_hash,
            "source_class": args.source_class,
            "target_class": args.target_class,
            "malicious_clients": malicious_clients,
            "clean_ema_threshold": clean_ema_threshold,
        }
        torch.save(
            checkpoint_payload,
            checkpoints_dir / "last_round_model.pt",
        )
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_round = round_id
            stale_rounds = 0
            best_payload = dict(checkpoint_payload)
            best_payload["checkpoint_type"] = "best_validation"
            torch.save(
                best_payload,
                checkpoints_dir / "best_validation_model.pt",
            )
        else:
            stale_rounds += 1

        pd.DataFrame(round_rows).to_csv(
            tables_dir / "round_metrics_partial.csv",
            index=False,
        )
        if (
            args.early_stopping_patience > 0
            and stale_rounds >= args.early_stopping_patience
        ):
            print(f"Early stopping after round {round_id}")
            break

    round_table = pd.DataFrame(round_rows)
    local_table = pd.DataFrame(local_rows)
    decisions_table = pd.concat(decision_tables, ignore_index=True)
    round_table.to_csv(tables_dir / "round_metrics.csv", index=False)
    local_table.to_csv(
        tables_dir / "local_client_training_metrics.csv", index=False
    )
    decisions_table.to_csv(
        tables_dir / "client_soft_trust_and_signals.csv", index=False
    )

    if not args.validation_only:
        metric_tables = []
        class_tables = []
        pair_tables = []
        for name, path in (
            ("best_validation", checkpoints_dir / "best_validation_model.pt"),
            ("last_round", checkpoints_dir / "last_round_model.pt"),
        ):
            metrics, per_class, pair = evaluate_checkpoint(
                name,
                path,
                arrays,
                source_id,
                target_id,
                args.evaluation_batch_size,
                tables_dir,
                args.save_predictions,
            )
            metric_tables.append(metrics)
            class_tables.append(per_class)
            pair_tables.append(pair)
        pd.concat(metric_tables, ignore_index=True).to_csv(
            tables_dir / "test_metrics.csv", index=False
        )
        pd.concat(class_tables, ignore_index=True).to_csv(
            tables_dir / "per_class_metrics.csv", index=False
        )
        pd.concat(pair_tables, ignore_index=True).to_csv(
            tables_dir / "source_target_metrics.csv", index=False
        )

    plot_validation(
        round_table,
        args.source_class,
        args.target_class,
        figures_dir / "validation_and_attack_dynamics",
    )
    plot_trust(round_table, figures_dir / "client_trust_dynamics")
    plot_influence(
        round_table,
        figures_dir / "malicious_influence_dynamics",
    )

    best_row = round_table[round_table["round"].eq(best_round)].iloc[0]
    metadata = {
        "experiment_version": "3.6.1",
        "experiment_type": "personalized_prediction_drift_soft_trust_defense",
        "status": "integrated_development_screen_not_final_paper_result",
        "mode": args.mode,
        "model": "ResidualMLP",
        "source_class": args.source_class,
        "target_class": args.target_class,
        "malicious_clients": malicious_clients,
        "partition_hash_sha256": partition_hash,
        "clean_seed_partition_hash_sha256": clean_hash,
        "poison_index_hash_sha256": poison_hash,
        "model_seed": args.model_seed,
        "attack_seed": args.attack_seed,
        "calibration_dir": str(calibration_dir),
        "clean_ema_threshold": clean_ema_threshold,
        "rounds_requested": args.rounds,
        "rounds_completed": int(round_table["round"].max()),
        "best_validation_round": int(best_round),
        "best_validation_macro_f1": float(best_macro_f1),
        "best_validation_source_to_target_rate": float(
            best_row["val_source_to_target_rate"]
        ),
        "best_round_malicious_influence_reduction": float(
            best_row["malicious_influence_reduction"]
        ),
        "best_round_mean_benign_trust": float(
            best_row["mean_benign_trust"]
        ),
        "test_sets_used_for_checkpoint_selection": False,
        "test_evaluation_performed": not args.validation_only,
        "total_seconds": float(time.time() - started),
        "parameters": {
            "ema_decay": args.ema_decay,
            "clean_ema_quantile": args.clean_ema_quantile,
            "trust_gamma": args.trust_gamma,
            "minimum_trust": args.minimum_trust,
            "count_cap_multiplier": args.count_cap_multiplier,
            "low_trust_cutoff": args.low_trust_cutoff,
        },
    }
    with (output_dir / "prediction_soft_trust_v361_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Prediction Soft Trust V3.6.1 complete")
    print("Mode:", args.mode)
    print("Best validation round:", best_round)
    print("Best validation macro F1:", f"{best_macro_f1:.6f}")
    print(
        "Best validation source-to-target rate:",
        f"{float(best_row['val_source_to_target_rate']):.6f}",
    )
    print(
        "Best-round malicious influence reduction:",
        f"{float(best_row['malicious_influence_reduction']):.6f}",
    )
    print(
        "Best-round mean benign trust:",
        f"{float(best_row['mean_benign_trust']):.6f}",
    )
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

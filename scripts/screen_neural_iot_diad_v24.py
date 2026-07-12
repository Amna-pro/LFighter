#!/usr/bin/env python3
"""Neural reliability screening for CIC IoT-DIAD Protocol V2.1.

This stage compares neural architectures and loss functions under leakage-safe
validation and test splits. Model selection uses validation macro F1 only.
Natural-test EM prior correction is estimated from unlabeled prediction
probabilities, while actual natural-test labels are used only for retrospective
evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neural_models_v24 import FocalLoss, build_model  # noqa: E402

CLASS_NAMES = [
    "Benign",
    "BruteForce",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web-Based",
]
NUM_CLASSES = len(CLASS_NAMES)


@dataclass(frozen=True)
class Candidate:
    name: str
    architecture: str
    loss_name: str


DEFAULT_CANDIDATES = [
    Candidate("mlp_ce", "mlp", "ce"),
    Candidate("mlp_sqrt", "mlp", "sqrt"),
    Candidate("resmlp_ce", "resmlp", "ce"),
    Candidate("resmlp_sqrt", "resmlp", "sqrt"),
    Candidate("resmlp_focal", "resmlp", "focal"),
    Candidate("cnn1d_ce", "cnn1d", "ce"),
    Candidate("cnn1d_sqrt", "cnn1d", "sqrt"),
    Candidate("cnn1d_focal", "cnn1d", "focal"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIC IoT-DIAD neural reliability screening V2.4.")
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--candidates",
        default=",".join(candidate.name for candidate in DEFAULT_CANDIDATES),
        help="Comma-separated candidate names.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--temperature-min", type=float, default=0.5)
    parser.add_argument("--temperature-max", type=float, default=5.0)
    parser.add_argument("--temperature-steps", type=int, default=80)
    parser.add_argument("--em-max-iter", type=int, default=200)
    parser.add_argument("--em-tolerance", type=float, default=1e-7)
    return parser.parse_args()


def parse_candidates(value: str) -> List[Candidate]:
    lookup = {candidate.name: candidate for candidate in DEFAULT_CANDIDATES}
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in lookup]
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}. Available: {sorted(lookup)}")
    if not names:
        raise ValueError("At least one candidate is required.")
    return [lookup[name] for name in names]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_arrays(path: Path) -> Dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prepared NPZ file not found: {path}")
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    required = {
        "X_train", "y_train", "X_val", "y_val",
        "X_test_natural", "y_test_natural",
        "X_test_diagnostic", "y_test_diagnostic",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise KeyError(f"Prepared NPZ is missing arrays: {missing}")
    return arrays


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32, copy=False)),
        torch.from_numpy(y.astype(np.int64, copy=False)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def class_weights(y: np.ndarray, max_weight: float) -> np.ndarray:
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    maximum = counts.max()
    weights = np.sqrt(maximum / np.clip(counts, 1.0, None))
    weights = np.clip(weights, 1.0, max_weight)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def build_criterion(loss_name: str, y_train: np.ndarray, max_weight: float, focal_gamma: float) -> nn.Module:
    weights = torch.tensor(class_weights(y_train, max_weight), dtype=torch.float32)
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    if loss_name == "sqrt":
        return nn.CrossEntropyLoss(weight=weights)
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=weights)
    raise ValueError(f"Unknown loss: {loss_name}")


def collect_logits(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            logits_list.append(model(features).cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return np.concatenate(logits_list), np.concatenate(labels_list)


def softmax_numpy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-8)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1e6, neginf=-1e6)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(np.clip(scaled, -700.0, 0.0))
    sums = exponentials.sum(axis=1, keepdims=True)
    zero_rows = sums[:, 0] <= 0.0
    if zero_rows.any():
        exponentials[zero_rows] = 1.0
        sums = exponentials.sum(axis=1, keepdims=True)
    return exponentials / sums


def optimize_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    minimum: float,
    maximum: float,
    steps: int,
) -> Tuple[float, pd.DataFrame]:
    temperatures = np.geomspace(minimum, maximum, steps)
    rows = []
    best_temperature = 1.0
    best_loss = math.inf
    for temperature in temperatures:
        probabilities = softmax_numpy(logits, temperature)
        loss = log_loss(labels, probabilities, labels=np.arange(NUM_CLASSES))
        rows.append({"temperature": float(temperature), "validation_log_loss": float(loss)})
        if loss < best_loss:
            best_loss = float(loss)
            best_temperature = float(temperature)
    return best_temperature, pd.DataFrame(rows)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correctness = (predicted == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    return ece


def metric_dict(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "log_loss": float(log_loss(y_true, probabilities, labels=np.arange(NUM_CLASSES))),
        "ece_15bin": float(expected_calibration_error(y_true, probabilities, bins=15)),
    }


def em_label_shift(
    source_probabilities: np.ndarray,
    source_prior: np.ndarray,
    max_iter: int,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Estimate target priors from unlabeled predictions via EM label-shift adjustment."""
    probabilities = np.clip(np.asarray(source_probabilities, dtype=np.float64), 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    source_prior = np.clip(np.asarray(source_prior, dtype=np.float64), 1e-12, None)
    source_prior /= source_prior.sum()
    target_prior = probabilities.mean(axis=0)
    target_prior = np.clip(target_prior, 1e-12, None)
    target_prior /= target_prior.sum()

    final_delta = math.inf
    adjusted = probabilities.copy()
    for iteration in range(1, max_iter + 1):
        ratios = target_prior / source_prior
        adjusted = probabilities * ratios[None, :]
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        new_prior = adjusted.mean(axis=0)
        new_prior = np.clip(new_prior, 1e-12, None)
        new_prior /= new_prior.sum()
        final_delta = float(np.max(np.abs(new_prior - target_prior)))
        target_prior = new_prior
        if final_delta < tolerance:
            break
    return adjusted, target_prior, iteration, final_delta


def parameter_size_mb(model: nn.Module) -> float:
    bytes_total = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    return float(bytes_total / (1024 * 1024))


def train_candidate(
    candidate: Candidate,
    arrays: Dict[str, np.ndarray],
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[nn.Module, pd.DataFrame, float, int]:
    set_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    X_train = arrays["X_train"].astype(np.float32, copy=False)
    y_train = arrays["y_train"].astype(np.int64, copy=False)
    X_val = arrays["X_val"].astype(np.float32, copy=False)
    y_val = arrays["y_val"].astype(np.int64, copy=False)

    model = build_model(candidate.architecture, X_train.shape[1], NUM_CLASSES)
    criterion = build_criterion(candidate.loss_name, y_train, args.max_class_weight, args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )
    train_loader = make_loader(X_train, y_train, args.batch_size, True)
    val_loader = make_loader(X_val, y_val, args.batch_size, False)

    best_state = copy.deepcopy(model.state_dict())
    best_macro_f1 = -1.0
    best_epoch = 0
    stale = 0
    history: List[Dict[str, float]] = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        row_count = 0
        train_predictions: List[np.ndarray] = []
        train_targets: List[np.ndarray] = []

        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for {candidate.name} at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            rows = len(targets)
            running_loss += float(loss.item()) * rows
            row_count += rows
            train_predictions.append(logits.detach().argmax(dim=1).cpu().numpy())
            train_targets.append(targets.cpu().numpy())

        val_logits, val_targets = collect_logits(model, val_loader)
        val_probabilities = softmax_numpy(val_logits)
        val_metrics = metric_dict(val_targets, val_probabilities)
        train_pred = np.concatenate(train_predictions)
        train_true = np.concatenate(train_targets)
        train_macro_f1 = f1_score(train_true, train_pred, average="macro", zero_division=0)
        scheduler.step(val_metrics["macro_f1"])

        history.append(
            {
                "candidate": candidate.name,
                "epoch": epoch,
                "train_loss": running_loss / max(row_count, 1),
                "train_macro_f1": float(train_macro_f1),
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_log_loss": val_metrics["log_loss"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"  epoch {epoch:02d} | loss {history[-1]['train_loss']:.4f} | "
            f"train macro F1 {train_macro_f1:.4f} | val macro F1 {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1 + 1e-5:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    train_seconds = time.time() - start
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "candidate": candidate.name,
            "architecture": candidate.architecture,
            "loss": candidate.loss_name,
            "seed": args.seed,
            "input_dim": X_train.shape[1],
            "num_classes": NUM_CLASSES,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_macro_f1,
            "state_dict": model.state_dict(),
        },
        checkpoint_dir / f"{candidate.name}_seed{args.seed}.pt",
    )
    return model, pd.DataFrame(history), train_seconds, best_epoch


def evaluate_candidate(
    candidate: Candidate,
    model: nn.Module,
    arrays: Dict[str, np.ndarray],
    args: argparse.Namespace,
    train_seconds: float,
    best_epoch: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    loaders = {
        "val": make_loader(arrays["X_val"], arrays["y_val"], args.batch_size, False),
        "test_natural_raw": make_loader(
            arrays["X_test_natural"], arrays["y_test_natural"], args.batch_size, False
        ),
        "test_diagnostic": make_loader(
            arrays["X_test_diagnostic"], arrays["y_test_diagnostic"], args.batch_size, False
        ),
    }
    logits: Dict[str, np.ndarray] = {}
    labels: Dict[str, np.ndarray] = {}
    for split, loader in loaders.items():
        logits[split], labels[split] = collect_logits(model, loader)

    temperature, temperature_grid = optimize_temperature(
        logits["val"],
        labels["val"],
        args.temperature_min,
        args.temperature_max,
        args.temperature_steps,
    )
    temperature_grid.insert(0, "candidate", candidate.name)

    probabilities = {
        split: softmax_numpy(split_logits, temperature)
        for split, split_logits in logits.items()
    }
    source_prior = np.bincount(arrays["y_train"], minlength=NUM_CLASSES).astype(np.float64)
    source_prior /= source_prior.sum()
    adjusted, estimated_prior, em_iterations, em_delta = em_label_shift(
        probabilities["test_natural_raw"],
        source_prior,
        args.em_max_iter,
        args.em_tolerance,
    )
    probabilities["test_natural_em"] = adjusted
    labels["test_natural_em"] = labels["test_natural_raw"]

    actual_natural_prior = np.bincount(labels["test_natural_raw"], minlength=NUM_CLASSES).astype(np.float64)
    actual_natural_prior /= actual_natural_prior.sum()
    prior_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        prior_rows.append(
            {
                "candidate": candidate.name,
                "class_id": class_id,
                "class_name": class_name,
                "source_train_prior": float(source_prior[class_id]),
                "estimated_natural_prior_unlabeled": float(estimated_prior[class_id]),
                "actual_natural_prior_retrospective": float(actual_natural_prior[class_id]),
                "absolute_estimation_error": float(abs(estimated_prior[class_id] - actual_natural_prior[class_id])),
                "em_iterations": em_iterations,
                "em_final_delta": em_delta,
            }
        )

    all_runs: List[Dict[str, object]] = []
    per_class: List[Dict[str, object]] = []
    predictions: Dict[str, np.ndarray] = {}
    for split in ("val", "test_natural_raw", "test_natural_em", "test_diagnostic"):
        probs = probabilities[split]
        y_true = labels[split]
        y_pred = probs.argmax(axis=1)
        predictions[split] = y_pred
        metrics = metric_dict(y_true, probs)
        all_runs.append(
            {
                "candidate": candidate.name,
                "architecture": candidate.architecture,
                "loss": candidate.loss_name,
                "seed": args.seed,
                "split": split,
                "train_seconds": train_seconds,
                "best_epoch": best_epoch,
                "model_size_mb": parameter_size_mb(model),
                "temperature": temperature,
                **metrics,
            }
        )
        report = classification_report(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        for class_id, class_name in enumerate(CLASS_NAMES):
            row = report[class_name]
            per_class.append(
                {
                    "candidate": candidate.name,
                    "architecture": candidate.architecture,
                    "loss": candidate.loss_name,
                    "seed": args.seed,
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1_score": float(row["f1-score"]),
                    "support": int(row["support"]),
                }
            )

    evaluation_arrays = {
        "val_true": labels["val"],
        "val_pred": predictions["val"],
        "natural_true": labels["test_natural_raw"],
        "natural_raw_pred": predictions["test_natural_raw"],
        "natural_em_pred": predictions["test_natural_em"],
        "diagnostic_true": labels["test_diagnostic"],
        "diagnostic_pred": predictions["test_diagnostic"],
    }
    return all_runs, per_class, temperature_grid, pd.DataFrame(prior_rows), evaluation_arrays


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_candidate_performance(all_runs: pd.DataFrame, output: Path) -> None:
    subset = all_runs[all_runs["split"].isin(["val", "test_diagnostic", "test_natural_raw", "test_natural_em"])]
    pivot = subset.pivot(index="candidate", columns="split", values="macro_f1")
    columns = [column for column in ["val", "test_diagnostic", "test_natural_raw", "test_natural_em"] if column in pivot]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pivot.index))
    width = 0.8 / len(columns)
    for index, column in enumerate(columns):
        ax.bar(x + (index - (len(columns)-1)/2) * width, pivot[column], width, label=column)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1")
    ax.set_title("Neural Candidate Reliability Across Leakage-Safe Splits")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def plot_runtime_tradeoff(all_runs: pd.DataFrame, output: Path) -> None:
    subset = all_runs[all_runs["split"] == "test_diagnostic"].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(subset["train_seconds"], subset["macro_f1"], s=70)
    for _, row in subset.iterrows():
        ax.annotate(str(row["candidate"]), (row["train_seconds"], row["macro_f1"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Training time, seconds")
    ax.set_ylabel("Diagnostic-test macro F1")
    ax.set_title("Neural Reliability and Computational Cost")
    ax.grid(alpha=0.25)
    save_figure(fig, output)


def plot_calibration(all_runs: pd.DataFrame, output: Path) -> None:
    subset = all_runs[all_runs["split"].isin(["val", "test_natural_raw", "test_natural_em", "test_diagnostic"])]
    pivot = subset.pivot(index="candidate", columns="split", values="ece_15bin")
    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=max(0.01, float(pivot.to_numpy().max())))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Expected Calibration Error, Lower Is Better")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="ECE")
    save_figure(fig, output)


def plot_per_class_heatmap(per_class: pd.DataFrame, split: str, output: Path) -> None:
    subset = per_class[per_class["split"] == split]
    pivot = subset.pivot(index="candidate", columns="class_name", values="f1_score").reindex(columns=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Per-Class F1, {split}")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="F1")
    save_figure(fig, output)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, output: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right")
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized proportion")
    save_figure(fig, output)


def plot_training_curves(history: pd.DataFrame, candidate: str, output: Path) -> None:
    subset = history[history["candidate"] == candidate]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(subset["epoch"], subset["train_macro_f1"], label="train macro F1")
    ax.plot(subset["epoch"], subset["val_macro_f1"], label="validation macro F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.set_title(f"Training Reliability, {candidate}")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def plot_prior_estimates(priors: pd.DataFrame, candidate: str, output: Path) -> None:
    subset = priors[priors["candidate"] == candidate].set_index("class_name").reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(NUM_CLASSES)
    width = 0.26
    ax.bar(x - width, subset["source_train_prior"], width, label="source train")
    ax.bar(x, subset["estimated_natural_prior_unlabeled"], width, label="EM estimate, unlabeled")
    ax.bar(x + width, subset["actual_natural_prior_retrospective"], width, label="actual, retrospective")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_ylabel("Class prior")
    ax.set_title(f"Unlabeled Natural-Prior Estimation, {candidate}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def main() -> int:
    args = parse_args()
    candidates = parse_candidates(args.candidates)
    arrays = load_arrays(args.data_file)
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_runs_rows: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []
    history_frames: List[pd.DataFrame] = []
    temperature_frames: List[pd.DataFrame] = []
    prior_frames: List[pd.DataFrame] = []
    evaluation_by_candidate: Dict[str, Dict[str, np.ndarray]] = {}

    for candidate in candidates:
        print(f"\nTraining {candidate.name}, seed {args.seed}")
        model, history, train_seconds, best_epoch = train_candidate(
            candidate, arrays, args, output_dir
        )
        run_rows, class_rows, temperature_grid, priors, evaluation_arrays = evaluate_candidate(
            candidate, model, arrays, args, train_seconds, best_epoch
        )
        all_runs_rows.extend(run_rows)
        per_class_rows.extend(class_rows)
        history_frames.append(history)
        temperature_frames.append(temperature_grid)
        prior_frames.append(priors)
        evaluation_by_candidate[candidate.name] = evaluation_arrays

    all_runs = pd.DataFrame(all_runs_rows)
    per_class = pd.DataFrame(per_class_rows)
    history = pd.concat(history_frames, ignore_index=True)
    temperatures = pd.concat(temperature_frames, ignore_index=True)
    priors = pd.concat(prior_frames, ignore_index=True)

    ranking = (
        all_runs[all_runs["split"] == "val"]
        .sort_values(["macro_f1", "balanced_accuracy", "log_loss"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    ranking.insert(0, "validation_rank", np.arange(1, len(ranking) + 1))
    best_candidate = str(ranking.iloc[0]["candidate"])

    all_runs.to_csv(tables_dir / "all_runs.csv", index=False)
    per_class.to_csv(tables_dir / "per_class_metrics.csv", index=False)
    history.to_csv(tables_dir / "training_history.csv", index=False)
    temperatures.to_csv(tables_dir / "temperature_grid.csv", index=False)
    priors.to_csv(tables_dir / "prior_estimates.csv", index=False)
    ranking.to_csv(tables_dir / "candidate_ranking_validation_only.csv", index=False)

    best_eval = evaluation_by_candidate[best_candidate]
    for split_name, y_true_key, y_pred_key in [
        ("natural_raw", "natural_true", "natural_raw_pred"),
        ("natural_em", "natural_true", "natural_em_pred"),
        ("diagnostic", "diagnostic_true", "diagnostic_pred"),
    ]:
        matrix = confusion_matrix(
            best_eval[y_true_key], best_eval[y_pred_key], labels=np.arange(NUM_CLASSES)
        )
        pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            tables_dir / f"best_{best_candidate}_{split_name}_confusion_matrix.csv"
        )

    plot_candidate_performance(all_runs, figures_dir / "candidate_macro_f1")
    plot_runtime_tradeoff(all_runs, figures_dir / "runtime_reliability_tradeoff")
    plot_calibration(all_runs, figures_dir / "candidate_calibration")
    plot_per_class_heatmap(per_class, "test_natural_raw", figures_dir / "natural_raw_per_class_f1")
    plot_per_class_heatmap(per_class, "test_natural_em", figures_dir / "natural_em_per_class_f1")
    plot_per_class_heatmap(per_class, "test_diagnostic", figures_dir / "diagnostic_per_class_f1")
    plot_training_curves(history, best_candidate, figures_dir / "best_candidate_training_curve")
    plot_prior_estimates(priors, best_candidate, figures_dir / "best_candidate_prior_estimation")
    plot_confusion(
        best_eval["natural_true"],
        best_eval["natural_raw_pred"],
        f"{best_candidate}, Natural Test, Raw",
        figures_dir / "best_candidate_natural_raw_confusion",
    )
    plot_confusion(
        best_eval["natural_true"],
        best_eval["natural_em_pred"],
        f"{best_candidate}, Natural Test, EM Prior Correction",
        figures_dir / "best_candidate_natural_em_confusion",
    )
    plot_confusion(
        best_eval["diagnostic_true"],
        best_eval["diagnostic_pred"],
        f"{best_candidate}, Diagnostic Test",
        figures_dir / "best_candidate_diagnostic_confusion",
    )

    metadata = {
        "screening_version": "2.4",
        "selection_rule": "Highest validation macro F1, then validation balanced accuracy, then validation log loss.",
        "test_sets_used_for_selection": False,
        "natural_em_uses_test_labels": False,
        "natural_actual_priors_are_retrospective_only": True,
        "data_file": str(args.data_file.expanduser().resolve()),
        "seed": args.seed,
        "candidates": [candidate.name for candidate in candidates],
        "best_candidate_validation_only": best_candidate,
        "tables": str(tables_dir),
        "figures_png_pdf": str(figures_dir),
        "notes": [
            "This is a one-seed screening stage, not a final paper estimate.",
            "BruteForce has very low natural-test support and requires uncertainty reporting.",
            "EM prior correction is exploratory because the shift audit also detected class-conditional feature shift.",
        ],
    }
    with (output_dir / "neural_screening_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nNeural screening V2.4 complete")
    print(ranking[["validation_rank", "candidate", "macro_f1", "balanced_accuracy", "log_loss", "train_seconds"]].to_string(index=False))
    print(f"\nBest candidate by validation only: {best_candidate}")
    print(f"CSV tables: {tables_dir}")
    print(f"PNG and PDF figures: {figures_dir}")
    print(f"Metadata: {output_dir / 'neural_screening_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

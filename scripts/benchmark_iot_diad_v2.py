#!/usr/bin/env python3
"""Multi-model, multi-seed centralized benchmark for Data Protocol V2."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from iot_diad_protocol_v2 import CLASS_NAMES  # noqa: E402
from research_plotting import save_figure  # noqa: E402
from tabular_models import TabularMLP  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CIC IoT-DIAD V2 centralized benchmark.")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "cic_iot_diad_2024_v2" / "arrays" / "behavioral_only.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "cic_iot_diad_benchmark_v2",
    )
    parser.add_argument("--models", default="sgd,randomforest,extratrees,mlp")
    parser.add_argument("--seeds", default="42,52,62,72,82")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--mlp-epochs", type=int, default=40)
    parser.add_argument("--mlp-patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_arrays(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Prepared data file not found: {path}")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}




def sanitize_probabilities(probabilities: np.ndarray, num_classes: int) -> Tuple[np.ndarray, Dict[str, int]]:
    """Return finite, normalized probabilities and diagnostics for numerical repairs."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"Probability matrix must be 2D, received shape {probs.shape}")

    nonfinite_values = int((~np.isfinite(probs)).sum())
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    negative_values = int((probs < 0.0).sum())
    probs = np.clip(probs, 0.0, None)

    row_sums = probs.sum(axis=1, keepdims=True)
    zero_sum_mask = row_sums[:, 0] <= 0.0
    zero_sum_rows = int(zero_sum_mask.sum())
    if zero_sum_rows:
        probs[zero_sum_mask] = 1.0 / num_classes
        row_sums = probs.sum(axis=1, keepdims=True)

    probs = probs / row_sums
    original = np.asarray(probabilities, dtype=np.float64)
    nonfinite_row_mask = np.any(~np.isfinite(original), axis=1)
    negative_row_mask = np.any(original < 0.0, axis=1)
    repaired_rows = int((nonfinite_row_mask | negative_row_mask | zero_sum_mask).sum())
    diagnostics = {
        "nonfinite_values": nonfinite_values,
        "negative_values": negative_values,
        "zero_sum_rows": zero_sum_rows,
        "repaired_rows": repaired_rows,
    }
    return probs, diagnostics


def predict_sklearn(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None, Dict[str, int]]:
    """Predict labels and stable probabilities without sklearn's unstable SGD normalization path."""
    prediction = model.predict(X)
    empty = {"nonfinite_values": 0, "negative_values": 0, "zero_sum_rows": 0, "repaired_rows": 0}

    if isinstance(model, SGDClassifier):
        scores = np.asarray(model.decision_function(X), dtype=np.float64)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        probabilities = np.exp(np.clip(scores, -700.0, 0.0))
        probabilities, diagnostics = sanitize_probabilities(probabilities, probabilities.shape[1])
        return prediction, probabilities, diagnostics

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        probabilities, diagnostics = sanitize_probabilities(probabilities, len(CLASS_NAMES))
        return prediction, probabilities, diagnostics

    return prediction, None, empty

def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray | None = None) -> Dict[str, float]:
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if probabilities is not None:
        probabilities, _ = sanitize_probabilities(probabilities, len(CLASS_NAMES))
        probabilities = np.clip(probabilities, 1e-12, None)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        result["log_loss"] = log_loss(y_true, probabilities, labels=np.arange(len(CLASS_NAMES)))
        confidence = probabilities.max(axis=1)
        correctness = (probabilities.argmax(axis=1) == y_true).astype(float)
        bins = np.linspace(0.0, 1.0, 16)
        ece = 0.0
        for left, right in zip(bins[:-1], bins[1:]):
            mask = (confidence > left) & (confidence <= right)
            if bool(mask.any()):
                ece += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
        result["ece_15bin"] = ece
    else:
        result["log_loss"] = math.nan
        result["ece_15bin"] = math.nan
    return result


def build_sklearn_model(name: str, seed: int, threads: int):
    if name == "sgd":
        return SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-5,
            l1_ratio=0.05,
            class_weight="balanced",
            max_iter=250,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=12,
            random_state=seed,
            n_jobs=threads,
        )
    if name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=threads,
            random_state=seed,
        )
    if name == "randomforest":
        return RandomForestClassifier(
            n_estimators=220,
            max_depth=28,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=threads,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64))),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def predict_mlp(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    with torch.no_grad():
        for features, _ in loader:
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            probabilities.append(probs.numpy())
            predictions.append(probs.argmax(dim=1).numpy())
    return np.concatenate(predictions), np.concatenate(probabilities)


def train_mlp(
    arrays: Dict[str, np.ndarray],
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    threads: int,
    output_path: Path,
) -> Tuple[nn.Module, pd.DataFrame]:
    set_seed(seed)
    torch.set_num_threads(max(1, threads))
    X_train = arrays["X_train"].astype(np.float32)
    y_train = arrays["y_train"].astype(np.int64)
    X_val = arrays["X_val"].astype(np.float32)
    y_val = arrays["y_val"].astype(np.int64)
    train_loader = make_loader(X_train, y_train, batch_size, True)
    val_loader = make_loader(X_val, y_val, batch_size, False)
    model = TabularMLP(input_dim=X_train.shape[1], num_classes=len(CLASS_NAMES), hidden_dims=(256, 128, 64), dropout=0.20)
    weights = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: List[Dict[str, float]] = []
    best_score = -1.0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            total_rows += len(targets)
        val_pred, _ = predict_mlp(model, val_loader)
        val_macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_rows, 1),
                "val_macro_f1": val_macro_f1,
            }
        )
        if val_macro_f1 > best_score + 1e-5:
            best_score = val_macro_f1
            stale = 0
            torch.save(model.state_dict(), output_path)
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(torch.load(output_path, map_location="cpu", weights_only=True))
    return model, pd.DataFrame(history)


def save_run_outputs(
    output_dir: Path,
    model_name: str,
    seed: int,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    per_class = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class_name"})
    per_class["model"] = model_name
    per_class["seed"] = seed
    per_class["split"] = split
    confusion = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES))),
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )
    run_dir = output_dir / "runs" / model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(run_dir / f"classification_report_{split}.csv", index=False)
    confusion.to_csv(run_dir / f"confusion_matrix_{split}.csv")
    return per_class, confusion


def plot_metric_summary(summary: pd.DataFrame, output_base: Path) -> None:
    metrics = ["macro_f1", "balanced_accuracy", "accuracy"]
    models = list(summary["model"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.8 / len(metrics)
    for index, metric in enumerate(metrics):
        subset = summary[summary["metric"] == metric].set_index("model").reindex(models)
        ax.bar(
            x + (index - 1) * width,
            subset["mean"].values,
            width,
            yerr=subset["std"].fillna(0).values,
            capsize=3,
            label=metric,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Centralized Benchmark on the Natural Test Set")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_base)


def plot_runtime_tradeoff(runs: pd.DataFrame, output_base: Path) -> None:
    data = runs[runs["split"] == "test_natural"].groupby("model", as_index=False).agg(
        macro_f1=("macro_f1", "mean"),
        train_seconds=("train_seconds", "mean"),
        model_size_mb=("model_size_mb", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data["train_seconds"], data["macro_f1"], s=90)
    for row in data.itertuples(index=False):
        ax.annotate(row.model, (row.train_seconds, row.macro_f1), xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Mean training time, seconds, log scale")
    ax.set_ylabel("Mean macro F1")
    ax.set_title("Accuracy and Computational-Cost Trade-off")
    ax.grid(alpha=0.25)
    save_figure(fig, output_base)


def plot_per_class_heatmap(per_class: pd.DataFrame, output_base: Path) -> None:
    data = per_class[(per_class["split"] == "test_natural") & (per_class["class_name"].isin(CLASS_NAMES))]
    pivot = data.groupby(["model", "class_name"])["f1-score"].mean().unstack().reindex(columns=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Per-Class F1 Across Centralized Models")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, output_base)


def plot_best_confusion(confusion: pd.DataFrame, output_base: Path, title: str) -> None:
    values = confusion.to_numpy(dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    image = ax.imshow(normalized, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, output_base)


def main() -> int:
    args = parse_args()
    models = parse_csv_list(args.models)
    seeds = [int(value) for value in parse_csv_list(args.seeds)]
    supported = {"sgd", "randomforest", "extratrees", "mlp"}
    unknown = sorted(set(models) - supported)
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    arrays = load_arrays(args.data_file.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, object]] = []
    per_class_rows: List[pd.DataFrame] = []
    confusions: Dict[Tuple[str, int, str], pd.DataFrame] = {}
    mlp_histories: List[pd.DataFrame] = []
    probability_diagnostic_rows: List[Dict[str, object]] = []

    for model_name in models:
        for seed in seeds:
            print(f"Training {model_name}, seed {seed}")
            set_seed(seed)
            start = time.time()
            model_path = output_dir / "models" / f"{model_name}_seed{seed}"
            if model_name == "mlp":
                checkpoint = model_path.with_suffix(".pt")
                model, history = train_mlp(
                    arrays,
                    seed=seed,
                    epochs=args.mlp_epochs,
                    patience=args.mlp_patience,
                    batch_size=args.batch_size,
                    threads=args.threads,
                    output_path=checkpoint,
                )
                history["model"] = model_name
                history["seed"] = seed
                mlp_histories.append(history)
                train_seconds = time.time() - start
                predictors = {}
                for split in ("val", "test_natural", "test_diagnostic"):
                    loader = make_loader(arrays[f"X_{split}"], arrays[f"y_{split}"], args.batch_size, False)
                    predictors[split] = predict_mlp(model, loader)
                model_size = checkpoint.stat().st_size / (1024 * 1024)
            else:
                model = build_sklearn_model(model_name, seed, args.threads)
                model.fit(arrays["X_train"], arrays["y_train"])
                train_seconds = time.time() - start
                saved = model_path.with_suffix(".joblib")
                joblib.dump(model, saved, compress=3)
                model_size = saved.stat().st_size / (1024 * 1024)
                predictors = {}
                for split in ("val", "test_natural", "test_diagnostic"):
                    prediction, probability, diagnostics = predict_sklearn(model, arrays[f"X_{split}"])
                    predictors[split] = (prediction, probability, diagnostics)

            if model_name == "mlp":
                predictors = {
                    split: (prediction, probability, {"nonfinite_values": 0, "negative_values": 0, "zero_sum_rows": 0, "repaired_rows": 0})
                    for split, (prediction, probability) in predictors.items()
                }

            for split, (prediction, probability, probability_diagnostics) in predictors.items():
                metrics = metric_dict(arrays[f"y_{split}"], prediction, probability)
                probability_diagnostic_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "split": split,
                        **probability_diagnostics,
                    }
                )
                run_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "split": split,
                        "train_seconds": train_seconds,
                        "model_size_mb": model_size,
                        **metrics,
                    }
                )
                per_class, confusion = save_run_outputs(
                    output_dir,
                    model_name,
                    seed,
                    split,
                    arrays[f"y_{split}"],
                    prediction,
                )
                per_class_rows.append(per_class)
                confusions[(model_name, seed, split)] = confusion

    runs = pd.DataFrame(run_rows)
    runs.to_csv(output_dir / "tables" / "all_runs.csv", index=False)
    pd.DataFrame(probability_diagnostic_rows).to_csv(
        output_dir / "tables" / "probability_diagnostics.csv", index=False
    )
    per_class_all = pd.concat(per_class_rows, ignore_index=True)
    per_class_all.to_csv(output_dir / "tables" / "per_class_metrics.csv", index=False)
    if mlp_histories:
        pd.concat(mlp_histories, ignore_index=True).to_csv(
            output_dir / "tables" / "mlp_training_history.csv", index=False
        )

    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc", "log_loss", "ece_15bin"]
    summary_rows: List[Dict[str, object]] = []
    natural = runs[runs["split"] == "test_natural"]
    for model_name, group in natural.groupby("model"):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            summary_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": float(np.mean(values)) if len(values) else math.nan,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_half_width": float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "tables" / "aggregate_metrics.csv", index=False)

    statistical_rows: List[Dict[str, object]] = []
    pivot = natural.pivot(index="seed", columns="model", values="macro_f1")
    model_names = list(pivot.columns)
    for left_index, left in enumerate(model_names):
        for right in model_names[left_index + 1 :]:
            paired = pivot[[left, right]].dropna()
            if len(paired) >= 3 and not np.allclose(paired[left], paired[right]):
                statistic, p_value = wilcoxon(paired[left], paired[right], zero_method="wilcox")
            else:
                statistic, p_value = math.nan, math.nan
            statistical_rows.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "paired_seeds": len(paired),
                    "mean_difference_macro_f1": float((paired[left] - paired[right]).mean()) if len(paired) else math.nan,
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
    pd.DataFrame(statistical_rows).to_csv(output_dir / "tables" / "pairwise_wilcoxon_macro_f1.csv", index=False)

    plot_metric_summary(summary, output_dir / "figures" / "centralized_model_performance")
    plot_runtime_tradeoff(runs, output_dir / "figures" / "runtime_performance_tradeoff")
    plot_per_class_heatmap(per_class_all, output_dir / "figures" / "per_class_f1_heatmap")

    best_row = natural.sort_values("macro_f1", ascending=False).iloc[0]
    best_key = (str(best_row["model"]), int(best_row["seed"]), "test_natural")
    plot_best_confusion(
        confusions[best_key],
        output_dir / "figures" / "best_model_normalized_confusion_matrix",
        f"Normalized Confusion Matrix, {best_key[0]}, seed {best_key[1]}",
    )

    metadata = {
        "benchmark_version": "2.2",
        "data_file": str(args.data_file.expanduser().resolve()),
        "models": models,
        "seeds": seeds,
        "best_natural_test_run": best_row.to_dict(),
        "tables": str(output_dir / "tables"),
        "figures_png_pdf": str(output_dir / "figures"),
    }
    with (output_dir / "benchmark_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)

    print()
    print("Benchmark complete")
    print(summary[summary["metric"].isin(["macro_f1", "balanced_accuracy", "accuracy"])].to_string(index=False))
    print(f"CSV tables: {output_dir / 'tables'}")
    print(f"PNG and PDF figures: {output_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

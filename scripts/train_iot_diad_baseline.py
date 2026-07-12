#!/usr/bin/env python3
"""Train and evaluate the clean centralized CIC IoT-DIAD 2024 MLP baseline."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from iot_diad_dataset import CLASS_NAMES  # noqa: E402
from tabular_models import TabularMLP  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the clean centralized MLP baseline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "cic_iot_diad_2024",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "cic_iot_diad_centralized",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Prepared data file not found: {path}")
    with np.load(path) as data:
        return data["X"].astype(np.float32), data["y"].astype(np.int64)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training split is missing classes: counts={counts.tolist()}")
    weights = len(y) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    all_targets = []
    all_predictions = []

    for features, targets in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_targets.append(targets.numpy())
        all_predictions.append(logits.argmax(dim=1).detach().numpy())

    targets_np = np.concatenate(all_targets)
    predictions_np = np.concatenate(all_predictions)
    return total_loss / max(total_examples, 1), targets_np, predictions_np


def metric_row(prefix: str, loss: float, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_loss": loss,
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_npz(data_dir / "train.npz")
    X_val, y_val = load_npz(data_dir / "val.npz")
    X_test, y_test = load_npz(data_dir / "test.npz")

    if X_train.shape[1] != X_val.shape[1] or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Feature dimensions do not match across splits")

    train_loader = make_loader(
        X_train, y_train, args.batch_size, True, args.num_workers
    )
    val_loader = make_loader(X_val, y_val, args.batch_size, False, args.num_workers)
    test_loader = make_loader(X_test, y_test, args.batch_size, False, args.num_workers)

    model = TabularMLP(input_dim=X_train.shape[1], num_classes=len(CLASS_NAMES))
    weights = class_weights(y_train, len(CLASS_NAMES))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_path = output_dir / "best_model.pt"
    history = []
    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0
    start = time.time()

    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Class weights: {weights.tolist()}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_true, train_pred = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_true, val_pred = run_epoch(model, val_loader, criterion, None)

        row = {"epoch": epoch}
        row.update(metric_row("train", train_loss, train_true, train_pred))
        row.update(metric_row("val", val_loss, val_true, val_pred))
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f}, macro F1 {row['train_macro_f1']:.4f} | "
            f"val loss {val_loss:.4f}, macro F1 {row['val_macro_f1']:.4f}"
        )

        if row["val_macro_f1"] > best_val_macro_f1 + 1e-5:
            best_val_macro_f1 = row["val_macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": X_train.shape[1],
                    "class_names": list(CLASS_NAMES),
                    "epoch": epoch,
                    "val_macro_f1": best_val_macro_f1,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_true, test_pred = run_epoch(model, test_loader, criterion, None)

    report_dict = classification_report(
        test_true,
        test_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        output_dir / "classification_report.csv"
    )

    cm = confusion_matrix(test_true, test_pred, labels=list(range(len(CLASS_NAMES))))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    test_metrics = metric_row("test", test_loss, test_true, test_pred)
    final_summary = {
        **test_metrics,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_macro_f1": float(checkpoint["val_macro_f1"]),
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "class_names": list(CLASS_NAMES),
        "class_weights": weights.tolist(),
        "elapsed_seconds": round(time.time() - start, 2),
        "torch_version": torch.__version__,
        "device": "cpu",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)

    print()
    print("Test results")
    print(json.dumps(test_metrics, indent=2))
    print(f"Best checkpoint: {best_path}")
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

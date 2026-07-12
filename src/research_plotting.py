"""Publication-oriented plotting helpers.

Each figure is saved as both high-resolution PNG and vector PDF. The source
values used to build the figure are saved separately by the caller as CSV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_raw_class_distribution(table: pd.DataFrame, output_base: Path) -> None:
    data = table.groupby("class_name", as_index=False)["rows"].sum().sort_values("rows", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(data["class_name"], data["rows"])
    ax.set_yscale("log")
    ax.set_xlabel("Class")
    ax.set_ylabel("Raw flow records, log scale")
    ax.set_title("CIC IoT-DIAD 2024 Raw Class Distribution")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_base)


def plot_split_distribution(table: pd.DataFrame, output_base: Path, title: str) -> None:
    pivot = table.pivot(index="class_name", columns="split", values="rows").fillna(0)
    desired = [column for column in ("train", "val", "test", "test_natural", "test_diagnostic") if column in pivot.columns]
    pivot = pivot.reindex(columns=desired)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(pivot.columns), 1)
    for index, column in enumerate(pivot.columns):
        ax.bar(x + (index - (len(pivot.columns) - 1) / 2) * width, pivot[column].values, width, label=column)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_ylabel("Flow records")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_base)


def plot_capture_counts(manifest: pd.DataFrame, output_base: Path) -> None:
    expanded = manifest.copy()
    expanded["display_split"] = expanded["assigned_split"].replace({"mixed": "within-capture"})
    counts = expanded.groupby(["class_name", "display_split"], as_index=False).size().rename(columns={"size": "files"})
    pivot = counts.pivot(index="class_name", columns="display_split", values="files").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(pivot))
    for column in pivot.columns:
        values = pivot[column].to_numpy()
        ax.bar(pivot.index, values, bottom=bottom, label=column)
        bottom += values
    ax.set_ylabel("Capture files")
    ax.set_title("Capture-Level Split Allocation")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    save_figure(fig, output_base)


def plot_missingness(table: pd.DataFrame, output_base: Path, top_n: int = 20) -> None:
    data = table.head(top_n).sort_values("missing_rate")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(data["feature"], data["missing_rate"])
    ax.set_xlabel("Missing or invalid value rate")
    ax.set_title(f"Top {min(top_n, len(data))} Features Requiring Imputation")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, output_base)


def plot_cleaning_summary(table: pd.DataFrame, output_base: Path) -> None:
    data = table.set_index("split")
    columns = [
        "within_split_duplicates_removed",
        "cross_split_duplicates_removed",
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = np.zeros(len(data))
    for column in columns:
        values = data[column].to_numpy()
        ax.bar(data.index, values, bottom=bottom, label=column.replace("_", " "))
        bottom += values
    ax.set_ylabel("Rows removed")
    ax.set_title("Duplicate Removal by Split")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    save_figure(fig, output_base)


def plot_pca(X: np.ndarray, y: np.ndarray, class_names: Sequence[str], output_base: Path, max_points: int = 12000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    if len(X) > max_points:
        indices = rng.choice(len(X), size=max_points, replace=False)
        X = X[indices]
        y = y[indices]
    projected = PCA(n_components=2, random_state=seed).fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    for class_id, class_name in enumerate(class_names):
        mask = y == class_id
        if bool(mask.any()):
            ax.scatter(projected[mask, 0], projected[mask, 1], s=8, alpha=0.45, label=class_name)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("PCA Projection of Leakage-Controlled Training Data")
    ax.legend(markerscale=2, fontsize=8)
    save_figure(fig, output_base)


def plot_correlation(frame: pd.DataFrame, output_base: Path, top_n: int = 20, max_rows: int = 25000, seed: int = 42) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).drop(columns=["target", "row_hash", "source_row"], errors="ignore")
    if len(numeric) > max_rows:
        numeric = numeric.sample(n=max_rows, random_state=seed)
    variances = numeric.var(numeric_only=True).sort_values(ascending=False)
    columns = list(variances.head(top_n).index)
    corr = numeric[columns].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=7)
    ax.set_yticklabels(columns, fontsize=7)
    ax.set_title("Correlation of High-Variance Behavioral Features")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, output_base)
    return corr

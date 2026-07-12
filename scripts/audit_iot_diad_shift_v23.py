#!/usr/bin/env python3
"""Distribution-shift and minority-class reliability audit for CIC IoT-DIAD Protocol V2.1.

The audit uses the already prepared NPZ arrays. It does not modify the dataset.
It produces source CSV tables plus publication-quality PNG and PDF figures.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit class-prior, feature, and capture-domain shift in CIC IoT-DIAD V2.1."
    )
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--feature-names", required=True, type=Path)
    parser.add_argument("--screening-per-class", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-per-class", type=int, default=5000)
    parser.add_argument("--pca-sample-per-class", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_arrays(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Prepared array file not found: {path}")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_feature_names(path: Path, expected_count: int) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature-name file not found: {path}")
    table = pd.read_csv(path)
    if "feature" not in table.columns:
        raise ValueError(f"Expected a 'feature' column in {path}")
    names = table["feature"].astype(str).tolist()
    if len(names) != expected_count:
        raise ValueError(
            f"Feature count mismatch, NPZ has {expected_count}, file has {len(names)}"
        )
    return names


def class_distribution(arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split in ("train", "val", "test_natural", "test_diagnostic"):
        labels = arrays[f"y_{split}"]
        counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        total = int(counts.sum())
        for class_id, class_name in enumerate(CLASS_NAMES):
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "rows": int(counts[class_id]),
                    "proportion": float(counts[class_id] / max(total, 1)),
                }
            )
    return pd.DataFrame(rows)


def prior_divergence(distribution: pd.DataFrame) -> pd.DataFrame:
    pivot = distribution.pivot(index="class_name", columns="split", values="proportion").reindex(CLASS_NAMES)
    train = pivot["train"].to_numpy(dtype=float)
    train = np.clip(train, 1e-12, None)
    train /= train.sum()
    rows = []
    for split in ("val", "test_natural", "test_diagnostic"):
        other = np.clip(pivot[split].to_numpy(dtype=float), 1e-12, None)
        other /= other.sum()
        rows.append(
            {
                "reference_split": "train",
                "comparison_split": split,
                "jensen_shannon_distance": float(jensenshannon(train, other, base=2.0)),
                "jensen_shannon_divergence": float(jensenshannon(train, other, base=2.0) ** 2),
                "total_variation_distance": float(0.5 * np.abs(train - other).sum()),
            }
        )
    return pd.DataFrame(rows)


def choose_sample(values: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(values) <= limit:
        return values
    indices = rng.choice(len(values), size=limit, replace=False)
    return values[indices]


def psi_from_train_bins(train: np.ndarray, test: np.ndarray, bins: int = 10) -> float:
    if len(train) == 0 or len(test) == 0:
        return math.nan
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(train, quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    train_hist = np.histogram(train, bins=edges)[0].astype(float)
    test_hist = np.histogram(test, bins=edges)[0].astype(float)
    train_prop = np.clip(train_hist / max(train_hist.sum(), 1.0), 1e-6, None)
    test_prop = np.clip(test_hist / max(test_hist.sum(), 1.0), 1e-6, None)
    return float(np.sum((test_prop - train_prop) * np.log(test_prop / train_prop)))


def per_feature_shift(
    arrays: Dict[str, np.ndarray],
    feature_names: List[str],
    sample_per_class: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    X_train = arrays["X_train"]
    y_train = arrays["y_train"]

    for comparison_split in ("test_natural", "test_diagnostic"):
        X_test = arrays[f"X_{comparison_split}"]
        y_test = arrays[f"y_{comparison_split}"]

        for class_id, class_name in enumerate(CLASS_NAMES):
            train_class = choose_sample(X_train[y_train == class_id], sample_per_class, rng)
            test_class = choose_sample(X_test[y_test == class_id], sample_per_class, rng)

            if len(train_class) == 0 or len(test_class) == 0:
                for feature_index, feature_name in enumerate(feature_names):
                    rows.append(
                        {
                            "comparison_split": comparison_split,
                            "class_id": class_id,
                            "class_name": class_name,
                            "feature_index": feature_index,
                            "feature": feature_name,
                            "train_rows_used": len(train_class),
                            "test_rows_used": len(test_class),
                            "train_mean": math.nan,
                            "test_mean": math.nan,
                            "standardized_mean_difference": math.nan,
                            "ks_statistic": math.nan,
                            "ks_p_value": math.nan,
                            "wasserstein_distance": math.nan,
                            "psi": math.nan,
                        }
                    )
                continue

            for feature_index, feature_name in enumerate(feature_names):
                train_values = train_class[:, feature_index].astype(np.float64)
                test_values = test_class[:, feature_index].astype(np.float64)
                train_values = train_values[np.isfinite(train_values)]
                test_values = test_values[np.isfinite(test_values)]

                if len(train_values) == 0 or len(test_values) == 0:
                    ks_stat = ks_p = wd = smd = psi = math.nan
                    train_mean = test_mean = math.nan
                else:
                    train_mean = float(np.mean(train_values))
                    test_mean = float(np.mean(test_values))
                    pooled = math.sqrt(
                        max(
                            (
                                float(np.var(train_values, ddof=1)) * max(len(train_values) - 1, 0)
                                + float(np.var(test_values, ddof=1)) * max(len(test_values) - 1, 0)
                            )
                            / max(len(train_values) + len(test_values) - 2, 1),
                            0.0,
                        )
                    )
                    smd = float(abs(test_mean - train_mean) / pooled) if pooled > 1e-12 else 0.0
                    ks = ks_2samp(train_values, test_values, alternative="two-sided", method="auto")
                    ks_stat = float(ks.statistic)
                    ks_p = float(ks.pvalue)
                    wd = float(wasserstein_distance(train_values, test_values))
                    psi = psi_from_train_bins(train_values, test_values)

                rows.append(
                    {
                        "comparison_split": comparison_split,
                        "class_id": class_id,
                        "class_name": class_name,
                        "feature_index": feature_index,
                        "feature": feature_name,
                        "train_rows_used": len(train_class),
                        "test_rows_used": len(test_class),
                        "train_mean": train_mean,
                        "test_mean": test_mean,
                        "standardized_mean_difference": smd,
                        "ks_statistic": ks_stat,
                        "ks_p_value": ks_p,
                        "wasserstein_distance": wd,
                        "psi": psi,
                    }
                )
    return pd.DataFrame(rows)


def shift_summary(feature_shift: pd.DataFrame) -> pd.DataFrame:
    return (
        feature_shift.groupby(["comparison_split", "class_id", "class_name"], as_index=False)
        .agg(
            mean_ks=("ks_statistic", "mean"),
            median_ks=("ks_statistic", "median"),
            max_ks=("ks_statistic", "max"),
            mean_abs_smd=("standardized_mean_difference", "mean"),
            max_abs_smd=("standardized_mean_difference", "max"),
            mean_wasserstein=("wasserstein_distance", "mean"),
            mean_psi=("psi", "mean"),
            max_psi=("psi", "max"),
        )
    )


def top_shifted_features(feature_shift: pd.DataFrame, top_k: int) -> pd.DataFrame:
    ranked = feature_shift.copy()
    ranked["composite_shift_score"] = (
        ranked["ks_statistic"].fillna(0.0)
        + 0.5 * ranked["standardized_mean_difference"].clip(upper=5).fillna(0.0)
        + 0.25 * ranked["psi"].clip(upper=10).fillna(0.0)
    )
    return (
        ranked.sort_values(
            ["comparison_split", "class_name", "composite_shift_score"],
            ascending=[True, True, False],
        )
        .groupby(["comparison_split", "class_name"], as_index=False, group_keys=False)
        .head(top_k)
    )


def support_reliability(
    distribution: pd.DataFrame,
    screening_path: Path | None,
) -> pd.DataFrame:
    natural = distribution[distribution["split"] == "test_natural"][
        ["class_name", "rows", "proportion"]
    ].rename(columns={"rows": "natural_test_support", "proportion": "natural_test_proportion"})

    def band(support: int) -> str:
        if support < 50:
            return "very_low"
        if support < 200:
            return "low"
        if support < 1000:
            return "moderate"
        return "high"

    natural["reliability_band"] = natural["natural_test_support"].map(band)
    natural["recommended_interpretation"] = natural["reliability_band"].map(
        {
            "very_low": "Do not draw strong per-class conclusions from one holdout.",
            "low": "Report uncertainty and treat as exploratory.",
            "moderate": "Interpret cautiously with confidence intervals.",
            "high": "Suitable for primary per-class reporting.",
        }
    )

    if screening_path is None or not screening_path.exists():
        return natural

    metrics = pd.read_csv(screening_path)
    metrics = metrics[
        (metrics["split"] == "test_natural")
        & (metrics["class_name"].isin(CLASS_NAMES))
    ].copy()
    return metrics.merge(natural, on="class_name", how="left")


def stratified_sample(
    X: np.ndarray,
    y: np.ndarray,
    per_class: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for class_id in range(len(CLASS_NAMES)):
        subset = X[y == class_id]
        if len(subset) == 0:
            continue
        subset = choose_sample(subset, per_class, rng)
        xs.append(subset)
        ys.append(np.full(len(subset), class_id, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def pca_projection(
    arrays: Dict[str, np.ndarray],
    sample_per_class: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces = []
    for split in ("train", "test_natural", "test_diagnostic"):
        X, y = stratified_sample(
            arrays[f"X_{split}"],
            arrays[f"y_{split}"],
            sample_per_class,
            rng,
        )
        pieces.append((split, X, y))

    train_X = pieces[0][1]
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(train_X)

    rows = []
    for split, X, y in pieces:
        points = pca.transform(X)
        for index in range(len(points)):
            rows.append(
                {
                    "split": split,
                    "class_id": int(y[index]),
                    "class_name": CLASS_NAMES[int(y[index])],
                    "pc1": float(points[index, 0]),
                    "pc2": float(points[index, 1]),
                }
            )
    return pd.DataFrame(rows)


def plot_class_support(distribution: pd.DataFrame, output: Path) -> None:
    pivot = distribution.pivot(index="class_name", columns="split", values="rows").reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.2
    for index, split in enumerate(["train", "val", "test_natural", "test_diagnostic"]):
        ax.bar(x + (index - 1.5) * width, pivot[split].values, width, label=split)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_ylabel("Rows, log scale")
    ax.set_title("Class Support Across Leakage-Safe Splits")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def plot_prior_shift(distribution: pd.DataFrame, output: Path) -> None:
    pivot = distribution.pivot(index="class_name", columns="split", values="proportion").reindex(CLASS_NAMES)
    columns = ["train", "val", "test_natural", "test_diagnostic"]
    values = pivot[columns].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(values, aspect="auto", vmin=0.0, vmax=max(0.01, float(values.max())))
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_title("Class-Prior Shift Across Experimental Splits")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Proportion")
    save_figure(fig, output)


def plot_shift_heatmap(summary: pd.DataFrame, output: Path) -> None:
    pivot = summary.pivot(
        index="class_name", columns="comparison_split", values="mean_ks"
    ).reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_title("Mean Per-Feature KS Shift by Class")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean KS statistic")
    save_figure(fig, output)


def plot_top_features(top: pd.DataFrame, output: Path) -> None:
    natural = top[top["comparison_split"] == "test_natural"]
    aggregate = (
        natural.groupby("feature", as_index=False)["composite_shift_score"]
        .mean()
        .sort_values("composite_shift_score", ascending=False)
        .head(15)
        .sort_values("composite_shift_score")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(aggregate["feature"], aggregate["composite_shift_score"])
    ax.set_xlabel("Mean composite shift score")
    ax.set_title("Most Shifted Behavioral Features, Train vs Natural Test")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, output)


def plot_pca(projection: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    markers = {"train": "o", "test_natural": "^", "test_diagnostic": "s"}
    for split, group in projection.groupby("split"):
        ax.scatter(
            group["pc1"],
            group["pc2"],
            s=7,
            alpha=0.18,
            marker=markers.get(split, "o"),
            label=split,
        )
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("Domain Projection of Train and Test Distributions")
    ax.grid(alpha=0.2)
    ax.legend(markerscale=2)
    save_figure(fig, output)


def plot_support_vs_f1(reliability: pd.DataFrame, output: Path) -> None:
    required = {"model", "f1-score", "natural_test_support", "class_name"}
    if not required.issubset(reliability.columns):
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for model, group in reliability.groupby("model"):
        ax.scatter(group["natural_test_support"], group["f1-score"], s=55, alpha=0.8, label=model)
    best_model = (
        reliability.groupby("model")["f1-score"].mean().sort_values(ascending=False).index[0]
    )
    for _, row in reliability[reliability["model"] == best_model].iterrows():
        ax.annotate(
            str(row["class_name"]),
            (float(row["natural_test_support"]), float(row["f1-score"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlim(left=10)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Natural-test class support, log scale")
    ax.set_ylabel("Per-class F1")
    ax.set_title("Per-Class F1 Versus Evaluation Support")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def plot_minority_precision_recall(reliability: pd.DataFrame, output: Path) -> None:
    required = {"model", "precision", "recall", "class_name"}
    if not required.issubset(reliability.columns):
        return
    minority = reliability[
        reliability["class_name"].isin(["BruteForce", "Mirai", "Recon", "Spoofing", "Web-Based"])
    ]
    fig, ax = plt.subplots(figsize=(9, 7))
    for model, group in minority.groupby("model"):
        ax.scatter(group["recall"], group["precision"], s=60, label=model, alpha=0.8)
    for row in minority.itertuples(index=False):
        if row.model == minority.groupby("model")["f1-score"].mean().idxmax():
            ax.annotate(row.class_name, (row.recall, row.precision), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Minority-Class Precision and Recall on the Natural Test")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    arrays = load_arrays(args.data_file.expanduser().resolve())
    feature_names = load_feature_names(
        args.feature_names.expanduser().resolve(),
        arrays["X_train"].shape[1],
    )
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    distribution = class_distribution(arrays)
    distribution.to_csv(tables_dir / "class_distribution.csv", index=False)

    divergence = prior_divergence(distribution)
    divergence.to_csv(tables_dir / "class_prior_divergence.csv", index=False)

    print("Computing per-feature distribution shift...")
    feature_shift = per_feature_shift(
        arrays,
        feature_names,
        sample_per_class=args.sample_per_class,
        seed=args.seed,
    )
    feature_shift.to_csv(tables_dir / "per_feature_shift.csv", index=False)

    summary = shift_summary(feature_shift)
    summary.to_csv(tables_dir / "per_class_shift_summary.csv", index=False)

    top = top_shifted_features(feature_shift, args.top_k)
    top.to_csv(tables_dir / "top_shifted_features.csv", index=False)

    screening_path = (
        args.screening_per_class.expanduser().resolve()
        if args.screening_per_class is not None
        else None
    )
    reliability = support_reliability(distribution, screening_path)
    reliability.to_csv(tables_dir / "support_reliability.csv", index=False)

    print("Computing PCA projection...")
    projection = pca_projection(
        arrays,
        sample_per_class=args.pca_sample_per_class,
        seed=args.seed,
    )
    projection.to_csv(tables_dir / "pca_projection.csv", index=False)

    plot_class_support(distribution, figures_dir / "class_support_by_split")
    plot_prior_shift(distribution, figures_dir / "class_prior_shift")
    plot_shift_heatmap(summary, figures_dir / "per_class_feature_shift")
    plot_top_features(top, figures_dir / "top_shifted_features_natural")
    plot_pca(projection, figures_dir / "domain_pca_projection")
    plot_support_vs_f1(reliability, figures_dir / "support_vs_f1")
    plot_minority_precision_recall(reliability, figures_dir / "minority_precision_recall")

    natural_counts = distribution[distribution["split"] == "test_natural"].set_index("class_name")["rows"]
    low_support = natural_counts[natural_counts < 200].index.tolist()
    very_low_support = natural_counts[natural_counts < 50].index.tolist()

    metadata = {
        "audit_version": "2.3",
        "data_file": str(args.data_file.expanduser().resolve()),
        "feature_names": str(args.feature_names.expanduser().resolve()),
        "screening_per_class": str(screening_path) if screening_path else None,
        "sample_per_class": args.sample_per_class,
        "pca_sample_per_class": args.pca_sample_per_class,
        "feature_count": len(feature_names),
        "low_support_natural_classes_under_200": low_support,
        "very_low_support_natural_classes_under_50": very_low_support,
        "maximum_mean_ks_class": (
            summary[summary["comparison_split"] == "test_natural"]
            .sort_values("mean_ks", ascending=False)
            .iloc[0]["class_name"]
        ),
        "tables": str(tables_dir),
        "figures_png_pdf": str(figures_dir),
    }
    with (output_dir / "shift_audit_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Shift audit complete")
    print(divergence.to_string(index=False))
    print()
    print("Natural-test support")
    print(
        distribution[distribution["split"] == "test_natural"][
            ["class_name", "rows", "proportion"]
        ].to_string(index=False)
    )
    print(f"CSV tables: {tables_dir}")
    print(f"PNG and PDF figures: {figures_dir}")
    print(f"Metadata: {output_dir / 'shift_audit_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.home() / "LFighter-research"
DATA_ROOT = ROOT / "data" / "processed" / "cic_iot_diad_2024_v2_1"
PREP_DIR = DATA_ROOT / "preprocessors" / "behavioral_only"
ARRAY_PATH = DATA_ROOT / "arrays" / "behavioral_only.npz"
FEATURES_PATH = PREP_DIR / "selected_features.csv"
METADATA_PATH = DATA_ROOT / "metadata_v2.json"

OUT_ROOT = (
    ROOT
    / "results"
    / "cic_iot_diad_task41a_trigger_feasibility_v410a"
    / "raw_semantic_audit"
)
TABLE_DIR = OUT_ROOT / "tables"
FIGURE_DIR = OUT_ROOT / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

branch = subprocess.check_output(
    ["git", "branch", "--show-current"], cwd=ROOT, text=True
).strip()
commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
).strip()

features = pd.read_csv(FEATURES_PATH)["feature"].astype(str).tolist()
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
class_names = [str(value) for value in metadata["classes"]]

scaler = joblib.load(PREP_DIR / "scaler.joblib")
imputer = joblib.load(PREP_DIR / "imputer.joblib")
selector = joblib.load(PREP_DIR / "variance_selector.joblib")

mean = np.asarray(scaler.mean_, dtype=np.float64)
scale = np.asarray(scaler.scale_, dtype=np.float64)
if len(features) != 69 or mean.shape != (69,) or scale.shape != (69,):
    raise RuntimeError(
        f"Feature/scaler mismatch: names={len(features)}, "
        f"mean={mean.shape}, scale={scale.shape}"
    )
if np.any(scale <= 0):
    raise RuntimeError("Scaler contains nonpositive scale values")

print("===== TASK 41A RAW SEMANTIC DOMAIN AUDIT =====")
print("Branch:", branch)
print("Commit:", commit)
print("Features:", len(features))
print("Classes:", class_names)
print("Scaler:", type(scaler).__name__)
print("Imputer:", type(imputer).__name__)
print("Variance selector:", type(selector).__name__)

with np.load(ARRAY_PATH, allow_pickle=False) as archive:
    X_train = np.asarray(archive["X_train"], dtype=np.float32)
    X_val = np.asarray(archive["X_val"], dtype=np.float32)
    y_train = np.asarray(archive["y_train"], dtype=np.int64)
    y_val = np.asarray(archive["y_val"], dtype=np.int64)

if X_train.shape != (len(y_train), len(features)):
    raise RuntimeError(f"Unexpected train shapes: {X_train.shape}, {y_train.shape}")
if X_val.shape != (len(y_val), len(features)):
    raise RuntimeError(f"Unexpected validation shapes: {X_val.shape}, {y_val.shape}")

quantiles = np.array(
    [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0],
    dtype=np.float64,
)
domain_rows: list[dict[str, object]] = []
class_rows: list[dict[str, object]] = []

for j, feature in enumerate(features):
    raw_train = X_train[:, j].astype(np.float64) * scale[j] + mean[j]
    raw_val = X_val[:, j].astype(np.float64) * scale[j] + mean[j]

    train_q = np.quantile(raw_train, quantiles)
    val_q = np.quantile(raw_val, quantiles)

    nearest_integer_error = np.abs(raw_train - np.rint(raw_train))
    integer_like_rate = float(np.mean(nearest_integer_error <= 1e-4))
    rounded_min = int(np.rint(np.min(raw_train)))
    rounded_max = int(np.rint(np.max(raw_train)))

    likely_binary = bool(
        integer_like_rate >= 0.9999
        and rounded_min >= 0
        and rounded_max <= 1
    )

    approximate_unique_count = ""
    likely_discrete = False
    if integer_like_rate >= 0.999:
        rounded = np.rint(raw_train).astype(np.int64)
        unique_values = np.unique(rounded)
        approximate_unique_count = int(unique_values.size)
        likely_discrete = bool(unique_values.size <= 64)

    row: dict[str, object] = {
        "feature_index": j,
        "feature": feature,
        "scaler_mean": float(mean[j]),
        "scaler_scale": float(scale[j]),
        "integer_like_rate": integer_like_rate,
        "likely_binary_raw": likely_binary,
        "likely_discrete_raw": likely_discrete,
        "rounded_unique_count_if_integer_like": approximate_unique_count,
        "raw_train_min": float(train_q[0]),
        "raw_train_max": float(train_q[-1]),
        "raw_validation_min": float(val_q[0]),
        "raw_validation_max": float(val_q[-1]),
        "raw_train_iqr": float(train_q[6] - train_q[4]),
        "raw_train_p99_p01_width": float(train_q[8] - train_q[2]),
        "validation_below_train_min_rate": float(np.mean(raw_val < train_q[0])),
        "validation_above_train_max_rate": float(np.mean(raw_val > train_q[-1])),
        "candidate_low_intensity_value_q75": float(train_q[6]),
        "candidate_high_intensity_value_q95": float(train_q[7]),
        "candidate_rare_value_q99": float(train_q[8]),
    }
    for q, value in zip(quantiles, train_q):
        row[f"raw_train_q_{str(q).replace('.', '_')}"] = float(value)
    for q, value in zip(quantiles, val_q):
        row[f"raw_val_q_{str(q).replace('.', '_')}"] = float(value)
    domain_rows.append(row)

    for class_id, class_name in enumerate(class_names):
        class_values = raw_train[y_train == class_id]
        if not len(class_values):
            continue
        q05, q25, q50, q75, q95 = np.quantile(
            class_values, [0.05, 0.25, 0.5, 0.75, 0.95]
        )
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "feature_index": j,
                "feature": feature,
                "rows": int(len(class_values)),
                "raw_q05": float(q05),
                "raw_q25": float(q25),
                "raw_q50": float(q50),
                "raw_q75": float(q75),
                "raw_q95": float(q95),
                "raw_iqr": float(q75 - q25),
            }
        )

domain_df = pd.DataFrame(domain_rows)
class_df = pd.DataFrame(class_rows)

rng = np.random.default_rng(410)
sample_size = min(100_000, len(X_train))
sample_indices = np.sort(rng.choice(len(X_train), size=sample_size, replace=False))
corr = np.corrcoef(X_train[sample_indices].astype(np.float64), rowvar=False)
corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

correlation_rows: list[dict[str, object]] = []
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        correlation_rows.append(
            {
                "feature_a_index": i,
                "feature_a": features[i],
                "feature_b_index": j,
                "feature_b": features[j],
                "pearson_correlation": float(corr[i, j]),
                "absolute_correlation": float(abs(corr[i, j])),
            }
        )
correlation_df = (
    pd.DataFrame(correlation_rows)
    .sort_values("absolute_correlation", ascending=False)
    .reset_index(drop=True)
)

max_corr = np.max(np.abs(corr - np.eye(len(features))), axis=1)
domain_df["maximum_absolute_correlation_with_other_feature"] = max_corr
domain_df["weakly_coupled_candidate"] = (
    (~domain_df["likely_binary_raw"])
    & (~domain_df["likely_discrete_raw"])
    & (domain_df["raw_train_p99_p01_width"] > 0)
    & (domain_df["maximum_absolute_correlation_with_other_feature"] < 0.90)
    & (
        domain_df["validation_below_train_min_rate"]
        + domain_df["validation_above_train_max_rate"]
        <= 0.001
    )
)

domain_csv = TABLE_DIR / "task41a_raw_feature_domain_quantiles.csv"
class_csv = TABLE_DIR / "task41a_raw_class_conditional_quantiles.csv"
correlation_csv = TABLE_DIR / "task41a_top_feature_correlations.csv"
candidate_csv = TABLE_DIR / "task41a_preliminary_trigger_feature_candidates.csv"

domain_df.to_csv(domain_csv, index=False)
class_df.to_csv(class_csv, index=False)
correlation_df.head(300).to_csv(correlation_csv, index=False)
(
    domain_df.loc[domain_df["weakly_coupled_candidate"]]
    .sort_values(
        [
            "maximum_absolute_correlation_with_other_feature",
            "raw_train_p99_p01_width",
        ],
        ascending=[True, False],
    )
    .to_csv(candidate_csv, index=False)
)

summary = {
    "experiment_version": "4.10C",
    "stage": "task41a_raw_semantic_domain_audit",
    "branch": branch,
    "commit": commit,
    "feature_count": len(features),
    "class_count": len(class_names),
    "train_shape": list(X_train.shape),
    "validation_shape": list(X_val.shape),
    "test_arrays_loaded": False,
    "scaler_type": f"{type(scaler).__module__}.{type(scaler).__name__}",
    "imputer_type": f"{type(imputer).__module__}.{type(imputer).__name__}",
    "variance_selector_type": f"{type(selector).__module__}.{type(selector).__name__}",
    "raw_binary_feature_count": int(domain_df["likely_binary_raw"].sum()),
    "raw_discrete_feature_count": int(domain_df["likely_discrete_raw"].sum()),
    "weakly_coupled_candidate_count": int(domain_df["weakly_coupled_candidate"].sum()),
    "correlation_sample_size": sample_size,
    "notes": [
        "Raw values were reconstructed from development arrays using scaler mean and scale.",
        "Final natural and diagnostic test arrays were not loaded.",
        "Preliminary candidate status is an audit aid, not a frozen trigger choice.",
        "Correlated multi-feature triggers must be built from empirical donor patterns, not independent arbitrary values.",
    ],
}
summary_json = TABLE_DIR / "task41a_raw_semantic_domain_audit.json"
summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

type_counts = pd.Series(
    {
        "Binary": int(domain_df["likely_binary_raw"].sum()),
        "Discrete": int(
            (
                domain_df["likely_discrete_raw"]
                & ~domain_df["likely_binary_raw"]
            ).sum()
        ),
        "Continuous": int((~domain_df["likely_discrete_raw"]).sum()),
    }
)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(type_counts.index, type_counts.values)
ax.set_ylabel("Feature count")
ax.set_title("Task 41A, reconstructed raw behavioral feature types")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
type_png = FIGURE_DIR / "task41a_raw_feature_type_counts.png"
type_pdf = FIGURE_DIR / "task41a_raw_feature_type_counts.pdf"
fig.savefig(type_png, dpi=300, bbox_inches="tight")
fig.savefig(type_pdf, bbox_inches="tight")
plt.close(fig)

top_corr = correlation_df.head(20).copy()
top_corr["pair"] = top_corr["feature_a"] + " ↔ " + top_corr["feature_b"]
fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(top_corr["pair"][::-1], top_corr["absolute_correlation"][::-1])
ax.set_xlabel("Absolute Pearson correlation")
ax.set_ylabel("Feature pair")
ax.set_title("Task 41A, strongest development feature dependencies")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
corr_png = FIGURE_DIR / "task41a_top_feature_dependencies.png"
corr_pdf = FIGURE_DIR / "task41a_top_feature_dependencies.pdf"
fig.savefig(corr_png, dpi=300, bbox_inches="tight")
fig.savefig(corr_pdf, bbox_inches="tight")
plt.close(fig)

print()
print("===== RAW SEMANTIC AUDIT SUMMARY =====")
print("Raw binary features:", summary["raw_binary_feature_count"])
print("Raw discrete features:", summary["raw_discrete_feature_count"])
print("Weakly coupled preliminary candidates:", summary["weakly_coupled_candidate_count"])
print("Correlation sample size:", sample_size)
print()
print("===== RAW BINARY FEATURES =====")
print(
    domain_df.loc[domain_df["likely_binary_raw"], ["feature_index", "feature"]]
    .to_string(index=False)
)
print()
print("===== PRELIMINARY WEAKLY COUPLED CANDIDATES =====")
print(
    domain_df.loc[
        domain_df["weakly_coupled_candidate"],
        [
            "feature_index",
            "feature",
            "candidate_low_intensity_value_q75",
            "candidate_high_intensity_value_q95",
            "candidate_rare_value_q99",
            "maximum_absolute_correlation_with_other_feature",
        ],
    ]
    .sort_values("maximum_absolute_correlation_with_other_feature")
    .head(20)
    .to_string(index=False)
)
print()
for path in (
    domain_csv,
    class_csv,
    correlation_csv,
    candidate_csv,
    summary_json,
    type_png,
    type_pdf,
    corr_png,
    corr_pdf,
):
    print("WROTE:", path)
print("FINAL TEST ARRAYS LOADED: False")

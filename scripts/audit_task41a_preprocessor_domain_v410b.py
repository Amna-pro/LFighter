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
FEATURES_PATH = PREP_DIR / "selected_features.csv"
ARRAY_PATH = DATA_ROOT / "arrays" / "behavioral_only.npz"
OUT_ROOT = (
    ROOT
    / "results"
    / "cic_iot_diad_task41a_trigger_feasibility_v410a"
    / "domain_audit"
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
if len(features) != 69:
    raise RuntimeError(f"Expected 69 behavioral features, found {len(features)}")

preprocessor_records: list[dict[str, object]] = []
loaded_objects: list[dict[str, object]] = []

print("===== TASK 41A PREPROCESSOR AND DOMAIN AUDIT =====")
print("Branch:", branch)
print("Commit:", commit)
print("Preprocessor directory:", PREP_DIR)
print("Feature count:", len(features))
print()
print("===== PREPROCESSOR FILES =====")

for path in sorted(PREP_DIR.rglob("*")):
    if not path.is_file():
        continue
    record = {
        "relative_path": str(path.relative_to(ROOT)),
        "suffix": path.suffix.lower(),
        "bytes": int(path.stat().st_size),
    }
    preprocessor_records.append(record)
    print(f"{path}\t{path.stat().st_size} bytes")

    if path.suffix.lower() not in {".joblib", ".pkl", ".pickle"}:
        continue
    try:
        obj = joblib.load(path)
        summary: dict[str, object] = {
            "relative_path": str(path.relative_to(ROOT)),
            "object_type": f"{type(obj).__module__}.{type(obj).__name__}",
        }
        for attr in (
            "n_features_in_",
            "feature_names_in_",
            "statistics_",
            "center_",
            "scale_",
            "mean_",
            "var_",
            "quantiles_",
        ):
            if not hasattr(obj, attr):
                continue
            value = getattr(obj, attr)
            array = np.asarray(value)
            summary[f"{attr}_shape"] = list(array.shape)
            if array.size and np.issubdtype(array.dtype, np.number):
                summary[f"{attr}_min"] = float(np.nanmin(array))
                summary[f"{attr}_max"] = float(np.nanmax(array))
        loaded_objects.append(summary)
        print("LOADED:", summary["object_type"], path.name)
    except Exception as exc:
        loaded_objects.append(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "load_error": repr(exc),
            }
        )
        print("LOAD ERROR:", path.name, repr(exc))

print()
print("===== DEVELOPMENT ARRAYS ONLY =====")
with np.load(ARRAY_PATH, allow_pickle=False) as archive:
    X_train = np.asarray(archive["X_train"], dtype=np.float64)
    X_val = np.asarray(archive["X_val"], dtype=np.float64)

if X_train.shape[1] != len(features) or X_val.shape[1] != len(features):
    raise RuntimeError(
        f"Feature mismatch: train={X_train.shape}, val={X_val.shape}, "
        f"names={len(features)}"
    )

quantiles = [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]
train_q = np.quantile(X_train, quantiles, axis=0)
val_q = np.quantile(X_val, quantiles, axis=0)

rows: list[dict[str, object]] = []
for index, feature in enumerate(features):
    train_col = X_train[:, index]
    val_col = X_val[:, index]
    unique_train = int(np.unique(train_col).size)
    finite_train = float(np.isfinite(train_col).mean())
    finite_val = float(np.isfinite(val_col).mean())
    row: dict[str, object] = {
        "feature_index": index,
        "feature": feature,
        "train_finite_rate": finite_train,
        "val_finite_rate": finite_val,
        "train_unique_count": unique_train,
        "likely_binary": bool(
            unique_train <= 2
            and np.all(np.isin(np.unique(train_col), [0.0, 1.0]))
        ),
        "likely_discrete": bool(unique_train <= 32),
        "train_iqr": float(train_q[6, index] - train_q[4, index]),
        "train_p99_p01_width": float(train_q[8, index] - train_q[2, index]),
        "val_below_train_min_rate": float(np.mean(val_col < train_q[0, index])),
        "val_above_train_max_rate": float(np.mean(val_col > train_q[-1, index])),
    }
    for q_index, q in enumerate(quantiles):
        label = str(q).replace(".", "_")
        row[f"train_q_{label}"] = float(train_q[q_index, index])
        row[f"val_q_{label}"] = float(val_q[q_index, index])
    rows.append(row)

domain_df = pd.DataFrame(rows)
domain_csv = TABLE_DIR / "task41a_feature_domain_quantiles.csv"
domain_df.to_csv(domain_csv, index=False)

preprocessor_csv = TABLE_DIR / "task41a_preprocessor_files.csv"
pd.DataFrame(preprocessor_records).to_csv(preprocessor_csv, index=False)

summary = {
    "experiment_version": "4.10B",
    "stage": "task41a_preprocessor_and_feature_domain_audit",
    "branch": branch,
    "commit": commit,
    "data_file": str(ARRAY_PATH),
    "preprocessor_directory": str(PREP_DIR),
    "feature_count": len(features),
    "train_shape": list(X_train.shape),
    "validation_shape": list(X_val.shape),
    "test_arrays_loaded": False,
    "preprocessor_files": preprocessor_records,
    "loaded_preprocessor_objects": loaded_objects,
    "all_train_values_finite": bool(np.isfinite(X_train).all()),
    "all_validation_values_finite": bool(np.isfinite(X_val).all()),
    "binary_feature_count": int(domain_df["likely_binary"].sum()),
    "discrete_feature_count": int(domain_df["likely_discrete"].sum()),
    "validation_outside_train_range_cell_count": int(
        sum(
            np.count_nonzero(X_val[:, j] < train_q[0, j])
            + np.count_nonzero(X_val[:, j] > train_q[-1, j])
            for j in range(len(features))
        )
    ),
}
summary_json = TABLE_DIR / "task41a_preprocessor_domain_audit.json"
summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

plot_df = domain_df.sort_values(
    "train_p99_p01_width", ascending=False
).head(20)
fig, ax = plt.subplots(figsize=(11, 7))
ax.barh(plot_df["feature"], plot_df["train_p99_p01_width"])
ax.invert_yaxis()
ax.set_xlabel("Training P99 minus P01 width, transformed feature space")
ax.set_ylabel("Feature")
ax.set_title("Task 41A, widest empirical behavioral feature domains")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
png_path = FIGURE_DIR / "task41a_feature_domain_widths.png"
pdf_path = FIGURE_DIR / "task41a_feature_domain_widths.pdf"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print("X_train:", X_train.shape, X_train.dtype)
print("X_val:", X_val.shape, X_val.dtype)
print("All train finite:", summary["all_train_values_finite"])
print("All validation finite:", summary["all_validation_values_finite"])
print("Likely binary features:", summary["binary_feature_count"])
print("Likely discrete features:", summary["discrete_feature_count"])
print(
    "Validation cells outside train min/max:",
    summary["validation_outside_train_range_cell_count"],
)
print()
print("WROTE:", domain_csv)
print("WROTE:", preprocessor_csv)
print("WROTE:", summary_json)
print("WROTE:", png_path)
print("WROTE:", pdf_path)
print("FINAL TEST ARRAYS LOADED: False")

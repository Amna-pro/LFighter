from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


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
    / "geometry_screen"
)
TABLE_DIR = OUT_ROOT / "tables"
FIGURE_DIR = OUT_ROOT / "figures"
SPEC_DIR = OUT_ROOT / "trigger_specs"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SPEC_DIR.mkdir(parents=True, exist_ok=True)

SEED = 4101
TARGET_CLASS_ID = 0
DONOR_CANDIDATES_PER_FAMILY = 24
SOURCE_EVAL_PER_CLASS = 500
PCA_FIT_ROWS = 50_000
TRIGGER_MATCH_TOLERANCE_STD = 0.05
LOW_INTENSITY_ALPHA = 0.35

branch = subprocess.check_output(
    ["git", "branch", "--show-current"], cwd=ROOT, text=True
).strip()
commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
).strip()

features = pd.read_csv(FEATURES_PATH)["feature"].astype(str).tolist()
feature_to_index = {name: index for index, name in enumerate(features)}
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
class_names = [str(value) for value in metadata["classes"]]

scaler = joblib.load(PREP_DIR / "scaler.joblib")
mean = np.asarray(scaler.mean_, dtype=np.float64)
scale = np.asarray(scaler.scale_, dtype=np.float64)

with np.load(ARRAY_PATH, allow_pickle=False) as archive:
    X_train = np.asarray(archive["X_train"], dtype=np.float32)
    X_val = np.asarray(archive["X_val"], dtype=np.float32)
    y_train = np.asarray(archive["y_train"], dtype=np.int64)
    y_val = np.asarray(archive["y_val"], dtype=np.int64)

if X_train.shape[1] != len(features):
    raise RuntimeError("Training feature count does not match selected features")
if X_val.shape[1] != len(features):
    raise RuntimeError("Validation feature count does not match selected features")
if len(class_names) != 8:
    raise RuntimeError(f"Expected eight classes, found {class_names}")

print("===== TASK 41A TRIGGER GEOMETRY SCREEN =====")
print("Branch:", branch)
print("Commit:", commit)
print("Target class:", class_names[TARGET_CLASS_ID])
print("Development train:", X_train.shape)
print("Development validation:", X_val.shape)

family_definitions = [
    {
        "family": "single_fwd_init_window",
        "features": ["FWD Init Win Bytes"],
        "modes": ["exact_template", "low_intensity_template"],
        "semantic_type": "initial_window",
    },
    {
        "family": "single_bwd_init_window",
        "features": ["Bwd Init Win Bytes"],
        "modes": ["exact_template", "low_intensity_template"],
        "semantic_type": "initial_window",
    },
    {
        "family": "single_fwd_psh_flag",
        "features": ["Fwd PSH Flags"],
        "modes": ["exact_template"],
        "semantic_type": "binary_flag",
    },
    {
        "family": "multi_init_windows",
        "features": ["FWD Init Win Bytes", "Bwd Init Win Bytes"],
        "modes": ["exact_template", "low_intensity_template"],
        "semantic_type": "initial_window",
    },
    {
        "family": "multi_flow_iat_bundle",
        "features": [
            "Flow IAT Mean",
            "Flow IAT Std",
            "Flow IAT Max",
            "Flow IAT Min",
        ],
        "modes": ["exact_template", "low_intensity_template"],
        "semantic_type": "iat_bundle",
    },
    {
        "family": "multi_fwd_packet_length_bundle",
        "features": [
            "Fwd Packet Length Max",
            "Fwd Packet Length Min",
            "Fwd Packet Length Mean",
            "Fwd Packet Length Std",
        ],
        "modes": ["exact_template", "low_intensity_template"],
        "semantic_type": "packet_length_bundle",
    },
    {
        "family": "multi_bwd_packet_length_bundle",
        "features": [
            "Bwd Packet Length Max",
            "Bwd Packet Length Min",
            "Bwd Packet Length Mean",
            "Bwd Packet Length Std",
        ],
        "modes": ["exact_template", "low_intensity_template"],
        "semantic_type": "packet_length_bundle",
    },
    {
        "family": "multi_active_idle_bundle",
        "features": [
            "Active Mean",
            "Active Std",
            "Active Max",
            "Active Min",
            "Idle Mean",
            "Idle Std",
            "Idle Max",
            "Idle Min",
        ],
        "modes": ["exact_template", "low_intensity_template"],
        "semantic_type": "active_idle_bundle",
    },
]

available_families = []
for definition in family_definitions:
    missing = [name for name in definition["features"] if name not in feature_to_index]
    if missing:
        print("SKIPPING FAMILY, MISSING FEATURES:", definition["family"], missing)
        continue
    available_families.append(definition)

if len(available_families) < 5:
    raise RuntimeError(
        f"Too few trigger families available: {[d['family'] for d in available_families]}"
    )

rng = np.random.default_rng(SEED)
target_train_indices = np.flatnonzero(y_train == TARGET_CLASS_ID)
target_train = X_train[target_train_indices].astype(np.float64)
target_median = np.median(target_train, axis=0)

source_eval_indices = []
for class_id in range(1, len(class_names)):
    indices = np.flatnonzero(y_val == class_id)
    if not len(indices):
        continue
    take = min(SOURCE_EVAL_PER_CLASS, len(indices))
    selected = np.sort(rng.choice(indices, size=take, replace=False))
    source_eval_indices.extend(selected.tolist())
source_eval_indices = np.asarray(source_eval_indices, dtype=np.int64)
X_source_eval = X_val[source_eval_indices].astype(np.float64)
y_source_eval = y_val[source_eval_indices]

pca_fit_size = min(PCA_FIT_ROWS, len(X_train))
pca_fit_indices = np.sort(rng.choice(len(X_train), size=pca_fit_size, replace=False))
pca = PCA(n_components=min(20, X_train.shape[1]), random_state=SEED)
pca.fit(X_train[pca_fit_indices].astype(np.float64))


def pca_metrics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    latent = pca.transform(values)
    reconstructed = pca.inverse_transform(latent)
    reconstruction_error = np.mean((values - reconstructed) ** 2, axis=1)
    latent_norm = np.sqrt(np.sum(latent**2, axis=1))
    return reconstruction_error, latent_norm


baseline_reconstruction_error, baseline_latent_norm = pca_metrics(X_source_eval)
baseline_reconstruction_mean = float(np.mean(baseline_reconstruction_error))
baseline_latent_mean = float(np.mean(baseline_latent_norm))

target_centroid = np.mean(target_train, axis=0)
baseline_target_distance = np.sqrt(
    np.sum((X_source_eval - target_centroid) ** 2, axis=1)
)
baseline_target_distance_mean = float(np.mean(baseline_target_distance))

train_q001 = np.quantile(X_train.astype(np.float64), 0.001, axis=0)
train_q999 = np.quantile(X_train.astype(np.float64), 0.999, axis=0)

def raw_values(std_values: np.ndarray, indices: list[int]) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    return std_values[:, idx] * scale[idx] + mean[idx]


def semantic_validity(
    values: np.ndarray,
    indices: list[int],
    names: list[str],
    semantic_type: str,
) -> np.ndarray:
    raw = raw_values(values, indices)
    valid = np.isfinite(raw).all(axis=1)

    if semantic_type == "binary_flag":
        valid &= np.isclose(raw[:, 0], 0.0, atol=1e-4) | np.isclose(
            raw[:, 0], 1.0, atol=1e-4
        )
    elif semantic_type == "initial_window":
        valid &= np.all(raw >= -1.0001, axis=1)
        valid &= np.all(np.abs(raw - np.rint(raw)) <= 1e-3, axis=1)
    elif semantic_type == "iat_bundle":
        mapping = {name: raw[:, i] for i, name in enumerate(names)}
        valid &= mapping["Flow IAT Min"] <= mapping["Flow IAT Mean"] + 1e-6
        valid &= mapping["Flow IAT Mean"] <= mapping["Flow IAT Max"] + 1e-6
        valid &= mapping["Flow IAT Std"] >= -1e-6
        valid &= np.all(raw >= -1e-6, axis=1)
    elif semantic_type == "packet_length_bundle":
        mapping = {name: raw[:, i] for i, name in enumerate(names)}
        prefix = "Fwd" if names[0].startswith("Fwd") else "Bwd"
        valid &= mapping[f"{prefix} Packet Length Min"] <= (
            mapping[f"{prefix} Packet Length Mean"] + 1e-6
        )
        valid &= mapping[f"{prefix} Packet Length Mean"] <= (
            mapping[f"{prefix} Packet Length Max"] + 1e-6
        )
        valid &= mapping[f"{prefix} Packet Length Std"] >= -1e-6
        valid &= np.all(raw >= -1e-6, axis=1)
    elif semantic_type == "active_idle_bundle":
        mapping = {name: raw[:, i] for i, name in enumerate(names)}
        for prefix in ("Active", "Idle"):
            valid &= mapping[f"{prefix} Min"] <= mapping[f"{prefix} Mean"] + 1e-6
            valid &= mapping[f"{prefix} Mean"] <= mapping[f"{prefix} Max"] + 1e-6
            valid &= mapping[f"{prefix} Std"] >= -1e-6
        valid &= np.all(raw >= -1e-6, axis=1)

    return valid


candidate_rows: list[dict[str, object]] = []
source_rows: list[dict[str, object]] = []
candidate_templates: dict[str, dict[str, object]] = {}

for definition in available_families:
    family = str(definition["family"])
    names = list(definition["features"])
    indices = [feature_to_index[name] for name in names]
    semantic_type = str(definition["semantic_type"])

    central_mask = np.ones(len(target_train), dtype=bool)
    for index in indices:
        low = np.quantile(target_train[:, index], 0.05)
        high = np.quantile(target_train[:, index], 0.95)
        central_mask &= target_train[:, index] >= low
        central_mask &= target_train[:, index] <= high
    donor_pool = np.flatnonzero(central_mask)
    if len(donor_pool) < DONOR_CANDIDATES_PER_FAMILY:
        donor_pool = np.arange(len(target_train))
    donor_take = min(DONOR_CANDIDATES_PER_FAMILY, len(donor_pool))
    donor_choices = np.sort(
        rng.choice(donor_pool, size=donor_take, replace=False)
    )

    for mode in definition["modes"]:
        for donor_rank, donor_local_index in enumerate(donor_choices):
            donor_global_index = int(target_train_indices[donor_local_index])
            donor_vector = target_train[donor_local_index, indices].copy()

            if mode == "exact_template":
                template = donor_vector
            elif mode == "low_intensity_template":
                template = target_median[indices] + LOW_INTENSITY_ALPHA * (
                    donor_vector - target_median[indices]
                )
            else:
                raise RuntimeError(f"Unsupported trigger mode: {mode}")

            triggered = X_source_eval.copy()
            triggered[:, indices] = template

            below = triggered[:, indices] < train_q001[indices]
            above = triggered[:, indices] > train_q999[indices]
            range_violation_rate = float(np.mean(below | above))

            semantic_valid = semantic_validity(
                triggered, indices, names, semantic_type
            )
            semantic_valid_rate = float(np.mean(semantic_valid))

            val_distance = np.max(
                np.abs(
                    X_val[:, indices].astype(np.float64)
                    - template.reshape(1, -1)
                ),
                axis=1,
            )
            accidental_prevalence = float(
                np.mean(val_distance <= TRIGGER_MATCH_TOLERANCE_STD)
            )

            perturbation = triggered - X_source_eval
            l0 = np.sum(np.abs(perturbation) > 1e-12, axis=1)
            l2 = np.sqrt(np.sum(perturbation**2, axis=1))

            reconstruction_error, latent_norm = pca_metrics(triggered)
            pca_reconstruction_ratio = float(
                np.mean(reconstruction_error)
                / max(baseline_reconstruction_mean, 1e-12)
            )
            pca_latent_norm_ratio = float(
                np.mean(latent_norm) / max(baseline_latent_mean, 1e-12)
            )

            target_distance = np.sqrt(
                np.sum((triggered - target_centroid) ** 2, axis=1)
            )
            target_distance_ratio = float(
                np.mean(target_distance)
                / max(baseline_target_distance_mean, 1e-12)
            )

            rarity_score = -math.log10(max(accidental_prevalence, 1e-8))
            plausibility_penalty = max(0.0, pca_reconstruction_ratio - 1.5)
            perturbation_penalty = max(0.0, float(np.median(l2)) - 4.0) * 0.1
            score = (
                rarity_score
                - 2.0 * plausibility_penalty
                - perturbation_penalty
                - 10.0 * range_violation_rate
                - 10.0 * (1.0 - semantic_valid_rate)
            )

            qualified_geometry = bool(
                accidental_prevalence <= 0.01
                and range_violation_rate == 0.0
                and semantic_valid_rate == 1.0
                and pca_reconstruction_ratio <= 2.5
                and float(np.median(l2)) <= 8.0
            )

            candidate_id = (
                f"{family}__{mode}__donor_{donor_rank:02d}"
            )
            candidate_templates[candidate_id] = {
                "family": family,
                "mode": mode,
                "semantic_type": semantic_type,
                "feature_names": names,
                "feature_indices": indices,
                "template_standardized": [float(value) for value in template],
                "template_raw": [
                    float(template[i] * scale[index] + mean[index])
                    for i, index in enumerate(indices)
                ],
                "donor_train_global_index": donor_global_index,
                "target_class_id": TARGET_CLASS_ID,
                "target_class_name": class_names[TARGET_CLASS_ID],
                "distributed_deployment_compatible": True,
                "clean_label_policy_compatible": True,
            }

            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "family": family,
                    "mode": mode,
                    "semantic_type": semantic_type,
                    "feature_count": len(indices),
                    "feature_names": "|".join(names),
                    "feature_indices": "|".join(map(str, indices)),
                    "donor_train_global_index": donor_global_index,
                    "accidental_prevalence_validation": accidental_prevalence,
                    "rarity_score": rarity_score,
                    "range_violation_rate": range_violation_rate,
                    "semantic_valid_rate": semantic_valid_rate,
                    "median_changed_feature_count": float(np.median(l0)),
                    "median_standardized_l2": float(np.median(l2)),
                    "pca_reconstruction_ratio": pca_reconstruction_ratio,
                    "pca_latent_norm_ratio": pca_latent_norm_ratio,
                    "target_centroid_distance_ratio": target_distance_ratio,
                    "geometry_score": score,
                    "qualified_geometry": qualified_geometry,
                    "distributed_deployment_compatible": True,
                    "clean_label_policy_compatible": True,
                }
            )

            for source_class_id in range(1, len(class_names)):
                mask = y_source_eval == source_class_id
                if not np.any(mask):
                    continue
                source_reconstruction, source_latent = pca_metrics(triggered[mask])
                source_base_reconstruction = baseline_reconstruction_error[mask]
                source_base_latent = baseline_latent_norm[mask]
                source_base_target = baseline_target_distance[mask]
                source_target = target_distance[mask]
                source_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "family": family,
                        "mode": mode,
                        "source_class_id": source_class_id,
                        "source_class_name": class_names[source_class_id],
                        "rows": int(np.sum(mask)),
                        "semantic_valid_rate": float(np.mean(semantic_valid[mask])),
                        "median_standardized_l2": float(np.median(l2[mask])),
                        "pca_reconstruction_ratio": float(
                            np.mean(source_reconstruction)
                            / max(float(np.mean(source_base_reconstruction)), 1e-12)
                        ),
                        "pca_latent_norm_ratio": float(
                            np.mean(source_latent)
                            / max(float(np.mean(source_base_latent)), 1e-12)
                        ),
                        "target_centroid_distance_ratio": float(
                            np.mean(source_target)
                            / max(float(np.mean(source_base_target)), 1e-12)
                        ),
                    }
                )

candidate_df = pd.DataFrame(candidate_rows)
source_df = pd.DataFrame(source_rows)

candidate_df = candidate_df.sort_values(
    ["qualified_geometry", "geometry_score"],
    ascending=[False, False],
).reset_index(drop=True)

selected_rows = []
for (family, mode), group in candidate_df.groupby(["family", "mode"], sort=True):
    qualified = group[group["qualified_geometry"]]
    source = qualified if len(qualified) else group
    selected_rows.append(source.iloc[0])
selected_df = pd.DataFrame(selected_rows).sort_values(
    ["qualified_geometry", "geometry_score"],
    ascending=[False, False],
).reset_index(drop=True)

family_summary = (
    candidate_df.groupby(["family", "mode"], as_index=False)
    .agg(
        candidate_count=("candidate_id", "count"),
        qualified_count=("qualified_geometry", "sum"),
        best_geometry_score=("geometry_score", "max"),
        minimum_accidental_prevalence=(
            "accidental_prevalence_validation",
            "min",
        ),
        minimum_pca_reconstruction_ratio=("pca_reconstruction_ratio", "min"),
        median_candidate_l2=("median_standardized_l2", "median"),
    )
    .sort_values(["qualified_count", "best_geometry_score"], ascending=[False, False])
)

candidate_csv = TABLE_DIR / "task41a_trigger_geometry_candidates.csv"
source_csv = TABLE_DIR / "task41a_trigger_candidate_source_metrics.csv"
selected_csv = TABLE_DIR / "task41a_selected_geometry_candidates.csv"
family_csv = TABLE_DIR / "task41a_trigger_family_summary.csv"
candidate_df.to_csv(candidate_csv, index=False)
source_df.to_csv(source_csv, index=False)
selected_df.to_csv(selected_csv, index=False)
family_summary.to_csv(family_csv, index=False)

selected_specs = {
    row["candidate_id"]: candidate_templates[row["candidate_id"]]
    for _, row in selected_df.iterrows()
}
spec_json = SPEC_DIR / "task41a_selected_trigger_specs.json"
spec_json.write_text(
    json.dumps(
        {
            "experiment_version": "4.10D",
            "stage": "task41a_trigger_geometry_screen",
            "branch": branch,
            "commit": commit,
            "seed": SEED,
            "target_class_id": TARGET_CLASS_ID,
            "target_class_name": class_names[TARGET_CLASS_ID],
            "test_arrays_loaded": False,
            "selection_status": "PRELIMINARY_GEOMETRY_ONLY",
            "notes": [
                "These triggers are not yet attack-qualified.",
                "Distributed and clean-label variants are deployment policies applied later.",
                "Final selection requires plain-FedAvg backdoor attack success and clean-utility qualification.",
            ],
            "selected_specs": selected_specs,
        },
        indent=2,
    ),
    encoding="utf-8",
)

npz_payload = {}
for candidate_id, spec in selected_specs.items():
    safe = candidate_id.replace("-", "_")
    npz_payload[f"{safe}__feature_indices"] = np.asarray(
        spec["feature_indices"], dtype=np.int64
    )
    npz_payload[f"{safe}__template_standardized"] = np.asarray(
        spec["template_standardized"], dtype=np.float32
    )
    npz_payload[f"{safe}__template_raw"] = np.asarray(
        spec["template_raw"], dtype=np.float64
    )
spec_npz = SPEC_DIR / "task41a_selected_trigger_specs.npz"
np.savez_compressed(spec_npz, **npz_payload)

plot_df = candidate_df.head(40).copy()
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(
    plot_df["pca_reconstruction_ratio"],
    plot_df["rarity_score"],
    s=35,
)
ax.set_xlabel("PCA reconstruction-error ratio")
ax.set_ylabel("Trigger rarity score")
ax.set_title("Task 41A, trigger rarity versus development plausibility")
ax.grid(alpha=0.3)
fig.tight_layout()
scatter_png = FIGURE_DIR / "task41a_trigger_rarity_plausibility.png"
scatter_pdf = FIGURE_DIR / "task41a_trigger_rarity_plausibility.pdf"
fig.savefig(scatter_png, dpi=300, bbox_inches="tight")
fig.savefig(scatter_pdf, bbox_inches="tight")
plt.close(fig)

summary_plot = family_summary.copy()
summary_plot["label"] = summary_plot["family"] + "\n" + summary_plot["mode"]
fig, ax = plt.subplots(figsize=(13, 7))
ax.bar(summary_plot["label"], summary_plot["qualified_count"])
ax.set_ylabel("Qualified geometry candidates")
ax.set_title("Task 41A, qualified trigger geometries by family")
ax.tick_params(axis="x", rotation=75)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
bar_png = FIGURE_DIR / "task41a_qualified_geometry_counts.png"
bar_pdf = FIGURE_DIR / "task41a_qualified_geometry_counts.pdf"
fig.savefig(bar_png, dpi=300, bbox_inches="tight")
fig.savefig(bar_pdf, bbox_inches="tight")
plt.close(fig)

summary_json = TABLE_DIR / "task41a_trigger_geometry_screen_summary.json"
summary_json.write_text(
    json.dumps(
        {
            "experiment_version": "4.10D",
            "stage": "task41a_trigger_geometry_screen",
            "branch": branch,
            "commit": commit,
            "target_class": class_names[TARGET_CLASS_ID],
            "source_classes": class_names[1:],
            "available_family_count": len(available_families),
            "candidate_count": int(len(candidate_df)),
            "qualified_candidate_count": int(candidate_df["qualified_geometry"].sum()),
            "selected_candidate_count": int(len(selected_df)),
            "source_evaluation_rows": int(len(X_source_eval)),
            "pca_fit_rows": int(pca_fit_size),
            "test_arrays_loaded": False,
            "selection_status": "PRELIMINARY_GEOMETRY_ONLY",
        },
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("===== FAMILY SUMMARY =====")
print(family_summary.to_string(index=False))
print()
print("===== SELECTED PRELIMINARY GEOMETRIES =====")
print(
    selected_df[
        [
            "candidate_id",
            "feature_count",
            "accidental_prevalence_validation",
            "median_standardized_l2",
            "pca_reconstruction_ratio",
            "semantic_valid_rate",
            "qualified_geometry",
        ]
    ].to_string(index=False)
)
print()
for path in (
    candidate_csv,
    source_csv,
    selected_csv,
    family_csv,
    summary_json,
    spec_json,
    spec_npz,
    scatter_png,
    scatter_pdf,
    bar_png,
    bar_pdf,
):
    print("WROTE:", path)
print("FINAL TEST ARRAYS LOADED: False")

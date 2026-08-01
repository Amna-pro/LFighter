from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_VERSION = "4.10F"
ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "cic_iot_diad_task41a_trigger_feasibility_v410a"
GEOMETRY_ROOT = RESULT_ROOT / "geometry_screen"
OUTPUT_ROOT = RESULT_ROOT / "frozen_protocol"
TABLE_DIR = OUTPUT_ROOT / "tables"
SPEC_DIR = OUTPUT_ROOT / "trigger_specs"
FIGURE_DIR = OUTPUT_ROOT / "figures"
for directory in (TABLE_DIR, SPEC_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CANDIDATES_CSV = (
    GEOMETRY_ROOT / "tables" / "task41a_trigger_geometry_candidates.csv"
)
SOURCE_METRICS_CSV = (
    GEOMETRY_ROOT / "tables" / "task41a_trigger_candidate_source_metrics.csv"
)
SELECTED_CSV = (
    GEOMETRY_ROOT / "tables" / "task41a_selected_geometry_candidates.csv"
)
SOURCE_SPECS_JSON = (
    GEOMETRY_ROOT / "trigger_specs" / "task41a_selected_trigger_specs.json"
)
SOURCE_SPECS_NPZ = (
    GEOMETRY_ROOT / "trigger_specs" / "task41a_selected_trigger_specs.npz"
)

PANEL_RULES = [
    {
        "panel_slot": "flow_iat_exact",
        "family": "multi_flow_iat_bundle",
        "mode": "exact_template",
        "role": "high_rarity_temporal_exact",
    },
    {
        "panel_slot": "flow_iat_low",
        "family": "multi_flow_iat_bundle",
        "mode": "low_intensity_template",
        "role": "paired_intensity_temporal",
    },
    {
        "panel_slot": "fwd_packet_low",
        "family": "multi_fwd_packet_length_bundle",
        "mode": "low_intensity_template",
        "role": "forward_direction_low_intensity",
    },
    {
        "panel_slot": "bwd_packet_low",
        "family": "multi_bwd_packet_length_bundle",
        "mode": "low_intensity_template",
        "role": "reverse_direction_low_intensity",
    },
    {
        "panel_slot": "init_windows_exact",
        "family": "multi_init_windows",
        "mode": "exact_template",
        "role": "transport_state_exact",
    },
    {
        "panel_slot": "active_idle_exact",
        "family": "multi_active_idle_bundle",
        "mode": "exact_template",
        "role": "high_dimensional_temporal_exact",
    },
]

QUALIFICATION_SEED = 7
CONFIRMATORY_SEEDS = [7, 99, 123, 2026]
POISON_FRACTIONS = [0.01, 0.05, 0.10, 0.20]
MALICIOUS_CLIENTS = [1, 7, 8, 10, 14, 15, 17, 18]
TARGET_CLASS_ID = 0
TARGET_CLASS_NAME = "Benign"
SOURCE_CLASS_NAMES = [
    "BruteForce",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web-Based",
]


def git_text(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


for path in (
    CANDIDATES_CSV,
    SOURCE_METRICS_CSV,
    SELECTED_CSV,
    SOURCE_SPECS_JSON,
    SOURCE_SPECS_NPZ,
):
    if not path.exists():
        raise FileNotFoundError(path)

branch = git_text("branch", "--show-current")
commit = git_text("rev-parse", "HEAD")

candidates = pd.read_csv(CANDIDATES_CSV)
source_metrics = pd.read_csv(SOURCE_METRICS_CSV)
preliminary = pd.read_csv(SELECTED_CSV)

required_candidate_columns = {
    "candidate_id",
    "family",
    "mode",
    "feature_count",
    "feature_names",
    "feature_indices",
    "accidental_prevalence_validation",
    "range_violation_rate",
    "semantic_valid_rate",
    "median_standardized_l2",
    "pca_reconstruction_ratio",
    "pca_latent_norm_ratio",
    "target_centroid_distance_ratio",
    "geometry_score",
    "qualified_geometry",
    "distributed_deployment_compatible",
    "clean_label_policy_compatible",
}
missing_columns = sorted(required_candidate_columns - set(candidates.columns))
if missing_columns:
    raise KeyError(f"Candidate table missing columns: {missing_columns}")

required_source_columns = {
    "candidate_id",
    "source_class_id",
    "source_class_name",
    "rows",
    "semantic_valid_rate",
    "median_standardized_l2",
    "pca_reconstruction_ratio",
    "pca_latent_norm_ratio",
    "target_centroid_distance_ratio",
}
missing_source_columns = sorted(required_source_columns - set(source_metrics.columns))
if missing_source_columns:
    raise KeyError(f"Source metrics missing columns: {missing_source_columns}")

eligible = candidates[
    candidates["qualified_geometry"].astype(bool)
    & candidates["distributed_deployment_compatible"].astype(bool)
    & candidates["clean_label_policy_compatible"].astype(bool)
    & np.isclose(candidates["range_violation_rate"].astype(float), 0.0)
    & np.isclose(candidates["semantic_valid_rate"].astype(float), 1.0)
].copy()

panel_rows: list[pd.Series] = []
for rule in PANEL_RULES:
    pool = eligible[
        (eligible["family"] == rule["family"])
        & (eligible["mode"] == rule["mode"])
    ].copy()
    if pool.empty:
        raise RuntimeError(
            "No eligible candidate for fixed panel rule: "
            + json.dumps(rule, sort_keys=True)
        )
    pool = pool.sort_values(
        by=[
            "geometry_score",
            "accidental_prevalence_validation",
            "candidate_id",
        ],
        ascending=[False, True, True],
        kind="mergesort",
    )
    chosen = pool.iloc[0].copy()
    chosen["panel_slot"] = rule["panel_slot"]
    chosen["panel_role"] = rule["role"]
    chosen["selection_rule"] = (
        "fixed_family_mode_stratum_then_max_geometry_score_"
        "then_min_accidental_prevalence_then_candidate_id"
    )
    panel_rows.append(chosen)

panel = pd.DataFrame(panel_rows).reset_index(drop=True)
if len(panel) != len(PANEL_RULES):
    raise AssertionError("Frozen panel size mismatch")
if panel["candidate_id"].duplicated().any():
    raise AssertionError("Frozen panel contains duplicate candidates")

source_panel = source_metrics[
    source_metrics["candidate_id"].isin(panel["candidate_id"])
].copy()

expected_source_set = set(SOURCE_CLASS_NAMES)
robustness_rows: list[dict[str, Any]] = []
for candidate_id in panel["candidate_id"]:
    group = source_panel[source_panel["candidate_id"] == candidate_id].copy()
    observed_source_set = set(group["source_class_name"].astype(str))
    if observed_source_set != expected_source_set:
        raise AssertionError(
            f"{candidate_id} source coverage mismatch: "
            f"{sorted(observed_source_set)}"
        )
    if len(group) != len(SOURCE_CLASS_NAMES):
        raise AssertionError(
            f"{candidate_id} expected 7 source rows, got {len(group)}"
        )
    minimum_semantic_valid_rate = float(group["semantic_valid_rate"].min())
    if not np.isclose(minimum_semantic_valid_rate, 1.0):
        raise AssertionError(
            f"{candidate_id} has source semantic validity below 1.0"
        )
    robustness_rows.append(
        {
            "candidate_id": candidate_id,
            "source_class_count": int(group["source_class_name"].nunique()),
            "minimum_source_semantic_valid_rate": minimum_semantic_valid_rate,
            "maximum_source_median_standardized_l2": float(
                group["median_standardized_l2"].max()
            ),
            "maximum_source_pca_reconstruction_ratio": float(
                group["pca_reconstruction_ratio"].max()
            ),
            "maximum_source_pca_latent_norm_ratio": float(
                group["pca_latent_norm_ratio"].max()
            ),
            "maximum_source_target_centroid_distance_ratio": float(
                group["target_centroid_distance_ratio"].max()
            ),
            "minimum_source_rows": int(group["rows"].min()),
        }
    )

robustness = pd.DataFrame(robustness_rows)
panel = panel.merge(robustness, on="candidate_id", how="left", validate="one_to_one")

with SOURCE_SPECS_JSON.open("r", encoding="utf-8") as handle:
    source_specs = json.load(handle)

selected_specs = source_specs.get("selected_specs")
if not isinstance(selected_specs, dict):
    raise TypeError("selected_specs is not a dictionary")

missing_specs = [
    candidate_id
    for candidate_id in panel["candidate_id"]
    if candidate_id not in selected_specs
]
if missing_specs:
    raise KeyError(f"Missing selected JSON specs: {missing_specs}")

frozen_specs = {
    candidate_id: selected_specs[candidate_id]
    for candidate_id in panel["candidate_id"]
}

frozen_spec_json = {
    "experiment_version": EXPERIMENT_VERSION,
    "stage": "task41a_frozen_trigger_panel",
    "branch": branch,
    "commit": commit,
    "selection_time_information_used": [
        "development_train_geometry",
        "development_validation_rarity",
        "development_source_class_geometry",
    ],
    "attack_training_outcomes_used": False,
    "test_arrays_loaded": False,
    "selection_policy": {
        "type": "fixed_family_mode_stratification",
        "rules": PANEL_RULES,
        "within_stratum_ranking": [
            "maximum geometry_score",
            "minimum accidental_prevalence_validation",
            "lexicographic candidate_id",
        ],
    },
    "target_class_id": TARGET_CLASS_ID,
    "target_class_name": TARGET_CLASS_NAME,
    "source_classes": SOURCE_CLASS_NAMES,
    "selected_specs": frozen_specs,
}

spec_json_path = SPEC_DIR / "task41a_frozen_trigger_specs.json"
spec_json_path.write_text(
    json.dumps(frozen_spec_json, indent=2),
    encoding="utf-8",
)

npz_payload: dict[str, np.ndarray] = {}
with np.load(SOURCE_SPECS_NPZ, allow_pickle=False) as archive:
    for candidate_id in panel["candidate_id"]:
        prefix = f"{candidate_id}__"
        keys = [key for key in archive.files if key.startswith(prefix)]
        expected_suffixes = {
            "feature_indices",
            "template_standardized",
            "template_raw",
        }
        observed_suffixes = {key[len(prefix):] for key in keys}
        if observed_suffixes != expected_suffixes:
            raise AssertionError(
                f"{candidate_id} NPZ fields mismatch: "
                f"{sorted(observed_suffixes)}"
            )
        for key in keys:
            npz_payload[key] = np.asarray(archive[key]).copy()

spec_npz_path = SPEC_DIR / "task41a_frozen_trigger_specs.npz"
np.savez_compressed(spec_npz_path, **npz_payload)

qualification_rows: list[dict[str, Any]] = []
for candidate_id in panel["candidate_id"]:
    for poison_fraction in POISON_FRACTIONS:
        qualification_rows.append(
            {
                "phase": "primary_dirty_label_qualification",
                "candidate_id": candidate_id,
                "attack_seed": QUALIFICATION_SEED,
                "model_seed": QUALIFICATION_SEED,
                "poison_fraction": poison_fraction,
                "label_policy": "dirty_label_all_nonbenign_to_benign",
                "deployment_policy": "centralized_full_trigger",
                "target_class": TARGET_CLASS_NAME,
                "source_scope": "all_seven_nonbenign_classes",
                "malicious_clients": "|".join(
                    str(value) for value in MALICIOUS_CLIENTS
                ),
                "post_warmup_rounds": 4,
                "plain_fedavg_required": True,
                "trusted_reconstruction_required": False,
                "test_arrays_permitted": False,
            }
        )

qualification_grid = pd.DataFrame(qualification_rows)

confirmatory_configuration_count = (
    len(panel)
    * len(POISON_FRACTIONS)
    * len(CONFIRMATORY_SEEDS)
)

protocol = {
    "experiment_version": EXPERIMENT_VERSION,
    "stage": "task41_frozen_backdoor_protocol",
    "branch": branch,
    "commit": commit,
    "method_reopened": False,
    "attack_specific_defense_retuning_permitted": False,
    "natural_test_access_permitted": False,
    "diagnostic_test_access_permitted": False,
    "target_class": {
        "id": TARGET_CLASS_ID,
        "name": TARGET_CLASS_NAME,
    },
    "source_classes": SOURCE_CLASS_NAMES,
    "malicious_clients": MALICIOUS_CLIENTS,
    "warmup_rounds": 4,
    "post_warmup_rounds": 4,
    "trigger_panel_size": int(len(panel)),
    "trigger_candidate_ids": panel["candidate_id"].tolist(),
    "primary_qualification": {
        "seed": QUALIFICATION_SEED,
        "poison_fractions": POISON_FRACTIONS,
        "label_policy": "dirty_label_all_nonbenign_to_benign",
        "deployment_policy": "centralized_full_trigger",
        "configuration_count": int(len(qualification_grid)),
        "required_outputs": [
            "per_round_and_per_source_triggered_ASR",
            "clean_validation_macro_F1",
            "clean_target_class_recall",
            "clean_benign_FPR",
            "poisoned_sample_counts_by_client_and_source",
            "CSV_tables",
            "PNG_figures",
            "PDF_figures",
        ],
        "qualification_gates_frozen_before_training": {
            "minimum_mean_triggered_ASR": 0.50,
            "minimum_final_round_triggered_ASR": 0.50,
            "minimum_source_classes_with_mean_ASR_at_least_0_40": 5,
            "maximum_clean_macro_F1_drop_from_matched_clean": 0.10,
            "minimum_improved_attack_round_count_over_clean_trigger_baseline": 3,
        },
    },
    "confirmatory_multiseed": {
        "seeds": CONFIRMATORY_SEEDS,
        "poison_fractions": POISON_FRACTIONS,
        "retain_complete_poison_fraction_curves": True,
        "plain_and_frozen_defense_paired": True,
        "attack_configuration_count_before_pairing": int(
            confirmatory_configuration_count
        ),
        "paired_branch_count": int(
            confirmatory_configuration_count * 2
        ),
        "selection_from_qualification": (
            "No trigger family may be removed solely for weak performance. "
            "Qualification labels attacks as successful or weak, and both "
            "statuses remain reportable. Confirmatory execution may be "
            "sequenced by qualification strength, but the frozen grid remains "
            "unchanged."
        ),
    },
    "secondary_breadth_extensions_after_primary_grid": {
        "required": True,
        "policies": [
            "clean_label_target_only",
            "distributed_subtrigger_dirty_label",
        ],
        "representative_slots": [
            "flow_iat_low",
            "init_windows_exact",
            "active_idle_exact",
        ],
        "poison_fractions": [0.05, 0.10, 0.20],
        "seeds": CONFIRMATORY_SEEDS,
        "defense_retuning_permitted": False,
    },
    "primary_metrics": [
        "triggered_attack_success_rate_all_nonbenign_sources",
        "macro_average_per_source_triggered_ASR",
        "worst_source_triggered_ASR",
        "clean_validation_macro_F1",
    ],
    "detector_metrics": [
        "malicious_client_recall",
        "benign_client_false_positive_rate",
        "replaced_client_count",
    ],
    "reporting_requirements": [
        "mean_standard_deviation_and_95_percent_CI_across_seeds",
        "per_trigger_per_fraction_curves",
        "per_source_class_triggered_ASR",
        "matched_clean_and_plain_FedAvg_references",
        "CSV_tables",
        "publication_quality_PNG_and_PDF_figures",
    ],
}

panel_path = TABLE_DIR / "task41a_frozen_trigger_panel.csv"
robustness_path = TABLE_DIR / "task41a_frozen_source_robustness.csv"
source_panel_path = TABLE_DIR / "task41a_frozen_source_metrics_long.csv"
qualification_path = TABLE_DIR / "task41a_primary_qualification_grid.csv"
protocol_path = TABLE_DIR / "task41_frozen_backdoor_protocol.json"

panel.to_csv(panel_path, index=False)
robustness.to_csv(robustness_path, index=False)
source_panel.to_csv(source_panel_path, index=False)
qualification_grid.to_csv(qualification_path, index=False)
protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

plot_frame = panel.sort_values(
    "accidental_prevalence_validation",
    ascending=True,
).reset_index(drop=True)

fig, axis = plt.subplots(figsize=(10.5, 6.2))
axis.scatter(
    plot_frame["accidental_prevalence_validation"],
    plot_frame["median_standardized_l2"],
    s=90,
)
for _, row in plot_frame.iterrows():
    axis.annotate(
        row["panel_slot"],
        (
            row["accidental_prevalence_validation"],
            row["median_standardized_l2"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )
axis.set_xscale("log")
axis.set_xlabel("Accidental validation prevalence, log scale")
axis.set_ylabel("Median standardized trigger L2")
axis.set_title("Task 41A Frozen Trigger Panel")
axis.grid(True, alpha=0.25)
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(
        FIGURE_DIR / f"task41a_frozen_trigger_panel_geometry.{suffix}",
        dpi=300,
        bbox_inches="tight",
    )
plt.close(fig)

heat = source_panel.pivot(
    index="candidate_id",
    columns="source_class_name",
    values="pca_reconstruction_ratio",
).reindex(
    index=panel["candidate_id"],
    columns=SOURCE_CLASS_NAMES,
)

fig, axis = plt.subplots(figsize=(10.5, 5.8))
image = axis.imshow(heat.to_numpy(), aspect="auto")
axis.set_xticks(np.arange(len(heat.columns)))
axis.set_xticklabels(heat.columns, rotation=35, ha="right")
axis.set_yticks(np.arange(len(heat.index)))
axis.set_yticklabels(
    [
        panel.set_index("candidate_id").loc[candidate_id, "panel_slot"]
        for candidate_id in heat.index
    ],
    fontsize=8,
)
axis.set_title("Source-Class PCA Reconstruction Ratio")
figure_colorbar = fig.colorbar(image, ax=axis)
figure_colorbar.set_label("Reconstruction ratio")
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(
        FIGURE_DIR / f"task41a_frozen_source_pca_ratios.{suffix}",
        dpi=300,
        bbox_inches="tight",
    )
plt.close(fig)

artifact_paths = [
    panel_path,
    robustness_path,
    source_panel_path,
    qualification_path,
    protocol_path,
    spec_json_path,
    spec_npz_path,
    FIGURE_DIR / "task41a_frozen_trigger_panel_geometry.png",
    FIGURE_DIR / "task41a_frozen_trigger_panel_geometry.pdf",
    FIGURE_DIR / "task41a_frozen_source_pca_ratios.png",
    FIGURE_DIR / "task41a_frozen_source_pca_ratios.pdf",
]
manifest_rows = [
    {
        "relative_path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    for path in artifact_paths
]
manifest_path = TABLE_DIR / "task41a_frozen_protocol_manifest_sha256.csv"
with manifest_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["relative_path", "bytes", "sha256"],
    )
    writer.writeheader()
    writer.writerows(manifest_rows)

print("===== TASK 41A FROZEN TRIGGER PANEL =====")
print("Branch:", branch)
print("Commit:", commit)
print("Selection used attack-training outcomes: False")
print("Final test arrays loaded: False")
print()
print(
    panel[
        [
            "panel_slot",
            "candidate_id",
            "family",
            "mode",
            "feature_count",
            "accidental_prevalence_validation",
            "median_standardized_l2",
            "geometry_score",
            "source_class_count",
            "minimum_source_semantic_valid_rate",
        ]
    ].to_string(index=False)
)
print()
print("Primary qualification configurations:", len(qualification_grid))
print(
    "Confirmatory attack configurations before plain/defense pairing:",
    confirmatory_configuration_count,
)
print(
    "Confirmatory paired branches:",
    confirmatory_configuration_count * 2,
)
print()
for path in artifact_paths + [manifest_path]:
    print("WROTE:", path)
print("FINAL TEST ARRAYS LOADED: False")

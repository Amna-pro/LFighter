from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARENT_TAG = "task57-c2-attribution-recovery-pass-v4192"
PARENT_COMMIT_SHORT = "20122c6"
FAMILY_LABELS = {
    "A_development_anchor": "A: development anchor",
    "B_hash_ranked": "B: hash ranked",
    "C_hash_ranked": "C: hash ranked",
}
FAMILY_COLORS = {
    "A_development_anchor": "#4063D8",
    "B_hash_ranked": "#E07A2D",
    "C_hash_ranked": "#2A9D78",
}
STATE_COLORS = {"Clean": "#4C78A8", "Suspicious": "#D1495B", "Reconstructed": "#2A9D78"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, figures: Path, stem: str, dpi: int, figure_id: str, title: str, selection: str, rows: int) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for fmt in ["png", "pdf"]:
        path = figures / f"{stem}.{fmt}"
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        width, height = png_size(path) if fmt == "png" else (None, None)
        entries.append(
            {
                "figure_id": figure_id,
                "stem": stem,
                "title": title,
                "selection_rule": selection,
                "source_rows_represented": rows,
                "format": fmt,
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "pixel_width": width,
                "pixel_height": height,
            }
        )
    plt.close(fig)
    return entries


def state_npz(seed7_root: Path, multiseed_root: Path, seed: int, family: str | None, state: str) -> Path:
    if seed == 7:
        base = seed7_root
        filename = "task55c2_attributions.npz"
    else:
        base = multiseed_root / f"seed_{seed}"
        filename = "task55c3_attributions.npz"
    if state == "clean_reference":
        return base / "common/clean_reference" / filename
    return base / "families" / str(family) / state / filename


def build_validation_dashboard(t56: Path, t57: Path) -> tuple[plt.Figure, pd.DataFrame, int]:
    stability = pd.read_csv(t56 / "task56c2_repeated_run_stability.csv")
    background = pd.read_csv(t56 / "task56c2_background_sensitivity.csv")
    probe = pd.read_csv(t56 / "task56c2_probe_size_sensitivity.csv")
    faith = pd.read_csv(t56 / "task56c2_faithfulness_state_summary.csv")
    specificity = pd.read_csv(t56 / "task56c2_class_specificity_state_summary.csv")
    cross = pd.read_csv(t56 / "task56c2_cross_seed_consistency.csv")
    recovery_decision = json.loads((t57.parent / "task57c2_attribution_recovery_decision.json").read_text(encoding="utf-8"))
    noise_q95 = float(recovery_decision["attack_signal"]["repeat_noise_q95"])
    domains = [
        ("Repeat", stability["spearman_rho"]),
        ("Background", background["spearman_rho"]),
        ("Probe size", probe["spearman_rho"]),
        ("Cross seed", cross["spearman_rho"]),
        ("Faithfulness", faith["feature_impact_spearman_rho"]),
    ]
    summary_rows: list[dict[str, object]] = []
    for name, values in domains:
        summary_rows.append({"metric": "Spearman rho", "domain": name, "n": len(values), "median": values.median(), "q25": values.quantile(0.25), "q75": values.quantile(0.75), "minimum": values.min(), "maximum": values.max()})
    for name, values in [("Repeat", stability["top10_jaccard"]), ("Background", background["top10_jaccard"]), ("Probe size", probe["top10_jaccard"])]:
        summary_rows.append({"metric": "Top 10 Jaccard", "domain": name, "n": len(values), "median": values.median(), "q25": values.quantile(0.25), "q75": values.quantile(0.75), "minimum": values.min(), "maximum": values.max()})
    summary_rows.append({"metric": "Normalized L1 drift", "domain": "Repeat", "n": len(stability), "median": stability.normalized_l1_drift.median(), "q25": stability.normalized_l1_drift.quantile(0.25), "q75": stability.normalized_l1_drift.quantile(0.75), "minimum": stability.normalized_l1_drift.min(), "maximum": stability.normalized_l1_drift.max()})
    summary_rows.append({"metric": "Class specificity", "domain": "All states", "n": len(specificity), "median": specificity.median_within_minus_between.median(), "q25": specificity.median_within_minus_between.quantile(0.25), "q75": specificity.median_within_minus_between.quantile(0.75), "minimum": specificity.median_within_minus_between.min(), "maximum": specificity.median_within_minus_between.max()})
    summary = pd.DataFrame(summary_rows)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.1), constrained_layout=True)
    ax = axes[0, 0]
    ax.boxplot([x[1].to_numpy() for x in domains], tick_labels=[x[0] for x in domains], showfliers=False, patch_artist=True, boxprops={"facecolor": "#BFD7EA"}, medianprops={"color": "#173F5F", "linewidth": 1.5})
    ax.set_ylim(0.8, 1.01)
    ax.set_ylabel("Spearman correlation")
    ax.set_title("A  Attribution consistency and faithfulness")
    ax.tick_params(axis="x", rotation=18)

    ax = axes[0, 1]
    j_domains = [("Repeat", stability.top10_jaccard), ("Background", background.top10_jaccard), ("Probe size", probe.top10_jaccard)]
    x = np.arange(3)
    med = [v.median() for _, v in j_domains]
    low = [v.quantile(0.25) for _, v in j_domains]
    high = [v.quantile(0.75) for _, v in j_domains]
    ax.bar(x, med, color=["#4C78A8", "#72B7B2", "#F2CF5B"], width=0.62)
    ax.errorbar(x, med, yerr=[np.array(med) - np.array(low), np.array(high) - np.array(med)], fmt="none", color="#222222", capsize=3)
    ax.set_xticks(x, [n for n, _ in j_domains])
    ax.set_ylim(0.5, 1.03)
    ax.set_ylabel("Top 10 Jaccard overlap")
    ax.set_title("B  Top feature set stability")

    ax = axes[1, 0]
    ax.hist(stability.normalized_l1_drift, bins=12, color="#8F6BB3", alpha=0.85, edgecolor="white")
    ax.axvline(noise_q95, color="#C73E1D", linestyle="--", linewidth=1.7, label=f"Frozen q95 = {noise_q95:.3f}")
    ax.set_xlabel("Normalized L1 attribution drift")
    ax.set_ylabel("Repeat evaluations")
    ax.set_title("C  Repeated run noise distribution")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    order = ["clean_reference", "suspicious", "reconstructed"]
    labels = ["Clean reference", "Suspicious", "Reconstructed"]
    data = [specificity.loc[specificity.state == state, "median_within_minus_between"].to_numpy() for state in order]
    ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, boxprops={"facecolor": "#CDE8D5"}, medianprops={"color": "#1B5E20", "linewidth": 1.5})
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_ylabel("Within class minus between class correlation")
    ax.set_title("D  Class specificity across states")
    ax.tick_params(axis="x", rotation=12)
    fig.suptitle("Task 56 validation: explanations are stable, faithful, and class specific", fontsize=13, fontweight="bold")
    source_rows = len(stability) + len(background) + len(probe) + len(faith) + len(specificity) + len(cross)
    return fig, summary, source_rows


def build_recovery_figure(triplets: pd.DataFrame) -> plt.Figure:
    data = triplets.sort_values(["seed", "family_id"]).reset_index(drop=True)
    labels = [f"{int(r.seed)}  {r.family_id[0]}" for r in data.itertuples()]
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    for idx, row in data.iterrows():
        color = FAMILY_COLORS[row.family_id]
        ax.plot([row.attack_distance, row.reconstructed_distance], [idx, idx], color=color, alpha=0.65, linewidth=2)
    ax.scatter(data.attack_distance, y, s=48, color=STATE_COLORS["Suspicious"], marker="o", label="Suspicious", zorder=3)
    ax.scatter(data.reconstructed_distance, y, s=54, color=STATE_COLORS["Reconstructed"], marker="D", label="Reconstructed", zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized L1 distance from paired clean reference")
    ax.set_ylabel("Seed and coalition family")
    ax.set_title("Attribution distance decreases after reconstruction in all 12 paired triplets", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.01, 0.015, "All 12 triplets shown; no case filtering", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="#555555")
    return fig


def build_trajectory_figure(triplets: pd.DataFrame) -> plt.Figure:
    states = ["Clean", "Suspicious", "Reconstructed"]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), constrained_layout=True)
    for family, family_data in triplets.groupby("family_id", sort=True):
        color = FAMILY_COLORS[family]
        for row in family_data.itertuples():
            axes[0].plot(x, [row.clean_mean_ddos_minus_benign_logit_margin, row.suspicious_mean_ddos_minus_benign_logit_margin, row.reconstructed_mean_ddos_minus_benign_logit_margin], color=color, alpha=0.42, linewidth=1.3)
            axes[1].plot(x, [row.clean_ddos_prediction_rate, row.suspicious_ddos_prediction_rate, row.reconstructed_ddos_prediction_rate], color=color, alpha=0.42, linewidth=1.3)
        axes[0].plot([], [], color=color, label=FAMILY_LABELS[family])
    margin_matrix = triplets[["clean_mean_ddos_minus_benign_logit_margin", "suspicious_mean_ddos_minus_benign_logit_margin", "reconstructed_mean_ddos_minus_benign_logit_margin"]].to_numpy()
    rate_matrix = triplets[["clean_ddos_prediction_rate", "suspicious_ddos_prediction_rate", "reconstructed_ddos_prediction_rate"]].to_numpy()
    axes[0].plot(x, np.median(margin_matrix, axis=0), color="#111111", linewidth=3, marker="o", label="Cohort median")
    axes[1].plot(x, np.median(rate_matrix, axis=0), color="#111111", linewidth=3, marker="o")
    for ax in axes:
        ax.set_xticks(x, states)
        ax.axvspan(0.92, 1.08, color="#D1495B", alpha=0.08)
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set_ylabel("Mean DDoS minus Benign logit margin")
    axes[0].set_title("A  Source target margin")
    axes[1].set_ylabel("Correct DDoS prediction rate")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("B  DDoS prediction recovery")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Prediction behavior across the clean, attacked, and reconstructed states", fontsize=13, fontweight="bold")
    return fig


def build_feature_figure(top10: pd.DataFrame) -> plt.Figure:
    data = top10.sort_values("rank", ascending=False).reset_index(drop=True)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    ax.barh(y + 0.18, data.median_absolute_attack_shift, height=0.34, color=STATE_COLORS["Suspicious"], label="Absolute attack shift")
    ax.barh(y - 0.18, data.median_absolute_reconstructed_shift, height=0.34, color=STATE_COLORS["Reconstructed"], label="Residual shift after reconstruction")
    ax.set_yticks(y, data.feature)
    ax.set_xlabel("Median absolute attribution shift")
    ax.set_title("Frozen top 10 poisoning associated features and reconstruction recovery", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    for i, row in data.iterrows():
        ax.text(row.median_absolute_attack_shift + 0.008, i + 0.18, f"{100 * row.feature_recovery_fraction:.0f}% recovered", va="center", fontsize=7, color="#333333")
    return fig


def build_case_figure(triplets: pd.DataFrame, seed7_root: Path, multiseed_root: Path) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame, list[Path]]:
    median_recovery = float(triplets.recovery_fraction.median())
    selected = triplets.assign(selection_gap=(triplets.recovery_fraction - median_recovery).abs()).sort_values(["selection_gap", "seed", "family_id"]).iloc[0]
    seed = int(selected.seed)
    family = str(selected.family_id)
    paths = {
        "Clean": state_npz(seed7_root, multiseed_root, seed, None, "clean_reference"),
        "Suspicious": state_npz(seed7_root, multiseed_root, seed, family, "suspicious"),
        "Reconstructed": state_npz(seed7_root, multiseed_root, seed, family, "reconstructed"),
    }
    arrays = {name: np.load(path, allow_pickle=False) for name, path in paths.items()}
    suspicious = arrays["Suspicious"]
    ddos_rows = np.flatnonzero(suspicious["true_class_id"] == 2)
    margins = suspicious["logits"][ddos_rows, 2] - suspicious["logits"][ddos_rows, 0]
    median_margin = float(np.median(margins))
    local_idx = int(np.argmin(np.abs(margins - median_margin)))
    row_idx = int(ddos_rows[local_idx])
    feature_names = suspicious["feature_names"].astype(str)
    suspicious_abs = np.abs(suspicious["source_minus_target_margin_shap"][row_idx])
    top_idx = np.argsort(-suspicious_abs, kind="stable")[:8]
    contributions: list[dict[str, object]] = []
    state_payload: dict[str, tuple[float, float, np.ndarray]] = {}
    for state, data in arrays.items():
        shap = data["source_minus_target_margin_shap"][row_idx].astype(float)
        margin = float(data["logits"][row_idx, 2] - data["logits"][row_idx, 0])
        baseline = margin - float(shap.sum())
        state_payload[state] = (baseline, margin, shap)
        for idx in top_idx:
            contributions.append({"state": state, "feature": feature_names[idx], "shap_contribution": shap[idx], "absolute_contribution": abs(shap[idx]), "probe_row_index": row_idx, "probe_validation_index": int(data["probe_validation_indices"][row_idx])})
        other = float(shap.sum() - shap[top_idx].sum())
        contributions.append({"state": state, "feature": "All other features", "shap_contribution": other, "absolute_contribution": abs(other), "probe_row_index": row_idx, "probe_validation_index": int(data["probe_validation_indices"][row_idx])})
    labels = [feature_names[i] for i in top_idx] + ["All other features"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.7), sharey=True, constrained_layout=True)
    max_abs = max(abs(float(x["shap_contribution"])) for x in contributions) * 1.15
    for ax, state in zip(axes, ["Clean", "Suspicious", "Reconstructed"]):
        state_rows = [x for x in contributions if x["state"] == state]
        values = np.array([float(x["shap_contribution"]) for x in state_rows])
        colors = np.where(values >= 0, "#2A9D78", "#D1495B")
        y = np.arange(len(labels))
        ax.barh(y, values, color=colors, alpha=0.9)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_xlim(-max_abs, max_abs)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        baseline, margin, _ = state_payload[state]
        ax.set_title(f"{state}\nmargin {margin:.3f}, base {baseline:.3f}")
        ax.set_xlabel("SHAP contribution to DDoS minus Benign margin")
    fig.suptitle(f"Deterministic representative case: seed {seed}, {FAMILY_LABELS[family]}\nTriplet nearest cohort median recovery; probe row nearest suspicious median margin", fontsize=12, fontweight="bold")
    case_selection = pd.DataFrame(
        [
            {
                "cohort_median_recovery_fraction": median_recovery,
                "selected_seed": seed,
                "selected_family_id": family,
                "selected_recovery_fraction": float(selected.recovery_fraction),
                "selection_gap": float(selected.selection_gap),
                "selected_probe_row_index": row_idx,
                "selected_probe_validation_index": int(suspicious["probe_validation_indices"][row_idx]),
                "suspicious_probe_margin": float(margins[local_idx]),
                "suspicious_median_probe_margin": median_margin,
                "probe_margin_selection_gap": abs(float(margins[local_idx]) - median_margin),
                "manual_substitution": False,
            }
        ]
    )
    return fig, case_selection, pd.DataFrame(contributions), list(paths.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--task56-root", default="results/cic_iot_diad_task56_c2_summary_v418")
    parser.add_argument("--task57-root", default="results/cic_iot_diad_task57_c2_summary_v419")
    parser.add_argument("--task55-seed7-root", default="results/cic_iot_diad_task55_c2_seed7_v417")
    parser.add_argument("--task55-multiseed-root", default="results/cic_iot_diad_task55_c3_multiseed_v417")
    parser.add_argument("--output-root", default="results/cic_iot_diad_task58_c1_publication_v420")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    t56 = (root / args.task56_root / "tables").resolve()
    t57_root = (root / args.task57_root).resolve()
    t57 = t57_root / "tables"
    seed7_root = (root / args.task55_seed7_root).resolve()
    multiseed_root = (root / args.task55_multiseed_root).resolve()
    output = (root / args.output_root).resolve()
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    set_style()
    source_files: list[Path] = []
    for name in ["task56c2_repeated_run_stability.csv", "task56c2_background_sensitivity.csv", "task56c2_probe_size_sensitivity.csv", "task56c2_faithfulness_state_summary.csv", "task56c2_class_specificity_state_summary.csv", "task56c2_cross_seed_consistency.csv"]:
        source_files.append(t56 / name)
    for name in ["task57c2_triplet_recovery_metrics.csv", "task57c2_top10_prespecified_features.csv", "task57c2_feature_association.csv", "task57c2_state_coverage.csv"]:
        source_files.append(t57 / name)
    source_files.append(t57_root / "task57c2_attribution_recovery_decision.json")
    triplets = pd.read_csv(t57 / "task57c2_triplet_recovery_metrics.csv")
    top10 = pd.read_csv(t57 / "task57c2_top10_prespecified_features.csv")
    manifest: list[dict[str, object]] = []

    fig, validation_summary, nrows = build_validation_dashboard(t56, t57)
    validation_summary.to_csv(tables / "task58c1_validation_metric_summary.csv", index=False)
    manifest.extend(save_figure(fig, figures, "task58_xai_validation_dashboard", args.dpi, "F1", "XAI validation dashboard", "All available Task 56 validation rows", nrows))
    fig = build_recovery_figure(triplets)
    manifest.extend(save_figure(fig, figures, "task58_all_triplet_recovery", args.dpi, "F2", "All triplet attribution recovery", "All twelve Task 57 triplets", len(triplets)))
    fig = build_trajectory_figure(triplets)
    manifest.extend(save_figure(fig, figures, "task58_prediction_state_trajectories", args.dpi, "F3", "Prediction state trajectories", "All twelve Task 57 triplets", len(triplets)))
    fig = build_feature_figure(top10)
    manifest.extend(save_figure(fig, figures, "task58_top10_poison_associated_features", args.dpi, "F4", "Top ten poisoning associated features", "Frozen Task 57 top ten prespecified ranking", len(top10)))
    fig, case_selection, case_contributions, case_paths = build_case_figure(triplets, seed7_root, multiseed_root)
    source_files.extend(case_paths)
    case_selection.to_csv(tables / "task58c1_case_selection.csv", index=False)
    case_contributions.to_csv(tables / "task58c1_case_feature_contributions.csv", index=False)
    manifest.extend(save_figure(fig, figures, "task58_representative_case_waterfall", args.dpi, "F5", "Deterministic representative case waterfall", "Triplet nearest cohort median recovery and probe nearest suspicious median margin", 1))
    manifest_df = pd.DataFrame(manifest)
    manifest_df["path"] = manifest_df["path"].map(lambda x: Path(x).relative_to(root).as_posix())
    manifest_df.to_csv(tables / "task58c1_figure_manifest.csv", index=False)
    with (tables / "task58c1_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        for path in sorted(set(source_files)):
            writer.writerow([path.relative_to(root).as_posix(), path.stat().st_size, sha256(path)])

    captions = """# Task 58 figure captions

**Figure F1. XAI validation dashboard.** Complete Task 56 evidence across 28 physical states, including repeated run stability, alternate background sensitivity, probe size sensitivity, feature perturbation faithfulness, class specificity, and cross seed consistency. The dashed drift threshold is the frozen repeated run q95 used by Task 57.

**Figure F2. Attribution recovery across every paired triplet.** Normalized L1 distance between each suspicious or reconstructed attribution profile and its paired round 4 preattack clean reference. All twelve combinations of four seeds and three coalition families are shown. Lower is closer to the paired clean reference.

**Figure F3. Prediction behavior across states.** DDoS minus Benign logit margin and correct DDoS prediction rate for all twelve triplets. Thin curves are individual triplets and the black curve is the cohort median.

**Figure F4. Frozen top ten poisoning associated features.** Median absolute attack shift and residual reconstructed shift for the ten features selected by the prespecified Task 57 ranking. Percent annotations report feature level recovery. No feature was selected during figure generation.

**Figure F5. Deterministic representative case.** Per feature SHAP contributions for the true DDoS probe row selected by the frozen mechanical rule. The triplet is closest to the cohort median recovery fraction and the row is closest to the suspicious state median DDoS minus Benign margin. Green supports the source target margin and red opposes it.

**Scope limitation.** Rejected and round 8 oracle clean attribution states were unavailable. No substitute states were created. Clean refers to the paired round 4 preattack reference, so recovery means movement toward that reference rather than proof of oracle clean restoration.
"""
    (output / "TASK58_FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")
    decision = {
        "experiment_version": "4.20.1",
        "parent_tag": PARENT_TAG,
        "parent_commit_short": PARENT_COMMIT_SHORT,
        "figure_stems": 5,
        "figure_files": 10,
        "triplets_represented": int(len(triplets)),
        "features_in_frozen_ranking": int(len(top10)),
        "case_selection_deterministic": True,
        "manual_case_substitution": False,
        "all_cohort_rows_used": True,
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
        "rejected_state_available": False,
        "oracle_clean_state_available": False,
        "clean_reference": "paired round 4 preattack state",
        "scientific_result_inherited_from_task57": "PASS",
        "task58_c1_complete": True,
    }
    (output / "task58c1_publication_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    closeout = f"""# Task 58 publication figure closeout

Task 58 generated five publication figure families in PNG and PDF formats from the frozen Task 56 and Task 57 evidence. Cohort figures include all twelve triplets, the feature panel uses the frozen top ten ranking, and the individual case follows the preregistered deterministic selection rule.

The selected case is seed {int(case_selection.iloc[0].selected_seed)} and family {case_selection.iloc[0].selected_family_id}. This selection was made because its recovery fraction was closest to the twelve triplet median, not because it maximized recovery.

No training or new SHAP evaluation occurred. Rejected and round 8 oracle clean states remain unavailable. The figures therefore support movement toward a paired round 4 preattack clean reference, not oracle clean restoration.
"""
    (output / "TASK58_C1_CLOSEOUT.md").write_text(closeout, encoding="utf-8")
    print("===== TASK 58 C1 PUBLICATION FIGURES =====")
    print("FIGURE STEMS: 5")
    print("FIGURE FILES: 10")
    print("TRIPLETS REPRESENTED:", len(triplets))
    print("CASE SELECTION DETERMINISTIC: True")
    print("MANUAL CASE SUBSTITUTION: False")
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("TASK 58 C1 COMPLETE: True")


if __name__ == "__main__":
    main()

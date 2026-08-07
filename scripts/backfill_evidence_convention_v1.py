#!/usr/bin/env python3
"""Evidence backfill: missing figures + SHA256 manifests for already-frozen stages.

Purpose
-------
Brings Task 42 C1a/C1b, Task 42/43 selection, and Task 44 C1 into
compliance with the project's Research Evidence Rule (every stage leaves
behind CSVs, PNG/PDF figures, a JSON decision file, and a SHA256
manifest). This is pure reprocessing of already-frozen CSVs -- it trains
nothing, reruns nothing, and does not modify any existing frozen file.
It only ADDS a figures/ folder and a manifest CSV where the convention
requires one and none currently exists.

Each stage's backfilled outputs are written into that stage's OWN
existing directory (not a new experiment folder), matching the
convention's <experiment>/<stage>/figures/ and
<experiment>/<stage>/tables/ structure exactly.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill missing figures and manifests.")
    parser.add_argument("--task42-c1a-dir", type=Path, required=True)
    parser.add_argument("--task42-c1b-dir", type=Path, required=True)
    parser.add_argument("--task42-c2-summary-dir", type=Path, required=True)
    parser.add_argument("--task43-c2-summary-dir", type=Path, required=True)
    parser.add_argument("--task44-c1-dir", type=Path, required=True)
    return parser.parse_args()


def save_figure(figure: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(stage_dir: Path, manifest_name: str) -> Path:
    """Hash every file already in this stage directory (including the
    figures just added) into a standard SHA256 manifest CSV, matching
    the convention. Excludes the manifest file itself.
    """
    manifest_path = stage_dir / "tables" / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(stage_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() == manifest_path.resolve():
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(stage_dir)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def bar_with_ceiling(
    labels: List[str], values: List[float], title: str, ylabel: str,
    ceiling: float | None, ceiling_label: str, out_base: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 5.5))
    bars = axis.bar(labels, values, color="#4C72B0")
    if ceiling is not None:
        axis.axhline(ceiling, color="#C44E52", linestyle="--", linewidth=1.5, label=ceiling_label)
        axis.legend()
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    plt.setp(axis.get_xticklabels(), rotation=20, ha="right")
    for bar, value in zip(bars, values):
        axis.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    save_figure(figure, out_base)


def backfill_task42_c1a(stage_dir: Path) -> None:
    tables = stage_dir / "tables"
    d1 = pd.read_csv(tables / "task42c1a_d1_norm_reference.csv")
    d2 = pd.read_csv(tables / "task42c1a_d2_direction_reference.csv")

    figure, axis = plt.subplots(figsize=(9.0, 5.5))
    for seed, group in d1.groupby("model_seed"):
        axis.plot(group["client_id"], group["log_norm_median"], marker="o", label=f"seed {seed}")
    axis.set_xlabel("Client ID")
    axis.set_ylabel("D1 log-norm median (per-client baseline)")
    axis.set_title("Task 42 C1a: D1 per-client norm baseline, all seeds")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    save_figure(figure, stage_dir / "figures" / "task42c1a_d1_baseline_by_client")

    figure, axis = plt.subplots(figsize=(9.0, 5.5))
    for seed, group in d2.groupby("model_seed"):
        axis.plot(group["client_id"], group["residual_l2_norm"], marker="o", label=f"seed {seed}")
    axis.set_xlabel("Client ID")
    axis.set_ylabel("D2 reference residual L2 norm")
    axis.set_title("Task 42 C1a: D2 per-client reference-direction magnitude, all seeds")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    save_figure(figure, stage_dir / "figures" / "task42c1a_d2_reference_magnitude_by_client")

    write_manifest(stage_dir, "task42c1a_evidence_manifest_sha256.csv")
    print(f"[OK] Task 42 C1a backfilled: {stage_dir}")


def backfill_task42_c1b(stage_dir: Path) -> None:
    tables = stage_dir / "tables"
    thresholds = pd.read_csv(tables / "task42c1b_pooled_threshold_candidates.csv")
    per_seed = pd.read_csv(tables / "task42c1b_per_seed_summary.csv")

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(thresholds["quantile"], thresholds["d1_instant_threshold"], marker="o", label="D1 threshold")
    axis2 = axis.twinx()
    axis2.plot(thresholds["quantile"], thresholds["d2_instant_threshold"], marker="s", color="#C44E52", label="D2 threshold")
    axis.set_xlabel("Quantile")
    axis.set_ylabel("D1 instant threshold", color="#4C72B0")
    axis2.set_ylabel("D2 instant threshold", color="#C44E52")
    axis.set_title("Task 42 C1b: D1/D2 threshold candidates by quantile")
    save_figure(figure, stage_dir / "figures" / "task42c1b_threshold_candidates")

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    width = 0.35
    x = range(len(per_seed))
    axis.bar([i - width / 2 for i in x], per_seed["mean_d1_z"], width=width, label="D1 mean z")
    axis.bar([i + width / 2 for i in x], per_seed["mean_d2_deviation"], width=width, label="D2 mean deviation")
    axis.set_xticks(list(x))
    axis.set_xticklabels([f"seed {s}" for s in per_seed["model_seed"]])
    axis.set_title("Task 42 C1b: pooled clean D1/D2 scores by seed")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, stage_dir / "figures" / "task42c1b_per_seed_clean_scores")

    write_manifest(stage_dir, "task42c1b_evidence_manifest_sha256.csv")
    print(f"[OK] Task 42 C1b backfilled: {stage_dir}")


def backfill_task42_c2_selection(summary_dir: Path) -> None:
    selection = pd.read_csv(summary_dir / "tables" / "task42c2_candidate_selection.csv")
    labels = selection["candidate"].str.replace("_", "\n", regex=False).tolist()

    bar_with_ceiling(labels, selection["maximum_benign_fpr"].tolist(),
                      "Task 42 C2: maximum benign FPR by candidate", "Max benign FPR",
                      0.05, "Frozen 0.05 ceiling", summary_dir / "figures" / "task42c2_max_fpr_by_candidate")
    bar_with_ceiling(labels, selection["mean_malicious_recall_all_conditions"].tolist(),
                      "Task 42 C2: mean malicious recall by candidate", "Mean recall",
                      0.50, "H1 floor (0.50)", summary_dir / "figures" / "task42c2_mean_recall_by_candidate")
    bar_with_ceiling(labels, selection["mean_damage_removed_fraction"].tolist(),
                      "Task 42 C2: mean damage-removed fraction by candidate", "Damage-removed fraction",
                      0.25, "H2 floor (0.25)", summary_dir / "figures" / "task42c2_damage_removed_by_candidate")

    write_manifest(summary_dir, "task42c2_evidence_manifest_sha256.csv")
    print(f"[OK] Task 42 C2 selection backfilled: {summary_dir}")


def backfill_task43_c2_selection(summary_dir: Path) -> None:
    selection = pd.read_csv(summary_dir / "tables" / "task43c2_candidate_selection.csv")
    h4 = pd.read_csv(summary_dir / "tables" / "task43c2_h4_baseline_comparison.csv")

    d0_only = selection[selection["arm"] == "D0_frozen_task40_lfighter"]
    bar_with_ceiling(d0_only["arm"].tolist(), d0_only["maximum_benign_fpr"].tolist(),
                      "Task 43 C2: D0 maximum benign FPR", "Max benign FPR",
                      0.05, "Frozen 0.05 ceiling", summary_dir / "figures" / "task43c2_d0_max_fpr")

    all_labels = selection["arm"].str.replace("_", "\n", regex=False).tolist()
    bar_with_ceiling(all_labels, selection["mean_clean_macro_f1_loss"].tolist(),
                      "Task 43 C2: mean clean macro-F1 loss by arm (incl. baselines)", "Mean clean F1 loss",
                      0.01, "H3 ceiling (0.01)", summary_dir / "figures" / "task43c2_clean_loss_all_arms")
    bar_with_ceiling(all_labels, selection["mean_damage_removed_fraction"].tolist(),
                      "Task 43 C2: mean damage-removed fraction by arm", "Damage-removed fraction",
                      None, "", summary_dir / "figures" / "task43c2_damage_removed_all_arms")

    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    width = 0.35
    x = range(len(h4))
    axis.bar([i - width / 2 for i in x], h4["d0_mean_damage_removed_fraction"], width=width, label="D0")
    axis.bar([i + width / 2 for i in x], h4["best_baseline_mean_damage_removed_fraction"], width=width, label="Best baseline")
    axis.set_xticks(list(x))
    axis.set_xticklabels(h4["attack_type"].tolist())
    axis.set_title("Task 43 H4: D0 vs best available baseline, by attack type")
    axis.set_ylabel("Mean damage-removed fraction")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, summary_dir / "figures" / "task43c2_h4_d0_vs_best_baseline")

    write_manifest(summary_dir, "task43c2_evidence_manifest_sha256.csv")
    print(f"[OK] Task 43 C2 selection backfilled: {summary_dir}")


def backfill_task44_c1(stage_dir: Path) -> None:
    anchor = pd.read_csv(stage_dir / "tables" / "task44c1_strength_curve_anchor_1p0.csv")
    labels = anchor["attack_type"].str.replace("_", "\n", regex=False).tolist()

    bar_with_ceiling(labels, anchor["mean_benign_fpr"].tolist(),
                      "Task 44 C1: benign FPR at poison_fraction=1.0 (frozen Task 40 anchor)", "Benign FPR",
                      0.05, "Frozen 0.05 ceiling", stage_dir / "figures" / "task44c1_anchor_fpr")
    bar_with_ceiling(labels, anchor["mean_damage_removed_fraction"].tolist(),
                      "Task 44 C1: damage-removed at poison_fraction=1.0 (frozen Task 40 anchor)",
                      "Damage-removed fraction", None, "",
                      stage_dir / "figures" / "task44c1_anchor_damage_removed")

    write_manifest(stage_dir, "task44c1_evidence_manifest_sha256.csv")
    print(f"[OK] Task 44 C1 backfilled: {stage_dir}")
    print("     NOTE: this is the poison_fraction=1.0 anchor point only, not")
    print("     the full strength curve -- 0.25/0.50/0.75 data does not exist yet.")


def main() -> int:
    args = parse_args()
    backfill_task42_c1a(args.task42_c1a_dir.expanduser().resolve())
    backfill_task42_c1b(args.task42_c1b_dir.expanduser().resolve())
    backfill_task42_c2_selection(args.task42_c2_summary_dir.expanduser().resolve())
    backfill_task43_c2_selection(args.task43_c2_summary_dir.expanduser().resolve())
    backfill_task44_c1(args.task44_c1_dir.expanduser().resolve())
    print()
    print("Evidence backfill complete. No existing frozen file was modified;")
    print("only figures/ and manifest CSVs were added.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

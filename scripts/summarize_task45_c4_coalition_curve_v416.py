#!/usr/bin/env python3
"""Produce the preregistered Task 45 C4 coalition size analysis."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


C3_TAG = "task45-c3-confirmatory-frozen-v4163"
SEEDS = (7, 99, 123, 2026)
FAMILIES = ("A_development_anchor", "B_hash_ranked", "C_hash_ranked")
SIZES = (1, 2, 4, 6, 8, 10)
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 451604


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True, type=Path)
    p.add_argument("--coalition-manifest", required=True, type=Path)
    p.add_argument("--warmup-root", required=True, type=Path)
    p.add_argument("--c2-root", required=True, type=Path)
    p.add_argument("--c3-root", required=True, type=Path)
    p.add_argument("--c3-audit-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_rounds(branch: Path, filename: str) -> pd.DataFrame:
    path = branch / "tables" / filename
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path).sort_values("monitoring_round").reset_index(drop=True)
    if frame["monitoring_round"].astype(int).tolist() != [1, 2, 3, 4]:
        raise ValueError(f"Expected four ordered monitoring rounds in {path}")
    return frame


def mean_ci(values: Iterable[float], rng: np.random.Generator) -> Tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    draws = rng.choice(array, size=(BOOTSTRAP_REPLICATES, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(array.mean()), float(low), float(high)


def exact_sign_flip_p(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    array = array[np.abs(array) > 1e-15]
    n = len(array)
    if n == 0:
        return 1.0
    observed = abs(float(array.mean()))
    if n <= 20:
        extreme = 0
        total = 1 << n
        for mask in range(total):
            signs = np.fromiter((1.0 if mask & (1 << i) else -1.0 for i in range(n)), dtype=float, count=n)
            if abs(float((array * signs).mean())) >= observed - 1e-15:
                extreme += 1
        return float(extreme / total)
    raise ValueError("Exact sign flip enumeration is limited to 20 paired units")


def holm_adjust(pvalues: List[float]) -> List[float]:
    count = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(pvalues[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> int:
    a = parse_args()
    root = a.project_root.expanduser().resolve()
    manifest_path = a.coalition_manifest.expanduser().resolve()
    warmup_root = a.warmup_root.expanduser().resolve()
    c2_root = a.c2_root.expanduser().resolve()
    c3_root = a.c3_root.expanduser().resolve()
    audit_dir = a.c3_audit_dir.expanduser().resolve()
    output = a.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    tag_commit = git(root, "rev-list", "-n", "1", C3_TAG)
    if len(tag_commit) != 40:
        raise RuntimeError(f"Required C3 tag does not resolve: {C3_TAG}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", C3_TAG, "HEAD"], cwd=root).returncode != 0:
        raise RuntimeError("C3 freeze tag is not an ancestor of HEAD")
    if git(root, "status", "--porcelain"):
        raise RuntimeError("Working tree must be clean before C4")

    audit_decision_path = audit_dir / "task45c3_integrity_decision.json"
    if not audit_decision_path.exists():
        raise FileNotFoundError(audit_decision_path)
    audit = json.loads(audit_decision_path.read_text(encoding="utf-8"))
    if not (
        audit.get("conditions_verified") == 72
        and audit.get("paired_branches_verified") == 144
        and audit.get("exact_poison_pairing_all") is True
        and audit.get("reserved_test_arrays_materialized") is False
        and audit.get("ready_for_c4_summary") is True
    ):
        raise RuntimeError("C3 audit decision does not authorize C4")

    manifest = pd.read_csv(manifest_path).sort_values(["family_id", "coalition_size"])
    if len(manifest) != 18 or manifest["coalition_hash_sha256"].nunique() != 18:
        raise RuntimeError("Expected the frozen 18 condition coalition manifest")

    condition_rows: List[Dict[str, object]] = []
    round_rows: List[pd.DataFrame] = []
    for seed in SEEDS:
        clean = load_rounds(warmup_root / f"seed_{seed}" / "clean_continuation", "continuation_round_metrics.csv")
        clean_rate = float(clean["val_source_to_target_rate"].mean())
        clean_f1 = float(clean["val_macro_f1"].mean())
        for row in manifest.itertuples(index=False):
            family = str(row.family_id)
            size = int(row.coalition_size)
            stage_root = c2_root if seed == 7 else c3_root
            condition_root = stage_root / family / f"size_{size:02d}" / f"seed_{seed}"
            plain = load_rounds(condition_root / "plain_attack", "continuation_round_metrics.csv")
            defense = load_rounds(condition_root / "trusted_reconstruction", "reconstruction_round_metrics.csv")
            plain_rate = float(plain["val_source_to_target_rate"].mean())
            defense_rate = float(defense["val_source_to_target_rate"].mean())
            attack_excess = plain_rate - clean_rate
            residual = defense_rate - clean_rate
            removed = (plain_rate - defense_rate) / attack_excess if attack_excess > 1e-12 else float("nan")
            condition_rows.append({
                "family_id": family,
                "coalition_size": size,
                "seed": seed,
                "selected_clients": row.selected_clients,
                "coalition_hash_sha256": row.coalition_hash_sha256,
                "clean_mean_source_to_target_rate": clean_rate,
                "plain_mean_source_to_target_rate": plain_rate,
                "defended_mean_source_to_target_rate": defense_rate,
                "plain_attack_excess_over_clean": attack_excess,
                "defended_residual_excess_over_clean": residual,
                "absolute_rate_reduction": plain_rate - defense_rate,
                "damage_removed_fraction": removed,
                "rounds_improved_vs_plain": int((defense["val_source_to_target_rate"].to_numpy() < plain["val_source_to_target_rate"].to_numpy()).sum()),
                "mean_malicious_recall": float(defense["malicious_recall"].mean()),
                "minimum_malicious_recall": float(defense["malicious_recall"].min()),
                "mean_benign_fpr": float(defense["benign_false_positive_rate"].mean()),
                "maximum_benign_fpr": float(defense["benign_false_positive_rate"].max()),
                "clean_mean_macro_f1": clean_f1,
                "plain_mean_macro_f1": float(plain["val_macro_f1"].mean()),
                "defended_mean_macro_f1": float(defense["val_macro_f1"].mean()),
                "exact_poison_pair": True,
                "reserved_test_arrays_materialized": False,
            })
            rounds = pd.DataFrame({
                "family_id": family,
                "coalition_size": size,
                "seed": seed,
                "monitoring_round": plain["monitoring_round"].astype(int),
                "clean_source_to_target_rate": clean["val_source_to_target_rate"],
                "plain_source_to_target_rate": plain["val_source_to_target_rate"],
                "defended_source_to_target_rate": defense["val_source_to_target_rate"],
                "malicious_recall": defense["malicious_recall"],
                "benign_fpr": defense["benign_false_positive_rate"],
            })
            rounds["defense_improved_vs_plain"] = rounds["defended_source_to_target_rate"] < rounds["plain_source_to_target_rate"]
            round_rows.append(rounds)

    conditions = pd.DataFrame(condition_rows).sort_values(["family_id", "coalition_size", "seed"])
    rounds = pd.concat(round_rows, ignore_index=True).sort_values(["family_id", "coalition_size", "seed", "monitoring_round"])
    if len(conditions) != 72 or len(rounds) != 288:
        raise RuntimeError(f"Unexpected C4 coverage: {len(conditions)} conditions, {len(rounds)} rounds")
    conditions.to_csv(tables / "task45c4_condition_seed_summary.csv", index=False)
    rounds.to_csv(tables / "task45c4_round_level_summary.csv", index=False)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    size_rows: List[Dict[str, object]] = []
    h2_pvalues: List[float] = []
    for size in SIZES:
        group = conditions[conditions["coalition_size"] == size]
        mean_recall, recall_low, recall_high = mean_ci(group["mean_malicious_recall"], rng)
        mean_removed, removed_low, removed_high = mean_ci(group["damage_removed_fraction"], rng)
        mean_reduction, reduction_low, reduction_high = mean_ci(group["absolute_rate_reduction"], rng)
        pvalue = exact_sign_flip_p(group["absolute_rate_reduction"])
        h2_pvalues.append(pvalue)
        mean_attack_excess = float(group["plain_attack_excess_over_clean"].mean())
        h1_recall = mean_recall >= 0.50
        maximum_fpr = float(group["maximum_benign_fpr"].max())
        h1_fpr = maximum_fpr <= 0.05
        h2_qualified = mean_attack_excess > 0.05
        h2_positive = (not h2_qualified) or mean_removed > 0.0
        size_rows.append({
            "coalition_size": size,
            "family_seed_unit_count": len(group),
            "mean_plain_attack_excess": mean_attack_excess,
            "mean_absolute_rate_reduction": mean_reduction,
            "absolute_rate_reduction_ci95_low": reduction_low,
            "absolute_rate_reduction_ci95_high": reduction_high,
            "mean_damage_removed_fraction": mean_removed,
            "damage_removed_fraction_ci95_low": removed_low,
            "damage_removed_fraction_ci95_high": removed_high,
            "mean_malicious_recall": mean_recall,
            "malicious_recall_ci95_low": recall_low,
            "malicious_recall_ci95_high": recall_high,
            "maximum_benign_fpr": maximum_fpr,
            "between_family_plain_excess_range": float(group.groupby("family_id")["plain_attack_excess_over_clean"].mean().max() - group.groupby("family_id")["plain_attack_excess_over_clean"].mean().min()),
            "between_family_plain_excess_std": float(group.groupby("family_id")["plain_attack_excess_over_clean"].mean().std(ddof=1)),
            "h1_mean_recall_at_least_0p50": h1_recall,
            "h1_maximum_fpr_at_most_0p05": h1_fpr,
            "h1_size_pass": h1_recall and h1_fpr,
            "h2_attack_excess_qualified": h2_qualified,
            "h2_mean_damage_removed_positive": h2_positive,
            "h2_exact_sign_flip_p_raw": pvalue,
        })
    size_table = pd.DataFrame(size_rows)
    size_table["h2_exact_sign_flip_p_holm"] = holm_adjust(h2_pvalues)
    size_table.to_csv(tables / "task45c4_size_aggregate_summary.csv", index=False)

    family_curve = conditions.groupby(["family_id", "coalition_size"], as_index=False).agg(
        mean_plain_attack_excess=("plain_attack_excess_over_clean", "mean"),
        mean_damage_removed_fraction=("damage_removed_fraction", "mean"),
        mean_malicious_recall=("mean_malicious_recall", "mean"),
        maximum_benign_fpr=("maximum_benign_fpr", "max"),
    )
    family_curve.to_csv(tables / "task45c4_family_curve_summary.csv", index=False)
    h3_rows = []
    for family in FAMILIES:
        group = family_curve[family_curve["family_id"] == family].sort_values("coalition_size")
        values = group["mean_plain_attack_excess"].to_numpy(dtype=float)
        differences = np.diff(values)
        monotonic = bool(np.all(differences >= -1e-12))
        h3_rows.append({
            "family_id": family,
            "sizes": "|".join(map(str, group["coalition_size"].astype(int))),
            "mean_plain_attack_excess_values": "|".join(f"{value:.12g}" for value in values),
            "successive_differences": "|".join(f"{value:.12g}" for value in differences),
            "nondecreasing": monotonic,
        })
    h3 = pd.DataFrame(h3_rows)
    h3.to_csv(tables / "task45c4_h3_family_monotonicity.csv", index=False)

    h1_pass = bool(size_table["h1_size_pass"].all())
    h2_pass = bool(size_table.loc[size_table["h2_attack_excess_qualified"], "h2_mean_damage_removed_positive"].all())
    h3_pass_count = int(h3["nondecreasing"].sum())
    h3_pass = h3_pass_count >= 2
    full_decision = "PASS" if h1_pass and h2_pass and h3_pass else "PARTIAL" if any([h1_pass, h2_pass, h3_pass]) else "FAIL"

    decision = {
        "experiment_version": "4.16.C4",
        "stage": "task45_coalition_size_summary_and_hypothesis_evaluation",
        "c3_tag": C3_TAG,
        "c3_commit": tag_commit,
        "seeds": list(SEEDS),
        "coalition_families": list(FAMILIES),
        "coalition_sizes": list(SIZES),
        "conditions_analyzed": 72,
        "paired_branches_analyzed": 144,
        "rounds_analyzed": 288,
        "h1_detection_robustness_pass": h1_pass,
        "h2_mitigation_pass": h2_pass,
        "h3_attack_scale_nondecreasing_family_count": h3_pass_count,
        "h3_attack_scale_required_family_count": 2,
        "h3_attack_scale_pass": h3_pass,
        "overall_preregistered_hypothesis_result": full_decision,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "multiple_comparison_control": "Holm adjustment across six H2 size tests",
        "negative_and_boundary_results_retained": True,
        "reserved_test_arrays_materialized": False,
        "task45_development_and_validation_closeout_complete": True,
        "c5_reserved_test_gate_opened": False,
        "method_reopened": False,
    }
    (output / "task45c4_coalition_size_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    axes[0].plot(size_table["coalition_size"], size_table["mean_plain_attack_excess"], marker="o", label="Plain attack excess")
    axes[0].plot(size_table["coalition_size"], size_table["mean_absolute_rate_reduction"], marker="s", label="D0 absolute reduction")
    axes[0].axhline(0.05, color="#C44E52", linestyle="--", linewidth=1, label="H2 damage threshold")
    axes[0].set_ylabel("Source to target rate")
    axes[0].set_title("Attack damage and mitigation")
    axes[1].errorbar(
        size_table["coalition_size"], size_table["mean_damage_removed_fraction"],
        yerr=[size_table["mean_damage_removed_fraction"] - size_table["damage_removed_fraction_ci95_low"], size_table["damage_removed_fraction_ci95_high"] - size_table["mean_damage_removed_fraction"]],
        marker="o", capsize=3,
    )
    axes[1].axhline(0, color="#C44E52", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Damage removed fraction")
    axes[1].set_title("Frozen D0 mitigation with 95% bootstrap CI")
    for ax in axes:
        ax.set_xlabel("Malicious coalition size")
        ax.set_xticks(SIZES)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "task45c4_attack_and_mitigation_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "task45c4_attack_and_mitigation_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    axes[0].plot(size_table["coalition_size"], size_table["mean_malicious_recall"], marker="o", color="#4C78A8")
    axes[0].fill_between(size_table["coalition_size"], size_table["malicious_recall_ci95_low"], size_table["malicious_recall_ci95_high"], alpha=0.2)
    axes[0].axhline(0.50, color="#C44E52", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Malicious recall")
    axes[0].set_title("Detection recall")
    axes[1].plot(size_table["coalition_size"], size_table["maximum_benign_fpr"], marker="s", color="#F58518")
    axes[1].axhline(0.05, color="#C44E52", linestyle="--", linewidth=1)
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("Maximum benign FPR")
    axes[1].set_title("Worst observed benign false positive rate")
    for ax in axes:
        ax.set_xlabel("Malicious coalition size")
        ax.set_xticks(SIZES)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "task45c4_detection_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "task45c4_detection_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    for family in FAMILIES:
        group = family_curve[family_curve["family_id"] == family].sort_values("coalition_size")
        axes[0].plot(group["coalition_size"], group["mean_plain_attack_excess"], marker="o", label=family)
        axes[1].plot(group["coalition_size"], group["mean_damage_removed_fraction"], marker="o", label=family)
    axes[0].set_title("Identity sensitivity of plain attack excess")
    axes[0].set_ylabel("Mean attack excess")
    axes[1].set_title("Identity sensitivity of mitigation")
    axes[1].set_ylabel("Mean damage removed fraction")
    axes[1].axhline(0, color="#C44E52", linestyle="--", linewidth=1)
    for ax in axes:
        ax.set_xlabel("Malicious coalition size")
        ax.set_xticks(SIZES)
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "task45c4_family_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "task45c4_family_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    report = [
        "# Task 45 Coalition Size Closeout",
        "",
        "Task 45 analyzed 72 frozen family by size by seed conditions, 144 exactly paired experiment branches, and 288 monitored rounds. The reserved natural and diagnostic tests remained closed.",
        "",
        "## Preregistered decisions",
        "",
        f"* H1 detection robustness: {'PASS' if h1_pass else 'FAIL'}.",
        f"* H2 mitigation: {'PASS' if h2_pass else 'FAIL'}.",
        f"* H3 attack scale: {'PASS' if h3_pass else 'FAIL'}, with {h3_pass_count} of 3 nested coalition families nondecreasing.",
        f"* Overall result: {full_decision}.",
        "",
        "All negative and boundary outcomes are retained. No threshold, detector feature, reconstruction rule, coalition identity, attack setting, seed, or method component was changed after observing results.",
        "",
        "C5 remains a separate reserved test gate. This closeout does not authorize test access.",
    ]
    (output / "TASK45_C4_CLOSEOUT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("===== TASK 45 C4 COALITION SIZE DECISION =====")
    print("CONDITIONS ANALYZED: 72")
    print("PAIRED BRANCHES ANALYZED: 144")
    print("ROUNDS ANALYZED: 288")
    print("H1 DETECTION ROBUSTNESS:", "PASS" if h1_pass else "FAIL")
    print("H2 MITIGATION:", "PASS" if h2_pass else "FAIL")
    print(f"H3 ATTACK SCALE: {'PASS' if h3_pass else 'FAIL'} ({h3_pass_count}/3 families nondecreasing)")
    print("OVERALL PREREGISTERED RESULT:", full_decision)
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("TASK 45 DEVELOPMENT AND VALIDATION CLOSEOUT COMPLETE: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

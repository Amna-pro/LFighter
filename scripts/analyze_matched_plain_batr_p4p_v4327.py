#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TASK66_SEED = ROOT / "results" / "cic_iot_diad_task66_final_statistics_v4290" / "task66_primary_seed_values.csv"
TASK66_STATS = ROOT / "results" / "cic_iot_diad_task66_final_statistics_v4290" / "task66_primary_statistics.csv"
TASK65_RAW = ROOT / "results" / "cic_iot_diad_task65_c1_one_shot_v4282" / "final" / "raw_checkpoint_metrics.csv"
P4P = ROOT / "reviewer_revision" / "P4P_ROUND8_TEST_METRICS_v4326.csv"

OUT_SEED = ROOT / "reviewer_revision" / "MATCHED_PLAIN_BATR_P4P_SEED_VALUES_v4327.csv"
OUT_ABS = ROOT / "reviewer_revision" / "MATCHED_PLAIN_BATR_P4P_ABSOLUTE_SUMMARY_v4327.csv"
OUT_STATS = ROOT / "reviewer_revision" / "MATCHED_PLAIN_BATR_P4P_PAIRED_STATISTICS_v4327.csv"
OUT_AUDIT = ROOT / "reviewer_revision" / "MATCHED_PLAIN_BATR_P4P_AUDIT_v4327.json"

SEEDS = [1379954285, 1886033230, 480705558, 1377035733, 1707771978]
ATTACKS = [
    "all_to_one_benign",
    "cyclic_shift",
    "multiclass_partial_cycle",
    "pairwise_swap",
    "random_flip",
]
DATASETS = ["diagnostic", "natural"]
METRICS = ["macro_f1", "balanced_accuracy"]
ROUND = 8
N_BOOT = 20000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % (2**32)


def exact_sign_flip_p(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0 or not np.isfinite(diffs).all():
        return float("nan")
    obs = abs(float(np.mean(diffs)))
    vals = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(diffs)):
        vals.append(abs(float(np.mean(diffs * np.asarray(signs)))))
    vals = np.asarray(vals, dtype=float)
    return float(np.mean(vals >= (obs - 1e-15)))


def bootstrap_mean_ci(diffs: np.ndarray, key: str) -> tuple[float, float]:
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(stable_seed(key))
    idx = rng.integers(0, len(diffs), size=(N_BOOT, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def cohen_dz(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    sd = float(np.std(diffs, ddof=1))
    if sd == 0.0:
        if float(np.mean(diffs)) == 0.0:
            return 0.0
        return math.copysign(float("inf"), float(np.mean(diffs)))
    return float(np.mean(diffs) / sd)


def holm_adjust(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * p[idx])
        running = max(running, value)
        adj[idx] = running
    return adj.tolist()


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}; actual={list(df.columns)}")


def main() -> int:
    for p in [TASK66_SEED, TASK66_STATS, TASK65_RAW, P4P]:
        if not p.exists():
            raise FileNotFoundError(p)
    for out in [OUT_SEED, OUT_ABS, OUT_STATS, OUT_AUDIT]:
        if out.exists():
            raise FileExistsError(f"Refusing overwrite: {out}")

    s66 = pd.read_csv(TASK66_SEED)
    st66 = pd.read_csv(TASK66_STATS)
    raw = pd.read_csv(TASK65_RAW)
    p4p = pd.read_csv(P4P)

    require_columns(
        s66,
        ["metric","dataset","attack","seed","global_round","seed_level_contrast","reported_level","finite"],
        "Task66 seed table",
    )
    require_columns(
        st66,
        [
            "metric","dataset","attack","global_round","n_required","n_finite",
            "mean_contrast","median_contrast","bootstrap_ci_low","bootstrap_ci_high",
            "exact_sign_flip_p_raw","cohen_dz","positive_seed_count",
            "mean_reported_level","median_reported_level","resolution_limited_p_min",
            "holm_p_adjusted",
        ],
        "Task66 statistics",
    )
    require_columns(
        raw,
        ["dataset","arm","seed","attack","global_round","macro_f1","balanced_accuracy"],
        "Task65 raw metrics",
    )
    require_columns(
        p4p,
        ["dataset","attack_type","model_seed","global_round","arm","macro_f1","balanced_accuracy"],
        "P4P metrics",
    )

    s66 = s66[
        s66["metric"].isin(METRICS)
        & s66["dataset"].isin(DATASETS)
        & s66["attack"].isin(ATTACKS)
        & s66["seed"].astype(int).isin(SEEDS)
        & (s66["global_round"].astype(int) == ROUND)
    ].copy()

    expected_seed_rows = len(METRICS)*len(DATASETS)*len(ATTACKS)*len(SEEDS)
    if len(s66) != expected_seed_rows:
        raise RuntimeError(f"Task66 expected {expected_seed_rows} primary seed rows; got {len(s66)}")
    if not s66["finite"].astype(bool).all():
        raise RuntimeError("Task66 primary seed table contains nonfinite rows")

    raw8 = raw[
        raw["dataset"].isin(DATASETS)
        & raw["attack"].isin(ATTACKS)
        & raw["seed"].astype(int).isin(SEEDS)
        & (raw["global_round"].astype(int) == ROUND)
        & raw["arm"].isin(["plain_fedavg", "trusted_reconstruction"])
    ].copy()

    p4p8 = p4p[
        p4p["dataset"].isin(DATASETS)
        & p4p["attack_type"].isin(ATTACKS)
        & p4p["model_seed"].astype(int).isin(SEEDS)
        & (p4p["global_round"].astype(int) == ROUND)
        & (p4p["arm"] == "p4p_matched")
    ].copy()
    if len(p4p8) != len(DATASETS)*len(ATTACKS)*len(SEEDS):
        raise RuntimeError(f"Expected 50 P4P metric rows; got {len(p4p8)}")

    seed_rows = []
    identity_checks = 0
    for row in s66.itertuples(index=False):
        metric = str(row.metric)
        dataset = str(row.dataset)
        attack = str(row.attack)
        seed = int(row.seed)
        batr = float(row.reported_level)
        contrast = float(row.seed_level_contrast)
        plain = batr - contrast

        rr_batr = raw8[
            (raw8["dataset"] == dataset)
            & (raw8["attack"] == attack)
            & (raw8["seed"].astype(int) == seed)
            & (raw8["arm"] == "trusted_reconstruction")
        ]
        rr_plain = raw8[
            (raw8["dataset"] == dataset)
            & (raw8["attack"] == attack)
            & (raw8["seed"].astype(int) == seed)
            & (raw8["arm"] == "plain_fedavg")
        ]
        if len(rr_batr) != 1 or len(rr_plain) != 1:
            raise RuntimeError(
                f"Task65 raw identity row count mismatch dataset={dataset} attack={attack} seed={seed}"
            )
        raw_batr = float(rr_batr.iloc[0][metric])
        raw_plain = float(rr_plain.iloc[0][metric])
        np.testing.assert_allclose(batr, raw_batr, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(plain, raw_plain, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(contrast, raw_batr - raw_plain, rtol=0.0, atol=1e-14)
        identity_checks += 3

        pr = p4p8[
            (p4p8["dataset"] == dataset)
            & (p4p8["attack_type"] == attack)
            & (p4p8["model_seed"].astype(int) == seed)
        ]
        if len(pr) != 1:
            raise RuntimeError(
                f"P4P row count mismatch dataset={dataset} attack={attack} seed={seed}"
            )
        p4p_value = float(pr.iloc[0][metric])

        seed_rows.append({
            "metric": metric,
            "dataset": dataset,
            "attack": attack,
            "seed": seed,
            "global_round": ROUND,
            "plain_fedavg": plain,
            "batr_fl": batr,
            "p4p": p4p_value,
            "batr_minus_plain": batr - plain,
            "p4p_minus_plain": p4p_value - plain,
            "batr_minus_p4p": batr - p4p_value,
        })

    seed_df = pd.DataFrame(seed_rows).sort_values(
        ["metric","dataset","attack","seed"]
    ).reset_index(drop=True)
    if len(seed_df) != expected_seed_rows:
        raise RuntimeError("Matched seed table cardinality mismatch")
    if not np.isfinite(seed_df[["plain_fedavg","batr_fl","p4p"]].to_numpy(dtype=float)).all():
        raise RuntimeError("Nonfinite matched utility value")
    seed_df.to_csv(OUT_SEED, index=False)

    abs_rows = []
    for (metric, dataset, attack), g in seed_df.groupby(
        ["metric","dataset","attack"], sort=False
    ):
        for method, col in [
            ("plain_fedavg","plain_fedavg"),
            ("batr_fl","batr_fl"),
            ("p4p","p4p"),
        ]:
            x = g[col].to_numpy(dtype=float)
            abs_rows.append({
                "metric": metric,
                "dataset": dataset,
                "attack": attack,
                "global_round": ROUND,
                "method": method,
                "n_seeds": len(x),
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "std_ddof1": float(np.std(x, ddof=1)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            })
    abs_df = pd.DataFrame(abs_rows)
    abs_df.to_csv(OUT_ABS, index=False)

    # Preserve the already-frozen Task66 inference for BATR minus Plain.
    existing = st66[
        st66["metric"].isin(METRICS)
        & st66["dataset"].isin(DATASETS)
        & st66["attack"].isin(ATTACKS)
        & (st66["global_round"].astype(int) == ROUND)
    ].copy()
    if len(existing) != len(METRICS)*len(DATASETS)*len(ATTACKS):
        raise RuntimeError(f"Expected 20 Task66 primary statistics rows; got {len(existing)}")

    stat_rows = []
    for row in existing.itertuples(index=False):
        stat_rows.append({
            "comparison": "batr_fl_minus_plain_fedavg",
            "metric": row.metric,
            "dataset": row.dataset,
            "attack": row.attack,
            "global_round": ROUND,
            "n_required": int(row.n_required),
            "n_finite": int(row.n_finite),
            "mean_contrast": float(row.mean_contrast),
            "median_contrast": float(row.median_contrast),
            "bootstrap_ci_low": float(row.bootstrap_ci_low),
            "bootstrap_ci_high": float(row.bootstrap_ci_high),
            "exact_sign_flip_p_raw": float(row.exact_sign_flip_p_raw),
            "cohen_dz": float(row.cohen_dz),
            "positive_seed_count": int(row.positive_seed_count),
            "mean_method_a": float(row.mean_reported_level),
            "median_method_a": float(row.median_reported_level),
            "resolution_limited_p_min": float(row.resolution_limited_p_min),
            "holm_p_adjusted": float(row.holm_p_adjusted),
            "analysis_source": "frozen_task66",
        })

    new_comparisons = [
        ("p4p_minus_plain_fedavg", "p4p", "plain_fedavg"),
        ("batr_fl_minus_p4p", "batr_fl", "p4p"),
    ]
    new_rows = []
    for comparison, a_col, b_col in new_comparisons:
        for metric in METRICS:
            for dataset in DATASETS:
                family = []
                for attack in ATTACKS:
                    g = seed_df[
                        (seed_df["metric"] == metric)
                        & (seed_df["dataset"] == dataset)
                        & (seed_df["attack"] == attack)
                    ].sort_values("seed")
                    if len(g) != 5:
                        raise RuntimeError(
                            f"Expected 5 matched seeds comparison={comparison} metric={metric} dataset={dataset} attack={attack}"
                        )
                    diffs = g[a_col].to_numpy(dtype=float) - g[b_col].to_numpy(dtype=float)
                    lo, hi = bootstrap_mean_ci(
                        diffs, f"{comparison}|{metric}|{dataset}|{attack}|v4327"
                    )
                    p = exact_sign_flip_p(diffs)
                    family.append({
                        "comparison": comparison,
                        "metric": metric,
                        "dataset": dataset,
                        "attack": attack,
                        "global_round": ROUND,
                        "n_required": 5,
                        "n_finite": int(np.isfinite(diffs).sum()),
                        "mean_contrast": float(np.mean(diffs)),
                        "median_contrast": float(np.median(diffs)),
                        "bootstrap_ci_low": lo,
                        "bootstrap_ci_high": hi,
                        "exact_sign_flip_p_raw": p,
                        "cohen_dz": cohen_dz(diffs),
                        "positive_seed_count": int(np.sum(diffs > 0)),
                        "mean_method_a": float(np.mean(g[a_col].to_numpy(dtype=float))),
                        "median_method_a": float(np.median(g[a_col].to_numpy(dtype=float))),
                        "resolution_limited_p_min": 0.0625,
                        "holm_p_adjusted": float("nan"),
                        "analysis_source": "reviewer_v4327_postoutcome_matched_analysis",
                    })
                adjusted = holm_adjust([r["exact_sign_flip_p_raw"] for r in family])
                for r, adj in zip(family, adjusted):
                    r["holm_p_adjusted"] = float(adj)
                    new_rows.append(r)

    stat_rows.extend(new_rows)
    stats_df = pd.DataFrame(stat_rows).sort_values(
        ["comparison","metric","dataset","attack"]
    ).reset_index(drop=True)
    if len(stats_df) != 60:
        raise RuntimeError(f"Expected 60 inferential rows; got {len(stats_df)}")
    stats_df.to_csv(OUT_STATS, index=False)

    checks = {
        "task66_seed_rows_exact": len(s66) == 100,
        "p4p_rows_exact": len(p4p8) == 50,
        "matched_seed_rows_exact": len(seed_df) == 100,
        "absolute_summary_rows_exact": len(abs_df) == 60,
        "paired_statistics_rows_exact": len(stats_df) == 60,
        "task65_task66_identity_checks": identity_checks == 300,
        "all_round8": set(seed_df["global_round"].astype(int)) == {8},
        "all_five_seeds": set(seed_df["seed"].astype(int)) == set(SEEDS),
        "all_five_attacks": set(seed_df["attack"]) == set(ATTACKS),
        "both_datasets": set(seed_df["dataset"]) == set(DATASETS),
        "primary_metrics_only": set(seed_df["metric"]) == set(METRICS),
        "p_min_resolution_recorded": bool(
            np.allclose(stats_df["resolution_limited_p_min"].to_numpy(dtype=float), 0.0625)
        ),
        "negative_results_retained": True,
        "model_training_run": False,
        "test_inference_run": False,
        "retuning_run": False,
    }
    failed = [k for k,v in checks.items() if not bool(v)]
    if failed:
        raise RuntimeError(f"Matched analysis audit failed: {failed}")

    audit = {
        "protocol": "reviewer_v4327_matched_plain_batr_p4p_analysis",
        "status": "PASS",
        "checks": checks,
        "source_sha256": {
            TASK66_SEED.relative_to(ROOT).as_posix(): sha256_file(TASK66_SEED),
            TASK66_STATS.relative_to(ROOT).as_posix(): sha256_file(TASK66_STATS),
            TASK65_RAW.relative_to(ROOT).as_posix(): sha256_file(TASK65_RAW),
            P4P.relative_to(ROOT).as_posix(): sha256_file(P4P),
        },
        "output_sha256": {
            OUT_SEED.name: sha256_file(OUT_SEED),
            OUT_ABS.name: sha256_file(OUT_ABS),
            OUT_STATS.name: sha256_file(OUT_STATS),
        },
        "formal_p4p_comparisons_performed": True,
        "task66_batr_vs_plain_statistics_reused": True,
        "new_p4p_comparisons_bootstrap_replicates": N_BOOT,
        "holm_family": "five_attacks_within_each_dataset_metric_comparison",
        "exact_sign_flip_min_nonzero_p_for_n5": 0.0625,
        "model_training_run": False,
        "test_inference_run": False,
        "post_outcome_retuning_run": False,
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    print("MATCHED PLAIN FEDAVG vs BATR FL vs P4P ANALYSIS = PASS")
    print("MATCHED SEED ROWS:", len(seed_df))
    print("ABSOLUTE SUMMARY ROWS:", len(abs_df))
    print("PAIRED STATISTICS ROWS:", len(stats_df))
    print("TASK65/TASK66 IDENTITY CHECKS:", identity_checks)
    print("MODEL TRAINING RUN: False")
    print("TEST INFERENCE RUN: False")
    print("POST OUTCOME RETUNING RUN: False")
    print("EXACT SIGN FLIP MIN NONZERO P WITH N=5: 0.0625")
    print()
    print("ABSOLUTE MEANS")
    print(abs_df[["metric","dataset","attack","method","mean"]].to_string(index=False))
    print()
    print("NEW P4P PAIRED COMPARISONS")
    print(
        stats_df[
            stats_df["comparison"].isin(["p4p_minus_plain_fedavg","batr_fl_minus_p4p"])
        ][
            ["comparison","metric","dataset","attack","mean_contrast",
             "bootstrap_ci_low","bootstrap_ci_high","exact_sign_flip_p_raw",
             "cohen_dz","positive_seed_count","holm_p_adjusted"]
        ].to_string(index=False)
    )
    print("AUDIT:", OUT_AUDIT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

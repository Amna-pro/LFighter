#!/usr/bin/env python3
"""Offline causal temporal-policy audit for the frozen V3.10 anchor signal.

The signal itself is not reopened. This audit only studies:
- EMA threshold quantile,
- EMA initialization after trusted warmup,
- an instantaneous rescue branch.

Seed 42 is used for policy selection. Seeds 7, 99, 123, and 2026 are held out
until the policy is frozen. No model training and no test-set access occur.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CANDIDATE = "candidate_max_anchor_absolute"
EMA_COLUMN = f"{CANDIDATE}_ema"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--development-warmup-dir", required=True, type=Path)
    p.add_argument("--development-clean-dir", required=True, type=Path)
    p.add_argument("--development-attack-dir", required=True, type=Path)
    p.add_argument("--heldout-root", required=True, type=Path)
    p.add_argument("--heldout-seeds", default="7,99,123,2026")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--development-seed", type=int, default=42)
    p.add_argument("--ema-decay", type=float, default=0.65)
    p.add_argument("--max-clean-mean-fpr", type=float, default=0.05)
    p.add_argument("--max-clean-round-fpr", type=float, default=0.10)
    p.add_argument("--max-attack-benign-mean-fpr", type=float, default=0.05)
    p.add_argument("--max-attack-benign-round-fpr", type=float, default=0.10)
    return p.parse_args()


def qhigher(values: Iterable[float], q: float) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


def load_seed(warmup_dir: Path, clean_dir: Path, attack_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    warmup = pd.read_csv(warmup_dir / "calibration" / "clean_leave_one_round_out_scores.csv")
    clean = pd.read_csv(clean_dir / "tables" / "continuation_client_anchor_scores.csv")
    attack = pd.read_csv(attack_dir / "tables" / "continuation_client_anchor_scores.csv")
    required = {"monitoring_round", "client_id", CANDIDATE, EMA_COLUMN}
    for name, table in (("warmup", warmup), ("clean", clean), ("attack", attack)):
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{name} table missing columns: {sorted(missing)}")
    if "actual_malicious" not in attack.columns:
        raise ValueError("Attack table lacks actual_malicious")
    return warmup, clean, attack


def recompute_ema(table: pd.DataFrame, decay: float, initial: Dict[int, float]) -> pd.Series:
    result = pd.Series(index=table.index, dtype=float)
    ordered = table.sort_values(["client_id", "monitoring_round"])
    for client_id, group in ordered.groupby("client_id", sort=False):
        previous = float(initial.get(int(client_id), 0.0))
        for idx in group.index:
            previous = decay * previous + (1.0 - decay) * float(table.at[idx, CANDIDATE])
            result.at[idx] = previous
    return result


def warmup_final_ema(warmup: pd.DataFrame) -> Dict[int, float]:
    final = (
        warmup.sort_values(["client_id", "monitoring_round"])
        .groupby("client_id", as_index=False)
        .tail(1)
    )
    return {int(row.client_id): float(getattr(row, EMA_COLUMN)) for row in final.itertuples(index=False)}


def score_policy(
    warmup: pd.DataFrame,
    clean: pd.DataFrame,
    attack: pd.DataFrame,
    ema_decay: float,
    ema_quantile: float,
    instant_quantile: float | None,
    ema_initialization: str,
) -> Dict[str, object]:
    ema_threshold = qhigher(warmup[EMA_COLUMN], ema_quantile)
    instant_threshold = (
        qhigher(warmup[CANDIDATE], instant_quantile)
        if instant_quantile is not None
        else float("nan")
    )
    initial = warmup_final_ema(warmup) if ema_initialization == "carryover" else {}

    clean_eval = clean.copy()
    attack_eval = attack.copy()
    clean_eval["policy_ema"] = recompute_ema(clean_eval, ema_decay, initial)
    attack_eval["policy_ema"] = recompute_ema(attack_eval, ema_decay, initial)

    for table in (clean_eval, attack_eval):
        ema_flag = table["policy_ema"].to_numpy(float) > ema_threshold
        if instant_quantile is None:
            flag = ema_flag
        else:
            instant_flag = table[CANDIDATE].to_numpy(float) > instant_threshold
            flag = ema_flag | instant_flag
        table["policy_flagged"] = flag

    clean_round = clean_eval.groupby("monitoring_round")["policy_flagged"].mean()
    labels = attack_eval["actual_malicious"].astype(bool)
    attack_benign = attack_eval.loc[~labels].groupby("monitoring_round")["policy_flagged"].mean()
    attack_malicious = attack_eval.loc[labels].groupby("monitoring_round")["policy_flagged"].mean()

    predicted = attack_eval["policy_flagged"].astype(bool)
    precision = float((predicted & labels).sum() / max(predicted.sum(), 1))

    return {
        "ema_initialization": ema_initialization,
        "ema_quantile": float(ema_quantile),
        "instant_quantile": "disabled" if instant_quantile is None else float(instant_quantile),
        "ema_threshold": float(ema_threshold),
        "instant_threshold": float(instant_threshold),
        "clean_mean_fpr": float(clean_round.mean()),
        "clean_max_round_fpr": float(clean_round.max()),
        "clean_final_round_fpr": float(clean_round.iloc[-1]),
        "attack_benign_mean_fpr": float(attack_benign.mean()),
        "attack_benign_max_round_fpr": float(attack_benign.max()),
        "attack_benign_final_round_fpr": float(attack_benign.iloc[-1]),
        "attack_mean_malicious_recall": float(attack_malicious.mean()),
        "attack_minimum_round_malicious_recall": float(attack_malicious.min()),
        "attack_first_round_malicious_recall": float(attack_malicious.iloc[0]),
        "attack_final_round_malicious_recall": float(attack_malicious.iloc[-1]),
        "attack_detection_precision": precision,
    }


def policy_grid() -> List[Tuple[str, float, float | None]]:
    rows = []
    for initialization in ("reset", "carryover"):
        for ema_q in (0.95, 0.975, 0.99):
            for instant_q in (None, 0.95, 0.975, 0.99):
                rows.append((initialization, ema_q, instant_q))
    return rows


def add_safety(table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    result = table.copy()
    result["passes_clean_mean_fpr"] = result["clean_mean_fpr"] <= args.max_clean_mean_fpr
    result["passes_clean_round_fpr"] = result["clean_max_round_fpr"] <= args.max_clean_round_fpr
    result["passes_attack_benign_mean_fpr"] = result["attack_benign_mean_fpr"] <= args.max_attack_benign_mean_fpr
    result["passes_attack_benign_round_fpr"] = result["attack_benign_max_round_fpr"] <= args.max_attack_benign_round_fpr
    result["passes_all_safety_constraints"] = result[
        [
            "passes_clean_mean_fpr",
            "passes_clean_round_fpr",
            "passes_attack_benign_mean_fpr",
            "passes_attack_benign_round_fpr",
        ]
    ].all(axis=1)
    return result


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    dev_warmup, dev_clean, dev_attack = load_seed(
        args.development_warmup_dir,
        args.development_clean_dir,
        args.development_attack_dir,
    )

    development_rows = []
    for initialization, ema_q, instant_q in policy_grid():
        development_rows.append(
            score_policy(
                dev_warmup,
                dev_clean,
                dev_attack,
                args.ema_decay,
                ema_q,
                instant_q,
                initialization,
            )
        )
    development = add_safety(pd.DataFrame(development_rows), args)
    eligible = development[development["passes_all_safety_constraints"]].copy()
    if eligible.empty:
        development.to_csv(tables / "development_policy_ranking.csv", index=False)
        raise RuntimeError("No temporal rescue policy satisfies the frozen safety constraints")

    eligible = eligible.sort_values(
        [
            "attack_first_round_malicious_recall",
            "attack_minimum_round_malicious_recall",
            "attack_mean_malicious_recall",
            "attack_detection_precision",
            "clean_mean_fpr",
        ],
        ascending=[False, False, False, False, True],
    )
    selected = eligible.iloc[0]
    development = development.sort_values(
        [
            "passes_all_safety_constraints",
            "attack_first_round_malicious_recall",
            "attack_minimum_round_malicious_recall",
            "attack_mean_malicious_recall",
        ],
        ascending=[False, False, False, False],
    )
    development.to_csv(tables / "development_policy_ranking.csv", index=False)
    eligible.to_csv(tables / "development_safety_eligible_policies.csv", index=False)

    heldout_rows = []
    heldout_seeds = [int(x.strip()) for x in args.heldout_seeds.split(",") if x.strip()]
    instant_q = None if selected["instant_quantile"] == "disabled" else float(selected["instant_quantile"])
    for seed in heldout_seeds:
        seed_root = args.heldout_root / f"seed_{seed}"
        warmup, clean, attack = load_seed(
            seed_root / "warmup",
            seed_root / "clean_continuation",
            seed_root / "attack_continuation",
        )
        row = score_policy(
            warmup,
            clean,
            attack,
            args.ema_decay,
            float(selected["ema_quantile"]),
            instant_q,
            str(selected["ema_initialization"]),
        )
        row["seed"] = seed
        heldout_rows.append(row)

    heldout = pd.DataFrame(heldout_rows)
    heldout.to_csv(tables / "selected_policy_heldout_metrics.csv", index=False)

    aggregate = pd.DataFrame(
        [
            {
                "selected_ema_initialization": selected["ema_initialization"],
                "selected_ema_quantile": selected["ema_quantile"],
                "selected_instant_quantile": selected["instant_quantile"],
                "development_seed": args.development_seed,
                "heldout_seed_count": len(heldout),
                "heldout_mean_clean_fpr": heldout["clean_mean_fpr"].mean(),
                "heldout_maximum_clean_round_fpr": heldout["clean_max_round_fpr"].max(),
                "heldout_mean_attack_benign_fpr": heldout["attack_benign_mean_fpr"].mean(),
                "heldout_maximum_attack_benign_round_fpr": heldout["attack_benign_max_round_fpr"].max(),
                "heldout_mean_malicious_recall": heldout["attack_mean_malicious_recall"].mean(),
                "heldout_minimum_round_malicious_recall": heldout["attack_minimum_round_malicious_recall"].min(),
                "heldout_mean_first_round_malicious_recall": heldout["attack_first_round_malicious_recall"].mean(),
                "heldout_minimum_first_round_malicious_recall": heldout["attack_first_round_malicious_recall"].min(),
                "heldout_minimum_final_round_malicious_recall": heldout["attack_final_round_malicious_recall"].min(),
                "selection_frozen_before_heldout_review": True,
                "candidate_selection_reopened": False,
                "defense_weights_applied": False,
                "test_sets_accessed": False,
            }
        ]
    )
    aggregate.to_csv(tables / "selected_policy_aggregate_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(heldout))
    width = 0.36
    ax.bar(x - width / 2, heldout["attack_first_round_malicious_recall"], width, label="First-round recall")
    ax.bar(x + width / 2, heldout["attack_mean_malicious_recall"], width, label="Mean recall")
    ax.set_xticks(x, heldout["seed"].astype(str))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Held-out development seed")
    ax.set_ylabel("Recall")
    ax.set_title("V3.10.2 frozen temporal rescue policy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, figures / "heldout_recall")

    with (output / "temporal_rescue_audit_v3102_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_version": "3.10.2",
                "frozen_signal": CANDIDATE,
                "development_seed": args.development_seed,
                "heldout_seeds": heldout_seeds,
                "policy_grid_frozen": True,
                "selection_frozen_before_heldout_review": True,
                "candidate_selection_reopened": False,
                "defense_weights_applied": False,
                "test_sets_accessed": False,
            },
            handle,
            indent=2,
        )

    print("Temporal Rescue Audit V3.10.2 complete")
    print("Selected EMA initialization:", selected["ema_initialization"])
    print("Selected EMA quantile:", selected["ema_quantile"])
    print("Selected instant quantile:", selected["instant_quantile"])
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

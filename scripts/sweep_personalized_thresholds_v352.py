#!/usr/bin/env python3
"""Clean-constrained temporal threshold sweep for V3.5.2 development.

This is an offline diagnostic over already-computed personalized residual scores.
It does not retrain the federated model.

Threshold values are always derived from clean-run score quantiles. Attack labels
are used only after calibration to measure diagnostic separation. Any candidate
identified here must later be frozen and validated on held-out seeds and attack
conditions before it can support a paper claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_QUANTILES = [0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995, 1.0]
DEFAULT_EMA_DECAYS = [0.40, 0.55, 0.65, 0.80]
DEFAULT_STRIKES = [1, 2, 3]
DEFAULT_WARMUPS = [1, 2]


def parse_csv_floats(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep clean-calibrated temporal thresholds for personalized residual scores."
    )
    parser.add_argument("--clean-scores", type=Path, required=True)
    parser.add_argument("--attack-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--quantiles",
        default=",".join(str(x) for x in DEFAULT_QUANTILES),
        help="Comma-separated clean quantiles for instantaneous and EMA thresholds.",
    )
    parser.add_argument(
        "--ema-decays",
        default=",".join(str(x) for x in DEFAULT_EMA_DECAYS),
        help="Comma-separated EMA decay values.",
    )
    parser.add_argument(
        "--required-strikes",
        default=",".join(str(x) for x in DEFAULT_STRIKES),
        help="Comma-separated strike requirements.",
    )
    parser.add_argument(
        "--warmup-rounds",
        default=",".join(str(x) for x in DEFAULT_WARMUPS),
        help="Comma-separated warmup round counts.",
    )
    return parser.parse_args()


def validate_table(name: str, table: pd.DataFrame) -> None:
    required = {
        "round",
        "client_id",
        "actual_malicious",
        "personalized_instantaneous_score",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{name} table missing required columns: {missing}")
    if table.duplicated(["round", "client_id"]).any():
        raise ValueError(f"{name} table has duplicate round/client rows")


def quantile_higher(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    try:
        return float(np.quantile(arr, q, method="higher"))
    except TypeError:
        return float(np.quantile(arr, q, interpolation="higher"))


def compute_ema(table: pd.DataFrame, decay: float) -> pd.DataFrame:
    result = table.copy().sort_values(["round", "client_id"]).reset_index(drop=True)
    memory: Dict[int, float] = {}
    ema = []
    for _, row in result.iterrows():
        client_id = int(row["client_id"])
        score = float(row["personalized_instantaneous_score"])
        previous = float(memory.get(client_id, 0.0))
        current = float(decay) * previous + (1.0 - float(decay)) * score
        memory[client_id] = current
        ema.append(current)
    result["sweep_ema"] = ema
    return result


def apply_rule(
    table_with_ema: pd.DataFrame,
    instantaneous_threshold: float,
    ema_threshold: float,
    required_strikes: int,
    warmup_rounds: int,
) -> pd.DataFrame:
    result = table_with_ema.copy().sort_values(["round", "client_id"]).reset_index(drop=True)
    strike_memory: Dict[int, int] = {}
    strikes = []
    rejected = []

    for _, row in result.iterrows():
        client_id = int(row["client_id"])
        round_id = int(row["round"])
        score = float(row["personalized_instantaneous_score"])
        prior = int(strike_memory.get(client_id, 0))

        if round_id > int(warmup_rounds) and score >= float(instantaneous_threshold):
            current = prior + 1
        else:
            current = max(prior - 1, 0)
        strike_memory[client_id] = current

        decision = (
            round_id > int(warmup_rounds)
            and current >= int(required_strikes)
            and float(row["sweep_ema"]) >= float(ema_threshold)
        )
        strikes.append(current)
        rejected.append(bool(decision))

    result["sweep_strikes"] = strikes
    result["sweep_rejected"] = rejected
    return result


def summarize_run(table: pd.DataFrame, clean_mode: bool) -> Dict[str, float]:
    rounds = sorted(int(x) for x in table["round"].unique())
    last_round = max(rounds)
    final = table[table["round"].astype(int).eq(last_round)].copy()

    rejected_clients = set(
        table.loc[table["sweep_rejected"].astype(bool), "client_id"].astype(int)
    )
    rejected_rows = int(table["sweep_rejected"].astype(bool).sum())

    if clean_mode:
        total_clients = int(table["client_id"].nunique())
        return {
            "clean_rejected_rows": rejected_rows,
            "clean_rejected_clients": int(len(rejected_clients)),
            "clean_client_false_rejection_rate": float(len(rejected_clients) / max(total_clients, 1)),
            "clean_final_benign_retention": float(
                1.0 - final["sweep_rejected"].astype(bool).mean()
            ),
            "clean_min_round_benign_retention": float(
                table.groupby("round")["sweep_rejected"]
                .apply(lambda s: 1.0 - s.astype(bool).mean())
                .min()
            ),
        }

    malicious = table["actual_malicious"].astype(bool)
    benign = ~malicious
    malicious_clients = set(table.loc[malicious, "client_id"].astype(int))
    benign_clients = set(table.loc[benign, "client_id"].astype(int))
    malicious_rejected_ever = rejected_clients.intersection(malicious_clients)
    benign_rejected_ever = rejected_clients.intersection(benign_clients)

    final_malicious = final["actual_malicious"].astype(bool)
    final_benign = ~final_malicious
    final_rejected = final["sweep_rejected"].astype(bool)
    final_tp = int((final_malicious & final_rejected).sum())
    final_fp = int((final_benign & final_rejected).sum())
    final_fn = int((final_malicious & ~final_rejected).sum())
    final_tn = int((final_benign & ~final_rejected).sum())

    ever_tp = len(malicious_rejected_ever)
    ever_fp = len(benign_rejected_ever)
    ever_fn = len(malicious_clients) - ever_tp
    ever_tn = len(benign_clients) - ever_fp

    first_detection = (
        table[
            table["actual_malicious"].astype(bool)
            & table["sweep_rejected"].astype(bool)
        ]
        .groupby("client_id")["round"]
        .min()
    )
    median_detection = (
        float(first_detection.median()) if len(first_detection) else math.nan
    )

    ever_precision = ever_tp / max(ever_tp + ever_fp, 1)
    ever_recall = ever_tp / max(ever_tp + ever_fn, 1)
    ever_f1 = 2 * ever_precision * ever_recall / max(ever_precision + ever_recall, 1e-12)

    final_precision = final_tp / max(final_tp + final_fp, 1)
    final_recall = final_tp / max(final_tp + final_fn, 1)
    final_f1 = 2 * final_precision * final_recall / max(final_precision + final_recall, 1e-12)

    return {
        "attack_malicious_clients": int(len(malicious_clients)),
        "attack_benign_clients": int(len(benign_clients)),
        "attack_ever_malicious_rejected": int(ever_tp),
        "attack_ever_benign_rejected": int(ever_fp),
        "attack_ever_precision": float(ever_precision),
        "attack_ever_malicious_recall": float(ever_recall),
        "attack_ever_detection_f1": float(ever_f1),
        "attack_ever_benign_retention": float(ever_tn / max(ever_tn + ever_fp, 1)),
        "attack_final_malicious_rejected": int(final_tp),
        "attack_final_benign_rejected": int(final_fp),
        "attack_final_precision": float(final_precision),
        "attack_final_malicious_recall": float(final_recall),
        "attack_final_detection_f1": float(final_f1),
        "attack_final_benign_retention": float(final_tn / max(final_tn + final_fp, 1)),
        "median_first_malicious_detection_round": median_detection,
    }


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(all_results: pd.DataFrame, base: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        all_results["clean_client_false_rejection_rate"],
        all_results["attack_ever_malicious_recall"],
        alpha=0.45,
    )
    ax.set_xlabel("Clean client false rejection rate")
    ax.set_ylabel("Attack malicious-client recall, ever rejected")
    ax.set_title("Clean false rejection versus attack detection tradeoff")
    ax.grid(alpha=0.25)
    save_figure(fig, base)


def plot_strict_top(strict: pd.DataFrame, base: Path) -> None:
    top = strict.head(20).copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    if top.empty:
        ax.text(0.5, 0.5, "No zero-clean-false-positive candidates", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [
            f"qI={r.instant_quantile:g}, qE={r.ema_quantile:g}, d={r.ema_decay:g}, s={int(r.required_strikes)}, w={int(r.warmup_rounds)}"
            for r in top.itertuples()
        ]
        x = np.arange(len(top))
        ax.bar(x, top["attack_ever_malicious_recall"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=70, ha="right")
        ax.set_ylabel("Attack malicious-client recall, ever rejected")
        ax.set_title("Top zero-clean-false-positive diagnostic configurations")
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, base)


def plot_final_vs_ever(all_results: pd.DataFrame, base: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        all_results["attack_ever_malicious_recall"],
        all_results["attack_final_malicious_recall"],
        alpha=0.45,
    )
    ax.set_xlabel("Ever-rejected malicious-client recall")
    ax.set_ylabel("Final-round malicious-client recall")
    ax.set_title("Persistent versus transient malicious-client detection")
    ax.grid(alpha=0.25)
    save_figure(fig, base)


def main() -> int:
    args = parse_args()
    clean_path = args.clean_scores.expanduser().resolve()
    attack_path = args.attack_scores.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(clean_path)
    attack = pd.read_csv(attack_path)
    validate_table("clean", clean)
    validate_table("attack", attack)

    quantiles = sorted(set(parse_csv_floats(args.quantiles)))
    ema_decays = sorted(set(parse_csv_floats(args.ema_decays)))
    strike_values = sorted(set(parse_csv_ints(args.required_strikes)))
    warmups = sorted(set(parse_csv_ints(args.warmup_rounds)))

    rows = []
    cached = {}
    for decay in ema_decays:
        clean_ema = compute_ema(clean, decay)
        attack_ema = compute_ema(attack, decay)
        cached[decay] = (clean_ema, attack_ema)

        for instant_q in quantiles:
            instant_threshold = quantile_higher(
                clean["personalized_instantaneous_score"], instant_q
            )
            for ema_q in quantiles:
                ema_threshold = quantile_higher(clean_ema["sweep_ema"], ema_q)
                for strikes in strike_values:
                    for warmup in warmups:
                        clean_decisions = apply_rule(
                            clean_ema,
                            instant_threshold,
                            ema_threshold,
                            strikes,
                            warmup,
                        )
                        attack_decisions = apply_rule(
                            attack_ema,
                            instant_threshold,
                            ema_threshold,
                            strikes,
                            warmup,
                        )
                        clean_summary = summarize_run(clean_decisions, clean_mode=True)
                        attack_summary = summarize_run(attack_decisions, clean_mode=False)
                        rows.append(
                            {
                                "instant_quantile": instant_q,
                                "ema_quantile": ema_q,
                                "ema_decay": decay,
                                "required_strikes": strikes,
                                "warmup_rounds": warmup,
                                "instantaneous_threshold": instant_threshold,
                                "ema_threshold": ema_threshold,
                                **clean_summary,
                                **attack_summary,
                            }
                        )

    results = pd.DataFrame(rows)
    results["strict_zero_clean_fp"] = (
        results["clean_rejected_clients"].eq(0)
        & results["clean_rejected_rows"].eq(0)
    )
    results["near_zero_clean_fp"] = (
        results["clean_rejected_clients"].le(1)
        & results["clean_rejected_rows"].le(3)
    )

    sort_columns = [
        "attack_ever_malicious_recall",
        "attack_final_malicious_recall",
        "attack_ever_precision",
        "attack_final_precision",
        "attack_ever_benign_retention",
        "median_first_malicious_detection_round",
    ]
    ascending = [False, False, False, False, False, True]

    strict = (
        results[results["strict_zero_clean_fp"]]
        .sort_values(sort_columns, ascending=ascending, na_position="last")
        .reset_index(drop=True)
    )
    near_zero = (
        results[results["near_zero_clean_fp"]]
        .sort_values(sort_columns, ascending=ascending, na_position="last")
        .reset_index(drop=True)
    )
    all_ranked = results.sort_values(
        [
            "clean_rejected_clients",
            "clean_rejected_rows",
            *sort_columns,
        ],
        ascending=[True, True, *ascending],
        na_position="last",
    ).reset_index(drop=True)

    results.to_csv(tables_dir / "all_threshold_sweep_results.csv", index=False)
    strict.to_csv(tables_dir / "strict_zero_clean_fp_ranked.csv", index=False)
    near_zero.to_csv(tables_dir / "near_zero_clean_fp_ranked.csv", index=False)
    all_ranked.to_csv(tables_dir / "all_candidates_ranked.csv", index=False)

    selected = strict.iloc[0] if not strict.empty else near_zero.iloc[0]
    selected_dict = {
        key: (
            None
            if pd.isna(value)
            else value.item()
            if hasattr(value, "item")
            else value
        )
        for key, value in selected.to_dict().items()
    }
    selected_dict["selection_status"] = (
        "exploratory_development_candidate_not_final"
    )
    selected_dict["selection_warning"] = (
        "Attack labels were used to rank clean-feasible candidates. Freeze the rule "
        "and validate on held-out seeds and attack conditions before any paper claim."
    )
    with (output_dir / "exploratory_candidate_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(selected_dict, handle, indent=2)

    decay = float(selected["ema_decay"])
    clean_ema, attack_ema = cached[decay]
    clean_selected = apply_rule(
        clean_ema,
        float(selected["instantaneous_threshold"]),
        float(selected["ema_threshold"]),
        int(selected["required_strikes"]),
        int(selected["warmup_rounds"]),
    )
    attack_selected = apply_rule(
        attack_ema,
        float(selected["instantaneous_threshold"]),
        float(selected["ema_threshold"]),
        int(selected["required_strikes"]),
        int(selected["warmup_rounds"]),
    )
    clean_selected.to_csv(tables_dir / "selected_candidate_clean_decisions.csv", index=False)
    attack_selected.to_csv(tables_dir / "selected_candidate_attack_decisions.csv", index=False)

    plot_tradeoff(results, figures_dir / "clean_fp_vs_attack_recall")
    plot_strict_top(strict, figures_dir / "strict_zero_fp_top_candidates")
    plot_final_vs_ever(results, figures_dir / "final_vs_ever_detection")

    metadata = {
        "experiment": "personalized_threshold_sweep_v352",
        "status": "offline_development_diagnostic",
        "clean_scores": str(clean_path),
        "attack_scores": str(attack_path),
        "quantiles": quantiles,
        "ema_decays": ema_decays,
        "required_strikes": strike_values,
        "warmup_rounds": warmups,
        "candidate_count": int(len(results)),
        "strict_zero_clean_fp_candidate_count": int(len(strict)),
        "near_zero_clean_fp_candidate_count": int(len(near_zero)),
        "threshold_source": "clean_scores_only",
        "attack_labels_used_for_threshold_calibration": False,
        "attack_labels_used_for_exploratory_ranking": True,
        "final_claim_allowed": False,
    }
    with (output_dir / "threshold_sweep_v352_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print("Personalized Threshold Sweep V3.5.2 complete")
    print("Candidates evaluated:", len(results))
    print("Strict zero-clean-FP candidates:", len(strict))
    print("Near-zero-clean-FP candidates:", len(near_zero))
    print("")
    print("Exploratory best clean-feasible candidate")
    display_columns = [
        "instant_quantile",
        "ema_quantile",
        "ema_decay",
        "required_strikes",
        "warmup_rounds",
        "instantaneous_threshold",
        "ema_threshold",
        "clean_rejected_clients",
        "clean_rejected_rows",
        "attack_ever_malicious_recall",
        "attack_final_malicious_recall",
        "attack_ever_precision",
        "attack_final_precision",
        "attack_ever_benign_retention",
        "attack_final_benign_retention",
        "median_first_malicious_detection_round",
    ]
    print(selected[display_columns].to_string())
    print("")
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("This is an exploratory diagnostic, not a final paper result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

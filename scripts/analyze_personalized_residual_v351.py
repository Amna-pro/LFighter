#!/usr/bin/env python3
"""Offline personalized residual diagnostic for V3.5 Stage 1 outputs.

Purpose
-------
Test, without retraining, whether bidirectional deviations from each client's
trusted clean behavior can separate targeted label-flip attackers from benign
non-IID client heterogeneity.

The detector threshold is calibrated only from clean-run rows. Malicious labels
are used only after calibration for diagnostic evaluation, never for threshold
selection.
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
from sklearn.metrics import roc_auc_score

SIGNALS = [
    "update_l2",
    "update_direction_distance",
    "max_offdiag_probability_drift",
    "explanation_cosine_distance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate personalized bidirectional residual signals from completed V3.5 clean and attack runs."
    )
    parser.add_argument("--clean-table", required=True, type=Path)
    parser.add_argument("--attack-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ema-decay", type=float, default=0.65)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--required-strikes", type=int, default=2)
    parser.add_argument(
        "--clean-quantile",
        type=float,
        default=0.995,
        help="Clean-only quantile used for instantaneous and EMA thresholds.",
    )
    parser.add_argument(
        "--pooled-scale-floor",
        type=float,
        default=0.25,
        help="Minimum client scale as a fraction of pooled within-client clean scale.",
    )
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 1.0
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        std = float(np.std(values))
        scale = std if np.isfinite(std) and std >= 1e-12 else 1.0
    return float(scale)


def transform_signal(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.maximum(array, 0.0)
    return np.log1p(array)


def validate_tables(clean: pd.DataFrame, attack: pd.DataFrame) -> None:
    required = {
        "round",
        "client_id",
        "actual_malicious",
        *SIGNALS,
    }
    for name, table in (("clean", clean), ("attack", attack)):
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(f"{name} table is missing columns: {missing}")
        if table.duplicated(["round", "client_id"]).any():
            raise ValueError(f"{name} table contains duplicate round/client rows")
    clean_clients = set(pd.to_numeric(clean["client_id"]).astype(int))
    attack_clients = set(pd.to_numeric(attack["client_id"]).astype(int))
    if clean_clients != attack_clients:
        raise ValueError("Clean and attack tables must contain the same clients")


def build_profiles(
    clean: pd.DataFrame,
    pooled_scale_floor: float,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, str], Tuple[float, float]], Dict[str, float]]:
    transformed = clean.copy()
    for signal in SIGNALS:
        transformed[f"t_{signal}"] = transform_signal(transformed[signal])

    client_centers: Dict[Tuple[int, str], float] = {}
    pooled_residuals: Dict[str, List[float]] = {signal: [] for signal in SIGNALS}
    for client_id, group in transformed.groupby("client_id", sort=True):
        for signal in SIGNALS:
            values = group[f"t_{signal}"].to_numpy(dtype=float)
            center = float(np.median(values))
            client_centers[(int(client_id), signal)] = center
            pooled_residuals[signal].extend((values - center).tolist())

    pooled_scales = {
        signal: robust_scale(np.asarray(residuals, dtype=float))
        for signal, residuals in pooled_residuals.items()
    }

    profile_rows = []
    profiles: Dict[Tuple[int, str], Tuple[float, float]] = {}
    for client_id, group in transformed.groupby("client_id", sort=True):
        for signal in SIGNALS:
            values = group[f"t_{signal}"].to_numpy(dtype=float)
            center = float(np.median(values))
            client_scale = robust_scale(values)
            scale_floor = max(float(pooled_scale_floor) * pooled_scales[signal], 1e-6)
            scale = max(client_scale, scale_floor)
            profiles[(int(client_id), signal)] = (center, scale)
            profile_rows.append(
                {
                    "client_id": int(client_id),
                    "signal": signal,
                    "clean_rows": int(len(values)),
                    "clean_transformed_median": center,
                    "clean_transformed_scale": scale,
                    "raw_client_scale": client_scale,
                    "pooled_scale": pooled_scales[signal],
                    "applied_scale_floor": scale_floor,
                }
            )
    return pd.DataFrame(profile_rows), profiles, pooled_scales


def top2_mean(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("At least two signal columns are required")
    partitioned = np.partition(matrix, kth=matrix.shape[1] - 2, axis=1)
    return partitioned[:, -2:].mean(axis=1)


def score_with_profiles(
    table: pd.DataFrame,
    profiles: Dict[Tuple[int, str], Tuple[float, float]],
) -> pd.DataFrame:
    result = table.copy().sort_values(["round", "client_id"]).reset_index(drop=True)
    component_columns = []
    for signal in SIGNALS:
        transformed = transform_signal(result[signal])
        scores = np.zeros(len(result), dtype=np.float64)
        for index, (client_id, value) in enumerate(
            zip(result["client_id"].astype(int), transformed)
        ):
            center, scale = profiles[(int(client_id), signal)]
            scores[index] = abs(float(value) - center) / max(scale, 1e-12)
        column = f"personalized_abs_z_{signal}"
        result[column] = np.clip(scores, 0.0, 20.0)
        component_columns.append(column)
    result["personalized_instantaneous_score"] = top2_mean(
        result[component_columns].to_numpy(dtype=float)
    )
    result["dominant_personalized_signal"] = (
        result[component_columns]
        .idxmax(axis=1)
        .str.replace("personalized_abs_z_", "", regex=False)
    )
    return result


def score_clean_leave_one_out(
    clean: pd.DataFrame,
    pooled_scales: Dict[str, float],
    pooled_scale_floor: float,
) -> pd.DataFrame:
    clean = clean.copy().sort_values(["client_id", "round"]).reset_index(drop=True)
    component_values = {signal: np.zeros(len(clean), dtype=np.float64) for signal in SIGNALS}
    transformed = {signal: transform_signal(clean[signal]) for signal in SIGNALS}

    for row_index, row in clean.iterrows():
        client_id = int(row["client_id"])
        client_mask = clean["client_id"].astype(int).eq(client_id).to_numpy()
        client_indices = np.where(client_mask)[0]
        other_indices = client_indices[client_indices != row_index]
        if len(other_indices) < 2:
            raise ValueError("At least three clean rounds per client are required")
        for signal in SIGNALS:
            values = transformed[signal][other_indices]
            center = float(np.median(values))
            client_scale = robust_scale(values)
            floor = max(float(pooled_scale_floor) * pooled_scales[signal], 1e-6)
            scale = max(client_scale, floor)
            component_values[signal][row_index] = abs(
                float(transformed[signal][row_index]) - center
            ) / max(scale, 1e-12)

    component_columns = []
    for signal in SIGNALS:
        column = f"personalized_abs_z_{signal}"
        clean[column] = np.clip(component_values[signal], 0.0, 20.0)
        component_columns.append(column)
    clean["personalized_instantaneous_score"] = top2_mean(
        clean[component_columns].to_numpy(dtype=float)
    )
    clean["dominant_personalized_signal"] = (
        clean[component_columns]
        .idxmax(axis=1)
        .str.replace("personalized_abs_z_", "", regex=False)
    )
    return clean


def add_temporal_decisions(
    scored: pd.DataFrame,
    instant_threshold: float,
    ema_threshold: float,
    ema_decay: float,
    warmup_rounds: int,
    required_strikes: int,
) -> pd.DataFrame:
    result = scored.copy().sort_values(["round", "client_id"]).reset_index(drop=True)
    ema_memory: Dict[int, float] = {}
    strike_memory: Dict[int, int] = {}
    ema_values, strikes, rejected = [], [], []

    for _, row in result.iterrows():
        client_id = int(row["client_id"])
        round_id = int(row["round"])
        current = float(row["personalized_instantaneous_score"])
        previous = float(ema_memory.get(client_id, 0.0))
        ema = float(ema_decay) * previous + (1.0 - float(ema_decay)) * current
        ema_memory[client_id] = ema

        prior_strikes = int(strike_memory.get(client_id, 0))
        if round_id > int(warmup_rounds) and current >= float(instant_threshold):
            current_strikes = prior_strikes + 1
        else:
            current_strikes = max(prior_strikes - 1, 0)
        strike_memory[client_id] = current_strikes

        is_rejected = (
            round_id > int(warmup_rounds)
            and current_strikes >= int(required_strikes)
            and ema >= float(ema_threshold)
        )
        ema_values.append(ema)
        strikes.append(current_strikes)
        rejected.append(bool(is_rejected))

    result["personalized_score_ema"] = ema_values
    result["personalized_strike_count"] = strikes
    result["personalized_rejected"] = rejected
    return result


def clean_ema_reference(
    clean_scored: pd.DataFrame,
    ema_decay: float,
) -> np.ndarray:
    ordered = clean_scored.copy().sort_values(["round", "client_id"]).reset_index(drop=True)
    memory: Dict[int, float] = {}
    values = []
    for _, row in ordered.iterrows():
        client_id = int(row["client_id"])
        current = float(row["personalized_instantaneous_score"])
        previous = float(memory.get(client_id, 0.0))
        ema = float(ema_decay) * previous + (1.0 - float(ema_decay)) * current
        memory[client_id] = ema
        values.append(ema)
    return np.asarray(values, dtype=float)


def quantile_higher(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if not 0.0 < q <= 1.0:
        raise ValueError("clean quantile must be in (0, 1]")
    try:
        return float(np.quantile(array, q, method="higher"))
    except TypeError:
        return float(np.quantile(array, q, interpolation="higher"))


def security_by_round(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for round_id, group in table.groupby("round", sort=True):
        malicious = group["actual_malicious"].astype(bool).to_numpy()
        rejected = group["personalized_rejected"].astype(bool).to_numpy()
        benign = ~malicious
        tp = int(np.sum(malicious & rejected))
        fn = int(np.sum(malicious & ~rejected))
        fp = int(np.sum(benign & rejected))
        tn = int(np.sum(benign & ~rejected))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "round": int(round_id),
                "malicious_clients": int(malicious.sum()),
                "benign_clients": int(benign.sum()),
                "malicious_rejected": tp,
                "malicious_admitted": fn,
                "benign_rejected": fp,
                "benign_admitted": tn,
                "malicious_detection_precision": float(precision),
                "malicious_rejection_recall": float(recall),
                "malicious_detection_f1": float(f1),
                "benign_retention_rate": float(tn / max(tn + fp, 1)),
                "benign_false_rejection_rate": float(fp / max(tn + fp, 1)),
            }
        )
    return pd.DataFrame(rows)


def client_summary(table: pd.DataFrame) -> pd.DataFrame:
    summary = (
        table.groupby("client_id", as_index=False)
        .agg(
            actual_malicious=("actual_malicious", "max"),
            mean_personalized_score=("personalized_instantaneous_score", "mean"),
            max_personalized_score=("personalized_instantaneous_score", "max"),
            mean_personalized_ema=("personalized_score_ema", "mean"),
            max_personalized_ema=("personalized_score_ema", "max"),
            max_strikes=("personalized_strike_count", "max"),
            rejected_any=("personalized_rejected", "max"),
            first_rejected_round=(
                "round",
                lambda s: int(s.iloc[0]),
            ),
        )
    )
    first_rejected = (
        table[table["personalized_rejected"].astype(bool)]
        .groupby("client_id")["round"]
        .min()
        .to_dict()
    )
    summary["first_rejected_round"] = [
        first_rejected.get(int(client_id), np.nan)
        for client_id in summary["client_id"]
    ]
    return summary


def plot_score_distribution(clean: pd.DataFrame, attack: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        clean["personalized_instantaneous_score"],
        bins=24,
        alpha=0.55,
        label="Clean leave-one-out",
        density=True,
    )
    attack_benign = attack[~attack["actual_malicious"].astype(bool)]
    attack_malicious = attack[attack["actual_malicious"].astype(bool)]
    ax.hist(
        attack_benign["personalized_instantaneous_score"],
        bins=24,
        alpha=0.55,
        label="Attack run, benign clients",
        density=True,
    )
    ax.hist(
        attack_malicious["personalized_instantaneous_score"],
        bins=24,
        alpha=0.55,
        label="Attack run, malicious clients",
        density=True,
    )
    ax.set_xlabel("Personalized bidirectional residual score")
    ax.set_ylabel("Density")
    ax.set_title("Personalized residual score distributions")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out)


def plot_security(round_table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        round_table["round"],
        round_table["malicious_rejection_recall"],
        marker="o",
        label="Malicious rejection recall",
    )
    ax.plot(
        round_table["round"],
        round_table["benign_retention_rate"],
        marker="s",
        label="Benign retention",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("Personalized residual diagnostic security metrics")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out)


def plot_client_ranking(summary: pd.DataFrame, out: Path) -> None:
    ordered = summary.sort_values("max_personalized_ema", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(ordered))
    ax.bar(x, ordered["max_personalized_ema"])
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{int(client_id)}{'*' if bool(malicious) else ''}"
            for client_id, malicious in zip(
                ordered["client_id"], ordered["actual_malicious"]
            )
        ],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("Client ID, * indicates malicious in diagnostic evaluation")
    ax.set_ylabel("Maximum personalized score EMA")
    ax.set_title("Client-level personalized residual ranking")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out)


def main() -> int:
    args = parse_args()
    clean_path = args.clean_table.expanduser().resolve()
    attack_path = args.attack_table.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(clean_path)
    attack = pd.read_csv(attack_path)
    validate_tables(clean, attack)

    profiles_table, profiles, pooled_scales = build_profiles(
        clean, args.pooled_scale_floor
    )
    clean_loo = score_clean_leave_one_out(
        clean, pooled_scales, args.pooled_scale_floor
    )
    attack_scored = score_with_profiles(attack, profiles)

    instant_threshold = quantile_higher(
        clean_loo["personalized_instantaneous_score"], args.clean_quantile
    )
    clean_ema = clean_ema_reference(clean_loo, args.ema_decay)
    ema_threshold = quantile_higher(clean_ema, args.clean_quantile)

    clean_temporal = add_temporal_decisions(
        clean_loo,
        instant_threshold=instant_threshold,
        ema_threshold=ema_threshold,
        ema_decay=args.ema_decay,
        warmup_rounds=args.warmup_rounds,
        required_strikes=args.required_strikes,
    )
    attack_temporal = add_temporal_decisions(
        attack_scored,
        instant_threshold=instant_threshold,
        ema_threshold=ema_threshold,
        ema_decay=args.ema_decay,
        warmup_rounds=args.warmup_rounds,
        required_strikes=args.required_strikes,
    )

    round_table = security_by_round(attack_temporal)
    clients = client_summary(attack_temporal)

    labels = attack_temporal["actual_malicious"].astype(int).to_numpy()
    row_auc = (
        float(roc_auc_score(labels, attack_temporal["personalized_instantaneous_score"]))
        if len(np.unique(labels)) == 2
        else math.nan
    )
    client_auc = (
        float(
            roc_auc_score(
                clients["actual_malicious"].astype(int),
                clients["max_personalized_ema"],
            )
        )
        if clients["actual_malicious"].nunique() == 2
        else math.nan
    )

    profiles_table.to_csv(tables_dir / "personalized_clean_profiles.csv", index=False)
    clean_temporal.to_csv(tables_dir / "clean_leave_one_out_scores.csv", index=False)
    attack_temporal.to_csv(tables_dir / "attack_personalized_scores.csv", index=False)
    round_table.to_csv(tables_dir / "round_security_metrics.csv", index=False)
    clients.to_csv(tables_dir / "client_security_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "clean_quantile": args.clean_quantile,
                "instantaneous_threshold": instant_threshold,
                "ema_threshold": ema_threshold,
                "clean_rows": len(clean_temporal),
                "clean_rejected_rows": int(clean_temporal["personalized_rejected"].sum()),
                "attack_rows": len(attack_temporal),
                "diagnostic_row_roc_auc": row_auc,
                "diagnostic_client_roc_auc": client_auc,
            }
        ]
    ).to_csv(tables_dir / "threshold_and_diagnostic_summary.csv", index=False)

    plot_score_distribution(
        clean_temporal, attack_temporal, figures_dir / "personalized_score_distributions"
    )
    plot_security(round_table, figures_dir / "personalized_security_by_round")
    plot_client_ranking(clients, figures_dir / "personalized_client_ranking")

    final_round = round_table.sort_values("round").iloc[-1]
    metadata = {
        "experiment_version": "3.5.1-offline-diagnostic",
        "status": "development_diagnostic_not_final_paper_result",
        "purpose": "test_personalized_bidirectional_residual_signal_before_retraining",
        "clean_table": str(clean_path),
        "attack_table": str(attack_path),
        "signals": SIGNALS,
        "score_rule": "mean_of_two_largest_absolute_personalized_robust_z_scores",
        "threshold_selection": "clean_leave_one_out_quantile_only",
        "malicious_labels_used_for_threshold_selection": False,
        "clean_quantile": args.clean_quantile,
        "instantaneous_threshold": instant_threshold,
        "ema_threshold": ema_threshold,
        "ema_decay": args.ema_decay,
        "warmup_rounds": args.warmup_rounds,
        "required_strikes": args.required_strikes,
        "diagnostic_row_roc_auc": row_auc,
        "diagnostic_client_roc_auc": client_auc,
        "final_round_malicious_rejection_recall": float(
            final_round["malicious_rejection_recall"]
        ),
        "final_round_benign_retention_rate": float(
            final_round["benign_retention_rate"]
        ),
    }
    with (output_dir / "personalized_residual_v351_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print("Personalized Residual Diagnostic V3.5.1 complete")
    print("Thresholds calibrated from clean rows only")
    print("Instantaneous threshold:", f"{instant_threshold:.6f}")
    print("EMA threshold:", f"{ema_threshold:.6f}")
    print("Clean rejected rows:", int(clean_temporal["personalized_rejected"].sum()))
    print("Diagnostic row ROC AUC:", f"{row_auc:.6f}")
    print("Diagnostic client ROC AUC:", f"{client_auc:.6f}")
    print(
        "Final-round malicious rejection recall:",
        f"{float(final_round['malicious_rejection_recall']):.6f}",
    )
    print(
        "Final-round benign retention:",
        f"{float(final_round['benign_retention_rate']):.6f}",
    )
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

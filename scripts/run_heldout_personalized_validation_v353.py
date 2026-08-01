#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FROZEN_INSTANT_QUANTILE = 0.80
FROZEN_EMA_QUANTILE = 0.90
FROZEN_EMA_DECAY = 0.55
FROZEN_REQUIRED_STRIKES = 1
FROZEN_WARMUP_ROUNDS = 2

STAGE_TABLE = Path("tables") / "client_decisions_and_signals.csv"
PERSONALIZED_CLEAN_TABLE = Path("tables") / "clean_leave_one_out_scores.csv"
PERSONALIZED_ATTACK_TABLE = Path("tables") / "attack_personalized_scores.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run held-out seed validation for the frozen personalized residual rule."
    )
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--partition-file", type=Path, required=True)
    p.add_argument("--clean-seed-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", default="7,99,123,2026")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--evaluation-batch-size", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument(
        "--stage-script",
        type=Path,
        default=Path("scripts/run_proposed_temporal_defense_v35.py"),
    )
    p.add_argument(
        "--personalized-script",
        type=Path,
        default=Path("scripts/analyze_personalized_residual_v351.py"),
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def parse_seeds(value: str) -> List[int]:
    seeds = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one held-out seed is required")
    if 42 in seeds:
        raise ValueError("Seed 42 is the development seed and cannot be held out")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate held-out seeds are not allowed")
    return seeds


def resolve_from_repo(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def run_live(command: Sequence[str], cwd: Path, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("\nRUNNING:")
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in command))
    print("")
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}. See log: {log_file}"
        )


def prepare_subrun(folder: Path, completion_file: Path, force: bool) -> bool:
    if force and folder.exists():
        shutil.rmtree(folder)
    if completion_file.exists():
        print(f"SKIP completed subrun: {folder}")
        return False
    if folder.exists():
        print(f"REMOVE incomplete subrun: {folder}")
        shutil.rmtree(folder)
    return True


def quantile_higher(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=float)
    try:
        return float(np.quantile(array, q, method="higher"))
    except TypeError:
        return float(np.quantile(array, q, interpolation="higher"))


def add_ema(table: pd.DataFrame, decay: float) -> pd.DataFrame:
    result = table.copy().sort_values(["round", "client_id"]).reset_index(drop=True)
    memory: Dict[int, float] = {}
    ema = []
    for _, row in result.iterrows():
        client_id = int(row["client_id"])
        score = float(row["personalized_instantaneous_score"])
        previous = float(memory.get(client_id, 0.0))
        current = decay * previous + (1.0 - decay) * score
        memory[client_id] = current
        ema.append(current)
    result["frozen_rule_ema"] = ema
    return result


def apply_frozen_rule(
    table: pd.DataFrame,
    instant_threshold: float,
    ema_threshold: float,
) -> pd.DataFrame:
    result = add_ema(table, FROZEN_EMA_DECAY)
    strike_memory: Dict[int, int] = {}
    strikes = []
    rejected = []

    for _, row in result.iterrows():
        client_id = int(row["client_id"])
        round_id = int(row["round"])
        score = float(row["personalized_instantaneous_score"])
        prior = int(strike_memory.get(client_id, 0))

        if round_id > FROZEN_WARMUP_ROUNDS and score >= instant_threshold:
            current = prior + 1
        else:
            current = max(prior - 1, 0)
        strike_memory[client_id] = current

        decision = (
            round_id > FROZEN_WARMUP_ROUNDS
            and current >= FROZEN_REQUIRED_STRIKES
            and float(row["frozen_rule_ema"]) >= ema_threshold
        )
        strikes.append(current)
        rejected.append(bool(decision))

    result["frozen_rule_strikes"] = strikes
    result["frozen_rule_rejected"] = rejected
    return result


def summarize_seed(seed: int, clean: pd.DataFrame, attack: pd.DataFrame) -> Dict[str, float]:
    clean_rejected = clean["frozen_rule_rejected"].astype(bool)
    clean_rejected_clients = set(
        clean.loc[clean_rejected, "client_id"].astype(int)
    )
    clean_round_retention = (
        clean.groupby("round")["frozen_rule_rejected"]
        .apply(lambda s: 1.0 - s.astype(bool).mean())
    )

    attack_rejected = attack["frozen_rule_rejected"].astype(bool)
    malicious = attack["actual_malicious"].astype(bool)
    benign = ~malicious

    malicious_clients = set(attack.loc[malicious, "client_id"].astype(int))
    benign_clients = set(attack.loc[benign, "client_id"].astype(int))
    rejected_clients = set(attack.loc[attack_rejected, "client_id"].astype(int))
    malicious_rejected = rejected_clients & malicious_clients
    benign_rejected = rejected_clients & benign_clients

    max_round = int(attack["round"].max())
    final = attack[attack["round"].astype(int).eq(max_round)]
    final_rejected = final["frozen_rule_rejected"].astype(bool)
    final_malicious = final["actual_malicious"].astype(bool)
    final_benign = ~final_malicious

    ever_tp = len(malicious_rejected)
    ever_fp = len(benign_rejected)
    ever_fn = len(malicious_clients) - ever_tp
    ever_tn = len(benign_clients) - ever_fp

    final_tp = int((final_malicious & final_rejected).sum())
    final_fp = int((final_benign & final_rejected).sum())
    final_fn = int((final_malicious & ~final_rejected).sum())
    final_tn = int((final_benign & ~final_rejected).sum())

    ever_precision = ever_tp / max(ever_tp + ever_fp, 1)
    ever_recall = ever_tp / max(ever_tp + ever_fn, 1)
    ever_f1 = 2 * ever_precision * ever_recall / max(ever_precision + ever_recall, 1e-12)

    final_precision = final_tp / max(final_tp + final_fp, 1)
    final_recall = final_tp / max(final_tp + final_fn, 1)
    final_f1 = 2 * final_precision * final_recall / max(final_precision + final_recall, 1e-12)

    first_detection = (
        attack[
            attack["actual_malicious"].astype(bool)
            & attack["frozen_rule_rejected"].astype(bool)
        ]
        .groupby("client_id")["round"]
        .min()
    )

    return {
        "seed": int(seed),
        "instant_quantile": FROZEN_INSTANT_QUANTILE,
        "ema_quantile": FROZEN_EMA_QUANTILE,
        "ema_decay": FROZEN_EMA_DECAY,
        "required_strikes": FROZEN_REQUIRED_STRIKES,
        "warmup_rounds": FROZEN_WARMUP_ROUNDS,
        "instantaneous_threshold": float(clean.attrs["instantaneous_threshold"]),
        "ema_threshold": float(clean.attrs["ema_threshold"]),
        "clean_rejected_rows": int(clean_rejected.sum()),
        "clean_rejected_clients": int(len(clean_rejected_clients)),
        "clean_final_benign_retention": float(
            1.0
            - clean[
                clean["round"].astype(int).eq(int(clean["round"].max()))
            ]["frozen_rule_rejected"].astype(bool).mean()
        ),
        "clean_min_round_benign_retention": float(clean_round_retention.min()),
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
        "median_first_malicious_detection_round": (
            float(first_detection.median()) if len(first_detection) else math.nan
        ),
    }


def aggregate_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "clean_rejected_rows",
        "clean_rejected_clients",
        "clean_final_benign_retention",
        "clean_min_round_benign_retention",
        "attack_ever_precision",
        "attack_ever_malicious_recall",
        "attack_ever_detection_f1",
        "attack_ever_benign_retention",
        "attack_final_precision",
        "attack_final_malicious_recall",
        "attack_final_detection_f1",
        "attack_final_benign_retention",
        "median_first_malicious_detection_round",
    ]
    rows = []
    n = len(per_seed)
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(max(n - 1, 1), 1.96)
    for metric in metrics:
        values = pd.to_numeric(per_seed[metric], errors="coerce").dropna().to_numpy(float)
        count = len(values)
        mean = float(np.mean(values)) if count else math.nan
        std = float(np.std(values, ddof=1)) if count > 1 else math.nan
        half = float(tcrit * std / math.sqrt(count)) if count > 1 else math.nan
        rows.append({
            "metric": metric,
            "n_seeds": count,
            "mean": mean,
            "sample_std": std,
            "minimum": float(np.min(values)) if count else math.nan,
            "maximum": float(np.max(values)) if count else math.nan,
            "t95_ci_lower": mean - half if count > 1 else math.nan,
            "t95_ci_upper": mean + half if count > 1 else math.nan,
        })
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_seed_security(per_seed: pd.DataFrame, base: Path) -> None:
    ordered = per_seed.sort_values("seed")
    x = np.arange(len(ordered))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - 1.5 * width, ordered["attack_ever_malicious_recall"], width, label="Ever malicious recall")
    ax.bar(x - 0.5 * width, ordered["attack_final_malicious_recall"], width, label="Final malicious recall")
    ax.bar(x + 0.5 * width, ordered["attack_ever_benign_retention"], width, label="Ever benign retention")
    ax.bar(x + 1.5 * width, ordered["attack_final_benign_retention"], width, label="Final benign retention")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in ordered["seed"]])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Held-out seed")
    ax.set_ylabel("Rate")
    ax.set_title("Frozen personalized rule on held-out seeds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, base)


def plot_clean_false_rejections(per_seed: pd.DataFrame, base: Path) -> None:
    ordered = per_seed.sort_values("seed")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([str(int(s)) for s in ordered["seed"]], ordered["clean_rejected_clients"])
    ax.set_xlabel("Held-out seed")
    ax.set_ylabel("Clean clients rejected at least once")
    ax.set_title("Clean false rejections under frozen rule")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, base)


def plot_thresholds(per_seed: pd.DataFrame, base: Path) -> None:
    ordered = per_seed.sort_values("seed")
    x = np.arange(len(ordered))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, ordered["instantaneous_threshold"], width, label="Instantaneous threshold")
    ax.bar(x + width / 2, ordered["ema_threshold"], width, label="EMA threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in ordered["seed"]])
    ax.set_xlabel("Held-out seed")
    ax.set_ylabel("Clean-calibrated threshold")
    ax.set_title("Threshold variation under frozen quantile rule")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, base)


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    seeds = parse_seeds(args.seeds)

    data_file = resolve_from_repo(args.data_file, repo_root)
    partition_file = resolve_from_repo(args.partition_file, repo_root)
    clean_seed_root = resolve_from_repo(args.clean_seed_root, repo_root)
    output_dir = resolve_from_repo(args.output_dir, repo_root)
    stage_script = resolve_from_repo(args.stage_script, repo_root)
    personalized_script = resolve_from_repo(args.personalized_script, repo_root)

    for required in [data_file, partition_file, clean_seed_root, stage_script, personalized_script]:
        if not required.exists():
            raise FileNotFoundError(f"Required path not found: {required}")

    logs_dir = output_dir / "logs"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    seed_runs_dir = output_dir / "seed_runs"
    for folder in [logs_dir, tables_dir, figures_dir, seed_runs_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    summaries = []

    for seed in seeds:
        print("\n" + "=" * 78)
        print(f"HELD-OUT SEED {seed}")
        print("=" * 78)

        clean_seed_dir = clean_seed_root / f"seed_{seed}"
        if not clean_seed_dir.exists():
            raise FileNotFoundError(f"Clean seed directory not found: {clean_seed_dir}")

        seed_root = seed_runs_dir / f"seed_{seed}"
        clean_out = seed_root / "stage1_clean"
        attack_out = seed_root / "stage1_strong_attack"
        personalized_out = seed_root / "personalized_scores"
        validation_out = seed_root / "frozen_rule_validation"
        validation_out.mkdir(parents=True, exist_ok=True)

        common = [
            "--data-file", str(data_file),
            "--partition-file", str(partition_file),
            "--clean-seed-dir", str(clean_seed_dir),
            "--model-seed", str(seed),
            "--rounds", str(args.rounds),
            "--batch-size", str(args.batch_size),
            "--evaluation-batch-size", str(args.evaluation_batch_size),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--threads", str(args.threads),
            "--early-stopping-patience", str(args.early_stopping_patience),
            "--validation-only",
        ]

        clean_completion = clean_out / STAGE_TABLE
        if prepare_subrun(clean_out, clean_completion, args.force):
            run_live(
                [sys.executable, str(stage_script), "--mode", "clean", *common, "--output-dir", str(clean_out)],
                repo_root,
                logs_dir / f"seed_{seed}_stage1_clean.log",
            )

        attack_completion = attack_out / STAGE_TABLE
        if prepare_subrun(attack_out, attack_completion, args.force):
            run_live(
                [sys.executable, str(stage_script), "--mode", "strong_attack", *common, "--output-dir", str(attack_out)],
                repo_root,
                logs_dir / f"seed_{seed}_stage1_strong_attack.log",
            )

        personalized_completion = personalized_out / PERSONALIZED_ATTACK_TABLE
        if prepare_subrun(personalized_out, personalized_completion, args.force):
            run_live(
                [
                    sys.executable,
                    str(personalized_script),
                    "--clean-table", str(clean_completion),
                    "--attack-table", str(attack_completion),
                    "--output-dir", str(personalized_out),
                ],
                repo_root,
                logs_dir / f"seed_{seed}_personalized_scores.log",
            )

        clean_scores = pd.read_csv(personalized_out / PERSONALIZED_CLEAN_TABLE)
        attack_scores = pd.read_csv(personalized_out / PERSONALIZED_ATTACK_TABLE)

        clean_with_ema = add_ema(clean_scores, FROZEN_EMA_DECAY)
        instant_threshold = quantile_higher(
            clean_scores["personalized_instantaneous_score"],
            FROZEN_INSTANT_QUANTILE,
        )
        ema_threshold = quantile_higher(
            clean_with_ema["frozen_rule_ema"],
            FROZEN_EMA_QUANTILE,
        )

        clean_decisions = apply_frozen_rule(clean_scores, instant_threshold, ema_threshold)
        attack_decisions = apply_frozen_rule(attack_scores, instant_threshold, ema_threshold)
        clean_decisions.attrs["instantaneous_threshold"] = instant_threshold
        clean_decisions.attrs["ema_threshold"] = ema_threshold

        clean_decisions.to_csv(validation_out / "clean_frozen_rule_decisions.csv", index=False)
        attack_decisions.to_csv(validation_out / "attack_frozen_rule_decisions.csv", index=False)

        seed_summary = summarize_seed(seed, clean_decisions, attack_decisions)
        summaries.append(seed_summary)
        pd.DataFrame([seed_summary]).to_csv(validation_out / "seed_validation_summary.csv", index=False)

        print("\nFrozen-rule seed summary")
        for key in [
            "clean_rejected_clients",
            "clean_rejected_rows",
            "attack_ever_malicious_recall",
            "attack_final_malicious_recall",
            "attack_ever_precision",
            "attack_final_precision",
            "attack_ever_benign_retention",
            "attack_final_benign_retention",
        ]:
            print(f"{key}: {seed_summary[key]}")

    per_seed = pd.DataFrame(summaries).sort_values("seed").reset_index(drop=True)
    aggregate = aggregate_metrics(per_seed)

    per_seed.to_csv(tables_dir / "heldout_per_seed_metrics.csv", index=False)
    aggregate.to_csv(tables_dir / "heldout_aggregate_metrics.csv", index=False)

    plot_seed_security(per_seed, figures_dir / "heldout_seed_security_metrics")
    plot_clean_false_rejections(per_seed, figures_dir / "heldout_clean_false_rejections")
    plot_thresholds(per_seed, figures_dir / "heldout_clean_calibrated_thresholds")

    all_clean_zero = bool(per_seed["clean_rejected_clients"].eq(0).all())
    mean_ever_recall = float(per_seed["attack_ever_malicious_recall"].mean())
    mean_final_recall = float(per_seed["attack_final_malicious_recall"].mean())
    mean_ever_benign_retention = float(per_seed["attack_ever_benign_retention"].mean())
    mean_final_benign_retention = float(per_seed["attack_final_benign_retention"].mean())

    metadata = {
        "experiment": "heldout_personalized_rule_validation_v353",
        "status": "held_out_development_validation_not_final_paper_result",
        "development_seed_excluded": 42,
        "heldout_seeds": seeds,
        "frozen_rule": {
            "instant_quantile": FROZEN_INSTANT_QUANTILE,
            "ema_quantile": FROZEN_EMA_QUANTILE,
            "ema_decay": FROZEN_EMA_DECAY,
            "required_strikes": FROZEN_REQUIRED_STRIKES,
            "warmup_rounds": FROZEN_WARMUP_ROUNDS,
        },
        "threshold_values_recalibrated_per_seed_from_clean_only": True,
        "attack_labels_used_for_thresholds": False,
        "all_heldout_clean_seeds_zero_rejected_clients": all_clean_zero,
        "mean_attack_ever_malicious_recall": mean_ever_recall,
        "mean_attack_final_malicious_recall": mean_final_recall,
        "mean_attack_ever_benign_retention": mean_ever_benign_retention,
        "mean_attack_final_benign_retention": mean_final_benign_retention,
        "resume_granularity": "completed clean, attack, and personalized subruns",
        "final_paper_claim_allowed": False,
    }
    with (output_dir / "heldout_validation_v353_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\n" + "=" * 78)
    print("Held-out Personalized Validation V3.5.3 complete")
    print("=" * 78)
    print("Held-out seeds:", seeds)
    print("All clean seeds had zero rejected clients:", all_clean_zero)
    print("Mean attack ever malicious recall:", f"{mean_ever_recall:.6f}")
    print("Mean attack final malicious recall:", f"{mean_final_recall:.6f}")
    print("Mean attack ever benign retention:", f"{mean_ever_benign_retention:.6f}")
    print("Mean attack final benign retention:", f"{mean_final_benign_retention:.6f}")
    print("CSV tables:", tables_dir)
    print("PNG and PDF figures:", figures_dir)
    print("This remains held-out development validation, not a final paper result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

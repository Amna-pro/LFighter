#!/usr/bin/env python3
"""Task 42 C1b: D1/D2 threshold calibration from captured clean updates.

Purpose
-------
C1a confirmed D1 (update-norm deviation) and D2 (update-direction
deviation) per-client reference statistics are free to compute from
existing Task 40 warmup artifacts, but found no raw per-round update
vectors existed to genuinely calibrate a threshold. Those vectors have
since been captured (V3.18B clean-mode capture, all four seeds,
zero-attack, post-warmup continuation rounds 5-8).

This script:
1. Re-derives each seed's D1 per-client historical baseline from the
   4 warmup rounds (same methodology as C1a).
2. Re-derives each seed's D2 per-client reference direction from the
   reconstruction bundle, but flattened using that seed's OWN captured
   parameter_layout.csv order -- NOT the arbitrary sorted() order C1a
   used for an exploratory check. This alignment is verified explicitly,
   not assumed.
3. Scores all 4 captured continuation rounds x 20 clients x 4 seeds
   (320 client-round observations, all genuinely clean: malicious_clients
   was empty in every capture) against those per-client references.
4. Proposes D1 and D2 thresholds at the same quantile conventions used
   throughout this project (0.95, 0.975, 0.99), pooled across all four
   seeds -- not seed-7-only, learning directly from the Task 41C mistake.

Scope
-----
Read-only relative to training: no new model training occurs here. Loads
already-captured update matrices and already-computed reconstruction
profiles only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

CONFIRMATORY_SEEDS = (7, 99, 123, 2026)
NUM_CLIENTS = 20
CONTINUATION_ROUNDS = 4
QUANTILES = (0.95, 0.975, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 42 C1b D1/D2 threshold calibration."
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        required=True,
        help=(
            "results\\cic_iot_diad_full_update_capture_v318b\\captures "
            "(contains clean_seed_<N> for all four seeds)"
        ),
    )
    parser.add_argument(
        "--reconstruction-root",
        type=Path,
        required=True,
        help="results\\cic_iot_diad_frozen_reconstruction_v3123_multiseed",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def d1_client_baseline(warmup_updates: pd.DataFrame) -> pd.DataFrame:
    """Per-client robust baseline of log1p(update_norm), from warmup rounds.

    Identical methodology to C1a / screen_multiview_oneclass_v317.py's
    client_norm_reference.
    """
    x = warmup_updates.copy()
    x["log_update_norm"] = np.log1p(x["update_norm"].to_numpy(dtype=float))
    rows: List[Dict[str, Any]] = []
    for client_id, group in x.groupby("client_id", sort=True):
        values = group["log_update_norm"].to_numpy(dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        iqr_scale = float((q75 - q25) / 1.349)
        scale = max(1.4826 * mad, iqr_scale, 1e-6)
        rows.append(
            {
                "client_id": int(client_id),
                "log_norm_median": median,
                "log_norm_robust_scale": scale,
            }
        )
    return pd.DataFrame(rows).sort_values("client_id").reset_index(drop=True)


def d2_reference_directions_aligned(
    reconstruction_bundle: Dict[str, Any], layout: pd.DataFrame
) -> np.ndarray:
    """Per-client reference direction, flattened in the EXACT order the
    captured update matrices use (layout order), not an arbitrary
    sorted() order. Returns shape (NUM_CLIENTS, parameter_count).

    This is the correctness-critical alignment step: the captured update
    matrix columns follow parameter_layout.csv's row order (start_index
    ascending, included_in_update_vector rows only). The reconstruction
    bundle's residual profile is a dict keyed by parameter name with
    arbitrary insertion order. We must slice/concatenate using layout's
    order, or every downstream cosine similarity is meaningless without
    any error being raised.
    """
    included = layout[layout["included_in_update_vector"].astype(bool)]
    included = included.sort_values("start_index")
    ordered_names = included["parameter_name"].astype(str).tolist()

    profiles = reconstruction_bundle["client_residual_profiles"]
    profile_names = set(profiles[0].keys())
    missing = set(ordered_names) - profile_names
    if missing:
        raise RuntimeError(
            f"Layout references parameter names not present in the "
            f"reconstruction profile: {sorted(missing)[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )

    total_params = int(included["numel"].sum())
    directions = np.empty((NUM_CLIENTS, total_params), dtype=np.float64)
    for client_id in range(NUM_CLIENTS):
        profile = profiles[client_id]
        pieces = [
            profile[name].detach().cpu().reshape(-1).to(torch.float64).numpy()
            for name in ordered_names
        ]
        flat = np.concatenate(pieces)
        if flat.shape[0] != total_params:
            raise RuntimeError(
                f"Client {client_id}: flattened residual has "
                f"{flat.shape[0]} elements, layout expects {total_params}."
            )
        norm = float(np.linalg.norm(flat))
        if norm <= 1e-12:
            raise RuntimeError(
                f"Client {client_id}: degenerate (near-zero) residual "
                f"direction after layout-aligned flattening."
            )
        directions[client_id] = flat / norm
    return directions


def score_round(
    matrix_path: Path,
    reference_directions: np.ndarray,
) -> pd.DataFrame:
    """Per-client D1 raw norm and D2 cosine similarity for one round."""
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if matrix.shape[0] != NUM_CLIENTS:
        raise RuntimeError(
            f"{matrix_path}: expected {NUM_CLIENTS} client rows, found "
            f"{matrix.shape[0]}."
        )
    if matrix.shape[1] != reference_directions.shape[1]:
        raise RuntimeError(
            f"{matrix_path}: matrix has {matrix.shape[1]} columns, "
            f"reference directions have {reference_directions.shape[1]}. "
            f"Parameter-layout alignment has failed silently upstream -- "
            f"stopping rather than producing meaningless cosine scores."
        )
    rows: List[Dict[str, Any]] = []
    for client_id in range(NUM_CLIENTS):
        row = np.asarray(matrix[client_id], dtype=np.float64)
        norm = float(np.linalg.norm(row))
        if norm > 1e-12:
            cosine = float(np.dot(row / norm, reference_directions[client_id]))
        else:
            cosine = float("nan")
        rows.append(
            {
                "client_id": client_id,
                "raw_update_l2_norm": norm,
                "cosine_similarity_to_reference": cosine,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    capture_root = args.capture_root.expanduser().resolve()
    reconstruction_root = args.reconstruction_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_scores: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, Any]] = []
    layout_hashes: Dict[int, str] = {}

    for seed in CONFIRMATORY_SEEDS:
        capture_dir = capture_root / f"clean_seed_{seed}"
        index_path = capture_dir / "tables" / "v318b_update_capture_index.csv"
        layout_path = capture_dir / "update_artifacts" / "parameter_layout.csv"
        calibration_dir = reconstruction_root / f"seed_{seed}" / "calibration"
        warmup_updates_path = (
            calibration_dir
            / "tables"
            / "trusted_warmup_replay_local_updates.csv"
        )
        bundle_path = (
            calibration_dir
            / "calibration"
            / "trusted_update_reconstruction_profiles.pt"
        )
        for path in (index_path, layout_path, warmup_updates_path, bundle_path):
            if not path.exists():
                raise FileNotFoundError(path)

        index = pd.read_csv(index_path)
        if len(index) != CONTINUATION_ROUNDS:
            raise RuntimeError(
                f"Seed {seed}: expected {CONTINUATION_ROUNDS} captured "
                f"rounds, found {len(index)}."
            )
        if index["mode"].astype(str).ne("clean").any():
            raise RuntimeError(f"Seed {seed}: capture index is not all-clean.")
        if index["capture_used_for_training_or_aggregation"].astype(bool).any():
            raise RuntimeError(
                f"Seed {seed}: capture reports it was used for training or "
                f"aggregation -- this would mean the capture influenced the "
                f"frozen trajectory, which must never happen."
            )

        layout = pd.read_csv(layout_path)
        layout_hashes[seed] = file_sha256(layout_path)

        warmup_updates = pd.read_csv(warmup_updates_path)
        d1_baseline = d1_client_baseline(warmup_updates)

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        if int(bundle["model_seed"]) != seed:
            raise RuntimeError(f"Seed {seed}: reconstruction bundle seed mismatch.")
        reference_directions = d2_reference_directions_aligned(bundle, layout)

        for _, row in index.iterrows():
            matrix_path = capture_dir / str(row["update_matrix_file"])
            observed_hash = file_sha256(matrix_path)
            if observed_hash != str(row["update_matrix_sha256"]):
                raise RuntimeError(
                    f"Seed {seed}, round {row['monitoring_round']}: update "
                    f"matrix hash mismatch against its own index record."
                )
            round_scores = score_round(matrix_path, reference_directions)
            round_scores["model_seed"] = seed
            round_scores["monitoring_round"] = int(row["monitoring_round"])
            round_scores = round_scores.merge(
                d1_baseline, on="client_id", how="left", validate="many_to_one"
            )
            round_scores["d1_log_robust_z"] = (
                np.log1p(round_scores["raw_update_l2_norm"])
                - round_scores["log_norm_median"]
            ) / round_scores["log_norm_robust_scale"]
            round_scores["d2_deviation"] = (
                1.0 - round_scores["cosine_similarity_to_reference"]
            )
            all_scores.append(round_scores)

        manifest_rows.append(
            {
                "model_seed": seed,
                "layout_sha256": layout_hashes[seed],
                "parameter_count": int(
                    layout[layout["included_in_update_vector"].astype(bool)][
                        "numel"
                    ].sum()
                ),
                "rounds_captured": int(len(index)),
                "malicious_clients_in_capture": "[]",
            }
        )

    unique_layout_hashes = set(layout_hashes.values())
    if len(unique_layout_hashes) != 1:
        raise RuntimeError(
            f"Parameter layout differs across seeds: {layout_hashes}. "
            f"This should be impossible for an identical architecture -- "
            f"stopping rather than silently pooling misaligned data."
        )

    scores = pd.concat(all_scores, ignore_index=True)
    expected_rows = len(CONFIRMATORY_SEEDS) * CONTINUATION_ROUNDS * NUM_CLIENTS
    if len(scores) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} scored client-rounds, found "
            f"{len(scores)}."
        )

    scores.to_csv(tables_dir / "task42c1b_all_clean_scores.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(
        tables_dir / "task42c1b_input_manifest.csv", index=False
    )

    d1_values = scores["d1_log_robust_z"].to_numpy(dtype=float)
    d2_values = scores["d2_deviation"].to_numpy(dtype=float)
    threshold_rows: List[Dict[str, Any]] = []
    for quantile in QUANTILES:
        threshold_rows.append(
            {
                "quantile": quantile,
                "d1_instant_threshold": float(np.quantile(d1_values, quantile)),
                "d2_instant_threshold": float(np.quantile(d2_values, quantile)),
                "pooled_clean_observation_count": int(len(scores)),
            }
        )
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(
        tables_dir / "task42c1b_pooled_threshold_candidates.csv", index=False
    )

    per_seed_summary = (
        scores.groupby("model_seed", as_index=False)
        .agg(
            mean_d1_z=("d1_log_robust_z", "mean"),
            max_d1_z=("d1_log_robust_z", "max"),
            mean_d2_deviation=("d2_deviation", "mean"),
            max_d2_deviation=("d2_deviation", "max"),
        )
    )
    per_seed_summary.to_csv(
        tables_dir / "task42c1b_per_seed_summary.csv", index=False
    )

    decision = {
        "experiment_version": "4.13.C1b",
        "stage": "task42_c1b_threshold_calibration",
        "seeds_used": list(CONFIRMATORY_SEEDS),
        "pooled_observation_count": int(len(scores)),
        "parameter_layout_consistent_across_seeds": True,
        "layout_sha256": next(iter(unique_layout_hashes)),
        "reserved_test_accessed": False,
        "attack_execution_performed": False,
        "candidate_thresholds_by_quantile": threshold_table.to_dict(
            orient="records"
        ),
        "next_stage": (
            "D1 and D2 now have pooled, four-seed, causally-derived "
            "threshold candidates (warmup-round reference vs. held-out "
            "continuation-round clean scoring, no leakage). Proceed to "
            "C2: seed-7 development screening of D1, D2, and D3 against "
            "the preregistered attack panel, using the quantile matching "
            "this project's established convention (recommend 0.99 for "
            "instant threshold, matching D0's frozen calibration) unless "
            "review of the candidate table suggests otherwise."
        ),
    }
    (output_dir / "task42c1b_calibration_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print("=" * 100)
    print("TASK 42 C1b D1/D2 THRESHOLD CALIBRATION")
    print("=" * 100)
    print("Seeds used:", list(CONFIRMATORY_SEEDS))
    print("Pooled clean observations:", len(scores))
    print("Parameter layout consistent across all seeds: True")
    print()
    print("Threshold candidates:")
    print(threshold_table.to_string(index=False))
    print()
    print("Per-seed summary:")
    print(per_seed_summary.to_string(index=False))
    print()
    print("Tables:", tables_dir)
    print(
        "Decision file:",
        output_dir / "task42c1b_calibration_decision.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

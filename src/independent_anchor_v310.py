"""Independent absolute transition-signature anchor utilities for V3.10."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

RAW_FEATURES = (
    "hist_js",
    "hist_l1",
    "hist_fro",
    "hist_diag_loss",
    "hist_offdiag_growth",
    "cons_js",
    "cons_l1",
    "cons_fro",
    "cons_diag_loss",
    "cons_offdiag_growth",
)
CANDIDATE = "candidate_max_anchor_absolute"
SCORE_COLUMN = f"{CANDIDATE}_ema"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def quantile_higher(values: Sequence[float], q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    try:
        return float(np.quantile(array, q, method="higher"))
    except TypeError:
        return float(np.quantile(array, q, interpolation="higher"))


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(matrix, dtype=np.float64), 1e-12, None)
    return values / np.clip(values.sum(axis=1, keepdims=True), 1e-12, None)


def rowwise_js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = normalize_rows(left)
    q = normalize_rows(right)
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        np.sum(p * np.log(p / midpoint), axis=1)
        + np.sum(q * np.log(q / midpoint), axis=1)
    )


def top5_offdiag_growth(local: np.ndarray, reference: np.ndarray) -> float:
    difference = np.asarray(local, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    mask = ~np.eye(difference.shape[0], dtype=bool)
    values = np.maximum(difference[mask], 0.0)
    count = min(5, len(values))
    if count == 0:
        return 0.0
    return float(np.partition(values, -count)[-count:].mean())


def raw_features(
    local: np.ndarray,
    historical: np.ndarray,
    consensus: np.ndarray,
    source_id: int,
    target_id: int,
) -> Dict[str, float]:
    local = normalize_rows(local)
    historical = normalize_rows(historical)
    consensus = normalize_rows(consensus)
    historical_difference = local - historical
    consensus_difference = local - consensus
    return {
        "hist_js": float(rowwise_js(local, historical).mean()),
        "hist_l1": float(np.abs(historical_difference).mean()),
        "hist_fro": float(np.linalg.norm(historical_difference)),
        "hist_diag_loss": float(
            np.maximum(np.diag(historical) - np.diag(local), 0.0).mean()
        ),
        "hist_offdiag_growth": top5_offdiag_growth(local, historical),
        "cons_js": float(rowwise_js(local, consensus).mean()),
        "cons_l1": float(np.abs(consensus_difference).mean()),
        "cons_fro": float(np.linalg.norm(consensus_difference)),
        "cons_diag_loss": float(
            np.maximum(np.diag(consensus) - np.diag(local), 0.0).mean()
        ),
        "cons_offdiag_growth": top5_offdiag_growth(local, consensus),
        "source_target_growth_from_profile": float(
            max(historical_difference[source_id, target_id], 0.0)
        ),
        "source_diagonal_loss_from_profile": float(
            max(-historical_difference[source_id, source_id], 0.0)
        ),
    }


def robust_center_scale(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    center = float(np.median(array))
    scale = 1.4826 * float(np.median(np.abs(array - center)))
    if scale < 1e-9:
        scale = float(np.std(array))
    return center, scale if scale >= 1e-9 else 1.0


def positive_z(values: Sequence[float], center: float, scale: float) -> np.ndarray:
    return np.clip(
        np.maximum(
            (np.asarray(values, dtype=np.float64) - float(center))
            / max(float(scale), 1e-12),
            0.0,
        ),
        0.0,
        20.0,
    )


def fit_feature_calibration(clean_rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scored = clean_rows.copy()
    calibration_rows = []
    for feature in RAW_FEATURES:
        center, scale = robust_center_scale(scored[feature])
        calibration_rows.append(
            {"feature": feature, "clean_median": center, "clean_scale": scale}
        )
        scored[f"z_{feature}"] = positive_z(scored[feature], center, scale)
    return scored, pd.DataFrame(calibration_rows)


def apply_feature_calibration(
    rows: pd.DataFrame, calibration: pd.DataFrame
) -> pd.DataFrame:
    result = rows.copy()
    lookup = calibration.set_index("feature")
    missing = set(RAW_FEATURES).difference(lookup.index)
    if missing:
        raise ValueError(f"Feature calibration is missing: {sorted(missing)}")
    for feature in RAW_FEATURES:
        result[f"z_{feature}"] = positive_z(
            result[feature],
            float(lookup.at[feature, "clean_median"]),
            float(lookup.at[feature, "clean_scale"]),
        )
    return result


def candidate_values(table: pd.DataFrame) -> np.ndarray:
    historical = table[
        [
            "z_hist_js",
            "z_hist_l1",
            "z_hist_diag_loss",
            "z_hist_offdiag_growth",
        ]
    ].mean(axis=1)
    consensus = table[
        [
            "z_cons_js",
            "z_cons_l1",
            "z_cons_diag_loss",
            "z_cons_offdiag_growth",
        ]
    ].mean(axis=1)
    return np.maximum(historical.to_numpy(dtype=np.float64), consensus.to_numpy(dtype=np.float64))


def add_candidate_and_ema(
    table: pd.DataFrame,
    ema_decay: float,
    initial_ema: Mapping[int, float] | None = None,
) -> Tuple[pd.DataFrame, Dict[int, float]]:
    result = table.sort_values(["client_id", "monitoring_round"]).copy()
    result[CANDIDATE] = candidate_values(result)
    ema_output = pd.Series(index=result.index, dtype=float)
    memory: Dict[int, float] = {
        int(key): float(value) for key, value in (initial_ema or {}).items()
    }
    for client_id, group in result.groupby("client_id", sort=False):
        previous = float(memory.get(int(client_id), 0.0))
        for row_index in group.index:
            previous = float(ema_decay) * previous + (1.0 - float(ema_decay)) * float(
                result.at[row_index, CANDIDATE]
            )
            ema_output.at[row_index] = previous
        memory[int(client_id)] = previous
    result[SCORE_COLUMN] = ema_output
    return result.sort_values(["monitoring_round", "client_id"]).reset_index(drop=True), memory


def weighted_average_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    sample_counts: Sequence[int],
    reference_state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    counts = np.asarray(sample_counts, dtype=np.float64)
    if len(states) == 0 or len(states) != len(counts):
        raise ValueError("State and sample-count lengths must match and be nonzero")
    if float(counts.sum()) <= 0:
        raise ValueError("Total sample count must be positive")
    weights = counts / counts.sum()
    result: Dict[str, torch.Tensor] = {}
    for key, reference in reference_state.items():
        if torch.is_floating_point(reference):
            accumulator = torch.zeros_like(reference, dtype=torch.float64)
            for state, weight in zip(states, weights):
                accumulator += state[key].detach().cpu().to(torch.float64) * float(weight)
            result[key] = accumulator.to(reference.dtype)
        else:
            result[key] = reference.detach().cpu().clone()
    return result


def save_profiles_npz(path: Path, profiles: Mapping[int, np.ndarray]) -> str:
    arrays = {f"client_{int(client_id):03d}": np.asarray(matrix, dtype=np.float64)
              for client_id, matrix in sorted(profiles.items())}
    np.savez_compressed(path, **arrays)
    return file_sha256(path)


def load_profiles_npz(path: Path, expected_clients: int) -> Dict[int, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = np.load(path)
    profiles: Dict[int, np.ndarray] = {}
    for key in bundle.files:
        if not key.startswith("client_"):
            continue
        client_id = int(key.split("_")[-1])
        profiles[client_id] = normalize_rows(bundle[key])
    if len(profiles) != expected_clients:
        raise ValueError(f"Expected {expected_clients} profiles, found {len(profiles)}")
    return profiles

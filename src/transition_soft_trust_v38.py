"""Frozen V3.8 generic dual-reference temporal soft-trust policy."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

StateDict = Dict[str, torch.Tensor]
HISTORICAL_FEATURES = (
    "max_offdiag_growth",
    "mean_diagonal_confidence_loss",
    "mean_row_js_reference",
    "frobenius_drift",
)
ROUND_FEATURES = (
    "max_offdiag_growth",
    "mean_row_js_consensus",
    "frobenius_to_consensus",
)
FROZEN_CANDIDATE = "candidate_generic_dual_reference"


@dataclass
class TemporalTrustMemory:
    candidate_ema: Dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationBundle:
    profiles: Dict[int, Dict[str, Tuple[float, float]]]
    instant_threshold: float
    ema_threshold: float
    calibration_seed: int
    profile_sha256: str
    clean_scores_sha256: str
    summary: pd.DataFrame


@dataclass(frozen=True)
class TrustAggregationResult:
    aggregated_state: StateDict
    client_table: pd.DataFrame
    summary: Dict[str, object]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile_higher(values: Sequence[float], q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    try:
        return float(np.quantile(array, q, method="higher"))
    except TypeError:
        return float(np.quantile(array, q, interpolation="higher"))


def robust_center_scale(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    center = float(np.median(array))
    mad = float(np.median(np.abs(array - center)))
    scale = 1.4826 * mad
    if scale < 1e-9:
        std = float(np.std(array))
        scale = std if std >= 1e-9 else 1.0
    return center, scale


def positive_z(values: Sequence[float], center: float, scale: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.clip(
        np.maximum((array - float(center)) / max(float(scale), 1e-12), 0.0),
        0.0,
        20.0,
    )


def load_calibration(
    analysis_dir: Path,
    calibration_seed: int,
    expected_clients: int,
    candidate: str = FROZEN_CANDIDATE,
    clean_threshold_quantile: float = 0.95,
) -> CalibrationBundle:
    if candidate != FROZEN_CANDIDATE:
        raise ValueError(f"V3.8 is frozen to {FROZEN_CANDIDATE}")
    tables_dir = analysis_dir / "tables"
    profile_path = tables_dir / "personalized_clean_feature_profiles.csv"
    clean_scores_path = tables_dir / f"seed_{calibration_seed}_clean_scored_rows.csv"
    for path in (profile_path, clean_scores_path):
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

    profiles = pd.read_csv(profile_path)
    if "seed" in profiles.columns:
        profiles = profiles[profiles["seed"].eq(calibration_seed)].copy()
    required_features = set(HISTORICAL_FEATURES)
    profiles = profiles[profiles["feature"].isin(required_features)].copy()
    clients = sorted(profiles["client_id"].astype(int).unique().tolist())
    if len(clients) != expected_clients:
        raise ValueError(
            f"Expected {expected_clients} calibration clients, found {len(clients)}"
        )
    profile_map: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for client_id in clients:
        client_rows = profiles[profiles["client_id"].eq(client_id)]
        found = set(client_rows["feature"].tolist())
        if found != required_features:
            raise ValueError(
                f"Client {client_id} calibration features do not match frozen set"
            )
        profile_map[int(client_id)] = {
            str(row.feature): (float(row.clean_median), float(row.clean_scale))
            for row in client_rows.itertuples()
        }

    clean_scores = pd.read_csv(clean_scores_path)
    ema_column = f"{candidate}_ema"
    for column in (candidate, ema_column):
        if column not in clean_scores.columns:
            raise ValueError(f"Clean calibration table is missing {column}")
    instant_threshold = quantile_higher(
        clean_scores[candidate], clean_threshold_quantile
    )
    ema_threshold = quantile_higher(
        clean_scores[ema_column], clean_threshold_quantile
    )
    summary = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "calibration_seed": int(calibration_seed),
                "clean_threshold_quantile": float(clean_threshold_quantile),
                "instant_clean_threshold": float(instant_threshold),
                "ema_clean_threshold": float(ema_threshold),
                "profile_clients": int(len(profile_map)),
                "profile_rows": int(len(profiles)),
                "clean_score_rows": int(len(clean_scores)),
                "profile_source": str(profile_path),
                "clean_score_source": str(clean_scores_path),
                "trusted_clean_history_required": True,
                "test_sets_accessed": False,
            }
        ]
    )
    return CalibrationBundle(
        profiles=profile_map,
        instant_threshold=float(instant_threshold),
        ema_threshold=float(ema_threshold),
        calibration_seed=int(calibration_seed),
        profile_sha256=file_sha256(profile_path),
        clean_scores_sha256=file_sha256(clean_scores_path),
        summary=summary,
    )


def _weighted_average_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
    reference_state: Mapping[str, torch.Tensor],
) -> StateDict:
    values = np.asarray(weights, dtype=np.float64)
    if len(states) != len(values):
        raise ValueError("states and weights must have identical lengths")
    if len(states) == 0:
        raise ValueError("At least one state is required")
    total = float(values.sum())
    if total <= 0:
        values = np.ones(len(states), dtype=np.float64)
        total = float(len(states))
    values = values / total

    result: StateDict = {}
    for key, reference in reference_state.items():
        if torch.is_floating_point(reference):
            accumulator = torch.zeros_like(reference, dtype=torch.float64)
            for state, weight in zip(states, values):
                accumulator += (
                    state[key].detach().cpu().to(torch.float64) * float(weight)
                )
            result[key] = accumulator.to(reference.dtype)
        else:
            result[key] = reference.detach().cpu().clone()
    return result


def score_and_aggregate(
    reference_state: Mapping[str, torch.Tensor],
    local_states: Sequence[Mapping[str, torch.Tensor]],
    client_rows: pd.DataFrame,
    calibration: CalibrationBundle,
    memory: TemporalTrustMemory,
    malicious_client_ids: Iterable[int] = (),
    ema_decay: float = 0.65,
    gamma: float = 2.0,
    minimum_trust: float = 0.10,
    count_cap_multiplier: float = 3.0,
    low_trust_cutoff: float = 0.50,
) -> TrustAggregationResult:
    table = client_rows.sort_values("client_id").reset_index(drop=True).copy()
    client_ids = table["client_id"].astype(int).tolist()
    if len(local_states) != len(client_ids):
        raise ValueError("Local states and client rows must align")
    if set(client_ids) != set(calibration.profiles):
        raise ValueError("Participating clients do not match calibration profiles")

    historical_columns = []
    for feature in HISTORICAL_FEATURES:
        column = f"hist_z_{feature}"
        historical_columns.append(column)
        values = []
        for row in table.itertuples():
            center, scale = calibration.profiles[int(row.client_id)][feature]
            values.append(
                float(positive_z([getattr(row, feature)], center, scale)[0])
            )
        table[column] = values

    round_columns = []
    for feature in ROUND_FEATURES:
        column = f"round_z_{feature}"
        round_columns.append(column)
        center, scale = robust_center_scale(table[feature].to_numpy(dtype=float))
        table[column] = positive_z(
            table[feature].to_numpy(dtype=float), center, scale
        )
        table[f"round_center_{feature}"] = float(center)
        table[f"round_scale_{feature}"] = float(scale)

    table["historical_generic_score"] = table[historical_columns].mean(axis=1)
    table["round_generic_score"] = table[round_columns].mean(axis=1)
    table[FROZEN_CANDIDATE] = (
        0.60 * table["historical_generic_score"]
        + 0.40 * table["round_generic_score"]
    )

    ema_values = []
    for row in table.itertuples():
        client_id = int(row.client_id)
        current = float(getattr(row, FROZEN_CANDIDATE))
        previous = float(memory.candidate_ema.get(client_id, 0.0))
        ema = float(ema_decay) * previous + (1.0 - float(ema_decay)) * current
        memory.candidate_ema[client_id] = ema
        ema_values.append(ema)
    table[f"{FROZEN_CANDIDATE}_ema"] = ema_values

    instant_ratio = (
        table[FROZEN_CANDIDATE].to_numpy(dtype=float)
        / max(calibration.instant_threshold, 1e-12)
    )
    ema_ratio = (
        table[f"{FROZEN_CANDIDATE}_ema"].to_numpy(dtype=float)
        / max(calibration.ema_threshold, 1e-12)
    )
    policy_ratio = np.maximum(instant_ratio, ema_ratio)
    excess = np.maximum(policy_ratio - 1.0, 0.0)
    trust = np.clip(
        np.exp(-float(gamma) * excess),
        float(minimum_trust),
        1.0,
    )

    counts = table["client_samples"].to_numpy(dtype=float)
    count_cap = float(np.median(counts) * float(count_cap_multiplier))
    bounded_counts = np.minimum(counts, max(count_cap, 1.0))
    base_weights = bounded_counts / max(float(bounded_counts.sum()), 1e-12)
    weighted_counts = bounded_counts * trust
    aggregation_weights = weighted_counts / max(
        float(weighted_counts.sum()), 1e-12
    )

    malicious_set = {int(value) for value in malicious_client_ids}
    actual_malicious = np.asarray(
        [client_id in malicious_set for client_id in client_ids], dtype=bool
    )
    benign = ~actual_malicious
    low_trust = trust < float(low_trust_cutoff)

    table["instant_threshold"] = float(calibration.instant_threshold)
    table["ema_threshold"] = float(calibration.ema_threshold)
    table["instant_ratio"] = instant_ratio
    table["ema_ratio"] = ema_ratio
    table["policy_ratio"] = policy_ratio
    table["threshold_excess"] = excess
    table["trust_factor"] = trust
    table["low_trust"] = low_trust
    table["bounded_client_samples"] = bounded_counts
    table["base_aggregation_weight"] = base_weights
    table["aggregation_weight"] = aggregation_weights
    table["actual_malicious"] = actual_malicious

    base_malicious_share = float(base_weights[actual_malicious].sum())
    adjusted_malicious_share = float(aggregation_weights[actual_malicious].sum())
    influence_reduction = (
        1.0 - adjusted_malicious_share / max(base_malicious_share, 1e-12)
        if actual_malicious.any()
        else 0.0
    )

    summary: Dict[str, object] = {
        "mean_trust": float(np.mean(trust)),
        "minimum_trust": float(np.min(trust)),
        "mean_benign_trust": float(np.mean(trust[benign])) if benign.any() else 1.0,
        "mean_malicious_trust": float(np.mean(trust[actual_malicious]))
        if actual_malicious.any()
        else 1.0,
        "benign_low_trust_rate": float(np.mean(low_trust[benign]))
        if benign.any()
        else 0.0,
        "malicious_low_trust_recall": float(np.mean(low_trust[actual_malicious]))
        if actual_malicious.any()
        else 0.0,
        "base_malicious_weight_share": base_malicious_share,
        "adjusted_malicious_weight_share": adjusted_malicious_share,
        "malicious_influence_reduction": float(influence_reduction),
        "mean_benign_candidate": float(
            table.loc[benign, FROZEN_CANDIDATE].mean()
        )
        if benign.any()
        else 0.0,
        "mean_malicious_candidate": float(
            table.loc[actual_malicious, FROZEN_CANDIDATE].mean()
        )
        if actual_malicious.any()
        else 0.0,
        "highest_suspicion_client": int(
            table.iloc[int(np.argmax(policy_ratio))]["client_id"]
        ),
        "highest_policy_ratio": float(np.max(policy_ratio)),
    }
    aggregated_state = _weighted_average_states(
        states=local_states,
        weights=aggregation_weights,
        reference_state=reference_state,
    )
    return TrustAggregationResult(
        aggregated_state=aggregated_state,
        client_table=table,
        summary=summary,
    )

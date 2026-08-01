"""Common-mode-corrected personalized prediction soft trust, V3.6.2.

Rationale
---------
V3.6.1 used an absolute personalized prediction-drift residual. Under sustained
poisoning, the poisoned global reference moved enough that several benign clients
also accumulated large residuals. V3.6.2 removes the round-wide common component
using a robust median and MAD before temporal smoothing.

The robust common-mode correction assumes fewer than half of participating
clients are malicious, a standard Byzantine-majority assumption. Malicious
labels are never used to compute aggregation weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

StateDict = Dict[str, torch.Tensor]


@dataclass
class CommonModeSoftTrustMemory:
    relative_residual_ema: Dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CommonModeSoftTrustRoundResult:
    aggregated_state: StateDict
    client_decisions: pd.DataFrame
    summary: Dict[str, object]


def _weighted_average_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
    reference_state: Mapping[str, torch.Tensor],
) -> StateDict:
    values = np.asarray(weights, dtype=np.float64)
    if len(states) != len(values):
        raise ValueError("states and weights must have identical lengths")
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


def _personalized_prediction_residual(
    client_ids: Sequence[int],
    prediction_drift: np.ndarray,
    prediction_profiles: Mapping[int, Tuple[float, float]],
    clip: float = 20.0,
) -> np.ndarray:
    transformed = np.log1p(
        np.maximum(np.asarray(prediction_drift, dtype=np.float64), 0.0)
    )
    residuals = np.zeros(len(client_ids), dtype=np.float64)
    for index, (client_id, value) in enumerate(zip(client_ids, transformed)):
        if int(client_id) not in prediction_profiles:
            raise KeyError(f"Missing clean prediction profile for client {client_id}")
        center, scale = prediction_profiles[int(client_id)]
        residuals[index] = abs(float(value) - float(center)) / max(
            float(scale), 1e-12
        )
    return np.clip(residuals, 0.0, float(clip))


def robust_common_mode_score(
    personalized_residual: np.ndarray,
    scale_floor: float = 0.25,
) -> Tuple[np.ndarray, float, float]:
    values = np.asarray(personalized_residual, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3:
        raise ValueError("At least three client residuals are required")
    if scale_floor <= 0:
        raise ValueError("scale_floor must be positive")

    round_median = float(np.median(values))
    mad = float(np.median(np.abs(values - round_median)))
    robust_scale = max(1.4826 * mad, float(scale_floor))

    score = np.maximum((values - round_median) / robust_scale, 0.0)
    return np.clip(score, 0.0, 20.0), round_median, robust_scale


def aggregate_common_mode_soft_trust(
    reference_state: Mapping[str, torch.Tensor],
    local_states: Sequence[Mapping[str, torch.Tensor]],
    client_ids: Sequence[int],
    sample_counts: Sequence[int],
    probe_signals: pd.DataFrame,
    prediction_profiles: Mapping[int, Tuple[float, float]],
    clean_relative_ema_threshold: float,
    memory: CommonModeSoftTrustMemory,
    round_id: int,
    malicious_client_ids: Iterable[int] = (),
    ema_decay: float = 0.65,
    trust_gamma: float = 1.20,
    minimum_trust: float = 0.15,
    count_cap_multiplier: float = 3.0,
    low_trust_cutoff: float = 0.50,
    round_scale_floor: float = 0.25,
) -> CommonModeSoftTrustRoundResult:
    if len(local_states) != len(client_ids) or len(local_states) != len(sample_counts):
        raise ValueError("local_states, client_ids, and sample_counts must align")
    if len(local_states) < 3:
        raise ValueError("At least three local states are required")
    if clean_relative_ema_threshold <= 0:
        raise ValueError("clean_relative_ema_threshold must be positive")

    client_ids = [int(value) for value in client_ids]
    probe = probe_signals.set_index("client_id").loc[client_ids].reset_index()
    prediction_drift = probe["max_offdiag_probability_drift"].to_numpy(
        dtype=np.float64
    )
    personalized_residual = _personalized_prediction_residual(
        client_ids=client_ids,
        prediction_drift=prediction_drift,
        prediction_profiles=prediction_profiles,
    )
    relative_score, round_median, round_scale = robust_common_mode_score(
        personalized_residual,
        scale_floor=round_scale_floor,
    )

    ema_values = []
    for client_id, current in zip(client_ids, relative_score):
        previous = float(memory.relative_residual_ema.get(client_id, 0.0))
        ema = float(ema_decay) * previous + (1.0 - float(ema_decay)) * float(
            current
        )
        memory.relative_residual_ema[client_id] = ema
        ema_values.append(ema)
    ema_array = np.asarray(ema_values, dtype=np.float64)

    threshold_excess = np.maximum(
        ema_array / max(float(clean_relative_ema_threshold), 1e-12) - 1.0,
        0.0,
    )
    trust = np.exp(-float(trust_gamma) * threshold_excess)
    trust = np.clip(trust, float(minimum_trust), 1.0)

    counts = np.asarray(sample_counts, dtype=np.float64)
    count_cap = float(np.median(counts) * float(count_cap_multiplier))
    bounded_counts = np.minimum(counts, max(count_cap, 1.0))
    base_weights = bounded_counts / max(float(bounded_counts.sum()), 1e-12)
    weighted_counts = bounded_counts * trust
    normalized_weights = weighted_counts / max(
        float(weighted_counts.sum()), 1e-12
    )

    aggregated_state = _weighted_average_states(
        local_states,
        normalized_weights,
        reference_state,
    )

    malicious_set = {int(value) for value in malicious_client_ids}
    actual_malicious = np.asarray(
        [client_id in malicious_set for client_id in client_ids],
        dtype=bool,
    )
    benign = ~actual_malicious
    low_trust = trust < float(low_trust_cutoff)

    base_malicious_share = float(base_weights[actual_malicious].sum())
    adjusted_malicious_share = float(normalized_weights[actual_malicious].sum())
    influence_reduction = (
        1.0
        - adjusted_malicious_share / max(base_malicious_share, 1e-12)
        if actual_malicious.any()
        else 0.0
    )

    table = probe.copy()
    table["round"] = int(round_id)
    table["client_samples"] = counts.astype(int)
    table["personalized_prediction_residual"] = personalized_residual.astype(float)
    table["round_residual_median"] = float(round_median)
    table["round_residual_robust_scale"] = float(round_scale)
    table["common_mode_relative_score"] = relative_score.astype(float)
    table["common_mode_relative_ema"] = ema_array.astype(float)
    table["clean_relative_ema_threshold"] = float(
        clean_relative_ema_threshold
    )
    table["relative_threshold_excess"] = threshold_excess.astype(float)
    table["trust_factor"] = trust.astype(float)
    table["low_trust"] = low_trust.astype(bool)
    table["base_aggregation_weight"] = base_weights.astype(float)
    table["aggregation_weight"] = normalized_weights.astype(float)
    table["actual_malicious"] = actual_malicious.astype(bool)

    malicious_mean_trust = (
        float(np.mean(trust[actual_malicious]))
        if actual_malicious.any()
        else 1.0
    )
    benign_mean_trust = (
        float(np.mean(trust[benign])) if benign.any() else 1.0
    )
    low_trust_malicious_recall = (
        float(np.mean(low_trust[actual_malicious]))
        if actual_malicious.any()
        else 0.0
    )
    low_trust_benign_rate = (
        float(np.mean(low_trust[benign])) if benign.any() else 0.0
    )

    summary: Dict[str, object] = {
        "mean_trust": float(np.mean(trust)),
        "minimum_trust": float(np.min(trust)),
        "mean_malicious_trust": malicious_mean_trust,
        "mean_benign_trust": benign_mean_trust,
        "low_trust_client_count": int(low_trust.sum()),
        "low_trust_malicious_recall": low_trust_malicious_recall,
        "low_trust_benign_rate": low_trust_benign_rate,
        "base_malicious_weight_share": base_malicious_share,
        "adjusted_malicious_weight_share": adjusted_malicious_share,
        "malicious_influence_reduction": float(influence_reduction),
        "round_residual_median": float(round_median),
        "round_residual_robust_scale": float(round_scale),
        "highest_relative_client": int(client_ids[int(np.argmax(ema_array))]),
        "highest_relative_ema": float(np.max(ema_array)),
        "dominant_probe_pair": str(
            table.iloc[int(np.argmax(prediction_drift))]["max_drift_pair"]
        ),
    }
    return CommonModeSoftTrustRoundResult(
        aggregated_state=aggregated_state,
        client_decisions=table,
        summary=summary,
    )

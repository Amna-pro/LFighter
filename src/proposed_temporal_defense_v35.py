"""Temporal, probe-aware robust aggregation for CIC IoT-DIAD, V3.5 stage 1.

This module is an independent proposed defense, not the original LFighter rule.
It combines four server-side signals that do not use malicious-client labels:

1. Full-model update magnitude relative to the client median.
2. Update direction distance from the coordinate-wise median update.
3. Maximum class-conditional off-diagonal prediction drift on a fixed validation probe.
4. Gradient-times-input attribution drift on the same fixed validation probe.

Signals are robustly normalized with median/MAD, accumulated through an exponential
moving suspicion score, and converted into soft aggregation weights. Persistently
suspicious clients are quarantined after a warm-up period, while a minimum-admission
safeguard prevents collapse under severe non-IID heterogeneity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

StateDict = Dict[str, torch.Tensor]


@dataclass
class TemporalDefenseMemory:
    suspicion_ema: Dict[int, float] = field(default_factory=dict)
    strike_count: Dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposedDefenseRoundResult:
    aggregated_state: StateDict
    client_decisions: pd.DataFrame
    signal_summary: pd.DataFrame
    summary: Dict[str, object]


def _flatten_update(
    reference_state: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor],
) -> np.ndarray:
    """Build a deterministic low-cost update sketch from the final representation layers.

    Using every model parameter made the robust median unnecessarily expensive on CPU.
    The penultimate projection and classifier are the layers most directly connected to
    class-decision changes, so V3.5 stage 1 freezes the sketch to those tensors.
    """
    floating_keys = [
        key for key, value in reference_state.items() if torch.is_floating_point(value)
    ]
    selected_keys = [
        key
        for key in floating_keys
        if key.startswith("penultimate.0.") or key.startswith("classifier.")
    ]
    if not selected_keys:
        selected_keys = floating_keys[-4:]
    pieces = []
    for key in selected_keys:
        reference = reference_state[key]
        delta = (
            local_state[key].detach().cpu().to(torch.float32)
            - reference.detach().cpu().to(torch.float32)
        )
        pieces.append(delta.reshape(-1).numpy())
    if not pieces:
        raise ValueError("No floating-point model parameters were found.")
    return np.concatenate(pieces).astype(np.float32, copy=False)


def _robust_positive_z(values: np.ndarray, clip: float = 8.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale < 1e-12:
        std = float(np.std(values))
        scale = std if std >= 1e-12 else 1.0
    z = (values - median) / scale
    return np.clip(np.maximum(z, 0.0), 0.0, clip)


def _cosine_distance_rows(matrix: np.ndarray, reference: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    row_norms = np.linalg.norm(matrix, axis=1)
    ref_norm = float(np.linalg.norm(reference))
    denominator = np.clip(row_norms * max(ref_norm, 1e-12), 1e-12, None)
    similarity = (matrix @ reference) / denominator
    return np.clip(1.0 - similarity, 0.0, 2.0)


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
                accumulator += state[key].detach().cpu().to(torch.float64) * float(weight)
            result[key] = accumulator.to(reference.dtype)
        else:
            result[key] = reference.detach().cpu().clone()
    return result


def _security_metrics(actual_malicious: np.ndarray, rejected: np.ndarray) -> Dict[str, float | int]:
    actual_malicious = actual_malicious.astype(bool)
    rejected = rejected.astype(bool)
    admitted = ~rejected
    benign = ~actual_malicious
    tp = int(np.sum(actual_malicious & rejected))
    fn = int(np.sum(actual_malicious & admitted))
    fp = int(np.sum(benign & rejected))
    tn = int(np.sum(benign & admitted))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "true_positive_malicious_rejected": tp,
        "false_negative_malicious_admitted": fn,
        "false_positive_benign_rejected": fp,
        "true_negative_benign_admitted": tn,
        "malicious_detection_precision": float(precision),
        "malicious_rejection_recall": float(recall),
        "malicious_detection_f1": float(f1),
        "benign_retention_rate": float(tn / max(tn + fp, 1)),
        "benign_false_rejection_rate": float(fp / max(tn + fp, 1)),
        "malicious_admission_rate": float(fn / max(tp + fn, 1)),
    }


def aggregate_proposed_temporal_defense(
    reference_state: Mapping[str, torch.Tensor],
    local_states: Sequence[Mapping[str, torch.Tensor]],
    client_ids: Sequence[int],
    sample_counts: Sequence[int],
    probe_signals: pd.DataFrame,
    memory: TemporalDefenseMemory,
    round_id: int,
    malicious_client_ids: Iterable[int] = (),
    ema_decay: float = 0.65,
    warmup_rounds: int = 2,
    quarantine_threshold: float = 2.25,
    strike_threshold: float = 1.50,
    required_strikes: int = 2,
    minimum_admitted_fraction: float = 0.60,
    soft_weight_temperature: float = 1.10,
    minimum_risk_weight: float = 0.05,
    weight_update_norm: float = 0.15,
    weight_update_direction: float = 0.20,
    weight_prediction_drift: float = 0.45,
    weight_explanation_drift: float = 0.20,
) -> ProposedDefenseRoundResult:
    if len(local_states) != len(client_ids) or len(local_states) != len(sample_counts):
        raise ValueError("local_states, client_ids, and sample_counts must align")
    if len(local_states) < 2:
        raise ValueError("At least two local states are required")

    client_ids = [int(value) for value in client_ids]
    probe = probe_signals.set_index("client_id").loc[client_ids].reset_index()
    updates = np.stack(
        [_flatten_update(reference_state, state) for state in local_states], axis=0
    )
    median_update = np.median(updates, axis=0)
    update_norm = np.linalg.norm(updates, axis=1)
    direction_distance = _cosine_distance_rows(updates, median_update)
    prediction_drift = probe["max_offdiag_probability_drift"].to_numpy(dtype=float)
    explanation_drift = probe["explanation_cosine_distance"].to_numpy(dtype=float)

    z_norm = _robust_positive_z(update_norm)
    z_direction = _robust_positive_z(direction_distance)
    z_prediction = _robust_positive_z(prediction_drift)
    z_explanation = _robust_positive_z(explanation_drift)

    coefficient_sum = (
        weight_update_norm
        + weight_update_direction
        + weight_prediction_drift
        + weight_explanation_drift
    )
    if coefficient_sum <= 0:
        raise ValueError("At least one signal weight must be positive")
    instantaneous = (
        weight_update_norm * z_norm
        + weight_update_direction * z_direction
        + weight_prediction_drift * z_prediction
        + weight_explanation_drift * z_explanation
    ) / coefficient_sum

    ema_values = []
    strikes = []
    for client_id, current in zip(client_ids, instantaneous):
        previous = float(memory.suspicion_ema.get(client_id, 0.0))
        ema = ema_decay * previous + (1.0 - ema_decay) * float(current)
        memory.suspicion_ema[client_id] = ema
        prior_strikes = int(memory.strike_count.get(client_id, 0))
        if round_id > warmup_rounds and float(current) >= strike_threshold:
            current_strikes = prior_strikes + 1
        else:
            current_strikes = max(prior_strikes - 1, 0)
        memory.strike_count[client_id] = current_strikes
        ema_values.append(ema)
        strikes.append(current_strikes)

    ema_values_array = np.asarray(ema_values, dtype=float)
    strikes_array = np.asarray(strikes, dtype=int)
    rejected = (
        (round_id > warmup_rounds)
        & (ema_values_array >= quarantine_threshold)
        & (strikes_array >= required_strikes)
    )

    minimum_admitted = max(2, int(np.ceil(minimum_admitted_fraction * len(client_ids))))
    if int((~rejected).sum()) < minimum_admitted:
        order = np.argsort(ema_values_array)
        rejected[:] = True
        rejected[order[:minimum_admitted]] = False

    counts = np.asarray(sample_counts, dtype=float)
    count_cap = float(np.median(counts) * 3.0)
    bounded_counts = np.minimum(counts, max(count_cap, 1.0))
    risk_factor = np.exp(
        -soft_weight_temperature * np.maximum(ema_values_array - 0.50, 0.0)
    )
    risk_factor = np.clip(risk_factor, minimum_risk_weight, 1.0)
    aggregation_weights = bounded_counts * risk_factor
    aggregation_weights[rejected] = 0.0
    if float(aggregation_weights.sum()) <= 0:
        aggregation_weights = (~rejected).astype(float)
    normalized_weights = aggregation_weights / float(aggregation_weights.sum())

    aggregated_state = _weighted_average_states(
        local_states, normalized_weights, reference_state
    )

    malicious_set = {int(value) for value in malicious_client_ids}
    actual_malicious = np.asarray([client_id in malicious_set for client_id in client_ids])
    admitted = ~rejected

    table = probe.copy()
    table["round"] = int(round_id)
    table["client_samples"] = counts.astype(int)
    table["update_l2"] = update_norm.astype(float)
    table["update_direction_distance"] = direction_distance.astype(float)
    table["z_update_l2"] = z_norm.astype(float)
    table["z_update_direction"] = z_direction.astype(float)
    table["z_prediction_drift"] = z_prediction.astype(float)
    table["z_explanation_drift"] = z_explanation.astype(float)
    table["instantaneous_suspicion"] = instantaneous.astype(float)
    table["temporal_suspicion_ema"] = ema_values_array.astype(float)
    table["strike_count"] = strikes_array.astype(int)
    table["admitted"] = admitted.astype(bool)
    table["rejected"] = rejected.astype(bool)
    table["aggregation_weight"] = normalized_weights.astype(float)
    table["actual_malicious"] = actual_malicious.astype(bool)
    table["correct_security_decision"] = (actual_malicious == rejected).astype(bool)

    signal_summary = pd.DataFrame(
        [
            {
                "signal": "update_l2",
                "median": float(np.median(update_norm)),
                "maximum": float(np.max(update_norm)),
            },
            {
                "signal": "update_direction_distance",
                "median": float(np.median(direction_distance)),
                "maximum": float(np.max(direction_distance)),
            },
            {
                "signal": "max_offdiag_probability_drift",
                "median": float(np.median(prediction_drift)),
                "maximum": float(np.max(prediction_drift)),
            },
            {
                "signal": "explanation_cosine_distance",
                "median": float(np.median(explanation_drift)),
                "maximum": float(np.max(explanation_drift)),
            },
        ]
    )
    signal_summary.insert(0, "round", int(round_id))

    security = _security_metrics(actual_malicious, rejected)
    summary: Dict[str, object] = {
        "admitted_client_count": int(admitted.sum()),
        "rejected_client_count": int(rejected.sum()),
        "admitted_clients": "|".join(str(client_ids[i]) for i in np.where(admitted)[0]),
        "rejected_clients": "|".join(str(client_ids[i]) for i in np.where(rejected)[0]),
        "mean_temporal_suspicion": float(np.mean(ema_values_array)),
        "max_temporal_suspicion": float(np.max(ema_values_array)),
        "highest_risk_client": int(client_ids[int(np.argmax(ema_values_array))]),
        "dominant_probe_pair": str(
            table.iloc[int(np.argmax(prediction_drift))]["max_drift_pair"]
        ),
        **security,
    }
    return ProposedDefenseRoundResult(
        aggregated_state=aggregated_state,
        client_decisions=table,
        signal_summary=signal_summary,
        summary=summary,
    )

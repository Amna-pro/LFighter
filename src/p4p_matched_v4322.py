"""Matched P4P comparator utilities for the BATR FL reviewer revision.

This module implements the defense logic specified in Khang et al. (2026)
and frozen by REVIEWER_PROTOCOL_AMENDMENT_C1_v4322.md.  It contains no
project-data access and no experiment orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import numpy as np
import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest

from trusted_update_reconstruction_v312 import (
    floating_update,
    update_cosine,
    update_norm,
    weighted_average_states,
)

State = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class P4PConfig:
    mad_k: float = 3.0
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    isolation_contamination: float = 0.1
    isolation_n_estimators: int = 100
    kmeans_clusters: int = 2
    ensemble_vote_threshold: int = 2
    suspicion_decay: float = 0.2
    suspicion_threshold: float = 2.0
    detector_random_state: int = 42
    kmeans_n_init: int = 10


@dataclass
class P4PRoundDecision:
    norms: np.ndarray
    magnitude_low: float
    magnitude_high: float
    magnitude_pass: np.ndarray
    probe_responses: np.ndarray
    kmeans_flag: np.ndarray
    dbscan_flag: np.ndarray
    isolation_flag: np.ndarray
    ensemble_votes: np.ndarray
    ensemble_anomaly: np.ndarray
    suspicion_scores: np.ndarray
    permanently_banned: np.ndarray
    rejected: np.ndarray
    trusted_ids: List[int]


def _validate_updates(updates: Sequence[Mapping[str, torch.Tensor]]) -> None:
    if not updates:
        raise ValueError("P4P requires at least one client update")
    keys = set(updates[0])
    if any(set(update) != keys for update in updates):
        raise ValueError("All P4P updates must have identical floating-state keys")


def global_probe(
    current_global_state: Mapping[str, torch.Tensor],
    previous_global_state: Mapping[str, torch.Tensor],
) -> State:
    """Return v_t = W_t - W_{t-1} over floating tensors only."""
    return floating_update(current_global_state, previous_global_state)


def mad_magnitude_filter(
    updates: Sequence[Mapping[str, torch.Tensor]],
    *,
    k: float = 3.0,
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Published P4P L2/MAD filter using median +/- k*MAD."""
    _validate_updates(updates)
    if k < 0:
        raise ValueError("MAD coefficient must be non-negative")
    norms = np.asarray([update_norm(update) for update in updates], dtype=np.float64)
    median_norm = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median_norm)))
    low = float(median_norm - float(k) * mad)
    high = float(median_norm + float(k) * mad)
    accepted = (norms >= low) & (norms <= high)
    return norms, low, high, accepted


def probe_responses(
    updates: Sequence[Mapping[str, torch.Tensor]],
    probe: Mapping[str, torch.Tensor],
) -> np.ndarray:
    """Server-side cosine response r_i^t for every client update."""
    _validate_updates(updates)
    if set(probe) != set(updates[0]):
        raise ValueError("Probe and client update keys do not match")
    return np.asarray([update_cosine(update, probe) for update in updates], dtype=np.float64)


def _kmeans_flags(values: np.ndarray, cfg: P4PConfig) -> np.ndarray:
    n = len(values)
    flags = np.zeros(n, dtype=bool)
    if n < cfg.kmeans_clusters or np.unique(values).size < cfg.kmeans_clusters:
        return flags
    model = KMeans(
        n_clusters=cfg.kmeans_clusters,
        random_state=cfg.detector_random_state,
        n_init=cfg.kmeans_n_init,
    )
    labels = model.fit_predict(values.reshape(-1, 1))
    unique = np.unique(labels)
    if unique.size < 2:
        return flags
    counts = {int(label): int(np.sum(labels == label)) for label in unique}
    minimum = min(counts.values())
    candidates = [label for label, count in counts.items() if count == minimum]
    if len(candidates) == 1:
        suspicious_label = candidates[0]
    else:
        means = {
            label: float(np.mean(values[labels == label]))
            for label in candidates
        }
        suspicious_label = min(candidates, key=lambda label: (means[label], label))
    return labels == suspicious_label


def _dbscan_flags(values: np.ndarray, cfg: P4PConfig) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    labels = DBSCAN(
        eps=cfg.dbscan_eps,
        min_samples=cfg.dbscan_min_samples,
    ).fit_predict(values.reshape(-1, 1))
    return labels == -1


def _isolation_flags(values: np.ndarray, cfg: P4PConfig) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros(len(values), dtype=bool)
    model = IsolationForest(
        contamination=cfg.isolation_contamination,
        n_estimators=cfg.isolation_n_estimators,
        random_state=cfg.detector_random_state,
    )
    labels = model.fit_predict(values.reshape(-1, 1))
    # sklearn semantic convention: -1 = outlier, +1 = inlier.
    return labels == -1


def ensemble_flags(
    all_responses: np.ndarray,
    magnitude_pass: np.ndarray,
    *,
    cfg: P4PConfig = P4PConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run P4P's 1D 3-detector ensemble on magnitude-passing clients.

    Returns full-client arrays: kmeans, dbscan, isolation, votes, anomaly.
    """
    responses = np.asarray(all_responses, dtype=np.float64)
    passed = np.asarray(magnitude_pass, dtype=bool)
    if responses.ndim != 1 or passed.ndim != 1 or len(responses) != len(passed):
        raise ValueError("Responses and magnitude mask must be same-length 1D arrays")

    n = len(responses)
    kflag = np.zeros(n, dtype=bool)
    dflag = np.zeros(n, dtype=bool)
    iflag = np.zeros(n, dtype=bool)
    passed_ids = np.flatnonzero(passed)
    values = responses[passed_ids]
    if len(passed_ids):
        kflag[passed_ids] = _kmeans_flags(values, cfg)
        dflag[passed_ids] = _dbscan_flags(values, cfg)
        iflag[passed_ids] = _isolation_flags(values, cfg)
    votes = kflag.astype(np.int64) + dflag.astype(np.int64) + iflag.astype(np.int64)
    anomaly = votes >= cfg.ensemble_vote_threshold
    return kflag, dflag, iflag, votes, anomaly


def update_temporal_suspicion(
    previous_scores: Sequence[float],
    ensemble_anomaly: Sequence[bool],
    previous_banned: Sequence[bool],
    *,
    cfg: P4PConfig = P4PConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(previous_scores, dtype=np.float64)
    anomaly = np.asarray(ensemble_anomaly, dtype=bool)
    banned = np.asarray(previous_banned, dtype=bool)
    if scores.ndim != 1 or anomaly.shape != scores.shape or banned.shape != scores.shape:
        raise ValueError("Temporal P4P arrays must be same-length 1D arrays")
    updated = np.maximum(
        0.0,
        scores + anomaly.astype(np.float64) - cfg.suspicion_decay,
    )
    updated_banned = banned | (updated > cfg.suspicion_threshold)
    return updated, updated_banned


def decide_round(
    updates: Sequence[Mapping[str, torch.Tensor]],
    probe: Mapping[str, torch.Tensor],
    previous_scores: Sequence[float],
    previous_banned: Sequence[bool],
    *,
    cfg: P4PConfig = P4PConfig(),
) -> P4PRoundDecision:
    """Apply the complete matched P4P filtering/temporal decision."""
    norms, low, high, mag_pass = mad_magnitude_filter(updates, k=cfg.mad_k)
    responses = probe_responses(updates, probe)
    kflag, dflag, iflag, votes, anomaly = ensemble_flags(
        responses, mag_pass, cfg=cfg
    )
    scores, banned = update_temporal_suspicion(
        previous_scores, anomaly, previous_banned, cfg=cfg
    )
    rejected = (~mag_pass) | anomaly | banned
    trusted_ids = np.flatnonzero(~rejected).astype(int).tolist()
    return P4PRoundDecision(
        norms=norms,
        magnitude_low=low,
        magnitude_high=high,
        magnitude_pass=mag_pass,
        probe_responses=responses,
        kmeans_flag=kflag,
        dbscan_flag=dflag,
        isolation_flag=iflag,
        ensemble_votes=votes,
        ensemble_anomaly=anomaly,
        suspicion_scores=scores,
        permanently_banned=banned,
        rejected=rejected,
        trusted_ids=trusted_ids,
    )


def aggregate_trusted_states(
    local_states: Sequence[Mapping[str, torch.Tensor]],
    sample_counts: Sequence[int],
    reference_state: Mapping[str, torch.Tensor],
    trusted_ids: Sequence[int],
) -> State:
    """Sample-count FedAvg over the P4P trusted set, renormalized."""
    ids = [int(client_id) for client_id in trusted_ids]
    if not ids:
        raise RuntimeError("P4P trusted set is empty; refusing hidden fallback aggregation")
    if any(client_id < 0 or client_id >= len(local_states) for client_id in ids):
        raise IndexError("P4P trusted client id is out of range")
    states = [local_states[client_id] for client_id in ids]
    counts = [int(sample_counts[client_id]) for client_id in ids]
    return weighted_average_states(states, counts, reference_state)

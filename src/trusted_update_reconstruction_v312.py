"""Trusted update-reconstruction utilities for V3.12."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

State = Dict[str, torch.Tensor]


def floating_update(
    local_state: Mapping[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor],
) -> State:
    update: State = {}
    for key, reference in reference_state.items():
        if torch.is_floating_point(reference):
            update[key] = (
                local_state[key].detach().cpu().to(torch.float32)
                - reference.detach().cpu().to(torch.float32)
            )
    return update


def zero_update(reference_state: Mapping[str, torch.Tensor]) -> State:
    return {
        key: torch.zeros_like(value.detach().cpu(), dtype=torch.float32)
        for key, value in reference_state.items()
        if torch.is_floating_point(value)
    }


def clone_update(update: Mapping[str, torch.Tensor]) -> State:
    return {key: value.detach().cpu().clone() for key, value in update.items()}


def add_updates(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    right_scale: float = 1.0,
) -> State:
    if set(left) != set(right):
        raise ValueError("Update keys do not match")
    return {
        key: (
            left[key].detach().cpu().to(torch.float32)
            + float(right_scale) * right[key].detach().cpu().to(torch.float32)
        )
        for key in left
    }


def subtract_updates(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> State:
    return add_updates(left, right, right_scale=-1.0)


def scale_update(update: Mapping[str, torch.Tensor], scale: float) -> State:
    return {
        key: value.detach().cpu().to(torch.float32) * float(scale)
        for key, value in update.items()
    }


def coordinate_median(updates: Sequence[Mapping[str, torch.Tensor]]) -> State:
    if not updates:
        raise ValueError("At least one update is required")
    keys = set(updates[0])
    if any(set(update) != keys for update in updates):
        raise ValueError("Update keys do not match")
    result: State = {}
    for key in sorted(keys):
        stacked = torch.stack(
            [update[key].detach().cpu().to(torch.float32) for update in updates],
            dim=0,
        )
        result[key] = torch.median(stacked, dim=0).values
    return result


def update_norm(update: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for tensor in update.values():
        values = tensor.detach().cpu().to(torch.float64)
        total += float(torch.sum(values * values))
    return float(np.sqrt(max(total, 0.0)))


def update_dot(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    if set(left) != set(right):
        raise ValueError("Update keys do not match")
    total = 0.0
    for key in left:
        total += float(
            torch.sum(
                left[key].detach().cpu().to(torch.float64)
                * right[key].detach().cpu().to(torch.float64)
            )
        )
    return total


def update_cosine(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    left_norm = update_norm(left)
    right_norm = update_norm(right)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(update_dot(left, right) / (left_norm * right_norm))


def relative_l2_error(
    reconstructed: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
) -> float:
    difference = subtract_updates(reconstructed, actual)
    return float(update_norm(difference) / max(update_norm(actual), 1e-12))


def robust_upper_norm(norms: Iterable[float], multiplier: float) -> float:
    values = np.asarray(list(norms), dtype=np.float64)
    if len(values) == 0:
        raise ValueError("At least one norm is required")
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    if mad <= 1e-12:
        mad = float(np.std(values))
    if mad <= 1e-12:
        mad = max(median * 0.10, 1e-12)
    return float(median + float(multiplier) * mad)


def clip_update_norm(
    update: Mapping[str, torch.Tensor],
    upper_norm: float,
) -> Tuple[State, float]:
    norm = update_norm(update)
    if norm <= float(upper_norm) or norm <= 1e-12:
        return clone_update(update), 1.0
    factor = float(upper_norm) / norm
    return scale_update(update, factor), factor


def state_from_update(
    reference_state: Mapping[str, torch.Tensor],
    update: Mapping[str, torch.Tensor],
) -> State:
    result: State = {}
    for key, reference in reference_state.items():
        reference_cpu = reference.detach().cpu()
        if torch.is_floating_point(reference):
            result[key] = (
                reference_cpu.to(torch.float32)
                + update[key].detach().cpu().to(torch.float32)
            ).to(reference.dtype)
        else:
            result[key] = reference_cpu.clone()
    return result


def weighted_average_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    sample_counts: Sequence[int],
    reference_state: Mapping[str, torch.Tensor],
) -> State:
    counts = np.asarray(sample_counts, dtype=np.float64)
    if len(states) == 0 or len(states) != len(counts):
        raise ValueError("State and sample-count lengths must match")
    if float(counts.sum()) <= 0:
        raise ValueError("Total sample count must be positive")
    weights = counts / counts.sum()
    result: State = {}
    for key, reference in reference_state.items():
        if torch.is_floating_point(reference):
            accumulator = torch.zeros_like(reference, dtype=torch.float64)
            for state, weight in zip(states, weights):
                accumulator += (
                    state[key].detach().cpu().to(torch.float64) * float(weight)
                )
            result[key] = accumulator.to(reference.dtype)
        else:
            result[key] = reference.detach().cpu().clone()
    return result


def state_max_abs_difference(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    if set(left) != set(right):
        raise ValueError("State keys do not match")
    maximum = 0.0
    for key in left:
        left_tensor = left[key].detach().cpu()
        right_tensor = right[key].detach().cpu()
        if torch.is_floating_point(left_tensor):
            difference = torch.max(
                torch.abs(
                    left_tensor.to(torch.float64)
                    - right_tensor.to(torch.float64)
                )
            )
            maximum = max(maximum, float(difference))
        elif not torch.equal(left_tensor, right_tensor):
            return float("inf")
    return maximum


def reconstruct_update(
    current_center: Mapping[str, torch.Tensor],
    historical_residual: Mapping[str, torch.Tensor],
    selected_policy: str,
    current_update_scale: float,
    warmup_update_scale: float,
    trusted_norms: Sequence[float],
    scale_lower: float,
    scale_upper: float,
    norm_clip_multiplier: float,
) -> Tuple[State, Dict[str, float]]:
    if selected_policy == "center_only":
        reconstructed = clone_update(current_center)
        residual_scale = 0.0
    elif selected_policy == "center_plus_residual":
        reconstructed = add_updates(current_center, historical_residual)
        residual_scale = 1.0
    elif selected_policy in {
        "center_plus_scaled_residual",
        "center_plus_scaled_residual_normclip",
    }:
        residual_scale = float(
            np.clip(
                float(current_update_scale)
                / max(float(warmup_update_scale), 1e-12),
                float(scale_lower),
                float(scale_upper),
            )
        )
        reconstructed = add_updates(
            current_center,
            historical_residual,
            right_scale=residual_scale,
        )
    else:
        raise ValueError(f"Unsupported reconstruction policy: {selected_policy}")

    norm_clip_factor = 1.0
    upper_norm = float("nan")
    if selected_policy == "center_plus_scaled_residual_normclip":
        upper_norm = robust_upper_norm(
            trusted_norms,
            multiplier=norm_clip_multiplier,
        )
        reconstructed, norm_clip_factor = clip_update_norm(
            reconstructed,
            upper_norm,
        )

    return reconstructed, {
        "residual_scale": float(residual_scale),
        "norm_clip_factor": float(norm_clip_factor),
        "norm_clip_upper": float(upper_norm),
        "reconstructed_update_norm": update_norm(reconstructed),
    }


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

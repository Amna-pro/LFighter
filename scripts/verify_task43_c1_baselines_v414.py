#!/usr/bin/env python3
"""Task 43 C1: robust-aggregation baseline implementation and unit tests.

Purpose
-------
Implements the two genuinely new aggregation rules Task 43 needs
(trimmed_mean, multi_krum) and verifies both against small, by-hand-
computed synthetic cases BEFORE either is used on any real federated
update. coordinate_median is NOT reimplemented here -- it already exists
in trusted_update_reconstruction_v312.py and has been used correctly
throughout Tasks 40-42; this script imports and reuses it directly rather
than retesting something already battle-tested.

This script trains nothing and touches no project data. It is pure
arithmetic verification, matching the same caution applied to Task 42's
parameter-order alignment (verify the mechanism in isolation before
trusting it inside a larger pipeline).

Usage
-----
Run with no arguments. Exits 0 and prints "ALL UNIT TESTS PASSED" if
every hand-computed case matches, or raises immediately with the first
mismatch otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for search_path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from trusted_update_reconstruction_v312 import coordinate_median  # noqa: E402


def trimmed_mean(
    updates: List[Mapping[str, torch.Tensor]], trim_fraction: float
) -> Dict[str, torch.Tensor]:
    """Per-parameter, per-coordinate symmetric trimmed mean.

    For each parameter tensor, and independently for each coordinate
    within that tensor, sorts the values submitted by all clients, drops
    the lowest and highest floor(trim_fraction * n_clients) values, and
    averages what remains. This is the standard coordinate-wise trimmed
    mean used in the Byzantine-robust aggregation literature (distinct
    from trimming based on a scalar summary like update norm).
    """
    if not updates:
        raise ValueError("trimmed_mean requires at least one update.")
    n = len(updates)
    trim_count = int(trim_fraction * n)
    if 2 * trim_count >= n:
        raise ValueError(
            f"trim_fraction={trim_fraction} with n={n} clients would trim "
            f"{2 * trim_count} of {n} values, leaving none to average."
        )
    keys = list(updates[0].keys())
    result: Dict[str, torch.Tensor] = {}
    for key in keys:
        stacked = torch.stack([u[key].to(torch.float64) for u in updates], dim=0)
        sorted_values, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_values[trim_count : n - trim_count]
        result[key] = trimmed.mean(dim=0).to(updates[0][key].dtype)
    return result


def multi_krum(
    updates: List[Mapping[str, torch.Tensor]], assumed_byzantine_count: int
) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    """Multi-Krum: select and average the m = n - f - 2 client updates
    with the smallest sum of squared distances to their own (n - f - 2)
    nearest neighbors among all other clients.

    Returns the averaged result and the list of selected client indices
    (0-based, in the order updates was given), for auditability.
    """
    n = len(updates)
    f = assumed_byzantine_count
    m = n - f - 2
    if m < 1:
        raise ValueError(
            f"n={n}, f={f} gives m=n-f-2={m} < 1 -- no valid Multi-Krum "
            f"selection size."
        )
    keys = list(updates[0].keys())
    flat = [
        torch.cat([u[key].detach().cpu().reshape(-1).to(torch.float64) for key in keys])
        for u in updates
    ]
    distances_sq = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(torch.sum((flat[i] - flat[j]) ** 2))
            distances_sq[i, j] = d
            distances_sq[j, i] = d

    scores = torch.empty(n, dtype=torch.float64)
    for i in range(n):
        others = torch.cat([distances_sq[i, :i], distances_sq[i, i + 1 :]])
        nearest, _ = torch.sort(others)
        scores[i] = nearest[:m].sum()

    # Deterministic tie-break: stable sort by (score, index).
    order = sorted(range(n), key=lambda idx: (float(scores[idx]), idx))
    selected = sorted(order[:m])

    result: Dict[str, torch.Tensor] = {}
    for key in keys:
        stacked = torch.stack(
            [updates[i][key].to(torch.float64) for i in selected], dim=0
        )
        result[key] = stacked.mean(dim=0).to(updates[0][key].dtype)
    return result, selected


def make_update(value: float) -> Dict[str, torch.Tensor]:
    return {"w": torch.tensor([value], dtype=torch.float32)}


def run_unit_tests() -> None:
    print("=" * 100)
    print("TASK 43 C1: ROBUST-AGGREGATION BASELINE UNIT TESTS")
    print("=" * 100)

    # --- coordinate_median: reused, sanity-checked only (not re-verified
    # from scratch -- already tested throughout Tasks 40-42). ---
    updates = [make_update(v) for v in (1, 2, 3, 100, 101)]
    result = coordinate_median(updates)
    expected = 3.0
    observed = float(result["w"].item())
    assert observed == expected, (
        f"coordinate_median sanity check failed: expected {expected}, "
        f"got {observed}"
    )
    print(f"[PASS] coordinate_median([1,2,3,100,101]) = {observed} (expected {expected})")

    # --- trimmed_mean: hand-computed case ---
    # Values [1,2,3,100,101], trim_fraction=0.2 on n=5 -> trim_count=1
    # per side. Drop {1, 101}, average {2,3,100} = 35.0.
    updates = [make_update(v) for v in (1, 2, 3, 100, 101)]
    result = trimmed_mean(updates, trim_fraction=0.2)
    expected = 35.0
    observed = float(result["w"].item())
    assert observed == expected, (
        f"trimmed_mean failed: expected {expected}, got {observed}"
    )
    print(f"[PASS] trimmed_mean([1,2,3,100,101], trim=0.2) = {observed} (expected {expected})")

    # --- multi_krum: hand-computed case ---
    # n=6, f=2 -> m=n-f-2=2. Values [0,1,2,3,50,51].
    # By-hand pairwise-nearest-2 sum-of-squared-distances scores:
    #   client0(0): nearest2={1,2} -> 1^2+2^2=5
    #   client1(1): nearest2={0,2} -> 1^2+1^2=2
    #   client2(2): nearest2={1,3} -> 1^2+1^2=2
    #   client3(3): nearest2={2,1} -> 1^2+2^2=5
    #   client4(50): nearest2={51,3} -> 1^2+47^2=2210
    #   client5(51): nearest2={50,3} -> 1^2+48^2=2305
    # Two smallest scores: clients 1 and 2 (tie at 2, index tiebreak keeps
    # both since they are the only two at the minimum). Selected={1,2},
    # average of values {1,2} = 1.5.
    updates = [make_update(v) for v in (0, 1, 2, 3, 50, 51)]
    result, selected = multi_krum(updates, assumed_byzantine_count=2)
    expected_selected = [1, 2]
    expected_value = 1.5
    observed_value = float(result["w"].item())
    assert selected == expected_selected, (
        f"multi_krum selection failed: expected {expected_selected}, "
        f"got {selected}"
    )
    assert observed_value == expected_value, (
        f"multi_krum average failed: expected {expected_value}, got "
        f"{observed_value}"
    )
    print(
        f"[PASS] multi_krum([0,1,2,3,50,51], f=2) selected={selected}, "
        f"average={observed_value} (expected selected={expected_selected}, "
        f"average={expected_value})"
    )

    # --- multi_krum: outliers must never be selected, sanity check on a
    # larger, more realistic client count matching this project's n=20. ---
    honest_values = [1.0 + 0.01 * i for i in range(12)]  # tight honest cluster
    malicious_values = [500.0, -500.0, 800.0, -800.0, 1000.0, -1000.0, 1200.0, -1200.0]
    all_values = honest_values + malicious_values
    updates = [make_update(v) for v in all_values]
    result, selected = multi_krum(updates, assumed_byzantine_count=8)
    malicious_indices = set(range(12, 20))
    assert not (set(selected) & malicious_indices), (
        f"multi_krum selected a known-extreme outlier at n=20,f=8: "
        f"selected={selected}, malicious_indices={sorted(malicious_indices)}"
    )
    print(
        f"[PASS] multi_krum(n=20, f=8, 8 extreme outliers) selected only "
        f"honest indices: {selected}"
    )

    # --- trimmed_mean: verify malicious extremes are fully excluded at
    # the project's real Task 43 parameters (n=20, beta=0.4). ---
    updates = [make_update(v) for v in all_values]
    result = trimmed_mean(updates, trim_fraction=0.2)
    # trim_count = int(0.2*20) = 4 per side -> drops the 4 lowest and 4
    # highest raw VALUES (not indices) after sorting. With 8 malicious
    # extremes (4 very negative, 4 very positive) and 12 honest values
    # clustered near 1.0, dropping 4 lowest + 4 highest removes exactly
    # the 4 most negative and 4 most positive malicious values, leaving
    # {-500,-800 no wait -- sorted ascending: [-1200,-1000,-800,-500,
    # 1.0..1.11 (12 values), 500,800,1000,1200]. Drop 4 lowest
    # (-1200,-1000,-800,-500) and 4 highest (500,800,1000,1200) ->
    # remaining is exactly the 12 honest values.
    observed = float(result["w"].item())
    expected = sum(honest_values) / len(honest_values)
    # Tolerance matches float32 precision (~1.2e-7 relative), since
    # trimmed_mean casts back to the original tensor dtype (float32 for
    # real update tensors) -- not a logic bug, just float32's real
    # precision limit. A genuine selection error would be many orders of
    # magnitude larger than this and would still be caught.
    assert abs(observed - expected) < 1e-5, (
        f"trimmed_mean at n=20,beta=0.4 failed to fully exclude malicious "
        f"extremes: expected {expected}, got {observed}"
    )
    print(
        f"[PASS] trimmed_mean(n=20, trim=0.2, 8 extreme outliers) = "
        f"{observed:.6f} (expected {expected:.6f}, pure honest average)"
    )

    print()
    print("ALL UNIT TESTS PASSED")


if __name__ == "__main__":
    run_unit_tests()

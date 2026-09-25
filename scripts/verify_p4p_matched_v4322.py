#!/usr/bin/env python3
"""Pure synthetic known-answer tests for the matched P4P comparator.

No project data, checkpoints, result files, or test arrays are accessed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from p4p_matched_v4322 import (  # noqa: E402
    P4PConfig,
    aggregate_trusted_states,
    decide_round,
    ensemble_flags,
    global_probe,
    mad_magnitude_filter,
    probe_responses,
    update_temporal_suspicion,
)


def state(value: float):
    return {
        "w": torch.tensor([value], dtype=torch.float32),
        "count": torch.tensor(1, dtype=torch.int64),
    }


def update(value: float):
    return {"w": torch.tensor([value], dtype=torch.float32)}


def assert_close(observed: float, expected: float, tol: float = 1e-7):
    assert abs(observed - expected) <= tol, (observed, expected)


def main() -> int:
    print("=" * 88)
    print("P4P MATCHED COMPARATOR v4.32.2 SYNTHETIC VERIFICATION")
    print("=" * 88)

    # Probe v_t = W_t - W_{t-1}.
    probe = global_probe(state(5.0), state(3.0))
    assert_close(float(probe["w"].item()), 2.0)
    print("[PASS] historical global probe")

    # Cosine response direction.
    responses = probe_responses([update(2.0), update(-3.0)], probe)
    assert_close(float(responses[0]), 1.0)
    assert_close(float(responses[1]), -1.0)
    print("[PASS] cosine probe responses")

    # MAD known answer: norms [1,1,1,10,1] => median 1, MAD 0, only norm 1 passes.
    norms, low, high, passed = mad_magnitude_filter(
        [update(v) for v in [1.0, 1.0, 1.0, 10.0, 1.0]], k=3.0
    )
    assert_close(low, 1.0)
    assert_close(high, 1.0)
    assert passed.tolist() == [True, True, True, False, True]
    print("[PASS] MAD magnitude filtering")

    # KMeans equal-size deterministic tie rule: lower cosine cluster is suspicious.
    cfg = P4PConfig(dbscan_min_samples=50, isolation_contamination=0.25)
    vals = np.asarray([0.95, 0.90, -0.90, -0.95], dtype=np.float64)
    mag = np.ones(4, dtype=bool)
    kf, _, _, _, _ = ensemble_flags(vals, mag, cfg=cfg)
    assert kf.tolist() == [False, False, True, True]
    print("[PASS] KMeans equal-size lower-cosine tie rule")

    # Degenerate KMeans values contribute no KMeans vote.
    kf, _, _, _, _ = ensemble_flags(
        np.asarray([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64),
        np.ones(5, dtype=bool),
    )
    assert not bool(np.any(kf))
    print("[PASS] degenerate KMeans no-vote rule")

    # Temporal score: anomalous rounds add 1 then decay by 0.2 -> +0.8 each.
    scores = np.zeros(2, dtype=np.float64)
    banned = np.zeros(2, dtype=bool)
    for _ in range(3):
        scores, banned = update_temporal_suspicion(
            scores, [True, False], banned
        )
    assert_close(float(scores[0]), 2.4)
    assert banned.tolist() == [True, False]
    print("[PASS] temporal suspicion and permanent ban")

    # Trusted sample-count weighted FedAvg: clients 0 and 2, counts 1 and 3.
    reference = state(0.0)
    aggregated = aggregate_trusted_states(
        [state(1.0), state(100.0), state(3.0)],
        [1, 5, 3],
        reference,
        trusted_ids=[0, 2],
    )
    assert_close(float(aggregated["w"].item()), 2.5)
    assert int(aggregated["count"].item()) == 1
    print("[PASS] trusted-set sample-count FedAvg")

    # Empty trusted set must abort rather than substitute a fallback.
    try:
        aggregate_trusted_states([state(1.0)], [1], reference, trusted_ids=[])
    except RuntimeError:
        pass
    else:
        raise AssertionError("empty trusted set did not abort")
    print("[PASS] empty trusted-set abort")

    # End-to-end decision dimensions and magnitude rejection semantics.
    probe = update(1.0)
    decision = decide_round(
        [update(v) for v in [1.0, 1.1, 0.9, 1.0, 10.0, 1.05, 0.95, 1.02, 0.98, 1.01]],
        probe,
        previous_scores=np.zeros(10),
        previous_banned=np.zeros(10, dtype=bool),
    )
    assert len(decision.trusted_ids) <= 10
    assert bool(decision.rejected[4])
    print("[PASS] complete decision path")

    print()
    print("ALL P4P SYNTHETIC TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

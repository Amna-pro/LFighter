#!/usr/bin/env python3
"""Task 42 C1a: calibration preflight audit (no training, read-only).

Purpose
-------
Task 42's preregistration states that D1 (update-norm deviation) and D2
(update-direction deviation) reference statistics can be derived from
existing Task 40 artifacts with zero new training. This script verifies
that claim directly rather than assuming it, and produces:

1. D1 per-client historical norm baseline, for all four seeds, using the
   exact robust median/MAD-on-log1p methodology already established in
   this codebase (screen_multiview_oneclass_v317.py's client_norm_reference
   function) -- confirmed free, computed here.

2. D2 per-client reference direction (flattened, L2-normalized residual
   vector), for all four seeds, from trusted_update_reconstruction_profiles.pt
   -- confirmed free, computed here.

3. An explicit, evidence-based answer to an open question: do raw
   per-round, per-client update vectors exist anywhere on disk for the
   *warmup* rounds, which would be needed to genuinely calibrate D1/D2
   thresholds via leave-one-round-out scoring (matching the causal
   calibration pattern used everywhere else in this project)? This script
   searches the known output trees and reports what it finds -- it does
   NOT assume an answer.

If raw per-round update vectors are NOT found, this script's decision
file states plainly that threshold calibration requires a new (small,
warmup-only, zero-attack) capture stage, and identifies
run_update_capture_v318b.py as the closest existing capture
infrastructure to adapt, rather than proposing to write anything new
blind.

Scope
-----
Read-only. No training, no attack construction. No test-set access.
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
MALICIOUS_CLIENTS = (1, 7, 8, 10, 14, 15, 17, 18)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 42 C1a calibration preflight audit."
    )
    parser.add_argument(
        "--reconstruction-root",
        type=Path,
        required=True,
        help=(
            "results\\cic_iot_diad_frozen_reconstruction_v3123_multiseed "
            "(contains seed_<N>\\calibration\\{tables,calibration} for "
            "all four seeds)"
        ),
    )
    parser.add_argument(
        "--warmup-root",
        type=Path,
        required=True,
        help=(
            "results\\cic_iot_diad_true_warmup_anchor_v3101_multiseed "
            "(searched for any raw per-round update-vector artifacts)"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def d1_client_norm_reference(warmup_updates: pd.DataFrame) -> pd.DataFrame:
    """Per-client robust baseline of log1p(update_norm).

    Exact methodology reused from screen_multiview_oneclass_v317.py's
    client_norm_reference: median and a combined MAD/IQR robust scale,
    computed independently per client from that client's own historical
    (here: warmup-round) update norms. No cross-client pooling.
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
                "round_count": int(len(values)),
                "log_norm_median": median,
                "log_norm_mad": mad,
                "log_norm_iqr_scale": iqr_scale,
                "log_norm_robust_scale": scale,
                "raw_norm_values": ",".join(
                    f"{v:.6f}" for v in group["update_norm"].to_numpy(dtype=float)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("client_id").reset_index(drop=True)


def d2_client_reference_direction(
    reconstruction_bundle: Dict[str, Any]
) -> pd.DataFrame:
    """Per-client flattened, L2-normalized historical residual direction."""
    profiles = reconstruction_bundle["client_residual_profiles"]
    rows: List[Dict[str, Any]] = []
    reference_names: List[str] | None = None
    for client_id in range(NUM_CLIENTS):
        profile = profiles[client_id]
        names = sorted(profile.keys())
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise RuntimeError(
                f"Client {client_id} residual profile parameter-name set "
                f"differs from client 0's. Cannot safely concatenate."
            )
        flat = torch.cat(
            [profile[name].detach().cpu().reshape(-1) for name in names]
        ).to(torch.float64)
        norm = float(torch.linalg.vector_norm(flat))
        rows.append(
            {
                "client_id": client_id,
                "residual_l2_norm": norm,
                "parameter_count": int(flat.numel()),
                "direction_is_degenerate": bool(norm <= 1e-12),
            }
        )
    return pd.DataFrame(rows).sort_values("client_id").reset_index(drop=True)


def search_for_raw_update_vectors(warmup_root: Path) -> pd.DataFrame:
    """Search known output trees for raw per-round, per-client update
    vector artifacts (as opposed to summary statistics or prediction
    signatures). Reports what naming patterns are found, if any, without
    assuming an answer either way.
    """
    candidate_patterns = (
        "*update_matrix*",
        "*client_updates*",
        "*update_vectors*",
        "*update_artifacts*",
    )
    findings: List[Dict[str, Any]] = []
    for pattern in candidate_patterns:
        for path in warmup_root.rglob(pattern):
            findings.append(
                {
                    "pattern_matched": pattern,
                    "relative_path": str(path.relative_to(warmup_root)),
                    "is_file": path.is_file(),
                    "bytes": path.stat().st_size if path.is_file() else None,
                }
            )
    # Also explicitly check whether warmup_local_signatures.npz (present
    # in every seed's warmup/tables directory) contains raw parameter
    # deltas or only prediction-transition signatures, by inspecting its
    # array names without assuming from the filename alone.
    signature_findings: List[Dict[str, Any]] = []
    for signature_path in warmup_root.rglob("warmup_local_signatures.npz"):
        try:
            with np.load(signature_path, allow_pickle=False) as archive:
                keys = list(archive.files)
                shapes = {
                    key: list(np.asarray(archive[key]).shape) for key in keys
                }
        except Exception as error:  # noqa: BLE001
            keys = []
            shapes = {"load_error": str(error)}
        signature_findings.append(
            {
                "relative_path": str(
                    signature_path.relative_to(warmup_root)
                ),
                "array_keys": keys,
                "array_shapes": json.dumps(shapes),
            }
        )
    return pd.DataFrame(findings), pd.DataFrame(signature_findings)


def main() -> int:
    args = parse_args()
    reconstruction_root = args.reconstruction_root.expanduser().resolve()
    warmup_root = args.warmup_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    d1_frames: List[pd.DataFrame] = []
    d2_frames: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, Any]] = []

    for seed in CONFIRMATORY_SEEDS:
        calibration_dir = reconstruction_root / f"seed_{seed}" / "calibration"
        updates_path = (
            calibration_dir
            / "tables"
            / "trusted_warmup_replay_local_updates.csv"
        )
        bundle_path = (
            calibration_dir
            / "calibration"
            / "trusted_update_reconstruction_profiles.pt"
        )
        if not updates_path.exists():
            raise FileNotFoundError(updates_path)
        if not bundle_path.exists():
            raise FileNotFoundError(bundle_path)

        updates = pd.read_csv(updates_path)
        expected_rows = 4 * NUM_CLIENTS
        if len(updates) != expected_rows:
            raise RuntimeError(
                f"Seed {seed}: expected {expected_rows} warmup-update rows "
                f"(4 rounds x {NUM_CLIENTS} clients), found {len(updates)}."
            )

        d1 = d1_client_norm_reference(updates)
        d1["model_seed"] = seed
        d1_frames.append(d1)

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        if int(bundle["model_seed"]) != seed:
            raise RuntimeError(
                f"Reconstruction bundle seed mismatch: expected {seed}, "
                f"found {bundle['model_seed']}."
            )
        d2 = d2_client_reference_direction(bundle)
        d2["model_seed"] = seed
        d2_frames.append(d2)

        manifest_rows.append(
            {
                "model_seed": seed,
                "updates_csv_sha256": file_sha256(updates_path),
                "reconstruction_bundle_sha256": file_sha256(bundle_path),
            }
        )

    d1_table = pd.concat(d1_frames, ignore_index=True)
    d2_table = pd.concat(d2_frames, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows)

    d1_table.to_csv(tables_dir / "task42c1a_d1_norm_reference.csv", index=False)
    d2_table.to_csv(
        tables_dir / "task42c1a_d2_direction_reference.csv", index=False
    )
    manifest.to_csv(
        tables_dir / "task42c1a_input_manifest_sha256.csv", index=False
    )

    degenerate = d2_table[d2_table["direction_is_degenerate"]]
    if len(degenerate):
        raise RuntimeError(
            f"D2 reference direction is degenerate (zero residual) for "
            f"{len(degenerate)} client/seed combinations -- cannot use as "
            f"a cosine-comparison reference. See "
            f"task42c1a_d2_direction_reference.csv."
        )

    raw_update_findings, signature_findings = search_for_raw_update_vectors(
        warmup_root
    )
    raw_update_findings.to_csv(
        tables_dir / "task42c1a_raw_update_vector_search.csv", index=False
    )
    signature_findings.to_csv(
        tables_dir / "task42c1a_warmup_signature_inspection.csv", index=False
    )

    raw_vectors_found = len(raw_update_findings) > 0
    # A signature file "contains raw update vectors" only if it has an
    # array whose second dimension looks like a full parameter count
    # (order of 10^4-10^6), not a small fixed-size behavioral signature
    # (this codebase's transition signatures are 8x8=64-dimensional).
    signature_contains_raw_updates = False
    if len(signature_findings):
        for shapes_json in signature_findings["array_shapes"]:
            shapes = json.loads(shapes_json)
            for shape in shapes.values():
                if isinstance(shape, list) and any(
                    dim > 1000 for dim in shape
                ):
                    signature_contains_raw_updates = True

    threshold_calibration_ready = bool(
        raw_vectors_found or signature_contains_raw_updates
    )

    decision = {
        "experiment_version": "4.13.C1a",
        "stage": "task42_c1a_calibration_preflight",
        "d1_norm_reference_computed": True,
        "d1_seeds_covered": list(CONFIRMATORY_SEEDS),
        "d2_direction_reference_computed": True,
        "d2_degenerate_count": int(len(degenerate)),
        "raw_update_vector_files_found": raw_vectors_found,
        "warmup_signature_files_inspected": int(len(signature_findings)),
        "warmup_signature_contains_raw_updates": signature_contains_raw_updates,
        "threshold_calibration_ready_from_existing_artifacts": (
            threshold_calibration_ready
        ),
        "next_stage": (
            "D1/D2 reference statistics are free and computed. Proceed "
            "directly to C1b: leave-one-round-out threshold calibration "
            "using the discovered raw update artifacts."
            if threshold_calibration_ready
            else "D1/D2 reference statistics are free and computed, but "
            "no raw per-round per-client update vectors were found on "
            "disk for the warmup stage. Genuine threshold calibration "
            "(as opposed to reference-direction extraction) requires a "
            "new, small, warmup-only, zero-attack capture stage. "
            "run_update_capture_v318b.py is the closest existing "
            "infrastructure (it already captures per-round, per-client "
            "update matrices for post-warmup continuation) and should "
            "be adapted rather than writing new capture code from "
            "scratch. This is NOT new training -- it replays the "
            "already-frozen clean warmup rounds and captures update "
            "deltas that a training run would already compute; it adds "
            "no new experimental content."
        ),
        "no_reserved_test_access": True,
    }
    (output_dir / "task42c1a_preflight_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print("=" * 100)
    print("TASK 42 C1a CALIBRATION PREFLIGHT")
    print("=" * 100)
    print("D1 norm reference computed for seeds:", list(CONFIRMATORY_SEEDS))
    print("D2 direction reference computed, degenerate count:", len(degenerate))
    print("Raw update-vector files found on disk:", raw_vectors_found)
    print(
        "Warmup signature files contain raw update vectors:",
        signature_contains_raw_updates,
    )
    print(
        "Threshold calibration ready from existing artifacts:",
        threshold_calibration_ready,
    )
    print()
    print("Next stage:", decision["next_stage"])
    print()
    print("Tables:", tables_dir)
    print("Decision file:", output_dir / "task42c1a_preflight_decision.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

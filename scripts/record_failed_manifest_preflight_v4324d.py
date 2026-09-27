#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTIAL = ROOT / "results" / "reviewer_task65_attack_manifest_recovery_v4324" / "all_to_one_benign" / "seed_1379954285" / "plain_fedavg"
EVID = ROOT / "reviewer_revision" / "TASK65_ATTACK_MANIFEST_RECOVERY_SEED_1379954285_ALL_TO_ONE_BENIGN_v4324.json"
OUT = ROOT / "reviewer_revision" / "TASK65_ATTACK_MANIFEST_RECOVERY_FAILED_PREFLIGHT_RECORD_v4324d.json"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def main() -> int:
    if EVID.exists():
        raise RuntimeError(f"PASS evidence unexpectedly exists; refusing cleanup: {EVID}")
    if not PARTIAL.exists():
        raise RuntimeError(f"Expected failed partial output is absent: {PARTIAL}")

    files = sorted(p for p in PARTIAL.rglob("*") if p.is_file())
    forbidden = []
    for p in files:
        low = p.name.lower()
        if p.suffix.lower() in {".pt", ".pth", ".ckpt"} or "checkpoint" in low or "model" in low:
            forbidden.append(str(p))
    if forbidden:
        raise RuntimeError(
            "Model/checkpoint-like files found in failed manifest-only output; refusing cleanup: "
            + repr(forbidden)
        )

    expected_rel = {
        "attack_manifest/attack_summary.csv",
        "attack_manifest/malicious_client_poison_manifest.csv",
        "attack_manifest/poisoned_indices.npz",
        "attack_manifest/poisoned_labels.npz",
        "exact_untargeted_plain_v320a3_metadata.json",
    }
    actual_rel = {p.relative_to(PARTIAL).as_posix() for p in files}
    unexpected = sorted(actual_rel - expected_rel)
    if unexpected:
        raise RuntimeError(f"Unexpected files in partial output; refusing cleanup: {unexpected}")

    record = {
        "protocol": "reviewer_v4324d_failed_manifest_preflight_preservation",
        "condition": {
            "attack_type": "all_to_one_benign",
            "model_seed": 1379954285,
            "attack_seed": 1379954285,
        },
        "failure_stage": "post_write_existing_exact_loader_dataframe_validation",
        "failure_cause": "csv_float_roundtrip_representation_in_derived_rate_columns",
        "scientific_parameter_change": False,
        "pass_evidence_existed_before_cleanup": False,
        "model_or_checkpoint_files_present": False,
        "local_model_training_run": False,
        "p4p_attack_outcome_run": False,
        "batr_attack_outcome_run": False,
        "files": {
            p.relative_to(PARTIAL).as_posix(): {
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
            for p in files
        },
        "status": "FAILED_PREFLIGHT_RECORDED_BEFORE_SAFE_CLEANUP",
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("FAILED PARTIAL PREFLIGHT RECORD = WRITTEN")
    print("FILES RECORDED:", len(files))
    print("PASS EVIDENCE EXISTS: False")
    print("MODEL OR CHECKPOINT FILES PRESENT: False")
    print("LOCAL MODEL TRAINING RUN: False")
    print("P4P ATTACK OUTCOME RUN: False")
    print("BATR ATTACK OUTCOME RUN: False")
    print("RECORD:", OUT)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

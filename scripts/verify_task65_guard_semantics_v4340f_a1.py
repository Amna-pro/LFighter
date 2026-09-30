#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "results" / "reviewer_v4340f_guard_access_log.jsonl"
OUTPUT = ROOT / "reviewer_revision" / "TASK65_GUARD_SEMANTIC_AUDIT_v4340f_a1.json"
EXPECTED_GUARD_SHA = "67909a8b30adf3e3332fd58a754364312df4b0b66830aecea41839edcff3c61c"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not GUARD.exists():
        raise FileNotFoundError(GUARD)

    observed = sha256_file(GUARD)
    if observed != EXPECTED_GUARD_SHA:
        raise RuntimeError(
            f"Guard log hash changed: expected {EXPECTED_GUARD_SHA}, observed {observed}"
        )

    records = [
        json.loads(line)
        for line in GUARD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    allowed = {"X_train", "y_train", "X_val", "y_val"}
    reserved = {
        "X_test_natural", "y_test_natural",
        "X_test_diagnostic", "y_test_diagnostic",
    }
    loads = [r for r in records if r.get("event") == "guarded_load_protocol_arrays"]
    all_materialized = [key for r in records for key in r.get("materialized_keys", [])]

    checks = {
        "guard_log_sha256_exact": observed == EXPECTED_GUARD_SHA,
        "guarded_load_events_present": bool(loads),
        "all_explicit_test_materialization_flags_false": all(
            r.get("test_arrays_materialized") is False
            for r in records
            if "test_arrays_materialized" in r
        ),
        "reserved_test_keys_absent_from_materialized_keys": not any(
            k in reserved for k in all_materialized
        ),
        "guarded_load_materialized_keys_train_val_only": all(
            set(r.get("materialized_keys", [])).issubset(allowed)
            for r in loads
        ),
        "reserved_names_are_verification_only": all(
            set(r.get("test_array_names_verified_only", [])) == reserved
            for r in loads
        ),
    }

    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError("Guard semantic audit failed: " + ", ".join(failed))

    result = {
        "protocol": "Task65 guard semantic audit v4340f-a1",
        "status": "PASS",
        "guard_log": str(GUARD),
        "guard_log_sha256": observed,
        "record_count": len(records),
        "guarded_load_event_count": len(loads),
        "materialized_keys_observed": sorted(set(all_materialized)),
        "reserved_test_names": sorted(reserved),
        "test_arrays_materialized": False,
        "checks": checks,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("TASK65 GUARD SEMANTIC AUDIT = PASS")
    print("GUARD LOG SHA256:", observed.upper())
    print("GUARDED LOAD EVENTS:", len(loads))
    print("MATERIALIZED KEYS:", sorted(set(all_materialized)))
    print("TEST ARRAYS MATERIALIZED: False")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

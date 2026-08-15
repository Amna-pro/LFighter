from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/task65_c2_postaccess_audit_v4283.json").read_text(encoding="utf-8"))
SRC = ROOT / CFG["source_output_root"]
TRAIN = SRC / "training"
FINAL = SRC / "final"
OUT = ROOT / "results/cic_iot_diad_task65_c2_postaccess_audit_v4283"

ACCESS = SRC / "TASK65_FINAL_ACCESS_STARTED.json"
COMPLETE = SRC / "TASK65_FINAL_EVALUATION_COMPLETE.json"
TRAINING_COMPLETE = TRAIN / "TASK65_TRAINING_COMPLETE.json"
GUARD_LOG = TRAIN / "task65_guard_access_log.jsonl"

def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def native_bool(value) -> bool:
    return bool(value)

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = {}

    # Provenance and marker checks only.
    frozen_commit = git("rev-list", "-n", "1", CFG["parent_tag"])
    head = git("rev-parse", "HEAD")
    checks["frozen_parent_tag_resolves"] = bool(frozen_commit)
    checks["head_descends_from_frozen_c1"] = subprocess.run(
        ["git", "merge-base", "--is-ancestor", frozen_commit, head], cwd=ROOT
    ).returncode == 0

    checks["access_marker_exists"] = ACCESS.exists()
    checks["completion_marker_exists"] = COMPLETE.exists()
    checks["training_complete_marker_exists"] = TRAINING_COMPLETE.exists()
    checks["guard_log_exists"] = GUARD_LOG.exists()

    if not all([ACCESS.exists(), COMPLETE.exists(), TRAINING_COMPLETE.exists(), GUARD_LOG.exists()]):
        raise RuntimeError("Required frozen C1 markers/logs are missing; audit cannot proceed.")

    access = json.loads(ACCESS.read_text(encoding="utf-8"))
    completion = json.loads(COMPLETE.read_text(encoding="utf-8"))
    training_complete = json.loads(TRAINING_COMPLETE.read_text(encoding="utf-8"))

    checks["access_head_matches_frozen_c1"] = access.get("head") == frozen_commit
    checks["completion_head_matches_frozen_c1"] = completion.get("head") == frozen_commit
    checks["training_head_matches_frozen_c1"] = training_complete.get("head") == frozen_commit
    checks["training_reports_no_reserved_test_materialization"] = (
        training_complete.get("reserved_test_arrays_materialized_during_training") is False
    )
    checks["completion_reports_reserved_test_materialized"] = (
        completion.get("reserved_test_arrays_materialized") is True
    )
    checks["negative_results_retained"] = completion.get("negative_results_retained") is True
    checks["best_round_selection_unused"] = completion.get("best_round_selection_used") is False
    checks["post_outcome_retuning_unused"] = completion.get("post_outcome_retuning_used") is False

    # Guard log must never report test-array materialization during training.
    guard_events = []
    forbidden_guard_events = []
    for line in GUARD_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        guard_events.append(event)
        if event.get("test_arrays_materialized") is True:
            forbidden_guard_events.append(event)
    checks["guard_log_nonempty"] = len(guard_events) > 0
    checks["guard_log_no_training_test_materialization"] = len(forbidden_guard_events) == 0

    required_files = {
        "raw_checkpoint_metrics.csv": FINAL / "raw_checkpoint_metrics.csv",
        "per_class_metrics.csv": FINAL / "per_class_metrics.csv",
        "confusion_matrix_long.csv": FINAL / "confusion_matrix_long.csv",
        "paired_primary_metrics.csv": FINAL / "paired_primary_metrics.csv",
    }
    for name, path in required_files.items():
        checks[f"{name}_exists"] = path.exists()
    if not all(path.exists() for path in required_files.values()):
        raise RuntimeError("One or more frozen C1 result CSV files are missing.")

    # Read existing result tables only. No model inference and no NPZ access.
    metrics = pd.read_csv(required_files["raw_checkpoint_metrics.csv"])
    per_class = pd.read_csv(required_files["per_class_metrics.csv"])
    confusion = pd.read_csv(required_files["confusion_matrix_long.csv"])
    paired = pd.read_csv(required_files["paired_primary_metrics.csv"])

    exp = CFG["expected_rows"]
    checks["checkpoint_metric_rows_exact"] = len(metrics) == exp["raw_checkpoint_metrics.csv"]
    checks["per_class_rows_exact"] = len(per_class) == exp["per_class_metrics.csv"]
    checks["confusion_rows_exact"] = len(confusion) == exp["confusion_matrix_long.csv"]
    checks["paired_primary_rows_exact"] = len(paired) == exp["paired_primary_metrics.csv"]

    checks["seeds_exact"] = set(metrics["seed"].astype(int)) == set(CFG["final_seeds"])
    checks["datasets_exact"] = set(metrics["dataset"].astype(str)) == set(CFG["datasets"])
    checks["rounds_exact"] = set(metrics["global_round"].astype(int)) == set(CFG["rounds"])
    checks["arms_exact"] = set(metrics["arm"].astype(str)) == {
        "clean_reference", "plain_fedavg", "trusted_reconstruction"
    }
    attack_metrics = metrics[metrics["arm"] != "clean_reference"]
    checks["attacks_complete"] = set(attack_metrics["attack"].astype(str)) == set(CFG["attacks"])
    checks["all_eight_classes_present"] = per_class["class_name"].nunique() == 8
    checks["primary_endpoint_rows_exact"] = int(paired["primary_endpoint"].astype(bool).sum()) == exp["primary_endpoint_rows"]
    checks["detector_metrics_complete"] = paired[
        ["malicious_client_recall", "benign_client_fpr", "reconstructed_client_round_count"]
    ].notna().all().all()

    # Verify the hashes recorded by the already-completed final evaluator.
    recorded_hashes = completion.get("output_sha256", {})
    hash_results = {}
    for name, expected_hash in recorded_hashes.items():
        path = FINAL / name
        hash_results[name] = path.exists() and sha256_file(path) == expected_hash
    checks["completion_output_hashes_present"] = len(recorded_hashes) >= 4
    checks["completion_output_hashes_match"] = len(recorded_hashes) >= 4 and all(hash_results.values())

    # Cardinalities recorded by the evaluator must agree with files.
    checks["completion_checkpoint_rows_match"] = int(completion.get("raw_checkpoint_metric_rows", -1)) == len(metrics)
    checks["completion_per_class_rows_match"] = int(completion.get("per_class_rows", -1)) == len(per_class)
    checks["completion_confusion_rows_match"] = int(completion.get("confusion_rows", -1)) == len(confusion)
    checks["completion_paired_rows_match"] = int(completion.get("paired_primary_rows", -1)) == len(paired)

    # Convert every value explicitly to native bool to fix the C1 serialization-only defect.
    checks = {key: native_bool(value) for key, value in checks.items()}
    passed = sum(checks.values())
    total = len(checks)
    ok = passed == total

    decision = {
        "task": 65,
        "phase": CFG["phase"],
        "protocol_id": CFG["protocol_id"],
        "source_protocol_id": CFG["source_protocol_id"],
        "parent_frozen_c1_commit": frozen_commit,
        "head_at_audit": head,
        "checks": checks,
        "checks_passed": passed,
        "checks_total": total,
        "all_checks_passed": ok,
        "scientific_results_recomputed": False,
        "reserved_test_arrays_opened_by_c2": False,
        "training_rerun_by_c2": False,
        "final_evaluator_rerun_by_c2": False,
        "original_c1_audit_failure_retained": True,
        "ready_for_task66_statistics": ok,
        "verified_output_hashes": hash_results,
    }
    out = OUT / "task65c2_postaccess_integrity_audit_v4283.json"
    out.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("===== TASK 65 C2 POST-ACCESS AUDIT V4.28.3 =====")
    for key, value in checks.items():
        print(f"{key}: {value}")
    print(f"PASS: {passed}/{total}")
    print("SCIENTIFIC RESULTS RECOMPUTED: False")
    print("RESERVED TEST ARRAYS OPENED BY C2: False")
    print("TRAINING RERUN BY C2: False")
    print("FINAL EVALUATOR RERUN BY C2: False")
    print("ORIGINAL C1 AUDIT FAILURE RETAINED: True")
    print("READY FOR TASK 66 STATISTICS:", ok)
    return 0 if ok else 2

if __name__ == "__main__":
    raise SystemExit(main())

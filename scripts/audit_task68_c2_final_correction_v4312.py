from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"configs/task68_c2_correction_v4312.json").read_text(encoding="utf-8"))
SRC=ROOT/CFG["source_output_root"]
OUT=ROOT/CFG["corrected_output_root"]
T65=ROOT/CFG["task65_root"]

def git(*args):
    return subprocess.run(["git",*args],cwd=ROOT,check=True,text=True,capture_output=True).stdout.strip()

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def parse_status_paths(status: str):
    paths=[]
    for line in status.splitlines():
        if not line.strip():
            continue
        # porcelain v1: XY<space>path
        p=line[3:].strip().replace("\\","/")
        if " -> " in p:
            p=p.split(" -> ",1)[1]
        paths.append(p)
    return paths

def main():
    if OUT.exists():
        raise RuntimeError("Task68 C2 V4.31.2 correction output already exists; rerun prohibited.")

    head=git("rev-parse","HEAD")
    parent=git("rev-list","-n","1",CFG["parent_tag"])
    if head!=parent:
        raise RuntimeError("Task68 C2 correction must run from the frozen Task68 C0 commit.")

    # Package installation necessarily creates four untracked correction files.
    status=git("status","--porcelain")
    seen=set(parse_status_paths(status))
    allowed=set(CFG["known_installed_files"])
    if seen != allowed:
        raise RuntimeError(
            "Unexpected repository dirt before Task68 C2 correction. "
            f"Expected only {sorted(allowed)}, observed {sorted(seen)}"
        )

    original_path=SRC/"TASK68_FINAL_AUDIT_COMPLETE.json"
    if not original_path.exists():
        raise RuntimeError("Original Task68 C1 decision is missing.")
    original=json.loads(original_path.read_text(encoding="utf-8"))
    checks0={k:bool(v) for k,v in original.get("checks",{}).items()}
    failed=[k for k,v in checks0.items() if not v]
    expected_failed=CFG["original_failed_checks"]

    checks={}
    checks["parent_tag_resolves"]=bool(parent)
    checks["head_equals_frozen_task68_c0"]=head==parent
    checks["only_expected_v4312_package_files_untracked"]=seen==allowed
    checks["original_c1_decision_exists"]=original_path.exists()
    checks["original_c1_was_72_of_74"]=(
        original.get("all_checks_passed") is False
        and int(original.get("checks_passed",-1))==72
        and int(original.get("checks_total",-1))==74
    )
    checks["original_c1_exact_failed_check_set"]=set(failed)==set(expected_failed)
    checks["original_c1_other_72_checks_all_true"]=len(checks0)==74 and sum(checks0.values())==72

    completion=json.loads((T65/"TASK65_FINAL_EVALUATION_COMPLETE.json").read_text(encoding="utf-8"))
    recorded=completion.get("output_sha256",{})
    hash_rows=[]
    hash_ok=bool(recorded) and len(recorded)>=4
    for name,expected in recorded.items():
        p=T65/"final"/name
        good=p.exists() and sha256(p)==expected
        hash_ok &= good
        hash_rows.append({
            "name":name,
            "resolved_path":str((Path("final")/name).as_posix()),
            "expected_sha256":expected,
            "observed_sha256":sha256(p) if p.exists() else None,
            "match":bool(good)
        })
    checks["corrected_task65_completion_hashes_match"]=bool(hash_ok)
    checks["task65_completion_hash_count_at_least_4"]=len(recorded)>=4

    training=json.loads((T65/"training/TASK65_TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
    checks["training_marker_has_actual_materialization_field"]="reserved_test_arrays_materialized_during_training" in training
    checks["corrected_task65_training_reports_no_reserved_test_materialization"]=(
        training.get("reserved_test_arrays_materialized_during_training") is False
    )

    t65audit=ROOT/"results/cic_iot_diad_task65_c2_postaccess_audit_v4283/task65c2_postaccess_integrity_audit_v4283.json"
    a65=json.loads(t65audit.read_text(encoding="utf-8"))
    checks["frozen_task65_c2_audit_was_38_of_38"]=(
        a65.get("all_checks_passed") is True
        and a65.get("checks_passed")==38
        and a65.get("checks_total")==38
    )
    checks["frozen_task65_c2_verified_training_no_test_materialization"]=(
        a65.get("checks",{}).get("training_reports_no_reserved_test_materialization") is True
    )
    checks["frozen_task65_c2_verified_completion_hashes"]=(
        a65.get("checks",{}).get("completion_output_hashes_match") is True
    )

    manifest=SRC/"task68_frozen_artifact_manifest_v4310.csv"
    trace=SRC/"task68_claim_to_evidence_trace_v4310.csv"
    checks["original_artifact_manifest_exists"]=manifest.exists()
    checks["original_claim_trace_exists"]=trace.exists()
    checks["claim_trace_has_8_rows"]=trace.exists() and len(pd.read_csv(trace))==8

    corrected=dict(checks0)
    corrected["task65_completion_hashes_match"]=checks["corrected_task65_completion_hashes_match"]
    corrected["task65_training_reports_no_reserved_test_materialization"]=checks["corrected_task65_training_reports_no_reserved_test_materialization"]
    corrected={k:bool(v) for k,v in corrected.items()}
    corrected_passed=sum(corrected.values())
    corrected_total=len(corrected)
    checks["corrected_final_check_set_still_74"]=corrected_total==74
    checks["corrected_final_audit_74_of_74"]=corrected_passed==74 and corrected_total==74

    checks={k:bool(v) for k,v in checks.items()}
    passed=sum(checks.values()); total=len(checks)
    ok=passed==total and corrected_passed==74

    OUT.mkdir(parents=True)
    pd.DataFrame(hash_rows).to_csv(OUT/"task68c2_task65_hash_resolution_v4312.csv",index=False)
    record={
        "task":68,
        "phase":CFG["phase"],
        "protocol_id":CFG["protocol_id"],
        "source_protocol_id":CFG["source_protocol_id"],
        "parent_frozen_c0_commit":parent,
        "head_at_correction":head,
        "original_c1_sha256":sha256(original_path),
        "original_c1_checks_passed":72,
        "original_c1_checks_total":74,
        "original_c1_failed_checks":failed,
        "original_c1_failure_retained":True,
        "v4311_execution_guard_failure_retained":True,
        "corrections":CFG["corrections"],
        "correction_checks":checks,
        "correction_checks_passed":passed,
        "correction_checks_total":total,
        "corrected_final_checks":corrected,
        "corrected_final_checks_passed":corrected_passed,
        "corrected_final_checks_total":corrected_total,
        "all_corrected_final_checks_passed":bool(ok),
        "scientific_results_recomputed":False,
        "training_rerun":False,
        "models_loaded":False,
        "reserved_npz_opened":False,
        "task65_rerun":False,
        "task66_inference_rerun":False,
        "task67_outputs_regenerated":False,
        "shap_recomputed":False,
        "llm_used":False,
        "ready_for_literature_review_and_manuscript_planning":bool(ok),
        "paper_drafting_authorized_only_after_50_plus_paper_review":bool(ok)
    }
    (OUT/"TASK68_C2_FINAL_AUDIT_CORRECTED_COMPLETE.json").write_text(
        json.dumps(record,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )

    print("===== TASK 68 C2 FINAL AUDIT CORRECTION V4.31.2 =====")
    for k,v in checks.items(): print(f"{k}: {v}")
    print(f"CORRECTION AUDIT PASS: {passed}/{total}")
    print(f"CORRECTED FINAL AUDIT: {corrected_passed}/{corrected_total}")
    print("ORIGINAL C1 72/74 FAILURE RETAINED: True")
    print("V4.31.1 EXECUTION-GUARD FAILURE RETAINED: True")
    print("SCIENTIFIC RESULTS RECOMPUTED: False")
    print("TRAINING RERUN: False")
    print("MODELS LOADED: False")
    print("RESERVED NPZ OPENED: False")
    print("TASK65 RERUN: False")
    print("TASK66 INFERENCE RERUN: False")
    print("TASK67 OUTPUTS REGENERATED: False")
    print("READY FOR LITERATURE REVIEW AND MANUSCRIPT PLANNING:",bool(ok))
    print("PAPER DRAFTING REQUIRES 50+ PAPER REVIEW FIRST:",bool(ok))
    print("DO NOT RERUN TASK68 C1.")
    return 0 if ok else 2

if __name__=="__main__":
    raise SystemExit(main())

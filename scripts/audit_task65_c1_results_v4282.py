from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"configs/task65_c1_preregistration_v4282.json").read_text(encoding="utf-8"))
OUT=ROOT/CFG["output_root"]; FINAL=OUT/"final"
def sha(p): h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def main():
    checks={}
    access=OUT/CFG["test_access_boundary"]["final_access_marker"]; complete=OUT/CFG["test_access_boundary"]["final_completion_marker"]
    checks["access_marker_exists"]=access.exists(); checks["completion_marker_exists"]=complete.exists()
    m=pd.read_csv(FINAL/"raw_checkpoint_metrics.csv"); pc=pd.read_csv(FINAL/"per_class_metrics.csv")
    cm=pd.read_csv(FINAL/"confusion_matrix_long.csv"); pp=pd.read_csv(FINAL/"paired_primary_metrics.csv")
    checks["checkpoint_metric_rows_exact"]=len(m)==440; checks["per_class_rows_exact"]=len(pc)==3520
    checks["confusion_rows_exact"]=len(cm)==28160; checks["paired_primary_rows_exact"]=len(pp)==200
    checks["seeds_exact"]=set(m.seed.astype(int))==set(CFG["final_seeds"])
    checks["datasets_exact"]=set(m.dataset)==set(CFG["final_datasets"]); checks["rounds_exact"]=set(m.global_round.astype(int))==set(CFG["monitored_rounds"])
    checks["arms_exact"]=set(m.arm)=={"clean_reference","plain_fedavg","trusted_reconstruction"}
    checks["attacks_complete"]=set(m[m.arm!="clean_reference"].attack)==set(CFG["attacks"])
    checks["all_classes_present"]=set(pc.class_name)==set(CFG["all_eight_classes"])
    checks["primary_endpoint_rows_exact"]=int(pp.primary_endpoint.astype(bool).sum())==50
    checks["no_missing_detector_metrics"]=pp[["malicious_client_recall","benign_client_fpr","reconstructed_client_round_count"]].notna().all().all()
    comp=json.loads(complete.read_text(encoding="utf-8")) if complete.exists() else {}
    hashes=comp.get("output_sha256",{})
    checks["output_hashes_match"]=len(hashes)>=4 and all((FINAL/n).exists() and sha(FINAL/n)==d for n,d in hashes.items())
    checks["negative_results_retained"]=comp.get("negative_results_retained") is True
    checks["best_round_selection_unused"]=comp.get("best_round_selection_used") is False
    checks["post_outcome_retuning_unused"]=comp.get("post_outcome_retuning_used") is False
    passed=sum(map(bool,checks.values())); total=len(checks); ok=passed==total
    (FINAL/"task65c1_integrity_audit.json").write_text(json.dumps({"task":65,"phase":"C1_one_shot_final_evaluation",
      "protocol_id":CFG["protocol_id"],"checks":checks,"checks_passed":passed,"checks_total":total,
      "all_checks_passed":ok,"reserved_test_arrays_materialized":True,"final_evaluation_complete":ok,
      "ready_for_task66_statistics":ok},indent=2,sort_keys=True)+"\n",encoding="utf-8")
    for k,v in checks.items(): print(f"{k}: {v}")
    print(f"PASS: {passed}/{total}"); print("READY FOR TASK 66 STATISTICS:",ok)
    return 0 if ok else 2
if __name__=="__main__": raise SystemExit(main())

#!/usr/bin/env python3
from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"reviewer_revision"/"MATCHED_PLAIN_BATR_P4P_FAILED_AUDIT_RECORD_v4327b.json"
FILES = [
    ROOT/"reviewer_revision"/"MATCHED_PLAIN_BATR_P4P_SEED_VALUES_v4327.csv",
    ROOT/"reviewer_revision"/"MATCHED_PLAIN_BATR_P4P_ABSOLUTE_SUMMARY_v4327.csv",
    ROOT/"reviewer_revision"/"MATCHED_PLAIN_BATR_P4P_PAIRED_STATISTICS_v4327.csv",
]

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    old=ROOT/"reviewer_revision"/"MATCHED_PLAIN_BATR_P4P_AUDIT_v4327.json"
    if old.exists(): raise RuntimeError(f"Unexpected original audit exists: {old}")
    for p in FILES:
        if not p.exists(): raise FileNotFoundError(p)
    obj={
      "protocol":"reviewer_v4327b_failed_postanalysis_audit_record",
      "status":"RECORDED",
      "failure_stage":"final_internal_boolean_audit_after_result_csv_write",
      "failure_reason":"Raw false operation-state flags were incorrectly interpreted as failed validation predicates.",
      "scientific_results_recomputed_during_recording":False,
      "model_training_run":False,
      "test_inference_run":False,
      "retuning_run":False,
      "existing_result_files":{p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in FILES},
    }
    OUT.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
    print("FAILED v4.32.7 AUDIT STATE RECORDED")
    for p in FILES: print(p.name, sha(p))
    return 0

if __name__=="__main__": raise SystemExit(main())

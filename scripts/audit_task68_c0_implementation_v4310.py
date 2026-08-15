from __future__ import annotations
import ast, json, re, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"configs/task68_c0_preregistration_v4310.json").read_text(encoding="utf-8"))
OUT=ROOT/"results/cic_iot_diad_task68_c0_final_audit_preflight_v4310"

def git(*args):
    return subprocess.run(["git",*args],cwd=ROOT,check=True,text=True,capture_output=True).stdout.strip()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks={}
    checks["task_exact"]=CFG["task"]==68
    checks["protocol_exact"]=CFG["protocol_id"]=="task68_c0_v4310"
    checks["parent_tag_exact"]=CFG["parent_required_tag"]=="task67-c3-publication-outputs-audited-v4302"
    parent=git("rev-list","-n","1",CFG["parent_required_tag"]); head=git("rev-parse","HEAD")
    checks["parent_tag_resolves"]=bool(parent)
    checks["head_equals_parent"]=head==parent
    checks["required_tag_chain_count_10"]=len(CFG["required_tag_chain"])==10
    checks["five_final_seeds_exact"]=CFG["final_seeds"]==[1379954285,1886033230,480705558,1377035733,1707771978]
    checks["five_attacks_exact"]=len(CFG["attacks"])==5
    checks["eight_malicious_clients_exact"]=CFG["malicious_clients"]==[1,7,8,10,14,15,17,18]
    checks["poison_fraction_exact"]=CFG["poison_fraction"]==1.0
    checks["datasets_exact"]=CFG["datasets"]==["diagnostic","natural"]
    checks["primary_round_exact"]=CFG["primary_round"]==8
    checks["monitored_rounds_exact"]=CFG["monitored_rounds"]==[5,6,7,8]
    checks["task66_bootstrap_exact"]=CFG["task66"]["bootstrap_replicates"]==20000 and CFG["task66"]["bootstrap_seed"]==650428
    checks["task66_signflip_exact"]=CFG["task66"]["sign_patterns"]==32 and CFG["task66"]["minimum_attainable_two_sided_p"]==0.0625
    checks["task67_output_counts_exact"]=CFG["task67"]["figure_stems"]==30 and CFG["task67"]["png_count"]==30 and CFG["task67"]["pdf_count"]==30 and CFG["task67"]["table_count"]==7

    script=ROOT/"scripts/run_task68_c1_independent_final_audit_v4310.py"
    src=script.read_text(encoding="utf-8"); ast.parse(src)
    checks["audit_script_parses"]=True
    checks["no_np_load"]="np.load(" not in src
    checks["no_torch"]="import torch" not in src and "torch.load" not in src
    checks["no_training_call"]="run_task65_c1_training" not in src and "run_true_warmup" not in src
    checks["no_task65_evaluator_call"]="evaluate_task65" not in src and "run_task65_c1_one_shot" not in src
    checks["no_task66_inference_call"]="run_task66_c1_final_statistics" not in src
    checks["no_task67_generator_call"]="run_task67_c1_publication_outputs" not in src
    checks["no_shap"]="import shap" not in src
    checks["no_llm"]="openai" not in src.lower() and "anthropic" not in src.lower()
    checks["requires_exact_frozen_execution_tag"]="Task68 C1 requires HEAD exactly" in src
    checks["rerun_blocked"]="Task68 final-audit output root already exists; rerun prohibited" in src
    checks["checks_tag_ancestry"]="merge-base" in src and "--is-ancestor" in src
    checks["checks_source_hashes"]="task65_source_sha256" in src
    checks["checks_task65_hashes"]="TASK65_FINAL_EVALUATION_COMPLETE.json" in src and "verify_completion_hashes" in src
    checks["checks_task66_hashes"]="TASK66_FINAL_STATISTICS_COMPLETE.json" in src
    checks["checks_task67_67_hashes"]="TASK67_PUBLICATION_OUTPUTS_COMPLETE.json" in src and "len(rec67)==67" in src
    checks["checks_skipped_tasks"]="task46_to_54_execution_filenames_absent" in src
    checks["checks_claim_boundaries"]="claim_worst_class_recall_improvement_not_supported" in src and "claim_attack_excess_partly_nonestimable" in src
    checks["writes_claim_trace"]="task68_claim_to_evidence_trace_v4310.csv" in src
    checks["writes_artifact_manifest"]="task68_frozen_artifact_manifest_v4310.csv" in src

    # C0 itself performs no independent final audit and reads no scientific CSV values.
    checks["c0_final_audit_checks_executed_zero"]=True
    checks["c0_scientific_csv_rows_loaded_zero"]=True
    checks["c0_reserved_npz_opened_zero"]=True

    checks={k:bool(v) for k,v in checks.items()}
    passed=sum(checks.values()); total=len(checks); ok=passed==total
    d={
      "task":68,"phase":CFG["phase"],"protocol_id":CFG["protocol_id"],
      "head_at_preflight":head,"parent_commit":parent,
      "checks":checks,"checks_passed":passed,"checks_total":total,"all_checks_passed":ok,
      "final_audit_checks_executed":0,"scientific_csv_rows_loaded":0,
      "reserved_npz_opened":False,"models_loaded":False,"training_runs_started":0,
      "ready_to_freeze_task68_c0_implementation":ok,
      "task68_c1_execution_authorized":False
    }
    (OUT/"task68c0_final_audit_implementation_preflight_v4310.json").write_text(json.dumps(d,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    for k,v in checks.items(): print(f"{k}: {v}")
    print(f"PASS: {passed}/{total}")
    print("FINAL AUDIT CHECKS EXECUTED: 0")
    print("SCIENTIFIC CSV ROWS LOADED: 0")
    print("RESERVED NPZ OPENED: False")
    print("READY TO FREEZE TASK68 C0 IMPLEMENTATION:",ok)
    print("TASK68 C1 EXECUTION AUTHORIZED: False")
    return 0 if ok else 2

if __name__=="__main__":
    raise SystemExit(main())

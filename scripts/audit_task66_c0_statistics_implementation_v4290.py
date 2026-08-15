from __future__ import annotations
import ast, importlib.util, json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"configs/task66_c0_preregistration_v4290.json").read_text(encoding="utf-8"))
OUT=ROOT/"results/cic_iot_diad_task66_c0_statistics_preflight_v4290"

def git(*args):
    return subprocess.run(["git",*args],cwd=ROOT,check=True,text=True,capture_output=True).stdout.strip()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks={}
    checks["task_exact"]=CFG["task"]==66
    checks["phase_exact"]=CFG["phase"]=="C0_statistics_implementation_freeze_before_inference"
    checks["protocol_exact"]=CFG["protocol_id"]=="task66_c0_v4290"
    checks["parent_tag_exact"]=CFG["parent_required_tag"]=="task65-c2-final-results-audited-v4283"
    parent=git("rev-list","-n","1",CFG["parent_required_tag"])
    head=git("rev-parse","HEAD")
    checks["parent_tag_resolves"]=bool(parent)
    checks["parent_is_ancestor"]=subprocess.run(["git","merge-base","--is-ancestor",parent,head],cwd=ROOT).returncode==0
    checks["five_seeds_exact"]=CFG["final_seeds"]==[1379954285,1886033230,480705558,1377035733,1707771978]
    checks["five_attacks_exact"]=CFG["attacks"]==["all_to_one_benign","cyclic_shift","multiclass_partial_cycle","pairwise_swap","random_flip"]
    checks["datasets_exact"]=CFG["datasets"]==["diagnostic","natural"]
    checks["primary_round_exact"]=CFG["primary_round"]==8
    checks["secondary_rounds_exact"]=CFG["secondary_rounds"]==[5,6,7,8]
    checks["six_primary_metrics_exact"]=CFG["primary_metrics"]==["macro_f1","balanced_accuracy","worst_class_recall","attack_excess_removed_fraction","malicious_client_recall","benign_client_fpr"]
    checks["bootstrap_replicates_exact"]=CFG["bootstrap"]["replicates"]==20000
    checks["bootstrap_seed_exact"]=CFG["bootstrap"]["rng_seed"]==650428
    checks["bootstrap_confidence_exact"]=CFG["bootstrap"]["confidence_level"]==0.95
    checks["sign_patterns_exact"]=CFG["exact_sign_flip"]["sign_patterns"]==32
    checks["minimum_p_exact"]=CFG["exact_sign_flip"]["minimum_attainable_two_sided_p"]==0.0625
    checks["holm_family_size_exact"]=CFG["holm"]["family_size"]==5
    checks["holm_scope_exact"]="five attacks separately within each dataset and primary metric family" in CFG["holm"]["family"]
    checks["no_seed_drop"]=CFG["no_flexibility_rules"]["drop_failed_seed_or_attack"] is False
    checks["best_round_blocked"]=CFG["no_flexibility_rules"]["best_round_selection"] is False
    checks["best_seed_blocked"]=CFG["no_flexibility_rules"]["best_seed_selection"] is False
    checks["task65_rerun_blocked"]=CFG["no_flexibility_rules"]["task65_rerun"] is False
    checks["reserved_npz_access_blocked"]=CFG["no_flexibility_rules"]["reserved_npz_access"] is False
    checks["model_loading_blocked"]=CFG["no_flexibility_rules"]["model_loading"] is False
    checks["new_training_blocked"]=CFG["no_flexibility_rules"]["new_training"] is False
    checks["new_shap_blocked"]=CFG["no_flexibility_rules"]["new_shap"] is False
    checks["llm_blocked"]=CFG["no_flexibility_rules"]["llm_calls"] is False

    checks["task65_primary_table_exists"]=(ROOT/CFG["task65_primary_table"]).exists()
    checks["task65_completion_exists"]=(ROOT/CFG["task65_completion_marker"]).exists()
    checks["task65_c2_audit_exists"]=(ROOT/CFG["task65_c2_audit"]).exists()

    c2=json.loads((ROOT/CFG["task65_c2_audit"]).read_text(encoding="utf-8"))
    checks["task65_audit_passed"]=c2.get("all_checks_passed") is True and c2.get("ready_for_task66_statistics") is True

    script=ROOT/"scripts/run_task66_c1_final_statistics_v4290.py"
    src=script.read_text(encoding="utf-8")
    ast.parse(src)
    checks["stats_script_parses"]=True
    checks["stats_script_has_no_np_load"]="np.load(" not in src
    checks["stats_script_has_no_torch"]="import torch" not in src and "torch.load" not in src
    checks["stats_script_has_no_training_runner"]="run_task65_c1_training" not in src
    checks["stats_script_has_no_final_evaluator"]="evaluate_task65_c1_one_shot" not in src
    checks["stats_script_requires_frozen_tag"]="required_execution_tag" in src and "Task66 C1 execution requires HEAD exactly" in src
    checks["stats_script_rerun_blocked"]="Task66 final statistics already completed; rerun prohibited" in src
    checks["exact_32_patterns"]="itertools.product((-1.0,1.0), repeat=5)" in src
    checks["holm_no_family_shrink"]="NOT_ESTIMABLE_FULL_FIVE_ATTACK_FAMILY_REQUIRED" in src
    checks["nonfinite_seed_not_dropped"]="NOT_ESTIMABLE_ALL_5_FINITE_REQUIRED" in src
    checks["round8_filter_present"]="global_round==int(CFG[\"primary_round\"])" in src
    checks["all_50_pairs_required"]="len(q)!=50" in src
    checks["detector_threshold_contrasts"]="return level-0.50, level" in src and "return 0.05-level, level" in src
    checks["clean_gate_not_recomputed"]="clean_utility_gate_recomputed_on_reserved_test" in src

    # C0 itself must not execute stats or read the Task65 CSV.
    checks["c0_statistics_rows_computed_zero"]=True
    checks["c0_task65_csv_loaded_zero"]=True
    checks["c0_reserved_npz_opened_zero"]=True

    passed=sum(bool(v) for v in checks.values()); total=len(checks); ok=passed==total
    decision={
      "task":66,"phase":CFG["phase"],"protocol_id":CFG["protocol_id"],
      "head_at_preflight":head,"parent_commit":parent,
      "checks":{k:bool(v) for k,v in checks.items()},
      "checks_passed":passed,"checks_total":total,"all_checks_passed":ok,
      "statistics_rows_computed":0,"task65_csv_rows_loaded":0,
      "reserved_npz_opened":False,"models_loaded":False,"training_runs_started":0,
      "ready_to_freeze_task66_c0_implementation":ok,
      "task66_c1_execution_authorized":False
    }
    (OUT/"task66c0_statistics_implementation_audit_v4290.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    for k,v in checks.items(): print(f"{k}: {bool(v)}")
    print(f"PASS: {passed}/{total}")
    print("STATISTICS ROWS COMPUTED: 0")
    print("TASK65 CSV ROWS LOADED: 0")
    print("RESERVED NPZ OPENED: False")
    print("READY TO FREEZE TASK 66 C0 IMPLEMENTATION:",ok)
    print("TASK 66 C1 EXECUTION AUTHORIZED: False")
    return 0 if ok else 2

if __name__=="__main__":
    raise SystemExit(main())

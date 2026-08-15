from __future__ import annotations
import hashlib, importlib.util, json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"configs/task65_c1_preregistration_v4282.json").read_text(encoding="utf-8"))
OUT=ROOT/"results/cic_iot_diad_task65_c1_implementation_preflight_v4282"
def sha(p): h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def git(*a): return subprocess.run(["git",*a],cwd=ROOT,check=True,text=True,capture_output=True).stdout.strip()
def main():
    OUT.mkdir(parents=True,exist_ok=True); c={}
    c["task_exact"]=CFG["task"]==65; c["protocol_exact"]=CFG["protocol_id"]=="task65_c1_v4282"
    parent=git("rev-parse",CFG["parent_tag"]+"^{commit}"); head=git("rev-parse","HEAD")
    c["parent_is_ancestor"]=subprocess.run(["git","merge-base","--is-ancestor",parent,head],cwd=ROOT).returncode==0
    c["v4281_unfrozen_recorded"]=CFG["correction_from_unfrozen_v4281"]["v4281_was_committed"] is False and CFG["correction_from_unfrozen_v4281"]["training_runs_started"]==0
    c["final_seeds_exact"]=CFG["final_seeds"]==[1379954285,1886033230,480705558,1377035733,1707771978]
    c["attacks_exact"]=CFG["attacks"]==["all_to_one_benign","cyclic_shift","multiclass_partial_cycle","pairwise_swap","random_flip"]
    c["coalition_exact"]=CFG["malicious_clients"]==[1,7,8,10,14,15,17,18]
    c["poison_exact"]=CFG["poison_fraction"]==1.0; c["rounds_exact"]=CFG["monitored_rounds"]==[5,6,7,8]
    c["data_exists"]=(ROOT/CFG["data_file"]).exists(); c["partition_exists"]=(ROOT/CFG["partition_file"]).exists()
    c["psutil_available"]=importlib.util.find_spec("psutil") is not None
    c["frozen_hashes_exact"]=all((ROOT/r).exists() and sha(ROOT/r)==h for r,h in CFG["frozen_source_hashes"].items())

    guard=(ROOT/"scripts/task65_guard_v4282/sitecustomize.py").read_text(encoding="utf-8")
    train=(ROOT/"scripts/run_task65_c1_training_v4282.py").read_text(encoding="utf-8")
    evals=(ROOT/"scripts/evaluate_task65_c1_one_shot_v4282.py").read_text(encoding="utf-8")
    c["captures_plain_checkpoint"]="continuation_last_round_model.pt" in guard
    c["captures_defense_checkpoint"]="v320b1_last_round_model.pt" in guard
    c["guard_materializes_train_val_only"]='arrays={k:data[k] for k in _ALLOWED}' in guard
    c["training_has_no_reserved_array_names"]=not any(x in train for x in ("X_test_natural","y_test_natural","X_test_diagnostic","y_test_diagnostic"))
    c["evaluator_has_all_reserved_names"]=all(x in evals for x in ("X_test_natural","y_test_natural","X_test_diagnostic","y_test_diagnostic"))
    c["access_marker_precedes_np_load"]=evals.index("ACCESS.write_text")<evals.index("with np.load(data_file")
    c["final_rerun_blocked"]="if ACCESS.exists() or COMPLETE.exists()" in evals
    c["reconstruction_arg_uses_builder_root"]='"--reconstruction-calibration-dir",str(recon)' in train and 'str(recon/"calibration")' not in train
    c["clean_partition_hash_key_exact"]='"partition_hash_sha256":ph' in train
    c["incomplete_stage_restart_before_access"]="shutil.rmtree(stage_dir)" in train and "Final access already started" in train
    c["clean_utility_scope_no_new_arm"]=CFG["clean_utility_threshold_scope"]["new_comparison_arm_added"] is False
    c["best_round_blocked"]=CFG["execution_policy"]["best_round_selection_permitted"] is False
    c["best_seed_blocked"]=CFG["execution_policy"]["best_seed_selection_permitted"] is False
    c["drop_failures_blocked"]=CFG["execution_policy"]["failed_seed_or_attack_dropping_permitted"] is False
    c["task46_54_blocked"]=CFG["execution_policy"]["task46_54_conditions_permitted"] is False
    c["no_access_marker"]=not (ROOT/CFG["output_root"]/CFG["test_access_boundary"]["final_access_marker"]).exists()
    c["no_completion_marker"]=not (ROOT/CFG["output_root"]/CFG["test_access_boundary"]["final_completion_marker"]).exists()
    passed=sum(map(bool,c.values())); total=len(c); ok=passed==total
    d={"task":65,"phase":CFG["phase"],"protocol_id":CFG["protocol_id"],"checks":c,
       "checks_passed":passed,"checks_total":total,"all_checks_passed":ok,
       "reserved_test_arrays_materialized":False,"training_runs_started":0,"final_test_metrics_computed":0,
       "ready_to_freeze_task65_c1_implementation":ok,"task65_c1_final_execution_authorized":False}
    (OUT/"task65c1_implementation_audit_v4282.json").write_text(json.dumps(d,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    for k,v in c.items(): print(f"{k}: {v}")
    print(f"PASS: {passed}/{total}"); print("READY TO FREEZE TASK 65 C1 V4.28.2:",ok)
    print("TRAINING RUNS STARTED: 0"); print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("TASK 65 C1 FINAL EXECUTION AUTHORIZED: False")
    return 0 if ok else 2
if __name__=="__main__": raise SystemExit(main())

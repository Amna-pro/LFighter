#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch

ROOT=Path(__file__).resolve().parents[1]; REV=ROOT/"reviewer_revision"
SEEDS=[1379954285,1886033230,480705558,1377035733,1707771978]
ATTACKS=["all_to_one_benign","cyclic_shift","multiclass_partial_cycle","pairwise_swap","random_flip"]
PA="all_to_one_benign"; PS=1379954285
PREFLIGHT=ROOT/"results"/"reviewer_geometric_median_attacked_preflight_v4332"/PA/f"seed_{PS}"
GRID=ROOT/"results"/"reviewer_geometric_median_matched_grid_v4333"
OR=REV/"GEOMETRIC_MEDIAN_MATCHED_FULL_GRID_ROUNDS_v4333.csv"
OC=REV/"GEOMETRIC_MEDIAN_MATCHED_FULL_GRID_CONDITIONS_v4333.csv"
OA=REV/"GEOMETRIC_MEDIAN_MATCHED_FULL_GRID_ATTACK_SUMMARY_v4333.csv"
OJ=REV/"GEOMETRIC_MEDIAN_MATCHED_FULL_GRID_AUDIT_v4333.json"
PH="5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()
def src(attack,seed):
    return PREFLIGHT if attack==PA and seed==PS else GRID/attack/f"seed_{seed}"
def evp(attack,seed):
    return REV/f"TASK65_ATTACK_MANIFEST_RECOVERY_SEED_{seed}_{attack.upper()}_v4324.json"

def main():
    for p in (OR,OC,OA,OJ):
        if p.exists(): raise FileExistsError(f"Refusing overwrite: {p}")
    rounds_all=[]; conds=[]; total_checks=0; ck_hash={}
    for attack in ATTACKS:
        for seed in SEEDS:
            d=src(attack,seed)
            mp=d/"REVIEWER_GEOMETRIC_MEDIAN_MATCHED_COMPLETE.json"
            rp=d/"tables"/"geometric_median_round_metrics.csv"
            cp=d/"tables"/"geometric_median_client_records.csv"
            pp=d/"tables"/"validation_class_metrics_long.csv"
            xp=d/"tables"/"validation_confusion_matrix_long.csv"
            kp=d/"checkpoints"/"reviewer_round_checkpoints"/"global_round_08_model.pt"
            ep=evp(attack,seed)
            for p in (mp,rp,cp,pp,xp,kp,ep):
                if not p.exists(): raise FileNotFoundError(p)
            m=json.loads(mp.read_text(encoding="utf-8")); e=json.loads(ep.read_text(encoding="utf-8"))
            r=pd.read_csv(rp); c=pd.read_csv(cp); pc=pd.read_csv(pp); cm=pd.read_csv(xp)
            k=torch.load(kp,map_location="cpu",weights_only=False)
            poison=e["poison_index_hash_sha256"]
            checks=[
              m.get("phase")=="reviewer_matched_geometric_median_comparator",
              m.get("mode")=="strong_attack",m.get("attack_type")==attack,int(m.get("model_seed",-1))==seed,
              int(m.get("num_clients",-1))==20,int(m.get("continuation_rounds",-1))==4,
              m.get("partition_hash_sha256")==PH,m.get("poison_index_hash_sha256")==poison,
              m.get("aggregation")=="geometric_median_all_20_submitted_updates",
              m.get("weighting")=="equal_client_geometric_median_objective",
              int(m.get("geomed_max_iter",-1))==60,float(m.get("geomed_tolerance",-1))==1e-7,
              float(m.get("geomed_near_threshold",-1))==1e-12,m.get("geomed_initialization")=="numpy_coordinate_median",
              m.get("geomed_numeric_dtype")=="float64",m.get("detector_used") is False,
              m.get("client_rejection_used") is False,m.get("client_reconstruction_used") is False,
              m.get("sample_count_weighting_used") is False,m.get("test_sets_accessed") is False,
              m.get("attack_specific_retuning") is False,len(r)==4,
              list(r["global_round"].astype(int))==[5,6,7,8],set(r["arm"])=={"geometric_median"},
              set(r["geomed_max_iter"].astype(int))=={60},set(r["geomed_tolerance"].astype(float))=={1e-7},
              bool(((r["algorithm_iterations"].astype(int)>=1)&(r["algorithm_iterations"].astype(int)<=60)).all()),
              len(c)==80,bool(c["submitted_to_geometric_median"].astype(bool).all()),len(pc)==32,len(cm)==256,
              k.get("arm")=="geometric_median",int(k.get("global_round",-1))==8,int(k.get("model_seed",-1))==seed,
              k.get("attack_type")==attack,k.get("partition_hash")==PH,k.get("poison_index_hash")==poison,
              k.get("aggregation")=="geometric_median_all_20_submitted_updates",
              k.get("weighting")=="equal_client_geometric_median_objective",
              int(k.get("geomed_max_iter",-1))==60,float(k.get("geomed_tolerance",-1))==1e-7,
              1<=int(k.get("algorithm_iterations",-1))<=60,isinstance(k.get("model_state_dict"),dict)
            ]
            if not all(checks):
                bad=[i for i,v in enumerate(checks,1) if not v]
                raise RuntimeError(f"Condition audit failed attack={attack} seed={seed} checks={bad}")
            total_checks+=len(checks)
            rr=r.copy(); rr.insert(0,"model_seed",seed); rr.insert(0,"attack",attack)
            rr["source_kind"]="audited_preflight_reuse" if attack==PA and seed==PS else "v4333_grid"
            rounds_all.append(rr)
            r8=r.loc[r["global_round"].astype(int)==8].iloc[0]
            conds.append({
              "attack":attack,"model_seed":seed,
              "source_kind":"audited_preflight_reuse" if attack==PA and seed==PS else "v4333_grid",
              "partition_hash_sha256":m["partition_hash_sha256"],"poison_index_hash_sha256":m["poison_index_hash_sha256"],
              "geomed_max_iter":60,"geomed_tolerance":1e-7,
              "all_rounds_converged":bool(r["algorithm_converged"].astype(bool).all()),
              "maximum_algorithm_iterations":int(r["algorithm_iterations"].astype(int).max()),
              "round8_algorithm_iterations":int(r8["algorithm_iterations"]),
              "round8_algorithm_converged":bool(r8["algorithm_converged"]),
              "round8_val_macro_f1":float(r8["val_macro_f1"]),
              "round8_val_balanced_accuracy":float(r8["val_balanced_accuracy"]),
              "mean_round_val_macro_f1":float(r["val_macro_f1"].mean()),
              "mean_round_val_balanced_accuracy":float(r["val_balanced_accuracy"].mean()),
              "total_seconds":float(m["total_seconds"]),"test_sets_accessed":bool(m["test_sets_accessed"]),
              "attack_specific_retuning":bool(m["attack_specific_retuning"]),"round8_checkpoint_sha256":sha(kp)})
            ck_hash[f"{attack}/seed_{seed}"]=sha(kp)
    rdf=pd.concat(rounds_all,ignore_index=True)
    cdf=pd.DataFrame(conds).sort_values(["attack","model_seed"]).reset_index(drop=True)
    if len(rdf)!=100 or len(cdf)!=25: raise RuntimeError("grid cardinality mismatch")
    rows=[]
    for attack,g in cdf.groupby("attack",sort=False):
        rows.append({"attack":attack,"n_seeds":len(g),
          "mean_round8_val_macro_f1":float(g["round8_val_macro_f1"].mean()),
          "median_round8_val_macro_f1":float(g["round8_val_macro_f1"].median()),
          "std_round8_val_macro_f1_ddof1":float(g["round8_val_macro_f1"].std(ddof=1)),
          "min_round8_val_macro_f1":float(g["round8_val_macro_f1"].min()),
          "max_round8_val_macro_f1":float(g["round8_val_macro_f1"].max()),
          "mean_round8_val_balanced_accuracy":float(g["round8_val_balanced_accuracy"].mean()),
          "all_conditions_all_rounds_converged":bool(g["all_rounds_converged"].all()),
          "maximum_algorithm_iterations_observed":int(g["maximum_algorithm_iterations"].max()),
          "mean_total_seconds":float(g["total_seconds"].mean())})
    adf=pd.DataFrame(rows)
    rdf.to_csv(OR,index=False); cdf.to_csv(OC,index=False); adf.to_csv(OA,index=False)
    checks={
      "conditions_exact_25":len(cdf)==25,"round_rows_exact_100":len(rdf)==100,
      "attacks_exact_5":set(cdf["attack"])==set(ATTACKS),"seeds_exact_5":set(cdf["model_seed"].astype(int))==set(SEEDS),
      "one_preflight_reuse":int((cdf["source_kind"]=="audited_preflight_reuse").sum())==1,
      "remaining_grid_conditions_24":int((cdf["source_kind"]=="v4333_grid").sum())==24,
      "max_iter_exact_60_all":set(cdf["geomed_max_iter"].astype(int))=={60},
      "tolerance_exact_1e_7_all":set(cdf["geomed_tolerance"].astype(float))=={1e-7},
      "partition_hash_all_exact":set(cdf["partition_hash_sha256"])=={PH},
      "test_sets_accessed_false_all":not cdf["test_sets_accessed"].astype(bool).any(),
      "attack_specific_retuning_false_all":not cdf["attack_specific_retuning"].astype(bool).any(),
      "round8_metrics_finite":bool(np.isfinite(cdf[["round8_val_macro_f1","round8_val_balanced_accuracy"]].to_numpy(float)).all()),
      "formal_cross_method_statistics_performed":False,"final_test_evaluation_performed":False,
      "negative_results_retained":True,"algorithm_convergence_used_as_selection_gate":False}
    failed=[k for k,v in checks.items() if k not in {"formal_cross_method_statistics_performed","final_test_evaluation_performed","algorithm_convergence_used_as_selection_gate"} and not bool(v)]
    if failed: raise RuntimeError(f"Full-grid audit failed: {failed}")
    obj={"protocol":"reviewer_v4333_geometric_median_five_attack_five_seed_grid","status":"PASS",
      "checks":checks,"condition_validation_checks_performed":total_checks,"conditions":25,"round_rows":100,
      "preflight_conditions_reused":1,"new_grid_conditions":24,"geomed_max_iter":60,"geomed_tolerance":1e-7,
      "all_25_conditions_all_rounds_converged_observed":bool(cdf["all_rounds_converged"].all()),
      "maximum_algorithm_iterations_observed":int(cdf["maximum_algorithm_iterations"].max()),
      "test_sets_accessed":False,"attack_specific_retuning":False,"formal_cross_method_statistics_performed":False,
      "final_test_evaluation_performed":False,"algorithm_convergence_used_as_selection_gate":False,
      "checkpoint_sha256":ck_hash,"summary_output_sha256":{OR.name:sha(OR),OC.name:sha(OC),OA.name:sha(OA)}}
    OJ.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
    print("GEOMETRIC MEDIAN FULL GRID SUMMARY AUDIT = PASS")
    print("CONDITIONS:",len(cdf)); print("ROUND ROWS:",len(rdf)); print("CONDITION VALIDATION CHECKS:",total_checks)
    print("PREFLIGHT REUSED: 1"); print("NEW GRID CONDITIONS: 24"); print("MAX ITER: 60"); print("TOLERANCE: 1e-7")
    print("ALL 25 CONDITIONS ALL ROUNDS CONVERGED OBSERVED:",bool(cdf["all_rounds_converged"].all()))
    print("MAXIMUM ALGORITHM ITERATIONS OBSERVED:",int(cdf["maximum_algorithm_iterations"].max()))
    print("TEST SETS ACCESSED: False"); print("ATTACK SPECIFIC RETUNING: False")
    print("FORMAL CROSS METHOD STATISTICS PERFORMED: False"); print("FINAL TEST EVALUATION PERFORMED: False")
    print("\nATTACK SUMMARY"); print(adf.to_string(index=False))
if __name__=="__main__": main()

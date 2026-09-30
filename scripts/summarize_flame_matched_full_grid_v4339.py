#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.metadata as md, json
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT=Path(__file__).resolve().parents[1]; REV=ROOT/"reviewer_revision"
SEEDS=[1379954285,1886033230,480705558,1377035733,1707771978]
ATTACKS=["all_to_one_benign","cyclic_shift","multiclass_partial_cycle","pairwise_swap","random_flip"]
PA,PS="all_to_one_benign",1379954285
PREF=ROOT/"results"/"reviewer_flame_attacked_preflight_v4338"/PA/f"seed_{PS}"
GRID=ROOT/"results"/"reviewer_flame_matched_grid_v4339"
OR=REV/"FLAME_MATCHED_FULL_GRID_ROUNDS_v4339.csv"
OC=REV/"FLAME_MATCHED_FULL_GRID_CONDITIONS_v4339.csv"
OA=REV/"FLAME_MATCHED_FULL_GRID_ATTACK_SUMMARY_v4339.csv"
OJ=REV/"FLAME_MATCHED_FULL_GRID_AUDIT_v4339.json"
PH="5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"
AGG="c964742e0279ef98cc3b3d0ce36c62282af1be9c7b94d3b8b484240085333f0d"

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for x in iter(lambda:f.read(1<<20),b""): h.update(x)
    return h.hexdigest()
def src(a,s): return PREF if (a==PA and s==PS) else GRID/a/f"seed_{s}"
def ev(a,s): return REV/f"TASK65_ATTACK_MANIFEST_RECOVERY_SEED_{s}_{a.upper()}_v4324.json"

def main():
    for p in (OR,OC,OA,OJ):
        if p.exists(): raise FileExistsError(f"Refusing overwrite: {p}")
    if md.version("hdbscan")!="0.8.44": raise RuntimeError("HDBSCAN version changed")
    if sha(ROOT/"src"/"aggregation.py")!=AGG: raise RuntimeError("FLAME source changed")

    rounds=[]; cond=[]; ckhash={}; totalchecks=0
    for a in ATTACKS:
      for s in SEEDS:
        d=src(a,s)
        mp=d/"REVIEWER_FLAME_MATCHED_COMPLETE.json"; rp=d/"tables"/"flame_round_metrics.csv"
        cp=d/"tables"/"flame_client_decisions.csv"; sp=d/"tables"/"flame_cosine_similarity_long.csv"
        pp=d/"tables"/"validation_class_metrics_long.csv"; cm=d/"tables"/"validation_confusion_matrix_long.csv"
        kp=d/"checkpoints"/"reviewer_round_checkpoints"/"global_round_08_model.pt"
        for p in (mp,rp,cp,sp,pp,cm,kp,ev(a,s)):
            if not p.exists(): raise FileNotFoundError(p)
        m=json.loads(mp.read_text(encoding="utf-8")); e=json.loads(ev(a,s).read_text(encoding="utf-8"))
        r=pd.read_csv(rp); c=pd.read_csv(cp); sim=pd.read_csv(sp); pc=pd.read_csv(pp); conf=pd.read_csv(cm)
        k=torch.load(kp,map_location="cpu",weights_only=False); poison=e["poison_index_hash_sha256"]
        seeds=[s+gr*1000+999 for gr in [5,6,7,8]]
        checks=[
          m.get("phase")=="reviewer_matched_flame_comparator",m.get("mode")=="strong_attack",
          m.get("attack_type")==a,int(m.get("model_seed",-1))==s,int(m.get("num_clients",-1))==20,
          int(m.get("continuation_rounds",-1))==4,m.get("partition_hash_sha256")==PH,
          m.get("poison_index_hash_sha256")==poison,m.get("aggregation_source")=="src/aggregation.py::FLAME",
          m.get("aggregation_source_sha256_expected")==AGG,m.get("hdbscan_version")=="0.8.44",
          m.get("hdbscan_version_is_reviewer_frozen_not_recovered_historical") is True,
          int(m.get("hdbscan_min_cluster_size",-1))==11,int(m.get("hdbscan_min_samples",-1))==1,
          m.get("hdbscan_allow_single_cluster") is True,float(m.get("lambda",-1))==0.001,
          float(m.get("noise_scalar",-1))==1.0,m.get("noise_scalar_adaptation_used") is False,
          m.get("preserved_source_uses_normal_std_sigma_squared") is True,int(m.get("noise_seed_offset",-1))==999,
          m.get("sample_count_weighting_used") is False,m.get("malicious_labels_used_for_aggregation") is False,
          m.get("malicious_labels_used_for_diagnostics_only") is True,m.get("test_sets_accessed") is False,
          m.get("attack_specific_retuning") is False,m.get("scientific_outcome_gate_used") is False,
          len(r)==4,list(r["global_round"].astype(int))==[5,6,7,8],set(r["arm"])=={"flame_preserved_project"},
          set(r["hdbscan_version"].astype(str))=={"0.8.44"},set(r["hdbscan_min_cluster_size"].astype(int))=={11},
          set(r["hdbscan_min_samples"].astype(int))=={1},set(r["noise_scalar"].astype(float))=={1.0},
          set(r["lambda"].astype(float))=={0.001},list(r["flame_noise_seed"].astype(int))==seeds,
          bool(((r.admitted_client_count.astype(int)>=1)&(r.admitted_client_count.astype(int)<=20)).all()),
          bool(((r.rejected_client_count.astype(int)>=0)&(r.rejected_client_count.astype(int)<=19)).all()),
          bool((r.admitted_client_count.astype(int)+r.rejected_client_count.astype(int)==20).all()),
          bool(np.isfinite(r[["malicious_rejection_recall","benign_false_rejection_rate","malicious_admission_rate"]].to_numpy(float)).all()),
          bool(np.isfinite(r[["val_macro_f1","val_balanced_accuracy"]].to_numpy(float)).all()),
          len(c)==80,len(sim)==1600,len(pc)==32,len(conf)==256,
          k.get("arm")=="flame_preserved_project",int(k.get("global_round",-1))==8,int(k.get("model_seed",-1))==s,
          k.get("attack_type")==a,k.get("partition_hash")==PH,k.get("poison_index_hash")==poison,
          k.get("hdbscan_version")=="0.8.44",float(k.get("noise_scalar",-1))==1.0,float(k.get("lambda",-1))==0.001,
          int(k.get("flame_noise_seed",-1))==seeds[-1],isinstance(k.get("model_state_dict"),dict)]
        if not all(checks): raise RuntimeError(f"Condition audit failed {a} {s}: {[i for i,v in enumerate(checks,1) if not v]}")
        totalchecks+=len(checks)

        rr=r.copy(); rr.insert(0,"model_seed",s); rr.insert(0,"attack",a)
        rr["source_kind"]="audited_preflight_reuse" if (a==PA and s==PS) else "v4339_grid"; rounds.append(rr)
        r8=r.loc[r.global_round.astype(int)==8].iloc[0]
        cond.append({
          "attack":a,"model_seed":s,"source_kind":"audited_preflight_reuse" if (a==PA and s==PS) else "v4339_grid",
          "partition_hash_sha256":m["partition_hash_sha256"],"poison_index_hash_sha256":m["poison_index_hash_sha256"],
          "hdbscan_version":"0.8.44","hdbscan_min_cluster_size":11,"hdbscan_min_samples":1,
          "lambda":0.001,"noise_scalar":1.0,
          "fallback_round_count":int(r.all_outliers_fallback.astype(bool).sum()),
          "round8_admitted_client_count":int(r8.admitted_client_count),"round8_rejected_client_count":int(r8.rejected_client_count),
          "round8_malicious_rejection_recall":float(r8.malicious_rejection_recall),
          "round8_benign_false_rejection_rate":float(r8.benign_false_rejection_rate),
          "round8_val_macro_f1":float(r8.val_macro_f1),"round8_val_balanced_accuracy":float(r8.val_balanced_accuracy),
          "mean_round_malicious_rejection_recall":float(r.malicious_rejection_recall.mean()),
          "mean_round_benign_false_rejection_rate":float(r.benign_false_rejection_rate.mean()),
          "mean_round_val_macro_f1":float(r.val_macro_f1.mean()),"mean_round_val_balanced_accuracy":float(r.val_balanced_accuracy.mean()),
          "total_seconds":float(m["total_seconds"]),"test_sets_accessed":bool(m["test_sets_accessed"]),
          "attack_specific_retuning":bool(m["attack_specific_retuning"]),"round8_checkpoint_sha256":sha(kp)})
        ckhash[f"{a}/seed_{s}"]=sha(kp)

    rdf=pd.concat(rounds,ignore_index=True); cdf=pd.DataFrame(cond).sort_values(["attack","model_seed"]).reset_index(drop=True)
    if len(rdf)!=100 or len(cdf)!=25: raise RuntimeError("Cardinality mismatch")
    rows=[]
    for a,g in cdf.groupby("attack",sort=False):
        rows.append({"attack":a,"n_seeds":len(g),"mean_round8_val_macro_f1":float(g.round8_val_macro_f1.mean()),
          "median_round8_val_macro_f1":float(g.round8_val_macro_f1.median()),
          "std_round8_val_macro_f1_ddof1":float(g.round8_val_macro_f1.std(ddof=1)),
          "min_round8_val_macro_f1":float(g.round8_val_macro_f1.min()),"max_round8_val_macro_f1":float(g.round8_val_macro_f1.max()),
          "mean_round8_val_balanced_accuracy":float(g.round8_val_balanced_accuracy.mean()),
          "mean_round8_malicious_rejection_recall":float(g.round8_malicious_rejection_recall.mean()),
          "mean_round8_benign_false_rejection_rate":float(g.round8_benign_false_rejection_rate.mean()),
          "fallback_rounds_total":int(g.fallback_round_count.sum()),"mean_total_seconds":float(g.total_seconds.mean())})
    adf=pd.DataFrame(rows); rdf.to_csv(OR,index=False); cdf.to_csv(OC,index=False); adf.to_csv(OA,index=False)
    checks={"conditions_exact_25":len(cdf)==25,"round_rows_exact_100":len(rdf)==100,
      "one_preflight_reuse":int((cdf.source_kind=="audited_preflight_reuse").sum())==1,
      "remaining_grid_conditions_24":int((cdf.source_kind=="v4339_grid").sum())==24,
      "hdbscan_version_exact_all":set(cdf.hdbscan_version)=={"0.8.44"},"noise_scalar_exact_all":set(cdf.noise_scalar)=={1.0},
      "lambda_exact_all":set(cdf["lambda"])=={0.001},"partition_hash_all_exact":set(cdf.partition_hash_sha256)=={PH},
      "test_sets_accessed_false_all":not cdf.test_sets_accessed.astype(bool).any(),
      "attack_specific_retuning_false_all":not cdf.attack_specific_retuning.astype(bool).any(),
      "formal_cross_method_statistics_performed":False,"final_test_evaluation_performed":False,"negative_results_retained":True}
    failed=[k for k,v in checks.items() if k not in {"formal_cross_method_statistics_performed","final_test_evaluation_performed"} and not bool(v)]
    if failed: raise RuntimeError(f"Full-grid audit failed: {failed}")
    obj={"protocol":"reviewer_v4339_flame_five_attack_five_seed_grid","status":"PASS","checks":checks,
      "condition_validation_checks_performed":totalchecks,"conditions":25,"round_rows":100,"preflight_conditions_reused":1,
      "new_grid_conditions":24,"hdbscan_version":"0.8.44","hdbscan_version_is_reviewer_frozen_not_recovered_historical":True,
      "hdbscan_min_cluster_size":11,"hdbscan_min_samples":1,"lambda":0.001,"noise_scalar":1.0,
      "test_sets_accessed":False,"attack_specific_retuning":False,"formal_cross_method_statistics_performed":False,
      "final_test_evaluation_performed":False,"checkpoint_sha256":ckhash,
      "summary_output_sha256":{OR.name:sha(OR),OC.name:sha(OC),OA.name:sha(OA)}}
    OJ.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
    print("FLAME FULL GRID SUMMARY AUDIT = PASS"); print("CONDITIONS:",len(cdf)); print("ROUND ROWS:",len(rdf))
    print("CONDITION VALIDATION CHECKS:",totalchecks); print("PREFLIGHT REUSED: 1"); print("NEW GRID CONDITIONS: 24")
    print("TEST SETS ACCESSED: False"); print("ATTACK SPECIFIC RETUNING: False")
    print("FORMAL CROSS METHOD STATISTICS PERFORMED: False"); print("FINAL TEST EVALUATION PERFORMED: False")
    print("\nATTACK SUMMARY"); print(adf.to_string(index=False)); return 0
if __name__=="__main__": raise SystemExit(main())

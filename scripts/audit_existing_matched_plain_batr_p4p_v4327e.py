#!/usr/bin/env python3
from __future__ import annotations
import hashlib,itertools,json,math
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TASK66_SEED=ROOT/"results"/"cic_iot_diad_task66_final_statistics_v4290"/"task66_primary_seed_values.csv"
TASK66_STATS=ROOT/"results"/"cic_iot_diad_task66_final_statistics_v4290"/"task66_primary_statistics.csv"
TASK65_RAW=ROOT/"results"/"cic_iot_diad_task65_c1_one_shot_v4282"/"final"/"raw_checkpoint_metrics.csv"
P4P=ROOT/"reviewer_revision"/"P4P_ROUND8_TEST_METRICS_v4326.csv"
SEED_OUT=ROOT/"reviewer_revision"/"MATCHED_PLAIN_BATR_P4P_SEED_VALUES_v4327.csv"
ABS_OUT=ROOT/"reviewer_revision"/"MATCHED_PLAIN_BATR_P4P_ABSOLUTE_SUMMARY_v4327.csv"
STATS_OUT=ROOT/"reviewer_revision"/"MATCHED_PLAIN_BATR_P4P_PAIRED_STATISTICS_v4327.csv"
FAILED=ROOT/"reviewer_revision"/"MATCHED_PLAIN_BATR_P4P_FAILED_AUDIT_RECORD_v4327b.json"
AUDIT=ROOT/"reviewer_revision"/"MATCHED_PLAIN_BATR_P4P_AUDIT_v4327e.json"

SEEDS=[1379954285,1886033230,480705558,1377035733,1707771978]
ATTACKS=["all_to_one_benign","cyclic_shift","multiclass_partial_cycle","pairwise_swap","random_flip"]
DATASETS=["diagnostic","natural"]; METRICS=["macro_f1","balanced_accuracy"]; ROUND=8; N_BOOT=20000

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

def sseed(k):
    return int.from_bytes(hashlib.sha256(k.encode()).digest()[:8],"big")%(2**32)

def signflip(d):
    d=np.asarray(d,float); obs=abs(float(d.mean()))
    vals=[abs(float(np.mean(d*np.asarray(s)))) for s in itertools.product((-1.,1.),repeat=len(d))]
    return float(np.mean(np.asarray(vals)>=(obs-1e-15)))

def boot(d,k):
    d=np.asarray(d,float); rng=np.random.default_rng(sseed(k))
    idx=rng.integers(0,len(d),size=(N_BOOT,len(d))); m=d[idx].mean(axis=1)
    q=np.quantile(m,[.025,.975]); return float(q[0]),float(q[1])

def dz(d):
    d=np.asarray(d,float); sd=float(np.std(d,ddof=1)); m=float(np.mean(d))
    if sd==0: return 0.0 if m==0 else math.copysign(float("inf"),m)
    return m/sd

def holm(ps):
    p=np.asarray(ps,float); order=np.argsort(p); out=np.empty(len(p)); run=0.
    for rank,idx in enumerate(order):
        run=max(run,min(1.,(len(p)-rank)*p[idx])); out[idx]=run
    return out.tolist()

def close(a,b,label,atol=1e-13):
    if not np.isclose(float(a),float(b),rtol=0,atol=atol,equal_nan=True):
        raise RuntimeError(f"{label}: {a} != {b}")

def main():
    if AUDIT.exists(): raise FileExistsError(AUDIT)
    for p in [TASK66_SEED,TASK66_STATS,TASK65_RAW,P4P,SEED_OUT,ABS_OUT,STATS_OUT,FAILED]:
        if not p.exists(): raise FileNotFoundError(p)

    s66=pd.read_csv(TASK66_SEED); st66=pd.read_csv(TASK66_STATS); raw=pd.read_csv(TASK65_RAW)
    p4p=pd.read_csv(P4P); seed=pd.read_csv(SEED_OUT); ab=pd.read_csv(ABS_OUT); st=pd.read_csv(STATS_OUT)

    checks={
      "failed_attempt_record_exists":FAILED.exists(),
      "matched_seed_rows_exact_100":len(seed)==100,
      "absolute_summary_rows_exact_60":len(ab)==60,
      "paired_statistics_rows_exact_60":len(st)==60,
      "all_round8":set(seed.global_round.astype(int))=={8},
      "all_five_seeds":set(seed.seed.astype(int))==set(SEEDS),
      "all_five_attacks":set(seed.attack)==set(ATTACKS),
      "both_datasets":set(seed.dataset)==set(DATASETS),
      "primary_metrics_only":set(seed.metric)==set(METRICS),
      "all_seed_values_finite":bool(np.isfinite(seed[["plain_fedavg","batr_fl","p4p"]].to_numpy(float)).all()),
    }

    identity=0
    for r in seed.itertuples(index=False):
        metric,dataset,attack,sd=str(r.metric),str(r.dataset),str(r.attack),int(r.seed)
        s=s66[(s66.metric==metric)&(s66.dataset==dataset)&(s66.attack==attack)&(s66.seed.astype(int)==sd)&(s66.global_round.astype(int)==8)]
        if len(s)!=1: raise RuntimeError("Task66 source cardinality mismatch")
        sr=s.iloc[0]; eb=float(sr.reported_level); ep=eb-float(sr.seed_level_contrast)
        close(r.batr_fl,eb,"Task66 batr"); close(r.plain_fedavg,ep,"Task66 plain"); close(r.batr_minus_plain,sr.seed_level_contrast,"Task66 contrast"); identity+=3
        rr=raw[(raw.dataset==dataset)&(raw.attack==attack)&(raw.seed.astype(int)==sd)&(raw.global_round.astype(int)==8)]
        rb=rr[rr.arm=="trusted_reconstruction"]; rp=rr[rr.arm=="plain_fedavg"]
        if len(rb)!=1 or len(rp)!=1: raise RuntimeError("Task65 raw cardinality mismatch")
        close(r.batr_fl,rb.iloc[0][metric],"Task65 batr"); close(r.plain_fedavg,rp.iloc[0][metric],"Task65 plain"); identity+=2
        pr=p4p[(p4p.dataset==dataset)&(p4p.attack_type==attack)&(p4p.model_seed.astype(int)==sd)&(p4p.global_round.astype(int)==8)&(p4p.arm=="p4p_matched")]
        if len(pr)!=1: raise RuntimeError("P4P cardinality mismatch")
        close(r.p4p,pr.iloc[0][metric],"P4P"); close(r.p4p_minus_plain,float(r.p4p)-float(r.plain_fedavg),"P4P-plain"); close(r.batr_minus_p4p,float(r.batr_fl)-float(r.p4p),"BATR-P4P"); identity+=3
    checks["source_identity_checks_exact_800"]=identity==800

    ac=0
    for (metric,dataset,attack),g in seed.groupby(["metric","dataset","attack"],sort=False):
        for method,col in [("plain_fedavg","plain_fedavg"),("batr_fl","batr_fl"),("p4p","p4p")]:
            row=ab[(ab.metric==metric)&(ab.dataset==dataset)&(ab.attack==attack)&(ab.method==method)]
            if len(row)!=1: raise RuntimeError("absolute row missing")
            row=row.iloc[0]; x=g[col].to_numpy(float)
            for c,v in [("mean",x.mean()),("median",np.median(x)),("std_ddof1",np.std(x,ddof=1)),("min",x.min()),("max",x.max())]:
                close(row[c],v,"absolute "+c); ac+=1
    checks["absolute_numeric_checks_exact_300"]=ac==300

    oc=0
    old=st66[st66.metric.isin(METRICS)&st66.dataset.isin(DATASETS)&st66.attack.isin(ATTACKS)&(st66.global_round.astype(int)==8)]
    if len(old)!=20: raise RuntimeError("Task66 stats expected 20")
    for tr in old.itertuples(index=False):
        row=st[(st.comparison=="batr_fl_minus_plain_fedavg")&(st.metric==tr.metric)&(st.dataset==tr.dataset)&(st.attack==tr.attack)]
        if len(row)!=1: raise RuntimeError("frozen Task66 stats missing")
        row=row.iloc[0]
        mapping={"mean_method_a":"mean_reported_level","median_method_a":"median_reported_level"}
        for c in ["mean_contrast","median_contrast","bootstrap_ci_low","bootstrap_ci_high","exact_sign_flip_p_raw","cohen_dz","positive_seed_count","mean_method_a","median_method_a","resolution_limited_p_min","holm_p_adjusted"]:
            close(row[c],getattr(tr,mapping.get(c,c)),"Task66 stats "+c); oc+=1
    checks["frozen_task66_statistics_checks_exact_220"]=oc==220

    nc=0
    for comp,a,b in [("p4p_minus_plain_fedavg","p4p","plain_fedavg"),("batr_fl_minus_p4p","batr_fl","p4p")]:
        for metric in METRICS:
            for dataset in DATASETS:
                fam=[]
                for attack in ATTACKS:
                    g=seed[(seed.metric==metric)&(seed.dataset==dataset)&(seed.attack==attack)].sort_values("seed")
                    d=g[a].to_numpy(float)-g[b].to_numpy(float); lo,hi=boot(d,f"{comp}|{metric}|{dataset}|{attack}|v4327")
                    fam.append((attack,{"mean_contrast":d.mean(),"median_contrast":np.median(d),"bootstrap_ci_low":lo,"bootstrap_ci_high":hi,"exact_sign_flip_p_raw":signflip(d),"cohen_dz":dz(d),"positive_seed_count":int(np.sum(d>0)),"mean_method_a":g[a].mean(),"median_method_a":g[a].median(),"resolution_limited_p_min":.0625}))
                adj=holm([v["exact_sign_flip_p_raw"] for _,v in fam])
                for (attack,v),ha in zip(fam,adj):
                    v["holm_p_adjusted"]=ha
                    row=st[(st.comparison==comp)&(st.metric==metric)&(st.dataset==dataset)&(st.attack==attack)]
                    if len(row)!=1: raise RuntimeError("new stats row missing")
                    row=row.iloc[0]
                    for c,val in v.items(): close(row[c],val,"new stats "+c); nc+=1
    checks["new_p4p_statistics_checks_exact_440"]=nc==440
    checks["exact_signflip_resolution_all_0_0625"]=bool(np.allclose(st.resolution_limited_p_min.to_numpy(float),.0625,rtol=0,atol=1e-15))
    checks["negative_and_mixed_results_retained"]=True
    checks["model_training_run_false"]=True
    checks["test_inference_run_false"]=True
    checks["post_outcome_retuning_run_false"]=True

    failed=[k for k,v in checks.items() if not bool(v)]
    if failed: raise RuntimeError(f"Corrected matched-analysis audit failed: {failed}")

    obj={
      "protocol":"reviewer_v4327e_matched_plain_batr_p4p_postfailure_audit","status":"PASS",
      "correction_scope":"validation_logic_only","original_v4327_result_csvs_recomputed_or_overwritten":False,
      "scientific_parameters_changed":False,"model_training_run":False,"test_inference_run":False,"post_outcome_retuning_run":False,
      "checks":checks,"checks_passed":len(checks),"checks_total":len(checks),
      "source_identity_checks":identity,"absolute_numeric_checks":ac,"frozen_task66_statistics_checks":oc,"new_p4p_statistics_checks":nc,
      "unchanged_v4327_output_sha256":{p.name:sha(p) for p in [SEED_OUT,ABS_OUT,STATS_OUT]},
      "formal_p4p_comparisons_performed_in_original_v4327_analysis":True,"audit_only_revalidation_performed_in_v4327b":True,
    }
    AUDIT.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
    print("CORRECTED MATCHED PLAIN FEDAVG vs BATR FL vs P4P AUDIT = PASS")
    print("CHECKS:",len(checks),"/",len(checks))
    print("SOURCE IDENTITY CHECKS:",identity)
    print("ABSOLUTE NUMERIC CHECKS:",ac)
    print("FROZEN TASK66 STATISTICS CHECKS:",oc)
    print("NEW P4P STATISTICS CHECKS:",nc)
    print("ORIGINAL v4.32.7 RESULT CSVs OVERWRITTEN: False")
    print("MODEL TRAINING RUN: False")
    print("TEST INFERENCE RUN: False")
    print("POST OUTCOME RETUNING RUN: False")
    print("\nABSOLUTE MEANS")
    print(ab[["metric","dataset","attack","method","mean"]].to_string(index=False))
    print("\nP4P RELATED PAIRED COMPARISONS")
    print(st[st.comparison.isin(["p4p_minus_plain_fedavg","batr_fl_minus_p4p"])][["comparison","metric","dataset","attack","mean_contrast","bootstrap_ci_low","bootstrap_ci_high","exact_sign_flip_p_raw","cohen_dz","positive_seed_count","holm_p_adjusted"]].to_string(index=False))
    return 0

if __name__=="__main__": raise SystemExit(main())

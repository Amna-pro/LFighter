from __future__ import annotations
import itertools, json, math, hashlib, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT/"configs/task66_c0_preregistration_v4290.json").read_text(encoding="utf-8"))
SRC = ROOT/CFG["task65_primary_table"]
RES = ROOT/CFG["task65_resource_table"]
COMPLETE = ROOT/CFG["task65_completion_marker"]
C2AUDIT = ROOT/CFG["task65_c2_audit"]
OUT = ROOT/CFG["output_root"]

def git(*args):
    return subprocess.run(["git",*args],cwd=ROOT,check=True,text=True,capture_output=True).stdout.strip()

def sha256_file(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def exact_sign_flip(values):
    x=np.asarray(values,dtype=float)
    if len(x)!=5 or not np.isfinite(x).all():
        return math.nan, "NOT_ESTIMABLE_ALL_5_FINITE_REQUIRED"
    obs=abs(float(np.mean(x)))
    stats=[]
    for signs in itertools.product((-1.0,1.0), repeat=5):
        stats.append(abs(float(np.mean(x*np.asarray(signs)))))
    p=float(sum(v >= obs-1e-15 for v in stats)/32.0)
    return p, "OK"

def bootstrap_mean_ci(values, metric_index, dataset_index, attack_index):
    x=np.asarray(values,dtype=float)
    if len(x)!=5 or not np.isfinite(x).all():
        return math.nan, math.nan, "NOT_ESTIMABLE_ALL_5_FINITE_REQUIRED"
    # Deterministic independent stream derived from the one frozen bootstrap seed.
    ss=np.random.SeedSequence([int(CFG["bootstrap"]["rng_seed"]), metric_index, dataset_index, attack_index])
    rng=np.random.default_rng(ss)
    idx=rng.integers(0,5,size=(int(CFG["bootstrap"]["replicates"]),5))
    means=x[idx].mean(axis=1)
    lo,hi=np.quantile(means,[0.025,0.975])
    return float(lo),float(hi),"OK"

def cohen_dz(values):
    x=np.asarray(values,dtype=float)
    if len(x)!=5 or not np.isfinite(x).all():
        return math.nan,"NOT_ESTIMABLE_ALL_5_FINITE_REQUIRED"
    sd=float(np.std(x,ddof=1))
    if not np.isfinite(sd) or sd<=0:
        return math.nan,"UNDEFINED_ZERO_OR_NONFINITE_SD"
    return float(np.mean(x)/sd),"OK"

def holm_adjust(pvals):
    if len(pvals)!=5 or not np.isfinite(np.asarray(pvals,dtype=float)).all():
        return [math.nan]*5,"NOT_ESTIMABLE_FULL_FIVE_ATTACK_FAMILY_REQUIRED"
    p=np.asarray(pvals,dtype=float)
    order=np.argsort(p,kind="mergesort")
    adj=np.empty(5,dtype=float)
    running=0.0
    m=5
    for rank,idx in enumerate(order):
        raw=(m-rank)*p[idx]
        running=max(running,raw)
        adj[idx]=min(1.0,running)
    return adj.tolist(),"OK"

def metric_seed_values(row, metric):
    if metric=="macro_f1":
        return float(row.defended_macro_f1-row.plain_macro_f1), float(row.defended_macro_f1)
    if metric=="balanced_accuracy":
        return float(row.defended_balanced_accuracy-row.plain_balanced_accuracy), float(row.defended_balanced_accuracy)
    if metric=="worst_class_recall":
        return float(row.defended_worst_class_recall-row.plain_worst_class_recall), float(row.defended_worst_class_recall)
    if metric=="attack_excess_removed_fraction":
        return float(row.attack_excess_removed_fraction), float(row.attack_excess_removed_fraction)
    if metric=="malicious_client_recall":
        level=float(row.malicious_client_recall)
        return level-0.50, level
    if metric=="benign_client_fpr":
        level=float(row.benign_client_fpr)
        return 0.05-level, level
    raise KeyError(metric)

def summarize_secondary(df):
    rows=[]
    for metric in CFG["primary_metrics"]:
        for dataset in CFG["datasets"]:
            for attack in CFG["attacks"]:
                for rnd in CFG["secondary_rounds"]:
                    q=df[(df.dataset==dataset)&(df.attack==attack)&(df.global_round==rnd)].copy()
                    vals=[]; levels=[]
                    for r in q.itertuples(index=False):
                        v,l=metric_seed_values(r,metric); vals.append(v); levels.append(l)
                    a=np.asarray(vals,float); lev=np.asarray(levels,float)
                    rows.append({
                        "metric":metric,"dataset":dataset,"attack":attack,"global_round":rnd,
                        "n_required":5,"n_finite":int(np.isfinite(a).sum()),
                        "mean_contrast":float(np.mean(a)) if np.isfinite(a).all() else math.nan,
                        "median_contrast":float(np.median(a)) if np.isfinite(a).all() else math.nan,
                        "mean_reported_level":float(np.mean(lev)) if np.isfinite(lev).all() else math.nan,
                        "median_reported_level":float(np.median(lev)) if np.isfinite(lev).all() else math.nan,
                        "confirmatory_inference":False
                    })
    return pd.DataFrame(rows)

def parse_resources(path):
    if not path.exists(): return pd.DataFrame()
    d=pd.read_csv(path)
    out=[]
    for r in d.itertuples(index=False):
        stage=str(r.stage)
        if stage.endswith("/warmup"):
            stage_type="warmup"; attack=""
        elif stage.endswith("/reconstruction_calibration"):
            stage_type="reconstruction_calibration"; attack=""
        elif stage.endswith("/clean_reference"):
            stage_type="clean_reference"; attack=""
        elif stage.endswith("/plain_fedavg"):
            stage_type="plain_fedavg"; attack=stage.split("/")[0]
        elif stage.endswith("/trusted_reconstruction"):
            stage_type="trusted_reconstruction"; attack=stage.split("/")[0]
        else:
            stage_type="other"; attack=""
        out.append({"stage":stage,"stage_type":stage_type,"attack":attack,
                    "runtime_seconds":float(r.runtime_seconds),"peak_rss_bytes":int(float(r.peak_rss_bytes))})
    raw=pd.DataFrame(out)
    summ=(raw.groupby(["stage_type","attack"],dropna=False)
          .agg(n=("stage","size"),runtime_mean_seconds=("runtime_seconds","mean"),
               runtime_median_seconds=("runtime_seconds","median"),
               peak_rss_mean_bytes=("peak_rss_bytes","mean"),
               peak_rss_max_bytes=("peak_rss_bytes","max")).reset_index())
    return summ

def main():
    required_tag=CFG["required_execution_tag"]
    head=git("rev-parse","HEAD"); tagc=git("rev-list","-n","1",required_tag)
    if head!=tagc: raise RuntimeError(f"Task66 C1 execution requires HEAD exactly at {required_tag}")
    if git("status","--porcelain"): raise RuntimeError("Repository must be clean for Task66 C1 execution.")
    if not SRC.exists() or not COMPLETE.exists() or not C2AUDIT.exists():
        raise FileNotFoundError("Frozen Task65 evidence missing")
    audit=json.loads(C2AUDIT.read_text(encoding="utf-8"))
    if audit.get("all_checks_passed") is not True or audit.get("ready_for_task66_statistics") is not True:
        raise RuntimeError("Task65 C2 audit does not authorize Task66.")
    completion=json.loads(COMPLETE.read_text(encoding="utf-8"))
    recorded=completion.get("output_sha256",{})
    if recorded.get(SRC.name)!=sha256_file(SRC):
        raise RuntimeError("Task65 paired table hash differs from frozen completion marker.")

    OUT.mkdir(parents=True,exist_ok=True)
    decision_path=OUT/"TASK66_FINAL_STATISTICS_COMPLETE.json"
    if decision_path.exists(): raise RuntimeError("Task66 final statistics already completed; rerun prohibited.")

    df=pd.read_csv(SRC)
    q=df[df.global_round==int(CFG["primary_round"])].copy()
    expected=set((s,a,d) for s in CFG["final_seeds"] for a in CFG["attacks"] for d in CFG["datasets"])
    got=set((int(r.seed),str(r.attack),str(r.dataset)) for r in q.itertuples(index=False))
    if got!=expected or len(q)!=50:
        raise RuntimeError("Round8 pairing coverage is not exactly 5 seeds x 5 attacks x 2 datasets.")

    seed_rows=[]; stat_rows=[]
    for mi,metric in enumerate(CFG["primary_metrics"]):
        for di,dataset in enumerate(CFG["datasets"]):
            for ai,attack in enumerate(CFG["attacks"]):
                g=q[(q.dataset==dataset)&(q.attack==attack)].sort_values("seed")
                vals=[]; levels=[]
                for r in g.itertuples(index=False):
                    v,l=metric_seed_values(r,metric)
                    vals.append(v); levels.append(l)
                    seed_rows.append({
                        "metric":metric,"dataset":dataset,"attack":attack,"seed":int(r.seed),
                        "global_round":int(r.global_round),"seed_level_contrast":v,
                        "reported_level":l,"finite":bool(np.isfinite(v))
                    })
                a=np.asarray(vals,dtype=float); lev=np.asarray(levels,dtype=float)
                full=bool(len(a)==5 and np.isfinite(a).all())
                mean=float(np.mean(a)) if full else math.nan
                median=float(np.median(a)) if full else math.nan
                lo,hi,bst=bootstrap_mean_ci(a,mi,di,ai)
                p,pst=exact_sign_flip(a)
                dz,dzst=cohen_dz(a)
                positive=int(np.sum(a>0)) if full else -1
                stat_rows.append({
                    "metric":metric,"dataset":dataset,"attack":attack,"global_round":8,
                    "n_required":5,"n_finite":int(np.isfinite(a).sum()),
                    "mean_contrast":mean,"median_contrast":median,
                    "bootstrap_ci_low":lo,"bootstrap_ci_high":hi,"bootstrap_status":bst,
                    "exact_sign_flip_p_raw":p,"sign_flip_status":pst,
                    "cohen_dz":dz,"cohen_dz_status":dzst,
                    "positive_seed_count":positive,
                    "mean_reported_level":float(np.mean(lev)) if np.isfinite(lev).all() else math.nan,
                    "median_reported_level":float(np.median(lev)) if np.isfinite(lev).all() else math.nan,
                    "dataset_invariant_training_side_metric":metric in {"malicious_client_recall","benign_client_fpr"},
                    "resolution_limited_p_min":0.0625
                })

    seed_df=pd.DataFrame(seed_rows)
    stats=pd.DataFrame(stat_rows)

    family_rows=[]
    stats["holm_p_adjusted"]=math.nan
    stats["holm_status"]=""
    for metric in CFG["primary_metrics"]:
        for dataset in CFG["datasets"]:
            mask=(stats.metric==metric)&(stats.dataset==dataset)
            idx=stats.index[mask].tolist()
            if len(idx)!=5: raise RuntimeError("Holm family does not contain exactly five attacks.")
            pvals=stats.loc[idx,"exact_sign_flip_p_raw"].to_numpy(dtype=float)
            adj,status=holm_adjust(pvals)
            stats.loc[idx,"holm_p_adjusted"]=adj
            stats.loc[idx,"holm_status"]=status
            family_rows.append({
                "metric":metric,"dataset":dataset,"family_size_required":5,
                "family_size_observed":5,"holm_status":status,
                "all_five_raw_p_finite":bool(np.isfinite(pvals).all())
            })

    # Frozen practical gates, without redefining inferential tests.
    gates=[]
    for dataset in CFG["datasets"]:
        for attack in CFG["attacks"]:
            sub=stats[(stats.dataset==dataset)&(stats.attack==attack)]
            removed=sub[sub.metric=="attack_excess_removed_fraction"].iloc[0]
            recall=sub[sub.metric=="malicious_client_recall"].iloc[0]
            fpr=sub[sub.metric=="benign_client_fpr"].iloc[0]
            gates.append({
                "dataset":dataset,"attack":attack,
                "mean_attack_excess_removed_fraction":float(removed.mean_reported_level),
                "mean_removed_fraction_gate_ge_0_25":bool(np.isfinite(removed.mean_reported_level) and removed.mean_reported_level>=0.25),
                "positive_mitigation_seed_count":int(removed.positive_seed_count),
                "positive_mitigation_gate_ge_4_of_5":bool(removed.positive_seed_count>=4),
                "mean_malicious_client_recall":float(recall.mean_reported_level),
                "mean_malicious_recall_gate_ge_0_50":bool(np.isfinite(recall.mean_reported_level) and recall.mean_reported_level>=0.50),
                "mean_benign_client_fpr":float(fpr.mean_reported_level),
                "mean_benign_fpr_gate_le_0_05":bool(np.isfinite(fpr.mean_reported_level) and fpr.mean_reported_level<=0.05),
                "clean_utility_gate_recomputed_on_reserved_test":False
            })

    secondary=summarize_secondary(df)
    resources=parse_resources(RES)

    seed_df.to_csv(OUT/"task66_primary_seed_values.csv",index=False)
    stats.to_csv(OUT/"task66_primary_statistics.csv",index=False)
    pd.DataFrame(family_rows).to_csv(OUT/"task66_holm_families.csv",index=False)
    pd.DataFrame(gates).to_csv(OUT/"task66_practical_gates.csv",index=False)
    secondary.to_csv(OUT/"task66_secondary_round_descriptives.csv",index=False)
    if not resources.empty: resources.to_csv(OUT/"task66_resource_summary.csv",index=False)

    hashes={p.name:sha256_file(p) for p in sorted(OUT.glob("*.csv"))}
    decision={
      "task":66,"protocol_id":CFG["protocol_id"],"head":head,
      "parent_task65_tag":CFG["parent_required_tag"],
      "primary_round":8,"final_seed_count":5,"attacks":CFG["attacks"],"datasets":CFG["datasets"],
      "bootstrap_replicates":20000,"bootstrap_seed":650428,
      "sign_patterns":32,"minimum_attainable_two_sided_p":0.0625,
      "holm_family_size":5,"negative_results_retained":True,
      "best_round_selection_used":False,"best_seed_selection_used":False,
      "task65_rerun_used":False,"reserved_npz_accessed":False,"models_loaded":False,
      "output_sha256":hashes
    }
    decision_path.write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("TASK 66 C1 FINAL STATISTICS COMPLETE")
    print("PRIMARY STATISTIC ROWS:",len(stats))
    print("PRIMARY SEED VALUE ROWS:",len(seed_df))
    print("HOLM FAMILIES:",len(family_rows))
    print("PRACTICAL GATE ROWS:",len(gates))
    print("SECONDARY DESCRIPTIVE ROWS:",len(secondary))
    print("MINIMUM ATTAINABLE TWO-SIDED P: 0.0625")
    print("TASK65 RERUN USED: False")
    print("RESERVED NPZ ACCESSED: False")
    print("DO NOT RERUN TASK66 C1.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())

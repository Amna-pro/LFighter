
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_independent_anchor_v392 as base

FEATURES=["hist_js","hist_l1","hist_fro","hist_diag_loss","hist_offdiag_growth",
          "cons_js","cons_l1","cons_fro","cons_diag_loss","cons_offdiag_growth"]
CANDIDATE="candidate_max_anchor_absolute"
SCORE=CANDIDATE+"_ema"

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--runs-root",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--seeds",default="42,7,99,123,2026")
    p.add_argument("--development-seed",type=int,default=42)
    p.add_argument("--ema-decay",type=float,default=.65)
    p.add_argument("--clean-threshold-quantile",type=float,default=.95)
    return p.parse_args()

def qhigh(x,q):
    try:return float(np.quantile(np.asarray(x,float),q,method="higher"))
    except TypeError:return float(np.quantile(np.asarray(x,float),q,interpolation="higher"))

def profiles(mats, rounds):
    clients=sorted({c for _,c in mats})
    out={}
    for c in clients:
        xs=[mats[(r,c)] for r in rounds if (r,c) in mats]
        if len(xs)!=len(rounds):
            raise ValueError(f"Missing calibration matrix client={c}, rounds={rounds}")
        out[c]=base.norm(np.median(np.stack(xs),axis=0))
    return out

def score_rows(meta,mats,prof):
    cons={int(r):base.consensus(mats,int(r)) for r in sorted(meta["round"].unique())}
    rows=[]
    for x in meta.itertuples(index=False):
        r,c=int(x.round),int(x.client_id)
        rows.append({"round":r,"client_id":c,"actual_malicious":bool(x.actual_malicious),
                     **base.feat(mats[(r,c)],prof[c],cons[r])})
    return pd.DataFrame(rows).sort_values(["round","client_id"]).reset_index(drop=True)

def clean_loo(meta,mats,prefix):
    rows=[]
    for r in prefix:
        other=[x for x in prefix if x!=r]
        if not other:
            raise ValueError("At least two calibration rounds are required")
        prof=profiles(mats,other)
        part=meta[meta["round"].eq(r)]
        cons=base.consensus(mats,r)
        for x in part.itertuples(index=False):
            c=int(x.client_id)
            rows.append({"round":r,"client_id":c,"actual_malicious":False,
                         **base.feat(mats[(r,c)],prof[c],cons)})
    return pd.DataFrame(rows).sort_values(["round","client_id"]).reset_index(drop=True)

def calibrate(clean_raw,attack_raw,decay):
    clean=clean_raw.copy(); attack=attack_raw.copy(); cal=[]
    for f in FEATURES:
        med,scale=base.robust(clean[f])
        cal.append({"feature":f,"clean_median":med,"clean_scale":scale})
        clean["z_"+f]=base.pz(clean[f],med,scale)
        attack["z_"+f]=base.pz(attack[f],med,scale)
    clean=base.add_ema(base.add_candidates(clean),decay)
    attack=base.add_ema(base.add_candidates(attack),decay)
    return clean,attack,pd.DataFrame(cal)

def evaluate(protocol,seed,clean,attack,q):
    th=qhigh(clean[SCORE],q)
    s=attack[SCORE].to_numpy(float)
    y=attack.actual_malicious.astype(bool).to_numpy()
    pred=s>th; b=~y
    final=attack["round"].eq(attack["round"].max()).to_numpy()
    return {"protocol":protocol,"seed":seed,"clean_threshold":th,
      "clean_false_positive_rate":float(np.mean(clean[SCORE].to_numpy(float)>th)),
      "attack_row_roc_auc":float(roc_auc_score(y,s)),
      "attack_row_pr_auc":float(average_precision_score(y,s)),
      "attack_row_malicious_recall":float(np.mean(pred[y])),
      "attack_row_precision":float(np.sum(pred&y)/max(np.sum(pred),1)),
      "attack_row_benign_false_positive_rate":float(np.mean(pred[b])),
      "final_round_malicious_recall":float(np.mean(pred[y&final])),
      "final_round_benign_false_positive_rate":float(np.mean(pred[b&final]))}

def run_protocol(protocol,seed,cal_meta,cal_mats,attack_meta,attack_mats,prefix,decay,q):
    clean_raw=clean_loo(cal_meta,cal_mats,prefix)
    attack_raw=score_rows(attack_meta,attack_mats,profiles(cal_mats,prefix))
    clean,attack,cal=calibrate(clean_raw,attack_raw,decay)
    return evaluate(protocol,seed,clean,attack,q),clean,attack,cal

def main():
    a=parse()
    seeds=[int(x) for x in a.seeds.split(",") if x.strip()]
    out=a.output_dir.resolve()
    tables=out/"tables"; figs=out/"figures"
    tables.mkdir(parents=True,exist_ok=True)
    figs.mkdir(parents=True,exist_ok=True)

    clean={}; attack={}
    for seed in seeds:
        audit=[]
        cm,cmats,n=base.records(base.load_long(a.runs_root,"clean",seed),seed,"clean",audit)
        am,amats,n2=base.records(base.load_long(a.runs_root,"strong_attack",seed),seed,"strong_attack",audit)
        if n!=n2:raise ValueError(f"Class mismatch seed {seed}")
        clean[seed]=(cm,cmats); attack[seed]=(am,amats)

    rows=[]
    for seed in seeds:
        cm,cmats=clean[seed]; am,amats=attack[seed]
        rounds=sorted(cm["round"].unique().tolist())
        for k in (2,4):
            protocol=f"same_seed_prefix_{k}"
            metric,cs,ats,cal=run_protocol(protocol,seed,cm,cmats,am,amats,rounds[:k],a.ema_decay,a.clean_threshold_quantile)
            rows.append(metric)
            cs.to_csv(tables/f"{protocol}_seed_{seed}_clean.csv",index=False)
            ats.to_csv(tables/f"{protocol}_seed_{seed}_attack.csv",index=False)
            cal.to_csv(tables/f"{protocol}_seed_{seed}_calibration.csv",index=False)

        dcm,dcmats=clean[a.development_seed]
        drounds=sorted(dcm["round"].unique().tolist())[:4]
        protocol=f"external_seed{a.development_seed}_prefix_4"
        metric,cs,ats,cal=run_protocol(protocol,seed,dcm,dcmats,am,amats,drounds,a.ema_decay,a.clean_threshold_quantile)
        rows.append(metric)
        ats.to_csv(tables/f"{protocol}_seed_{seed}_attack.csv",index=False)

    metrics=pd.DataFrame(rows)
    metrics.to_csv(tables/"causal_calibration_metrics_all_seeds.csv",index=False)
    metrics[metrics.seed!=a.development_seed].to_csv(tables/"causal_calibration_heldout_metrics.csv",index=False)
    agg=metrics.groupby("protocol",as_index=False).agg(
      seed_count=("seed","size"),mean_roc_auc=("attack_row_roc_auc","mean"),
      minimum_roc_auc=("attack_row_roc_auc","min"),mean_pr_auc=("attack_row_pr_auc","mean"),
      minimum_pr_auc=("attack_row_pr_auc","min"),
      mean_malicious_recall=("attack_row_malicious_recall","mean"),
      minimum_malicious_recall=("attack_row_malicious_recall","min"),
      mean_benign_false_positive_rate=("attack_row_benign_false_positive_rate","mean"),
      maximum_benign_false_positive_rate=("attack_row_benign_false_positive_rate","max"),
      mean_final_malicious_recall=("final_round_malicious_recall","mean"),
      minimum_final_malicious_recall=("final_round_malicious_recall","min"),
      maximum_final_benign_false_positive_rate=("final_round_benign_false_positive_rate","max"))
    agg.to_csv(tables/"causal_calibration_aggregate_summary.csv",index=False)

    fig,ax=plt.subplots(figsize=(10,6))
    x=np.arange(len(agg)); w=.35
    ax.bar(x-w/2,agg.mean_pr_auc,w,label="Mean PR-AUC")
    ax.bar(x+w/2,agg.mean_malicious_recall,w,label="Mean malicious recall")
    ax.set_xticks(x,agg.protocol,rotation=20,ha="right")
    ax.set_ylim(0,1); ax.set_title("V3.9.3 causal calibration audit")
    ax.legend(); ax.grid(axis="y",alpha=.25)
    fig.tight_layout()
    fig.savefig(figs/"causal_calibration.png",dpi=300)
    fig.savefig(figs/"causal_calibration.pdf")
    plt.close(fig)

    meta={"experiment_version":"3.9.3","frozen_candidate":CANDIDATE,
      "frozen_score_variant":"ema","candidate_selection_reopened":False,
      "uses_current_global_reference":False,"uses_future_clean_rounds":False,
      "stable_client_identity_required":True,"test_sets_accessed":False,
      "protocols":["same_seed_prefix_2","same_seed_prefix_4",f"external_seed{a.development_seed}_prefix_4"]}
    with open(out/"causal_anchor_v393_metadata.json","w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)

    print("Causal Anchor Calibration Audit V3.9.3 complete")
    print("Frozen candidate:",CANDIDATE)
    print("Frozen score variant: ema")
    print("Uses future clean rounds: False")
    print("Tables:",tables)
    print("PNG and PDF figures:",figs)

if __name__=="__main__":
    main()

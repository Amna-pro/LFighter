
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

CANDIDATES = [
    "candidate_historical_absolute",
    "candidate_consensus_absolute",
    "candidate_dual_anchor_absolute",
    "candidate_max_anchor_absolute",
]

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--runs-root",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--seeds",default="42,7,99,123,2026")
    p.add_argument("--development-seed",type=int,default=42)
    p.add_argument("--ema-decay",type=float,default=.65)
    p.add_argument("--clean-threshold-quantile",type=float,default=.95)
    p.add_argument("--max-development-clean-fpr",type=float,default=.05)
    p.add_argument("--max-development-attack-benign-fpr",type=float,default=.05)
    p.add_argument("--max-development-final-benign-fpr",type=float,default=.10)
    return p.parse_args()

def qhigh(x,q):
    try: return float(np.quantile(np.asarray(x,float),q,method="higher"))
    except TypeError: return float(np.quantile(np.asarray(x,float),q,interpolation="higher"))

def norm(m):
    m=np.clip(np.asarray(m,float),1e-12,None)
    return m/np.clip(m.sum(1,keepdims=True),1e-12,None)

def js(a,b):
    a,b=norm(a),norm(b); m=.5*(a+b)
    return .5*(np.sum(a*np.log(a/m),1)+np.sum(b*np.log(b/m),1))

def top5_growth(a,b):
    d=a-b; v=np.maximum(d[~np.eye(d.shape[0],dtype=bool)],0)
    k=min(5,len(v)); return float(np.partition(v,-k)[-k:].mean()) if k else 0.

def robust(x):
    x=np.asarray(x,float); med=float(np.median(x)); mad=1.4826*float(np.median(np.abs(x-med)))
    if mad<1e-9:
        mad=float(np.std(x))
    return med, mad if mad>=1e-9 else 1.

def pz(x,c,s): return np.clip(np.maximum((np.asarray(x,float)-c)/max(s,1e-12),0),0,20)

def load_long(root,mode,seed):
    p=root/mode/f"seed_{seed}"/"tables"/"transition_signature_long.csv"
    if not p.exists(): raise FileNotFoundError(p)
    return pd.read_csv(p)

def records(df, seed, mode, audit_rows):
    required={"round","client_id","actual_malicious","source_id","target_id","local_probability"}
    missing=required.difference(df.columns)
    if missing:
        raise ValueError(f"{mode} seed {seed} transition table is missing columns: {sorted(missing)}")

    source_ids=sorted(pd.Series(df.source_id).dropna().astype(int).unique().tolist())
    target_ids=sorted(pd.Series(df.target_id).dropna().astype(int).unique().tolist())
    if source_ids != target_ids:
        raise ValueError(
            f"{mode} seed {seed} has different source/target class IDs: "
            f"source={source_ids}, target={target_ids}"
        )
    n=len(source_ids)
    expected_cells=n*n
    meta=df[["round","client_id","actual_malicious"]].drop_duplicates().sort_values(["round","client_id"])
    mats={}
    total_exact_duplicate_rows=0

    for r in meta.itertuples(index=False):
        raw=df[(df.round==r.round)&(df.client_id==r.client_id)].copy()
        raw_rows=len(raw)
        key_cols=["source_id","target_id"]
        duplicate_mask=raw.duplicated(key_cols,keep=False)
        duplicate_rows=int(duplicate_mask.sum())
        duplicate_groups=int(raw.loc[duplicate_mask,key_cols].drop_duplicates().shape[0])

        conflicts=0
        if duplicate_groups:
            spread=(
                raw.groupby(key_cols,as_index=False)["local_probability"]
                .agg(lambda values: float(np.max(values)-np.min(values)))
            )
            conflicts=int((spread["local_probability"]>1e-12).sum())
            if conflicts:
                bad=spread[spread["local_probability"]>1e-12].head(10)
                raise ValueError(
                    f"Conflicting duplicate transition cells in {mode} seed {seed}, "
                    f"round={r.round}, client={r.client_id}. First conflicts: "
                    f"{bad.to_dict(orient='records')}"
                )

        # Collapse only numerically identical duplicate coordinate rows.
        clean=(
            raw.groupby(key_cols,as_index=False)
            .agg(local_probability=("local_probability","last"))
        )
        unique_cells=len(clean)
        missing_cells=expected_cells-unique_cells
        extra_cells=max(unique_cells-expected_cells,0)
        audit_rows.append({
            "seed":int(seed),
            "mode":str(mode),
            "round":int(r.round),
            "client_id":int(r.client_id),
            "raw_rows":int(raw_rows),
            "expected_cells":int(expected_cells),
            "unique_coordinate_cells":int(unique_cells),
            "duplicate_rows_involved":int(duplicate_rows),
            "duplicate_coordinate_groups":int(duplicate_groups),
            "conflicting_duplicate_groups":int(conflicts),
            "missing_coordinate_cells":int(max(missing_cells,0)),
            "extra_coordinate_cells":int(extra_cells),
            "exact_duplicate_rows_collapsed":int(raw_rows-unique_cells),
        })
        total_exact_duplicate_rows += int(raw_rows-unique_cells)

        if unique_cells != expected_cells:
            observed=set(zip(clean.source_id.astype(int),clean.target_id.astype(int)))
            expected={(s,t) for s in source_ids for t in target_ids}
            missing_pairs=sorted(expected-observed)[:20]
            extra_pairs=sorted(observed-expected)[:20]
            raise ValueError(
                f"Invalid transition matrix in {mode} seed {seed}, round={r.round}, "
                f"client={r.client_id}: raw_rows={raw_rows}, unique_cells={unique_cells}, "
                f"expected={expected_cells}, missing_pairs={missing_pairs}, "
                f"extra_pairs={extra_pairs}"
            )

        pivot=clean.pivot(index="source_id",columns="target_id",values="local_probability")
        pivot=pivot.reindex(index=source_ids,columns=target_ids)
        if pivot.isna().any().any():
            raise ValueError(
                f"NaN transition cells after pivot in {mode} seed {seed}, "
                f"round={r.round}, client={r.client_id}"
            )
        mats[(int(r.round),int(r.client_id))]=norm(pivot.to_numpy(float))

    print(
        f"Input integrity {mode} seed {seed}: matrices={len(mats)}, "
        f"classes={n}, exact duplicate rows collapsed={total_exact_duplicate_rows}"
    )
    return meta.reset_index(drop=True),mats,n

def profile(clean_mats,cid,exclude=None):
    xs=[m for (r,c),m in clean_mats.items() if c==cid and (exclude is None or r!=exclude)]
    if len(xs)<2: raise ValueError(f"Insufficient clean history for client {cid}")
    return norm(np.median(np.stack(xs),0))

def consensus(mats,round_id):
    xs=[m for (r,_),m in mats.items() if r==round_id]
    return norm(np.median(np.stack(xs),0))

def feat(local,hist,cons):
    hd,cd=local-hist,local-cons
    return {
      "hist_js":float(js(local,hist).mean()),
      "hist_l1":float(np.abs(hd).mean()),
      "hist_fro":float(np.linalg.norm(hd)),
      "hist_diag_loss":float(np.maximum(np.diag(hist)-np.diag(local),0).mean()),
      "hist_offdiag_growth":top5_growth(local,hist),
      "cons_js":float(js(local,cons).mean()),
      "cons_l1":float(np.abs(cd).mean()),
      "cons_fro":float(np.linalg.norm(cd)),
      "cons_diag_loss":float(np.maximum(np.diag(cons)-np.diag(local),0).mean()),
      "cons_offdiag_growth":top5_growth(local,cons),
    }

def score(meta,mats,clean_mats,loo):
    cons={int(r):consensus(mats,int(r)) for r in meta["round"].unique()}
    rows=[]
    for x in meta.itertuples(index=False):
        r,c=int(x.round),int(x.client_id)
        rows.append({"round":r,"client_id":c,"actual_malicious":bool(x.actual_malicious),
                     **feat(mats[(r,c)],profile(clean_mats,c,r if loo else None),cons[r])})
    return pd.DataFrame(rows).sort_values(["round","client_id"]).reset_index(drop=True)

def normalize_features(clean,attack):
    cols=["hist_js","hist_l1","hist_fro","hist_diag_loss","hist_offdiag_growth",
          "cons_js","cons_l1","cons_fro","cons_diag_loss","cons_offdiag_growth"]
    prof=[]
    for c in cols:
        med,sc=robust(clean[c]); prof.append({"feature":c,"clean_median":med,"clean_scale":sc})
        clean["z_"+c]=pz(clean[c],med,sc); attack["z_"+c]=pz(attack[c],med,sc)
    return clean,attack,pd.DataFrame(prof)

def add_candidates(d):
    h=d[["z_hist_js","z_hist_l1","z_hist_diag_loss","z_hist_offdiag_growth"]].mean(1)
    c=d[["z_cons_js","z_cons_l1","z_cons_diag_loss","z_cons_offdiag_growth"]].mean(1)
    d["candidate_historical_absolute"]=h
    d["candidate_consensus_absolute"]=c
    d["candidate_dual_anchor_absolute"]=.7*h+.3*c
    d["candidate_max_anchor_absolute"]=np.maximum(h,c)
    return d

def add_ema(d,decay):
    d=d.sort_values(["client_id","round"]).copy()
    for c in CANDIDATES:
        out=pd.Series(index=d.index,dtype=float)
        for _,g in d.groupby("client_id",sort=False):
            prev=0.
            for idx in g.index:
                prev=decay*prev+(1-decay)*float(d.at[idx,c]); out.at[idx]=prev
        d[c+"_ema"]=out
        d[c+"_max_instant_ema"]=np.maximum(d[c],d[c+"_ema"])
    return d.sort_values(["round","client_id"]).reset_index(drop=True)

def metrics(seed,candidate,variant,clean,attack,q):
    col=candidate if variant=="instant" else candidate+"_"+variant
    th=qhigh(clean[col],q); y=attack.actual_malicious.astype(bool).to_numpy()
    s=attack[col].to_numpy(float); pred=s>th; b=~y; final=attack["round"].eq(attack["round"].max()).to_numpy()
    return {
      "seed":seed,"candidate":candidate,"score_variant":variant,"score_column":col,
      "clean_threshold":th,"clean_false_positive_rate":float(np.mean(clean[col].to_numpy(float)>th)),
      "attack_row_roc_auc":float(roc_auc_score(y,s)),
      "attack_row_pr_auc":float(average_precision_score(y,s)),
      "attack_row_malicious_recall":float(np.mean(pred[y])),
      "attack_row_precision":float(np.sum(pred&y)/max(np.sum(pred),1)),
      "attack_row_benign_false_positive_rate":float(np.mean(pred[b])),
      "final_round_malicious_recall":float(np.mean(pred[y&final])),
      "final_round_benign_false_positive_rate":float(np.mean(pred[b&final])),
    }

def main():
    a=args(); seeds=[int(x) for x in a.seeds.split(",") if x.strip()]
    out=a.output_dir.resolve(); tables=out/"tables"; figs=out/"figures"
    tables.mkdir(parents=True,exist_ok=True); figs.mkdir(parents=True,exist_ok=True)
    allm=[]; profs=[]; integrity_audit=[]
    for seed in seeds:
        cm,cmats,n=records(
            load_long(a.runs_root,"clean",seed),
            seed=seed,
            mode="clean",
            audit_rows=integrity_audit,
        )
        am,amats,n2=records(
            load_long(a.runs_root,"strong_attack",seed),
            seed=seed,
            mode="strong_attack",
            audit_rows=integrity_audit,
        )
        if n!=n2: raise ValueError("Class count mismatch")
        c=score(cm,cmats,cmats,True); x=score(am,amats,cmats,False)
        c,x,p=normalize_features(c,x); c=add_ema(add_candidates(c),a.ema_decay); x=add_ema(add_candidates(x),a.ema_decay)
        c.insert(0,"seed",seed); x.insert(0,"seed",seed); p.insert(0,"seed",seed); profs.append(p)
        c.to_csv(tables/f"seed_{seed}_clean_scores.csv",index=False)
        x.to_csv(tables/f"seed_{seed}_attack_scores.csv",index=False)
        for cand in CANDIDATES:
            for var in ["instant","ema","max_instant_ema"]:
                allm.append(metrics(seed,cand,var,c,x,a.clean_threshold_quantile))
    m=pd.DataFrame(allm); m.to_csv(tables/"candidate_metrics_all_seeds.csv",index=False)
    pd.concat(profs,ignore_index=True).to_csv(tables/"clean_feature_profiles.csv",index=False)
    integrity_table=pd.DataFrame(integrity_audit)
    integrity_table.to_csv(tables/"input_transition_integrity_audit.csv",index=False)
    integrity_summary=(
        integrity_table.groupby(["seed","mode"],as_index=False)
        .agg(
            matrices=("client_id","size"),
            raw_rows=("raw_rows","sum"),
            unique_coordinate_cells=("unique_coordinate_cells","sum"),
            exact_duplicate_rows_collapsed=("exact_duplicate_rows_collapsed","sum"),
            conflicting_duplicate_groups=("conflicting_duplicate_groups","sum"),
            missing_coordinate_cells=("missing_coordinate_cells","sum"),
            extra_coordinate_cells=("extra_coordinate_cells","sum"),
        )
    )
    integrity_summary.to_csv(tables/"input_transition_integrity_summary.csv",index=False)
    d=m[m.seed==a.development_seed].copy()
    d["passes_all_safety_constraints"]=(d.clean_false_positive_rate<=a.max_development_clean_fpr)&(d.attack_row_benign_false_positive_rate<=a.max_development_attack_benign_fpr)&(d.final_round_benign_false_positive_rate<=a.max_development_final_benign_fpr)
    e=d[d.passes_all_safety_constraints].sort_values(["attack_row_pr_auc","attack_row_roc_auc","final_round_malicious_recall"],ascending=[False,False,False])
    if e.empty: raise RuntimeError("No V3.9 candidate satisfies frozen development safety constraints")
    sel=e.iloc[0]; d.sort_values(["passes_all_safety_constraints","attack_row_pr_auc"],ascending=[False,False]).to_csv(tables/"development_candidate_ranking.csv",index=False)
    e.to_csv(tables/"development_safety_eligible_candidates.csv",index=False)
    held=m[(m.candidate==sel.candidate)&(m.score_variant==sel.score_variant)&(m.seed!=a.development_seed)].copy()
    held.to_csv(tables/"selected_candidate_heldout_metrics.csv",index=False)
    agg=pd.DataFrame([{
      "selected_candidate":sel.candidate,"selected_score_variant":sel.score_variant,
      "heldout_seed_count":len(held),"heldout_mean_roc_auc":held.attack_row_roc_auc.mean(),
      "heldout_mean_pr_auc":held.attack_row_pr_auc.mean(),
      "heldout_mean_malicious_recall":held.attack_row_malicious_recall.mean(),
      "heldout_mean_benign_false_positive_rate":held.attack_row_benign_false_positive_rate.mean(),
      "heldout_mean_final_malicious_recall":held.final_round_malicious_recall.mean(),
      "heldout_mean_final_benign_false_positive_rate":held.final_round_benign_false_positive_rate.mean(),
      "uses_current_global_reference":False,"trusted_clean_history_required":True,
      "selection_frozen_before_heldout_review":True,"test_sets_accessed":False
    }]); agg.to_csv(tables/"selected_candidate_aggregate_summary.csv",index=False)
    s=m.groupby(["candidate","score_variant"],as_index=False).attack_row_pr_auc.mean().sort_values("attack_row_pr_auc")
    fig,ax=plt.subplots(figsize=(10,7)); ax.barh(s.candidate.str.replace("candidate_","",regex=False)+" | "+s.score_variant,s.attack_row_pr_auc); ax.set_xlabel("Mean PR-AUC"); ax.set_title("V3.9 independent-anchor candidates"); fig.tight_layout(); fig.savefig(figs/"candidate_pr_auc.png",dpi=300); fig.savefig(figs/"candidate_pr_auc.pdf"); plt.close(fig)
    json.dump({"experiment_version":"3.9.1","selected_candidate":str(sel.candidate),"selected_score_variant":str(sel.score_variant),"uses_current_global_reference":False,"test_sets_accessed":False},open(out/"independent_anchor_v39_metadata.json","w"),indent=2)
    print("Independent Anchor Audit V3.9.1 complete")
    print("Selected candidate:",sel.candidate)
    print("Selected score variant:",sel.score_variant)
    print("Uses current global reference: False")
    print("Tables:",tables)
    print("PNG and PDF figures:",figs)

if __name__=="__main__":
    main()

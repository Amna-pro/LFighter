from __future__ import annotations
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT/"configs/task67_c0_preregistration_v4300.json").read_text(encoding="utf-8"))
IN = ROOT/CFG["input_root"]
AUDIT = ROOT/CFG["audit_input"]
OUT = ROOT/CFG["output_root"]
FIG = OUT/"figures"
TAB = OUT/"tables"

PRIMARY = IN/"task66_primary_statistics.csv"
SEED = IN/"task66_primary_seed_values.csv"
HOLM = IN/"task66_holm_families.csv"
GATES = IN/"task66_practical_gates.csv"
SECONDARY = IN/"task66_secondary_round_descriptives.csv"
RESOURCES = IN/"task66_resource_summary.csv"
COMPLETE = IN/"TASK66_FINAL_STATISTICS_COMPLETE.json"

ATTACK_LABEL = {
    "all_to_one_benign":"All→Benign",
    "cyclic_shift":"Cyclic shift",
    "multiclass_partial_cycle":"Partial cycle",
    "pairwise_swap":"Pairwise swap",
    "random_flip":"Random flip",
}
METRIC_LABEL = {
    "macro_f1":"Macro F1",
    "balanced_accuracy":"Balanced accuracy",
    "worst_class_recall":"Worst-class recall",
    "attack_excess_removed_fraction":"Attack excess removed",
    "malicious_client_recall":"Malicious-client recall margin",
    "benign_client_fpr":"Benign-client FPR margin",
}

def git(*args):
    return subprocess.run(["git",*args],cwd=ROOT,check=True,text=True,capture_output=True).stdout.strip()

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def require_frozen_execution():
    head=git("rev-parse","HEAD")
    tag=git("rev-list","-n","1",CFG["required_execution_tag"])
    if head != tag:
        raise RuntimeError(f"Task67 C1 requires HEAD exactly at {CFG['required_execution_tag']}")
    if git("status","--porcelain"):
        raise RuntimeError("Repository must be clean for Task67 C1.")
    if OUT.exists():
        raise RuntimeError("Task67 output root already exists; rerun prohibited.")
    a=json.loads(AUDIT.read_text(encoding="utf-8"))
    if a.get("all_checks_passed") is not True or a.get("ready_for_task67_publication_outputs") is not True:
        raise RuntimeError("Task66 C2 audit does not authorize Task67.")

def savefig(fig, stem):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/f"{stem}.png",dpi=int(CFG["png_dpi"]),bbox_inches="tight")
    fig.savefig(FIG/f"{stem}.pdf",bbox_inches="tight")
    plt.close(fig)

def ordered_attack_frame(df):
    d=df.copy()
    d["attack"]=pd.Categorical(d["attack"],categories=CFG["attacks"],ordered=True)
    return d.sort_values("attack")

def contrast_bar(stats, metric, dataset, stem):
    d=ordered_attack_frame(stats[(stats.metric==metric)&(stats.dataset==dataset)])
    fig,ax=plt.subplots(figsize=(8.2,4.8))
    x=np.arange(len(d))
    y=pd.to_numeric(d["mean_contrast"],errors="coerce").to_numpy(float)
    ax.bar(x,np.nan_to_num(y,nan=0.0))
    ax.axhline(0,linewidth=1)
    for i,v in enumerate(y):
        if np.isfinite(v):
            ax.text(i,v,f"{v:.3f}",ha="center",va="bottom" if v>=0 else "top",fontsize=8)
        else:
            ax.text(i,0,"NE",ha="center",va="bottom",fontsize=8)
    ax.set_xticks(x, [ATTACK_LABEL[a] for a in d["attack"].astype(str)], rotation=25, ha="right")
    ax.set_ylabel("Mean seed-level contrast")
    ax.set_title(f"{METRIC_LABEL[metric]} — {dataset.capitalize()} — Round 8")
    ax.text(0.01,0.01,"Positive values follow the frozen favorable direction; NE = not estimable.",transform=ax.transAxes,fontsize=8)
    fig.tight_layout()
    savefig(fig,stem)

def seed_strip(seed, metric, dataset, stem):
    d=seed[(seed.metric==metric)&(seed.dataset==dataset)].copy()
    fig,ax=plt.subplots(figsize=(8.4,4.8))
    for i,attack in enumerate(CFG["attacks"]):
        q=d[d.attack==attack].sort_values("seed")
        vals=pd.to_numeric(q["seed_level_contrast"],errors="coerce").to_numpy(float)
        xs=np.full(len(vals),i,dtype=float)
        finite=np.isfinite(vals)
        ax.scatter(xs[finite],vals[finite],s=35)
        if finite.any():
            ax.plot([i-0.18,i+0.18],[np.mean(vals[finite]),np.mean(vals[finite])],linewidth=2)
    ax.axhline(0,linewidth=1)
    ax.set_xticks(range(len(CFG["attacks"])),[ATTACK_LABEL[a] for a in CFG["attacks"]],rotation=25,ha="right")
    ax.set_ylabel("Seed-level contrast")
    ax.set_title(f"Seed-level {METRIC_LABEL[metric]} — {dataset.capitalize()} — Round 8")
    fig.tight_layout()
    savefig(fig,stem)

def heatmap(stats, value_col, title, stem, fmt):
    rows=[]
    labels=[]
    for metric in CFG["metrics"]:
        for dataset in CFG["datasets"]:
            q=ordered_attack_frame(stats[(stats.metric==metric)&(stats.dataset==dataset)])
            rows.append(pd.to_numeric(q[value_col],errors="coerce").to_numpy(float))
            labels.append(f"{METRIC_LABEL[metric]} | {dataset}")
    arr=np.vstack(rows)
    fig,ax=plt.subplots(figsize=(10.5,7.2))
    im=ax.imshow(np.ma.masked_invalid(arr),aspect="auto")
    ax.set_xticks(range(len(CFG["attacks"])),[ATTACK_LABEL[a] for a in CFG["attacks"]],rotation=25,ha="right")
    ax.set_yticks(range(len(labels)),labels)
    ax.set_title(title)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            v=arr[r,c]
            ax.text(c,r,("NE" if not np.isfinite(v) else format(v,fmt)),ha="center",va="center",fontsize=7)
    fig.colorbar(im,ax=ax,fraction=0.03,pad=0.02)
    fig.tight_layout()
    savefig(fig,stem)

def bootstrap_forest(stats, metric, stem):
    d=stats[stats.metric==metric].copy()
    fig,ax=plt.subplots(figsize=(9.2,6.4))
    y=[]; labels=[]; means=[]; lo=[]; hi=[]
    idx=0
    for dataset in CFG["datasets"]:
        for attack in CFG["attacks"]:
            q=d[(d.dataset==dataset)&(d.attack==attack)]
            if len(q)!=1: continue
            r=q.iloc[0]
            y.append(idx)
            labels.append(f"{dataset[:4].upper()} | {ATTACK_LABEL[attack]}")
            means.append(float(r["mean_contrast"]) if pd.notna(r["mean_contrast"]) else math.nan)
            lo.append(float(r["bootstrap_ci_low"]) if pd.notna(r["bootstrap_ci_low"]) else math.nan)
            hi.append(float(r["bootstrap_ci_high"]) if pd.notna(r["bootstrap_ci_high"]) else math.nan)
            idx+=1
    means=np.asarray(means,float); lo=np.asarray(lo,float); hi=np.asarray(hi,float); y=np.asarray(y)
    finite=np.isfinite(means)&np.isfinite(lo)&np.isfinite(hi)
    ax.errorbar(means[finite],y[finite],xerr=np.vstack([means[finite]-lo[finite],hi[finite]-means[finite]]),fmt="o",capsize=3)
    for yy,m in zip(y[~finite],means[~finite]):
        ax.text(0,yy,"NE",ha="center",va="center",fontsize=8)
    ax.axvline(0,linewidth=1)
    ax.set_yticks(y,labels)
    ax.set_xlabel("Mean contrast with frozen 95% percentile-bootstrap CI")
    ax.set_title(f"{METRIC_LABEL[metric]} — Round 8")
    fig.tight_layout()
    savefig(fig,stem)

def trajectory(secondary, metric, stem):
    d=secondary[secondary.metric==metric].copy()
    fig,ax=plt.subplots(figsize=(9.2,5.4))
    for dataset in CFG["datasets"]:
        for attack in CFG["attacks"]:
            q=d[(d.dataset==dataset)&(d.attack==attack)].sort_values("global_round")
            ax.plot(q["global_round"],q["mean_contrast"],marker="o",label=f"{dataset[:4].upper()} | {ATTACK_LABEL[attack]}")
    ax.axhline(0,linewidth=1)
    ax.set_xticks([5,6,7,8])
    ax.set_xlabel("Global round")
    ax.set_ylabel("Mean seed-level contrast")
    ax.set_title(f"Descriptive round 5–8 trajectory — {METRIC_LABEL[metric]}")
    ax.legend(fontsize=7,ncol=2,loc="best")
    fig.tight_layout()
    savefig(fig,stem)

def resource_bar(resources, value_col, ylabel, title, stem):
    d=resources.copy()
    labels=[]
    vals=[]
    for r in d.itertuples(index=False):
        attack="" if pd.isna(r.attack) else str(r.attack)
        labels.append(str(r.stage_type) + (f" | {ATTACK_LABEL.get(attack,attack)}" if attack else ""))
        vals.append(float(getattr(r,value_col)))
    fig,ax=plt.subplots(figsize=(10.2,5.8))
    x=np.arange(len(vals))
    ax.bar(x,vals)
    ax.set_xticks(x,labels,rotation=45,ha="right",fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    savefig(fig,stem)

def write_tables(stats,seed,holm,gates,secondary,resources):
    TAB.mkdir(parents=True,exist_ok=True)
    stats.to_csv(TAB/"table01_primary_statistics.csv",index=False)
    seed.to_csv(TAB/"table02_primary_seed_values.csv",index=False)
    holm.to_csv(TAB/"table03_holm_families.csv",index=False)
    gates.to_csv(TAB/"table04_practical_gates.csv",index=False)
    secondary.to_csv(TAB/"table05_secondary_round_descriptives.csv",index=False)
    resources.to_csv(TAB/"table06_resource_summary.csv",index=False)
    boundary=pd.DataFrame([
      {"claim":"Macro F1 / balanced accuracy directionality","status":"Consistent positive mean contrasts across all five attacks in both datasets; exact p resolution and Holm-adjusted p-values must be reported."},
      {"claim":"Conventional p<0.05 significance","status":"Not supported by five-seed exact sign-flip test; minimum attainable two-sided raw p is 0.0625."},
      {"claim":"Worst-class recall improvement","status":"Not supported by frozen Task66 outcome synopsis; negative/null results retained."},
      {"claim":"Attack-excess-removed fraction","status":"Partly non-estimable because frozen seed-level denominators can be non-finite; no seed dropping allowed."},
      {"claim":"Detector safety","status":"Report malicious-client recall and benign-client FPR against frozen practical thresholds; do not convert into unregistered superiority claims."},
      {"claim":"Best run / best round selection","status":"Prohibited."},
    ])
    boundary.to_csv(TAB/"table07_claim_boundary_summary.csv",index=False)

def main():
    require_frozen_execution()
    required=[PRIMARY,SEED,HOLM,GATES,SECONDARY,RESOURCES,COMPLETE,AUDIT]
    if not all(p.exists() for p in required):
        missing=[str(p) for p in required if not p.exists()]
        raise FileNotFoundError(missing)

    stats=pd.read_csv(PRIMARY)
    seed=pd.read_csv(SEED)
    holm=pd.read_csv(HOLM)
    gates=pd.read_csv(GATES)
    secondary=pd.read_csv(SECONDARY)
    resources=pd.read_csv(RESOURCES)

    OUT.mkdir(parents=True)
    write_tables(stats,seed,holm,gates,secondary,resources)

    contrast_bar(stats,"macro_f1","diagnostic","fig01_macro_f1_contrast_diagnostic")
    contrast_bar(stats,"macro_f1","natural","fig02_macro_f1_contrast_natural")
    contrast_bar(stats,"balanced_accuracy","diagnostic","fig03_balanced_accuracy_contrast_diagnostic")
    contrast_bar(stats,"balanced_accuracy","natural","fig04_balanced_accuracy_contrast_natural")
    contrast_bar(stats,"worst_class_recall","diagnostic","fig05_worst_class_recall_contrast_diagnostic")
    contrast_bar(stats,"worst_class_recall","natural","fig06_worst_class_recall_contrast_natural")
    contrast_bar(stats,"attack_excess_removed_fraction","diagnostic","fig07_attack_excess_removed_fraction_diagnostic")
    contrast_bar(stats,"attack_excess_removed_fraction","natural","fig08_attack_excess_removed_fraction_natural")
    contrast_bar(stats,"malicious_client_recall","diagnostic","fig09_malicious_client_recall_margin_diagnostic")
    contrast_bar(stats,"malicious_client_recall","natural","fig10_malicious_client_recall_margin_natural")
    contrast_bar(stats,"benign_client_fpr","diagnostic","fig11_benign_client_fpr_margin_diagnostic")
    contrast_bar(stats,"benign_client_fpr","natural","fig12_benign_client_fpr_margin_natural")

    seed_strip(seed,"macro_f1","diagnostic","fig13_seed_level_macro_f1_diagnostic")
    seed_strip(seed,"macro_f1","natural","fig14_seed_level_macro_f1_natural")
    seed_strip(seed,"balanced_accuracy","diagnostic","fig15_seed_level_balanced_accuracy_diagnostic")
    seed_strip(seed,"balanced_accuracy","natural","fig16_seed_level_balanced_accuracy_natural")
    seed_strip(seed,"worst_class_recall","diagnostic","fig17_seed_level_worst_class_recall_diagnostic")
    seed_strip(seed,"worst_class_recall","natural","fig18_seed_level_worst_class_recall_natural")

    heatmap(stats,"exact_sign_flip_p_raw","Exact paired sign-flip raw p-values (NE retained)","fig19_raw_exact_p_value_heatmap",".3f")
    heatmap(stats,"holm_p_adjusted","Holm-adjusted p-values by metric × dataset × attack","fig20_holm_adjusted_p_value_heatmap",".3f")
    heatmap(stats,"cohen_dz","Cohen dz where defined","fig21_cohen_dz_heatmap",".2f")
    heatmap(stats,"positive_seed_count","Positive seed counts out of five","fig22_positive_seed_count_heatmap",".0f")

    bootstrap_forest(stats,"macro_f1","fig23_bootstrap_ci_macro_f1")
    bootstrap_forest(stats,"balanced_accuracy","fig24_bootstrap_ci_balanced_accuracy")
    bootstrap_forest(stats,"worst_class_recall","fig25_bootstrap_ci_worst_class_recall")

    trajectory(secondary,"macro_f1","fig26_secondary_macro_f1_trajectories")
    trajectory(secondary,"balanced_accuracy","fig27_secondary_balanced_accuracy_trajectories")
    trajectory(secondary,"worst_class_recall","fig28_secondary_worst_class_recall_trajectories")

    resource_bar(resources,"runtime_mean_seconds","Mean runtime (seconds)","Mean runtime by frozen training stage","fig29_runtime_summary")
    resource_bar(resources,"peak_rss_max_bytes","Maximum peak RSS (bytes)","Peak memory by frozen training stage","fig30_peak_rss_summary")

    pngs=sorted(FIG.glob("*.png")); pdfs=sorted(FIG.glob("*.pdf"))
    stems_png={p.stem for p in pngs}; stems_pdf={p.stem for p in pdfs}
    required_stems=set(CFG["required_figure_stems"])
    if stems_png!=required_stems or stems_pdf!=required_stems:
        raise RuntimeError("Publication figure set does not exactly match frozen 30-stem plan.")

    hashes={}
    for p in sorted([*FIG.glob("*"),*TAB.glob("*")]):
        if p.is_file():
            hashes[str(p.relative_to(OUT)).replace("\\","/")]=sha256_file(p)

    decision={
      "task":67,"protocol_id":CFG["protocol_id"],"head":git("rev-parse","HEAD"),
      "parent_task66_tag":CFG["parent_required_tag"],
      "figure_stem_count":len(required_stems),"png_count":len(pngs),"pdf_count":len(pdfs),
      "publication_table_count":len(list(TAB.glob("*.csv"))),
      "task65_rerun_used":False,"task66_inference_rerun_used":False,
      "reserved_npz_accessed":False,"models_loaded":False,"training_used":False,
      "shap_recomputed":False,"llm_used":False,
      "negative_null_nonestimable_results_retained":True,
      "best_seed_selection_used":False,"best_round_selection_used":False,
      "output_sha256":hashes
    }
    (OUT/"TASK67_PUBLICATION_OUTPUTS_COMPLETE.json").write_text(
        json.dumps(decision,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("TASK 67 C1 PUBLICATION OUTPUTS COMPLETE")
    print("FIGURE STEMS:",len(required_stems))
    print("PNG FILES:",len(pngs))
    print("PDF FILES:",len(pdfs))
    print("PUBLICATION TABLES:",len(list(TAB.glob('*.csv'))))
    print("NEGATIVE/NULL/NON-ESTIMABLE RESULTS RETAINED: True")
    print("TASK66 INFERENCE RERUN: False")
    print("TASK65 RERUN: False")
    print("RESERVED NPZ ACCESSED: False")
    print("DO NOT RERUN TASK67 C1.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())

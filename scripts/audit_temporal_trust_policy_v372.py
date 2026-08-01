#!/usr/bin/env python3
"""Offline audit of temporal soft-trust mappings for the frozen V3.7 signal."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POLICY_MODES=('ema_only','instant_only','max_instant_ema','blend_70ema_30instant')
GAMMAS=(0.50,0.75,1.00,1.25,1.50,2.00)
MIN_TRUSTS=(0.10,0.15,0.25)

def args_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--analysis-dir',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--seeds',default='42,7,99,123,2026')
    p.add_argument('--development-seed',type=int,default=42)
    p.add_argument('--candidate',default='candidate_generic_dual_reference')
    p.add_argument('--clean-threshold-quantile',type=float,default=0.95)
    p.add_argument('--low-trust-cutoff',type=float,default=0.50)
    p.add_argument('--count-cap-multiplier',type=float,default=3.0)
    p.add_argument('--min-development-clean-mean-trust',type=float,default=0.97)
    p.add_argument('--max-development-clean-low-trust-rate',type=float,default=0.05)
    p.add_argument('--min-development-attack-benign-mean-trust',type=float,default=0.95)
    p.add_argument('--max-development-attack-benign-low-trust-rate',type=float,default=0.05)
    p.add_argument('--min-development-final-benign-mean-trust',type=float,default=0.90)
    p.add_argument('--max-development-final-benign-low-trust-rate',type=float,default=0.10)
    return p.parse_args()

def qhigher(x,q):
    x=np.asarray(x,float)
    try:return float(np.quantile(x,q,method='higher'))
    except TypeError:return float(np.quantile(x,q,interpolation='higher'))

def savefig(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout(); fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight'); fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight'); plt.close(fig)

def load_seed(tables,seed):
    c=tables/f'seed_{seed}_clean_scored_rows.csv'; a=tables/f'seed_{seed}_attack_scored_rows.csv'
    if not c.exists() or not a.exists(): raise FileNotFoundError(f'Missing scored rows for seed {seed}')
    return pd.read_csv(c),pd.read_csv(a)

def ratio(inst,ema,ti,te,mode):
    ri=inst/max(ti,1e-12); re=ema/max(te,1e-12)
    if mode=='ema_only': return re
    if mode=='instant_only': return ri
    if mode=='max_instant_ema': return np.maximum(ri,re)
    if mode=='blend_70ema_30instant': return 0.7*re+0.3*ri
    raise ValueError(mode)

def apply_policy(df,candidate,ti,te,mode,gamma,min_trust,capmult):
    out=df.copy(); r=ratio(out[candidate].to_numpy(float),out[f'{candidate}_ema'].to_numpy(float),ti,te,mode)
    excess=np.maximum(r-1.0,0.0); out['policy_ratio']=r; out['trust_factor']=np.clip(np.exp(-gamma*excess),min_trust,1.0)
    out['base_aggregation_weight']=0.0; out['aggregation_weight']=0.0
    for _,g in out.groupby('round',sort=True):
        counts=g['client_samples'].to_numpy(float); cap=float(np.median(counts)*capmult); bounded=np.minimum(counts,max(cap,1.0))
        base=bounded/max(float(bounded.sum()),1e-12); weighted=bounded*g['trust_factor'].to_numpy(float); adj=weighted/max(float(weighted.sum()),1e-12)
        out.loc[g.index,'base_aggregation_weight']=base; out.loc[g.index,'aggregation_weight']=adj
    return out

def summarize(seed,mode,gamma,min_trust,clean,attack,cutoff,ti,te):
    clow=clean['trust_factor'].to_numpy(float)<cutoff
    mal=attack['actual_malicious'].astype(bool).to_numpy(); ben=~mal; trust=attack['trust_factor'].to_numpy(float); low=trust<cutoff; final=attack['round'].eq(attack['round'].max()).to_numpy()
    reds=[]
    for _,g in attack.groupby('round',sort=True):
        labs=g['actual_malicious'].astype(bool).to_numpy(); b=float(g.loc[labs,'base_aggregation_weight'].sum()); a=float(g.loc[labs,'aggregation_weight'].sum()); reds.append(1.0-a/max(b,1e-12))
    return {'seed':seed,'policy_mode':mode,'gamma':gamma,'minimum_trust':min_trust,'instant_clean_threshold':ti,'ema_clean_threshold':te,
      'clean_mean_trust':float(clean['trust_factor'].mean()),'clean_minimum_trust':float(clean['trust_factor'].min()),'clean_low_trust_rate':float(clow.mean()),
      'attack_benign_mean_trust':float(np.mean(trust[ben])),'attack_malicious_mean_trust':float(np.mean(trust[mal])),
      'attack_benign_low_trust_rate':float(np.mean(low[ben])),'attack_malicious_low_trust_recall':float(np.mean(low[mal])),
      'final_benign_mean_trust':float(np.mean(trust[ben&final])),'final_malicious_mean_trust':float(np.mean(trust[mal&final])),
      'final_benign_low_trust_rate':float(np.mean(low[ben&final])),'final_malicious_low_trust_recall':float(np.mean(low[mal&final])),
      'mean_malicious_influence_reduction':float(np.mean(reds)),'final_malicious_influence_reduction':float(reds[-1])}

def main():
    a=args_parser(); seeds=[int(x.strip()) for x in a.seeds.split(',') if x.strip()]
    tables=a.analysis_dir.expanduser().resolve()/'tables'; out=a.output_dir.expanduser().resolve(); td=out/'tables'; fd=out/'figures'; td.mkdir(parents=True,exist_ok=True); fd.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        clean,attack=load_seed(tables,seed); ema=f'{a.candidate}_ema'
        if a.candidate not in clean or ema not in clean: raise ValueError('Frozen candidate columns are missing')
        ti=qhigher(clean[a.candidate],a.clean_threshold_quantile); te=qhigher(clean[ema],a.clean_threshold_quantile)
        for mode in POLICY_MODES:
            for gamma in GAMMAS:
                for mt in MIN_TRUSTS:
                    cp=apply_policy(clean,a.candidate,ti,te,mode,gamma,mt,a.count_cap_multiplier); ap=apply_policy(attack,a.candidate,ti,te,mode,gamma,mt,a.count_cap_multiplier)
                    rows.append(summarize(seed,mode,gamma,mt,cp,ap,a.low_trust_cutoff,ti,te))
    m=pd.DataFrame(rows); m.to_csv(td/'trust_policy_metrics_all_seeds.csv',index=False)
    d=m[m.seed.eq(a.development_seed)].copy();
    if d.empty: raise ValueError('Development seed missing')
    d['passes_clean_mean_trust']=d.clean_mean_trust.ge(a.min_development_clean_mean_trust)
    d['passes_clean_low_trust_rate']=d.clean_low_trust_rate.le(a.max_development_clean_low_trust_rate)
    d['passes_attack_benign_mean_trust']=d.attack_benign_mean_trust.ge(a.min_development_attack_benign_mean_trust)
    d['passes_attack_benign_low_trust_rate']=d.attack_benign_low_trust_rate.le(a.max_development_attack_benign_low_trust_rate)
    d['passes_final_benign_mean_trust']=d.final_benign_mean_trust.ge(a.min_development_final_benign_mean_trust)
    d['passes_final_benign_low_trust_rate']=d.final_benign_low_trust_rate.le(a.max_development_final_benign_low_trust_rate)
    pc=[c for c in d.columns if c.startswith('passes_')]; d['passes_all_safety_constraints']=d[pc].all(axis=1)
    e=d[d.passes_all_safety_constraints].copy()
    if e.empty: raise RuntimeError('No trust policy satisfies frozen development safety constraints. Do not tune on held-out seeds.')
    e=e.sort_values(['mean_malicious_influence_reduction','final_malicious_influence_reduction','attack_malicious_mean_trust','attack_benign_mean_trust'],ascending=[False,False,True,False])
    s=e.iloc[0]; d=d.sort_values(['passes_all_safety_constraints','mean_malicious_influence_reduction','attack_benign_mean_trust'],ascending=[False,False,False])
    d.to_csv(td/'development_policy_ranking.csv',index=False); e.to_csv(td/'development_safety_eligible_policies.csv',index=False)
    mask=m.policy_mode.eq(s.policy_mode)&m.gamma.eq(float(s.gamma))&m.minimum_trust.eq(float(s.minimum_trust)); allsel=m[mask].copy(); held=allsel[~allsel.seed.eq(a.development_seed)].copy()
    allsel.to_csv(td/'selected_policy_all_seed_metrics.csv',index=False); held.to_csv(td/'selected_policy_heldout_metrics.csv',index=False)
    agg=pd.DataFrame([{'candidate':a.candidate,'selected_policy_mode':s.policy_mode,'selected_gamma':float(s.gamma),'selected_minimum_trust':float(s.minimum_trust),'development_seed':a.development_seed,'heldout_seed_count':len(held),
      'heldout_mean_clean_mean_trust':float(held.clean_mean_trust.mean()),'heldout_mean_attack_benign_mean_trust':float(held.attack_benign_mean_trust.mean()),'heldout_mean_attack_malicious_mean_trust':float(held.attack_malicious_mean_trust.mean()),
      'heldout_mean_malicious_influence_reduction':float(held.mean_malicious_influence_reduction.mean()),'heldout_mean_final_malicious_influence_reduction':float(held.final_malicious_influence_reduction.mean()),
      'heldout_mean_attack_benign_low_trust_rate':float(held.attack_benign_low_trust_rate.mean()),'heldout_mean_final_benign_low_trust_rate':float(held.final_benign_low_trust_rate.mean()),
      'selection_frozen_before_heldout_policy_review':True,'selection_rule':'highest development mean malicious influence reduction among policies satisfying frozen clean and benign-safety constraints','test_sets_accessed':False,'malicious_labels_used_for_aggregation':False,'malicious_labels_used_for_development_policy_selection':True}])
    agg.to_csv(td/'selected_policy_aggregate_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,7))
    for mode,g in d.groupby('policy_mode'): ax.scatter(g.attack_benign_mean_trust,g.mean_malicious_influence_reduction,label=mode,alpha=.75)
    ax.set_xlabel('Attack-trajectory benign mean trust'); ax.set_ylabel('Mean malicious influence reduction'); ax.set_title('V3.7.2 development trust-policy frontier'); ax.grid(alpha=.25); ax.legend(); savefig(fig,fd/'development_policy_frontier')
    fig,ax=plt.subplots(figsize=(10,6)); x=np.arange(len(held)); w=.36; ax.bar(x-w/2,held.attack_benign_mean_trust,w,label='Benign mean trust'); ax.bar(x+w/2,held.mean_malicious_influence_reduction,w,label='Malicious influence reduction'); ax.set_xticks(x,held.seed.astype(str)); ax.set_ylim(0,1); ax.set_xlabel('Held-out development seed'); ax.set_ylabel('Rate'); ax.set_title('Frozen V3.7.2 trust policy on held-out seeds'); ax.grid(axis='y',alpha=.25); ax.legend(); savefig(fig,fd/'selected_policy_heldout_summary')
    meta={'experiment_version':'3.7.2','status':'development_policy_audit_not_final_paper_result','candidate':a.candidate,'policy_modes':POLICY_MODES,'gammas':GAMMAS,'minimum_trusts':MIN_TRUSTS,'development_seed':a.development_seed,'heldout_development_seeds':[x for x in seeds if x!=a.development_seed],'selected_policy':{'mode':str(s.policy_mode),'gamma':float(s.gamma),'minimum_trust':float(s.minimum_trust)},'test_sets_accessed':False}
    (out/'temporal_trust_policy_audit_v372_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Temporal Trust Policy Audit V3.7.2 complete'); print('Selected candidate:',a.candidate); print('Selected policy mode:',s.policy_mode); print('Selected gamma:',f'{float(s.gamma):.2f}'); print('Selected minimum trust:',f'{float(s.minimum_trust):.2f}'); print('Tables:',td); print('PNG and PDF figures:',fd)
if __name__=='__main__': main()

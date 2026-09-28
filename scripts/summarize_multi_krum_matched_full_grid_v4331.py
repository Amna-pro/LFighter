#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT=Path(__file__).resolve().parents[1]; REV=ROOT/'reviewer_revision'
SEEDS=[1379954285,1886033230,480705558,1377035733,1707771978]
ATTACKS=['all_to_one_benign','cyclic_shift','multiclass_partial_cycle','pairwise_swap','random_flip']
PA='all_to_one_benign'; PS=1379954285
PREFLIGHT=ROOT/'results'/'reviewer_multi_krum_attacked_preflight_v4330'/PA/f'seed_{PS}'
GRID=ROOT/'results'/'reviewer_multi_krum_matched_grid_v4331'
OR=REV/'MULTI_KRUM_MATCHED_FULL_GRID_ROUNDS_v4331.csv'; OC=REV/'MULTI_KRUM_MATCHED_FULL_GRID_CONDITIONS_v4331.csv'; OA=REV/'MULTI_KRUM_MATCHED_FULL_GRID_ATTACK_SUMMARY_v4331.csv'; OJ=REV/'MULTI_KRUM_MATCHED_FULL_GRID_AUDIT_v4331.json'
PH='5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48'; F=8; M=10

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
def src(a,s): return PREFLIGHT if (a==PA and s==PS) else GRID/a/f'seed_{s}'
def ev(a,s): return REV/f'TASK65_ATTACK_MANIFEST_RECOVERY_SEED_{s}_{a.upper()}_v4324.json'

def main():
 for p in (OR,OC,OA,OJ):
  if p.exists(): raise FileExistsError(f'Refusing overwrite: {p}')
 rounds_all=[]; cond=[]; total_checks=0; hashes={}
 for a in ATTACKS:
  for s in SEEDS:
   d=src(a,s); mp=d/'REVIEWER_MULTI_KRUM_MATCHED_COMPLETE.json'; rp=d/'tables'/'multi_krum_round_metrics.csv'; clp=d/'tables'/'multi_krum_client_records.csv'; pcp=d/'tables'/'validation_class_metrics_long.csv'; cmp=d/'tables'/'validation_confusion_matrix_long.csv'; ckp=d/'checkpoints'/'reviewer_round_checkpoints'/'global_round_08_model.pt'; ep=ev(a,s)
   for p in (mp,rp,clp,pcp,cmp,ckp,ep):
    if not p.exists(): raise FileNotFoundError(p)
   meta=json.loads(mp.read_text()); e=json.loads(ep.read_text()); r=pd.read_csv(rp); c=pd.read_csv(clp); pc=pd.read_csv(pcp); cm=pd.read_csv(cmp); ck=torch.load(ckp,map_location='cpu',weights_only=False); poison=e['poison_index_hash_sha256']; sels=[json.loads(x) for x in r.selected_client_ids]
   checks=[meta.get('phase')=='reviewer_matched_multi_krum_comparator',meta.get('mode')=='strong_attack',meta.get('attack_type')==a,int(meta.get('model_seed',-1))==s,int(meta.get('num_clients',-1))==20,int(meta.get('continuation_rounds',-1))==4,meta.get('partition_hash_sha256')==PH,meta.get('poison_index_hash_sha256')==poison,meta.get('aggregation')=='multi_krum_all_20_submitted_updates',int(meta.get('assumed_byzantine_count',-1))==F,int(meta.get('selected_clients_count',-1))==M,meta.get('detector_used') is False,meta.get('client_reconstruction_used') is False,meta.get('sample_count_weighting_used') is False,meta.get('test_sets_accessed') is False,meta.get('attack_specific_retuning') is False,len(r)==4,list(r.global_round.astype(int))==[5,6,7,8],set(r.arm)=={'multi_krum'},set(r.assumed_byzantine_count.astype(int))=={F},set(r.selected_clients_count.astype(int))=={M},all(len(x)==M for x in sels),all(len(set(x))==M for x in sels),all(all(0<=int(v)<20 for v in x) for x in sels),len(c)==80,int(c.selected_by_multi_krum.astype(bool).sum())==40,len(pc)==32,len(cm)==256,ck.get('arm')=='multi_krum',int(ck.get('global_round',-1))==8,int(ck.get('model_seed',-1))==s,ck.get('attack_type')==a,ck.get('partition_hash')==PH,ck.get('poison_index_hash')==poison,ck.get('aggregation')=='multi_krum_all_20_submitted_updates',int(ck.get('assumed_byzantine_count',-1))==F,int(ck.get('selected_clients_count',-1))==M,len(ck.get('selected_client_ids',[]))==M,isinstance(ck.get('model_state_dict'),dict)]
   if not all(checks): raise RuntimeError(f'Condition audit failed attack={a} seed={s} checks={[i+1 for i,x in enumerate(checks) if not x]}')
   total_checks+=len(checks); rr=r.copy(); rr.insert(0,'model_seed',s); rr.insert(0,'attack',a); rr['source_kind']='audited_preflight_reuse' if (a==PA and s==PS) else 'v4331_grid'; rounds_all.append(rr); r8=r.loc[r.global_round.astype(int)==8].iloc[0]
   cond.append({'attack':a,'model_seed':s,'source_kind':'audited_preflight_reuse' if (a==PA and s==PS) else 'v4331_grid','partition_hash_sha256':meta['partition_hash_sha256'],'poison_index_hash_sha256':meta['poison_index_hash_sha256'],'assumed_byzantine_count':F,'selected_clients_count':M,'round8_selected_client_ids':r8['selected_client_ids'],'round8_selected_malicious_clients_count':int(r8['selected_malicious_clients_count']),'mean_selected_malicious_clients_per_round':float(r.selected_malicious_clients_count.astype(int).mean()),'round8_val_macro_f1':float(r8['val_macro_f1']),'round8_val_balanced_accuracy':float(r8['val_balanced_accuracy']),'mean_round_val_macro_f1':float(r.val_macro_f1.mean()),'mean_round_val_balanced_accuracy':float(r.val_balanced_accuracy.mean()),'total_seconds':float(meta['total_seconds']),'test_sets_accessed':bool(meta['test_sets_accessed']),'attack_specific_retuning':bool(meta['attack_specific_retuning']),'round8_checkpoint_sha256':sha(ckp)}); hashes[f'{a}/seed_{s}']=sha(ckp)
 rdf=pd.concat(rounds_all,ignore_index=True); cdf=pd.DataFrame(cond).sort_values(['attack','model_seed']).reset_index(drop=True)
 if len(rdf)!=100 or len(cdf)!=25: raise RuntimeError(f'Cardinality mismatch rounds={len(rdf)} conds={len(cdf)}')
 rows=[]
 for a,g in cdf.groupby('attack',sort=False): rows.append({'attack':a,'n_seeds':len(g),'mean_round8_val_macro_f1':float(g.round8_val_macro_f1.mean()),'median_round8_val_macro_f1':float(g.round8_val_macro_f1.median()),'std_round8_val_macro_f1_ddof1':float(g.round8_val_macro_f1.std(ddof=1)),'min_round8_val_macro_f1':float(g.round8_val_macro_f1.min()),'max_round8_val_macro_f1':float(g.round8_val_macro_f1.max()),'mean_round8_val_balanced_accuracy':float(g.round8_val_balanced_accuracy.mean()),'median_round8_val_balanced_accuracy':float(g.round8_val_balanced_accuracy.median()),'mean_round8_selected_malicious_clients_count':float(g.round8_selected_malicious_clients_count.mean()),'mean_selected_malicious_clients_per_round':float(g.mean_selected_malicious_clients_per_round.mean()),'mean_total_seconds':float(g.total_seconds.mean())})
 adf=pd.DataFrame(rows); rdf.to_csv(OR,index=False); cdf.to_csv(OC,index=False); adf.to_csv(OA,index=False)
 checks={'conditions_exact_25':len(cdf)==25,'round_rows_exact_100':len(rdf)==100,'attacks_exact_5':set(cdf.attack)==set(ATTACKS),'seeds_exact_5':set(cdf.model_seed.astype(int))==set(SEEDS),'one_preflight_reuse':int((cdf.source_kind=='audited_preflight_reuse').sum())==1,'remaining_grid_conditions_24':int((cdf.source_kind=='v4331_grid').sum())==24,'f_exact_8_all':set(cdf.assumed_byzantine_count.astype(int))=={F},'m_exact_10_all':set(cdf.selected_clients_count.astype(int))=={M},'partition_hash_all_exact':set(cdf.partition_hash_sha256)=={PH},'test_sets_accessed_false_all':not cdf.test_sets_accessed.astype(bool).any(),'attack_specific_retuning_false_all':not cdf.attack_specific_retuning.astype(bool).any(),'round8_metrics_finite':bool(np.isfinite(cdf[['round8_val_macro_f1','round8_val_balanced_accuracy']].to_numpy(float)).all()),'formal_cross_method_statistics_performed':False,'final_test_evaluation_performed':False,'negative_results_retained':True}
 failed=[k for k,v in checks.items() if k not in {'formal_cross_method_statistics_performed','final_test_evaluation_performed'} and not bool(v)]
 if failed: raise RuntimeError(f'Full-grid summary audit failed: {failed}')
 obj={'protocol':'reviewer_v4331_multi_krum_five_attack_five_seed_grid','status':'PASS','checks':checks,'condition_validation_checks_performed':total_checks,'conditions':25,'round_rows':100,'preflight_conditions_reused':1,'new_grid_conditions':24,'assumed_byzantine_count':F,'selected_clients_per_round':M,'test_sets_accessed':False,'attack_specific_retuning':False,'formal_cross_method_statistics_performed':False,'final_test_evaluation_performed':False,'checkpoint_sha256':hashes,'summary_output_sha256':{OR.name:sha(OR),OC.name:sha(OC),OA.name:sha(OA)}}; OJ.write_text(json.dumps(obj,indent=2)+'\n')
 print('MULTI-KRUM FULL GRID SUMMARY AUDIT = PASS'); print('CONDITIONS:',len(cdf)); print('ROUND ROWS:',len(rdf)); print('CONDITION VALIDATION CHECKS:',total_checks); print('PREFLIGHT REUSED: 1'); print('NEW GRID CONDITIONS: 24'); print('ASSUMED BYZANTINE COUNT:',F); print('SELECTED CLIENTS PER ROUND:',M); print('TEST SETS ACCESSED: False'); print('ATTACK SPECIFIC RETUNING: False'); print('FORMAL CROSS METHOD STATISTICS PERFORMED: False'); print('FINAL TEST EVALUATION PERFORMED: False'); print('\nATTACK SUMMARY'); print(adf.to_string(index=False)); return 0
if __name__=='__main__': raise SystemExit(main())

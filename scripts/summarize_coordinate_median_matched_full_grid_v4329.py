#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1]
SEEDS=[1379954285,1886033230,480705558,1377035733,1707771978]
ATTACKS=['all_to_one_benign','cyclic_shift','multiclass_partial_cycle','pairwise_swap','random_flip']
PREFLIGHT_ATTACK='all_to_one_benign'; PREFLIGHT_SEED=1379954285
PREFLIGHT=ROOT/'results'/'reviewer_coordinate_median_attacked_preflight_v4328'/PREFLIGHT_ATTACK/f'seed_{PREFLIGHT_SEED}'
GRID=ROOT/'results'/'reviewer_coordinate_median_matched_grid_v4329'; REV=ROOT/'reviewer_revision'
ROUNDS=REV/'COORDINATE_MEDIAN_MATCHED_FULL_GRID_ROUNDS_v4329.csv'; CONDS=REV/'COORDINATE_MEDIAN_MATCHED_FULL_GRID_CONDITIONS_v4329.csv'; ATTACKSUM=REV/'COORDINATE_MEDIAN_MATCHED_FULL_GRID_ATTACK_SUMMARY_v4329.csv'; AUDIT=REV/'COORDINATE_MEDIAN_MATCHED_FULL_GRID_AUDIT_v4329.json'
EXPECTED_PARTITION='5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
def src(attack,seed): return PREFLIGHT if (attack==PREFLIGHT_ATTACK and seed==PREFLIGHT_SEED) else GRID/attack/f'seed_{seed}'
def manifest(attack,seed): return REV/f'TASK65_ATTACK_MANIFEST_RECOVERY_SEED_{seed}_{attack.upper()}_v4324.json'
def main():
    for p in (ROUNDS,CONDS,ATTACKSUM,AUDIT):
        if p.exists(): raise FileExistsError(p)
    rr=[]; cc=[]; checks=0; ckhash={}
    for attack in ATTACKS:
        for seed in SEEDS:
            d=src(attack,seed); mp=d/'REVIEWER_COORDINATE_MEDIAN_MATCHED_COMPLETE.json'; rp=d/'tables'/'coordinate_median_round_metrics.csv'; cl=d/'tables'/'validation_class_metrics_long.csv'; cf=d/'tables'/'validation_confusion_matrix_long.csv'; ck=d/'checkpoints'/'reviewer_round_checkpoints'/'global_round_08_model.pt'; ep=manifest(attack,seed)
            for p in (mp,rp,cl,cf,ck,ep):
                if not p.exists(): raise FileNotFoundError(p)
            m=json.loads(mp.read_text()); e=json.loads(ep.read_text()); r=pd.read_csv(rp); c=pd.read_csv(cl); f=pd.read_csv(cf); k=torch.load(ck,map_location='cpu',weights_only=False)
            exp=e['poison_index_hash_sha256']
            qs=[m.get('phase')=='reviewer_matched_coordinate_median_comparator',m.get('mode')=='strong_attack',m.get('attack_type')==attack,int(m.get('model_seed',-1))==seed,int(m.get('num_clients',-1))==20,int(m.get('continuation_rounds',-1))==4,m.get('partition_hash_sha256')==EXPECTED_PARTITION,m.get('poison_index_hash_sha256')==exp,m.get('aggregation')=='coordinate_median_all_20_submitted_updates',m.get('detector_used') is False,m.get('client_rejection_used') is False,m.get('client_reconstruction_used') is False,m.get('sample_count_weighting_used') is False,m.get('test_sets_accessed') is False,m.get('attack_specific_retuning') is False,len(r)==4,list(r.global_round.astype(int))==[5,6,7,8],set(r.arm)=={'coordinate_median'},len(c)==32,len(f)==256,k.get('arm')=='coordinate_median',int(k.get('global_round',-1))==8,int(k.get('model_seed',-1))==seed,k.get('attack_type')==attack,k.get('partition_hash')==EXPECTED_PARTITION,k.get('poison_index_hash')==exp,k.get('aggregation')=='coordinate_median_all_20_submitted_updates',isinstance(k.get('model_state_dict'),dict)]
            if not all(qs): raise RuntimeError(f'Condition audit failed {attack} {seed}: {[i+1 for i,x in enumerate(qs) if not x]}')
            checks+=len(qs)
            x=r.copy(); x.insert(0,'model_seed',seed); x.insert(0,'attack',attack); x['source_kind']='audited_preflight_reuse' if (attack==PREFLIGHT_ATTACK and seed==PREFLIGHT_SEED) else 'v4329_grid'; rr.append(x)
            r8=r[r.global_round.astype(int)==8].iloc[0]
            cc.append({'attack':attack,'model_seed':seed,'source_kind':'audited_preflight_reuse' if (attack==PREFLIGHT_ATTACK and seed==PREFLIGHT_SEED) else 'v4329_grid','partition_hash_sha256':m['partition_hash_sha256'],'poison_index_hash_sha256':m['poison_index_hash_sha256'],'round8_val_macro_f1':float(r8.val_macro_f1),'round8_val_balanced_accuracy':float(r8.val_balanced_accuracy),'mean_round_val_macro_f1':float(r.val_macro_f1.mean()),'mean_round_val_balanced_accuracy':float(r.val_balanced_accuracy.mean()),'total_seconds':float(m['total_seconds']),'test_sets_accessed':bool(m['test_sets_accessed']),'attack_specific_retuning':bool(m['attack_specific_retuning']),'round8_checkpoint_sha256':sha(ck)})
            ckhash[f'{attack}/seed_{seed}']=sha(ck)
    rdf=pd.concat(rr,ignore_index=True); cdf=pd.DataFrame(cc).sort_values(['attack','model_seed']).reset_index(drop=True)
    if len(rdf)!=100 or len(cdf)!=25: raise RuntimeError('Grid cardinality mismatch')
    ars=[]
    for attack,g in cdf.groupby('attack',sort=False):
        ars.append({'attack':attack,'n_seeds':len(g),'mean_round8_val_macro_f1':float(g.round8_val_macro_f1.mean()),'median_round8_val_macro_f1':float(g.round8_val_macro_f1.median()),'std_round8_val_macro_f1_ddof1':float(g.round8_val_macro_f1.std(ddof=1)),'min_round8_val_macro_f1':float(g.round8_val_macro_f1.min()),'max_round8_val_macro_f1':float(g.round8_val_macro_f1.max()),'mean_round8_val_balanced_accuracy':float(g.round8_val_balanced_accuracy.mean()),'median_round8_val_balanced_accuracy':float(g.round8_val_balanced_accuracy.median()),'std_round8_val_balanced_accuracy_ddof1':float(g.round8_val_balanced_accuracy.std(ddof=1)),'min_round8_val_balanced_accuracy':float(g.round8_val_balanced_accuracy.min()),'max_round8_val_balanced_accuracy':float(g.round8_val_balanced_accuracy.max()),'mean_total_seconds':float(g.total_seconds.mean())})
    adf=pd.DataFrame(ars); rdf.to_csv(ROUNDS,index=False); cdf.to_csv(CONDS,index=False); adf.to_csv(ATTACKSUM,index=False)
    passchecks={'conditions_exact_25':len(cdf)==25,'round_rows_exact_100':len(rdf)==100,'attacks_exact_5':set(cdf.attack)==set(ATTACKS),'seeds_exact_5':set(cdf.model_seed.astype(int))==set(SEEDS),'one_preflight_reuse':int((cdf.source_kind=='audited_preflight_reuse').sum())==1,'remaining_grid_conditions_24':int((cdf.source_kind=='v4329_grid').sum())==24,'partition_hash_all_exact':set(cdf.partition_hash_sha256)=={EXPECTED_PARTITION},'test_sets_accessed_false_all':not cdf.test_sets_accessed.astype(bool).any(),'attack_specific_retuning_false_all':not cdf.attack_specific_retuning.astype(bool).any(),'round8_metrics_finite':bool(np.isfinite(cdf[['round8_val_macro_f1','round8_val_balanced_accuracy']].to_numpy(float)).all()),'negative_results_retained':True}
    bad=[k for k,v in passchecks.items() if not v]
    if bad: raise RuntimeError(f'Summary audit failed: {bad}')
    obj={'protocol':'reviewer_v4329_coordinate_median_five_attack_five_seed_grid','status':'PASS','checks':passchecks,'condition_validation_checks_performed':checks,'conditions':25,'round_rows':100,'preflight_conditions_reused':1,'new_grid_conditions':24,'test_sets_accessed':False,'attack_specific_retuning':False,'formal_cross_method_statistics_performed':False,'final_test_evaluation_performed':False,'checkpoint_sha256':ckhash,'summary_output_sha256':{ROUNDS.name:sha(ROUNDS),CONDS.name:sha(CONDS),ATTACKSUM.name:sha(ATTACKSUM)}}
    AUDIT.write_text(json.dumps(obj,indent=2)+'\n')
    print('COORDINATE MEDIAN FULL GRID SUMMARY AUDIT = PASS'); print('CONDITIONS: 25'); print('ROUND ROWS: 100'); print('CONDITION VALIDATION CHECKS:',checks); print('PREFLIGHT REUSED: 1'); print('NEW GRID CONDITIONS: 24'); print('TEST SETS ACCESSED: False'); print('ATTACK SPECIFIC RETUNING: False'); print('FORMAL CROSS METHOD STATISTICS PERFORMED: False'); print('FINAL TEST EVALUATION PERFORMED: False'); print('\nATTACK SUMMARY'); print(adf.to_string(index=False)); return 0
if __name__=='__main__': raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/'scripts'/'run_reviewer_multi_krum_matched_v4330.py'
SEEDS=[1379954285,1886033230,480705558,1377035733,1707771978]
ATTACKS=['all_to_one_benign','cyclic_shift','multiclass_partial_cycle','pairwise_swap','random_flip']
PREFLIGHT_ATTACK='all_to_one_benign'; PREFLIGHT_SEED=1379954285
DATA=ROOT/'data'/'processed'/'cic_iot_diad_2024_v2_1_recovery_check'/'arrays'/'behavioral_only.npz'
PARTITION=ROOT/'results'/'cic_iot_diad_federated_tuning_v27_alpha05_seed42_recovery_check'/'partitions'/'client_partitions.npz'
GRID=ROOT/'results'/'reviewer_multi_krum_matched_grid_v4331'
PREFLIGHT=ROOT/'results'/'reviewer_multi_krum_attacked_preflight_v4330'/PREFLIGHT_ATTACK/f'seed_{PREFLIGHT_SEED}'
EXPECTED_PARTITION='5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48'
F=8; M=10

def evpath(a,s): return ROOT/'reviewer_revision'/f'TASK65_ATTACK_MANIFEST_RECOVERY_SEED_{s}_{a.upper()}_v4324.json'
def clean(s): return ROOT/'results'/'reviewer_task65_warmup_recovery_v4323'/f'seed_{s}'/'clean_seed_record'
def warm(s): return ROOT/'results'/'reviewer_task65_warmup_recovery_v4323'/f'seed_{s}'/'warmup'
def plain(a,s): return ROOT/'results'/'reviewer_task65_attack_manifest_recovery_v4324'/a/f'seed_{s}'/'plain_fedavg'

def validate(out,a,s):
    mp=out/'REVIEWER_MULTI_KRUM_MATCHED_COMPLETE.json'; rp=out/'tables'/'multi_krum_round_metrics.csv'; cp=out/'checkpoints'/'reviewer_round_checkpoints'/'global_round_08_model.pt'; ep=evpath(a,s)
    for p in (mp,rp,cp,ep):
        if not p.exists(): raise FileNotFoundError(p)
    meta=json.loads(mp.read_text(encoding='utf-8')); e=json.loads(ep.read_text(encoding='utf-8')); r=pd.read_csv(rp)
    checks=[
      meta.get('phase')=='reviewer_matched_multi_krum_comparator', meta.get('mode')=='strong_attack', meta.get('attack_type')==a,
      int(meta.get('model_seed',-1))==s, int(meta.get('num_clients',-1))==20, int(meta.get('continuation_rounds',-1))==4,
      meta.get('partition_hash_sha256')==EXPECTED_PARTITION, meta.get('poison_index_hash_sha256')==e['poison_index_hash_sha256'],
      meta.get('aggregation')=='multi_krum_all_20_submitted_updates', int(meta.get('assumed_byzantine_count',-1))==F,
      int(meta.get('selected_clients_count',-1))==M, meta.get('detector_used') is False, meta.get('client_reconstruction_used') is False,
      meta.get('sample_count_weighting_used') is False, meta.get('test_sets_accessed') is False, meta.get('attack_specific_retuning') is False,
      len(r)==4, list(r.global_round.astype(int))==[5,6,7,8], set(r.arm)=={'multi_krum'}, set(r.assumed_byzantine_count.astype(int))=={F},
      set(r.selected_clients_count.astype(int))=={M}
    ]
    if not all(checks): raise RuntimeError(f'Condition validation failed: attack={a} seed={s}, failed={[i+1 for i,x in enumerate(checks) if not x]}')
    for x in r.selected_client_ids:
        z=json.loads(x)
        if len(z)!=M or len(set(z))!=M or any(int(v)<0 or int(v)>=20 for v in z): raise RuntimeError(f'Invalid selected ids: attack={a} seed={s}')

def run(a,s,out):
    cmd=[sys.executable,str(RUNNER),'--mode','strong_attack','--attack-type',a,'--data-file',str(DATA),'--partition-file',str(PARTITION),'--clean-seed-dir',str(clean(s)),'--warmup-dir',str(warm(s)),'--plain-branch-dir',str(plain(a,s)),'--output-dir',str(out),'--model-seed',str(s),'--num-clients','20','--continuation-rounds','4','--batch-size','2048','--evaluation-batch-size','4096','--learning-rate','0.0003','--weight-decay','0.0001','--max-class-weight','4.0','--gradient-clip-norm','5.0','--threads','6','--source-class','DDoS','--target-class','Benign']
    subprocess.run(cmd,cwd=ROOT,check=True)

def main():
    for p in (RUNNER,DATA,PARTITION,PREFLIGHT):
        if not p.exists(): raise FileNotFoundError(p)
    print('='*96); print('MULTI-KRUM MATCHED FULL GRID v4.33.1'); print('25 total conditions; one audited preflight reused; 24 remaining conditions to run'); print('Frozen n=20, f=8, m=10'); print('No outcome tuning; no final test access'); print('='*96)
    validate(PREFLIGHT,PREFLIGHT_ATTACK,PREFLIGHT_SEED); print(f'AUDITED PREFLIGHT REUSE = PASS: {PREFLIGHT_ATTACK} seed {PREFLIGHT_SEED}')
    newly=0
    for a in ATTACKS:
      for s in SEEDS:
        if a==PREFLIGHT_ATTACK and s==PREFLIGHT_SEED: continue
        out=GRID/a/f'seed_{s}'; print('\n'+'-'*96); print(f'CONDITION attack={a} seed={s}'); print('-'*96)
        if out.exists():
            if not any(out.iterdir()): raise RuntimeError(f'Empty pre-existing output directory; inspect manually: {out}')
            validate(out,a,s); print('EXISTING COMPLETE CONDITION REUSED = PASS'); continue
        run(a,s,out); validate(out,a,s); newly+=1; print('CONDITION COMPLETION VALIDATION = PASS')
    print('\n'+'='*96); print('MULTI-KRUM FULL GRID EXECUTION COMPLETE'); print('NEW CONDITIONS RUN:',newly); print('TOTAL CONDITIONS:',25); print('TEST SETS ACCESSED: False'); print('ATTACK SPECIFIC RETUNING: False')
    return 0
if __name__=='__main__': raise SystemExit(main())

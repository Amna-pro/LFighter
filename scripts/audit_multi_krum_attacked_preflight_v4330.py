#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'reviewer_multi_krum_attacked_preflight_v4330'/'all_to_one_benign'/'seed_1379954285'
META=OUT/'REVIEWER_MULTI_KRUM_MATCHED_COMPLETE.json'
ROUND=OUT/'tables'/'multi_krum_round_metrics.csv'
CLIENT=OUT/'tables'/'multi_krum_client_records.csv'
CLASS=OUT/'tables'/'validation_class_metrics_long.csv'
CONF=OUT/'tables'/'validation_confusion_matrix_long.csv'
CKPT=OUT/'checkpoints'/'reviewer_round_checkpoints'/'global_round_08_model.pt'
MANIFEST=ROOT/'reviewer_revision'/'TASK65_ATTACK_MANIFEST_RECOVERY_SEED_1379954285_ALL_TO_ONE_BENIGN_v4324.json'
AUDIT=ROOT/'reviewer_revision'/'MULTI_KRUM_ATTACKED_PREFLIGHT_AUDIT_v4330.json'
EXPECTED_PARTITION='5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48'
F=8
M=10

def sha256_file(path: Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    if AUDIT.exists(): raise FileExistsError(AUDIT)
    for p in (META,ROUND,CLIENT,CLASS,CONF,CKPT,MANIFEST):
        if not p.exists(): raise FileNotFoundError(p)
    meta=json.loads(META.read_text(encoding='utf-8'))
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    r=pd.read_csv(ROUND); c=pd.read_csv(CLIENT); pc=pd.read_csv(CLASS); cm=pd.read_csv(CONF)
    ckpt=torch.load(CKPT,map_location='cpu',weights_only=False)
    expected_poison=manifest['poison_index_hash_sha256']
    parsed=[json.loads(x) for x in r['selected_client_ids']]

    checks={
        'metadata_mode_strong_attack':meta.get('mode')=='strong_attack',
        'metadata_attack_exact':meta.get('attack_type')=='all_to_one_benign',
        'metadata_seed_exact':int(meta.get('model_seed',-1))==1379954285,
        'metadata_clients_20':int(meta.get('num_clients',-1))==20,
        'metadata_rounds_4':int(meta.get('continuation_rounds',-1))==4,
        'partition_hash_exact':meta.get('partition_hash_sha256')==EXPECTED_PARTITION,
        'poison_hash_matches_recovery':meta.get('poison_index_hash_sha256')==expected_poison,
        'aggregation_exact':meta.get('aggregation')=='multi_krum_all_20_submitted_updates',
        'assumed_f_exact_8':int(meta.get('assumed_byzantine_count',-1))==F,
        'selection_count_exact_10':int(meta.get('selected_clients_count',-1))==M,
        'detector_false':meta.get('detector_used') is False,
        'reconstruction_false':meta.get('client_reconstruction_used') is False,
        'sample_weighting_false':meta.get('sample_count_weighting_used') is False,
        'test_access_false':meta.get('test_sets_accessed') is False,
        'attack_retuning_false':meta.get('attack_specific_retuning') is False,
        'round_rows_4':len(r)==4,
        'rounds_exact_5_to_8':list(r['global_round'].astype(int))==[5,6,7,8],
        'all_round_arm_multi_krum':set(r['arm'])=={'multi_krum'},
        'submitted_clients_20_all_rounds':set(r['submitted_clients'].astype(int))=={20},
        'assumed_f_8_all_rounds':set(r['assumed_byzantine_count'].astype(int))=={F},
        'selected_10_all_rounds':set(r['selected_clients_count'].astype(int))=={M},
        'selection_lists_length_10':all(len(x)==M for x in parsed),
        'selection_lists_unique':all(len(set(x))==M for x in parsed),
        'selection_ids_valid':all(all(0<=int(v)<20 for v in x) for x in parsed),
        'client_rows_80':len(c)==80,
        'selected_client_records_40':int(c['selected_by_multi_krum'].astype(bool).sum())==40,
        'class_rows_32':len(pc)==32,
        'confusion_rows_256':len(cm)==256,
        'ckpt_arm_exact':ckpt.get('arm')=='multi_krum',
        'ckpt_round_8':int(ckpt.get('global_round',-1))==8,
        'ckpt_seed_exact':int(ckpt.get('model_seed',-1))==1379954285,
        'ckpt_attack_exact':ckpt.get('attack_type')=='all_to_one_benign',
        'ckpt_partition_exact':ckpt.get('partition_hash')==EXPECTED_PARTITION,
        'ckpt_poison_exact':ckpt.get('poison_index_hash')==expected_poison,
        'ckpt_aggregation_exact':ckpt.get('aggregation')=='multi_krum_all_20_submitted_updates',
        'ckpt_f_exact_8':int(ckpt.get('assumed_byzantine_count',-1))==F,
        'ckpt_selection_count_exact_10':int(ckpt.get('selected_clients_count',-1))==M,
        'ckpt_selected_ids_len_10':len(ckpt.get('selected_client_ids',[]))==M,
        'model_state_present':isinstance(ckpt.get('model_state_dict'),dict),
    }
    failed=[k for k,v in checks.items() if not bool(v)]
    if failed: raise RuntimeError(f'Multi-Krum preflight audit failed: {failed}')
    obj={
        'protocol':'reviewer_v4330_multi_krum_attacked_preflight_audit',
        'status':'PASS','checks':checks,'checks_passed':len(checks),'checks_total':len(checks),
        'scientific_outcome_gate_used':False,'test_sets_accessed':False,'attack_specific_retuning':False,
        'assumed_byzantine_count':F,'selected_clients_per_round':M,
        'output_sha256':{META.name:sha256_file(META),ROUND.name:sha256_file(ROUND),CLIENT.name:sha256_file(CLIENT),CLASS.name:sha256_file(CLASS),CONF.name:sha256_file(CONF),CKPT.name:sha256_file(CKPT)}
    }
    AUDIT.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    print('MULTI KRUM ATTACKED PREFLIGHT AUDIT = PASS')
    print('CHECKS:',len(checks),'/',len(checks))
    print('ROUND ROWS:',len(r)); print('CLIENT ROWS:',len(c)); print('CLASS ROWS:',len(pc)); print('CONFUSION ROWS:',len(cm))
    print('ASSUMED BYZANTINE COUNT:',F); print('SELECTED CLIENTS PER ROUND:',M)
    print('TEST SETS ACCESSED: False'); print('ATTACK SPECIFIC RETUNING: False'); print('SCIENTIFIC OUTCOME GATE USED: False')
    print('ROUND8 VALIDATION MACRO F1:',float(r.loc[r.global_round==8,'val_macro_f1'].iloc[0]))
    print('ROUND8 SELECTED IDS:',r.loc[r.global_round==8,'selected_client_ids'].iloc[0])
    return 0

if __name__=='__main__': raise SystemExit(main())

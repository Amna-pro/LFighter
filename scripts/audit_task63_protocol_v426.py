import json, pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=ROOT/'configs'/'task63_preregistration_v4260.json'
d=json.loads(p.read_text(encoding='utf-8'))
checks={}
checks['task_exact']=d.get('task')==63
checks['protocol_exact']=d.get('protocol_id')=='task63_v4260'
checks['deferred_exact']=d['scope_reconciliation']['deferred_unexecuted_tasks']==list(range(46,55))
checks['no_backfill']=d['scope_reconciliation']['no_backfill'] is True
checks['reserved_test_blocked']=d['data_boundary']['task63_may_materialize_reserved_test_arrays'] is False
checks['training_blocked']=d['data_boundary']['task63_may_train'] is False
checks['new_shap_blocked']=d['data_boundary']['task63_may_run_new_shap'] is False
checks['llm_blocked']=d['data_boundary']['task63_may_call_llm'] is False
checks['task61_fail_retained']='Task61 automated LLM evaluation failure' in d['claim_classes']['negative']
checks['task62_superiority_blocked']=any('LLM superiority' in x for x in d['claim_classes']['prohibited'])
checks['oracle_blocked']=any('Oracle-clean' in x for x in d['claim_classes']['prohibited'])
checks['development_seeds_blocked']=all(str(x) in d['final_evaluation_freeze']['task64_seed_policy'] for x in [7,42,99,123,2026])
checks['one_shot']= 'No retuning' in d['final_evaluation_freeze']['task65_test_policy']
checks['holm_frozen']='Holm' in d['statistics_freeze']['multiple_comparisons']
checks['md_exists']=(ROOT/'TASK63_PREREGISTRATION_V426.md').exists()
checks['scope_exists']=(ROOT/'configs'/'TASK63_SCOPE_RECONCILIATION_V426.md').exists()
checks['ledger_exists']=(ROOT/'configs'/'TASK63_CLAIM_LEDGER_V426.md').exists()
for k,v in checks.items(): print(f'{k}: {v}')
print(f'PASS: {sum(checks.values())}/{len(checks)}')
if not all(checks.values()): sys.exit(1)
print('READY TO FREEZE TASK 63 C0/C1: True')

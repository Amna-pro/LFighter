#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"/"reviewer_geometric_median_attacked_preflight_v4332"/"all_to_one_benign"/"seed_1379954285"
META=OUT/"REVIEWER_GEOMETRIC_MEDIAN_MATCHED_COMPLETE.json"
ROUND=OUT/"tables"/"geometric_median_round_metrics.csv"
CLIENT=OUT/"tables"/"geometric_median_client_records.csv"
CLASS=OUT/"tables"/"validation_class_metrics_long.csv"
CONF=OUT/"tables"/"validation_confusion_matrix_long.csv"
CKPT=OUT/"checkpoints"/"reviewer_round_checkpoints"/"global_round_08_model.pt"
MANIFEST=ROOT/"reviewer_revision"/"TASK65_ATTACK_MANIFEST_RECOVERY_SEED_1379954285_ALL_TO_ONE_BENIGN_v4324.json"
AUDIT=ROOT/"reviewer_revision"/"GEOMETRIC_MEDIAN_ATTACKED_PREFLIGHT_AUDIT_v4332.json"
PH="5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()
def main():
    if AUDIT.exists(): raise FileExistsError(AUDIT)
    for p in (META,ROUND,CLIENT,CLASS,CONF,CKPT,MANIFEST):
        if not p.exists(): raise FileNotFoundError(p)
    m=json.loads(META.read_text(encoding="utf-8")); ev=json.loads(MANIFEST.read_text(encoding="utf-8"))
    r=pd.read_csv(ROUND); c=pd.read_csv(CLIENT); pc=pd.read_csv(CLASS); cm=pd.read_csv(CONF)
    k=torch.load(CKPT,map_location="cpu",weights_only=False); poison=ev["poison_index_hash_sha256"]
    checks={
      "metadata_mode_strong_attack":m.get("mode")=="strong_attack",
      "metadata_attack_exact":m.get("attack_type")=="all_to_one_benign",
      "metadata_seed_exact":int(m.get("model_seed",-1))==1379954285,
      "metadata_clients_20":int(m.get("num_clients",-1))==20,
      "metadata_rounds_4":int(m.get("continuation_rounds",-1))==4,
      "partition_hash_exact":m.get("partition_hash_sha256")==PH,
      "poison_hash_matches_recovery":m.get("poison_index_hash_sha256")==poison,
      "aggregation_exact":m.get("aggregation")=="geometric_median_all_20_submitted_updates",
      "weighting_exact":m.get("weighting")=="equal_client_geometric_median_objective",
      "geomed_source_exact":m.get("geometric_median_source")=="scripts/audit_robust_centers_v319a.py",
      "max_iter_exact_60":int(m.get("geomed_max_iter",-1))==60,
      "tolerance_exact_1e_7":float(m.get("geomed_tolerance",-1))==1e-7,
      "near_threshold_exact":float(m.get("geomed_near_threshold",-1))==1e-12,
      "initialization_exact":m.get("geomed_initialization")=="numpy_coordinate_median",
      "numeric_dtype_float64":m.get("geomed_numeric_dtype")=="float64",
      "adapter_self_test_passed":m.get("adapter_self_test_passed") is True,
      "detector_false":m.get("detector_used") is False,
      "rejection_false":m.get("client_rejection_used") is False,
      "reconstruction_false":m.get("client_reconstruction_used") is False,
      "sample_weighting_false":m.get("sample_count_weighting_used") is False,
      "test_access_false":m.get("test_sets_accessed") is False,
      "attack_retuning_false":m.get("attack_specific_retuning") is False,
      "round_rows_4":len(r)==4,
      "rounds_exact_5_to_8":list(r["global_round"].astype(int))==[5,6,7,8],
      "all_round_arm_geometric_median":set(r["arm"])=={"geometric_median"},
      "submitted_clients_20_all_rounds":set(r["submitted_clients"].astype(int))=={20},
      "max_iter_60_all_rounds":set(r["geomed_max_iter"].astype(int))=={60},
      "tol_1e_7_all_rounds":bool(np.allclose(r["geomed_tolerance"].astype(float),1e-7,rtol=0,atol=0)),
      "iteration_counts_valid":bool(((r["algorithm_iterations"].astype(int)>=1)&(r["algorithm_iterations"].astype(int)<=60)).all()),
      "round_metrics_finite":bool(np.isfinite(r[["val_macro_f1","val_balanced_accuracy","algorithm_iterations"]].to_numpy(float)).all()),
      "client_rows_80":len(c)==80,
      "all_client_rows_submitted":bool(c["submitted_to_geometric_median"].astype(bool).all()),
      "class_rows_32":len(pc)==32,
      "confusion_rows_256":len(cm)==256,
      "ckpt_arm_exact":k.get("arm")=="geometric_median",
      "ckpt_round_8":int(k.get("global_round",-1))==8,
      "ckpt_seed_exact":int(k.get("model_seed",-1))==1379954285,
      "ckpt_attack_exact":k.get("attack_type")=="all_to_one_benign",
      "ckpt_partition_exact":k.get("partition_hash")==PH,
      "ckpt_poison_exact":k.get("poison_index_hash")==poison,
      "ckpt_aggregation_exact":k.get("aggregation")=="geometric_median_all_20_submitted_updates",
      "ckpt_weighting_exact":k.get("weighting")=="equal_client_geometric_median_objective",
      "ckpt_max_iter_exact":int(k.get("geomed_max_iter",-1))==60,
      "ckpt_tolerance_exact":float(k.get("geomed_tolerance",-1))==1e-7,
      "ckpt_iterations_valid":1<=int(k.get("algorithm_iterations",-1))<=60,
      "model_state_present":isinstance(k.get("model_state_dict"),dict),
    }
    bad=[x for x,v in checks.items() if not bool(v)]
    if bad: raise RuntimeError(f"Geometric-median preflight audit failed: {bad}")
    obj={"protocol":"reviewer_v4332_geometric_median_attacked_preflight_audit","status":"PASS",
      "checks":checks,"checks_passed":len(checks),"checks_total":len(checks),
      "scientific_outcome_gate_used":False,"algorithm_convergence_gate_used":False,
      "test_sets_accessed":False,"attack_specific_retuning":False,"geomed_max_iter":60,"geomed_tolerance":1e-7,
      "all_rounds_converged_observed":bool(r["algorithm_converged"].astype(bool).all()),
      "algorithm_iterations_by_round":[int(v) for v in r["algorithm_iterations"].astype(int)],
      "output_sha256":{p.name:sha(p) for p in (META,ROUND,CLIENT,CLASS,CONF,CKPT)}}
    AUDIT.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
    print("GEOMETRIC MEDIAN ATTACKED PREFLIGHT AUDIT = PASS")
    print("CHECKS:",len(checks),"/",len(checks)); print("ROUND ROWS:",len(r)); print("CLIENT ROWS:",len(c))
    print("CLASS ROWS:",len(pc)); print("CONFUSION ROWS:",len(cm)); print("MAX ITER: 60"); print("TOLERANCE: 1e-7")
    print("ITERATIONS BY ROUND:",r["algorithm_iterations"].astype(int).tolist())
    print("ALL ROUNDS CONVERGED OBSERVED:",bool(r["algorithm_converged"].astype(bool).all()))
    print("TEST SETS ACCESSED: False"); print("ATTACK SPECIFIC RETUNING: False")
    print("SCIENTIFIC OUTCOME GATE USED: False"); print("ALGORITHM CONVERGENCE GATE USED: False")
    print("ROUND8 VALIDATION MACRO F1:",float(r.loc[r.global_round==8,"val_macro_f1"].iloc[0]))
if __name__=="__main__": main()

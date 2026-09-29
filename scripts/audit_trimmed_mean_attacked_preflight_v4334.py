#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"/"reviewer_trimmed_mean_attacked_preflight_v4334"/"all_to_one_benign"/"seed_1379954285"
META=OUT/"REVIEWER_TRIMMED_MEAN_MATCHED_COMPLETE.json"
ROUND=OUT/"tables"/"trimmed_mean_round_metrics.csv"
CLIENT=OUT/"tables"/"trimmed_mean_client_records.csv"
CLASS=OUT/"tables"/"validation_class_metrics_long.csv"
CONF=OUT/"tables"/"validation_confusion_matrix_long.csv"
CKPT=OUT/"checkpoints"/"reviewer_round_checkpoints"/"global_round_08_model.pt"
MANIFEST=ROOT/"reviewer_revision"/"TASK65_ATTACK_MANIFEST_RECOVERY_SEED_1379954285_ALL_TO_ONE_BENIGN_v4324.json"
AUDIT=ROOT/"reviewer_revision"/"TRIMMED_MEAN_ATTACKED_PREFLIGHT_AUDIT_v4334.json"
PH="5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):
            h.update(c)
    return h.hexdigest()

def main():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    for p in (META,ROUND,CLIENT,CLASS,CONF,CKPT,MANIFEST):
        if not p.exists():
            raise FileNotFoundError(p)

    m=json.loads(META.read_text(encoding="utf-8"))
    e=json.loads(MANIFEST.read_text(encoding="utf-8"))
    r=pd.read_csv(ROUND)
    c=pd.read_csv(CLIENT)
    pc=pd.read_csv(CLASS)
    cm=pd.read_csv(CONF)
    k=torch.load(CKPT,map_location="cpu",weights_only=False)
    poison=e["poison_index_hash_sha256"]

    checks={
        "mode_strong":m.get("mode")=="strong_attack",
        "attack_exact":m.get("attack_type")=="all_to_one_benign",
        "seed_exact":int(m.get("model_seed",-1))==1379954285,
        "clients_20":int(m.get("num_clients",-1))==20,
        "rounds_4":int(m.get("continuation_rounds",-1))==4,
        "partition_exact":m.get("partition_hash_sha256")==PH,
        "poison_exact":m.get("poison_index_hash_sha256")==poison,
        "aggregation_exact":m.get("aggregation")=="coordinate_wise_symmetric_trimmed_mean_all_20_submitted_updates",
        "trim_fraction_exact":float(m.get("trim_fraction",-1))==0.2,
        "trim_count_per_side_exact":int(m.get("trim_count_per_side",-1))==4,
        "retained_count_exact":int(m.get("retained_count_per_coordinate",-1))==12,
        "detector_false":m.get("detector_used") is False,
        "rejection_false":m.get("client_rejection_used") is False,
        "reconstruction_false":m.get("client_reconstruction_used") is False,
        "sample_weighting_false":m.get("sample_count_weighting_used") is False,
        "test_access_false":m.get("test_sets_accessed") is False,
        "retuning_false":m.get("attack_specific_retuning") is False,
        "round_rows_4":len(r)==4,
        "rounds_5_to_8":list(r["global_round"].astype(int))==[5,6,7,8],
        "arm_exact":set(r["arm"])=={"trimmed_mean"},
        "trim_fraction_all":set(r["trim_fraction"].astype(float))=={0.2},
        "trim_per_side_all":set(r["trim_count_per_side"].astype(int))=={4},
        "retained_all":set(r["retained_count_per_coordinate"].astype(int))=={12},
        "round_metrics_finite":bool(np.isfinite(r[["val_macro_f1","val_balanced_accuracy"]].to_numpy(float)).all()),
        "client_rows_80":len(c)==80,
        "all_submitted":bool(c["submitted_to_trimmed_mean"].astype(bool).all()),
        "class_rows_32":len(pc)==32,
        "confusion_rows_256":len(cm)==256,
        "ckpt_arm":k.get("arm")=="trimmed_mean",
        "ckpt_round8":int(k.get("global_round",-1))==8,
        "ckpt_seed":int(k.get("model_seed",-1))==1379954285,
        "ckpt_attack":k.get("attack_type")=="all_to_one_benign",
        "ckpt_partition":k.get("partition_hash")==PH,
        "ckpt_poison":k.get("poison_index_hash")==poison,
        "ckpt_trim_fraction":float(k.get("trim_fraction",-1))==0.2,
        "ckpt_trim_per_side":int(k.get("trim_count_per_side",-1))==4,
        "ckpt_retained":int(k.get("retained_count_per_coordinate",-1))==12,
        "model_state_present":isinstance(k.get("model_state_dict"),dict),
    }
    bad=[x for x,v in checks.items() if not bool(v)]
    if bad:
        raise RuntimeError(f"trimmed-mean preflight audit failed: {bad}")

    obj={
        "protocol":"reviewer_v4334_trimmed_mean_attacked_preflight_audit",
        "status":"PASS",
        "checks":checks,
        "checks_passed":len(checks),
        "checks_total":len(checks),
        "scientific_outcome_gate_used":False,
        "test_sets_accessed":False,
        "attack_specific_retuning":False,
        "trim_fraction":0.2,
        "trim_count_per_side":4,
        "retained_count_per_coordinate":12,
        "output_sha256":{p.name:sha(p) for p in (META,ROUND,CLIENT,CLASS,CONF,CKPT)},
    }
    AUDIT.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")

    print("TRIMMED MEAN ATTACKED PREFLIGHT AUDIT = PASS")
    print("CHECKS:",len(checks),"/",len(checks))
    print("TRIM FRACTION: 0.2")
    print("TRIM COUNT PER SIDE: 4")
    print("RETAINED COUNT PER COORDINATE: 12")
    print("TEST SETS ACCESSED: False")
    print("ATTACK SPECIFIC RETUNING: False")
    print("SCIENTIFIC OUTCOME GATE USED: False")
    print("ROUND8 VALIDATION MACRO F1:",float(r.loc[r.global_round==8,"val_macro_f1"].iloc[0]))

if __name__=="__main__":
    main()

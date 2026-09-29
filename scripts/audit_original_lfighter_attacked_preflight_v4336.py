#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"/"reviewer_original_lfighter_attacked_preflight_v4336"/"all_to_one_benign"/"seed_1379954285"

META=OUT/"REVIEWER_ORIGINAL_LFIGHTER_MATCHED_COMPLETE.json"
ROUND=OUT/"tables"/"original_lfighter_round_metrics.csv"
CLIENT=OUT/"tables"/"original_lfighter_client_decisions.csv"
SALIENCE=OUT/"tables"/"original_lfighter_class_salience.csv"
CLUSTER=OUT/"tables"/"original_lfighter_cluster_diagnostics.csv"
CLASS=OUT/"tables"/"validation_class_metrics_long.csv"
CONF=OUT/"tables"/"validation_confusion_matrix_long.csv"
CKPT=OUT/"checkpoints"/"reviewer_round_checkpoints"/"global_round_08_model.pt"
MANIFEST=ROOT/"reviewer_revision"/"TASK65_ATTACK_MANIFEST_RECOVERY_SEED_1379954285_ALL_TO_ONE_BENIGN_v4324.json"
AUDIT=ROOT/"reviewer_revision"/"ORIGINAL_LFIGHTER_ATTACKED_PREFLIGHT_AUDIT_v4336.json"

PH="5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if AUDIT.exists():
        raise FileExistsError(AUDIT)

    for p in (META,ROUND,CLIENT,SALIENCE,CLUSTER,CLASS,CONF,CKPT,MANIFEST):
        if not p.exists():
            raise FileNotFoundError(p)

    meta=json.loads(META.read_text(encoding="utf-8"))
    recovery=json.loads(MANIFEST.read_text(encoding="utf-8"))
    r=pd.read_csv(ROUND)
    c=pd.read_csv(CLIENT)
    s=pd.read_csv(SALIENCE)
    cl=pd.read_csv(CLUSTER)
    pc=pd.read_csv(CLASS)
    cm=pd.read_csv(CONF)
    ckpt=torch.load(CKPT,map_location="cpu",weights_only=False)

    poison=recovery["poison_index_hash_sha256"]

    checks={
        "metadata_mode_strong_attack":meta.get("mode")=="strong_attack",
        "metadata_attack_exact":meta.get("attack_type")=="all_to_one_benign",
        "metadata_seed_exact":int(meta.get("model_seed",-1))==1379954285,
        "metadata_clients_20":int(meta.get("num_clients",-1))==20,
        "metadata_rounds_4":int(meta.get("continuation_rounds",-1))==4,
        "partition_hash_exact":meta.get("partition_hash_sha256")==PH,
        "poison_hash_matches_recovery":meta.get("poison_index_hash_sha256")==poison,
        "aggregation_exact":meta.get("aggregation")=="original_lfighter_v34_equal_average_admitted_states",
        "source_exact":meta.get("source")=="src/lfighter_original_v34.py",
        "kmeans_seed_zero":int(meta.get("kmeans_seed",-1))==0,
        "kmeans_clusters_two":int(meta.get("kmeans_clusters",-1))==2,
        "kmeans_n_init_ten":int(meta.get("kmeans_n_init",-1))==10,
        "sample_weighting_false":meta.get("sample_count_weighting_used") is False,
        "malicious_labels_decision_false":meta.get("malicious_labels_used_for_decision") is False,
        "malicious_labels_diagnostics_true":meta.get("malicious_labels_used_for_diagnostics_only") is True,
        "test_access_false":meta.get("test_sets_accessed") is False,
        "attack_retuning_false":meta.get("attack_specific_retuning") is False,
        "round_rows_4":len(r)==4,
        "rounds_exact_5_to_8":list(r["global_round"].astype(int))==[5,6,7,8],
        "arm_exact":set(r["arm"])=={"original_lfighter"},
        "kmeans_seed_zero_all_rounds":set(r["kmeans_seed"].astype(int))=={0},
        "kmeans_clusters_two_all_rounds":set(r["kmeans_clusters"].astype(int))=={2},
        "kmeans_n_init_ten_all_rounds":set(r["kmeans_n_init"].astype(int))=={10},
        "admitted_counts_valid":bool(((r["admitted_client_count"].astype(int)>=1)&(r["admitted_client_count"].astype(int)<=20)).all()),
        "rejected_counts_valid":bool(((r["rejected_client_count"].astype(int)>=0)&(r["rejected_client_count"].astype(int)<=19)).all()),
        "counts_sum_20":bool((r["admitted_client_count"].astype(int)+r["rejected_client_count"].astype(int)==20).all()),
        "security_metrics_finite":bool(np.isfinite(r[["malicious_rejection_recall","benign_false_rejection_rate","malicious_admission_rate"]].to_numpy(float)).all()),
        "validation_metrics_finite":bool(np.isfinite(r[["val_macro_f1","val_balanced_accuracy"]].to_numpy(float)).all()),
        "client_rows_80":len(c)==80,
        "class_salience_rows_32":len(s)==32,
        "exact_two_selected_classes_per_round":bool((s.groupby("global_round")["selected_top_two"].sum().astype(int)==2).all()),
        "cluster_rows_8":len(cl)==8,
        "two_clusters_per_round":bool((cl.groupby("global_round").size()==2).all()),
        "validation_class_rows_32":len(pc)==32,
        "confusion_rows_256":len(cm)==256,
        "ckpt_arm_exact":ckpt.get("arm")=="original_lfighter",
        "ckpt_round_8":int(ckpt.get("global_round",-1))==8,
        "ckpt_seed_exact":int(ckpt.get("model_seed",-1))==1379954285,
        "ckpt_attack_exact":ckpt.get("attack_type")=="all_to_one_benign",
        "ckpt_partition_exact":ckpt.get("partition_hash")==PH,
        "ckpt_poison_exact":ckpt.get("poison_index_hash")==poison,
        "ckpt_aggregation_exact":ckpt.get("aggregation")=="original_lfighter_v34_equal_average_admitted_states",
        "ckpt_kmeans_seed_zero":int(ckpt.get("kmeans_seed",-1))==0,
        "ckpt_kmeans_clusters_two":int(ckpt.get("kmeans_clusters",-1))==2,
        "ckpt_kmeans_n_init_ten":int(ckpt.get("kmeans_n_init",-1))==10,
        "model_state_present":isinstance(ckpt.get("model_state_dict"),dict),
    }

    bad=[name for name,value in checks.items() if not bool(value)]
    if bad:
        raise RuntimeError(f"Original LFighter preflight audit failed: {bad}")

    obj={
        "protocol":"reviewer_v4336_original_lfighter_attacked_preflight_audit",
        "status":"PASS",
        "checks":checks,
        "checks_passed":len(checks),
        "checks_total":len(checks),
        "scientific_outcome_gate_used":False,
        "test_sets_accessed":False,
        "attack_specific_retuning":False,
        "kmeans_seed":0,
        "kmeans_clusters":2,
        "kmeans_n_init":10,
        "fallback_round_count_observed":int((r["fallback_reason"].fillna("")!="").sum()),
        "round8_selected_class_pair":str(r.loc[r.global_round==8,"selected_class_pair"].iloc[0]),
        "round8_admitted_client_count":int(r.loc[r.global_round==8,"admitted_client_count"].iloc[0]),
        "round8_rejected_client_count":int(r.loc[r.global_round==8,"rejected_client_count"].iloc[0]),
        "round8_malicious_rejection_recall":float(r.loc[r.global_round==8,"malicious_rejection_recall"].iloc[0]),
        "round8_benign_false_rejection_rate":float(r.loc[r.global_round==8,"benign_false_rejection_rate"].iloc[0]),
        "round8_validation_macro_f1":float(r.loc[r.global_round==8,"val_macro_f1"].iloc[0]),
        "output_sha256":{
            META.name:sha256_file(META),
            ROUND.name:sha256_file(ROUND),
            CLIENT.name:sha256_file(CLIENT),
            SALIENCE.name:sha256_file(SALIENCE),
            CLUSTER.name:sha256_file(CLUSTER),
            CLASS.name:sha256_file(CLASS),
            CONF.name:sha256_file(CONF),
            CKPT.name:sha256_file(CKPT),
        },
    }
    AUDIT.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")

    print("ORIGINAL LFIGHTER ATTACKED PREFLIGHT AUDIT = PASS")
    print("CHECKS:",len(checks),"/",len(checks))
    print("KMEANS SEED: 0")
    print("KMEANS CLUSTERS: 2")
    print("KMEANS N_INIT: 10")
    print("FALLBACK ROUNDS OBSERVED:",obj["fallback_round_count_observed"])
    print("ROUND8 SELECTED CLASS PAIR:",obj["round8_selected_class_pair"])
    print("ROUND8 ADMITTED:",obj["round8_admitted_client_count"])
    print("ROUND8 REJECTED:",obj["round8_rejected_client_count"])
    print("ROUND8 MALICIOUS REJECTION RECALL:",obj["round8_malicious_rejection_recall"])
    print("ROUND8 BENIGN FALSE REJECTION RATE:",obj["round8_benign_false_rejection_rate"])
    print("ROUND8 VALIDATION MACRO F1:",obj["round8_validation_macro_f1"])
    print("TEST SETS ACCESSED: False")
    print("ATTACK SPECIFIC RETUNING: False")
    print("SCIENTIFIC OUTCOME GATE USED: False")
    return 0


if __name__=="__main__":
    raise SystemExit(main())

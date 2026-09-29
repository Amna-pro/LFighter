#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
REV=ROOT/"reviewer_revision"

SEEDS=[1379954285,1886033230,480705558,1377035733,1707771978]
ATTACKS=["all_to_one_benign","cyclic_shift","multiclass_partial_cycle","pairwise_swap","random_flip"]
PA="all_to_one_benign"; PS=1379954285

PREFLIGHT=ROOT/"results"/"reviewer_original_lfighter_attacked_preflight_v4336"/PA/f"seed_{PS}"
GRID=ROOT/"results"/"reviewer_original_lfighter_matched_grid_v4337"

OR=REV/"ORIGINAL_LFIGHTER_MATCHED_FULL_GRID_ROUNDS_v4337.csv"
OC=REV/"ORIGINAL_LFIGHTER_MATCHED_FULL_GRID_CONDITIONS_v4337.csv"
OA=REV/"ORIGINAL_LFIGHTER_MATCHED_FULL_GRID_ATTACK_SUMMARY_v4337.csv"
OJ=REV/"ORIGINAL_LFIGHTER_MATCHED_FULL_GRID_AUDIT_v4337.json"

PH="5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def source_dir(attack,seed):
    return PREFLIGHT if (attack==PA and seed==PS) else GRID/attack/f"seed_{seed}"


def evidence_path(attack,seed):
    return REV/f"TASK65_ATTACK_MANIFEST_RECOVERY_SEED_{seed}_{attack.upper()}_v4324.json"


def main() -> int:
    for p in (OR,OC,OA,OJ):
        if p.exists():
            raise FileExistsError(f"Refusing overwrite: {p}")

    round_frames=[]
    condition_rows=[]
    checkpoint_hashes={}
    total_checks=0

    for attack in ATTACKS:
        for seed in SEEDS:
            d=source_dir(attack,seed)

            mp=d/"REVIEWER_ORIGINAL_LFIGHTER_MATCHED_COMPLETE.json"
            rp=d/"tables"/"original_lfighter_round_metrics.csv"
            cp=d/"tables"/"original_lfighter_client_decisions.csv"
            sp=d/"tables"/"original_lfighter_class_salience.csv"
            clp=d/"tables"/"original_lfighter_cluster_diagnostics.csv"
            pcp=d/"tables"/"validation_class_metrics_long.csv"
            cmp=d/"tables"/"validation_confusion_matrix_long.csv"
            kp=d/"checkpoints"/"reviewer_round_checkpoints"/"global_round_08_model.pt"
            ep=evidence_path(attack,seed)

            for p in (mp,rp,cp,sp,clp,pcp,cmp,kp,ep):
                if not p.exists():
                    raise FileNotFoundError(p)

            m=json.loads(mp.read_text(encoding="utf-8"))
            e=json.loads(ep.read_text(encoding="utf-8"))
            r=pd.read_csv(rp)
            c=pd.read_csv(cp)
            s=pd.read_csv(sp)
            cl=pd.read_csv(clp)
            pc=pd.read_csv(pcp)
            cm=pd.read_csv(cmp)
            ckpt=torch.load(kp,map_location="cpu",weights_only=False)

            poison=e["poison_index_hash_sha256"]

            checks=[
                m.get("phase")=="reviewer_matched_original_lfighter_comparator",
                m.get("mode")=="strong_attack",
                m.get("attack_type")==attack,
                int(m.get("model_seed",-1))==seed,
                int(m.get("num_clients",-1))==20,
                int(m.get("continuation_rounds",-1))==4,
                m.get("partition_hash_sha256")==PH,
                m.get("poison_index_hash_sha256")==poison,
                m.get("aggregation")=="original_lfighter_v34_equal_average_admitted_states",
                m.get("source")=="src/lfighter_original_v34.py",
                int(m.get("kmeans_seed",-1))==0,
                int(m.get("kmeans_clusters",-1))==2,
                int(m.get("kmeans_n_init",-1))==10,
                m.get("sample_count_weighting_used") is False,
                m.get("malicious_labels_used_for_decision") is False,
                m.get("malicious_labels_used_for_diagnostics_only") is True,
                m.get("test_sets_accessed") is False,
                m.get("attack_specific_retuning") is False,
                len(r)==4,
                list(r["global_round"].astype(int))==[5,6,7,8],
                set(r["arm"])=={"original_lfighter"},
                set(r["kmeans_seed"].astype(int))=={0},
                set(r["kmeans_clusters"].astype(int))=={2},
                set(r["kmeans_n_init"].astype(int))=={10},
                bool(((r["admitted_client_count"].astype(int)>=1)&(r["admitted_client_count"].astype(int)<=20)).all()),
                bool(((r["rejected_client_count"].astype(int)>=0)&(r["rejected_client_count"].astype(int)<=19)).all()),
                bool((r["admitted_client_count"].astype(int)+r["rejected_client_count"].astype(int)==20).all()),
                bool(np.isfinite(r[["malicious_rejection_recall","benign_false_rejection_rate","malicious_admission_rate"]].to_numpy(float)).all()),
                bool(np.isfinite(r[["val_macro_f1","val_balanced_accuracy"]].to_numpy(float)).all()),
                len(c)==80,
                len(s)==32,
                bool((s.groupby("global_round")["selected_top_two"].sum().astype(int)==2).all()),
                len(cl)==8,
                bool((cl.groupby("global_round").size()==2).all()),
                len(pc)==32,
                len(cm)==256,
                ckpt.get("arm")=="original_lfighter",
                int(ckpt.get("global_round",-1))==8,
                int(ckpt.get("model_seed",-1))==seed,
                ckpt.get("attack_type")==attack,
                ckpt.get("partition_hash")==PH,
                ckpt.get("poison_index_hash")==poison,
                ckpt.get("aggregation")=="original_lfighter_v34_equal_average_admitted_states",
                int(ckpt.get("kmeans_seed",-1))==0,
                int(ckpt.get("kmeans_clusters",-1))==2,
                int(ckpt.get("kmeans_n_init",-1))==10,
                isinstance(ckpt.get("model_state_dict"),dict),
            ]

            if not all(checks):
                bad=[i for i,v in enumerate(checks,1) if not v]
                raise RuntimeError(
                    f"Condition audit failed attack={attack} seed={seed} checks={bad}"
                )
            total_checks += len(checks)

            rr=r.copy()
            rr.insert(0,"model_seed",seed)
            rr.insert(0,"attack",attack)
            rr["source_kind"]="audited_preflight_reuse" if (attack==PA and seed==PS) else "v4337_grid"
            round_frames.append(rr)

            r8=r.loc[r["global_round"].astype(int)==8].iloc[0]

            condition_rows.append({
                "attack":attack,
                "model_seed":seed,
                "source_kind":"audited_preflight_reuse" if (attack==PA and seed==PS) else "v4337_grid",
                "partition_hash_sha256":m["partition_hash_sha256"],
                "poison_index_hash_sha256":m["poison_index_hash_sha256"],
                "kmeans_seed":0,
                "kmeans_clusters":2,
                "kmeans_n_init":10,
                "fallback_round_count":int((r["fallback_reason"].fillna("")!="").sum()),
                "round8_selected_class_pair":str(r8["selected_class_pair"]),
                "round8_admitted_client_count":int(r8["admitted_client_count"]),
                "round8_rejected_client_count":int(r8["rejected_client_count"]),
                "round8_malicious_rejection_recall":float(r8["malicious_rejection_recall"]),
                "round8_benign_false_rejection_rate":float(r8["benign_false_rejection_rate"]),
                "round8_malicious_admission_rate":float(r8["malicious_admission_rate"]),
                "round8_val_macro_f1":float(r8["val_macro_f1"]),
                "round8_val_balanced_accuracy":float(r8["val_balanced_accuracy"]),
                "mean_round_malicious_rejection_recall":float(r["malicious_rejection_recall"].mean()),
                "mean_round_benign_false_rejection_rate":float(r["benign_false_rejection_rate"].mean()),
                "mean_round_val_macro_f1":float(r["val_macro_f1"].mean()),
                "mean_round_val_balanced_accuracy":float(r["val_balanced_accuracy"].mean()),
                "total_seconds":float(m["total_seconds"]),
                "test_sets_accessed":bool(m["test_sets_accessed"]),
                "attack_specific_retuning":bool(m["attack_specific_retuning"]),
                "round8_checkpoint_sha256":sha256_file(kp),
            })
            checkpoint_hashes[f"{attack}/seed_{seed}"]=sha256_file(kp)

    rdf=pd.concat(round_frames,ignore_index=True)
    cdf=pd.DataFrame(condition_rows).sort_values(["attack","model_seed"]).reset_index(drop=True)

    if len(rdf)!=100 or len(cdf)!=25:
        raise RuntimeError("Full-grid cardinality mismatch")

    attack_rows=[]
    for attack,g in cdf.groupby("attack",sort=False):
        attack_rows.append({
            "attack":attack,
            "n_seeds":len(g),
            "mean_round8_val_macro_f1":float(g["round8_val_macro_f1"].mean()),
            "median_round8_val_macro_f1":float(g["round8_val_macro_f1"].median()),
            "std_round8_val_macro_f1_ddof1":float(g["round8_val_macro_f1"].std(ddof=1)),
            "min_round8_val_macro_f1":float(g["round8_val_macro_f1"].min()),
            "max_round8_val_macro_f1":float(g["round8_val_macro_f1"].max()),
            "mean_round8_val_balanced_accuracy":float(g["round8_val_balanced_accuracy"].mean()),
            "mean_round8_malicious_rejection_recall":float(g["round8_malicious_rejection_recall"].mean()),
            "mean_round8_benign_false_rejection_rate":float(g["round8_benign_false_rejection_rate"].mean()),
            "fallback_rounds_total":int(g["fallback_round_count"].sum()),
            "mean_total_seconds":float(g["total_seconds"].mean()),
        })
    adf=pd.DataFrame(attack_rows)

    rdf.to_csv(OR,index=False)
    cdf.to_csv(OC,index=False)
    adf.to_csv(OA,index=False)

    checks={
        "conditions_exact_25":len(cdf)==25,
        "round_rows_exact_100":len(rdf)==100,
        "attacks_exact_5":set(cdf["attack"])==set(ATTACKS),
        "seeds_exact_5":set(cdf["model_seed"].astype(int))==set(SEEDS),
        "one_preflight_reuse":int((cdf["source_kind"]=="audited_preflight_reuse").sum())==1,
        "remaining_grid_conditions_24":int((cdf["source_kind"]=="v4337_grid").sum())==24,
        "kmeans_seed_zero_all":set(cdf["kmeans_seed"].astype(int))=={0},
        "kmeans_clusters_two_all":set(cdf["kmeans_clusters"].astype(int))=={2},
        "kmeans_n_init_ten_all":set(cdf["kmeans_n_init"].astype(int))=={10},
        "partition_hash_all_exact":set(cdf["partition_hash_sha256"])=={PH},
        "test_sets_accessed_false_all":not cdf["test_sets_accessed"].astype(bool).any(),
        "attack_specific_retuning_false_all":not cdf["attack_specific_retuning"].astype(bool).any(),
        "round8_metrics_finite":bool(np.isfinite(
            cdf[[
                "round8_val_macro_f1",
                "round8_val_balanced_accuracy",
                "round8_malicious_rejection_recall",
                "round8_benign_false_rejection_rate",
            ]].to_numpy(float)
        ).all()),
        "formal_cross_method_statistics_performed":False,
        "final_test_evaluation_performed":False,
        "negative_results_retained":True,
    }

    failed=[
        k for k,v in checks.items()
        if k not in {
            "formal_cross_method_statistics_performed",
            "final_test_evaluation_performed",
        } and not bool(v)
    ]
    if failed:
        raise RuntimeError(f"Full-grid audit failed: {failed}")

    audit={
        "protocol":"reviewer_v4337_original_lfighter_five_attack_five_seed_grid",
        "status":"PASS",
        "checks":checks,
        "condition_validation_checks_performed":total_checks,
        "conditions":25,
        "round_rows":100,
        "preflight_conditions_reused":1,
        "new_grid_conditions":24,
        "kmeans_seed":0,
        "kmeans_clusters":2,
        "kmeans_n_init":10,
        "test_sets_accessed":False,
        "attack_specific_retuning":False,
        "formal_cross_method_statistics_performed":False,
        "final_test_evaluation_performed":False,
        "checkpoint_sha256":checkpoint_hashes,
        "summary_output_sha256":{
            OR.name:sha256_file(OR),
            OC.name:sha256_file(OC),
            OA.name:sha256_file(OA),
        },
    }
    OJ.write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")

    print("ORIGINAL LFIGHTER FULL GRID SUMMARY AUDIT = PASS")
    print("CONDITIONS:",len(cdf))
    print("ROUND ROWS:",len(rdf))
    print("CONDITION VALIDATION CHECKS:",total_checks)
    print("PREFLIGHT REUSED: 1")
    print("NEW GRID CONDITIONS: 24")
    print("KMEANS SEED: 0")
    print("KMEANS CLUSTERS: 2")
    print("KMEANS N_INIT: 10")
    print("TEST SETS ACCESSED: False")
    print("ATTACK SPECIFIC RETUNING: False")
    print("FORMAL CROSS METHOD STATISTICS PERFORMED: False")
    print("FINAL TEST EVALUATION PERFORMED: False")
    print("\nATTACK SUMMARY")
    print(adf.to_string(index=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

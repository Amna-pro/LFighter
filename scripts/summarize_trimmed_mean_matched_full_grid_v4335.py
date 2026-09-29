#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch

ROOT=Path(__file__).resolve().parents[1]
REV=ROOT/"reviewer_revision"
SEEDS=[1379954285,1886033230,480705558,1377035733,1707771978]
ATTACKS=["all_to_one_benign","cyclic_shift","multiclass_partial_cycle","pairwise_swap","random_flip"]
PA="all_to_one_benign"; PS=1379954285
PREFLIGHT=ROOT/"results"/"reviewer_trimmed_mean_attacked_preflight_v4334"/PA/f"seed_{PS}"
GRID=ROOT/"results"/"reviewer_trimmed_mean_matched_grid_v4335"
OR=REV/"TRIMMED_MEAN_MATCHED_FULL_GRID_ROUNDS_v4335.csv"
OC=REV/"TRIMMED_MEAN_MATCHED_FULL_GRID_CONDITIONS_v4335.csv"
OA=REV/"TRIMMED_MEAN_MATCHED_FULL_GRID_ATTACK_SUMMARY_v4335.csv"
OJ=REV/"TRIMMED_MEAN_MATCHED_FULL_GRID_AUDIT_v4335.json"
PH="5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):
            h.update(c)
    return h.hexdigest()

def src(attack,seed):
    return PREFLIGHT if attack==PA and seed==PS else GRID/attack/f"seed_{seed}"

def evp(attack,seed):
    return REV/f"TASK65_ATTACK_MANIFEST_RECOVERY_SEED_{seed}_{attack.upper()}_v4324.json"

def main():
    for p in (OR,OC,OA,OJ):
        if p.exists():
            raise FileExistsError(f"Refusing overwrite: {p}")

    round_frames=[]
    conditions=[]
    total_checks=0
    ck_hash={}

    for attack in ATTACKS:
        for seed in SEEDS:
            d=src(attack,seed)
            mp=d/"REVIEWER_TRIMMED_MEAN_MATCHED_COMPLETE.json"
            rp=d/"tables"/"trimmed_mean_round_metrics.csv"
            cp=d/"tables"/"trimmed_mean_client_records.csv"
            pp=d/"tables"/"validation_class_metrics_long.csv"
            xp=d/"tables"/"validation_confusion_matrix_long.csv"
            kp=d/"checkpoints"/"reviewer_round_checkpoints"/"global_round_08_model.pt"
            ep=evp(attack,seed)

            for p in (mp,rp,cp,pp,xp,kp,ep):
                if not p.exists():
                    raise FileNotFoundError(p)

            m=json.loads(mp.read_text(encoding="utf-8"))
            e=json.loads(ep.read_text(encoding="utf-8"))
            r=pd.read_csv(rp)
            c=pd.read_csv(cp)
            pc=pd.read_csv(pp)
            cm=pd.read_csv(xp)
            k=torch.load(kp,map_location="cpu",weights_only=False)
            poison=e["poison_index_hash_sha256"]

            checks=[
                m.get("phase")=="reviewer_matched_trimmed_mean_comparator",
                m.get("mode")=="strong_attack",
                m.get("attack_type")==attack,
                int(m.get("model_seed",-1))==seed,
                int(m.get("num_clients",-1))==20,
                int(m.get("continuation_rounds",-1))==4,
                m.get("partition_hash_sha256")==PH,
                m.get("poison_index_hash_sha256")==poison,
                m.get("aggregation")=="coordinate_wise_symmetric_trimmed_mean_all_20_submitted_updates",
                float(m.get("trim_fraction",-1))==0.2,
                int(m.get("trim_count_per_side",-1))==4,
                int(m.get("retained_count_per_coordinate",-1))==12,
                m.get("detector_used") is False,
                m.get("client_rejection_used") is False,
                m.get("client_reconstruction_used") is False,
                m.get("sample_count_weighting_used") is False,
                m.get("test_sets_accessed") is False,
                m.get("attack_specific_retuning") is False,
                len(r)==4,
                list(r["global_round"].astype(int))==[5,6,7,8],
                set(r["arm"])=={"trimmed_mean"},
                set(r["trim_fraction"].astype(float))=={0.2},
                set(r["trim_count_per_side"].astype(int))=={4},
                set(r["retained_count_per_coordinate"].astype(int))=={12},
                len(c)==80,
                bool(c["submitted_to_trimmed_mean"].astype(bool).all()),
                len(pc)==32,
                len(cm)==256,
                k.get("arm")=="trimmed_mean",
                int(k.get("global_round",-1))==8,
                int(k.get("model_seed",-1))==seed,
                k.get("attack_type")==attack,
                k.get("partition_hash")==PH,
                k.get("poison_index_hash")==poison,
                float(k.get("trim_fraction",-1))==0.2,
                int(k.get("trim_count_per_side",-1))==4,
                int(k.get("retained_count_per_coordinate",-1))==12,
                isinstance(k.get("model_state_dict"),dict),
            ]
            if not all(checks):
                bad=[i for i,v in enumerate(checks,1) if not v]
                raise RuntimeError(f"Condition audit failed attack={attack} seed={seed} checks={bad}")
            total_checks+=len(checks)

            rr=r.copy()
            rr.insert(0,"model_seed",seed)
            rr.insert(0,"attack",attack)
            rr["source_kind"]="audited_preflight_reuse" if attack==PA and seed==PS else "v4335_grid"
            round_frames.append(rr)

            r8=r.loc[r["global_round"].astype(int)==8].iloc[0]
            conditions.append({
                "attack":attack,
                "model_seed":seed,
                "source_kind":"audited_preflight_reuse" if attack==PA and seed==PS else "v4335_grid",
                "partition_hash_sha256":m["partition_hash_sha256"],
                "poison_index_hash_sha256":m["poison_index_hash_sha256"],
                "trim_fraction":0.2,
                "trim_count_per_side":4,
                "retained_count_per_coordinate":12,
                "round8_val_macro_f1":float(r8["val_macro_f1"]),
                "round8_val_balanced_accuracy":float(r8["val_balanced_accuracy"]),
                "mean_round_val_macro_f1":float(r["val_macro_f1"].mean()),
                "mean_round_val_balanced_accuracy":float(r["val_balanced_accuracy"].mean()),
                "total_seconds":float(m["total_seconds"]),
                "test_sets_accessed":bool(m["test_sets_accessed"]),
                "attack_specific_retuning":bool(m["attack_specific_retuning"]),
                "round8_checkpoint_sha256":sha(kp),
            })
            ck_hash[f"{attack}/seed_{seed}"]=sha(kp)

    rdf=pd.concat(round_frames,ignore_index=True)
    cdf=pd.DataFrame(conditions).sort_values(["attack","model_seed"]).reset_index(drop=True)

    if len(rdf)!=100 or len(cdf)!=25:
        raise RuntimeError("grid cardinality mismatch")

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
        "remaining_grid_conditions_24":int((cdf["source_kind"]=="v4335_grid").sum())==24,
        "trim_fraction_exact_0_2_all":set(cdf["trim_fraction"].astype(float))=={0.2},
        "trim_count_per_side_exact_4_all":set(cdf["trim_count_per_side"].astype(int))=={4},
        "retained_count_exact_12_all":set(cdf["retained_count_per_coordinate"].astype(int))=={12},
        "partition_hash_all_exact":set(cdf["partition_hash_sha256"])=={PH},
        "test_sets_accessed_false_all":not cdf["test_sets_accessed"].astype(bool).any(),
        "attack_specific_retuning_false_all":not cdf["attack_specific_retuning"].astype(bool).any(),
        "round8_metrics_finite":bool(np.isfinite(
            cdf[["round8_val_macro_f1","round8_val_balanced_accuracy"]].to_numpy(float)
        ).all()),
        "formal_cross_method_statistics_performed":False,
        "final_test_evaluation_performed":False,
        "negative_results_retained":True,
    }

    failed=[k for k,v in checks.items()
            if k not in {"formal_cross_method_statistics_performed","final_test_evaluation_performed"}
            and not bool(v)]
    if failed:
        raise RuntimeError(f"Full-grid audit failed: {failed}")

    obj={
        "protocol":"reviewer_v4335_trimmed_mean_five_attack_five_seed_grid",
        "status":"PASS",
        "checks":checks,
        "condition_validation_checks_performed":total_checks,
        "conditions":25,
        "round_rows":100,
        "preflight_conditions_reused":1,
        "new_grid_conditions":24,
        "trim_fraction":0.2,
        "trim_count_per_side":4,
        "retained_count_per_coordinate":12,
        "test_sets_accessed":False,
        "attack_specific_retuning":False,
        "formal_cross_method_statistics_performed":False,
        "final_test_evaluation_performed":False,
        "checkpoint_sha256":ck_hash,
        "summary_output_sha256":{
            OR.name:sha(OR),
            OC.name:sha(OC),
            OA.name:sha(OA),
        },
    }
    OJ.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")

    print("TRIMMED MEAN FULL GRID SUMMARY AUDIT = PASS")
    print("CONDITIONS:",len(cdf))
    print("ROUND ROWS:",len(rdf))
    print("CONDITION VALIDATION CHECKS:",total_checks)
    print("PREFLIGHT REUSED: 1")
    print("NEW GRID CONDITIONS: 24")
    print("TRIM FRACTION: 0.2")
    print("TRIM COUNT PER SIDE: 4")
    print("RETAINED COUNT PER COORDINATE: 12")
    print("TEST SETS ACCESSED: False")
    print("ATTACK SPECIFIC RETUNING: False")
    print("FORMAL CROSS METHOD STATISTICS PERFORMED: False")
    print("FINAL TEST EVALUATION PERFORMED: False")
    print("\nATTACK SUMMARY")
    print(adf.to_string(index=False))

if __name__=="__main__":
    main()

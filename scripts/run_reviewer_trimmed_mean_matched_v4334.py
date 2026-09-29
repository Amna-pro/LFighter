#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json, shutil, sys, time
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/"src",ROOT/"scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0,str(p))

from federated_iot_v26 import CLASS_NAMES, NUM_CLASSES, set_seed, sqrt_class_weights, train_local_model
from neural_models_v24 import build_model
from run_frozen_untargeted_defense_v320b1 import (
    ATTACK_TYPES, DEFAULT_MALICIOUS_CLIENTS,
    load_exact_v320a3_poison_plan, parse_client_ids,
)
from run_reviewer_p4p_matched_v4322 import class_confusion_rows, load_train_val_only
from run_targeted_label_flip_v292 import load_fixed_partitions, read_clean_partition_hash
from transition_signature_features_v38 import evaluate_validation
from trusted_update_reconstruction_v312 import (
    checkpoint_sha256, floating_update, state_from_update,
    state_max_abs_difference, update_norm,
)
from verify_task43_c1_baselines_v414 import trimmed_mean

EXPERIMENT_VERSION="4.33.4-TRIMMED-MEAN"
TRIM_FRACTION=0.2
TRIM_COUNT_PER_SIDE=4
RETAINED_COUNT=12


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["clean","strong_attack"],required=True)
    p.add_argument("--attack-type",choices=ATTACK_TYPES,required=True)
    p.add_argument("--data-file",type=Path,required=True)
    p.add_argument("--partition-file",type=Path,required=True)
    p.add_argument("--clean-seed-dir",type=Path,required=True)
    p.add_argument("--warmup-dir",type=Path,required=True)
    p.add_argument("--plain-branch-dir",type=Path)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--model-seed",type=int,required=True)
    p.add_argument("--num-clients",type=int,default=20)
    p.add_argument("--continuation-rounds",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=2048)
    p.add_argument("--evaluation-batch-size",type=int,default=4096)
    p.add_argument("--learning-rate",type=float,default=3e-4)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--max-class-weight",type=float,default=4.0)
    p.add_argument("--gradient-clip-norm",type=float,default=5.0)
    p.add_argument("--threads",type=int,default=6)
    p.add_argument("--source-class",default="DDoS")
    p.add_argument("--target-class",default="Benign")
    p.add_argument("--malicious-clients",default=DEFAULT_MALICIOUS_CLIENTS)
    return p.parse_args()


def main():
    a=parse_args()
    if a.mode=="strong_attack" and a.plain_branch_dir is None:
        raise ValueError("--plain-branch-dir required")
    if a.mode=="strong_attack" and len(parse_client_ids(a.malicious_clients))!=8:
        raise ValueError("eight malicious clients required")
    if a.num_clients!=20 or a.continuation_rounds!=4:
        raise ValueError("frozen primary configuration mismatch")

    out=a.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)

    tables=out/"tables"
    ckpts=out/"checkpoints"/"reviewer_round_checkpoints"
    attackdir=out/"attack_manifest"
    for p in (tables,ckpts,attackdir):
        p.mkdir(parents=True,exist_ok=True)

    torch.set_num_threads(max(1,a.threads))
    set_seed(a.model_seed)

    X_train,y_train,X_val,y_val=load_train_val_only(a.data_file)
    client_indices,ph=load_fixed_partitions(
        a.partition_file.expanduser().resolve(),
        expected_clients=20,
        train_rows=len(y_train),
    )
    if read_clean_partition_hash(a.clean_seed_dir.expanduser().resolve())!=ph:
        raise RuntimeError("clean partition mismatch")

    wdir=a.warmup_dir.expanduser().resolve()
    wm=json.loads((wdir/"true_warmup_v310_metadata.json").read_text(encoding="utf-8"))
    w4p=wdir/"checkpoints"/"common_round4_warmup_model.pt"
    w4=torch.load(w4p,map_location="cpu",weights_only=False)
    if int(wm["model_seed"])!=a.model_seed or wm["partition_hash_sha256"]!=ph:
        raise RuntimeError("warmup mismatch")

    model=build_model("resmlp",X_train.shape[1],NUM_CLASSES)
    model.load_state_dict(w4["model_state_dict"])
    if state_max_abs_difference(model.state_dict(),w4["model_state_dict"])!=0.0:
        raise RuntimeError("W4 mismatch")

    if a.mode=="strong_attack":
        requested=parse_client_ids(a.malicious_clients)
        malicious,ppos,plab,_pmanifest,poison_hash=load_exact_v320a3_poison_plan(
            plain_branch_dir=a.plain_branch_dir,
            client_indices=client_indices,
            attack_type=a.attack_type,
            model_seed=a.model_seed,
            expected_malicious_clients=requested,
        )
        src=a.plain_branch_dir.expanduser().resolve()/"attack_manifest"
        for name in ("poisoned_indices.npz","poisoned_labels.npz","malicious_client_poison_manifest.csv"):
            if (src/name).exists():
                shutil.copy2(src/name,attackdir/name)
    else:
        malicious=[]
        ppos={i:np.empty(0,dtype=np.int64) for i in range(20)}
        plab={i:np.empty(0,dtype=np.int64) for i in range(20)}
        poison_hash="clean_no_poison"

    cw=sqrt_class_weights(y_train,a.max_class_weight)
    sid=CLASS_NAMES.index(a.source_class)
    tid=CLASS_NAMES.index(a.target_class)

    rr: List[Dict[str,object]]=[]
    cr: List[Dict[str,object]]=[]
    pcl: List[Dict[str,object]]=[]
    cml: List[Dict[str,object]]=[]
    started=time.perf_counter()

    for mr in range(1,5):
        gr=4+mr
        rt=time.perf_counter()
        ref=copy.deepcopy(model.state_dict())

        t=time.perf_counter()
        states: List[Mapping[str,torch.Tensor]]=[]
        counts=[]
        for cid in range(20):
            idx=client_indices[cid]
            ly=y_train[idx].copy()
            pos=ppos[cid]
            if len(pos):
                repl=plab[cid]
                if len(repl)!=len(pos):
                    raise RuntimeError("poison length mismatch")
                ly[pos]=repl
            lm=build_model("resmlp",X_train.shape[1],NUM_CLASSES)
            lm.load_state_dict(ref)
            st,_=train_local_model(
                model=lm,
                X=X_train[idx],
                y=ly,
                class_weights=cw,
                local_epochs=1,
                batch_size=a.batch_size,
                learning_rate=a.learning_rate,
                weight_decay=a.weight_decay,
                gradient_clip_norm=a.gradient_clip_norm,
                seed=a.model_seed+gr*1000+cid,
            )
            states.append(st)
            counts.append(int(len(idx)))
            del lm
        train_s=time.perf_counter()-t

        t=time.perf_counter()
        updates=[floating_update(st,ref) for st in states]
        agg=trimmed_mean(updates,trim_fraction=TRIM_FRACTION)
        model.load_state_dict(state_from_update(ref,agg))
        agg_s=time.perf_counter()-t

        t=time.perf_counter()
        vm,vp=evaluate_validation(model,X_val,y_val,sid,tid,a.evaluation_batch_size)
        pc,cm=class_confusion_rows(model,X_val,y_val,a.evaluation_batch_size,mr,gr,a.attack_type)
        pcl.extend(pc); cml.extend(cm)
        val_s=time.perf_counter()-t

        labels=np.asarray([i in malicious for i in range(20)],dtype=bool)
        for cid in range(20):
            cr.append({
                "monitoring_round":mr,
                "global_round":gr,
                "client_id":cid,
                "actual_malicious":bool(labels[cid]),
                "client_samples":counts[cid],
                "poisoned_rows":int(len(ppos[cid])),
                "update_l2_norm":float(update_norm(updates[cid])),
                "submitted_to_trimmed_mean":True,
                "client_detection_applicable":False,
                "reconstructed":False,
            })

        rr.append({
            "monitoring_round":mr,
            "global_round":gr,
            "mode":a.mode,
            "attack_type":a.attack_type,
            "arm":"trimmed_mean",
            "submitted_clients":20,
            "trim_fraction":TRIM_FRACTION,
            "trim_count_per_side":TRIM_COUNT_PER_SIDE,
            "retained_count_per_coordinate":RETAINED_COUNT,
            "phase_local_training_seconds":train_s,
            "phase_detector_seconds":0.0,
            "phase_reconstruction_seconds":0.0,
            "phase_aggregation_seconds":agg_s,
            "phase_validation_seconds":val_s,
            "round_seconds":time.perf_counter()-rt,
            **{f"val_{k}":v for k,v in vm.items()},
            **{f"val_{k}":v for k,v in vp.items()},
        })

        torch.save({
            "experiment_version":EXPERIMENT_VERSION,
            "arm":"trimmed_mean",
            "monitoring_round":mr,
            "global_round":gr,
            "model_seed":a.model_seed,
            "attack_type":a.attack_type,
            "model_state_dict":model.state_dict(),
            "previous_global_state_dict":ref,
            "partition_hash":ph,
            "poison_index_hash":poison_hash,
            "aggregation":"coordinate_wise_symmetric_trimmed_mean_all_20_submitted_updates",
            "trim_fraction":TRIM_FRACTION,
            "trim_count_per_side":TRIM_COUNT_PER_SIDE,
            "retained_count_per_coordinate":RETAINED_COUNT,
        },ckpts/f"global_round_{gr:02d}_model.pt")

        print(
            f"round {gr}: macroF1={vm['macro_f1']:.4f}, "
            f"trim_fraction=0.2, trim_per_side=4, seconds={rr[-1]['round_seconds']:.1f}"
        )

    rdf=pd.DataFrame(rr)
    rdf.to_csv(tables/"trimmed_mean_round_metrics.csv",index=False)
    pd.DataFrame(cr).to_csv(tables/"trimmed_mean_client_records.csv",index=False)
    pd.DataFrame(pcl).to_csv(tables/"validation_class_metrics_long.csv",index=False)
    pd.DataFrame(cml).to_csv(tables/"validation_confusion_matrix_long.csv",index=False)

    meta={
        "experiment_version":EXPERIMENT_VERSION,
        "phase":"reviewer_matched_trimmed_mean_comparator",
        "mode":a.mode,
        "attack_type":a.attack_type,
        "model_seed":a.model_seed,
        "num_clients":20,
        "continuation_rounds":4,
        "malicious_clients":malicious,
        "partition_hash_sha256":ph,
        "poison_index_hash_sha256":poison_hash,
        "warmup_round4_checkpoint_sha256":checkpoint_sha256(w4p),
        "aggregation":"coordinate_wise_symmetric_trimmed_mean_all_20_submitted_updates",
        "trim_fraction":TRIM_FRACTION,
        "trim_count_per_side":TRIM_COUNT_PER_SIDE,
        "retained_count_per_coordinate":RETAINED_COUNT,
        "trimmed_mean_source":"scripts/verify_task43_c1_baselines_v414.py",
        "historical_setting_source":"scripts/run_task43_c2_seed7_screen_v414.py",
        "detector_used":False,
        "client_rejection_used":False,
        "client_reconstruction_used":False,
        "sample_count_weighting_used":False,
        "test_sets_accessed":False,
        "attack_specific_retuning":False,
        "total_seconds":float(time.perf_counter()-started),
        "mean_validation_macro_f1":float(rdf["val_macro_f1"].mean()),
    }
    (out/"REVIEWER_TRIMMED_MEAN_MATCHED_COMPLETE.json").write_text(
        json.dumps(meta,indent=2)+"\n",encoding="utf-8"
    )

    print("TRIMMED MEAN MATCHED BRANCH COMPLETE")
    print("TRIM FRACTION: 0.2")
    print("TRIM COUNT PER SIDE: 4")
    print("RETAINED COUNT PER COORDINATE: 12")
    print("TEST SETS ACCESSED: False")
    print("ATTACK SPECIFIC RETUNING: False")
    return 0

if __name__=="__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PARTITION_HASH = "5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"
PROBE_HASH = "4f4243b5725fd80f8f17be179711241a0fd9b459acc5ea6f1725b880c54865bf"
EXPECTED_EMA_Q99_6DP = 2.749450
EXPECTED_INSTANT_Q95_6DP = 3.977139

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def checkpoint_state(path: Path):
    obj=torch.load(path,map_location="cpu",weights_only=False)
    if isinstance(obj,dict) and isinstance(obj.get("model_state_dict"),dict):
        return obj["model_state_dict"]
    if isinstance(obj,dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        return obj
    raise RuntimeError(f"Cannot identify state dict in {path}")

def compare_states(a: Path,b: Path):
    sa=checkpoint_state(a); sb=checkpoint_state(b)
    if set(sa)!=set(sb):
        return False, float("inf")
    maxdiff=0.0
    for k in sa:
        xa=sa[k].detach().cpu()
        xb=sb[k].detach().cpu()
        if xa.shape!=xb.shape:
            return False,float("inf")
        if xa.numel():
            d=float((xa-xb).abs().max().item())
            maxdiff=max(maxdiff,d)
        if not torch.equal(xa,xb):
            return False,maxdiff
    return True,maxdiff

def compare_npz(a: Path,b: Path):
    aa=np.load(a,allow_pickle=False); bb=np.load(b,allow_pickle=False)
    if set(aa.files)!=set(bb.files):
        return False, sorted(aa.files), sorted(bb.files)
    for k in aa.files:
        if not np.array_equal(aa[k],bb[k]):
            return False, sorted(aa.files), sorted(bb.files)
    return True, sorted(aa.files), sorted(bb.files)

def load_runner_module(root: Path):
    p=root/"scripts"/"run_reviewer_reconstruction_ablation_v4340e.py"
    spec=importlib.util.spec_from_file_location("v4340e_runner_for_calibration_check",p)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,required=True)
    ap.add_argument("--reference-warmup",type=Path,required=True)
    ap.add_argument("--recovered-warmup",type=Path,required=True)
    ap.add_argument("--record",type=Path,required=True)
    a=ap.parse_args()

    root=a.project_root.resolve()
    ref=a.reference_warmup.resolve()
    rec=a.recovered_warmup.resolve()

    paths={
        "reference_w4":ref/"checkpoints"/"common_round4_warmup_model.pt",
        "recovered_w4":rec/"checkpoints"/"common_round4_warmup_model.pt",
        "reference_profiles":ref/"calibration"/"trusted_client_profiles.npz",
        "recovered_profiles":rec/"calibration"/"trusted_client_profiles.npz",
        "feature_calibration":rec/"calibration"/"feature_calibration.csv",
        "clean_scores":rec/"calibration"/"clean_leave_one_round_out_scores.csv",
        "metadata":rec/"true_warmup_v310_metadata.json",
    }
    for name,p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{name}: {p}")

    state_equal,maxdiff=compare_states(paths["reference_w4"],paths["recovered_w4"])
    profile_equal,profile_keys_ref,profile_keys_rec=compare_npz(
        paths["reference_profiles"],paths["recovered_profiles"]
    )

    meta=json.loads(paths["metadata"].read_text(encoding="utf-8"))
    partition=meta.get("partition_hash_sha256")
    probe=meta.get("probe_hash_sha256")

    feature=pd.read_csv(paths["feature_calibration"])
    scores=pd.read_csv(paths["clean_scores"])
    if feature.empty or scores.empty:
        raise RuntimeError("Recovered calibration table is empty")
    if not np.isfinite(feature.select_dtypes(include=[np.number]).to_numpy(float)).all():
        raise RuntimeError("Recovered feature calibration contains non-finite numeric values")
    if not np.isfinite(scores.select_dtypes(include=[np.number]).to_numpy(float)).all():
        raise RuntimeError("Recovered clean score table contains non-finite numeric values")

    sys.path.insert(0,str(root/"src"))
    sys.path.insert(0,str(root/"scripts"))
    mod=load_runner_module(root)
    ema=float(mod.quantile_higher(
        scores[mod.SCORE_COLUMN].to_numpy(dtype=np.float64),
        mod.FROZEN_EMA_QUANTILE,
    ))
    inst=float(mod.quantile_higher(
        scores[mod.CANDIDATE].to_numpy(dtype=np.float64),
        mod.FROZEN_INSTANT_QUANTILE,
    ))

    checks={
        "reference_w4_exists":paths["reference_w4"].exists(),
        "recovered_w4_exists":paths["recovered_w4"].exists(),
        "w4_tensor_exact":state_equal,
        "w4_max_abs_difference_zero":maxdiff==0.0,
        "trusted_profile_arrays_exact":profile_equal,
        "partition_hash_exact":partition==PARTITION_HASH,
        "probe_hash_exact":probe==PROBE_HASH,
        "feature_calibration_present":paths["feature_calibration"].exists(),
        "clean_scores_present":paths["clean_scores"].exists(),
        "ema_q99_historical_6dp":round(ema,6)==round(EXPECTED_EMA_Q99_6DP,6),
        "instant_q95_historical_6dp":round(inst,6)==round(EXPECTED_INSTANT_Q95_6DP,6),
    }
    failed=[k for k,v in checks.items() if not bool(v)]
    if failed:
        raise RuntimeError("Warmup calibration recovery verification failed: "+", ".join(failed))

    out={
        "protocol":"reviewer_v4340f_task65_full_warmup_calibration_recovery",
        "status":"PASS",
        "checks":checks,
        "partition_hash_sha256":partition,
        "probe_hash_sha256":probe,
        "ema_q99":ema,
        "instant_q95":inst,
        "reference_w4_sha256":sha256_file(paths["reference_w4"]),
        "recovered_w4_sha256":sha256_file(paths["recovered_w4"]),
        "w4_tensor_max_abs_difference":maxdiff,
        "reference_profile_sha256":sha256_file(paths["reference_profiles"]),
        "recovered_profile_sha256":sha256_file(paths["recovered_profiles"]),
        "profile_npz_keys":profile_keys_rec,
        "feature_calibration_sha256":sha256_file(paths["feature_calibration"]),
        "clean_scores_sha256":sha256_file(paths["clean_scores"]),
        "test_arrays_materialized":False,
        "attack_outcomes_run":False,
    }
    a.record.parent.mkdir(parents=True,exist_ok=True)
    a.record.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("FULL TASK65 WARMUP CALIBRATION RECOVERY = PASS")
    print("W4 TENSOR EXACT:",state_equal)
    print("W4 MAX ABS DIFFERENCE:",maxdiff)
    print("TRUSTED PROFILE ARRAYS EXACT:",profile_equal)
    print("PARTITION HASH:",partition)
    print("PROBE HASH:",probe)
    print("EMA Q99:",f"{ema:.6f}")
    print("INSTANT Q95:",f"{inst:.6f}")
    print("TEST ARRAYS MATERIALIZED: False")
    print("ATTACK OUTCOMES RUN: False")
    return 0

if __name__=="__main__":
    raise SystemExit(main())

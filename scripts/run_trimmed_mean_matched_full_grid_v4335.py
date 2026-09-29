#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/"scripts"/"run_reviewer_trimmed_mean_matched_v4334.py"
SEEDS=[1379954285,1886033230,480705558,1377035733,1707771978]
ATTACKS=["all_to_one_benign","cyclic_shift","multiclass_partial_cycle","pairwise_swap","random_flip"]
PA="all_to_one_benign"; PS=1379954285
DATA=ROOT/"data"/"processed"/"cic_iot_diad_2024_v2_1_recovery_check"/"arrays"/"behavioral_only.npz"
PARTITION=ROOT/"results"/"cic_iot_diad_federated_tuning_v27_alpha05_seed42_recovery_check"/"partitions"/"client_partitions.npz"
PREFLIGHT=ROOT/"results"/"reviewer_trimmed_mean_attacked_preflight_v4334"/PA/f"seed_{PS}"
GRID=ROOT/"results"/"reviewer_trimmed_mean_matched_grid_v4335"
PH="5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"

def evidence_path(attack,seed):
    return ROOT/"reviewer_revision"/f"TASK65_ATTACK_MANIFEST_RECOVERY_SEED_{seed}_{attack.upper()}_v4324.json"
def clean_seed_dir(seed):
    return ROOT/"results"/"reviewer_task65_warmup_recovery_v4323"/f"seed_{seed}"/"clean_seed_record"
def warmup_dir(seed):
    return ROOT/"results"/"reviewer_task65_warmup_recovery_v4323"/f"seed_{seed}"/"warmup"
def plain_branch_dir(attack,seed):
    return ROOT/"results"/"reviewer_task65_attack_manifest_recovery_v4324"/attack/f"seed_{seed}"/"plain_fedavg"

def validate_condition(out,attack,seed):
    mp=out/"REVIEWER_TRIMMED_MEAN_MATCHED_COMPLETE.json"
    rp=out/"tables"/"trimmed_mean_round_metrics.csv"
    kp=out/"checkpoints"/"reviewer_round_checkpoints"/"global_round_08_model.pt"
    ep=evidence_path(attack,seed)
    for p in (mp,rp,kp,ep):
        if not p.exists(): raise FileNotFoundError(p)
    m=json.loads(mp.read_text(encoding="utf-8"))
    e=json.loads(ep.read_text(encoding="utf-8"))
    r=pd.read_csv(rp)
    checks=[
      m.get("phase")=="reviewer_matched_trimmed_mean_comparator",
      m.get("mode")=="strong_attack",
      m.get("attack_type")==attack,
      int(m.get("model_seed",-1))==seed,
      int(m.get("num_clients",-1))==20,
      int(m.get("continuation_rounds",-1))==4,
      m.get("partition_hash_sha256")==PH,
      m.get("poison_index_hash_sha256")==e["poison_index_hash_sha256"],
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
    ]
    if not all(checks):
        bad=[i for i,v in enumerate(checks,1) if not v]
        raise RuntimeError(f"Condition validation failed attack={attack} seed={seed} checks={bad}")

def run_condition(attack,seed,out):
    cmd=[sys.executable,str(RUNNER),
      "--mode","strong_attack","--attack-type",attack,
      "--data-file",str(DATA),"--partition-file",str(PARTITION),
      "--clean-seed-dir",str(clean_seed_dir(seed)),"--warmup-dir",str(warmup_dir(seed)),
      "--plain-branch-dir",str(plain_branch_dir(attack,seed)),"--output-dir",str(out),
      "--model-seed",str(seed),"--num-clients","20","--continuation-rounds","4",
      "--batch-size","2048","--evaluation-batch-size","4096","--learning-rate","0.0003",
      "--weight-decay","0.0001","--max-class-weight","4.0","--gradient-clip-norm","5.0",
      "--threads","6","--source-class","DDoS","--target-class","Benign"]
    subprocess.run(cmd,cwd=ROOT,check=True)

def main():
    for p in (RUNNER,DATA,PARTITION,PREFLIGHT):
        if not p.exists(): raise FileNotFoundError(p)
    print("="*96)
    print("TRIMMED MEAN MATCHED FULL GRID v4.33.5")
    print("25 total conditions; audited preflight reused; remaining 24 run with frozen v4.33.4 runner")
    print("trim_fraction=0.2, trim 4 per side, retain 12 per coordinate")
    print("No final-test access; no outcome-based retuning")
    print("="*96)
    validate_condition(PREFLIGHT,PA,PS)
    print("AUDITED PREFLIGHT REUSE = PASS")
    new=0
    for attack in ATTACKS:
        for seed in SEEDS:
            if attack==PA and seed==PS: continue
            out=GRID/attack/f"seed_{seed}"
            print("\n"+"-"*96)
            print(f"CONDITION attack={attack} seed={seed}")
            print("-"*96)
            if out.exists():
                if not any(out.iterdir()):
                    raise RuntimeError(f"Empty pre-existing output directory; preserve and inspect: {out}")
                validate_condition(out,attack,seed)
                print("EXISTING COMPLETE CONDITION REUSED = PASS")
                continue
            run_condition(attack,seed,out)
            validate_condition(out,attack,seed)
            new+=1
            print("CONDITION COMPLETION VALIDATION = PASS")
    print("\n"+"="*96)
    print("TRIMMED MEAN FULL GRID EXECUTION COMPLETE")
    print("NEW CONDITIONS RUN:",new)
    print("TOTAL CONDITIONS: 25")
    print("TEST SETS ACCESSED: False")
    print("ATTACK SPECIFIC RETUNING: False")
    return 0

if __name__=="__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/"scripts"/"run_reviewer_flame_matched_v4338.py"
SEEDS=[1379954285,1886033230,480705558,1377035733,1707771978]
ATTACKS=["all_to_one_benign","cyclic_shift","multiclass_partial_cycle","pairwise_swap","random_flip"]
PA,PS="all_to_one_benign",1379954285
DATA=ROOT/"data"/"processed"/"cic_iot_diad_2024_v2_1_recovery_check"/"arrays"/"behavioral_only.npz"
PART=ROOT/"results"/"cic_iot_diad_federated_tuning_v27_alpha05_seed42_recovery_check"/"partitions"/"client_partitions.npz"
PREFLIGHT=ROOT/"results"/"reviewer_flame_attacked_preflight_v4338"/PA/f"seed_{PS}"
GRID=ROOT/"results"/"reviewer_flame_matched_grid_v4339"
PH="5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"

def ev(a,s): return ROOT/"reviewer_revision"/f"TASK65_ATTACK_MANIFEST_RECOVERY_SEED_{s}_{a.upper()}_v4324.json"
def clean(s): return ROOT/"results"/"reviewer_task65_warmup_recovery_v4323"/f"seed_{s}"/"clean_seed_record"
def warm(s): return ROOT/"results"/"reviewer_task65_warmup_recovery_v4323"/f"seed_{s}"/"warmup"
def plain(a,s): return ROOT/"results"/"reviewer_task65_attack_manifest_recovery_v4324"/a/f"seed_{s}"/"plain_fedavg"

def validate(out,a,s):
    mp=out/"REVIEWER_FLAME_MATCHED_COMPLETE.json"; rp=out/"tables"/"flame_round_metrics.csv"
    cp=out/"tables"/"flame_client_decisions.csv"; sp=out/"tables"/"flame_cosine_similarity_long.csv"
    for p in (mp,rp,cp,sp,ev(a,s)):
        if not p.exists(): raise FileNotFoundError(p)
    m=json.loads(mp.read_text(encoding="utf-8")); e=json.loads(ev(a,s).read_text(encoding="utf-8"))
    r=pd.read_csv(rp); c=pd.read_csv(cp); sim=pd.read_csv(sp)
    checks=[
      m.get("phase")=="reviewer_matched_flame_comparator",m.get("mode")=="strong_attack",
      m.get("attack_type")==a,int(m.get("model_seed",-1))==s,int(m.get("num_clients",-1))==20,
      int(m.get("continuation_rounds",-1))==4,m.get("partition_hash_sha256")==PH,
      m.get("poison_index_hash_sha256")==e["poison_index_hash_sha256"],
      m.get("aggregation_source")=="src/aggregation.py::FLAME",m.get("hdbscan_version")=="0.8.44",
      m.get("hdbscan_version_is_reviewer_frozen_not_recovered_historical") is True,
      int(m.get("hdbscan_min_cluster_size",-1))==11,int(m.get("hdbscan_min_samples",-1))==1,
      m.get("hdbscan_allow_single_cluster") is True,float(m.get("lambda",-1))==0.001,
      float(m.get("noise_scalar",-1))==1.0,m.get("noise_scalar_adaptation_used") is False,
      m.get("preserved_source_uses_normal_std_sigma_squared") is True,int(m.get("noise_seed_offset",-1))==999,
      m.get("test_sets_accessed") is False,m.get("attack_specific_retuning") is False,
      m.get("scientific_outcome_gate_used") is False,len(r)==4,
      list(r["global_round"].astype(int))==[5,6,7,8],set(r["arm"])=={"flame_preserved_project"},
      set(r["hdbscan_min_cluster_size"].astype(int))=={11},set(r["noise_scalar"].astype(float))=={1.0},
      set(r["lambda"].astype(float))=={0.001},len(c)==80,len(sim)==1600]
    if not all(checks):
        raise RuntimeError(f"Condition validation failed attack={a} seed={s} checks={[i for i,v in enumerate(checks,1) if not v]}")

def run(a,s,out):
    cmd=[sys.executable,str(RUNNER),"--mode","strong_attack","--attack-type",a,
      "--data-file",str(DATA),"--partition-file",str(PART),"--clean-seed-dir",str(clean(s)),
      "--warmup-dir",str(warm(s)),"--plain-branch-dir",str(plain(a,s)),"--output-dir",str(out),
      "--model-seed",str(s),"--num-clients","20","--continuation-rounds","4","--batch-size","2048",
      "--evaluation-batch-size","4096","--learning-rate","0.0003","--weight-decay","0.0001",
      "--max-class-weight","4.0","--gradient-clip-norm","5.0","--threads","6",
      "--source-class","DDoS","--target-class","Benign"]
    subprocess.run(cmd,cwd=ROOT,check=True)

def main():
    for p in (RUNNER,DATA,PART,PREFLIGHT):
        if not p.exists(): raise FileNotFoundError(p)
    validate(PREFLIGHT,PA,PS); print("AUDITED FLAME PREFLIGHT REUSE = PASS")
    new=reused=0
    for a in ATTACKS:
      for s in SEEDS:
        if a==PA and s==PS: continue
        out=GRID/a/f"seed_{s}"
        print(f"\nCONDITION attack={a} seed={s}")
        if out.exists():
            if not any(out.iterdir()): raise RuntimeError(f"Empty pre-existing output; preserve: {out}")
            validate(out,a,s); reused+=1; print("EXISTING COMPLETE CONDITION REUSED = PASS")
        else:
            run(a,s,out); validate(out,a,s); new+=1; print("CONDITION COMPLETION VALIDATION = PASS")
    print("FLAME FULL GRID EXECUTION COMPLETE")
    print("NEW CONDITIONS RUN:",new); print("EXISTING GRID CONDITIONS REUSED:",reused)
    print("AUDITED PREFLIGHT REUSED: 1"); print("TOTAL CONDITIONS: 25")
    print("TEST SETS ACCESSED: False"); print("ATTACK SPECIFIC RETUNING: False")
    return 0
if __name__=="__main__": raise SystemExit(main())

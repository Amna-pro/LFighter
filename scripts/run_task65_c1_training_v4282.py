from __future__ import annotations
import csv, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"configs/task65_c1_preregistration_v4282.json").read_text(encoding="utf-8"))
OUT=ROOT/CFG["output_root"]; TRAIN=OUT/"training"
GUARD_DIR=ROOT/"scripts/task65_guard_v4282"; GUARD_LOG=TRAIN/"task65_guard_access_log.jsonl"

def sha256_file(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def git(*args):
    return subprocess.run(["git",*args],cwd=ROOT,check=True,text=True,capture_output=True).stdout.strip()

def partition_hash(path):
    digest=hashlib.sha256()
    with np.load(path,allow_pickle=False) as payload:
        keys=sorted(payload.files)
        if len(keys)!=CFG["num_clients"]: raise RuntimeError("partition client count mismatch")
        for client_id,key in enumerate(keys):
            arr=np.asarray(payload[key],dtype=np.int64)
            digest.update(f"client_{client_id:03d}".encode())
            digest.update(arr.tobytes())
    return digest.hexdigest()

def env_guarded():
    e=os.environ.copy()
    pieces=[str(GUARD_DIR),str(ROOT/"src"),str(ROOT/"scripts")]
    if e.get("PYTHONPATH"): pieces.append(e["PYTHONPATH"])
    e["PYTHONPATH"]=os.pathsep.join(pieces)
    e["LFIGHTER_TASK65_GUARD"]="1"
    e["LFIGHTER_TASK65_GUARD_LOG"]=str(GUARD_LOG)
    return e

def verify_guard(e):
    code="import federated_iot_v26 as f; print(f.load_protocol_arrays.__name__); assert f.load_protocol_arrays.__name__=='task65_guarded_load_protocol_arrays'"
    subprocess.run([sys.executable,"-c",code],cwd=ROOT,env=e,check=True)

def run_stage(name, stage_dir, command, completion, rows, env):
    if completion.exists():
        print("SKIPPING COMPLETE TRAINING STAGE:",name); return
    if (OUT/CFG["test_access_boundary"]["final_access_marker"]).exists():
        raise RuntimeError("Final access already started; training restart is forbidden.")
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True,exist_ok=True)
    print("\n"+"="*80+"\nTRAINING STAGE:",name,"\n"+" ".join(command)+"\n"+"="*80)
    started=time.time(); proc=subprocess.Popen(command,cwd=ROOT,env=env); peak=0
    import psutil
    p=psutil.Process(proc.pid)
    while proc.poll() is None:
        try:
            rss=p.memory_info().rss
            for c in p.children(recursive=True):
                try: rss+=c.memory_info().rss
                except psutil.Error: pass
            peak=max(peak,int(rss))
        except psutil.Error: pass
        time.sleep(0.25)
    rc=proc.wait()
    if rc!=0: raise subprocess.CalledProcessError(rc,command)
    if not completion.exists(): raise RuntimeError(f"Missing completion file: {completion}")
    rows.append({"stage":name,"runtime_seconds":float(time.time()-started),
                 "peak_rss_bytes":int(peak),"completion_file":completion.relative_to(ROOT).as_posix()})
    csvp=TRAIN/"branch_resource_metrics.csv"; csvp.parent.mkdir(parents=True,exist_ok=True)
    with csvp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def ensure_rounds(branch):
    d=branch/"checkpoints/task65_round_checkpoints"
    for r in CFG["monitored_rounds"]:
        p=d/f"global_round_{r:02d}_model.pt"
        if not p.exists(): raise RuntimeError(f"Missing captured round checkpoint: {p}")

def main():
    tag=CFG["required_execution_tag"]; head=git("rev-parse","HEAD"); tc=git("rev-parse",tag+"^{commit}")
    if head!=tc: raise RuntimeError(f"Execution requires HEAD exactly at {tag}")
    if git("status","--porcelain"): raise RuntimeError("Repository must be clean.")

    data=ROOT/CFG["data_file"]; part=ROOT/CFG["partition_file"]
    if not data.exists() or not part.exists(): raise FileNotFoundError("Required data/partition file missing")
    OUT.mkdir(parents=True,exist_ok=True); TRAIN.mkdir(parents=True,exist_ok=True)
    if (OUT/CFG["test_access_boundary"]["final_completion_marker"]).exists():
        raise RuntimeError("Final evaluation already completed.")

    for rel,expected in CFG["frozen_source_hashes"].items():
        if sha256_file(ROOT/rel)!=expected: raise RuntimeError(f"Frozen source hash mismatch: {rel}")

    ph=partition_hash(part)
    records=TRAIN/"clean_seed_records"
    for seed in CFG["final_seeds"]:
        d=records/f"seed_{seed}"; d.mkdir(parents=True,exist_ok=True)
        (d/"seed_metadata.json").write_text(json.dumps({
            "task":65,"seed":seed,"partition_hash_sha256":ph,
            "record_type":"partition_hash_plumbing_only","test_sets_accessed":False
        },indent=2)+"\n",encoding="utf-8")

    e=env_guarded(); verify_guard(e)
    rows=[]
    resource=TRAIN/"branch_resource_metrics.csv"
    if resource.exists():
        with resource.open("r",encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))

    P=CFG["training_parameters"]; py=sys.executable
    warm=ROOT/"scripts/run_true_warmup_v310.py"
    recon_builder=ROOT/"scripts/build_frozen_reconstruction_calibration_v3123.py"
    plain_runner=ROOT/"scripts/run_exact_untargeted_plain_v320a3.py"
    defense_runner=ROOT/"scripts/run_frozen_untargeted_defense_v320b1.py"
    malicious=",".join(map(str,CFG["malicious_clients"]))

    for seed in CFG["final_seeds"]:
        sr=TRAIN/f"seed_{seed}"; clean_record=records/f"seed_{seed}"
        warmup=sr/"warmup"; recon=sr/"reconstruction_calibration"; clean=sr/"clean_reference"

        run_stage(f"seed_{seed}/warmup",warmup,
          [py,str(warm),"--data-file",str(data),"--partition-file",str(part),
           "--clean-seed-dir",str(clean_record),"--output-dir",str(warmup),
           "--model-seed",str(seed),"--num-clients","20","--warmup-rounds","4",
           "--batch-size",str(P["batch_size"]),"--evaluation-batch-size",str(P["evaluation_batch_size"]),
           "--learning-rate",str(P["learning_rate"]),"--weight-decay",str(P["weight_decay"]),
           "--threads",str(P["threads"]),"--probe-per-class",str(P["probe_per_class"]),
           "--probe-seed",str(P["probe_seed"]),"--source-class",P["source_class"],
           "--target-class",P["target_class"],"--ema-decay",str(P["ema_decay"]),
           "--clean-threshold-quantile",str(P["clean_threshold_quantile"])],
          warmup/"true_warmup_v310_metadata.json",rows,e)

        run_stage(f"seed_{seed}/reconstruction_calibration",recon,
          [py,str(recon_builder),"--data-file",str(data),"--partition-file",str(part),
           "--clean-seed-dir",str(clean_record),"--warmup-dir",str(warmup),
           "--output-dir",str(recon),"--model-seed",str(seed),"--num-clients","20","--warmup-rounds","4",
           "--batch-size",str(P["batch_size"]),"--evaluation-batch-size",str(P["evaluation_batch_size"]),
           "--learning-rate",str(P["learning_rate"]),"--weight-decay",str(P["weight_decay"]),
           "--threads",str(P["threads"]),"--probe-per-class",str(P["probe_per_class"]),
           "--probe-seed",str(P["probe_seed"]),"--force-selected-policy",P["reconstruction_policy"]],
          recon/"trusted_update_reconstruction_v312_metadata.json",rows,e)

        common=[
          "--data-file",str(data),"--partition-file",str(part),"--clean-seed-dir",str(clean_record),
          "--warmup-dir",str(warmup),"--model-seed",str(seed),"--num-clients","20",
          "--continuation-rounds","4","--batch-size",str(P["batch_size"]),
          "--evaluation-batch-size",str(P["evaluation_batch_size"]),
          "--learning-rate",str(P["learning_rate"]),"--weight-decay",str(P["weight_decay"]),
          "--threads",str(P["threads"]),"--probe-per-class",str(P["probe_per_class"]),
          "--probe-seed",str(P["probe_seed"]),"--source-class",P["source_class"],
          "--target-class",P["target_class"],"--ema-decay",str(P["ema_decay"]),
          "--malicious-clients",malicious,"--poison-fraction",str(CFG["poison_fraction"]),
          "--attack-seed",str(seed)
        ]
        run_stage(f"seed_{seed}/clean_reference",clean,
          [py,str(plain_runner),"--mode","clean","--attack-type","random_flip","--output-dir",str(clean),*common],
          clean/"exact_untargeted_plain_v320a3_metadata.json",rows,e)
        ensure_rounds(clean)

        for attack in CFG["attacks"]:
            plain=TRAIN/"attacks"/attack/f"seed_{seed}"/"plain_fedavg"
            defended=TRAIN/"attacks"/attack/f"seed_{seed}"/"trusted_reconstruction"
            run_stage(f"{attack}/seed_{seed}/plain_fedavg",plain,
              [py,str(plain_runner),"--mode","strong_attack","--attack-type",attack,"--output-dir",str(plain),*common],
              plain/"exact_untargeted_plain_v320a3_metadata.json",rows,e)
            ensure_rounds(plain)
            run_stage(f"{attack}/seed_{seed}/trusted_reconstruction",defended,
              [py,str(defense_runner),"--mode","strong_attack","--replacement-policy","trusted_reconstruction",
               "--attack-type",attack,"--plain-branch-dir",str(plain),
               "--reconstruction-calibration-dir",str(recon),"--output-dir",str(defended),*common],
              defended/"frozen_untargeted_defense_v320b1_metadata.json",rows,e)
            ensure_rounds(defended)

    if not GUARD_LOG.exists(): raise RuntimeError("Guard log missing")
    for line in GUARD_LOG.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("test_arrays_materialized") is True:
            raise RuntimeError("Forbidden test-array materialization during training")

    (TRAIN/"TASK65_TRAINING_COMPLETE.json").write_text(json.dumps({
      "task":65,"protocol_id":CFG["protocol_id"],"head":head,"partition_hash_sha256":ph,
      "seeds":CFG["final_seeds"],"attacks":CFG["attacks"],"training_complete":True,
      "reserved_test_arrays_materialized_during_training":False,"final_access_started":False
    },indent=2)+"\n",encoding="utf-8")
    print("TASK 65 C1 TRAINING COMPLETE")
    print("FINAL TEST ARRAYS MATERIALIZED DURING TRAINING: False")
    return 0

if __name__=="__main__":
    raise SystemExit(main())

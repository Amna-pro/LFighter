from __future__ import annotations
import hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"configs/task65_c1_preregistration_v4282.json").read_text(encoding="utf-8"))
OUT=ROOT/CFG["output_root"]; TRAIN=OUT/"training"; FINAL=OUT/"final"
ACCESS=OUT/CFG["test_access_boundary"]["final_access_marker"]; COMPLETE=OUT/CFG["test_access_boundary"]["final_completion_marker"]
CLASS_NAMES=CFG["all_eight_classes"]
sys.path.insert(0,str(ROOT/"src"))
from neural_models_v24 import build_model
from federated_iot_v26 import stable_softmax, make_loader

def git(*args): return subprocess.run(["git",*args],cwd=ROOT,check=True,text=True,capture_output=True).stdout.strip()
def sha256_file(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def predict(model,X,batch_size):
    loader=make_loader(X,np.zeros(len(X),dtype=np.int64),batch_size=batch_size,shuffle=False)
    out=[]; model.eval()
    with torch.no_grad():
        for features,_ in loader: out.append(model(features).cpu().numpy())
    return stable_softmax(np.concatenate(out,axis=0))
def checkpoint(branch,r):
    p=branch/"checkpoints/task65_round_checkpoints"/f"global_round_{r:02d}_model.pt"
    if not p.exists(): raise FileNotFoundError(p)
    return p
def eval_ckpt(path,X,y,dataset,arm,seed,attack,r,batch):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    if int(ck.get("global_round",-1))!=r or int(ck.get("model_seed",-1))!=seed: raise RuntimeError("checkpoint identity mismatch")
    model=build_model("resmlp",X.shape[1],len(CLASS_NAMES)); model.load_state_dict(ck["model_state_dict"])
    pred=predict(model,X,batch).argmax(axis=1)
    cm=confusion_matrix(y,pred,labels=np.arange(8))
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=np.arange(8),zero_division=0)
    m={"seed":seed,"attack":attack,"dataset":dataset,"arm":arm,"global_round":r,
       "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
       "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
       "worst_class_recall":float(np.min(recall)),"checkpoint_sha256":sha256_file(path)}
    cr=[{"seed":seed,"attack":attack,"dataset":dataset,"arm":arm,"global_round":r,"class_id":i,
         "class_name":CLASS_NAMES[i],"precision":float(precision[i]),"recall":float(recall[i]),
         "f1":float(f1[i]),"support":int(support[i])} for i in range(8)]
    mr=[]
    for ti,tn in enumerate(CLASS_NAMES):
        for pi,pn in enumerate(CLASS_NAMES):
            mr.append({"seed":seed,"attack":attack,"dataset":dataset,"arm":arm,"global_round":r,
                       "true_id":ti,"true_class":tn,"predicted_id":pi,"predicted_class":pn,"count":int(cm[ti,pi])})
    return m,cr,mr
def main():
    tag=CFG["required_execution_tag"]; head=git("rev-parse","HEAD"); tc=git("rev-parse",tag+"^{commit}")
    if head!=tc: raise RuntimeError("HEAD must equal frozen C1 implementation tag")
    if git("status","--porcelain"): raise RuntimeError("Repository must be clean at final access")
    if ACCESS.exists() or COMPLETE.exists(): raise RuntimeError("Final evaluator rerun prohibited")
    if not (TRAIN/"TASK65_TRAINING_COMPLETE.json").exists(): raise RuntimeError("Training incomplete")

    for seed in CFG["final_seeds"]:
        clean=TRAIN/f"seed_{seed}"/"clean_reference"
        for r in CFG["monitored_rounds"]: checkpoint(clean,r)
        for attack in CFG["attacks"]:
            for arm in ("plain_fedavg","trusted_reconstruction"):
                b=TRAIN/"attacks"/attack/f"seed_{seed}"/arm
                for r in CFG["monitored_rounds"]: checkpoint(b,r)

    data_file=ROOT/CFG["data_file"]
    ACCESS.write_text(json.dumps({"task":65,"protocol_id":CFG["protocol_id"],"time_unix":time.time(),
      "head":head,"data_file_sha256":sha256_file(data_file),"final_evaluation_rerun_permitted":False},indent=2)+"\n",encoding="utf-8")

    with np.load(data_file,allow_pickle=False) as data:
        required=("X_test_natural","y_test_natural","X_test_diagnostic","y_test_diagnostic")
        missing=[k for k in required if k not in data.files]
        if missing: raise KeyError(missing)
        tests={"natural":(data["X_test_natural"].astype(np.float32,copy=False),data["y_test_natural"].astype(np.int64,copy=False)),
               "diagnostic":(data["X_test_diagnostic"].astype(np.float32,copy=False),data["y_test_diagnostic"].astype(np.int64,copy=False))}

    FINAL.mkdir(parents=True,exist_ok=True)
    metrics=[]; classes=[]; conf=[]; batch=int(CFG["training_parameters"]["evaluation_batch_size"])
    for dataset,(X,y) in tests.items():
        if sorted(np.unique(y).tolist())!=list(range(8)): raise RuntimeError(f"{dataset} lacks all 8 classes")
    for seed in CFG["final_seeds"]:
        clean=TRAIN/f"seed_{seed}"/"clean_reference"
        for dataset in CFG["final_datasets"]:
            X,y=tests[dataset]
            for r in CFG["monitored_rounds"]:
                a,b,c=eval_ckpt(checkpoint(clean,r),X,y,dataset,"clean_reference",seed,"clean",r,batch)
                metrics.append(a); classes+=b; conf+=c
        for attack in CFG["attacks"]:
            for arm in ("plain_fedavg","trusted_reconstruction"):
                branch=TRAIN/"attacks"/attack/f"seed_{seed}"/arm
                for dataset in CFG["final_datasets"]:
                    X,y=tests[dataset]
                    for r in CFG["monitored_rounds"]:
                        a,b,c=eval_ckpt(checkpoint(branch,r),X,y,dataset,arm,seed,attack,r,batch)
                        metrics.append(a); classes+=b; conf+=c

    m=pd.DataFrame(metrics); pc=pd.DataFrame(classes); cm=pd.DataFrame(conf)
    m.to_csv(FINAL/"raw_checkpoint_metrics.csv",index=False)
    pc.to_csv(FINAL/"per_class_metrics.csv",index=False)
    cm.to_csv(FINAL/"confusion_matrix_long.csv",index=False)

    detector=[]
    for attack in CFG["attacks"]:
        for seed in CFG["final_seeds"]:
            p=TRAIN/"attacks"/attack/f"seed_{seed}"/"trusted_reconstruction"/"tables/continuation_round_metrics.csv"
            t=pd.read_csv(p)
            for _,row in t.iterrows():
                detector.append({"attack":attack,"seed":seed,"global_round":int(row["global_round"]),
                                 "malicious_client_recall":float(row["malicious_recall"]),
                                 "benign_client_fpr":float(row["benign_false_positive_rate"]),
                                 "reconstructed_client_round_count":int(row["replaced_clients"])})
    dtab=pd.DataFrame(detector)

    paired=[]
    for attack in CFG["attacks"]:
        for seed in CFG["final_seeds"]:
            for dataset in CFG["final_datasets"]:
                for r in CFG["monitored_rounds"]:
                    def one(arm,att):
                        q=m[(m.seed==seed)&(m.dataset==dataset)&(m.arm==arm)&(m.global_round==r)&(m.attack==att)]
                        if len(q)!=1: raise RuntimeError("metric pairing failure")
                        return q.iloc[0]
                    clean=one("clean_reference","clean"); plain=one("plain_fedavg",attack); defense=one("trusted_reconstruction",attack)
                    dq=dtab[(dtab.attack==attack)&(dtab.seed==seed)&(dtab.global_round==r)]
                    if len(dq)!=1: raise RuntimeError("detector pairing failure")
                    dr=dq.iloc[0]; denom=float(clean.macro_f1-plain.macro_f1)
                    removed=float((defense.macro_f1-plain.macro_f1)/denom) if denom>1e-12 else float("nan")
                    paired.append({"seed":seed,"attack":attack,"dataset":dataset,"global_round":r,
                      "clean_macro_f1":float(clean.macro_f1),"plain_macro_f1":float(plain.macro_f1),
                      "defended_macro_f1":float(defense.macro_f1),
                      "clean_balanced_accuracy":float(clean.balanced_accuracy),"plain_balanced_accuracy":float(plain.balanced_accuracy),
                      "defended_balanced_accuracy":float(defense.balanced_accuracy),
                      "clean_worst_class_recall":float(clean.worst_class_recall),"plain_worst_class_recall":float(plain.worst_class_recall),
                      "defended_worst_class_recall":float(defense.worst_class_recall),
                      "attack_excess_macro_f1":denom,"attack_excess_removed_fraction":removed,
                      "malicious_client_recall":float(dr.malicious_client_recall),"benign_client_fpr":float(dr.benign_client_fpr),
                      "reconstructed_client_round_count":int(dr.reconstructed_client_round_count),
                      "primary_endpoint":bool(r==CFG["primary_endpoint_round"])})
    pp=pd.DataFrame(paired); pp.to_csv(FINAL/"paired_primary_metrics.csv",index=False)

    hashes={p.name:sha256_file(p) for p in sorted(FINAL.glob("*.csv"))}
    COMPLETE.write_text(json.dumps({"task":65,"protocol_id":CFG["protocol_id"],"time_unix":time.time(),"head":head,
      "reserved_test_arrays_materialized":True,"raw_checkpoint_metric_rows":len(m),"per_class_rows":len(pc),
      "confusion_rows":len(cm),"paired_primary_rows":len(pp),"negative_results_retained":True,
      "best_round_selection_used":False,"post_outcome_retuning_used":False,"output_sha256":hashes},indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("TASK 65 C1 ONE-SHOT FINAL EVALUATION COMPLETE")
    print("RAW CHECKPOINT METRIC ROWS:",len(m)); print("PAIRED PRIMARY ROWS:",len(pp)); print("DO NOT RERUN.")
    return 0
if __name__=="__main__": raise SystemExit(main())

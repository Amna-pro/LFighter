#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "reviewer_reconstruction_ablation_preflight_v4340c"
AUDIT = ROOT / "reviewer_revision" / "RECONSTRUCTION_ABLATION_PREFLIGHT_AUDIT_v4340c.json"
SEED = 1379954285
PART = "5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"
EXPECTED = {
    "scripts/run_frozen_untargeted_defense_v320b1.py":"5f3852fc13959301b31adf47f57df3b65456fb028a647978b355f976e5aee951",
    "src/trusted_update_reconstruction_v312.py":"af5083ff363ae2de23f802fba5778111171c23c95537793bf2453f5235cec42c",
    "src/independent_anchor_v310.py":"9fee9bb10e6b4381b30e5488aab7650e8e2d3919529f4f7652526bd9e3b452fa",
    "scripts/run_aggregation_headroom_v3112.py":"bfe57c2564d4b38773c390bdba30788cd7fee048bcadecb1d03625a0234ef154",
}
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def load(name):
    d=BASE/name
    meta=json.loads((d/"frozen_untargeted_defense_v320b1_metadata.json").read_text())
    r=pd.read_csv(d/"tables"/"continuation_round_metrics.csv")
    s=pd.read_csv(d/"tables"/"continuation_client_anchor_scores.csv")
    x=pd.read_csv(d/"tables"/"reconstruction_client_rows.csv")
    c=pd.read_csv(d/"tables"/"validation_class_metrics_long.csv")
    f=pd.read_csv(d/"tables"/"validation_confusion_matrix_long.csv")
    k=torch.load(d/"checkpoints"/"v320b1_last_round_model.pt",map_location="cpu",weights_only=False)
    return meta,r,s,x,c,f,k
def main():
    if AUDIT.exists(): raise FileExistsError(AUDIT)
    checks={}
    for p,h in EXPECTED.items():
        checks["hash_"+Path(p).name]=sha(ROOT/p)==h
    C=load("center_only"); H=load("hard_rejection")
    for label,bundle,arm in [("center",C,"center_only"),("hard",H,"hard_rejection")]:
        m,r,s,x,c,f,k=bundle
        checks[label+"_version"]=m.get("experiment_version")=="4.34.0c-RECONSTRUCTION-ABLATION"
        checks[label+"_phase"]=m.get("phase")=="reviewer_reconstruction_mitigation_ablation"
        checks[label+"_arm"]=m.get("ablation_arm")==arm
        checks[label+"_seed"]=int(m.get("model_seed",-1))==SEED
        checks[label+"_partition"]=m.get("partition_hash_sha256")==PART
        checks[label+"_attack"]=m.get("attack_type")=="all_to_one_benign"
        checks[label+"_test_false"]=m.get("test_sets_accessed") is False
        checks[label+"_retune_false"]=m.get("attack_specific_retuning") is False
        checks[label+"_reconcal_false"]=m.get("reconstruction_calibration_used") is False
        checks[label+"_historical_residual_false"]=m.get("historical_residual_used") is False
        checks[label+"_detector_algo_false"]=m.get("detector_algorithm_changed") is False
        checks[label+"_detector_threshold_false"]=m.get("detector_thresholds_changed") is False
        checks[label+"_rounds4"]=len(r)==4 and list(r.global_round.astype(int))==[5,6,7,8]
        checks[label+"_scores80"]=len(s)==80
        checks[label+"_reconrows80"]=len(x)==80
        checks[label+"_classrows32"]=len(c)==32
        checks[label+"_confusion256"]=len(f)==256
        checks[label+"_checkpoint_round8"]=int(k.get("global_round",-1))==8 and k.get("ablation_arm")==arm
        checks[label+"_finite"]=bool(np.isfinite(r[["val_macro_f1","val_balanced_accuracy","malicious_recall","benign_false_positive_rate"]].to_numpy(float)).all())

    cm,cr,cs,cx,_,_,_=C
    hm,hr,hs,hx,_,_,_=H
    cf=cx.flagged.astype(bool); hf=hx.flagged.astype(bool)
    checks["center_flagged_replaced"]=bool(cx.loc[cf,"update_replaced"].astype(bool).all())
    checks["center_none_rejected"]=bool((~cx.update_rejected.astype(bool)).all())
    checks["center_residual_zero"]=bool(np.allclose(cx.loc[cf,"residual_scale"].to_numpy(float),0.0,atol=0,rtol=0))
    checks["center_retains20"]=bool((cr.aggregation_retained_clients.astype(int)==20).all())
    checks["hard_flagged_rejected"]=bool(hx.loc[hf,"update_rejected"].astype(bool).all())
    checks["hard_none_replaced"]=bool((~hx.update_replaced.astype(bool)).all())
    checks["hard_rejected_eq_flagged"]=bool((hr.rejected_clients.astype(int)==hr.flagged_clients.astype(int)).all())
    checks["hard_retained_eq"]=bool((hr.aggregation_retained_clients.astype(int)==20-hr.flagged_clients.astype(int)).all())
    checks["hard_min8_preserved"]=bool((hr.aggregation_retained_clients.astype(int)>=8).all())

    c5=cs[cs.global_round.astype(int)==5].sort_values("client_id").reset_index(drop=True)
    h5=hs[hs.global_round.astype(int)==5].sort_values("client_id").reset_index(drop=True)
    checks["round5_flags_identical"]=np.array_equal(c5.flagged.astype(bool).to_numpy(),h5.flagged.astype(bool).to_numpy())
    for col in ["policy_ratio","ema_ratio","instant_ratio"]:
        checks["round5_"+col+"_identical"]=bool(np.allclose(c5[col].to_numpy(float),h5[col].to_numpy(float),atol=1e-12,rtol=0))

    guard=BASE/"guard_access_log.jsonl"
    txt=guard.read_text(encoding="utf-8",errors="replace").lower() if guard.exists() else ""
    checks["guard_no_test_tokens"]=not any(t in txt for t in ["x_test","y_test","diagnostic_test","natural_test"])

    bad=[k for k,v in checks.items() if not bool(v)]
    if bad: raise RuntimeError("Audit failed: "+", ".join(bad))

    out={
        "protocol":"v4.34.0c reconstruction/mitigation ablation preflight",
        "status":"PASS",
        "checks_passed":len(checks),
        "checks_total":len(checks),
        "checks":checks,
        "scientific_outcome_gate_used":False,
        "test_sets_accessed":False,
        "attack_specific_retuning":False,
        "center_only_round8":cr.loc[cr.global_round.astype(int)==8,["val_macro_f1","val_balanced_accuracy","malicious_recall","benign_false_positive_rate"]].iloc[0].to_dict(),
        "hard_rejection_round8":hr.loc[hr.global_round.astype(int)==8,["val_macro_f1","val_balanced_accuracy","malicious_recall","benign_false_positive_rate"]].iloc[0].to_dict(),
    }
    AUDIT.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("RECONSTRUCTION ABLATION PREFLIGHT AUDIT = PASS")
    print("CHECKS:",len(checks),"/",len(checks))
    print("CENTER ONLY ROUND8 MACRO F1:",out["center_only_round8"]["val_macro_f1"])
    print("HARD REJECTION ROUND8 MACRO F1:",out["hard_rejection_round8"]["val_macro_f1"])
    print("TEST SETS ACCESSED: False")
    print("ATTACK SPECIFIC RETUNING: False")
    print("SCIENTIFIC OUTCOME GATE USED: False")
    return 0
if __name__=="__main__":
    raise SystemExit(main())

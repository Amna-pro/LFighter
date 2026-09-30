# v4.34.0f-a2 AUDIT-ONLY CORRECTION
# Parent audit SHA256: 15F00ADD47C74B4E6CE751DC630E7B974F5798A6772542752D219ADB221D0E63
# No scientific experiment is rerun or modified.
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def load_arm(base: Path,name: str):
    d=base/name
    candidates=list(d.glob("*metadata.json"))
    if not candidates:
        raise FileNotFoundError(f"No metadata JSON in {d}")
    meta_path=candidates[0]
    table=d/"tables"
    round_path=table/"continuation_round_metrics.csv"
    score_path=table/"continuation_client_anchor_scores.csv"
    recon_path=table/"reconstruction_client_rows.csv"
    class_path=table/"validation_class_metrics_long.csv"
    conf_path=table/"validation_confusion_matrix_long.csv"
    ckpt=d/"checkpoints"/"v320b1_last_round_model.pt"
    for p in [round_path,score_path,recon_path,class_path,conf_path,ckpt]:
        if not p.exists(): raise FileNotFoundError(p)
    return (
        json.loads(meta_path.read_text(encoding="utf-8")),
        pd.read_csv(round_path),
        pd.read_csv(score_path),
        pd.read_csv(recon_path),
        pd.read_csv(class_path),
        pd.read_csv(conf_path),
        torch.load(ckpt,map_location="cpu",weights_only=False),
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",type=Path,required=True)
    ap.add_argument("--warmup-recovery-record",type=Path,required=True)
    ap.add_argument("--guard-log",type=Path,required=True)
    ap.add_argument("--record",type=Path,required=True)
    a=ap.parse_args()

    warm=json.loads(a.warmup_recovery_record.read_text(encoding="utf-8"))
    checks={"warmup_recovery_pass":warm.get("status")=="PASS"}

    C=load_arm(a.base,"center_only")
    H=load_arm(a.base,"hard_rejection")
    for label,bundle,arm in [("center",C,"center_only"),("hard",H,"hard_rejection")]:
        m,r,s,x,c,f,k=bundle
        checks[label+"_experiment_version"]=m.get("experiment_version")=="4.34.0e-RECONSTRUCTION-ABLATION"
        checks[label+"_phase"]=m.get("phase")=="reviewer_reconstruction_mitigation_ablation"
        checks[label+"_arm"]=m.get("ablation_arm")==arm
        checks[label+"_seed"]=int(m.get("model_seed",-1))==1379954285
        checks[label+"_attack"]=m.get("attack_type")=="all_to_one_benign"
        checks[label+"_test_false"]=m.get("test_sets_accessed") is False
        checks[label+"_retune_false"]=m.get("attack_specific_retuning") is False
        checks[label+"_recon_calibration_false"]=m.get("reconstruction_calibration_used") is False
        checks[label+"_historical_residual_false"]=m.get("historical_residual_used") is False
        checks[label+"_rounds_5_8"]=len(r)==4 and list(r["global_round"].astype(int))==[5,6,7,8]
        checks[label+"_score_rows_80"]=len(s)==80
        checks[label+"_reconstruction_rows_80"]=len(x)==80
        checks[label+"_class_rows_32"]=len(c)==32
        checks[label+"_confusion_rows_256"]=len(f)==256
        checks[label+"_metrics_finite"]=bool(np.isfinite(
            r[["val_macro_f1","val_balanced_accuracy","malicious_recall","benign_false_positive_rate"]].to_numpy(float)
        ).all())
        checks[label+"_checkpoint_round8"]=int(k.get("global_round",-1))==8
        checks[label+"_checkpoint_arm"]=k.get("ablation_arm")==arm

    cm,cr,cs,cx,_,_,_=C
    hm,hr,hs,hx,_,_,_=H
    cf=cx["flagged"].astype(bool)
    hf=hx["flagged"].astype(bool)

    checks["center_flagged_replaced"]=bool(cx.loc[cf,"update_replaced"].astype(bool).all())
    checks["center_none_rejected"]=bool((~cx["update_rejected"].astype(bool)).all())
    checks["center_residual_scale_zero"]=bool(np.allclose(
        cx.loc[cf,"residual_scale"].to_numpy(float),0.0,atol=0,rtol=0
    ))
    checks["center_retains_all_20"]=bool((cr["aggregation_retained_clients"].astype(int)==20).all())

    checks["hard_flagged_rejected"]=bool(hx.loc[hf,"update_rejected"].astype(bool).all())
    checks["hard_none_replaced"]=bool((~hx["update_replaced"].astype(bool)).all())
    checks["hard_rejected_equals_flagged"]=bool(
        (hr["rejected_clients"].astype(int)==hr["flagged_clients"].astype(int)).all()
    )
    checks["hard_retained_equals_20_minus_flagged"]=bool(
        (hr["aggregation_retained_clients"].astype(int)==20-hr["flagged_clients"].astype(int)).all()
    )
    checks["hard_min8_guard_observed"]=bool((hr["aggregation_retained_clients"].astype(int)>=8).all())

    c5=cs.loc[cs["global_round"].astype(int)==5].sort_values("client_id").reset_index(drop=True)
    h5=hs.loc[hs["global_round"].astype(int)==5].sort_values("client_id").reset_index(drop=True)
    checks["round5_flags_identical"]=np.array_equal(
        c5["flagged"].astype(bool).to_numpy(),h5["flagged"].astype(bool).to_numpy()
    )
    for col in ["policy_ratio","ema_ratio","instant_ratio"]:
        checks["round5_"+col+"_identical"]=bool(np.allclose(
            c5[col].to_numpy(float),h5[col].to_numpy(float),atol=1e-12,rtol=0
        ))

    guard_text=a.guard_log.read_text(encoding="utf-8",errors="replace").lower() if a.guard_log.exists() else ""
    # v4340f-a1: semantic Task-65 guard audit.
    _guard_path = a.guard_log
    _guard_records = [
        json.loads(_line)
        for _line in _guard_path.read_text(encoding="utf-8").splitlines()
        if _line.strip()
    ]
    _allowed_materialized = {"X_train", "y_train", "X_val", "y_val"}
    _reserved_test = {
        "X_test_natural", "y_test_natural",
        "X_test_diagnostic", "y_test_diagnostic",
    }
    _load_events = [
        _r for _r in _guard_records
        if _r.get("event") == "guarded_load_protocol_arrays"
    ]
    _all_materialized = [
        _k
        for _r in _guard_records
        for _k in _r.get("materialized_keys", [])
    ]
    checks["guard_reserved_tests_not_materialized"] = bool(_load_events) and all(
        _r.get("test_arrays_materialized") is False
        for _r in _guard_records
        if "test_arrays_materialized" in _r
    ) and not any(
        _k in _reserved_test for _k in _all_materialized
    ) and all(
        set(_r.get("materialized_keys", [])).issubset(_allowed_materialized)
        for _r in _load_events
    ) and all(
        set(_r.get("test_array_names_verified_only", [])) == _reserved_test
        for _r in _load_events
    )

    failed=[k for k,v in checks.items() if not bool(v)]
    if failed:
        raise RuntimeError("v4340f ablation audit failed: "+", ".join(failed))

    out={
        "protocol":"reviewer_v4340f_reconstruction_mitigation_ablation_preflight",
        "status":"PASS",
        "checks_passed":len(checks),
        "checks_total":len(checks),
        "checks":checks,
        "warmup_recovery_record":str(a.warmup_recovery_record),
        "scientific_outcome_gate_used":False,
        "test_sets_accessed":False,
        "attack_specific_retuning":False,
        "center_only_round8":cr.loc[
            cr["global_round"].astype(int)==8,
            ["val_macro_f1","val_balanced_accuracy","malicious_recall","benign_false_positive_rate"]
        ].iloc[0].to_dict(),
        "hard_rejection_round8":hr.loc[
            hr["global_round"].astype(int)==8,
            ["val_macro_f1","val_balanced_accuracy","malicious_recall","benign_false_positive_rate"]
        ].iloc[0].to_dict(),
    }
    a.record.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
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

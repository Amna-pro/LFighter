#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "reviewer_reconstruction_ablation_preflight_v4340e"
AUDIT = ROOT / "reviewer_revision" / "RECONSTRUCTION_ABLATION_PREFLIGHT_AUDIT_v4340e.json"
MANIFEST = ROOT / "reviewer_revision" / "RECONSTRUCTION_ABLATION_PREFLIGHT_SHA256_v4340e.txt"
SEED = 1379954285
PART = "5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48"

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def manifest_hash(path_text: str) -> str:
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.lower().endswith(path_text.lower()):
            return line.split()[0].lower()
    raise RuntimeError(f"Missing manifest entry: {path_text}")

def load_arm(name: str):
    d=BASE/name
    needed=[
        d/"frozen_untargeted_defense_v320b1_metadata.json",
        d/"tables"/"continuation_round_metrics.csv",
        d/"tables"/"continuation_client_anchor_scores.csv",
        d/"tables"/"reconstruction_client_rows.csv",
        d/"tables"/"validation_class_metrics_long.csv",
        d/"tables"/"validation_confusion_matrix_long.csv",
        d/"checkpoints"/"v320b1_last_round_model.pt",
    ]
    for p in needed:
        if not p.exists(): raise FileNotFoundError(p)
    return (
        json.loads(needed[0].read_text(encoding="utf-8")),
        pd.read_csv(needed[1]),
        pd.read_csv(needed[2]),
        pd.read_csv(needed[3]),
        pd.read_csv(needed[4]),
        pd.read_csv(needed[5]),
        torch.load(needed[6],map_location="cpu",weights_only=False),
    )

def main() -> int:
    if AUDIT.exists(): raise FileExistsError(AUDIT)
    checks={}

    runner=ROOT/"scripts"/"run_reviewer_reconstruction_ablation_v4340e.py"
    checks["derived_runner_matches_preoutcome_manifest"] = (
        sha256_file(runner) == manifest_hash("scripts\\run_reviewer_reconstruction_ablation_v4340e.py")
    )
    checks["parent_runner_hash_exact"] = (
        sha256_file(ROOT/"scripts"/"run_reviewer_reconstruction_ablation_v4340c.py")
        == "d2eb3c8cd2803a2807705c317528830897dbeac5901747c70356db4d54a55d7b"
    )

    C=load_arm("center_only")
    H=load_arm("hard_rejection")

    for label,bundle,arm in [("center",C,"center_only"),("hard",H,"hard_rejection")]:
        m,r,s,x,c,f,k=bundle
        checks[label+"_version"] = m.get("experiment_version")=="4.34.0e-RECONSTRUCTION-ABLATION"
        checks[label+"_phase"] = m.get("phase")=="reviewer_reconstruction_mitigation_ablation"
        checks[label+"_arm"] = m.get("ablation_arm")==arm
        checks[label+"_seed"] = int(m.get("model_seed",-1))==SEED
        checks[label+"_partition"] = m.get("partition_hash_sha256")==PART
        checks[label+"_attack"] = m.get("attack_type")=="all_to_one_benign"
        checks[label+"_test_false"] = m.get("test_sets_accessed") is False
        checks[label+"_retune_false"] = m.get("attack_specific_retuning") is False
        checks[label+"_reconcal_false"] = m.get("reconstruction_calibration_used") is False
        checks[label+"_residual_false"] = m.get("historical_residual_used") is False
        checks[label+"_correction_recorded"] = m.get("preoutcome_correction_from_v4340c") is True
        checks[label+"_guard_removed_recorded"] = m.get("reconstruction_calibration_cli_guard_removed") is True
        checks[label+"_detector_algo_unchanged"] = m.get("detector_algorithm_changed") is False
        checks[label+"_detector_thresholds_unchanged"] = m.get("detector_thresholds_changed") is False
        checks[label+"_rounds"] = len(r)==4 and list(r["global_round"].astype(int))==[5,6,7,8]
        checks[label+"_scores80"] = len(s)==80
        checks[label+"_reconstruction_rows80"] = len(x)==80
        checks[label+"_class_rows32"] = len(c)==32
        checks[label+"_confusion_rows256"] = len(f)==256
        checks[label+"_checkpoint_round8"] = int(k.get("global_round",-1))==8 and k.get("ablation_arm")==arm
        checks[label+"_metrics_finite"] = bool(np.isfinite(
            r[["val_macro_f1","val_balanced_accuracy","malicious_recall","benign_false_positive_rate"]].to_numpy(float)
        ).all())

    cm,cr,cs,cx,_,_,_=C
    hm,hr,hs,hx,_,_,_=H
    cf=cx["flagged"].astype(bool)
    hf=hx["flagged"].astype(bool)

    checks["center_flagged_replaced"] = bool(cx.loc[cf,"update_replaced"].astype(bool).all())
    checks["center_none_rejected"] = bool((~cx["update_rejected"].astype(bool)).all())
    checks["center_flagged_residual_scale_zero"] = bool(
        np.allclose(cx.loc[cf,"residual_scale"].to_numpy(float),0.0,atol=0,rtol=0)
    )
    checks["center_retains_all20"] = bool((cr["aggregation_retained_clients"].astype(int)==20).all())

    checks["hard_flagged_rejected"] = bool(hx.loc[hf,"update_rejected"].astype(bool).all())
    checks["hard_none_replaced"] = bool((~hx["update_replaced"].astype(bool)).all())
    checks["hard_rejected_equals_flagged"] = bool(
        (hr["rejected_clients"].astype(int)==hr["flagged_clients"].astype(int)).all()
    )
    checks["hard_retained_equals_20_minus_flagged"] = bool(
        (hr["aggregation_retained_clients"].astype(int)==20-hr["flagged_clients"].astype(int)).all()
    )
    checks["hard_min8_safety_preserved"] = bool((hr["aggregation_retained_clients"].astype(int)>=8).all())

    c5=cs.loc[cs["global_round"].astype(int)==5].sort_values("client_id").reset_index(drop=True)
    h5=hs.loc[hs["global_round"].astype(int)==5].sort_values("client_id").reset_index(drop=True)
    checks["round5_flags_identical"] = np.array_equal(
        c5["flagged"].astype(bool).to_numpy(), h5["flagged"].astype(bool).to_numpy()
    )
    for col in ["policy_ratio","ema_ratio","instant_ratio"]:
        checks["round5_"+col+"_identical"] = bool(np.allclose(
            c5[col].to_numpy(float), h5[col].to_numpy(float), atol=1e-12, rtol=0
        ))

    guard=BASE/"guard_access_log.jsonl"
    guard_txt=guard.read_text(encoding="utf-8",errors="replace").lower() if guard.exists() else ""
    checks["guard_no_reserved_test_tokens"] = not any(
        t in guard_txt for t in ["x_test","y_test","diagnostic_test","natural_test"]
    )

    bad=[k for k,v in checks.items() if not bool(v)]
    if bad:
        raise RuntimeError("Ablation preflight audit failed: "+", ".join(bad))

    out={
        "protocol":"v4.34.0e reconstruction/mitigation ablation preflight",
        "status":"PASS",
        "checks_passed":len(checks),
        "checks_total":len(checks),
        "checks":checks,
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

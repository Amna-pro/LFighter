from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT/"configs/task67_c3_audit_v4302.json").read_text(encoding="utf-8"))
SRC = ROOT/CFG["source_output_root"]
T66 = ROOT/CFG["task66_root"]
FAILED = ROOT/CFG["failed_audit_json"]
OUT = ROOT/"results/cic_iot_diad_task67_c3_publication_audit_v4302"
FIG = SRC/"figures"
TAB = SRC/"tables"
DEC = SRC/"TASK67_PUBLICATION_OUTPUTS_COMPLETE.json"

def git(*args):
    return subprocess.run(["git",*args],cwd=ROOT,check=True,text=True,capture_output=True).stdout.strip()

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def semantically_equal_csv(a: Path, b: Path):
    da=pd.read_csv(a)
    db=pd.read_csv(b)
    details={
        "columns_equal": list(da.columns)==list(db.columns),
        "row_count_equal": len(da)==len(db),
        "column_count_equal": len(da.columns)==len(db.columns),
        "nan_positions_equal": False,
        "values_equal": False,
    }
    if not (details["columns_equal"] and details["row_count_equal"] and details["column_count_equal"]):
        return False, details
    details["nan_positions_equal"] = bool(da.isna().equals(db.isna()))
    if not details["nan_positions_equal"]:
        return False, details

    ok=True
    for c in da.columns:
        xa=da[c]; xb=db[c]
        # Prefer numeric comparison only if both columns are genuinely numeric after parsing.
        if pd.api.types.is_numeric_dtype(xa) and pd.api.types.is_numeric_dtype(xb):
            va=pd.to_numeric(xa,errors="coerce").to_numpy(float)
            vb=pd.to_numeric(xb,errors="coerce").to_numpy(float)
            mask=np.isfinite(va)&np.isfinite(vb)
            if not np.array_equal(np.isnan(va),np.isnan(vb)):
                ok=False; break
            if mask.any() and not np.allclose(va[mask],vb[mask],rtol=1e-12,atol=1e-12,equal_nan=True):
                ok=False; break
            # infinities, if any, must agree exactly
            nonfinite=~mask & ~np.isnan(va)
            if nonfinite.any() and not np.array_equal(va[nonfinite],vb[nonfinite]):
                ok=False; break
        else:
            sa=xa.fillna("<NA>").astype(str).to_numpy()
            sb=xb.fillna("<NA>").astype(str).to_numpy()
            if not np.array_equal(sa,sb):
                ok=False; break
    details["values_equal"]=bool(ok)
    return bool(ok), details

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks={}

    parent=git("rev-list","-n","1",CFG["parent_tag"])
    head=git("rev-parse","HEAD")
    checks["parent_tag_resolves"]=bool(parent)
    checks["head_equals_frozen_task67_c0"]=head==parent

    checks["original_c2_failed_audit_exists"]=FAILED.exists()
    if not FAILED.exists():
        raise RuntimeError("Original Task67 C2 failed audit record is missing.")
    old=json.loads(FAILED.read_text(encoding="utf-8"))
    checks["original_c2_failure_retained"]=(
        old.get("protocol_id")=="task67_c2_v4301"
        and old.get("all_checks_passed") is False
        and int(old.get("checks_passed",-1))==40
        and int(old.get("checks_total",-1))==41
        and old.get("checks",{}).get("task66_tables_1_to_6_byte_identical") is False
    )

    checks["completion_marker_exists"]=DEC.exists()
    checks["figure_directory_exists"]=FIG.exists()
    checks["table_directory_exists"]=TAB.exists()
    if not all([DEC.exists(),FIG.exists(),TAB.exists()]):
        raise RuntimeError("Task67 C1 output structure is incomplete.")

    d=json.loads(DEC.read_text(encoding="utf-8"))
    checks["decision_head_matches_frozen_c0"]=d.get("head")==parent
    checks["figure_stem_count_recorded_30"]=int(d.get("figure_stem_count",-1))==30
    checks["png_count_recorded_30"]=int(d.get("png_count",-1))==30
    checks["pdf_count_recorded_30"]=int(d.get("pdf_count",-1))==30
    checks["publication_table_count_recorded_7"]=int(d.get("publication_table_count",-1))==7
    checks["negative_null_nonestimable_retained"]=d.get("negative_null_nonestimable_results_retained") is True
    checks["best_seed_selection_unused"]=d.get("best_seed_selection_used") is False
    checks["best_round_selection_unused"]=d.get("best_round_selection_used") is False
    checks["task65_rerun_unused"]=d.get("task65_rerun_used") is False
    checks["task66_inference_rerun_unused"]=d.get("task66_inference_rerun_used") is False
    checks["reserved_npz_access_unused"]=d.get("reserved_npz_accessed") is False
    checks["models_loaded_unused"]=d.get("models_loaded") is False
    checks["training_unused"]=d.get("training_used") is False
    checks["shap_recompute_unused"]=d.get("shap_recomputed") is False
    checks["llm_unused"]=d.get("llm_used") is False

    pngs=sorted(FIG.glob("*.png")); pdfs=sorted(FIG.glob("*.pdf")); tables=sorted(TAB.glob("*.csv"))
    checks["png_files_exact_30"]=len(pngs)==30
    checks["pdf_files_exact_30"]=len(pdfs)==30
    checks["publication_tables_exact_7"]=len(tables)==7
    stems_png={p.stem for p in pngs}; stems_pdf={p.stem for p in pdfs}
    checks["png_pdf_stem_sets_match"]=stems_png==stems_pdf
    checks["figure_stems_unique_30"]=len(stems_png)==30 and len(stems_pdf)==30

    png_readable=True; png_nonblank=True; png_dimensions_ok=True; png_dpi_ok=True
    manifest=[]
    for p in pngs:
        try:
            with Image.open(p) as im:
                im.load()
                w,h=im.size
                dpi=im.info.get("dpi",(None,None))
                stat=ImageStat.Stat(im.convert("RGB"))
                std=max(stat.stddev) if stat.stddev else 0.0
                readable=True
                nonblank=std>0.5
                dimensions_ok=w>=1200 and h>=800
                dx=dpi[0] if isinstance(dpi,(tuple,list)) and len(dpi)>=1 else None
                dy=dpi[1] if isinstance(dpi,(tuple,list)) and len(dpi)>=2 else None
                dpi_ok=(dx is not None and dy is not None and 295<=float(dx)<=305 and 295<=float(dy)<=305)
        except Exception:
            w=h=0; dx=dy=None
            readable=nonblank=dimensions_ok=dpi_ok=False
        png_readable &= readable; png_nonblank &= nonblank
        png_dimensions_ok &= dimensions_ok; png_dpi_ok &= dpi_ok
        manifest.append({
            "stem":p.stem,"png_bytes":p.stat().st_size,"png_sha256":sha256_file(p),
            "width_px":w,"height_px":h,"dpi_x":dx,"dpi_y":dy,
            "png_readable":bool(readable),"png_nonblank":bool(nonblank),
            "png_dimensions_ok":bool(dimensions_ok),"png_dpi_ok":bool(dpi_ok)
        })
    checks["all_png_readable"]=png_readable
    checks["all_png_nonblank"]=png_nonblank
    checks["all_png_dimensions_publication_scale"]=png_dimensions_ok
    checks["all_png_dpi_approximately_300"]=png_dpi_ok

    pdf_by_stem={p.stem:p for p in pdfs}
    pdf_signature_ok=True; pdf_nontrivial_size=True
    for row in manifest:
        p=pdf_by_stem.get(row["stem"])
        sig=False; nontrivial=False
        if p is not None and p.exists():
            with p.open("rb") as f: sig=f.read(5)==b"%PDF-"
            nontrivial=p.stat().st_size>=5000
            row["pdf_bytes"]=p.stat().st_size; row["pdf_sha256"]=sha256_file(p)
        else:
            row["pdf_bytes"]=0; row["pdf_sha256"]=None
        row["pdf_signature_ok"]=bool(sig); row["pdf_nontrivial_size"]=bool(nontrivial)
        pdf_signature_ok &= sig; pdf_nontrivial_size &= nontrivial
    checks["all_pdf_signatures_valid"]=pdf_signature_ok
    checks["all_pdf_files_nontrivial"]=pdf_nontrivial_size

    expected_tables={
        "table01_primary_statistics.csv","table02_primary_seed_values.csv",
        "table03_holm_families.csv","table04_practical_gates.csv",
        "table05_secondary_round_descriptives.csv","table06_resource_summary.csv",
        "table07_claim_boundary_summary.csv",
    }
    checks["publication_table_names_exact"]=set(p.name for p in tables)==expected_tables

    row_counts_ok=True
    for name,n in CFG["expected"]["table_row_counts"].items():
        p=TAB/name
        row_counts_ok &= p.exists() and len(pd.read_csv(p))==int(n)
    checks["publication_table_fixed_row_counts_match"]=row_counts_ok

    semantic_details={}
    semantic_all=True
    for pub,src in CFG["source_table_map"].items():
        ok,details=semantically_equal_csv(TAB/pub,T66/src)
        semantic_details[pub]={"source":src,"semantic_equal":bool(ok),**details}
        semantic_all &= ok
    checks["task66_tables_1_to_6_semantically_identical"]=semantic_all

    # Record that the earlier byte mismatch is expected under reserialization, without changing outputs.
    byte_identity={}
    for pub,src in CFG["source_table_map"].items():
        byte_identity[pub]=sha256_file(TAB/pub)==sha256_file(T66/src)
    checks["at_least_one_reserialized_table_byte_differs"]=not all(byte_identity.values())

    claim=TAB/"table07_claim_boundary_summary.csv"
    cdf=pd.read_csv(claim)
    ctext=" | ".join(cdf.astype(str).fillna("").agg(" ".join,axis=1).tolist())
    checks["claim_boundary_table_six_rows"]=len(cdf)==6
    checks["claim_boundary_min_p_0_0625_present"]="0.0625" in ctext
    checks["claim_boundary_worst_class_recall_present"]="Worst-class recall improvement" in ctext
    checks["claim_boundary_attack_excess_present"]="Attack-excess-removed fraction" in ctext
    checks["claim_boundary_best_run_round_prohibited"]="Best run / best round selection" in ctext and "Prohibited" in ctext

    recorded=d.get("output_sha256",{})
    checks["completion_output_hash_count_exact_67"]=len(recorded)==67
    hash_ok=True; missing=[]
    for rel,expected in recorded.items():
        p=SRC/rel
        if not p.exists():
            missing.append(rel); hash_ok=False
        elif sha256_file(p)!=expected:
            hash_ok=False
    checks["completion_output_hashes_match"]=hash_ok
    checks["completion_output_hashes_missing_zero"]=len(missing)==0
    actual_rel={str(p.relative_to(SRC)).replace("\\","/") for p in [*pngs,*pdfs,*tables]}
    checks["completion_hash_key_set_matches_exact_outputs"]=set(recorded.keys())==actual_rel

    pd.DataFrame(manifest).sort_values("stem").to_csv(OUT/"task67c3_figure_file_manifest_v4302.csv",index=False)
    (OUT/"task67c3_table_semantic_equivalence_v4302.json").write_text(
        json.dumps({
            "comparison_policy":CFG["semantic_comparison"],
            "tables":semantic_details,
            "byte_identity":byte_identity,
            "all_six_semantically_identical":bool(semantic_all),
            "original_c2_byte_identity_failure_retained":True
        },indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )

    checks={k:bool(v) for k,v in checks.items()}
    passed=sum(checks.values()); total=len(checks); ok=passed==total
    out={
      "task":67,"phase":CFG["phase"],"protocol_id":CFG["protocol_id"],
      "source_protocol_id":CFG["source_protocol_id"],
      "failed_audit_protocol_id":CFG["failed_audit_protocol_id"],
      "parent_frozen_c0_commit":parent,"head_at_audit":head,
      "checks":checks,"checks_passed":passed,"checks_total":total,"all_checks_passed":ok,
      "original_c2_failure_retained":True,
      "audit_criterion_correction":"byte identity replaced by strict parsed semantic equivalence because frozen C1 reserializes CSVs with pandas.to_csv",
      "publication_outputs_modified_by_c3":False,
      "publication_outputs_regenerated_by_c3":False,
      "task66_inference_rerun_by_c3":False,
      "task65_rerun_by_c3":False,
      "reserved_npz_opened_by_c3":False,
      "models_loaded_by_c3":False,
      "training_run_by_c3":False,
      "shap_recomputed_by_c3":False,
      "llm_used_by_c3":False,
      "ready_to_freeze_task67_publication_outputs":ok,
      "ready_for_task68_independent_final_audit":ok
    }
    (OUT/"task67c3_publication_integrity_audit_v4302.json").write_text(
        json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )

    print("===== TASK 67 C3 PUBLICATION SEMANTIC AUDIT V4.30.2 =====")
    for k,v in checks.items(): print(f"{k}: {v}")
    print(f"PASS: {passed}/{total}")
    print("ORIGINAL C2 40/41 FAILURE RETAINED: True")
    print("PUBLICATION OUTPUTS MODIFIED BY C3: False")
    print("PUBLICATION OUTPUTS REGENERATED BY C3: False")
    print("TASK66 INFERENCE RERUN BY C3: False")
    print("TASK65 RERUN BY C3: False")
    print("RESERVED NPZ OPENED BY C3: False")
    print("READY TO FREEZE TASK67 PUBLICATION OUTPUTS:",ok)
    print("READY FOR TASK68 INDEPENDENT FINAL AUDIT:",ok)
    return 0 if ok else 2

if __name__=="__main__":
    raise SystemExit(main())

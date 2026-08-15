from __future__ import annotations
import hashlib, json, math, re, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"configs/task68_c0_preregistration_v4310.json").read_text(encoding="utf-8"))
OUT=ROOT/CFG["output_root"]

def git(*args, check=True):
    r=subprocess.run(["git",*args],cwd=ROOT,text=True,capture_output=True)
    if check and r.returncode!=0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    return r.stdout.strip(), r.returncode

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def jsonload(rel):
    return json.loads((ROOT/rel).read_text(encoding="utf-8"))

def exact_csv_rows(rel, n):
    p=ROOT/rel
    return p.exists() and len(pd.read_csv(p))==n

def verify_completion_hashes(root_rel, marker_name):
    root=ROOT/root_rel
    d=json.loads((root/marker_name).read_text(encoding="utf-8"))
    rec=d.get("output_sha256",{})
    ok=bool(rec)
    rows=[]
    for name,h in rec.items():
        p=root/name
        good=p.exists() and sha256(p)==h
        ok &= good
        rows.append((name,h,good))
    return bool(ok),d,rows

def main():
    if OUT.exists():
        raise RuntimeError("Task68 final-audit output root already exists; rerun prohibited.")
    head,_=git("rev-parse","HEAD")
    frozen,_=git("rev-list","-n","1",CFG["required_execution_tag"])
    if head!=frozen:
        raise RuntimeError("Task68 C1 requires HEAD exactly at the frozen Task68 C0 tag.")
    status,_=git("status","--porcelain")
    if status:
        raise RuntimeError("Repository must be clean before independent final audit.")

    OUT.mkdir(parents=True)
    checks={}
    artifacts=[]

    # 1. Independent provenance chain.
    tag_commits=[]
    for tag in CFG["required_tag_chain"]:
        commit,rc=git("rev-list","-n","1",tag,check=False)
        checks[f"tag_resolves__{tag}"]=bool(commit) and rc==0
        tag_commits.append((tag,commit))
    ancestry=True
    for (ta,ca),(tb,cb) in zip(tag_commits,tag_commits[1:]):
        if not ca or not cb:
            ancestry=False; continue
        _,rc=git("merge-base","--is-ancestor",ca,cb,check=False)
        ancestry &= rc==0
    checks["required_tag_chain_is_ancestral"]=ancestry
    checks["task68_parent_is_task67_c3"]=tag_commits[-1][1]==git("rev-list","-n","1",CFG["parent_required_tag"])[0]

    # 2. Final seeds, attacks, and no skipped-task artifacts.
    final=set(CFG["final_seeds"]); dev=set(CFG["development_seeds"]); old=set(CFG["superseded_c1_seeds"])
    checks["five_final_seeds_unique"]=len(final)==5
    checks["final_seeds_disjoint_from_development"]=final.isdisjoint(dev)
    checks["final_seeds_disjoint_from_superseded_c1"]=final.isdisjoint(old)
    tracked,_=git("ls-files","scripts","configs","results")
    names=[x.replace("\\","/").lower() for x in tracked.splitlines()]
    pat=re.compile(r"(^|[/_.-])task(?:46|47|48|49|50|51|52|53|54)(?:[/_.-]|$)")
    skipped=[x for x in names if pat.search(x)]
    checks["task46_to_54_execution_filenames_absent"]=len(skipped)==0

    # 3. Frozen Task65 implementation source hashes.
    source_hash_ok=True
    for rel,expected in CFG["task65_source_sha256"].items():
        p=ROOT/rel
        good=p.exists() and sha256(p)==expected
        source_hash_ok &= good
        artifacts.append({"layer":"Task65 source","path":rel,"sha256":sha256(p) if p.exists() else None,"verified":bool(good)})
    checks["task65_frozen_source_hashes_match"]=source_hash_ok

    # 4. Task65 final evidence independently checked.
    t65=CFG["task65"]; root65=ROOT/t65["root"]
    checks["task65_access_marker_exists"]=(root65/"TASK65_FINAL_ACCESS_STARTED.json").exists()
    checks["task65_completion_marker_exists"]=(root65/"TASK65_FINAL_EVALUATION_COMPLETE.json").exists()
    checks["task65_training_marker_exists"]=(root65/"training/TASK65_TRAINING_COMPLETE.json").exists()
    t65hash,t65dec,t65hashrows=verify_completion_hashes(t65["root"],"TASK65_FINAL_EVALUATION_COMPLETE.json")
    checks["task65_completion_hashes_match"]=t65hash
    checks["task65_reserved_test_materialized_only_final_recorded"]=t65dec.get("reserved_test_arrays_materialized") is True
    checks["task65_negative_results_retained"]=t65dec.get("negative_results_retained") is True
    checks["task65_best_round_unused"]=t65dec.get("best_round_selection_used") is False
    checks["task65_post_outcome_retuning_unused"]=t65dec.get("post_outcome_retuning_used") is False
    tr=json.loads((root65/"training/TASK65_TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
    checks["task65_training_reports_no_reserved_test_materialization"]=tr.get("reserved_test_arrays_materialized") is False
    for rel,n in t65["expected_rows"].items():
        checks[f"task65_rows_exact__{Path(rel).name}"]=exact_csv_rows(f"{t65['root']}/{rel}",n)
    pair=pd.read_csv(root65/"final/paired_primary_metrics.csv")
    checks["task65_pair_seed_set_exact"]=set(pair["seed"].astype(int))==final
    checks["task65_pair_attack_set_exact"]=set(pair["attack"].astype(str))==set(CFG["attacks"])
    checks["task65_pair_dataset_set_exact"]=set(pair["dataset"].astype(str))==set(CFG["datasets"])
    checks["task65_monitored_rounds_exact"]=set(pair["global_round"].astype(int))==set(CFG["monitored_rounds"])
    a65=jsonload(t65["audit"])
    checks["task65_c2_audit_38_of_38"]=a65.get("all_checks_passed") is True and a65.get("checks_passed")==38 and a65.get("checks_total")==38
    checks["task65_c2_no_reruns"]=all([
        a65.get("scientific_results_recomputed") is False,
        a65.get("reserved_test_arrays_opened_by_c2") is False,
        a65.get("training_rerun_by_c2") is False,
        a65.get("final_evaluator_rerun_by_c2") is False,
    ])

    # 5. Task66 stats: independent configuration/result-hash checks.
    t66=CFG["task66"]; root66=ROOT/t66["root"]
    t66hash,t66dec,t66hashrows=verify_completion_hashes(t66["root"],"TASK66_FINAL_STATISTICS_COMPLETE.json")
    checks["task66_completion_hashes_match"]=t66hash
    checks["task66_primary_round_8"]=t66dec.get("primary_round")==8
    checks["task66_final_seed_count_5"]=t66dec.get("final_seed_count")==5
    checks["task66_bootstrap_replicates_20000"]=t66dec.get("bootstrap_replicates")==20000
    checks["task66_bootstrap_seed_650428"]=t66dec.get("bootstrap_seed")==650428
    checks["task66_sign_patterns_32"]=t66dec.get("sign_patterns")==32
    checks["task66_min_exact_p_0_0625"]=float(t66dec.get("minimum_attainable_two_sided_p",-1))==0.0625
    checks["task66_holm_family_size_5"]=t66dec.get("holm_family_size")==5
    checks["task66_negative_results_retained"]=t66dec.get("negative_results_retained") is True
    checks["task66_no_task65_rerun_npz_model"]=all([
        t66dec.get("task65_rerun_used") is False,
        t66dec.get("reserved_npz_accessed") is False,
        t66dec.get("models_loaded") is False,
    ])
    for rel,n in t66["expected_rows"].items():
        checks[f"task66_rows_exact__{Path(rel).name}"]=exact_csv_rows(f"{t66['root']}/{rel}",n)
    s=pd.read_csv(root66/"task66_primary_statistics.csv")
    checks["task66_exact_60_metric_dataset_attack_cells"]=len(s)==60 and len(s.groupby(["metric","dataset","attack"]))==60
    checks["task66_each_holm_family_five_attacks"]=bool((s.groupby(["metric","dataset"]).size()==5).all())
    p=pd.to_numeric(s["exact_sign_flip_p_raw"],errors="coerce").to_numpy(float)
    fp=p[np.isfinite(p)]
    checks["task66_finite_raw_p_not_below_0_0625"]=bool((fp>=0.0625-1e-12).all())
    checks["task66_finite_raw_p_on_1_over_32_grid"]=bool((np.abs(fp*32-np.round(fp*32))<1e-10).all())
    a66=jsonload(t66["audit"])
    checks["task66_c2_audit_47_of_47"]=a66.get("all_checks_passed") is True and a66.get("checks_passed")==47 and a66.get("checks_total")==47

    # 6. Task67 publication layer: independently hash all 67 frozen outputs.
    t67=CFG["task67"]; root67=ROOT/t67["root"]
    dec67=json.loads((root67/"TASK67_PUBLICATION_OUTPUTS_COMPLETE.json").read_text(encoding="utf-8"))
    png=list((root67/"figures").glob("*.png")); pdf=list((root67/"figures").glob("*.pdf")); csv=list((root67/"tables").glob("*.csv"))
    checks["task67_png_count_30"]=len(png)==30
    checks["task67_pdf_count_30"]=len(pdf)==30
    checks["task67_table_count_7"]=len(csv)==7
    checks["task67_completion_counts_exact"]=dec67.get("figure_stem_count")==30 and dec67.get("png_count")==30 and dec67.get("pdf_count")==30 and dec67.get("publication_table_count")==7
    rec67=dec67.get("output_sha256",{})
    checks["task67_completion_hash_count_67"]=len(rec67)==67
    checks["task67_all_67_hashes_match"]=len(rec67)==67 and all((root67/r).exists() and sha256(root67/r)==h for r,h in rec67.items())
    checks["task67_negative_null_nonestimable_retained"]=dec67.get("negative_null_nonestimable_results_retained") is True
    checks["task67_no_reruns_npz_model_training_shap_llm"]=all([
        dec67.get("task65_rerun_used") is False,
        dec67.get("task66_inference_rerun_used") is False,
        dec67.get("reserved_npz_accessed") is False,
        dec67.get("models_loaded") is False,
        dec67.get("training_used") is False,
        dec67.get("shap_recomputed") is False,
        dec67.get("llm_used") is False,
    ])
    failed67=jsonload(t67["failed_c2_audit"])
    checks["task67_original_c2_40_of_41_failure_retained"]=failed67.get("all_checks_passed") is False and failed67.get("checks_passed")==40 and failed67.get("checks_total")==41 and failed67.get("checks",{}).get("task66_tables_1_to_6_byte_identical") is False
    a67=jsonload(t67["audit"])
    checks["task67_c3_audit_46_of_46"]=a67.get("all_checks_passed") is True and a67.get("checks_passed")==46 and a67.get("checks_total")==46 and a67.get("original_c2_failure_retained") is True
    sem=jsonload(t67["semantic_equivalence"])
    checks["task67_tables_1_to_6_semantically_identical"]=sem.get("all_six_semantically_identical") is True and all(v.get("semantic_equal") is True for v in sem.get("tables",{}).values())

    # 7. Claim boundaries from frozen Task66/67 outputs.
    synopsis_path=ROOT/"results/cic_iot_diad_task66_c2_statistics_audit_v4291/task66c2_outcome_synopsis_v4291.json"
    synopsis=json.loads(synopsis_path.read_text(encoding="utf-8"))
    def find(metric,dataset):
        for r in synopsis:
            if r.get("metric")==metric and r.get("dataset")==dataset: return r
        return {}
    mf_d=find("macro_f1","diagnostic"); mf_n=find("macro_f1","natural")
    ba_d=find("balanced_accuracy","diagnostic"); ba_n=find("balanced_accuracy","natural")
    wc_d=find("worst_class_recall","diagnostic"); wc_n=find("worst_class_recall","natural")
    ae_d=find("attack_excess_removed_fraction","diagnostic"); ae_n=find("attack_excess_removed_fraction","natural")
    checks["claim_macro_f1_positive_mean_all_attacks_both_datasets"]=mf_d.get("positive_mean_contrast_attack_count")==5 and mf_n.get("positive_mean_contrast_attack_count")==5
    checks["claim_balanced_accuracy_positive_mean_all_attacks_both_datasets"]=ba_d.get("positive_mean_contrast_attack_count")==5 and ba_n.get("positive_mean_contrast_attack_count")==5
    checks["claim_no_conventional_significance_supported"]=min(x for x in [mf_d.get("minimum_raw_p"),mf_n.get("minimum_raw_p"),ba_d.get("minimum_raw_p"),ba_n.get("minimum_raw_p")] if x is not None)>=0.0625
    checks["claim_worst_class_recall_improvement_not_supported"]=wc_d.get("positive_mean_contrast_attack_count")==0 and wc_n.get("positive_mean_contrast_attack_count")==0
    checks["claim_attack_excess_partly_nonestimable"]=ae_d.get("estimable_attack_count",5)<5 or ae_n.get("estimable_attack_count",5)<5
    cbt=pd.read_csv(root67/"tables/table07_claim_boundary_summary.csv").astype(str).agg(" ".join,axis=1).str.cat(sep=" | ")
    checks["claim_boundary_table_contains_0_0625"]="0.0625" in cbt
    checks["claim_boundary_table_prohibits_best_run_round"]="Best run / best round selection" in cbt and "Prohibited" in cbt

    # 8. Artifact manifest and claim-to-evidence trace.
    for rel in [
        t65["root"]+"/TASK65_FINAL_EVALUATION_COMPLETE.json",
        t65["audit"],
        t66["root"]+"/TASK66_FINAL_STATISTICS_COMPLETE.json",
        t66["audit"],
        t67["root"]+"/TASK67_PUBLICATION_OUTPUTS_COMPLETE.json",
        t67["failed_c2_audit"], t67["audit"], t67["semantic_equivalence"],
    ]:
        p=ROOT/rel
        artifacts.append({"layer":"Final evidence","path":rel,"sha256":sha256(p) if p.exists() else None,"verified":bool(p.exists())})
    for tag,commit in tag_commits:
        artifacts.append({"layer":"Git tag","path":tag,"sha256":commit,"verified":bool(commit)})

    pd.DataFrame(artifacts).to_csv(OUT/"task68_frozen_artifact_manifest_v4310.csv",index=False)
    claims=[
      ["C1","LFighter shows positive mean Macro-F1 contrast for all five frozen attacks on both final datasets.","Task66 outcome synopsis + primary statistics","SUPPORTED DIRECTIONALLY; not conventional p<0.05 significance"],
      ["C2","LFighter shows positive mean balanced-accuracy contrast for all five frozen attacks on both final datasets.","Task66 outcome synopsis + primary statistics","SUPPORTED DIRECTIONALLY; not conventional p<0.05 significance"],
      ["C3","The five-seed exact paired test cannot attain two-sided p<0.05.","Task66 completion marker + exact sign-flip grid","SUPPORTED; minimum p=0.0625"],
      ["C4","Worst-class recall improvement is supported.","Task66 outcome synopsis","NOT SUPPORTED; retain negative/null finding"],
      ["C5","Attack-excess-removed fraction is fully estimable across all attacks/datasets.","Task66 outcome synopsis","NOT SUPPORTED; partly non-estimable"],
      ["C6","Final results were obtained without post-outcome retuning/best-round selection.","Task65 completion + Task66/67 decisions","SUPPORTED within recorded provenance"],
      ["C7","Publication tables preserve Task66 scientific values.","Task67 C3 semantic-equivalence audit","SUPPORTED; C2 byte-identity failure retained"],
      ["C8","Task46-54 contributed executed experiments to the final evidence.","Tracked filenames + frozen protocol history","NOT SUPPORTED; deliberately unexecuted/deferred"],
    ]
    pd.DataFrame(claims,columns=["claim_id","claim","evidence","final_status"]).to_csv(OUT/"task68_claim_to_evidence_trace_v4310.csv",index=False)

    checks={k:bool(v) for k,v in checks.items()}
    passed=sum(checks.values()); total=len(checks); ok=passed==total
    decision={
      "task":68,"phase":"C1_independent_final_audit","protocol_id":"task68_c1_v4310",
      "implementation_protocol_id":CFG["protocol_id"],"head":head,
      "checks":checks,"checks_passed":passed,"checks_total":total,"all_checks_passed":ok,
      "scientific_results_recomputed":False,"training_rerun":False,"models_loaded":False,
      "reserved_npz_opened":False,"task65_rerun":False,"task66_inference_rerun":False,
      "task67_outputs_regenerated":False,"shap_recomputed":False,"llm_used":False,
      "ready_for_literature_review_and_manuscript_planning":ok,
      "paper_drafting_authorized_only_after_50_plus_paper_review":ok
    }
    (OUT/"TASK68_FINAL_AUDIT_COMPLETE.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("===== TASK 68 C1 INDEPENDENT FINAL AUDIT V4.31.0 =====")
    for k,v in checks.items(): print(f"{k}: {v}")
    print(f"PASS: {passed}/{total}")
    print("SCIENTIFIC RESULTS RECOMPUTED: False")
    print("TRAINING RERUN: False")
    print("MODELS LOADED: False")
    print("RESERVED NPZ OPENED: False")
    print("TASK65 RERUN: False")
    print("TASK66 INFERENCE RERUN: False")
    print("TASK67 OUTPUTS REGENERATED: False")
    print("READY FOR LITERATURE REVIEW AND MANUSCRIPT PLANNING:",ok)
    print("PAPER DRAFTING REQUIRES 50+ PAPER REVIEW FIRST:",ok)
    print("DO NOT RERUN TASK68 C1.")
    return 0 if ok else 2

if __name__=="__main__":
    raise SystemExit(main())

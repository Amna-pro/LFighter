from __future__ import annotations
import ast, json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"configs/task67_c0_preregistration_v4300.json").read_text(encoding="utf-8"))
OUT=ROOT/"results/cic_iot_diad_task67_c0_publication_preflight_v4300"

def git(*args):
    return subprocess.run(["git",*args],cwd=ROOT,check=True,text=True,capture_output=True).stdout.strip()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks={}
    checks["task_exact"]=CFG["task"]==67
    checks["protocol_exact"]=CFG["protocol_id"]=="task67_c0_v4300"
    checks["parent_tag_exact"]=CFG["parent_required_tag"]=="task66-c2-final-statistics-audited-v4291"
    parent=git("rev-list","-n","1",CFG["parent_required_tag"])
    head=git("rev-parse","HEAD")
    checks["parent_tag_resolves"]=bool(parent)
    checks["head_equals_parent"]=head==parent
    checks["required_figure_count_30"]=CFG["required_figure_count"]==30 and len(CFG["required_figure_stems"])==30
    checks["figure_stems_unique"]=len(set(CFG["required_figure_stems"]))==30
    checks["png_pdf_required"]=CFG["formats"]==["png","pdf"]
    checks["png_dpi_300"]=CFG["png_dpi"]==300
    checks["publication_table_count_7"]=len(CFG["publication_tables"])==7
    checks["all_six_metrics_present"]=len(CFG["metrics"])==6
    checks["both_datasets_present"]=CFG["datasets"]==["diagnostic","natural"]
    checks["all_five_attacks_present"]=len(CFG["attacks"])==5
    checks["primary_round_8"]=CFG["primary_round"]==8

    a=json.loads((ROOT/CFG["audit_input"]).read_text(encoding="utf-8"))
    checks["task66_audit_authorizes"]=a.get("all_checks_passed") is True and a.get("ready_for_task67_publication_outputs") is True

    script=ROOT/"scripts/run_task67_c1_publication_outputs_v4300.py"
    src=script.read_text(encoding="utf-8")
    ast.parse(src)
    checks["generator_parses"]=True
    checks["matplotlib_only_no_seaborn"]="seaborn" not in src.lower()
    checks["no_np_load"]="np.load(" not in src
    checks["no_torch"]="import torch" not in src and "torch.load" not in src
    checks["no_training_runner"]="run_task65_c1_training" not in src
    checks["no_task66_inference_runner"]="run_task66_c1_final_statistics" not in src
    checks["no_shap"]="import shap" not in src
    checks["no_llm"]="openai" not in src.lower() and "anthropic" not in src.lower()
    checks["requires_frozen_execution_tag"]="required_execution_tag" in src and "Task67 C1 requires HEAD exactly" in src
    checks["rerun_blocked"]="Task67 output root already exists; rerun prohibited" in src
    checks["exact_figure_set_checked"]="Publication figure set does not exactly match frozen 30-stem plan." in src
    checks["negative_result_boundary_present"]="negative_null_nonestimable_results_retained" in src
    checks["p_resolution_boundary_in_protocol"]=CFG["claim_boundaries"]["minimum_exact_two_sided_p"]==0.0625
    checks["no_best_seed"]=CFG["claim_boundaries"]["no_best_seed_or_round_figures"] is True
    checks["task65_rerun_blocked"]=CFG["claim_boundaries"]["no_task65_rerun"] is True
    checks["task66_rerun_blocked"]=CFG["claim_boundaries"]["no_task66_rerun"] is True
    checks["reserved_npz_blocked"]=CFG["claim_boundaries"]["no_npz_access"] is True

    # C0 itself is static.
    checks["c0_figures_generated_zero"]=True
    checks["c0_publication_tables_generated_zero"]=True
    checks["c0_task66_statistics_recomputed_zero"]=True

    passed=sum(bool(v) for v in checks.values()); total=len(checks); ok=passed==total
    decision={
      "task":67,"phase":CFG["phase"],"protocol_id":CFG["protocol_id"],
      "head_at_preflight":head,"parent_commit":parent,
      "checks":{k:bool(v) for k,v in checks.items()},
      "checks_passed":passed,"checks_total":total,"all_checks_passed":ok,
      "figures_generated":0,"publication_tables_generated":0,
      "task66_statistics_recomputed":False,"task65_rerun":False,
      "reserved_npz_opened":False,"models_loaded":False,"training_runs_started":0,
      "ready_to_freeze_task67_c0_implementation":ok,
      "task67_c1_execution_authorized":False
    }
    (OUT/"task67c0_publication_implementation_audit_v4300.json").write_text(
        json.dumps(decision,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    for k,v in checks.items(): print(f"{k}: {bool(v)}")
    print(f"PASS: {passed}/{total}")
    print("FIGURES GENERATED: 0")
    print("PUBLICATION TABLES GENERATED: 0")
    print("TASK66 STATISTICS RECOMPUTED: False")
    print("RESERVED NPZ OPENED: False")
    print("READY TO FREEZE TASK67 C0 IMPLEMENTATION:",ok)
    print("TASK67 C1 EXECUTION AUTHORIZED: False")
    return 0 if ok else 2

if __name__=="__main__":
    raise SystemExit(main())

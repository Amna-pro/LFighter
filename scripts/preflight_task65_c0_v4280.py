
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "task65_c0_preregistration_v4280.json"
OUT = ROOT / "results" / "cic_iot_diad_task65_c0_preflight_v4280"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="strict")

def extract_argparse_flags(source: str) -> list[str]:
    flags = sorted(set(re.findall(r"add_argument\(\s*['\"](--[A-Za-z0-9_-]+)['\"]", source)))
    return flags

def extract_top_level_functions(source: str) -> list[str]:
    tree = ast.parse(source)
    return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}

    checks["task_exact"] = cfg.get("task") == 65
    checks["phase_exact"] = cfg.get("phase") == "C0_preflight_only"
    checks["protocol_exact"] = cfg.get("protocol_id") == "task65_c0_v4280"

    parent_tag = cfg["parent_tag"]
    try:
        parent_commit = git("rev-parse", parent_tag + "^{commit}")
        head_commit = git("rev-parse", "HEAD")
        checks["parent_tag_resolves"] = True
        checks["parent_is_ancestor"] = subprocess.run(
            ["git","merge-base","--is-ancestor",parent_commit,head_commit], cwd=ROOT
        ).returncode == 0
    except Exception:
        parent_commit = ""
        head_commit = ""
        checks["parent_tag_resolves"] = False
        checks["parent_is_ancestor"] = False

    manifest_path = ROOT / cfg["task63_manifest"]
    seed_manifest_path = ROOT / cfg["task64c2_seed_manifest"]
    checks["task63_manifest_exists"] = manifest_path.exists()
    checks["task64c2_seed_manifest_exists"] = seed_manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    seed_rows = []
    if seed_manifest_path.exists():
        with seed_manifest_path.open("r", encoding="utf-8", newline="") as f:
            seed_rows = list(csv.DictReader(f))
    observed_seeds = [int(r["seed"]) for r in seed_rows] if seed_rows else []

    checks["final_seeds_exact"] = observed_seeds == cfg["required_final_seeds"]
    checks["attack_panel_exact"] = manifest.get("final_attack_panel", {}).get("attacks") == cfg["required_attacks"]
    checks["poison_fraction_exact"] = float(manifest.get("final_attack_panel", {}).get("poison_fraction", -1)) == 1.0
    checks["malicious_clients_exact"] = manifest.get("final_attack_panel", {}).get("malicious_clients") == cfg["required_malicious_clients"]
    checks["primary_round_exact"] = manifest.get("final_method", {}).get("primary_endpoint_round") == 8
    checks["datasets_exact"] = manifest.get("reserved_test_evaluation", {}).get("datasets") == ["diagnostic", "natural"]
    checks["best_round_blocked"] = manifest.get("reserved_test_evaluation", {}).get("best_round_selection_permitted") is False
    checks["task46_54_blocked"] = manifest.get("final_attack_panel", {}).get("task46_54_conditions_permitted") is False

    runner_inventory = []
    all_critical_exist = True
    all_sources_parse = True
    for role, rel in cfg["critical_runners"].items():
        p = ROOT / rel
        exists = p.exists()
        all_critical_exist &= exists
        item = {"role": role, "path": rel, "exists": exists}
        if exists:
            src = read_text(p)
            item["sha256"] = sha256_file(p)
            item["bytes"] = p.stat().st_size
            item["argparse_flags"] = extract_argparse_flags(src)
            try:
                item["top_level_functions"] = extract_top_level_functions(src)
                item["ast_parse_ok"] = True
            except SyntaxError:
                item["top_level_functions"] = []
                item["ast_parse_ok"] = False
                all_sources_parse = False
            item["mentions_load_protocol_arrays"] = "load_protocol_arrays" in src
            item["mentions_x_test_natural"] = "X_test_natural" in src
            item["mentions_x_test_diagnostic"] = "X_test_diagnostic" in src
            item["mentions_center_plus_residual"] = "center_plus_residual" in src
        runner_inventory.append(item)

    checks["all_critical_runners_exist"] = all_critical_exist
    checks["all_critical_sources_parse"] = all_sources_parse

    plain_path = ROOT / cfg["critical_runners"]["plain"]
    defense_path = ROOT / cfg["critical_runners"]["defense"]
    plain_src = read_text(plain_path) if plain_path.exists() else ""
    defense_src = read_text(defense_path) if defense_path.exists() else ""

    checks["plain_runner_all_attacks_present"] = all(a in plain_src for a in cfg["required_attacks"])
    checks["defense_runner_all_attacks_present"] = all(a in defense_src for a in cfg["required_attacks"])
    checks["plain_runner_poison_arg_present"] = "--poison-fraction" in plain_src
    checks["defense_runner_poison_arg_present"] = "--poison-fraction" in defense_src
    checks["defense_center_plus_residual_present"] = "center_plus_residual" in defense_src
    checks["defense_trusted_reconstruction_present"] = "trusted_reconstruction" in defense_src
    checks["plain_uses_protocol_loader"] = "load_protocol_arrays" in plain_src
    checks["defense_uses_protocol_loader"] = "load_protocol_arrays" in defense_src
    checks["plain_source_has_no_explicit_test_array_use"] = "X_test_natural" not in plain_src and "X_test_diagnostic" not in plain_src
    checks["defense_source_has_no_explicit_test_array_use"] = "X_test_natural" not in defense_src and "X_test_diagnostic" not in defense_src

    data_candidates = []
    for p in ROOT.rglob("*.npz"):
        rel = p.relative_to(ROOT).as_posix()
        low = rel.lower()
        if ".venv/" in low or "__pycache__/" in low:
            continue
        data_candidates.append({"path": rel, "bytes": p.stat().st_size})
    data_candidates.sort(key=lambda x: (0 if "processed" in x["path"].lower() else 1, x["path"]))

    partition_candidates = []
    for p in ROOT.rglob("*partition*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        low = rel.lower()
        if ".venv/" in low or "__pycache__/" in low:
            continue
        partition_candidates.append({"path": rel, "bytes": p.stat().st_size})
    partition_candidates.sort(key=lambda x: x["path"])

    script_candidates = []
    keywords = ("warmup", "reconstruction", "multiseed", "untargeted")
    scripts_dir = ROOT / "scripts"
    for p in sorted(scripts_dir.glob("*.py")):
        low = p.name.lower()
        if any(k in low for k in keywords):
            src = read_text(p)
            script_candidates.append({
                "path": p.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(p),
                "argparse_flags": extract_argparse_flags(src),
                "mentions_load_protocol_arrays": "load_protocol_arrays" in src,
                "mentions_reserved_test_name": ("X_test_natural" in src or "X_test_diagnostic" in src)
            })

    new_seed_path_hits = []
    for seed in cfg["required_final_seeds"]:
        token = f"seed_{seed}"
        for p in ROOT.rglob(f"*{token}*"):
            if ".venv" in p.parts or ".git" in p.parts:
                continue
            new_seed_path_hits.append({
                "seed": seed,
                "path": p.relative_to(ROOT).as_posix(),
                "is_dir": p.is_dir()
            })

    inventory = {
        "task": 65,
        "phase": "C0_static_inventory",
        "parent_tag": parent_tag,
        "parent_commit": parent_commit,
        "head_commit": head_commit,
        "final_seeds": observed_seeds,
        "runner_inventory": runner_inventory,
        "data_npz_candidates_static_only": data_candidates,
        "partition_candidates_static_only": partition_candidates,
        "candidate_pipeline_scripts": script_candidates,
        "existing_new_seed_path_hits": new_seed_path_hits,
        "npz_files_opened": 0,
        "experiment_modules_imported": 0,
        "experiment_scripts_executed": 0,
        "training_runs": 0,
        "final_test_metrics_computed": 0,
        "reserved_test_arrays_materialized": False,
        "execution_design_requirement": "Task65 C1 must isolate train/validation loader use during training/warmup/reconstruction and load natural/diagnostic arrays only in the explicit final checkpoint evaluation stage."
    }
    (OUT/"task65c0_execution_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True)+"\n", encoding="utf-8"
    )

    checks["npz_files_opened_zero"] = True
    checks["experiment_modules_imported_zero"] = True
    checks["experiment_scripts_executed_zero"] = True
    checks["training_runs_zero"] = True
    checks["final_metrics_computed_zero"] = True
    checks["reserved_test_arrays_materialized_false"] = True
    checks["c1_still_blocked"] = cfg["preflight_rules"]["task65_c1_permitted_before_preflight_freeze"] is False

    passed = sum(bool(v) for v in checks.values())
    total = len(checks)
    all_pass = passed == total

    decision = {
        "task": 65,
        "phase": "C0_preflight_only",
        "protocol_id": cfg["protocol_id"],
        "experiment_version": cfg["experiment_version"],
        "checks": checks,
        "checks_passed": passed,
        "checks_total": total,
        "all_checks_passed": all_pass,
        "parent_tag": parent_tag,
        "parent_commit": parent_commit,
        "head_commit_at_preflight": head_commit,
        "final_seeds_verified": observed_seeds,
        "reserved_test_arrays_materialized": False,
        "npz_files_opened": 0,
        "training_permitted": False,
        "final_metrics_computed": False,
        "ready_to_freeze_task65_c0": all_pass,
        "task65_c1_execution_authorized": False
    }
    (OUT/"task65c0_preflight_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True)+"\n", encoding="utf-8"
    )

    print("===== TASK 65 C0 STATIC PREFLIGHT =====")
    for k, v in checks.items():
        print(f"{k}: {v}")
    print(f"PASS: {passed}/{total}")
    print("READY TO FREEZE TASK 65 C0:", all_pass)
    print("TASK 65 C1 EXECUTION AUTHORIZED: False")
    print("NPZ FILES OPENED: 0")
    print("TRAINING RUNS: 0")
    print("FINAL TEST METRICS COMPUTED: 0")
    print()
    print("===== CANDIDATE PIPELINE SCRIPTS =====")
    for row in script_candidates:
        print(row["path"])
    print()
    print("===== NEW-SEED PREEXISTING PATH HITS =====")
    if new_seed_path_hits:
        for row in new_seed_path_hits[:100]:
            print(row)
    else:
        print("NONE")
    return 0 if all_pass else 2

if __name__ == "__main__":
    raise SystemExit(main())

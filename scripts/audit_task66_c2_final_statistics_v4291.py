from __future__ import annotations
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/task66_c2_audit_v4291.json").read_text(encoding="utf-8"))
SRC = ROOT / CFG["source_output_root"]
OUT = ROOT / "results/cic_iot_diad_task66_c2_statistics_audit_v4291"

FILES = {
    "primary_stats": SRC / "task66_primary_statistics.csv",
    "seed_values": SRC / "task66_primary_seed_values.csv",
    "holm": SRC / "task66_holm_families.csv",
    "gates": SRC / "task66_practical_gates.csv",
    "secondary": SRC / "task66_secondary_round_descriptives.csv",
    "decision": SRC / "TASK66_FINAL_STATISTICS_COMPLETE.json",
}

def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def native_bool(v) -> bool:
    return bool(v)

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = {}

    parent_commit = git("rev-list", "-n", "1", CFG["parent_tag"])
    head = git("rev-parse", "HEAD")
    checks["parent_tag_resolves"] = bool(parent_commit)
    checks["head_equals_frozen_task66_c0"] = head == parent_commit

    for name, path in FILES.items():
        checks[f"{name}_exists"] = path.exists()
    if not all(FILES[k].exists() for k in FILES):
        raise RuntimeError("Required Task66 C1 output file is missing.")

    decision = json.loads(FILES["decision"].read_text(encoding="utf-8"))
    exp = CFG["expected"]

    checks["decision_head_matches_frozen_c0"] = decision.get("head") == parent_commit
    checks["primary_round_recorded_exact"] = int(decision.get("primary_round", -1)) == exp["primary_round"]
    checks["final_seed_count_recorded_exact"] = int(decision.get("final_seed_count", -1)) == exp["seeds"]
    checks["bootstrap_replicates_recorded_exact"] = int(decision.get("bootstrap_replicates", -1)) == exp["bootstrap_replicates"]
    checks["bootstrap_seed_recorded_exact"] = int(decision.get("bootstrap_seed", -1)) == exp["bootstrap_seed"]
    checks["sign_patterns_recorded_exact"] = int(decision.get("sign_patterns", -1)) == exp["sign_patterns"]
    checks["minimum_p_recorded_exact"] = float(decision.get("minimum_attainable_two_sided_p", -1)) == exp["minimum_attainable_two_sided_p"]
    checks["holm_family_size_recorded_exact"] = int(decision.get("holm_family_size", -1)) == exp["holm_family_size"]
    checks["negative_results_retained"] = decision.get("negative_results_retained") is True
    checks["best_round_selection_unused"] = decision.get("best_round_selection_used") is False
    checks["best_seed_selection_unused"] = decision.get("best_seed_selection_used") is False
    checks["task65_rerun_unused"] = decision.get("task65_rerun_used") is False
    checks["reserved_npz_access_unused"] = decision.get("reserved_npz_accessed") is False
    checks["models_loaded_unused"] = decision.get("models_loaded") is False

    stats = pd.read_csv(FILES["primary_stats"])
    seed = pd.read_csv(FILES["seed_values"])
    holm = pd.read_csv(FILES["holm"])
    gates = pd.read_csv(FILES["gates"])
    secondary = pd.read_csv(FILES["secondary"])

    checks["primary_statistics_rows_exact"] = len(stats) == exp["primary_statistics_rows"]
    checks["primary_seed_value_rows_exact"] = len(seed) == exp["primary_seed_value_rows"]
    checks["holm_family_rows_exact"] = len(holm) == exp["holm_family_rows"]
    checks["practical_gate_rows_exact"] = len(gates) == exp["practical_gate_rows"]
    checks["secondary_descriptive_rows_exact"] = len(secondary) == exp["secondary_descriptive_rows"]

    checks["metrics_exact"] = set(stats["metric"].astype(str)) == set(CFG["metrics"])
    checks["datasets_exact"] = set(stats["dataset"].astype(str)) == set(CFG["datasets"])
    checks["attacks_exact"] = set(stats["attack"].astype(str)) == set(CFG["attacks"])
    checks["seed_set_exact"] = set(seed["seed"].astype(int)) == set(CFG["final_seeds"])
    checks["primary_round_only"] = set(stats["global_round"].astype(int)) == {8}
    checks["seed_values_primary_round_only"] = set(seed["global_round"].astype(int)) == {8}

    # Every metric x dataset x attack confirmatory cell must contain exactly five seed rows.
    counts = seed.groupby(["metric", "dataset", "attack"]).size()
    checks["every_confirmatory_cell_has_five_seed_rows"] = (
        len(counts) == 60 and bool((counts == 5).all())
    )

    # Each Holm family remains exactly five attacks; no family shrinkage.
    family_counts = stats.groupby(["metric", "dataset"]).size()
    checks["every_holm_family_has_five_attacks"] = (
        len(family_counts) == 12 and bool((family_counts == 5).all())
    )
    checks["holm_manifest_family_size_exact"] = bool((holm["family_size_required"].astype(int) == 5).all()) and bool((holm["family_size_observed"].astype(int) == 5).all())

    # Exact sign-flip p values are either NaN (explicit not-estimable) or multiples of 1/32,
    # and no finite value can be below the frozen two-sided resolution limit.
    p = pd.to_numeric(stats["exact_sign_flip_p_raw"], errors="coerce").to_numpy(float)
    finite_p = p[np.isfinite(p)]
    checks["finite_raw_p_not_below_resolution"] = bool((finite_p >= 0.0625 - 1e-12).all())
    checks["finite_raw_p_on_32_pattern_grid"] = bool((np.abs(finite_p * 32.0 - np.round(finite_p * 32.0)) < 1e-10).all())

    padj = pd.to_numeric(stats["holm_p_adjusted"], errors="coerce").to_numpy(float)
    both = np.isfinite(p) & np.isfinite(padj)
    checks["holm_adjusted_not_below_raw"] = bool((padj[both] + 1e-12 >= p[both]).all())

    # Status consistency: no silent dropping of non-finite seed values.
    nfin = stats["n_finite"].astype(int).to_numpy()
    bstat = stats["bootstrap_status"].astype(str).to_numpy()
    sstat = stats["sign_flip_status"].astype(str).to_numpy()
    checks["nonfinite_cells_explicitly_not_estimable"] = all(
        (nf == 5) or ("NOT_ESTIMABLE" in bs and "NOT_ESTIMABLE" in ss)
        for nf, bs, ss in zip(nfin, bstat, sstat)
    )
    checks["finite_cells_bootstrap_signflip_ok"] = all(
        (nf != 5) or (bs == "OK" and ss == "OK")
        for nf, bs, ss in zip(nfin, bstat, sstat)
    )

    # Confirmatory and descriptive separation.
    checks["secondary_rounds_exact"] = set(secondary["global_round"].astype(int)) == {5, 6, 7, 8}
    checks["secondary_confirmatory_flag_false"] = not bool(secondary["confirmatory_inference"].astype(bool).any())

    # Practical gate table is exactly 2 datasets x 5 attacks and never claims a recomputed clean-defense gate.
    gate_pairs = set(zip(gates["dataset"].astype(str), gates["attack"].astype(str)))
    expected_gate_pairs = set((d, a) for d in CFG["datasets"] for a in CFG["attacks"])
    checks["practical_gate_coverage_exact"] = gate_pairs == expected_gate_pairs
    checks["clean_utility_gate_not_recomputed"] = not bool(gates["clean_utility_gate_recomputed_on_reserved_test"].astype(bool).any())

    # Completion hashes must match all recorded CSV outputs.
    recorded = decision.get("output_sha256", {})
    hash_checks = {}
    for name, expected_hash in recorded.items():
        path = SRC / name
        hash_checks[name] = path.exists() and sha256_file(path) == expected_hash
    checks["completion_hashes_present"] = len(recorded) >= 5
    checks["completion_hashes_match"] = len(recorded) >= 5 and all(hash_checks.values())

    # Outcome synopsis is copied/derived only from already-computed C1 CSV values; no inference is rerun.
    synopsis = []
    for metric in CFG["metrics"]:
        for dataset in CFG["datasets"]:
            q = stats[(stats.metric == metric) & (stats.dataset == dataset)]
            raw = pd.to_numeric(q["exact_sign_flip_p_raw"], errors="coerce")
            adj = pd.to_numeric(q["holm_p_adjusted"], errors="coerce")
            synopsis.append({
                "metric": metric,
                "dataset": dataset,
                "finite_raw_p_count": int(raw.notna().sum()),
                "minimum_raw_p": float(raw.min()) if raw.notna().any() else None,
                "minimum_holm_adjusted_p": float(adj.min()) if adj.notna().any() else None,
                "positive_mean_contrast_attack_count": int((pd.to_numeric(q["mean_contrast"], errors="coerce") > 0).sum()),
                "estimable_attack_count": int((q["n_finite"].astype(int) == 5).sum()),
            })
    (OUT / "task66c2_outcome_synopsis_v4291.json").write_text(
        json.dumps(synopsis, indent=2) + "\n", encoding="utf-8"
    )

    checks = {k: native_bool(v) for k, v in checks.items()}
    passed = sum(checks.values())
    total = len(checks)
    ok = passed == total

    out = {
        "task": 66,
        "phase": CFG["phase"],
        "protocol_id": CFG["protocol_id"],
        "source_protocol_id": CFG["source_protocol_id"],
        "parent_frozen_c0_commit": parent_commit,
        "head_at_audit": head,
        "checks": checks,
        "checks_passed": passed,
        "checks_total": total,
        "all_checks_passed": ok,
        "statistics_recomputed_by_c2": False,
        "task65_rerun_by_c2": False,
        "reserved_npz_opened_by_c2": False,
        "models_loaded_by_c2": False,
        "training_run_by_c2": False,
        "verified_output_hashes": hash_checks,
        "ready_to_freeze_task66_final_statistics": ok,
        "ready_for_task67_publication_outputs": ok,
    }
    (OUT / "task66c2_statistics_integrity_audit_v4291.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("===== TASK 66 C2 FINAL STATISTICS AUDIT V4.29.1 =====")
    for k, v in checks.items():
        print(f"{k}: {v}")
    print(f"PASS: {passed}/{total}")
    print("STATISTICS RECOMPUTED BY C2: False")
    print("TASK65 RERUN BY C2: False")
    print("RESERVED NPZ OPENED BY C2: False")
    print("MODELS LOADED BY C2: False")
    print("READY TO FREEZE TASK66 FINAL STATISTICS:", ok)
    print("READY FOR TASK67 PUBLICATION OUTPUTS:", ok)
    print()
    print("===== OUTCOME SYNOPSIS (EXISTING C1 VALUES ONLY) =====")
    for row in synopsis:
        print(
            f"{row['metric']} | {row['dataset']} | "
            f"estimable={row['estimable_attack_count']}/5 | "
            f"positive_mean={row['positive_mean_contrast_attack_count']}/5 | "
            f"min_raw_p={row['minimum_raw_p']} | "
            f"min_holm_p={row['minimum_holm_adjusted_p']}"
        )
    return 0 if ok else 2

if __name__ == "__main__":
    raise SystemExit(main())

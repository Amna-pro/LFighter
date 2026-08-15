from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

TEXT_EXTENSIONS = {
    ".bat", ".cfg", ".csv", ".ini", ".ipynb", ".json", ".log", ".md",
    ".ps1", ".py", ".rst", ".sh", ".toml", ".tsv", ".txt", ".xml",
    ".yaml", ".yml",
}
SKIP_DIR_NAMES = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
RESERVED_TEST_TOKENS = {"x_test_natural", "y_test_natural", "x_test_diagnostic", "y_test_diagnostic"}


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def derive(template: str, idx: int, minimum: int, maximum: int):
    text = template.format(candidate_index=idx)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    raw = int.from_bytes(bytes.fromhex(digest)[:8], "big", signed=False)
    return minimum + (raw % (maximum - minimum + 1)), digest


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def skip_path(rel: Path) -> bool:
    parts = [p.lower() for p in rel.parts]
    if any(p in SKIP_DIR_NAMES for p in parts):
        return True
    joined = "/".join(parts)
    if "task64" in joined:
        return True
    if any(token in joined for token in RESERVED_TEST_TOKENS):
        return True
    return False


def independent_usage_scan(root: Path, candidates: set[int]):
    hits = defaultdict(list)
    values = {str(v): v for v in candidates}
    alt = "|".join(sorted((re.escape(x) for x in values), key=len, reverse=True))
    token_re = re.compile(rf"(?<!\d)({alt})(?!\d)")
    unreadable = []
    text_read = 0

    for path in root.rglob("*"):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if skip_path(rel) or not path.is_file():
            continue
        rel_text = rel.as_posix()
        for m in token_re.finditer(rel_text):
            hits[values[m.group(1)]].append((rel_text, "path"))
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            with path.open("r", encoding="utf-8", errors="strict") as handle:
                text_read += 1
                for line_no, line in enumerate(handle, start=1):
                    for m in token_re.finditer(line):
                        hits[values[m.group(1)]].append((f"{rel_text}:{line_no}", "content"))
        except UnicodeDecodeError:
            unreadable.append(rel_text + " [UnicodeDecodeError]")
        except OSError as exc:
            unreadable.append(rel_text + f" [{type(exc).__name__}]")
    return hits, unreadable, text_read


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/task64_c2_preregistration_v4271.json").read_text(encoding="utf-8"))
    out = root / cfg["outputs"]["root"]
    manifest_path = out / cfg["outputs"]["seed_manifest"]
    decision_path = out / cfg["outputs"]["reservation_decision"]
    audit_path = out / cfg["outputs"]["audit_decision"]

    manifest = load_csv(manifest_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    policy = cfg["seed_policy"]

    development = {int(x) for x in policy["development_seeds_ineligible"]}
    superseded = {int(x) for x in policy["superseded_task64_c1_seeds_ineligible"]}
    banned = development | superseded
    seeds = [int(row["seed"]) for row in manifest]

    checks = {}
    checks["task_exact"] = cfg.get("task") == 64
    checks["phase_exact"] = cfg.get("phase") == "C2_replacement_seed_reservation"
    checks["protocol_exact"] = cfg.get("protocol_id") == "task64_c2_v4271"
    checks["parent_tag_exact"] = cfg.get("parent_tag") == "task63-c2-final-manifest-frozen-v4262"

    parent_commit_now = git(root, "rev-parse", cfg["parent_tag"] + "^{commit}")
    checks["parent_commit_matches_decision"] = parent_commit_now == decision.get("parent_commit")
    checks["parent_is_ancestor"] = subprocess.run(
        ["git", "merge-base", "--is-ancestor", parent_commit_now, git(root, "rev-parse", "HEAD")], cwd=root
    ).returncode == 0

    checks["final_seed_count_exact"] = len(seeds) == int(policy["final_seed_count"]) == 5
    checks["seeds_unique"] = len(set(seeds)) == len(seeds)
    checks["development_seeds_excluded"] = not (set(seeds) & development)
    checks["superseded_c1_seeds_excluded"] = not (set(seeds) & superseded)
    checks["all_banned_seeds_excluded"] = not (set(seeds) & banned)
    checks["manifest_hit_counts_zero"] = all(int(row["preexisting_usage_hit_count"]) == 0 for row in manifest)
    checks["manifest_status_exact"] = all(
        row["status"] == "RESERVED_UNTOUCHED_REPLACEMENT_FINAL_SEED" for row in manifest
    )

    derivation_ok = True
    for row in manifest:
        expected_seed, expected_digest = derive(
            policy["derivation_namespace"],
            int(row["candidate_index"]),
            int(policy["minimum_seed"]),
            int(policy["maximum_seed"]),
        )
        if expected_seed != int(row["seed"]) or expected_digest != row["sha256"]:
            derivation_ok = False
            break
    checks["manifest_derivation_recomputed"] = derivation_ok

    pool = []
    pool_values = set()
    for idx in range(int(policy["candidate_pool_size"])):
        value, digest = derive(
            policy["derivation_namespace"], idx,
            int(policy["minimum_seed"]), int(policy["maximum_seed"])
        )
        pool.append((idx, value, digest))
        pool_values.add(value)

    independent_hits, independent_unreadable, audit_text_read = independent_usage_scan(root, pool_values)

    independent_selected = []
    independent_selected_values = set()
    for idx, value, digest in pool:
        if value in banned or value in independent_selected_values or independent_hits.get(value):
            continue
        independent_selected.append((idx, value, digest))
        independent_selected_values.add(value)
        if len(independent_selected) == int(policy["final_seed_count"]):
            break

    manifest_triples = [(int(r["candidate_index"]), int(r["seed"]), r["sha256"]) for r in manifest]
    checks["independent_prior_usage_rescan_zero"] = all(not independent_hits.get(seed) for seed in seeds)
    checks["independent_unreadable_text_zero"] = len(independent_unreadable) == 0
    checks["earliest_eligible_selection_recomputed"] = independent_selected == manifest_triples
    checks["independent_text_files_read_positive"] = audit_text_read > 0

    checks["manifest_sha256_matches"] = (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() == decision.get("seed_manifest_sha256")
    )
    checks["decision_development_list_exact"] = set(decision.get("development_seeds_excluded", [])) == development
    checks["decision_superseded_list_exact"] = set(decision.get("superseded_task64_c1_seeds_excluded", [])) == superseded
    checks["selection_outcome_free"] = decision.get("selection_uses_model_or_test_outcomes") is False
    checks["manual_substitution_blocked"] = decision.get("manual_seed_substitution_permitted") is False
    checks["no_selected_prior_usage_hits"] = int(decision.get("selected_seed_prior_usage_hits", -1)) == 0
    checks["no_unreadable_eligible_text"] = int(decision.get("unreadable_eligible_text_files", -1)) == 0
    checks["reserved_test_paths_unopened"] = int(decision.get("reserved_test_paths_opened", -1)) == 0
    checks["binary_arrays_unloaded"] = int(decision.get("binary_arrays_loaded", -1)) == 0
    checks["training_blocked"] = decision.get("training_permitted") is False and cfg["data_boundary"]["training_permitted"] is False
    checks["new_shap_blocked"] = int(decision.get("new_shap_evaluations", -1)) == 0 and cfg["data_boundary"]["new_shap_evaluations_permitted"] is False
    checks["llm_blocked"] = int(decision.get("llm_calls", -1)) == 0 and cfg["data_boundary"]["llm_calls_permitted"] is False
    checks["task64_c1_seeds_forbidden_in_task65"] = cfg["task65_handoff"]["task64_c1_superseded_seeds_permitted"] is False
    checks["task65_seed_lock_frozen"] = cfg["task65_handoff"]["seed_set_change_after_task64_c2_freeze_permitted"] is False
    checks["one_shot_frozen"] = cfg["task65_handoff"]["one_shot_final_evaluation"] is True
    checks["task65_waits_for_freeze"] = cfg["task65_handoff"]["task65_unblocked_only_after_task64_c2_freeze"] is True
    checks["task65_not_permitted_before_freeze"] = decision.get("task65_permitted_before_freeze") is False

    passed = sum(bool(v) for v in checks.values())
    total = len(checks)
    all_pass = passed == total

    audit = {
        "task": 64,
        "phase": "C2_replacement_seed_reservation",
        "protocol_id": cfg["protocol_id"],
        "experiment_version": "4.27.1.audit",
        "checks": checks,
        "checks_passed": passed,
        "checks_total": total,
        "all_checks_passed": all_pass,
        "final_replacement_seeds_verified": seeds,
        "superseded_task64_c1_seeds_verified_excluded": sorted(superseded),
        "independent_text_files_read": audit_text_read,
        "independent_selected_seed_usage_hits": sum(len(independent_hits.get(seed, [])) for seed in seeds),
        "reserved_test_arrays_materialized": False,
        "ready_to_freeze_task64_c2": all_pass,
        "task65_unblocked_after_freeze_only": all_pass,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for key, value in checks.items():
        print(f"{key}: {value}")
    print(f"PASS: {passed}/{total}")
    print("READY TO FREEZE TASK 64 C2:", all_pass)
    print("TASK 65 MAY START ONLY AFTER TASK 64 C2 COMMIT AND TAG:", all_pass)
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

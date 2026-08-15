from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

TEXT_EXTENSIONS = {
    ".bat", ".cfg", ".csv", ".ini", ".ipynb", ".json", ".log", ".md",
    ".ps1", ".py", ".rst", ".sh", ".toml", ".tsv", ".txt", ".xml",
    ".yaml", ".yml",
}
SKIP_DIR_NAMES = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
RESERVED_TEST_TOKENS = {"x_test_natural", "y_test_natural", "x_test_diagnostic", "y_test_diagnostic"}


def run_git(root: Path, *args: str) -> str:
    cp = subprocess.run(["git", *args], cwd=root, check=True, text=True, capture_output=True)
    return cp.stdout.strip()


def candidate_value(namespace_template: str, candidate_index: int, minimum: int, maximum: int) -> Tuple[int, str]:
    namespace = namespace_template.format(candidate_index=candidate_index)
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    raw = int.from_bytes(bytes.fromhex(digest)[:8], "big", signed=False)
    seed = minimum + (raw % (maximum - minimum + 1))
    return seed, digest


def should_skip_path(rel: Path) -> bool:
    lower_parts = [p.lower() for p in rel.parts]
    if any(part in SKIP_DIR_NAMES for part in lower_parts):
        return True
    joined = "/".join(lower_parts)
    if "task64" in joined:
        return True
    if any(token in joined for token in RESERVED_TEST_TOKENS):
        return True
    return False


def scan_prior_usage(root: Path, candidates: Set[int]) -> Tuple[Dict[int, List[Tuple[str, str]]], int, int, List[str]]:
    hits: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    candidate_strings = {str(x): x for x in candidates}
    alt = "|".join(sorted((re.escape(s) for s in candidate_strings), key=len, reverse=True))
    token_re = re.compile(rf"(?<!\d)({alt})(?!\d)")
    files_considered = 0
    text_files_read = 0
    unreadable: List[str] = []

    for path in root.rglob("*"):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if should_skip_path(rel):
            continue
        if not path.is_file():
            continue
        files_considered += 1

        rel_text = rel.as_posix()
        for m in token_re.finditer(rel_text):
            seed = candidate_strings[m.group(1)]
            hits[seed].append((rel_text, "path"))

        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            with path.open("r", encoding="utf-8", errors="strict") as handle:
                text_files_read += 1
                for line_no, line in enumerate(handle, start=1):
                    for m in token_re.finditer(line):
                        seed = candidate_strings[m.group(1)]
                        hits[seed].append((f"{rel_text}:{line_no}", "content"))
        except UnicodeDecodeError:
            # A text-like extension that is not UTF-8 is still part of the audit obligation.
            unreadable.append(rel_text + " [UnicodeDecodeError]")
        except OSError as exc:
            unreadable.append(rel_text + f" [{type(exc).__name__}]")

    return hits, files_considered, text_files_read, unreadable


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()

    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "task64_preregistration_v4270.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    parent_tag = cfg["parent_tag"]
    parent_commit = run_git(root, "rev-parse", f"{parent_tag}^{{commit}}")
    head_commit = run_git(root, "rev-parse", "HEAD")
    ancestor_rc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", parent_commit, head_commit], cwd=root
    ).returncode
    if ancestor_rc != 0:
        raise RuntimeError(f"Parent Task63 tag {parent_tag} is not an ancestor of HEAD")

    policy = cfg["seed_policy"]
    minimum = int(policy["minimum_seed"])
    maximum = int(policy["maximum_seed"])
    pool_size = int(policy["candidate_pool_size"])
    final_count = int(policy["final_seed_count"])
    banned = {int(x) for x in policy["development_seeds_ineligible"]}
    namespace_template = policy["derivation_namespace"]

    candidate_rows = []
    candidate_set: Set[int] = set()
    for idx in range(pool_size):
        seed, digest = candidate_value(namespace_template, idx, minimum, maximum)
        candidate_rows.append((idx, seed, digest))
        candidate_set.add(seed)

    hits, files_considered, text_files_read, unreadable = scan_prior_usage(root, candidate_set)
    if unreadable:
        raise RuntimeError("Unreadable eligible text files found; Task64 fails conservatively: " + "; ".join(unreadable[:10]))

    selected: List[dict] = []
    rejected: List[dict] = []
    selected_values: Set[int] = set()
    usage_hit_rows: List[dict] = []

    for seed, seed_hits in sorted(hits.items()):
        for location, hit_type in seed_hits:
            usage_hit_rows.append({"seed": seed, "hit_type": hit_type, "location": location})

    for idx, seed, digest in candidate_rows:
        reasons: List[str] = []
        if seed in banned:
            reasons.append("development_seed")
        if seed in selected_values:
            reasons.append("duplicate_selected_seed")
        hit_count = len(hits.get(seed, []))
        if hit_count:
            reasons.append("preexisting_usage_hit")
        if reasons:
            rejected.append({
                "candidate_index": idx,
                "seed": seed,
                "sha256": digest,
                "usage_hit_count": hit_count,
                "reason": ";".join(reasons),
            })
            continue
        selected.append({
            "reservation_rank": len(selected) + 1,
            "candidate_index": idx,
            "seed": seed,
            "sha256": digest,
            "preexisting_usage_hit_count": 0,
            "status": "RESERVED_UNTOUCHED_FINAL_SEED",
        })
        selected_values.add(seed)
        if len(selected) == final_count:
            break

    if len(selected) != final_count:
        raise RuntimeError(f"Only {len(selected)} untouched candidates survived; required {final_count}")

    out_dir = root / cfg["outputs"]["root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / cfg["outputs"]["seed_manifest"]
    rejected_path = out_dir / cfg["outputs"]["rejected_candidates"]
    hits_path = out_dir / cfg["outputs"]["usage_hits"]
    decision_path = out_dir / cfg["outputs"]["reservation_decision"]

    write_csv(manifest_path, selected[0].keys(), selected)
    write_csv(
        rejected_path,
        ["candidate_index", "seed", "sha256", "usage_hit_count", "reason"],
        rejected,
    )
    write_csv(hits_path, ["seed", "hit_type", "location"], usage_hit_rows)

    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    decision = {
        "task": 64,
        "protocol_id": cfg["protocol_id"],
        "experiment_version": cfg["experiment_version"],
        "parent_tag": parent_tag,
        "parent_commit": parent_commit,
        "head_commit_at_reservation": head_commit,
        "final_seed_count": len(selected),
        "final_seeds": [row["seed"] for row in selected],
        "development_seeds_excluded": sorted(banned),
        "selection_uses_model_or_test_outcomes": False,
        "manual_seed_substitution_permitted": False,
        "files_considered": files_considered,
        "text_files_read": text_files_read,
        "unreadable_eligible_text_files": 0,
        "selected_seed_prior_usage_hits": 0,
        "reserved_test_paths_opened": 0,
        "binary_arrays_loaded": 0,
        "training_permitted": False,
        "new_shap_evaluations": 0,
        "llm_calls": 0,
        "seed_manifest_sha256": manifest_sha,
        "ready_for_independent_task64_audit": True,
    }
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("TASK 64 SEED RESERVATION COMPLETE")
    print("PARENT TAG:", parent_tag)
    print("PARENT COMMIT:", parent_commit)
    print("FINAL SEEDS:", ",".join(str(x) for x in decision["final_seeds"]))
    print("FINAL SEED COUNT:", len(selected))
    print("SELECTED PRIOR USAGE HITS: 0")
    print("UNREADABLE ELIGIBLE TEXT FILES: 0")
    print("RESERVED TEST PATHS OPENED: 0")
    print("TRAINING PERMITTED: False")
    print("READY FOR TASK64 AUDIT: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

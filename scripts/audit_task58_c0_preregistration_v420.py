from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

PARENT_TAG = "task57-c2-attribution-recovery-pass-v4192"
PARENT_COMMIT_SHORT = "20122c6"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="results/cic_iot_diad_task58_c0_preregistration_v420")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = (root / args.output_root).resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    cfg_path = root / "configs/task58_preregistration_v4200.json"
    md_path = root / "configs/TASK58_PREREGISTRATION_V420.md"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append((name, bool(passed), str(detail)))

    add("config_exists", cfg_path.is_file(), cfg_path)
    add("narrative_exists", md_path.is_file(), md_path)
    add("task_number", cfg.get("task") == 58, cfg.get("task"))
    add("version", cfg.get("experiment_version") == "4.20.0", cfg.get("experiment_version"))
    add("parent_tag", cfg.get("parent_tag") == PARENT_TAG, cfg.get("parent_tag"))
    add("parent_commit", cfg.get("parent_commit_short") == PARENT_COMMIT_SHORT, cfg.get("parent_commit_short"))
    add("outcomes_uninspected", cfg.get("outcomes_inspected_before_freeze") is False, cfg.get("outcomes_inspected_before_freeze"))
    add("no_training", cfg.get("training_permitted") is False, cfg.get("training_permitted"))
    add("no_new_shap", cfg.get("new_shap_evaluations_permitted") is False, cfg.get("new_shap_evaluations_permitted"))
    add("reserved_closed", cfg.get("reserved_test_arrays_permitted") is False, cfg.get("reserved_test_arrays_permitted"))
    plan = cfg.get("figure_plan", [])
    add("five_figure_families", len(plan) == 5, len(plan))
    add("unique_figure_ids", len({x.get("id") for x in plan}) == 5, [x.get("id") for x in plan])
    add("unique_stems", len({x.get("stem") for x in plan}) == 5, [x.get("stem") for x in plan])
    anti = cfg.get("anti_cherry_picking", {})
    add("all_cohort_rows", anti.get("cohort_figures_use_all_rows") is True, anti)
    add("frozen_feature_ranking", anti.get("feature_figure_uses_frozen_ranking") is True, anti)
    add("deterministic_case", anti.get("case_selection_is_deterministic") is True, anti)
    add("no_manual_substitution", anti.get("manual_case_substitution_permitted") is False, anti)
    add("formats", cfg.get("formats") == ["png", "pdf"], cfg.get("formats"))
    add("dpi", cfg.get("png_dpi") == 300, cfg.get("png_dpi"))
    expected = cfg.get("minimum_expected_outputs", {})
    for key, value in {"figure_stems": 5, "figure_files": 10, "triplets": 12, "seeds": 4, "families": 3, "features": 69}.items():
        add(f"expected_{key}", expected.get(key) == value, expected.get(key))
    limitation = str(cfg.get("state_limitation", ""))
    add("rejected_limitation", "Rejected" in limitation and "unavailable" in limitation, limitation)
    add("oracle_limitation", "oracle clean" in limitation and "unavailable" in limitation, limitation)
    add("clean_reference_qualified", "round 4 preattack" in limitation, limitation)
    try:
        tag_commit = subprocess.check_output(["git", "rev-list", "-n", "1", PARENT_TAG], cwd=root, text=True).strip()
        add("parent_tag_resolves", tag_commit.startswith(PARENT_COMMIT_SHORT), tag_commit)
    except Exception as exc:
        add("parent_tag_resolves", False, exc)

    with (tables / "task58c0_preregistration_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check", "passed", "detail"])
        writer.writerows(checks)
    with (tables / "task58c0_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        for path in [cfg_path, md_path]:
            writer.writerow([path.relative_to(root).as_posix(), path.stat().st_size, sha256(path)])
    passed = sum(x[1] for x in checks)
    decision = {
        "experiment_version": "4.20.0",
        "checks_passed": passed,
        "checks_total": len(checks),
        "all_checks_passed": passed == len(checks),
        "outcomes_inspected": False,
        "training_permitted": False,
        "new_shap_evaluations_permitted": False,
        "reserved_test_arrays_materialized": False,
        "ready_for_c1_publication_figures": passed == len(checks),
    }
    (output / "task58c0_preregistration_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print("===== TASK 58 C0 FIGURE PREREGISTRATION AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("OUTCOMES INSPECTED: False")
    print("TRAINING PERMITTED: False")
    print("NEW SHAP EVALUATIONS PERMITTED: False")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("READY FOR C1 PUBLICATION FIGURES:", passed == len(checks))
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

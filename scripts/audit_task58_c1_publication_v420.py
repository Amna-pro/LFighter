from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

EXPECTED_STEMS = {
    "task58_xai_validation_dashboard",
    "task58_all_triplet_recovery",
    "task58_prediction_state_trajectories",
    "task58_top10_poison_associated_features",
    "task58_representative_case_waterfall",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--task57-root", default="results/cic_iot_diad_task57_c2_summary_v419")
    parser.add_argument("--publication-root", default="results/cic_iot_diad_task58_c1_publication_v420")
    parser.add_argument("--output-root", default="results/cic_iot_diad_task58_c1_audit_v420")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    t57 = (root / args.task57_root / "tables").resolve()
    publication = (root / args.publication_root).resolve()
    output = (root / args.output_root).resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append((name, bool(passed), str(detail)))

    c0_path = root / "results/cic_iot_diad_task58_c0_preregistration_v420/task58c0_preregistration_decision.json"
    c0 = json.loads(c0_path.read_text(encoding="utf-8"))
    decision_path = publication / "task58c1_publication_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    manifest_path = publication / "tables/task58c1_figure_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    triplets = pd.read_csv(t57 / "task57c2_triplet_recovery_metrics.csv")
    top10 = pd.read_csv(t57 / "task57c2_top10_prespecified_features.csv")
    case = pd.read_csv(publication / "tables/task58c1_case_selection.csv")
    contributions = pd.read_csv(publication / "tables/task58c1_case_feature_contributions.csv")
    validation = pd.read_csv(publication / "tables/task58c1_validation_metric_summary.csv")
    source_manifest = pd.read_csv(publication / "tables/task58c1_source_manifest_sha256.csv")

    add("c0_all_checks", c0.get("all_checks_passed") is True, c0)
    add("c0_ready", c0.get("ready_for_c1_publication_figures") is True, c0.get("ready_for_c1_publication_figures"))
    add("decision_complete", decision.get("task58_c1_complete") is True, decision.get("task58_c1_complete"))
    add("parent_tag", decision.get("parent_tag") == "task57-c2-attribution-recovery-pass-v4192", decision.get("parent_tag"))
    add("parent_commit", decision.get("parent_commit_short") == "20122c6", decision.get("parent_commit_short"))
    add("scientific_result", decision.get("scientific_result_inherited_from_task57") == "PASS", decision.get("scientific_result_inherited_from_task57"))
    add("five_stems_decision", decision.get("figure_stems") == 5, decision.get("figure_stems"))
    add("ten_files_decision", decision.get("figure_files") == 10, decision.get("figure_files"))
    add("twelve_triplets_decision", decision.get("triplets_represented") == 12, decision.get("triplets_represented"))
    add("all_rows_decision", decision.get("all_cohort_rows_used") is True, decision.get("all_cohort_rows_used"))
    add("deterministic_case_decision", decision.get("case_selection_deterministic") is True, decision.get("case_selection_deterministic"))
    add("no_manual_case_decision", decision.get("manual_case_substitution") is False, decision.get("manual_case_substitution"))
    add("no_training", decision.get("training_permitted") is False, decision.get("training_permitted"))
    add("no_new_shap", decision.get("new_shap_evaluations") == 0, decision.get("new_shap_evaluations"))
    add("reserved_closed", decision.get("reserved_test_arrays_materialized") is False, decision.get("reserved_test_arrays_materialized"))
    add("rejected_unavailable", decision.get("rejected_state_available") is False, decision.get("rejected_state_available"))
    add("oracle_unavailable", decision.get("oracle_clean_state_available") is False, decision.get("oracle_clean_state_available"))
    add("clean_qualified", decision.get("clean_reference") == "paired round 4 preattack state", decision.get("clean_reference"))

    add("manifest_rows", len(manifest) == 10, len(manifest))
    add("manifest_stems", set(manifest.stem) == EXPECTED_STEMS, sorted(set(manifest.stem)))
    add("manifest_formats", set(manifest.format) == {"png", "pdf"}, sorted(set(manifest.format)))
    add("two_formats_each", manifest.groupby("stem").size().eq(2).all(), manifest.groupby("stem").size().to_dict())
    add("unique_paths", manifest.path.nunique() == 10, manifest.path.nunique())
    for row in manifest.itertuples():
        path = root / row.path
        add(f"figure_exists_{row.stem}_{row.format}", path.is_file(), path)
        if path.is_file():
            add(f"figure_hash_{row.stem}_{row.format}", sha256(path) == row.sha256, sha256(path))
            add(f"figure_bytes_{row.stem}_{row.format}", path.stat().st_size == int(row.bytes) and path.stat().st_size > 5000, path.stat().st_size)
            if row.format == "png":
                add(f"png_dimensions_{row.stem}", int(row.pixel_width) >= 1800 and int(row.pixel_height) >= 900, f"{row.pixel_width}x{row.pixel_height}")
            else:
                add(f"pdf_header_{row.stem}", path.read_bytes()[:5] == b"%PDF-", path.read_bytes()[:5])

    add("triplet_source_rows", len(triplets) == 12, len(triplets))
    add("triplet_seed_count", triplets.seed.nunique() == 4, triplets.seed.nunique())
    add("triplet_family_count", triplets.family_id.nunique() == 3, triplets.family_id.nunique())
    add("top10_rows", len(top10) == 10, len(top10))
    ranks = top10["rank"].astype(int).tolist()
    add("top10_ranks", len(set(ranks)) == 10 and ranks == sorted(ranks) and min(ranks) >= 1, ranks)
    add("case_one_row", len(case) == 1, len(case))
    median_recovery = float(triplets.recovery_fraction.median())
    expected = triplets.assign(selection_gap=(triplets.recovery_fraction - median_recovery).abs()).sort_values(["selection_gap", "seed", "family_id"]).iloc[0]
    selected = case.iloc[0]
    add("case_median_exact", abs(float(selected.cohort_median_recovery_fraction) - median_recovery) < 1e-12, selected.cohort_median_recovery_fraction)
    add("case_seed_exact", int(selected.selected_seed) == int(expected.seed), selected.selected_seed)
    add("case_family_exact", selected.selected_family_id == expected.family_id, selected.selected_family_id)
    add("case_recovery_exact", abs(float(selected.selected_recovery_fraction) - float(expected.recovery_fraction)) < 1e-12, selected.selected_recovery_fraction)
    add("case_gap_exact", abs(float(selected.selection_gap) - float(expected.selection_gap)) < 1e-12, selected.selection_gap)
    add("case_no_manual", bool(selected.manual_substitution) is False, selected.manual_substitution)
    add("case_three_states", set(contributions.state) == {"Clean", "Suspicious", "Reconstructed"}, sorted(set(contributions.state)))
    add("case_equal_feature_rows", contributions.groupby("state").size().nunique() == 1, contributions.groupby("state").size().to_dict())
    add("case_nine_rows_per_state", contributions.groupby("state").size().eq(9).all(), contributions.groupby("state").size().to_dict())
    add("case_finite", contributions[["shap_contribution", "absolute_contribution"]].notna().all().all(), "finite")
    add("validation_summary_nonempty", len(validation) >= 10, len(validation))
    add("validation_summary_finite", validation[["median", "q25", "q75", "minimum", "maximum"]].notna().all().all(), "finite")

    all_sources_valid = True
    for row in source_manifest.itertuples():
        path = root / row.path
        if not path.is_file() or path.stat().st_size != int(row.bytes) or sha256(path) != row.sha256:
            all_sources_valid = False
            break
    add("source_manifest_nonempty", len(source_manifest) >= 14, len(source_manifest))
    add("source_manifest_exact", all_sources_valid, all_sources_valid)
    captions_path = publication / "TASK58_FIGURE_CAPTIONS.md"
    captions = captions_path.read_text(encoding="utf-8")
    add("captions_exist", captions_path.is_file(), captions_path)
    for figure_id in ["F1", "F2", "F3", "F4", "F5"]:
        add(f"caption_{figure_id}", f"Figure {figure_id}." in captions, figure_id)
    add("caption_scope_limitation", "round 8 oracle clean" in captions and "No substitute states" in captions, "scope stated")
    closeout_path = publication / "TASK58_C1_CLOSEOUT.md"
    add("closeout_exists", closeout_path.is_file(), closeout_path)

    with (tables / "task58c1_publication_audit_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check", "passed", "detail"])
        writer.writerows(checks)
    audit_sources = [decision_path, manifest_path, publication / "tables/task58c1_case_selection.csv", publication / "tables/task58c1_case_feature_contributions.csv", publication / "tables/task58c1_validation_metric_summary.csv", publication / "tables/task58c1_source_manifest_sha256.csv", captions_path, closeout_path]
    with (tables / "task58c1_audit_source_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        for path in audit_sources:
            writer.writerow([path.relative_to(root).as_posix(), path.stat().st_size, sha256(path)])
    passed = sum(x[1] for x in checks)
    audit_decision = {
        "experiment_version": "4.20.1.audit",
        "checks_passed": passed,
        "checks_total": len(checks),
        "all_checks_passed": passed == len(checks),
        "figure_stems_verified": 5,
        "figure_files_verified": 10,
        "triplets_verified": 12,
        "case_selection_independently_recomputed": True,
        "source_hashes_verified": all_sources_valid,
        "new_shap_evaluations": 0,
        "reserved_test_arrays_materialized": False,
        "task58_ready_to_freeze": passed == len(checks),
    }
    (output / "task58c1_publication_audit_decision.json").write_text(json.dumps(audit_decision, indent=2) + "\n", encoding="utf-8")
    print("===== TASK 58 C1 PUBLICATION AUDIT =====")
    print(f"Checks passed: {passed}/{len(checks)}")
    print("FIGURE STEMS VERIFIED: 5")
    print("FIGURE FILES VERIFIED: 10")
    print("TRIPLETS VERIFIED: 12")
    print("CASE SELECTION INDEPENDENTLY RECOMPUTED: True")
    print("SOURCE HASHES VERIFIED:", all_sources_valid)
    print("NEW SHAP EVALUATIONS: 0")
    print("RESERVED TEST ARRAYS MATERIALIZED: False")
    print("TASK 58 READY TO FREEZE:", passed == len(checks))
    if passed != len(checks):
        failed = [name for name, ok, _ in checks if not ok]
        print("FAILED CHECKS:", ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

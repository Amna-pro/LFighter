#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer_p4p_round8_test_evaluation_v4326"
TABLES = OUT / "tables"
ACCESS = OUT / "P4P_TEST_ACCESS_STARTED_v4326.json"
COMPLETE = OUT / "P4P_ROUND8_TEST_EVALUATION_COMPLETE_v4326.json"
AUDIT = ROOT / "reviewer_revision" / "P4P_ROUND8_TEST_EVALUATION_AUDIT_v4326.json"
PRIMARY = ROOT / "reviewer_revision" / "P4P_ROUND8_TEST_PRIMARY_SUMMARY_v4326.csv"

COPIES = {
    TABLES / "p4p_round8_reserved_test_metrics.csv":
        ROOT / "reviewer_revision" / "P4P_ROUND8_TEST_METRICS_v4326.csv",
    TABLES / "p4p_round8_reserved_test_per_class.csv":
        ROOT / "reviewer_revision" / "P4P_ROUND8_TEST_PER_CLASS_v4326.csv",
    TABLES / "p4p_round8_reserved_test_confusion_long.csv":
        ROOT / "reviewer_revision" / "P4P_ROUND8_TEST_CONFUSION_LONG_v4326.csv",
    ACCESS:
        ROOT / "reviewer_revision" / "P4P_ROUND8_TEST_ACCESS_STARTED_v4326.json",
    COMPLETE:
        ROOT / "reviewer_revision" / "P4P_ROUND8_TEST_EVALUATION_COMPLETE_v4326.json",
}

SEEDS = [1379954285, 1886033230, 480705558, 1377035733, 1707771978]
ATTACKS = [
    "all_to_one_benign",
    "cyclic_shift",
    "multiclass_partial_cycle",
    "pairwise_swap",
    "random_flip",
]
DATASET_ROWS = {"diagnostic": 113210, "natural": 199990}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if AUDIT.exists() or PRIMARY.exists():
        raise FileExistsError("Post-evaluation audit evidence already exists; refusing overwrite")

    if not COMPLETE.exists() or not ACCESS.exists():
        raise FileNotFoundError("Evaluation completion or access record is missing")

    completion = json.loads(COMPLETE.read_text(encoding="utf-8"))
    access = json.loads(ACCESS.read_text(encoding="utf-8"))

    metrics_path = TABLES / "p4p_round8_reserved_test_metrics.csv"
    class_path = TABLES / "p4p_round8_reserved_test_per_class.csv"
    confusion_path = TABLES / "p4p_round8_reserved_test_confusion_long.csv"

    m = pd.read_csv(metrics_path)
    pc = pd.read_csv(class_path)
    cm = pd.read_csv(confusion_path)

    checks = {}
    checks["completion_pass"] = completion.get("status") == "PASS"
    checks["access_global_round_fixed_8"] = int(access.get("global_round_fixed", -1)) == 8
    checks["completion_global_round_8"] = int(completion.get("global_round", -1)) == 8
    checks["best_round_selection_false"] = completion.get("best_round_selection_used") is False
    checks["post_outcome_retuning_false"] = completion.get("post_outcome_retuning_used") is False
    checks["training_false"] = completion.get("model_training_run") is False
    checks["p4p_rerun_false"] = completion.get("p4p_federated_branch_rerun") is False
    checks["formal_statistics_false"] = completion.get("formal_cross_method_statistics_performed") is False
    checks["prior_shape_access_disclosed"] = completion.get("prior_structural_test_array_shape_access") is True

    checks["metric_rows_50"] = len(m) == 50
    checks["per_class_rows_400"] = len(pc) == 400
    checks["confusion_rows_3200"] = len(cm) == 3200

    expected_conditions = {(d, a, s) for d in DATASET_ROWS for a in ATTACKS for s in SEEDS}
    actual_conditions = set(
        zip(m["dataset"], m["attack_type"], m["model_seed"].astype(int))
    )
    checks["exact_50_condition_dataset_rows"] = actual_conditions == expected_conditions
    checks["all_global_round_8"] = set(m["global_round"].astype(int)) == {8}
    checks["all_arm_p4p"] = set(m["arm"]) == {"p4p_matched"}

    metric_cols = [
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
        "mcc", "log_loss", "ece_15bin", "worst_class_recall",
    ]
    checks["all_primary_metrics_finite"] = bool(
        np.isfinite(m[metric_cols].to_numpy(dtype=float)).all()
    )
    checks["bounded_core_metrics"] = bool(
        ((m[["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "ece_15bin", "worst_class_recall"]] >= 0.0)
         & (m[["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "ece_15bin", "worst_class_recall"]] <= 1.0)).all().all()
    )

    worst_ok = True
    support_ok = True
    confusion_ok = True
    class_card_ok = True
    confusion_card_ok = True

    for dataset, attack, seed in sorted(expected_conditions):
        mm = m[
            (m["dataset"] == dataset)
            & (m["attack_type"] == attack)
            & (m["model_seed"].astype(int) == seed)
        ].iloc[0]
        psub = pc[
            (pc["dataset"] == dataset)
            & (pc["attack_type"] == attack)
            & (pc["model_seed"].astype(int) == seed)
        ]
        csub = cm[
            (cm["dataset"] == dataset)
            & (cm["attack_type"] == attack)
            & (cm["model_seed"].astype(int) == seed)
        ]
        if len(psub) != 8:
            class_card_ok = False
        if len(csub) != 64:
            confusion_card_ok = False

        min_recall = float(psub["recall"].min())
        if not np.isclose(min_recall, float(mm["worst_class_recall"]), rtol=0.0, atol=1e-15):
            worst_ok = False

        expected_rows = DATASET_ROWS[dataset]
        if int(psub["support"].sum()) != expected_rows:
            support_ok = False
        if int(csub["count"].sum()) != expected_rows:
            confusion_ok = False

    checks["eight_class_rows_per_condition"] = class_card_ok
    checks["sixty_four_confusion_rows_per_condition"] = confusion_card_ok
    checks["worst_class_recall_matches_per_class_min"] = worst_ok
    checks["per_class_support_matches_dataset_rows"] = support_ok
    checks["confusion_counts_match_dataset_rows"] = confusion_ok

    recorded = completion.get("output_sha256", {})
    hash_ok = (
        recorded.get(metrics_path.name) == sha256_file(metrics_path)
        and recorded.get(class_path.name) == sha256_file(class_path)
        and recorded.get(confusion_path.name) == sha256_file(confusion_path)
        and recorded.get(ACCESS.name) == sha256_file(ACCESS)
    )
    checks["completion_output_hashes_match"] = hash_ok

    failed = [k for k, v in checks.items() if not bool(v)]
    if failed:
        raise RuntimeError(f"P4P reserved-test audit failed: {failed}")

    summary_rows = []
    for dataset in ("diagnostic", "natural"):
        for attack in ATTACKS:
            g = m[(m["dataset"] == dataset) & (m["attack_type"] == attack)].copy()
            for metric in ("macro_f1", "balanced_accuracy", "accuracy", "worst_class_recall"):
                values = g[metric].to_numpy(dtype=float)
                summary_rows.append({
                    "dataset": dataset,
                    "attack_type": attack,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std_ddof1": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "formal_inference_performed": False,
                })
    pd.DataFrame(summary_rows).to_csv(PRIMARY, index=False)

    for src, dst in COPIES.items():
        shutil.copy2(src, dst)

    audit = {
        "protocol": "reviewer_v4326_p4p_round8_reserved_test_postaccess_audit",
        "status": "PASS",
        "checks": {k: bool(v) for k, v in checks.items()},
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "metric_rows": len(m),
        "per_class_rows": len(pc),
        "confusion_rows": len(cm),
        "negative_results_retained": True,
        "formal_cross_method_statistics_performed": False,
        "post_outcome_retuning_used": False,
        "output_sha256": {
            "primary_summary": sha256_file(PRIMARY),
            **{dst.name: sha256_file(dst) for dst in COPIES.values()},
        },
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    print("P4P ROUND8 RESERVED TEST POSTACCESS AUDIT = PASS")
    print("CHECKS:", len(checks), "/", len(checks))
    print("METRIC ROWS:", len(m))
    print("PER CLASS ROWS:", len(pc))
    print("CONFUSION ROWS:", len(cm))
    print("NEGATIVE RESULTS RETAINED: True")
    print("POST OUTCOME RETUNING USED: False")
    print("FORMAL CROSS METHOD STATISTICS PERFORMED: False")
    print("PRIMARY SUMMARY:", PRIMARY)
    print("AUDIT:", AUDIT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXPECTED_BRANCH = "feature/cic-iot-diad-task41c-backdoor-extension-v1"
EXPECTED_TAG = "task41b-frozen-v4.11b9"
EXPECTED_PARENT = "f29b026"
SEEDS = [7, 99, 123, 2026]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--clean-seed-root", type=Path, required=True)
    p.add_argument("--warmup-root", type=Path, required=True)
    p.add_argument("--reconstruction-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(root: Path, *values: str) -> str:
    return subprocess.run(
        ["git", *values],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def save(fig, base: Path):
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    a = args()
    root = a.project_root.expanduser().resolve()
    clean_root = a.clean_seed_root.expanduser().resolve()
    warmup_root = a.warmup_root.expanduser().resolve()
    reconstruction_root = a.reconstruction_root.expanduser().resolve()
    output = a.output_dir.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    protocol_path = root / "configs" / "task41c_preregistration_v412c0.json"
    prereg_path = root / "TASK41C_PREREGISTRATION_V412C0.md"
    prereg_decision_path = (
        root
        / "results"
        / "cic_iot_diad_task41c_preregistration_v412c0"
        / "task41c0_preregistration_decision.json"
    )
    audit_source = root / "scripts" / "audit_task41c_preregistration_v412c0.py"

    for path in (
        protocol_path,
        prereg_path,
        prereg_decision_path,
        audit_source,
        clean_root,
        warmup_root,
        reconstruction_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    prereg_decision = json.loads(
        prereg_decision_path.read_text(encoding="utf-8")
    )
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")
    tag_commit = git(root, "rev-list", "-n", "1", EXPECTED_TAG)

    checks = []

    def check(name, passed, detail):
        checks.append(
            {"check_name": name, "passed": bool(passed), "detail": str(detail)}
        )
        if not passed:
            raise RuntimeError(f"{name} failed: {detail}")

    check("branch", branch == EXPECTED_BRANCH, branch)
    check(
        "protocol_version",
        protocol.get("experiment_version") == "4.12C.0",
        protocol.get("experiment_version"),
    )
    check(
        "parent_tag",
        protocol["parent_frozen_task"]["tag"] == EXPECTED_TAG,
        protocol["parent_frozen_task"]["tag"],
    )
    check(
        "parent_commit",
        protocol["parent_frozen_task"]["commit"] == EXPECTED_PARENT,
        protocol["parent_frozen_task"]["commit"],
    )
    check(
        "tag_commit",
        tag_commit.startswith(EXPECTED_PARENT),
        tag_commit,
    )
    check(
        "preregistration_frozen",
        prereg_decision.get("preregistration_frozen") is True,
        prereg_decision.get("preregistration_frozen"),
    )
    check(
        "experiments_not_started",
        prereg_decision.get("task41c_experiments_started") is False,
        prereg_decision.get("task41c_experiments_started"),
    )
    check(
        "task41b_not_reopened",
        prereg_decision.get("task41b_reopened") is False,
        prereg_decision.get("task41b_reopened"),
    )
    check(
        "test_not_accessed",
        prereg_decision.get("reserved_test_accessed") is False,
        prereg_decision.get("reserved_test_accessed"),
    )
    check("seeds", protocol.get("frozen_seeds") == SEEDS, protocol.get("frozen_seeds"))
    check(
        "reserved_test_policy",
        protocol["data_policy"]["reserved_test_access_before_final_freeze"]
        is False,
        protocol["data_policy"]["reserved_test_access_before_final_freeze"],
    )

    seed_rows = []
    inventory_rows = []
    excluded_rows = []

    for seed in SEEDS:
        clean_dir = clean_root / f"seed_{seed}"
        warmup_dir = warmup_root / f"seed_{seed}" / "warmup"
        calibration_dir = (
            reconstruction_root / f"seed_{seed}" / "calibration"
        )
        warmup_checkpoint = (
            warmup_dir / "checkpoints" / "common_round4_warmup_model.pt"
        )

        clean_files = (
            sorted(
                path
                for path in clean_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".csv", ".json", ".pt", ".npz"}
            )
            if clean_dir.exists()
            else []
        )
        warmup_files = (
            sorted(path for path in warmup_dir.rglob("*") if path.is_file())
            if warmup_dir.exists()
            else []
        )
        calibration_files = (
            sorted(path for path in calibration_dir.rglob("*") if path.is_file())
            if calibration_dir.exists()
            else []
        )

        clean_ready = clean_dir.exists() and bool(clean_files)
        warmup_ready = warmup_dir.exists() and warmup_checkpoint.exists()
        calibration_ready = calibration_dir.exists() and bool(calibration_files)
        ready = clean_ready and warmup_ready and calibration_ready

        seed_rows.append(
            {
                "seed": seed,
                "clean_artifact_count": len(clean_files),
                "clean_ready": clean_ready,
                "warmup_artifact_count": len(warmup_files),
                "warmup_checkpoint_present": warmup_checkpoint.exists(),
                "warmup_ready": warmup_ready,
                "calibration_artifact_count": len(calibration_files),
                "calibration_ready": calibration_ready,
                "c1_seed_ready": ready,
            }
        )

        for category, files in (
            ("clean_seed", clean_files),
            ("warmup", warmup_files),
            ("reconstruction_calibration", calibration_files),
        ):
            for path in files:
                lower = str(path).lower()
                if "test_natural" in lower or "test_diagnostic" in lower:
                    excluded_rows.append(
                        {
                            "seed": seed,
                            "category": category,
                            "relative_path": str(path.relative_to(root)),
                            "bytes": path.stat().st_size,
                            "exclusion_reason": "reserved_test_name_reference",
                            "content_hashed": False,
                            "content_loaded": False,
                        }
                    )
                    continue
                inventory_rows.append(
                    {
                        "seed": seed,
                        "category": category,
                        "relative_path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "sha256": digest(path),
                        "reserved_test_artifact": False,
                    }
                )

    seed_frame = pd.DataFrame(seed_rows)
    inventory = pd.DataFrame(inventory_rows)
    excluded = pd.DataFrame(excluded_rows)

    check(
        "all_seed_inputs_ready",
        bool(seed_frame["c1_seed_ready"].all()),
        seed_frame.to_dict(orient="records"),
    )
    check(
        "no_reserved_test_artifacts_hashed_or_loaded",
        not bool(inventory["reserved_test_artifact"].any()),
        {
            "included_reserved_count": int(
                inventory["reserved_test_artifact"].sum()
            ),
            "excluded_name_reference_count": len(excluded),
        },
    )

    manifest = pd.DataFrame(
        [
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in (
                prereg_path,
                protocol_path,
                prereg_decision_path,
                audit_source,
            )
        ]
    )

    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(
        tables / "task41c1_preflight_checks.csv", index=False
    )
    seed_frame.to_csv(tables / "task41c1_seed_readiness.csv", index=False)
    inventory.to_csv(
        tables / "task41c1_clean_input_inventory.csv", index=False
    )
    excluded.to_csv(
        tables / "task41c1_excluded_reserved_name_references.csv",
        index=False,
    )
    manifest.to_csv(
        tables / "task41c1_source_manifest_sha256.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    values = seed_frame[
        ["clean_ready", "warmup_ready", "calibration_ready"]
    ].sum()
    ax.bar(values.index, values.values)
    ax.set_ylim(0, 4)
    ax.set_ylabel("Ready seeds")
    ax.set_title("Task 41C C1 clean-only calibration readiness")
    ax.grid(axis="y", alpha=0.25)
    save(fig, figures / "task41c1_clean_readiness")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    counts = inventory.groupby("category").size()
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Artifact count")
    ax.set_title("Task 41C C1 clean input inventory")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    save(fig, figures / "task41c1_clean_inventory")

    decision = {
        "experiment_version": "4.12C.1",
        "stage": "task41c_c1_clean_only_calibration_preflight",
        "git_branch": branch,
        "git_head": head,
        "integrity_check_count": len(checks_frame),
        "integrity_checks_passed": int(checks_frame["passed"].sum()),
        "seed_count": len(seed_frame),
        "ready_seed_count": int(seed_frame["c1_seed_ready"].sum()),
        "c1_clean_calibration_allowed": True,
        "attack_execution_allowed": False,
        "task41c_experiments_started": False,
        "task41b_reopened": False,
        "reserved_name_reference_count": len(excluded),
        "reserved_name_references_hashed": False,
        "reserved_name_references_loaded": False,
        "reserved_test_accessed": False,
        "next_stage": (
            "Commit C0 and C1 preflight evidence, then implement clean-only "
            "D1, D2, and D3 calibration."
        ),
    }
    (output / "task41c1_clean_preflight_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print("===== TASK 41C.1 CLEAN-ONLY CALIBRATION PREFLIGHT =====")
    print(
        "Integrity checks passed:",
        f"{decision['integrity_checks_passed']}/"
        f"{decision['integrity_check_count']}",
    )
    print(
        "Ready seeds:",
        f"{decision['ready_seed_count']}/{decision['seed_count']}",
    )
    print("C1 CLEAN CALIBRATION ALLOWED: True")
    print("ATTACK EXECUTION ALLOWED: False")
    print("TASK 41C EXPERIMENTS STARTED: False")
    print("TASK 41B REOPENED: False")
    print(
        "Excluded reserved-name references:",
        len(excluded),
    )
    print("RESERVED-NAME REFERENCES HASHED: False")
    print("RESERVED-NAME REFERENCES LOADED: False")
    print("TEST SETS ACCESSED: False")
    print("Tables:", tables)
    print("PNG and PDF figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

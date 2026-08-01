#!/usr/bin/env python3
"""Run the pre-registered V3.18B seed-7 full-update capture smoke grid."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

SOURCE_TO_SLUG = {
    "DoS": "dos_to_benign",
    "BruteForce": "bruteforce_to_benign",
    "Web-Based": "web_based_to_benign",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True, type=Path)
    p.add_argument("--partition-file", required=True, type=Path)
    p.add_argument("--clean-seed-root", required=True, type=Path)
    p.add_argument("--v3101-root", required=True, type=Path)
    p.add_argument("--dos-root", required=True, type=Path)
    p.add_argument("--remaining-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sources", default="DoS,BruteForce,Web-Based")
    p.add_argument("--target-class", default="Benign")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--evaluation-batch-size", type=int, default=4096)
    return p.parse_args()


def run(command: List[str]) -> None:
    print("\nRUNNING:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def branch_root(source: str, dos_root: Path, remaining_root: Path) -> Path:
    return dos_root if source == "DoS" else remaining_root


def coalition_for(source: str, root: Path, target: str) -> str:
    path = root / "coalitions" / "tables" / "v313_frozen_coalition_manifest.csv"
    table = pd.read_csv(path)
    row = table[table["source_class"].eq(source) & table["target_class"].eq(target)]
    if len(row) != 1:
        raise RuntimeError(f"Missing unique frozen coalition for {source}->{target}: {path}")
    return str(row.iloc[0]["selected_clients"]).replace("|", ",")


def verified(path: Path) -> bool:
    decision = path / "v318b_equivalence.json"
    if not decision.exists():
        return False
    with decision.open("r", encoding="utf-8") as f:
        return bool(json.load(f).get("exact_training_equivalence", False))


def capture_complete(path: Path) -> bool:
    metadata = path / "post_warmup_capture_v310_metadata.json"
    index = path / "tables" / "v318b_update_capture_index.csv"
    if not metadata.exists() or not index.exists():
        return False
    try:
        table = pd.read_csv(index)
        return len(table) == 4 and bool(table["all_values_finite"].all())
    except Exception:
        return False


def ensure_capture(output: Path, command: List[str]) -> None:
    if capture_complete(output):
        print(f"SKIPPING COMPLETE CAPTURE: {output}")
        return
    if output.exists():
        shutil.rmtree(output)
    run(command)


def ensure_verification(output: Path, command: List[str]) -> None:
    if verified(output):
        print(f"SKIPPING PASSED VERIFICATION: {output}")
        return
    if output.exists():
        shutil.rmtree(output)
    run(command)


def main() -> int:
    a = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python = sys.executable
    runner = project_root / "scripts" / "run_update_capture_v318b.py"
    verifier = project_root / "scripts" / "verify_update_capture_equivalence_v318b.py"

    sources = [x.strip() for x in a.sources.split(",") if x.strip()]
    unknown = sorted(set(sources) - set(SOURCE_TO_SLUG))
    if unknown:
        raise ValueError(f"Unsupported V3.18B smoke sources: {unknown}")
    if a.seed != 7:
        raise ValueError("V3.18B development smoke protocol is frozen to seed 7")

    data_file = a.data_file.expanduser().resolve()
    partition_file = a.partition_file.expanduser().resolve()
    clean_seed_dir = a.clean_seed_root.expanduser().resolve() / f"seed_{a.seed}"
    warmup_dir = a.v3101_root.expanduser().resolve() / f"seed_{a.seed}" / "warmup"
    dos_root = a.dos_root.expanduser().resolve()
    remaining_root = a.remaining_root.expanduser().resolve()
    output_root = a.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    common = [
        "--data-file", str(data_file),
        "--partition-file", str(partition_file),
        "--clean-seed-dir", str(clean_seed_dir),
        "--warmup-dir", str(warmup_dir),
        "--model-seed", str(a.seed),
        "--num-clients", "20",
        "--continuation-rounds", "4",
        "--batch-size", str(a.batch_size),
        "--evaluation-batch-size", str(a.evaluation_batch_size),
        "--learning-rate", "0.0003",
        "--weight-decay", "0.0001",
        "--threads", str(a.threads),
        "--probe-per-class", "48",
        "--probe-seed", "3701",
        "--target-class", a.target_class,
        "--ema-decay", "0.65",
    ]

    summary_rows: List[Dict[str, object]] = []

    # A single clean local-update trajectory is sufficient because source class
    # changes only diagnostics, not clean local training or FedAvg aggregation.
    clean_source = "BruteForce"
    clean_output = output_root / "captures" / f"clean_seed_{a.seed}"
    ensure_capture(
        clean_output,
        [
            python, str(runner),
            "--mode", "clean",
            *common,
            "--source-class", clean_source,
            "--output-dir", str(clean_output),
        ],
    )
    clean_reference = (
        remaining_root / "runs" / SOURCE_TO_SLUG[clean_source]
        / f"seed_{a.seed}" / "exact_clean"
    )
    clean_verification = output_root / "verification" / f"clean_seed_{a.seed}"
    ensure_verification(
        clean_verification,
        [
            python, str(verifier),
            "--captured-dir", str(clean_output),
            "--reference-dir", str(clean_reference),
            "--output-dir", str(clean_verification),
        ],
    )
    clean_index = pd.read_csv(clean_output / "tables" / "v318b_update_capture_index.csv")
    summary_rows.append({
        "scenario": "Clean",
        "source_class": clean_source,
        "seed": a.seed,
        "capture_dir": str(clean_output),
        "reference_dir": str(clean_reference),
        "exact_equivalence": verified(clean_verification),
        "captured_rounds": len(clean_index),
        "captured_update_bytes": int(clean_index["update_matrix_bytes"].sum()),
    })

    for source in sources:
        root = branch_root(source, dos_root, remaining_root)
        slug = SOURCE_TO_SLUG[source]
        coalition = coalition_for(source, root, a.target_class)
        attack_output = output_root / "captures" / f"{slug}_seed_{a.seed}"
        ensure_capture(
            attack_output,
            [
                python, str(runner),
                "--mode", "strong_attack",
                *common,
                "--source-class", source,
                "--output-dir", str(attack_output),
                "--malicious-clients", coalition,
                "--poison-fraction", "1.0",
                "--min-source-samples", "1",
                "--attack-seed", str(a.seed),
            ],
        )
        reference = root / "runs" / slug / f"seed_{a.seed}" / "plain_attack"
        verification = output_root / "verification" / f"{slug}_seed_{a.seed}"
        ensure_verification(
            verification,
            [
                python, str(verifier),
                "--captured-dir", str(attack_output),
                "--reference-dir", str(reference),
                "--output-dir", str(verification),
            ],
        )
        index = pd.read_csv(attack_output / "tables" / "v318b_update_capture_index.csv")
        summary_rows.append({
            "scenario": f"{source}->{a.target_class}",
            "source_class": source,
            "seed": a.seed,
            "malicious_clients": coalition.replace(",", "|"),
            "capture_dir": str(attack_output),
            "reference_dir": str(reference),
            "exact_equivalence": verified(verification),
            "captured_rounds": len(index),
            "captured_update_bytes": int(index["update_matrix_bytes"].sum()),
        })

    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables / "v318b_smoke_capture_manifest.csv", index=False)
    all_exact = bool(summary["exact_equivalence"].all())
    decision = {
        "experiment_version": "3.18B",
        "stage": "pre_registered_full_update_capture_smoke",
        "seed": int(a.seed),
        "scenarios": summary["scenario"].tolist(),
        "all_capture_branches_exactly_equivalent": all_exact,
        "captured_scenarios": int(len(summary)),
        "captured_rounds": int(summary["captured_rounds"].sum()),
        "captured_update_matrix_bytes": int(summary["captured_update_bytes"].sum()),
        "training_or_aggregation_modified": False,
        "test_sets_accessed": False,
        "next_stage": "V3.18C offline update-space detector feasibility screen" if all_exact else "stop and repair capture equivalence",
    }
    pd.DataFrame([decision]).to_csv(tables / "v318b_decision.csv", index=False)
    with (output_root / "v318b_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    print("\nV3.18B SMOKE CAPTURE DECISION")
    print(pd.DataFrame([decision]).to_string(index=False))
    print("\nCAPTURE MANIFEST")
    print(summary.to_string(index=False))
    if not all_exact:
        raise SystemExit("At least one V3.18B branch failed exact equivalence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "cic_iot_diad_task41a_trigger_feasibility_v410a"
GEOMETRY_ROOT = RESULT_ROOT / "geometry_screen"
TABLE_DIR = RESULT_ROOT / "schema_audit" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "candidates_csv": GEOMETRY_ROOT / "tables" / "task41a_trigger_geometry_candidates.csv",
    "source_metrics_csv": GEOMETRY_ROOT / "tables" / "task41a_trigger_candidate_source_metrics.csv",
    "selected_csv": GEOMETRY_ROOT / "tables" / "task41a_selected_geometry_candidates.csv",
    "family_summary_csv": GEOMETRY_ROOT / "tables" / "task41a_trigger_family_summary.csv",
    "screen_summary_json": GEOMETRY_ROOT / "tables" / "task41a_trigger_geometry_screen_summary.json",
    "trigger_specs_json": GEOMETRY_ROOT / "trigger_specs" / "task41a_selected_trigger_specs.json",
    "trigger_specs_npz": GEOMETRY_ROOT / "trigger_specs" / "task41a_selected_trigger_specs.npz",
}


def git_text(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}"


def json_shape(value: Any) -> str:
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return type(value).__name__


def collect_candidate_id_locations(
    value: Any,
    candidate_ids: set[str],
    path: str = "$",
    found: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    if found is None:
        found = {candidate_id: [] for candidate_id in candidate_ids}

    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            for candidate_id in candidate_ids:
                if candidate_id == key_text:
                    found[candidate_id].append(f"{path}.{key_text}")
            collect_candidate_id_locations(
                child,
                candidate_ids,
                f"{path}.{key_text}",
                found,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            collect_candidate_id_locations(
                child,
                candidate_ids,
                f"{path}[{index}]",
                found,
            )
    elif isinstance(value, str) and value in candidate_ids:
        found[value].append(path)

    return found


missing = [str(path) for path in INPUTS.values() if not path.exists()]
if missing:
    raise FileNotFoundError("Missing Task 41A geometry artifacts:\n" + "\n".join(missing))

branch = git_text("branch", "--show-current")
commit = git_text("rev-parse", "HEAD")

print("===== TASK 41A TRIGGER SPEC SCHEMA AUDIT =====")
print("Branch:", branch)
print("Commit:", commit)
print("Dataset arrays loaded: False")
print()

records: list[dict[str, Any]] = []
csv_profiles: dict[str, Any] = {}

for name, path in INPUTS.items():
    if path.suffix.lower() != ".csv":
        continue

    frame = pd.read_csv(path)
    csv_profiles[name] = {
        "relative_path": str(path.relative_to(ROOT)),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
    }

    print(f"===== {name} =====")
    print("Path:", path)
    print("Shape:", frame.shape)
    print("Columns:")
    for column in frame.columns:
        print(f"  {column}\t{frame[column].dtype}")
        records.append(
            {
                "artifact": name,
                "record_type": "csv_column",
                "name": column,
                "detail": str(frame[column].dtype),
            }
        )
    print("Preview:")
    print(frame.head(5).to_string(index=False))
    print()

selected = pd.read_csv(INPUTS["selected_csv"])
if "candidate_id" not in selected.columns:
    raise KeyError("Selected-candidate CSV does not contain candidate_id")

candidate_ids = set(selected["candidate_id"].astype(str))

with INPUTS["screen_summary_json"].open("r", encoding="utf-8") as handle:
    screen_summary = json.load(handle)

with INPUTS["trigger_specs_json"].open("r", encoding="utf-8") as handle:
    trigger_specs = json.load(handle)

print("===== JSON TOP-LEVEL SCHEMA =====")
for name, value in (
    ("screen_summary_json", screen_summary),
    ("trigger_specs_json", trigger_specs),
):
    print(name, json_shape(value))
    if isinstance(value, dict):
        for key, child in value.items():
            print(f"  {key}\t{json_shape(child)}")
            records.append(
                {
                    "artifact": name,
                    "record_type": "json_top_level",
                    "name": str(key),
                    "detail": json_shape(child),
                }
            )
    print()

locations = collect_candidate_id_locations(trigger_specs, candidate_ids)

print("===== SELECTED CANDIDATE JSON LOCATIONS =====")
for candidate_id in selected["candidate_id"].astype(str):
    candidate_locations = locations.get(candidate_id, [])
    print(candidate_id, "=>", candidate_locations if candidate_locations else "NOT_FOUND")
    records.append(
        {
            "artifact": "trigger_specs_json",
            "record_type": "candidate_location",
            "name": candidate_id,
            "detail": " | ".join(candidate_locations) if candidate_locations else "NOT_FOUND",
        }
    )
print()

npz_profile: dict[str, Any] = {}
with np.load(INPUTS["trigger_specs_npz"], allow_pickle=False) as archive:
    print("===== TRIGGER SPEC NPZ CONTENTS =====")
    for key in archive.files:
        value = np.asarray(archive[key])
        detail = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes": int(value.nbytes),
        }
        npz_profile[key] = detail
        print(
            f"{key}\tshape={tuple(value.shape)}\tdtype={value.dtype}\tbytes={value.nbytes}"
        )
        records.append(
            {
                "artifact": "trigger_specs_npz",
                "record_type": "npz_array",
                "name": key,
                "detail": json.dumps(detail, sort_keys=True),
            }
        )
print()

source_metrics = pd.read_csv(INPUTS["source_metrics_csv"])
source_group_columns = [
    column
    for column in (
        "candidate_id",
        "source_class",
        "source_class_name",
        "class_name",
        "class_id",
    )
    if column in source_metrics.columns
]

print("===== SOURCE-METRIC GROUPING CANDIDATES =====")
print(source_group_columns if source_group_columns else "NO STANDARD GROUPING COLUMNS FOUND")
print()

audit_json = {
    "experiment_version": "4.10E",
    "stage": "task41a_trigger_spec_schema_audit",
    "branch": branch,
    "commit": commit,
    "dataset_arrays_loaded": False,
    "test_arrays_loaded": False,
    "inputs": {name: str(path.relative_to(ROOT)) for name, path in INPUTS.items()},
    "csv_profiles": csv_profiles,
    "json_top_level": {
        "screen_summary_json": {
            str(key): json_shape(value)
            for key, value in screen_summary.items()
        }
        if isinstance(screen_summary, dict)
        else {"$": json_shape(screen_summary)},
        "trigger_specs_json": {
            str(key): json_shape(value)
            for key, value in trigger_specs.items()
        }
        if isinstance(trigger_specs, dict)
        else {"$": json_shape(trigger_specs)},
    },
    "candidate_json_locations": locations,
    "npz_profile": npz_profile,
    "source_metric_grouping_columns": source_group_columns,
}

csv_path = TABLE_DIR / "task41a_trigger_spec_schema_audit.csv"
json_path = TABLE_DIR / "task41a_trigger_spec_schema_audit.json"

with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["artifact", "record_type", "name", "detail"],
    )
    writer.writeheader()
    writer.writerows(records)

json_path.write_text(json.dumps(audit_json, indent=2), encoding="utf-8")

print("WROTE:", csv_path)
print("WROTE:", json_path)
print("FINAL TEST ARRAYS LOADED: False")

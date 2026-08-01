from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np

root = Path.home() / "LFighter-research"
data_root = root / "data" / "processed" / "cic_iot_diad_2024_v2_1"
array_path = data_root / "arrays" / "behavioral_only.npz"
output_dir = root / "results" / "cic_iot_diad_task41a_trigger_feasibility_v410a" / "inventory"
table_dir = output_dir / "tables"
table_dir.mkdir(parents=True, exist_ok=True)

if not array_path.is_file():
    raise FileNotFoundError(array_path)

branch = subprocess.check_output(
    ["git", "branch", "--show-current"], cwd=root, text=True
).strip()
commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=root, text=True
).strip()

name_tokens = (
    "feature",
    "schema",
    "metadata",
    "manifest",
    "range",
    "stat",
    "column",
    "class",
    "label",
)
text_suffixes = {".json", ".csv", ".txt", ".md", ".yaml", ".yml"}

metadata_files = sorted(
    p
    for p in data_root.rglob("*")
    if p.is_file() and any(token in p.name.lower() for token in name_tokens)
)

rows: list[dict[str, object]] = []
for path in metadata_files:
    rows.append(
        {
            "record_type": "metadata_file",
            "name": path.name,
            "relative_path": str(path.relative_to(root)),
            "shape": "",
            "dtype": "",
            "bytes": path.stat().st_size,
            "access_status": "metadata_only",
        }
    )

print("===== TASK 41A DATA INVENTORY =====")
print("Branch:", branch)
print("Commit:", commit)
print("Data root:", data_root)
print("Array file:", array_path)
print()
print("===== FULL METADATA PATHS =====")
for path in metadata_files:
    print(f"{path}\t{path.stat().st_size} bytes")

print()
print("===== NPZ CONTENTS =====")
with np.load(array_path, allow_pickle=False) as archive:
    for key in archive.files:
        if "test" in key.lower():
            print(f"{key}\tRESERVED_NOT_LOADED")
            rows.append(
                {
                    "record_type": "npz_array",
                    "name": key,
                    "relative_path": str(array_path.relative_to(root)),
                    "shape": "RESERVED",
                    "dtype": "RESERVED",
                    "bytes": "",
                    "access_status": "final_test_reserved_not_loaded",
                }
            )
            continue

        value = np.asarray(archive[key])
        print(f"{key}\tshape={value.shape}\tdtype={value.dtype}")
        rows.append(
            {
                "record_type": "npz_array",
                "name": key,
                "relative_path": str(array_path.relative_to(root)),
                "shape": "x".join(str(v) for v in value.shape),
                "dtype": str(value.dtype),
                "bytes": int(value.nbytes),
                "access_status": "development_allowed",
            }
        )

csv_path = table_dir / "task41a_dataset_inventory.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

json_path = table_dir / "task41a_dataset_inventory.json"
json_path.write_text(
    json.dumps(
        {
            "branch": branch,
            "commit": commit,
            "data_root": str(data_root),
            "array_file": str(array_path),
            "test_arrays_loaded": False,
            "records": rows,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("===== SMALL METADATA PREVIEWS =====")
for path in metadata_files:
    if path.suffix.lower() not in text_suffixes or path.stat().st_size > 2_000_000:
        continue
    print()
    print("-----", path, "-----")
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number > 20:
                    break
                print(f"{line_number:02d}: {line.rstrip()}")
    except Exception as exc:
        print("PREVIEW_ERROR:", repr(exc))

print()
print("WROTE:", csv_path)
print("WROTE:", json_path)
print("FINAL TEST ARRAYS LOADED: False")

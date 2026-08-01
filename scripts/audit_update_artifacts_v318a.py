#!/usr/bin/env python3
"""V3.18A update-artifact recoverability audit.

Determines whether full per-client model-update information already exists and
can be reused. This stage does not retrain models, alter prior results, or
access natural/diagnostic test sets.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

TEXT_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".md", ".txt"}
BINARY_EXTENSIONS = {".pt", ".pth", ".ckpt", ".npz", ".npy"}
TABLE_EXTENSIONS = {".csv", ".parquet", ".feather"}

UPDATE_TERMS = [
    "client_update", "client_updates", "local_update", "local_updates",
    "update_vector", "update_vectors", "delta_vector", "delta_vectors",
    "model_delta", "model_deltas", "parameter_delta", "parameter_deltas",
    "flattened_update", "flat_update", "gradient_vector", "grad_vector",
    "layer_update", "layer_updates", "local_state_dict", "local_model_state",
    "client_state_dict", "actual_update_norm", "update_norm",
    "cosine_similarity", "layer_norm", "residual_norm",
]
SAVE_TERMS = [
    "torch.save", "np.save", "np.savez", "to_csv", "to_parquet",
    "pickle.dump", "joblib.dump", "save_checkpoint",
]
PATH_TOKENS = [
    "update", "delta", "gradient", "grad", "client", "local", "checkpoint",
    "state", "reconstruction", "signature", "anchor",
]
VECTOR_PATTERNS = [
    re.compile(r"(client|local).*(update|delta|gradient).*(vector|flat|tensor)?", re.I),
    re.compile(r"(update|delta|gradient).*(vector|flat|tensor)", re.I),
    re.compile(r"(model|parameter).*(delta|difference)", re.I),
]
STATE_PATTERN = re.compile(r"(state_dict|model_state|local_state|client_state)", re.I)
NORM_ONLY_PATTERN = re.compile(r"(norm|magnitude|cosine|similarity|score|probability|signature)", re.I)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-binary-mb", type=int, default=512)
    p.add_argument("--max-text-mb", type=int, default=8)
    return p.parse_args()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def is_candidate_path(path: Path) -> bool:
    value = str(path).lower()
    return any(token in value for token in PATH_TOKENS)


def scan_code(repo_root: Path, max_bytes: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for scan_root in [repo_root / "scripts", repo_root / "src", repo_root / "lfighter", repo_root]:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                if path.stat().st_size > max_bytes:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                lower = line.lower()
                updates = sorted({x for x in UPDATE_TERMS if x.lower() in lower})
                saves = sorted({x for x in SAVE_TERMS if x.lower() in lower})
                if updates or saves:
                    rows.append({
                        "file": rel(path, repo_root),
                        "line_number": lineno,
                        "matched_update_terms": "|".join(updates),
                        "matched_save_terms": "|".join(saves),
                        "possible_update_save_same_line": bool(updates and saves),
                        "line": line.strip()[:1000],
                    })
    return pd.DataFrame(rows)


def scan_table(path: Path, repo_root: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "file": rel(path, repo_root), "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size, "read_success": False,
        "row_count": np.nan, "column_count": np.nan,
        "update_related_columns": "", "explicit_vector_columns": "",
        "norm_or_score_columns": "", "error": "",
    }
    try:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, nrows=0)
            columns = [str(c) for c in frame.columns]
            try:
                with path.open("rb") as f:
                    row_count = max(sum(1 for _ in f) - 1, 0)
            except Exception:
                row_count = np.nan
        elif path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
            columns = [str(c) for c in frame.columns]
            row_count = len(frame)
        else:
            frame = pd.read_feather(path)
            columns = [str(c) for c in frame.columns]
            row_count = len(frame)
        update_cols = [c for c in columns if any(t.lower() in c.lower() for t in UPDATE_TERMS)]
        vector_cols = [
            c for c in columns
            if any(p.search(c) for p in VECTOR_PATTERNS) and not NORM_ONLY_PATTERN.search(c)
        ]
        norm_cols = [c for c in columns if NORM_ONLY_PATTERN.search(c)]
        row.update({
            "read_success": True, "row_count": row_count,
            "column_count": len(columns),
            "update_related_columns": "|".join(update_cols),
            "explicit_vector_columns": "|".join(vector_cols),
            "norm_or_score_columns": "|".join(norm_cols),
        })
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:1000]
    return row


def tensor_record(path: Path, key: str, shape: Tuple[int, ...], dtype: str, count: int, repo_root: Path) -> Dict[str, Any]:
    key_text = key or path.stem
    explicit_key = any(p.search(key_text) for p in VECTOR_PATTERNS) and not NORM_ONLY_PATTERN.search(key_text)
    state_key = bool(STATE_PATTERN.search(key_text))
    likely_client_axis = any(x in {20, 80, 320} for x in shape)
    likely_parameter_axis = count >= 1000 or any(x >= 1000 for x in shape)
    return {
        "file": rel(path, repo_root), "container_key": key,
        "shape": str(shape), "dtype": dtype, "element_count": count,
        "likely_client_axis": likely_client_axis,
        "likely_parameter_axis": likely_parameter_axis,
        "explicit_vector_key": explicit_key,
        "state_key": state_key,
        "contains_explicit_client_update_vector": bool(explicit_key and likely_parameter_axis),
        "contains_model_state_tensor": bool(state_key and likely_parameter_axis),
        "inspection_success": True, "error": "",
    }


def inspect_numpy(path: Path, repo_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as z:
                for key in z.files:
                    arr = np.asarray(z[key])
                    rows.append(tensor_record(path, key, tuple(int(x) for x in arr.shape), str(arr.dtype), int(arr.size), repo_root))
        else:
            arr = np.load(path, allow_pickle=False, mmap_mode="r")
            rows.append(tensor_record(path, path.stem, tuple(int(x) for x in arr.shape), str(arr.dtype), int(arr.size), repo_root))
    except Exception as exc:
        rows.append({
            "file": rel(path, repo_root), "container_key": "", "shape": "", "dtype": "",
            "element_count": np.nan, "likely_client_axis": False,
            "likely_parameter_axis": False, "explicit_vector_key": False,
            "state_key": False, "contains_explicit_client_update_vector": False,
            "contains_model_state_tensor": False, "inspection_success": False,
            "error": f"{type(exc).__name__}: {exc}"[:1000],
        })
    return rows


def flatten_object(obj: Any, prefix: str = "", depth: int = 0) -> Iterable[Tuple[str, Any]]:
    if depth > 4:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{prefix}.{k}" if prefix else str(k)
            yield from flatten_object(v, child, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:100]):
            yield from flatten_object(v, f"{prefix}[{i}]", depth + 1)
    else:
        yield prefix, obj


def inspect_torch(path: Path, repo_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        import torch
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        for key, value in flatten_object(obj):
            if torch.is_tensor(value):
                rows.append(tensor_record(
                    path, key, tuple(int(x) for x in value.shape), str(value.dtype), int(value.numel()), repo_root
                ))
        if not rows:
            rows.append({
                "file": rel(path, repo_root), "container_key": "", "shape": "", "dtype": "",
                "element_count": np.nan, "likely_client_axis": False,
                "likely_parameter_axis": False, "explicit_vector_key": False,
                "state_key": False, "contains_explicit_client_update_vector": False,
                "contains_model_state_tensor": False, "inspection_success": True,
                "error": "No tensor leaves found",
            })
    except Exception as exc:
        rows.append({
            "file": rel(path, repo_root), "container_key": "", "shape": "", "dtype": "",
            "element_count": np.nan, "likely_client_axis": False,
            "likely_parameter_axis": False, "explicit_vector_key": False,
            "state_key": False, "contains_explicit_client_update_vector": False,
            "contains_model_state_tensor": False, "inspection_success": False,
            "error": f"{type(exc).__name__}: {exc}"[:1000],
        })
    return rows


def inventory(repo_root: Path, max_binary_bytes: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artifact_rows: List[Dict[str, Any]] = []
    binary_rows: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, Any]] = []
    seen = set()
    for root in [repo_root / "results", repo_root / "data" / "processed", repo_root / "checkpoints"]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in BINARY_EXTENSIONS | TABLE_EXTENSIONS:
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            suffix = path.suffix.lower()
            size = path.stat().st_size
            candidate = is_candidate_path(path)
            record = {
                "file": rel(path, repo_root), "extension": suffix,
                "size_bytes": size, "candidate_path": candidate,
                "inspected": False, "skip_reason": "",
            }
            artifact_rows.append(record)
            if suffix in TABLE_EXTENSIONS and candidate:
                table_rows.append(scan_table(path, repo_root))
                record["inspected"] = True
            elif suffix in BINARY_EXTENSIONS and candidate:
                if size > max_binary_bytes:
                    record["skip_reason"] = f"larger_than_{max_binary_bytes}_bytes"
                    continue
                if suffix in {".npz", ".npy"}:
                    binary_rows.extend(inspect_numpy(path, repo_root))
                else:
                    binary_rows.extend(inspect_torch(path, repo_root))
                record["inspected"] = True
    return pd.DataFrame(artifact_rows), pd.DataFrame(binary_rows), pd.DataFrame(table_rows)


def checkpoint_evidence(binary: pd.DataFrame) -> Dict[str, Any]:
    if binary.empty:
        return {
            "per_client_local_state_files": 0,
            "global_checkpoint_files": 0,
            "sufficient_local_checkpoint_pattern": False,
        }
    state_files = binary.loc[
        binary["contains_model_state_tensor"].fillna(False), "file"
    ].drop_duplicates().tolist()
    local_files = [
        f for f in state_files
        if re.search(r"(client|local).*(round|r\d+)|(round|r\d+).*(client|local)", f, re.I)
    ]
    global_files = [f for f in state_files if re.search(r"(global|continuation|checkpoint|model)", f, re.I)]
    return {
        "per_client_local_state_files": len(set(local_files)),
        "global_checkpoint_files": len(set(global_files)),
        "sufficient_local_checkpoint_pattern": bool(len(set(local_files)) >= 80 and len(set(global_files)) >= 1),
    }


def main() -> int:
    a = parse_args()
    repo_root = a.repo_root.expanduser().resolve()
    out = a.output_dir.expanduser().resolve()
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if not repo_root.exists():
        raise FileNotFoundError(repo_root)

    code = scan_code(repo_root, int(a.max_text_mb) * 1024 * 1024)
    artifacts, binary, schemas = inventory(repo_root, int(a.max_binary_mb) * 1024 * 1024)

    code.to_csv(tables / "v318a_code_evidence.csv", index=False)
    artifacts.to_csv(tables / "v318a_candidate_artifact_inventory.csv", index=False)
    binary.to_csv(tables / "v318a_binary_structure_inventory.csv", index=False)
    schemas.to_csv(tables / "v318a_table_schema_inventory.csv", index=False)

    explicit_binary = int(binary["contains_explicit_client_update_vector"].fillna(False).sum()) if not binary.empty else 0
    explicit_tables = int(schemas["explicit_vector_columns"].fillna("").astype(str).str.len().gt(0).sum()) if not schemas.empty else 0
    norm_tables = int(schemas["norm_or_score_columns"].fillna("").astype(str).str.len().gt(0).sum()) if not schemas.empty else 0
    save_lines = int(code["possible_update_save_same_line"].fillna(False).sum()) if not code.empty else 0
    ckpt = checkpoint_evidence(binary)

    full_vectors = explicit_binary > 0 or explicit_tables > 0
    reconstructable = (not full_vectors) and ckpt["sufficient_local_checkpoint_pattern"]
    if full_vectors:
        decision = "FULL_UPDATE_VECTORS_FOUND"
        next_stage = "Build V3.18B offline update-space feature extraction; do not rerun federated training."
    elif reconstructable:
        decision = "LOCAL_CHECKPOINT_RECONSTRUCTION_POSSIBLE"
        next_stage = "Verify exact per-client/round checkpoint coverage and reconstruct updates offline."
    else:
        decision = "INSTRUMENTATION_REQUIRED"
        next_stage = "Patch the continuation runner to save deterministic client update vectors/sketches, then run the pre-registered smoke grid only."

    decision_row = {
        "decision": decision,
        "explicit_binary_update_vector_records": explicit_binary,
        "tables_with_explicit_vector_columns": explicit_tables,
        "tables_with_norm_or_score_columns": norm_tables,
        "code_lines_with_update_and_save_terms": save_lines,
        **ckpt,
        "federated_training_rerun": False,
        "prior_results_modified": False,
        "test_sets_accessed": False,
        "recommended_next_stage": next_stage,
    }
    pd.DataFrame([decision_row]).to_csv(tables / "v318a_decision.csv", index=False)
    with (out / "v318a_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "experiment_version": "3.18A",
            "stage": "update_artifact_recoverability_audit",
            "repo_root": str(repo_root),
            "decision": decision,
            "federated_training_rerun": False,
            "prior_results_modified": False,
            "test_sets_accessed": False,
            "recommended_next_stage": next_stage,
        }, f, indent=2)

    print("V3.18A update-artifact recoverability audit complete")
    print()
    print("DECISION")
    print(pd.DataFrame([decision_row]).to_string(index=False))
    print()
    if not schemas.empty:
        relevant = schemas[
            schemas["update_related_columns"].fillna("").astype(str).str.len().gt(0)
            | schemas["norm_or_score_columns"].fillna("").astype(str).str.len().gt(0)
        ]
        print("UPDATE-RELATED TABLE SCHEMAS")
        if len(relevant):
            print(relevant[[
                "file", "row_count", "column_count", "update_related_columns",
                "explicit_vector_columns", "norm_or_score_columns"
            ]].head(50).to_string(index=False))
        else:
            print("No update-related table columns found.")
        print()
    if not binary.empty:
        relevant_binary = binary[
            binary["contains_explicit_client_update_vector"].fillna(False)
            | binary["contains_model_state_tensor"].fillna(False)
        ]
        print("RELEVANT BINARY STRUCTURES")
        if len(relevant_binary):
            print(relevant_binary[[
                "file", "container_key", "shape", "element_count",
                "contains_explicit_client_update_vector", "contains_model_state_tensor"
            ]].head(80).to_string(index=False))
        else:
            print("No explicit update vectors or reconstructable local states found.")
        print()
    print("Tables:", tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

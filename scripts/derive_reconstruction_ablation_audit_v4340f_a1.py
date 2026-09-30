#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "audit_reconstruction_ablation_preflight_v4340f.py"
OUTPUT = ROOT / "scripts" / "audit_reconstruction_ablation_preflight_v4340f_a1.py"

EXPECTED_SOURCE_SHA = "9d9d52b3b16b06cdd2fac5be18578c2db2d3bcf75439abe79b9287e9f75c3198"
TARGET_KEY = "guard_no_reserved_test_tokens"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def assignment_key(node: ast.AST):
    targets = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    for t in targets:
        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "checks":
            sl = t.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                return sl.value
    return None

def main() -> int:
    got = sha256_file(SOURCE)
    if got != EXPECTED_SOURCE_SHA:
        raise RuntimeError(
            f"Original v4340f audit hash mismatch: expected {EXPECTED_SOURCE_SHA}, observed {got}"
        )

    s = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(s)
    hits = [n for n in ast.walk(tree) if assignment_key(n) == TARGET_KEY]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one {TARGET_KEY} assignment, found {len(hits)}")
    node = hits[0]
    lines = s.splitlines(keepends=True)
    indent = " " * node.col_offset

    replacement_lines = [
        f'{indent}# v4340f-a1: semantic Task-65 guard audit.\n',
        f'{indent}_guard_path = ROOT / "results" / "reviewer_v4340f_guard_access_log.jsonl"\n',
        f'{indent}_guard_records = [\n',
        f'{indent}    json.loads(_line)\n',
        f'{indent}    for _line in _guard_path.read_text(encoding="utf-8").splitlines()\n',
        f'{indent}    if _line.strip()\n',
        f'{indent}]\n',
        f'{indent}_allowed_materialized = {{"X_train", "y_train", "X_val", "y_val"}}\n',
        f'{indent}_reserved_test = {{\n',
        f'{indent}    "X_test_natural", "y_test_natural",\n',
        f'{indent}    "X_test_diagnostic", "y_test_diagnostic",\n',
        f'{indent}}}\n',
        f'{indent}_load_events = [\n',
        f'{indent}    _r for _r in _guard_records\n',
        f'{indent}    if _r.get("event") == "guarded_load_protocol_arrays"\n',
        f'{indent}]\n',
        f'{indent}_all_materialized = [\n',
        f'{indent}    _k\n',
        f'{indent}    for _r in _guard_records\n',
        f'{indent}    for _k in _r.get("materialized_keys", [])\n',
        f'{indent}]\n',
        f'{indent}checks["guard_reserved_tests_not_materialized"] = bool(_load_events) and all(\n',
        f'{indent}    _r.get("test_arrays_materialized") is False\n',
        f'{indent}    for _r in _guard_records\n',
        f'{indent}    if "test_arrays_materialized" in _r\n',
        f'{indent}) and not any(\n',
        f'{indent}    _k in _reserved_test for _k in _all_materialized\n',
        f'{indent}) and all(\n',
        f'{indent}    set(_r.get("materialized_keys", [])).issubset(_allowed_materialized)\n',
        f'{indent}    for _r in _load_events\n',
        f'{indent}) and all(\n',
        f'{indent}    set(_r.get("test_array_names_verified_only", [])) == _reserved_test\n',
        f'{indent}    for _r in _load_events\n',
        f'{indent})\n',
    ]
    lines[node.lineno - 1:node.end_lineno] = replacement_lines
    out = "".join(lines)

    if "import json" not in out:
        marker = "from __future__ import annotations\n"
        if marker in out:
            out = out.replace(marker, marker + "import json\n", 1)
        else:
            out = "import json\n" + out

    header = (
        "# v4.34.0f-a1 AUDIT-ONLY CORRECTION\n"
        "# Parent audit SHA256: "
        "9D9D52B3B16B06CDD2FAC5BE18578C2DB2D3BCF75439ABE79B9287E9F75C3198\n"
        "# No scientific experiment is rerun or modified.\n"
    )
    out = header + out

    parsed = ast.parse(out)
    remaining = [n for n in ast.walk(parsed) if assignment_key(n) == TARGET_KEY]
    if remaining:
        raise RuntimeError("Obsolete token-presence guard check still exists")

    semantic = [
        n for n in ast.walk(parsed)
        if assignment_key(n) == "guard_reserved_tests_not_materialized"
    ]
    if len(semantic) != 1:
        raise RuntimeError("Correct semantic guard assignment was not created exactly once")

    OUTPUT.write_text(out, encoding="utf-8", newline="\n")
    print("V4340F-A1 CORRECTED AUDIT CREATED")
    print("PARENT AUDIT SHA256:", got.upper())
    print("CORRECTED AUDIT SHA256:", sha256_file(OUTPUT).upper())
    print("OBSOLETE TOKEN CHECK REMAINS: False")
    print("SEMANTIC MATERIALIZATION CHECK PRESENT: True")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

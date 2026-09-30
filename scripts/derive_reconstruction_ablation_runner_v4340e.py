#!/usr/bin/env python3
from __future__ import annotations
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "run_reviewer_reconstruction_ablation_v4340c.py"
OUTPUT = ROOT / "scripts" / "run_reviewer_reconstruction_ablation_v4340e.py"
EXPECTED_SOURCE_SHA = "d2eb3c8cd2803a2807705c317528830897dbeac5901747c70356db4d54a55d7b"
TARGET_MESSAGE = "--reconstruction-calibration-dir is required for trusted_reconstruction"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def raise_contains(node: ast.AST, phrase: str) -> bool:
    for x in ast.walk(node):
        if isinstance(x, ast.Raise) and x.exc is not None:
            for y in ast.walk(x.exc):
                if isinstance(y, ast.Constant) and isinstance(y.value, str) and phrase in y.value:
                    return True
    return False

def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    got = sha256_file(SOURCE)
    if got != EXPECTED_SOURCE_SHA:
        raise RuntimeError(
            f"Frozen v4.34.0c runner hash mismatch: expected {EXPECTED_SOURCE_SHA}, observed {got}"
        )

    s = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(s)

    candidates = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and raise_contains(node, TARGET_MESSAGE)
    ]
    if not candidates:
        raise RuntimeError("Could not locate inherited reconstruction-calibration CLI guard")

    # Remove the smallest enclosing if-block containing the target raise.
    candidates.sort(key=lambda n: (n.end_lineno - n.lineno, n.lineno))
    guard = candidates[0]

    lines = s.splitlines(keepends=True)
    removed = "".join(lines[guard.lineno - 1:guard.end_lineno])

    # Verify the AST-selected source region is the intended calibration-dir guard,
    # without requiring the adjacent string literals to be contiguous in raw source.
    if "reconstruction_calibration_dir" not in removed:
        raise RuntimeError("AST-selected guard does not reference reconstruction_calibration_dir")
    if "trusted_reconstruction" not in removed:
        raise RuntimeError("AST-selected guard does not reference trusted_reconstruction")

    replacement = (
        " " * guard.col_offset
        + "# v4.34.0e: inherited center_plus_residual calibration-dir CLI guard removed.\\n"
        + " " * guard.col_offset
        + "# center_only needs no historical residual; hard_rejection performs no reconstruction.\\n"
    )
    lines[guard.lineno - 1:guard.end_lineno] = [replacement]
    out = "".join(lines)

    old_version = '"4.34.0c-RECONSTRUCTION-ABLATION"'
    new_version = '"4.34.0e-RECONSTRUCTION-ABLATION"'
    if old_version not in out:
        raise RuntimeError("Expected v4.34.0c experiment version not found")
    out = out.replace(old_version, new_version)

    anchor = '        "reconstruction_calibration_required": False,\n'
    if anchor not in out:
        raise RuntimeError("Expected reviewer-ablation metadata anchor not found")
    out = out.replace(
        anchor,
        anchor
        + '        "preoutcome_correction_from_v4340c": True,\n'
        + '        "reconstruction_calibration_cli_guard_removed": True,\n',
        1,
    )

    header = (
        "# v4.34.0e PRE-OUTCOME SETUP CORRECTION\\n"
        "# Parent runner: v4.34.0c SHA256 "
        "D2EB3C8CD2803A2807705C317528830897DBEAC5901747C70356DB4D54A55D7B\\n"
        "# Scientific settings unchanged; only inherited calibration-dir CLI guard removed.\\n"
    )
    out = header + out

    parsed = ast.parse(out)
    target_raises = []
    for n in ast.walk(parsed):
        if isinstance(n, ast.Raise) and n.exc is not None:
            vals = [
                c.value for c in ast.walk(n.exc)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
            ]
            if any(TARGET_MESSAGE in v for v in vals):
                target_raises.append(n.lineno)
    if target_raises:
        raise RuntimeError(f"Target inherited CLI guard still present at lines {target_raises}")

    OUTPUT.write_text(out, encoding="utf-8", newline="\n")
    print("V4340E RUNNER CREATED")
    print("PARENT SHA256:", got.upper())
    print("DERIVED SHA256:", sha256_file(OUTPUT).upper())
    print("REMOVED AST GUARD LINES:", guard.lineno, "-", guard.end_lineno)
    print("REMOVED GUARD REFERENCES reconstruction_calibration_dir: True")
    print("REMOVED GUARD REFERENCES trusted_reconstruction: True")
    print("TARGET RAISE REMAINS: False")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

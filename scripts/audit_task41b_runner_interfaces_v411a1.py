from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib import format as npformat


EXPERIMENT_VERSION = "4.11A.1"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT
    / "results"
    / "cic_iot_diad_task41b_backdoor_v411a"
    / "interface_audit"
)
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
for directory in (TABLE_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SOURCE_FILES = [
    ROOT / "scripts" / "run_exact_untargeted_plain_v320a3.py",
    ROOT / "scripts" / "run_frozen_untargeted_defense_v320b1.py",
    ROOT / "scripts" / "verify_frozen_untargeted_v320b1.py",
    ROOT / "scripts" / "run_untargeted_label_poisoning_v320a.py",
    ROOT / "scripts" / "run_targeted_attack_breadth_v313.py",
    ROOT / "scripts" / "run_exact_plain_fedavg_v3132.py",
    ROOT / "src" / "trusted_update_reconstruction_v312.py",
    ROOT / "src" / "independent_anchor_v310.py",
]

DATA_NPZ = (
    ROOT
    / "data"
    / "processed"
    / "cic_iot_diad_2024_v2_1"
    / "arrays"
    / "behavioral_only.npz"
)
PARTITION_NPZ = (
    ROOT
    / "results"
    / "cic_iot_diad_federated_tuning_v27_alpha05_seed42"
    / "partitions"
    / "client_partitions.npz"
)
FROZEN_PROTOCOL_ROOT = (
    ROOT
    / "results"
    / "cic_iot_diad_task41a_trigger_feasibility_v410a"
    / "frozen_protocol"
)
FROZEN_PANEL_CSV = (
    FROZEN_PROTOCOL_ROOT
    / "tables"
    / "task41a_frozen_trigger_panel.csv"
)
FROZEN_PROTOCOL_JSON = (
    FROZEN_PROTOCOL_ROOT
    / "tables"
    / "task41_frozen_backdoor_protocol.json"
)
FROZEN_SPECS_JSON = (
    FROZEN_PROTOCOL_ROOT
    / "trigger_specs"
    / "task41a_frozen_trigger_specs.json"
)
FROZEN_SPECS_NPZ = (
    FROZEN_PROTOCOL_ROOT
    / "trigger_specs"
    / "task41a_frozen_trigger_specs.npz"
)

REFERENCE_DIRS = {
    "clean_seed_7": (
        ROOT
        / "results"
        / "cic_iot_diad_federated_multiseed_v28_fixed_partition"
        / "seed_runs"
        / "seed_7"
    ),
    "warmup_seed_7": (
        ROOT
        / "results"
        / "cic_iot_diad_true_warmup_anchor_v3101_multiseed"
        / "seed_7"
        / "warmup"
    ),
    "v320a3_seed7_plain": (
        ROOT
        / "results"
        / "cic_iot_diad_untargeted_exact_qualification_v320a3"
        / "runs"
        / "all_to_one_benign"
        / "seed_7"
        / "plain_fedavg"
    ),
    "v320b1_seed7_adapter_plain": (
        ROOT
        / "results"
        / "cic_iot_diad_frozen_untargeted_defense_v320b1"
        / "adapter_attack"
        / "all_to_one_benign"
        / "seed_7"
        / "plain_fedavg"
    ),
    "v320b1_seed7_defended": (
        ROOT
        / "results"
        / "cic_iot_diad_frozen_untargeted_defense_v320b1"
        / "runs"
        / "all_to_one_benign"
        / "seed_7"
        / "trusted_reconstruction"
    ),
}

KEYWORDS = (
    "X_train",
    "X_val",
    "X_test_natural",
    "X_test_diagnostic",
    "client_partitions",
    "malicious_clients",
    "poison_fraction",
    "local_positions",
    "global_indices",
    "replacement_policy",
    "plain_fedavg",
    "trusted_reconstruction",
    "state_dict",
    "common_round4_warmup_model",
    "source_class",
    "target_class",
)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
        if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict)):
            return value
    except Exception:
        pass
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    parts: list[str] = []
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(
        node.args.defaults
    )
    for argument, default in zip(positional, defaults):
        text = argument.arg
        if argument.annotation is not None:
            text += f": {safe_literal(argument.annotation)}"
        if default is not None:
            text += f" = {safe_literal(default)!r}"
        parts.append(text)
    if node.args.vararg is not None:
        parts.append(f"*{node.args.vararg.arg}")
    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        text = argument.arg
        if default is not None:
            text += f" = {safe_literal(default)!r}"
        parts.append(text)
    if node.args.kwarg is not None:
        parts.append(f"**{node.args.kwarg.arg}")
    return f"{node.name}({', '.join(parts)})"


def read_npy_header(handle: Any) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = npformat.read_magic(handle)
    if version == (1, 0):
        shape, fortran_order, dtype = npformat.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran_order, dtype = npformat.read_array_header_2_0(handle)
    elif version == (3, 0):
        shape, fortran_order, dtype = npformat.read_array_header_2_0(handle)
    else:
        raise ValueError(f"Unsupported NPY version: {version}")
    return shape, fortran_order, dtype


def npz_header_rows(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path, "r") as archive:
        for member in sorted(archive.namelist()):
            if not member.endswith(".npy"):
                continue
            with archive.open(member, "r") as handle:
                shape, fortran_order, dtype = read_npy_header(handle)
            rows.append(
                {
                    "archive": label,
                    "relative_path": str(path.relative_to(ROOT)),
                    "array_key": member[:-4],
                    "shape": "x".join(str(value) for value in shape),
                    "dtype": str(dtype),
                    "fortran_order": bool(fortran_order),
                    "header_only": True,
                    "reserved_test_array": member[:-4]
                    in {
                        "X_test_natural",
                        "X_test_diagnostic",
                        "y_test_natural",
                        "y_test_diagnostic",
                    },
                }
            )
    return rows


branch = git_text("branch", "--show-current")
commit = git_text("rev-parse", "HEAD")

for path in SOURCE_FILES + [
    DATA_NPZ,
    PARTITION_NPZ,
    FROZEN_PANEL_CSV,
    FROZEN_PROTOCOL_JSON,
    FROZEN_SPECS_JSON,
    FROZEN_SPECS_NPZ,
]:
    if not path.exists():
        raise FileNotFoundError(path)

definition_rows: list[dict[str, Any]] = []
argument_rows: list[dict[str, Any]] = []
import_rows: list[dict[str, Any]] = []
keyword_rows: list[dict[str, Any]] = []
source_summary_rows: list[dict[str, Any]] = []

for path in SOURCE_FILES:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    tree = ast.parse(text, filename=str(path))
    relative = str(path.relative_to(ROOT))
    definitions = 0
    arguments = 0
    imports = 0

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions += 1
            definition_rows.append(
                {
                    "relative_path": relative,
                    "definition_type": "function",
                    "name": node.name,
                    "signature": function_signature(node),
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno),
                }
            )
        elif isinstance(node, ast.ClassDef):
            definitions += 1
            definition_rows.append(
                {
                    "relative_path": relative,
                    "definition_type": "class",
                    "name": node.name,
                    "signature": node.name,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno),
                }
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports += 1
                import_rows.append(
                    {
                        "relative_path": relative,
                        "module": alias.name,
                        "imported_name": "",
                        "alias": alias.asname or "",
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports += 1
                import_rows.append(
                    {
                        "relative_path": relative,
                        "module": module,
                        "imported_name": alias.name,
                        "alias": alias.asname or "",
                    }
                )
        elif isinstance(node, ast.Call):
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "add_argument"
                and node.args
            ):
                flags = [
                    safe_literal(argument)
                    for argument in node.args
                    if isinstance(safe_literal(argument), str)
                ]
                keyword_map = {
                    keyword.arg: safe_literal(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
                arguments += 1
                argument_rows.append(
                    {
                        "relative_path": relative,
                        "flags": "|".join(str(flag) for flag in flags),
                        "dest": keyword_map.get("dest", ""),
                        "required": keyword_map.get("required", False),
                        "default": json.dumps(
                            keyword_map.get("default"),
                            ensure_ascii=False,
                            default=str,
                        ),
                        "type": str(keyword_map.get("type", "")),
                        "choices": json.dumps(
                            keyword_map.get("choices"),
                            ensure_ascii=False,
                            default=str,
                        ),
                        "action": str(keyword_map.get("action", "")),
                        "line": node.lineno,
                    }
                )

    for line_number, line in enumerate(text.splitlines(), start=1):
        matched = [keyword for keyword in KEYWORDS if keyword in line]
        if matched:
            keyword_rows.append(
                {
                    "relative_path": relative,
                    "line": line_number,
                    "keywords": "|".join(matched),
                    "source_text": line.strip()[:500],
                }
            )

    source_summary_rows.append(
        {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "top_level_definitions": definitions,
            "argparse_arguments": arguments,
            "imports": imports,
        }
    )

npz_rows = npz_header_rows(DATA_NPZ, "behavioral_only")
npz_rows.extend(npz_header_rows(PARTITION_NPZ, "client_partitions"))
npz_rows.extend(npz_header_rows(FROZEN_SPECS_NPZ, "frozen_trigger_specs"))

artifact_rows: list[dict[str, Any]] = []
for label, directory in REFERENCE_DIRS.items():
    if not directory.exists():
        artifact_rows.append(
            {
                "reference": label,
                "relative_path": str(directory.relative_to(ROOT)),
                "file_type": "directory",
                "bytes": 0,
                "sha256": "",
                "csv_rows": "",
                "csv_columns": "",
                "status": "MISSING",
            }
        )
        continue

    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in {".csv", ".json", ".pt", ".pth", ".npz"}
    )
    for path in files:
        row = {
            "reference": label,
            "relative_path": str(path.relative_to(ROOT)),
            "file_type": path.suffix.lower(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "csv_rows": "",
            "csv_columns": "",
            "status": "PRESENT",
        }
        if path.suffix.lower() == ".csv":
            try:
                frame = pd.read_csv(path)
                row["csv_rows"] = int(len(frame))
                row["csv_columns"] = "|".join(frame.columns)
            except pd.errors.EmptyDataError:
                row["csv_rows"] = 0
                row["csv_columns"] = ""
                row["status"] = "PRESENT_EMPTY_CSV"
            except pd.errors.ParserError as exc:
                row["csv_rows"] = ""
                row["csv_columns"] = ""
                row["status"] = f"PRESENT_CSV_PARSE_ERROR:{type(exc).__name__}"
        artifact_rows.append(row)

panel = pd.read_csv(FROZEN_PANEL_CSV)
with FROZEN_PROTOCOL_JSON.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)
with FROZEN_SPECS_JSON.open("r", encoding="utf-8") as handle:
    frozen_specs = json.load(handle)

flow_row = panel.loc[panel["panel_slot"] == "flow_iat_exact"]
if len(flow_row) != 1:
    raise AssertionError("Expected exactly one flow_iat_exact panel row")
flow_candidate_id = str(flow_row.iloc[0]["candidate_id"])
selected_specs = frozen_specs.get("selected_specs", {})
if flow_candidate_id not in selected_specs:
    raise KeyError(f"Missing frozen spec: {flow_candidate_id}")

smoke_contract = {
    "experiment_version": EXPERIMENT_VERSION,
    "stage": "task41b_interface_audit_empty_csv_safe",
    "branch": branch,
    "commit": commit,
    "test_arrays_loaded": False,
    "test_arrays_header_inspected_only": True,
    "frozen_protocol_commit": "698bbf2",
    "smoke_configuration": {
        "panel_slot": "flow_iat_exact",
        "candidate_id": flow_candidate_id,
        "target_class_id": protocol["target_class"]["id"],
        "target_class_name": protocol["target_class"]["name"],
        "source_classes": protocol["source_classes"],
        "model_seed": 7,
        "attack_seed": 7,
        "poison_fraction": 0.01,
        "malicious_clients": protocol["malicious_clients"],
        "warmup_rounds": protocol["warmup_rounds"],
        "post_warmup_rounds": protocol["post_warmup_rounds"],
        "label_policy": protocol["primary_qualification"]["label_policy"],
        "deployment_policy": protocol["primary_qualification"][
            "deployment_policy"
        ],
        "required_pretraining_gate": (
            "zero-poison adapter must exactly reproduce the matched clean "
            "reference at metric and model-state level"
        ),
        "required_attack_gate": (
            "triggered-label poison plan must be generated once and reused "
            "unchanged by all paired branches"
        ),
        "natural_test_accessed": False,
        "diagnostic_test_accessed": False,
    },
    "frozen_trigger_spec": selected_specs[flow_candidate_id],
    "source_files": source_summary_rows,
}

definitions_path = TABLE_DIR / "task41b_source_definitions.csv"
arguments_path = TABLE_DIR / "task41b_argparse_interfaces.csv"
imports_path = TABLE_DIR / "task41b_import_inventory.csv"
keywords_path = TABLE_DIR / "task41b_keyword_locations.csv"
source_summary_path = TABLE_DIR / "task41b_source_file_summary.csv"
npz_path = TABLE_DIR / "task41b_npz_header_inventory.csv"
artifacts_path = TABLE_DIR / "task41b_reference_artifact_schemas.csv"
contract_path = TABLE_DIR / "task41b_smoke_contract.json"

pd.DataFrame(definition_rows).to_csv(definitions_path, index=False)
pd.DataFrame(argument_rows).to_csv(arguments_path, index=False)
pd.DataFrame(import_rows).to_csv(imports_path, index=False)
pd.DataFrame(keyword_rows).to_csv(keywords_path, index=False)
pd.DataFrame(source_summary_rows).to_csv(source_summary_path, index=False)
pd.DataFrame(npz_rows).to_csv(npz_path, index=False)
pd.DataFrame(artifact_rows).to_csv(artifacts_path, index=False)
contract_path.write_text(json.dumps(smoke_contract, indent=2), encoding="utf-8")

summary_frame = pd.DataFrame(source_summary_rows)
fig, axis = plt.subplots(figsize=(11.0, 6.2))
x = np.arange(len(summary_frame))
axis.bar(
    x - 0.22,
    summary_frame["top_level_definitions"],
    width=0.22,
    label="Definitions",
)
axis.bar(
    x,
    summary_frame["argparse_arguments"],
    width=0.22,
    label="CLI arguments",
)
axis.bar(
    x + 0.22,
    summary_frame["imports"],
    width=0.22,
    label="Imports",
)
axis.set_xticks(x)
axis.set_xticklabels(
    [Path(value).name for value in summary_frame["relative_path"]],
    rotation=35,
    ha="right",
)
axis.set_ylabel("Count")
axis.set_title("Task 41B Existing Runner Interface Inventory")
axis.legend()
axis.grid(True, axis="y", alpha=0.25)
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(
        FIGURE_DIR / f"task41b_interface_inventory.{suffix}",
        dpi=300,
        bbox_inches="tight",
    )
plt.close(fig)

print("===== TASK 41B RUNNER INTERFACE AUDIT =====")
print("Branch:", branch)
print("Commit:", commit)
print("Frozen Task 41A commit:", "698bbf2")
print("Smoke panel slot:", "flow_iat_exact")
print("Smoke candidate:", flow_candidate_id)
print("Smoke poison fraction:", 0.01)
print("Smoke seed:", 7)
print()

print("===== SOURCE INTERFACE SUMMARY =====")
print(summary_frame.to_string(index=False))
print()

argument_frame = pd.DataFrame(argument_rows)
interesting_flags = argument_frame[
    argument_frame["flags"].str.contains(
        "data-file|partition-file|clean-seed-dir|warmup-dir|"
        "plain-branch-dir|replacement-policy|attack-type|"
        "poison-fraction|malicious-clients|output-dir|"
        "continuation-rounds|model-seed|attack-seed",
        case=False,
        regex=True,
        na=False,
    )
]
print("===== RELEVANT CLI ARGUMENTS =====")
print(
    interesting_flags[
        [
            "relative_path",
            "flags",
            "required",
            "default",
            "choices",
            "line",
        ]
    ].to_string(index=False)
)
print()

print("===== NPZ HEADER-ONLY INVENTORY =====")
print(pd.DataFrame(npz_rows).to_string(index=False))
print()

artifact_frame = pd.DataFrame(artifact_rows)
print("===== REFERENCE ARTIFACT COUNTS =====")
print(
    artifact_frame.groupby(["reference", "file_type"], dropna=False)
    .size()
    .reset_index(name="count")
    .to_string(index=False)
)
print()

for path in (
    definitions_path,
    arguments_path,
    imports_path,
    keywords_path,
    source_summary_path,
    npz_path,
    artifacts_path,
    contract_path,
    FIGURE_DIR / "task41b_interface_inventory.png",
    FIGURE_DIR / "task41b_interface_inventory.pdf",
):
    print("WROTE:", path)

print("DATA ARRAYS LOADED: False")
print("RESERVED TEST ARRAY HEADERS INSPECTED ONLY: True")
print("FINAL TEST ARRAYS LOADED: False")

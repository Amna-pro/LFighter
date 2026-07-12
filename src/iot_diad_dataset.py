"""Utilities for preparing CIC IoT-DIAD 2024 flow-based data.

This module is intentionally independent from the original LFighter sampling code.
It supports the first clean centralized baseline before federated learning and poisoning.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CLASS_NAMES: Tuple[str, ...] = (
    "Benign",
    "BruteForce",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web-Based",
)

CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

CLASS_ALIASES: Mapping[str, str] = {
    "benign": "Benign",
    "bruteforce": "BruteForce",
    "brute force": "BruteForce",
    "brute_force": "BruteForce",
    "ddos": "DDoS",
    "dos": "DoS",
    "mirai": "Mirai",
    "recon": "Recon",
    "reconnaissance": "Recon",
    "spoofing": "Spoofing",
    "web-based": "Web-Based",
    "web based": "Web-Based",
    "web_based": "Web-Based",
}

# Counts verified in the user's full audit. They are used only to determine
# one-pass sampling probabilities. The script still reports observed samples.
AUDITED_CLASS_ROWS: Mapping[str, int] = {
    "Benign": 398_330,
    "BruteForce": 3_619,
    "DDoS": 3_478_814,
    "DoS": 14_853_092,
    "Mirai": 174_588,
    "Recon": 442_158,
    "Spoofing": 157_238,
    "Web-Based": 11_328,
}

DEFAULT_CLASS_CAPS: Mapping[str, int] = {
    "Benign": 50_000,
    "BruteForce": 3_619,
    "DDoS": 50_000,
    "DoS": 50_000,
    "Mirai": 50_000,
    "Recon": 50_000,
    "Spoofing": 50_000,
    "Web-Based": 11_328,
}

DROP_COLUMNS: Tuple[str, ...] = (
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Label",
)


@dataclass(frozen=True)
class PreparedPaths:
    root: Path
    train_npz: Path
    val_npz: Path
    test_npz: Path
    preprocessor_joblib: Path
    metadata_json: Path


def normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def infer_broad_class(csv_path: Path, dataset_root: Path) -> str:
    """Infer the broad class from parent folder names or the file name."""
    try:
        parts = csv_path.relative_to(dataset_root).parts[:-1]
    except ValueError:
        parts = csv_path.parts[:-1]

    for part in parts:
        key = normalize_name(part)
        if key in CLASS_ALIASES:
            return CLASS_ALIASES[key]

    file_key = normalize_name(csv_path.stem)
    for key, label in CLASS_ALIASES.items():
        if key in file_key:
            return label
    raise ValueError(f"Could not infer class for file: {csv_path}")


def list_csvs(dataset_root: Path) -> List[Path]:
    files = sorted(dataset_root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {dataset_root}")
    return files


def find_canonical_columns(csv_files: Iterable[Path]) -> List[str]:
    for path in csv_files:
        try:
            columns = [str(c).strip() for c in pd.read_csv(path, nrows=0).columns]
        except Exception:
            continue
        required = {"Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"}
        if required.issubset(columns):
            return columns
    raise RuntimeError("Could not find a valid 84-column CSV header.")


def has_valid_header(csv_path: Path) -> bool:
    try:
        columns = {str(c).strip() for c in pd.read_csv(csv_path, nrows=0).columns}
    except Exception:
        return False
    return {"Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"}.issubset(columns)


def stable_seed(base_seed: int, path: Path, chunk_index: int) -> int:
    digest = hashlib.sha256(f"{base_seed}|{path}|{chunk_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def sample_dataset(
    dataset_root: Path,
    class_caps: Mapping[str, int],
    seed: int,
    chunksize: int,
    oversample_factor: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Read all CSV files once and retain a reproducible class-capped sample.

    Sampling uses a fixed Bernoulli probability per class, with an oversampling
    margin. The final sample is trimmed to the exact cap after cleaning.
    """
    csv_files = list_csvs(dataset_root)
    canonical_columns = find_canonical_columns(csv_files)
    feature_columns = [c for c in canonical_columns if c not in DROP_COLUMNS]

    sampled_parts: Dict[str, List[pd.DataFrame]] = {name: [] for name in CLASS_NAMES}
    read_rows: Dict[str, int] = {name: 0 for name in CLASS_NAMES}
    sampled_before_clean: Dict[str, int] = {name: 0 for name in CLASS_NAMES}
    recovered_headerless: List[str] = []
    skipped_files: List[Dict[str, str]] = []

    probabilities: Dict[str, float] = {}
    for class_name in CLASS_NAMES:
        cap = int(class_caps[class_name])
        total = int(AUDITED_CLASS_ROWS[class_name])
        probabilities[class_name] = min(1.0, oversample_factor * cap / max(total, 1))

    for file_index, csv_path in enumerate(csv_files, start=1):
        class_name = infer_broad_class(csv_path, dataset_root)
        probability = probabilities[class_name]
        relative = str(csv_path.relative_to(dataset_root))
        print(f"[{file_index}/{len(csv_files)}] {class_name}: {relative}")

        try:
            if has_valid_header(csv_path):
                reader = pd.read_csv(
                    csv_path,
                    usecols=feature_columns,
                    chunksize=chunksize,
                    low_memory=False,
                    on_bad_lines="warn",
                )
            else:
                recovered_headerless.append(relative)
                reader = pd.read_csv(
                    csv_path,
                    header=None,
                    names=canonical_columns,
                    usecols=feature_columns,
                    chunksize=chunksize,
                    low_memory=False,
                    on_bad_lines="warn",
                )

            for chunk_index, chunk in enumerate(reader):
                chunk.columns = [str(c).strip() for c in chunk.columns]
                read_rows[class_name] += len(chunk)

                if probability >= 1.0:
                    selected = chunk
                else:
                    rng = np.random.default_rng(stable_seed(seed, csv_path, chunk_index))
                    mask = rng.random(len(chunk)) < probability
                    if len(chunk) > 0 and not bool(mask.any()):
                        mask[int(rng.integers(0, len(chunk)))] = True
                    selected = chunk.loc[mask]

                if not selected.empty:
                    selected = selected.copy()
                    selected["target"] = CLASS_TO_ID[class_name]
                    sampled_parts[class_name].append(selected)
                    sampled_before_clean[class_name] += len(selected)
        except Exception as exc:
            skipped_files.append({"file": relative, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    if skipped_files:
        raise RuntimeError(
            "One or more CSV files could not be read. See error details: "
            + json.dumps(skipped_files[:5], indent=2)
        )

    final_parts: List[pd.DataFrame] = []
    final_counts: Dict[str, int] = {}
    dropped_duplicates: Dict[str, int] = {}
    invalid_numeric_cells: Dict[str, int] = {}

    for class_name in CLASS_NAMES:
        if not sampled_parts[class_name]:
            raise RuntimeError(f"No rows sampled for class {class_name}")

        class_df = pd.concat(sampled_parts[class_name], ignore_index=True)
        target = class_df.pop("target").astype(np.int64)

        for column in class_df.columns:
            class_df[column] = pd.to_numeric(class_df[column], errors="coerce")

        numeric = class_df.to_numpy(dtype=np.float64, copy=False)
        invalid_numeric_cells[class_name] = int((~np.isfinite(numeric)).sum())
        class_df = class_df.replace([np.inf, -np.inf], np.nan)
        class_df["target"] = target.values

        before = len(class_df)
        class_df = class_df.drop_duplicates(ignore_index=True)
        dropped_duplicates[class_name] = before - len(class_df)

        cap = int(class_caps[class_name])
        if len(class_df) > cap:
            class_df = class_df.sample(n=cap, random_state=seed).reset_index(drop=True)

        final_counts[class_name] = len(class_df)
        final_parts.append(class_df)

    combined = pd.concat(final_parts, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    report: Dict[str, object] = {
        "csv_files": len(csv_files),
        "canonical_column_count": len(canonical_columns),
        "raw_feature_count": len(feature_columns),
        "read_rows": read_rows,
        "sampling_probabilities": probabilities,
        "sampled_before_clean": sampled_before_clean,
        "dropped_duplicates": dropped_duplicates,
        "invalid_numeric_cells": invalid_numeric_cells,
        "final_counts": final_counts,
        "headerless_files_recovered": recovered_headerless,
        "total_prepared_rows": len(combined),
    }
    return combined, report


def stratified_split_and_transform(
    dataframe: pd.DataFrame,
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object], Dict[str, object]]:
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be positive and sum to less than 1")

    y = dataframe["target"].to_numpy(dtype=np.int64)
    X = dataframe.drop(columns=["target"])
    raw_feature_names = list(X.columns)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size + val_size,
        random_state=seed,
        stratify=y,
    )
    relative_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test,
        random_state=seed,
        stratify=y_temp,
    )

    imputer = SimpleImputer(strategy="median")
    selector = VarianceThreshold(threshold=0.0)
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_train_sel = selector.fit_transform(X_train_imp)
    X_train_out = scaler.fit_transform(X_train_sel).astype(np.float32)

    X_val_out = scaler.transform(selector.transform(imputer.transform(X_val))).astype(np.float32)
    X_test_out = scaler.transform(selector.transform(imputer.transform(X_test))).astype(np.float32)

    selected_mask = selector.get_support()
    selected_feature_names = [
        name for name, keep in zip(raw_feature_names, selected_mask) if bool(keep)
    ]

    arrays = {
        "X_train": X_train_out,
        "y_train": y_train.astype(np.int64),
        "X_val": X_val_out,
        "y_val": y_val.astype(np.int64),
        "X_test": X_test_out,
        "y_test": y_test.astype(np.int64),
    }
    preprocessors: Dict[str, object] = {
        "imputer": imputer,
        "variance_selector": selector,
        "scaler": scaler,
    }
    metadata: Dict[str, object] = {
        "raw_feature_names": raw_feature_names,
        "selected_feature_names": selected_feature_names,
        "raw_feature_count": len(raw_feature_names),
        "selected_feature_count": len(selected_feature_names),
        "train_rows": len(y_train),
        "val_rows": len(y_val),
        "test_rows": len(y_test),
        "split_class_counts": {
            "train": np.bincount(y_train, minlength=len(CLASS_NAMES)).tolist(),
            "val": np.bincount(y_val, minlength=len(CLASS_NAMES)).tolist(),
            "test": np.bincount(y_test, minlength=len(CLASS_NAMES)).tolist(),
        },
    }
    return arrays, preprocessors, metadata


def save_prepared_data(
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    preprocessors: Mapping[str, object],
    metadata: Mapping[str, object],
) -> PreparedPaths:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_npz = output_dir / "train.npz"
    val_npz = output_dir / "val.npz"
    test_npz = output_dir / "test.npz"
    preprocessor_joblib = output_dir / "preprocessor.joblib"
    metadata_json = output_dir / "metadata.json"

    np.savez_compressed(train_npz, X=arrays["X_train"], y=arrays["y_train"])
    np.savez_compressed(val_npz, X=arrays["X_val"], y=arrays["y_val"])
    np.savez_compressed(test_npz, X=arrays["X_test"], y=arrays["y_test"])
    joblib.dump(dict(preprocessors), preprocessor_joblib)

    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, indent=2)

    return PreparedPaths(
        root=output_dir,
        train_npz=train_npz,
        val_npz=val_npz,
        test_npz=test_npz,
        preprocessor_joblib=preprocessor_joblib,
        metadata_json=metadata_json,
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

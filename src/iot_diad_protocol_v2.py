"""Leakage-aware data protocol for CIC IoT-DIAD 2024.

The module implements a reproducible two-pass preparation pipeline:
1. Inventory every capture file and build a group-aware split manifest.
2. Sample each split without crossing capture boundaries where possible.
3. Use chronological row blocks when a class has fewer than three captures.
4. Remove exact feature duplicates across train, validation, and test.
5. Fit all preprocessing objects on training data only.

The original CSV files are never modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
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
CLASS_TO_ID: Dict[str, int] = {name: index for index, name in enumerate(CLASS_NAMES)}

CLASS_ALIASES: Dict[str, str] = {
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

IDENTIFIER_COLUMNS: Tuple[str, ...] = (
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Label",
)
NETWORK_METADATA_COLUMNS: Tuple[str, ...] = (
    "Src Port",
    "Dst Port",
    "Protocol",
)
AUXILIARY_COLUMNS: Tuple[str, ...] = (
    "class_name",
    "target",
    "fine_label",
    "source_file",
    "source_row",
    "row_hash",
)


@dataclass(frozen=True)
class ProtocolPaths:
    root: Path
    samples_dir: Path
    arrays_dir: Path
    preprocessors_dir: Path
    tables_dir: Path
    figures_dir: Path
    manifest_csv: Path
    metadata_json: Path


def make_protocol_paths(root: Path) -> ProtocolPaths:
    root = root.expanduser().resolve()
    samples = root / "samples"
    arrays = root / "arrays"
    preprocessors = root / "preprocessors"
    tables = root / "tables"
    figures = root / "figures"
    for directory in (root, samples, arrays, preprocessors, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)
    return ProtocolPaths(
        root=root,
        samples_dir=samples,
        arrays_dir=arrays,
        preprocessors_dir=preprocessors,
        tables_dir=tables,
        figures_dir=figures,
        manifest_csv=tables / "capture_split_manifest.csv",
        metadata_json=root / "metadata_v2.json",
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def infer_broad_class(csv_path: Path, dataset_root: Path) -> str:
    relative_parts = csv_path.relative_to(dataset_root).parts[:-1]
    for part in relative_parts:
        key = normalize_text(part)
        if key in CLASS_ALIASES:
            return CLASS_ALIASES[key]
    filename_key = normalize_text(csv_path.stem)
    for key, label in CLASS_ALIASES.items():
        if key in filename_key:
            return label
    raise ValueError(f"Cannot infer broad class for {csv_path}")


def infer_fine_label(csv_path: Path, dataset_root: Path, broad_label: str) -> str:
    relative_parts = list(csv_path.relative_to(dataset_root).parts[:-1])
    ignored = {normalize_text(key) for key in CLASS_ALIASES}
    for part in reversed(relative_parts):
        if normalize_text(part) not in ignored:
            return part
    stem = csv_path.stem.replace(".pcap_Flow", "").replace("_Flow", "")
    return stem or broad_label


def list_csv_files(dataset_root: Path) -> List[Path]:
    files = sorted(dataset_root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {dataset_root}")
    return files


def find_canonical_columns(csv_files: Sequence[Path]) -> List[str]:
    for csv_path in csv_files:
        try:
            columns = [str(column).strip() for column in pd.read_csv(csv_path, nrows=0).columns]
        except Exception:
            continue
        required = {"Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"}
        if required.issubset(columns):
            return columns
    raise RuntimeError("No CSV with the canonical CICFlowMeter header was found")


def has_valid_header(csv_path: Path) -> bool:
    try:
        columns = {str(column).strip() for column in pd.read_csv(csv_path, nrows=0).columns}
    except Exception:
        return False
    return {"Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"}.issubset(columns)


def count_csv_rows(csv_path: Path, has_header: bool, block_size: int = 8 * 1024 * 1024) -> int:
    line_count = 0
    with csv_path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            line_count += block.count(b"\n")
    if csv_path.stat().st_size > 0:
        with csv_path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                line_count += 1
    return max(0, line_count - (1 if has_header else 0))


def stable_uint32(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def build_capture_inventory(dataset_root: Path) -> Tuple[pd.DataFrame, List[str]]:
    csv_files = list_csv_files(dataset_root)
    canonical_columns = find_canonical_columns(csv_files)
    records: List[Dict[str, object]] = []
    for index, csv_path in enumerate(csv_files, start=1):
        broad = infer_broad_class(csv_path, dataset_root)
        fine = infer_fine_label(csv_path, dataset_root, broad)
        valid_header = has_valid_header(csv_path)
        rows = count_csv_rows(csv_path, valid_header)
        records.append(
            {
                "file_index": index,
                "relative_path": str(csv_path.relative_to(dataset_root)),
                "absolute_path": str(csv_path),
                "class_name": broad,
                "fine_label": fine,
                "row_count": rows,
                "size_mb": round(csv_path.stat().st_size / (1024 * 1024), 4),
                "has_header": valid_header,
            }
        )
    inventory = pd.DataFrame(records)
    return inventory, canonical_columns


def _ensure_nonempty_file_splits(assignments: MutableMapping[int, str], group: pd.DataFrame) -> None:
    split_names = ("train", "val", "test")
    present = set(assignments.values())
    for missing in split_names:
        if missing in present:
            continue
        donor_counts = pd.Series(assignments).value_counts()
        donor = str(donor_counts.index[0])
        donor_indices = [idx for idx, split in assignments.items() if split == donor]
        smallest_idx = min(donor_indices, key=lambda idx: int(group.loc[idx, "row_count"]))
        assignments[smallest_idx] = missing
        present.add(missing)


def assign_capture_splits(
    inventory: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> pd.DataFrame:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-9):
        raise ValueError("train, validation, and test ratios must sum to 1")

    manifest = inventory.copy()
    manifest["split_mode"] = ""
    manifest["assigned_split"] = ""
    manifest["train_end_row"] = -1
    manifest["val_end_row"] = -1

    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}

    for class_name, group in manifest.groupby("class_name", sort=False):
        indices = list(group.index)
        if len(indices) >= 3:
            target_rows = {
                split: float(group["row_count"].sum()) * ratio
                for split, ratio in ratios.items()
            }
            current_rows = {split: 0.0 for split in ratios}
            assignments: Dict[int, str] = {}
            ordered = sorted(
                indices,
                key=lambda idx: (
                    -int(group.loc[idx, "row_count"]),
                    stable_uint32(seed, class_name, group.loc[idx, "relative_path"]),
                ),
            )
            for idx in ordered:
                rows = float(group.loc[idx, "row_count"])
                split = min(
                    ratios,
                    key=lambda candidate: (
                        (current_rows[candidate] + rows) / max(target_rows[candidate], 1.0),
                        stable_uint32(seed, class_name, candidate, group.loc[idx, "relative_path"]),
                    ),
                )
                assignments[idx] = split
                current_rows[split] += rows

            _ensure_nonempty_file_splits(assignments, group)
            for idx, split in assignments.items():
                manifest.loc[idx, "split_mode"] = "capture_held_out"
                manifest.loc[idx, "assigned_split"] = split
        else:
            for idx in indices:
                rows = int(manifest.loc[idx, "row_count"])
                train_end = int(math.floor(rows * train_ratio))
                val_end = int(math.floor(rows * (train_ratio + val_ratio)))
                train_end = min(max(train_end, 1), max(rows - 2, 1)) if rows >= 3 else rows
                val_end = min(max(val_end, train_end + 1), max(rows - 1, train_end)) if rows >= 3 else rows
                manifest.loc[idx, "split_mode"] = "chronological_within_capture"
                manifest.loc[idx, "assigned_split"] = "mixed"
                manifest.loc[idx, "train_end_row"] = train_end
                manifest.loc[idx, "val_end_row"] = val_end

    return manifest


def raw_split_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for record in manifest.to_dict(orient="records"):
        total = int(record["row_count"])
        if record["split_mode"] == "capture_held_out":
            counts = {"train": 0, "val": 0, "test": 0}
            counts[str(record["assigned_split"])] = total
        else:
            train_end = int(record["train_end_row"])
            val_end = int(record["val_end_row"])
            counts = {
                "train": train_end,
                "val": max(0, val_end - train_end),
                "test": max(0, total - val_end),
            }
        for split, count in counts.items():
            rows.append(
                {
                    "class_name": record["class_name"],
                    "split": split,
                    "rows": int(count),
                }
            )
    result = pd.DataFrame(rows).groupby(["class_name", "split"], as_index=False)["rows"].sum()
    return result


def _read_csv_chunks(
    csv_path: Path,
    canonical_columns: Sequence[str],
    chunksize: int,
    valid_header: bool,
) -> Iterator[pd.DataFrame]:
    kwargs = dict(chunksize=chunksize, low_memory=False, on_bad_lines="warn")
    if valid_header:
        reader = pd.read_csv(csv_path, **kwargs)
    else:
        reader = pd.read_csv(csv_path, header=None, names=list(canonical_columns), **kwargs)
    for chunk in reader:
        chunk.columns = [str(column).strip() for column in chunk.columns]
        yield chunk


def _sampling_probability(target: int, available: int, margin: float) -> float:
    if available <= 0:
        return 0.0
    return min(1.0, max(0.0, margin * target / available))


def _sample_chunk(
    chunk: pd.DataFrame,
    probability: float,
    seed: int,
) -> pd.DataFrame:
    if chunk.empty or probability <= 0:
        return chunk.iloc[0:0].copy()
    if probability >= 1:
        return chunk.copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(len(chunk)) < probability
    return chunk.loc[mask].copy()


def _split_chunk_by_row_position(
    chunk: pd.DataFrame,
    chunk_start: int,
    train_end: int,
    val_end: int,
) -> Dict[str, pd.DataFrame]:
    positions = np.arange(chunk_start, chunk_start + len(chunk))
    return {
        "train": chunk.loc[positions < train_end],
        "val": chunk.loc[(positions >= train_end) & (positions < val_end)],
        "test": chunk.loc[positions >= val_end],
    }


def collect_protocol_samples(
    dataset_root: Path,
    manifest: pd.DataFrame,
    canonical_columns: Sequence[str],
    raw_counts: pd.DataFrame,
    train_cap_per_class: int,
    val_cap_per_class: int,
    diagnostic_test_cap_per_class: int,
    natural_test_total: int,
    seed: int,
    chunksize: int,
    sampling_margin: float = 1.30,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, object]]:
    raw_lookup = {
        (str(row.class_name), str(row.split)): int(row.rows)
        for row in raw_counts.itertuples(index=False)
    }
    target_caps = {
        "train": train_cap_per_class,
        "val": val_cap_per_class,
        "test_diagnostic": diagnostic_test_cap_per_class,
    }

    parts: Dict[Tuple[str, str], List[pd.DataFrame]] = defaultdict(list)
    natural_parts: List[pd.DataFrame] = []
    sampled_before_clean: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    read_rows: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    raw_test_total = int(raw_counts.loc[raw_counts["split"] == "test", "rows"].sum())
    natural_probability = _sampling_probability(natural_test_total, raw_test_total, sampling_margin)

    model_input_columns = [column for column in canonical_columns if column != "Label"]

    for file_number, record in enumerate(manifest.to_dict(orient="records"), start=1):
        csv_path = Path(str(record["absolute_path"]))
        class_name = str(record["class_name"])
        fine_label = str(record["fine_label"])
        relative_path = str(record["relative_path"])
        valid_header = bool(record["has_header"])
        mode = str(record["split_mode"])
        print(f"[{file_number}/{len(manifest)}] {class_name}: {relative_path}")
        row_offset = 0

        for chunk_index, chunk in enumerate(
            _read_csv_chunks(csv_path, canonical_columns, chunksize, valid_header)
        ):
            chunk = chunk.reindex(columns=model_input_columns)
            chunk["class_name"] = class_name
            chunk["target"] = CLASS_TO_ID[class_name]
            chunk["fine_label"] = fine_label
            chunk["source_file"] = relative_path
            chunk["source_row"] = np.arange(row_offset, row_offset + len(chunk), dtype=np.int64)

            if mode == "capture_held_out":
                split_chunks = {str(record["assigned_split"]): chunk}
            else:
                split_chunks = _split_chunk_by_row_position(
                    chunk,
                    row_offset,
                    int(record["train_end_row"]),
                    int(record["val_end_row"]),
                )

            for split, split_chunk in split_chunks.items():
                if split_chunk.empty:
                    continue
                read_rows[class_name][split] += len(split_chunk)
                if split in {"train", "val"}:
                    cap = target_caps[split]
                    probability = _sampling_probability(
                        cap,
                        raw_lookup.get((class_name, split), len(split_chunk)),
                        sampling_margin,
                    )
                    selected = _sample_chunk(
                        split_chunk,
                        probability,
                        stable_uint32(seed, relative_path, chunk_index, class_name, split),
                    )
                    if not selected.empty:
                        parts[(split, class_name)].append(selected)
                        sampled_before_clean[class_name][split] += len(selected)
                elif split == "test":
                    probability = _sampling_probability(
                        diagnostic_test_cap_per_class,
                        raw_lookup.get((class_name, "test"), len(split_chunk)),
                        sampling_margin,
                    )
                    selected = _sample_chunk(
                        split_chunk,
                        probability,
                        stable_uint32(seed, relative_path, chunk_index, class_name, "test_diagnostic"),
                    )
                    if not selected.empty:
                        parts[("test_diagnostic", class_name)].append(selected)
                        sampled_before_clean[class_name]["test_diagnostic"] += len(selected)

                    natural_selected = _sample_chunk(
                        split_chunk,
                        natural_probability,
                        stable_uint32(seed, relative_path, chunk_index, class_name, "test_natural"),
                    )
                    if not natural_selected.empty:
                        natural_parts.append(natural_selected)
                        sampled_before_clean[class_name]["test_natural"] += len(natural_selected)

            row_offset += len(chunk)

    datasets: Dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test_diagnostic"):
        split_parts: List[pd.DataFrame] = []
        for class_name in CLASS_NAMES:
            class_parts = parts.get((split, class_name), [])
            if not class_parts:
                raise RuntimeError(f"No sampled rows for {class_name} in {split}")
            class_df = pd.concat(class_parts, ignore_index=True)
            cap = target_caps[split]
            if len(class_df) > cap:
                class_df = class_df.sample(n=cap, random_state=stable_uint32(seed, split, class_name))
            split_parts.append(class_df)
        datasets[split] = pd.concat(split_parts, ignore_index=True)

    if not natural_parts:
        raise RuntimeError("No rows were sampled for the natural test set")
    natural_df = pd.concat(natural_parts, ignore_index=True)
    if len(natural_df) > natural_test_total:
        natural_df = natural_df.sample(n=natural_test_total, random_state=stable_uint32(seed, "natural"))
    datasets["test_natural"] = natural_df

    report = {
        "read_rows": {key: dict(value) for key, value in read_rows.items()},
        "sampled_before_clean": {
            key: dict(value) for key, value in sampled_before_clean.items()
        },
        "natural_sampling_probability": natural_probability,
    }
    return datasets, report


def numeric_feature_columns(canonical_columns: Sequence[str], strict_behavioral: bool) -> List[str]:
    dropped = set(IDENTIFIER_COLUMNS)
    if strict_behavioral:
        dropped.update(NETWORK_METADATA_COLUMNS)
    return [column for column in canonical_columns if column not in dropped]


def _coerce_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out[list(columns)] = out[list(columns)].replace([np.inf, -np.inf], np.nan)
    return out


def _feature_hash(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    normalized = frame[list(columns)].copy()
    normalized = normalized.fillna(-1.7976931348623157e308)
    return pd.util.hash_pandas_object(normalized, index=False).astype("uint64")


def clean_and_deduplicate_samples(
    datasets: Mapping[str, pd.DataFrame],
    canonical_columns: Sequence[str],
    seed: int,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Clean samples while enforcing leakage rules between model-development splits.

    Train and validation must be disjoint. Both test views must be disjoint from
    train and validation. The diagnostic and natural test views are alternative
    evaluations drawn from the same held-out capture pool, so overlap between
    those two test views is allowed and is reported rather than removed.
    """
    full_columns = numeric_feature_columns(canonical_columns, strict_behavioral=False)
    cleaned: Dict[str, pd.DataFrame] = {}
    quality_rows: List[Dict[str, object]] = []
    overlap_rows: List[Dict[str, object]] = []

    protected_hashes: set[int] = set()

    for split in ("train", "val"):
        frame = _coerce_numeric(datasets[split], full_columns)
        invalid = int(frame[full_columns].isna().sum().sum())
        before = len(frame)
        frame["row_hash"] = _feature_hash(frame, full_columns)
        within_duplicates = int(frame["row_hash"].duplicated().sum())
        frame = frame.drop_duplicates(subset=["row_hash"], keep="first")

        cross_mask = frame["row_hash"].isin(protected_hashes)
        cross_duplicates = int(cross_mask.sum())
        frame = frame.loc[~cross_mask].copy()

        protected_hashes.update(int(value) for value in frame["row_hash"].tolist())
        frame = frame.sample(
            frac=1.0,
            random_state=stable_uint32(seed, split),
        ).reset_index(drop=True)
        cleaned[split] = frame
        quality_rows.append(
            {
                "split": split,
                "rows_before_clean": before,
                "rows_after_clean": len(frame),
                "invalid_numeric_cells": invalid,
                "within_split_duplicates_removed": within_duplicates,
                "cross_split_duplicates_removed": cross_duplicates,
                "cross_split_reference": "earlier development splits",
            }
        )

    diagnostic = _coerce_numeric(datasets["test_diagnostic"], full_columns)
    invalid = int(diagnostic[full_columns].isna().sum().sum())
    before = len(diagnostic)
    diagnostic["row_hash"] = _feature_hash(diagnostic, full_columns)
    within_duplicates = int(diagnostic["row_hash"].duplicated().sum())
    diagnostic = diagnostic.drop_duplicates(subset=["row_hash"], keep="first")
    cross_mask = diagnostic["row_hash"].isin(protected_hashes)
    cross_duplicates = int(cross_mask.sum())
    diagnostic = diagnostic.loc[~cross_mask].copy()
    diagnostic = diagnostic.sample(
        frac=1.0,
        random_state=stable_uint32(seed, "test_diagnostic"),
    ).reset_index(drop=True)
    cleaned["test_diagnostic"] = diagnostic
    quality_rows.append(
        {
            "split": "test_diagnostic",
            "rows_before_clean": before,
            "rows_after_clean": len(diagnostic),
            "invalid_numeric_cells": invalid,
            "within_split_duplicates_removed": within_duplicates,
            "cross_split_duplicates_removed": cross_duplicates,
            "cross_split_reference": "train and validation only",
        }
    )

    natural = _coerce_numeric(datasets["test_natural"], full_columns)
    invalid = int(natural[full_columns].isna().sum().sum())
    before = len(natural)
    natural["row_hash"] = _feature_hash(natural, full_columns)
    within_duplicates = int(natural["row_hash"].duplicated().sum())
    natural = natural.drop_duplicates(subset=["row_hash"], keep="first")
    cross_mask = natural["row_hash"].isin(protected_hashes)
    cross_duplicates = int(cross_mask.sum())
    natural = natural.loc[~cross_mask].copy()
    natural = natural.sample(
        frac=1.0,
        random_state=stable_uint32(seed, "test_natural"),
    ).reset_index(drop=True)
    cleaned["test_natural"] = natural
    quality_rows.append(
        {
            "split": "test_natural",
            "rows_before_clean": before,
            "rows_after_clean": len(natural),
            "invalid_numeric_cells": invalid,
            "within_split_duplicates_removed": within_duplicates,
            "cross_split_duplicates_removed": cross_duplicates,
            "cross_split_reference": "train and validation only",
        }
    )

    split_names = list(cleaned)
    for left_index, left in enumerate(split_names):
        left_hashes = set(int(value) for value in cleaned[left]["row_hash"].tolist())
        for right in split_names[left_index + 1 :]:
            right_hashes = set(int(value) for value in cleaned[right]["row_hash"].tolist())
            allowed = {left, right} == {"test_natural", "test_diagnostic"}
            overlap_rows.append(
                {
                    "split_a": left,
                    "split_b": right,
                    "overlap_rows": len(left_hashes.intersection(right_hashes)),
                    "allowed_overlap": allowed,
                }
            )

    quality = pd.DataFrame(quality_rows)
    overlap = pd.DataFrame(overlap_rows)
    return cleaned, quality, overlap

def save_sample_parquets(datasets: Mapping[str, pd.DataFrame], paths: ProtocolPaths) -> Dict[str, str]:
    saved: Dict[str, str] = {}
    for split, frame in datasets.items():
        parquet_path = paths.samples_dir / f"{split}.parquet"
        try:
            frame.to_parquet(parquet_path, index=False, compression="zstd")
            saved[split] = str(parquet_path)
        except ImportError:
            fallback_path = paths.samples_dir / f"{split}.pkl.gz"
            frame.to_pickle(fallback_path, compression="gzip")
            saved[split] = str(fallback_path)
    return saved


def fit_and_save_feature_configurations(
    datasets: Mapping[str, pd.DataFrame],
    canonical_columns: Sequence[str],
    paths: ProtocolPaths,
    compact_k: int,
) -> Dict[str, Dict[str, object]]:
    configurations = {
        "full_flow": numeric_feature_columns(canonical_columns, strict_behavioral=False),
        "behavioral_only": numeric_feature_columns(canonical_columns, strict_behavioral=True),
    }
    result: Dict[str, Dict[str, object]] = {}

    for config_name, raw_features in configurations.items():
        config_dir = paths.preprocessors_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        imputer = SimpleImputer(strategy="median")
        selector = VarianceThreshold(threshold=0.0)
        scaler = StandardScaler()

        train_raw = datasets["train"][raw_features]
        train_imp = imputer.fit_transform(train_raw)
        train_var = selector.fit_transform(train_imp)
        train_out = scaler.fit_transform(train_var).astype(np.float32)
        selected_features = [
            feature for feature, keep in zip(raw_features, selector.get_support()) if bool(keep)
        ]

        arrays: Dict[str, np.ndarray] = {"X_train": train_out}
        for split in ("val", "test_natural", "test_diagnostic"):
            transformed = scaler.transform(
                selector.transform(imputer.transform(datasets[split][raw_features]))
            ).astype(np.float32)
            arrays[f"X_{split}"] = transformed

        for split in ("train", "val", "test_natural", "test_diagnostic"):
            arrays[f"y_{split}"] = datasets[split]["target"].to_numpy(dtype=np.int64)

        npz_path = paths.arrays_dir / f"{config_name}.npz"
        np.savez_compressed(npz_path, **arrays)
        joblib.dump(imputer, config_dir / "imputer.joblib")
        joblib.dump(selector, config_dir / "variance_selector.joblib")
        joblib.dump(scaler, config_dir / "scaler.joblib")
        pd.DataFrame({"feature": selected_features}).to_csv(
            config_dir / "selected_features.csv", index=False
        )
        result[config_name] = {
            "raw_feature_count": len(raw_features),
            "selected_feature_count": len(selected_features),
            "selected_features": selected_features,
            "arrays": str(npz_path),
        }

    base = configurations["behavioral_only"]
    compact_dir = paths.preprocessors_dir / "compact_k"
    compact_dir.mkdir(parents=True, exist_ok=True)
    imputer = SimpleImputer(strategy="median")
    variance = VarianceThreshold(threshold=0.0)
    train_imp = imputer.fit_transform(datasets["train"][base])
    train_var = variance.fit_transform(train_imp)
    variance_features = [
        feature for feature, keep in zip(base, variance.get_support()) if bool(keep)
    ]
    k = min(compact_k, len(variance_features))
    selector = SelectKBest(score_func=f_classif, k=k)
    train_selected = selector.fit_transform(train_var, datasets["train"]["target"].to_numpy())
    scaler = StandardScaler()
    train_out = scaler.fit_transform(train_selected).astype(np.float32)
    selected_features = [
        feature for feature, keep in zip(variance_features, selector.get_support()) if bool(keep)
    ]
    arrays = {"X_train": train_out}
    for split in ("val", "test_natural", "test_diagnostic"):
        transformed = scaler.transform(
            selector.transform(variance.transform(imputer.transform(datasets[split][base])))
        ).astype(np.float32)
        arrays[f"X_{split}"] = transformed
    for split in ("train", "val", "test_natural", "test_diagnostic"):
        arrays[f"y_{split}"] = datasets[split]["target"].to_numpy(dtype=np.int64)

    compact_npz = paths.arrays_dir / f"compact_k{k}.npz"
    np.savez_compressed(compact_npz, **arrays)
    joblib.dump(imputer, compact_dir / "imputer.joblib")
    joblib.dump(variance, compact_dir / "variance_selector.joblib")
    joblib.dump(selector, compact_dir / "anova_selector.joblib")
    joblib.dump(scaler, compact_dir / "scaler.joblib")
    scores = pd.DataFrame(
        {
            "feature": variance_features,
            "f_score": selector.scores_,
            "p_value": selector.pvalues_,
            "selected": selector.get_support(),
        }
    ).sort_values("f_score", ascending=False)
    scores.to_csv(compact_dir / "feature_selection_scores.csv", index=False)
    pd.DataFrame({"feature": selected_features}).to_csv(
        compact_dir / "selected_features.csv", index=False
    )
    result[f"compact_k{k}"] = {
        "raw_feature_count": len(base),
        "variance_feature_count": len(variance_features),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "arrays": str(compact_npz),
    }
    return result


def class_distribution_table(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split, frame in datasets.items():
        counts = frame["class_name"].value_counts()
        for class_name in CLASS_NAMES:
            count = int(counts.get(class_name, 0))
            rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "class_id": CLASS_TO_ID[class_name],
                    "rows": count,
                    "proportion": count / max(len(frame), 1),
                }
            )
    return pd.DataFrame(rows)


def missingness_table(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    missing = frame[list(feature_columns)].isna().sum()
    result = pd.DataFrame(
        {
            "feature": missing.index,
            "missing_cells": missing.values.astype(np.int64),
            "missing_rate": missing.values / max(len(frame), 1),
        }
    )
    return result.sort_values(["missing_cells", "feature"], ascending=[False, True])


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

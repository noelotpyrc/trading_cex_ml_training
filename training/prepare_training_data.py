import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import fnmatch

import numpy as np
import pandas as pd


@dataclass
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    cutoff_dates: Optional[Tuple[Optional[str], Optional[str]]] = None


@dataclass
class PrepMetadata:
    input_path: str
    num_rows_before: int
    num_rows_after: int
    num_features_before: int
    num_features_after: int
    target_column: str
    split_strategy: str
    split_params: Dict[str, str]
    dropped_constant_columns: List[str]
    dropped_na_rows: int
    split_timestamps: Dict[str, List[str]]
    split_timestamp_ranges: Dict[str, Dict[str, Optional[str]]]
    merged_output_csv: Optional[str] = None
    selected_feature_columns: Optional[List[str]] = None
    excluded_feature_columns: Optional[List[str]] = None
    extra_feature_sources: Optional[List[Dict[str, object]]] = None


def _load_merged(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    return df


def _apply_feature_filters(
    df: pd.DataFrame,
    target_col: str,
    include: Optional[Sequence[str]],
    include_patterns: Optional[Sequence[str]],
    exclude: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, List[str]]:
    filtered_cols: List[str]
    if include:
        matches: set[str] = set()
        for pattern in include:
            pattern_matches = fnmatch.filter(df.columns, pattern)
            matches.update(pattern_matches)
        missing = [p for p in include if not fnmatch.filter(df.columns, p)]
        if missing:
            raise KeyError(f"Included feature columns not found for patterns: {missing}")
        filtered_cols = [c for c in df.columns if c == "timestamp" or c == target_col or c in matches]
    else:
        filtered_cols = list(df.columns)

    if include_patterns:
        pattern_matches: set[str] = set()
        for pattern in include_patterns:
            pattern_matches.update(fnmatch.filter(df.columns, pattern))
        filtered_cols.extend([c for c in pattern_matches if c not in filtered_cols or c == target_col])

    # Always drop auxiliary target columns (y_*) except for the chosen target
    filtered_cols = [
        c for c in filtered_cols
        if c == "timestamp" or c == target_col or not c.startswith("y_")
    ]

    if exclude:
        exclude_matches: set[str] = set()
        for pattern in exclude:
            exclude_matches.update(fnmatch.filter(filtered_cols, pattern))
        filtered_cols = [c for c in filtered_cols if c not in exclude_matches or c in ("timestamp", target_col)]

    filtered_df = df.loc[:, filtered_cols]
    selected = [c for c in filtered_cols if c not in {"timestamp", target_col}]
    return filtered_df, selected


def _clean_dataframe(
    df: pd.DataFrame,
    target_col: str,
    include_features: Optional[Sequence[str]] = None,
    include_patterns: Optional[Sequence[str]] = None,
    exclude_features: Optional[Sequence[str]] = None,
    *,
    warmup_rows: int = 0,
) -> Tuple[pd.DataFrame, List[str], int, List[str]]:
    df, selected_columns = _apply_feature_filters(df, target_col, include_features, include_patterns, exclude_features)
    cols_to_numeric = [c for c in df.columns if c not in ('timestamp', target_col)]
    for c in cols_to_numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(axis=1, how='all')

    constant_cols: List[str] = []
    for c in [c for c in df.columns if c not in ('timestamp', target_col)]:
        series = df[c]
        if series.nunique(dropna=True) <= 1:
            constant_cols.append(c)
    if constant_cols:
        df = df.drop(columns=constant_cols)

    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    warmup_rows = max(int(warmup_rows or 0), 0)
    if warmup_rows:
        if len(df) > warmup_rows:
            df = df.iloc[warmup_rows:].reset_index(drop=True)
        else:
            df = df.iloc[0:0].reset_index(drop=True)

    before_rows = len(df)
    # Only enforce non-NA on features (exclude all y_* leakage columns) plus the selected target
    feature_cols_no_y = [c for c in df.columns if c != 'timestamp' and not c.startswith('y_')]
    df = df.dropna(axis=0, how='any', subset=feature_cols_no_y + [target_col])
    dropped_na_rows = before_rows - len(df)

    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    return df, constant_cols, dropped_na_rows, selected_columns


def _time_based_split(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if 'timestamp' not in df.columns:
        n = len(df)
        n_train = int(n * cfg.train_ratio)
        n_val = int(n * cfg.val_ratio)
        train = df.iloc[:n_train]
        val = df.iloc[n_train:n_train + n_val]
        test = df.iloc[n_train + n_val:]
        return train, val, test

    if cfg.cutoff_dates and (cfg.cutoff_dates[0] or cfg.cutoff_dates[1]):
        start, mid = cfg.cutoff_dates
        ts = df['timestamp']
        if start:
            train = df[ts < pd.to_datetime(start)]
            remain = df[ts >= pd.to_datetime(start)]
        else:
            train = pd.DataFrame(columns=df.columns)
            remain = df
        if mid:
            val = remain[remain['timestamp'] < pd.to_datetime(mid)]
            test = remain[remain['timestamp'] >= pd.to_datetime(mid)]
        else:
            n_remain = len(remain)
            n_val = int(n_remain * cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio))
            val = remain.iloc[:n_val]
            test = remain.iloc[n_val:]
        return train, val, test

    n = len(df)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test


def _write_outputs(out_dir: Path, target_col: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        # Exclude timestamp and all y_* columns from X to avoid leakage
        x_cols = [c for c in df.columns if c != 'timestamp' and not c.startswith('y_')]
        X = df[x_cols]
        y = df[target_col].astype(float)
        return X, y

    for name, part in [('train', train), ('val', val), ('test', test)]:
        X, y = split_xy(part)
        X.to_csv(out_dir / f'X_{name}.csv', index=False)
        y.to_csv(out_dir / f'y_{name}.csv', index=False, header=[target_col])


def prepare_splits(
    input_path: Path,
    output_dir: Path,
    target: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    cutoff_start: Optional[str] = None,
    cutoff_mid: Optional[str] = None,
    include_features: Optional[Sequence[str]] = None,
    include_patterns: Optional[Sequence[str]] = None,
    exclude_features: Optional[Sequence[str]] = None,
    extra_feature_files: Optional[Sequence[dict]] = None,
    warmup_rows: int = 0,
) -> Path:
    """Programmatic API to prepare train/val/test splits.

    Returns the final output directory where X_*/y_* and prep_metadata.json are written.
    """
    merged = _load_merged(input_path)
    num_rows_before, num_cols_before = merged.shape

    if target not in merged.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {len(merged.columns)} columns")

    merged_with_extras = merged.copy()
    extra_sources: List[dict] = []
    if extra_feature_files:
        for entry in extra_feature_files:
            extra_df = _load_extra_feature_file(entry)
            merged_with_extras = merged_with_extras.merge(extra_df, on="timestamp", how="left")
            extra_sources.append({
                "path": str(entry.get("path")),
                "include": entry.get("include"),
                "exclude": entry.get("exclude"),
                "added_columns": [c for c in extra_df.columns if c != "timestamp"],
            })

    cleaned, dropped_constants, dropped_na_rows, selected_columns = _clean_dataframe(
        merged_with_extras,
        target_col=target,
        include_features=include_features,
        include_patterns=include_patterns,
        exclude_features=exclude_features,
        warmup_rows=warmup_rows,
    )
    num_rows_after, num_cols_after = cleaned.shape

    split_cfg = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        cutoff_dates=(cutoff_start, cutoff_mid),
    )
    train, val, test = _time_based_split(cleaned, split_cfg)

    final_out_dir = output_dir.parent / f"{output_dir.name}_{target}"

    _write_outputs(final_out_dir, target, train, val, test)

    def _ts_list(df: pd.DataFrame) -> List[str]:
        if 'timestamp' in df.columns:
            return df['timestamp'].astype(str).tolist()
        return []

    def _ts_range(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        if 'timestamp' in df.columns and len(df) > 0:
            return {
                'min': str(df['timestamp'].min()),
                'max': str(df['timestamp'].max()),
            }
        return {'min': None, 'max': None}

    meta = PrepMetadata(
        input_path=str(input_path),
        num_rows_before=num_rows_before,
        num_rows_after=num_rows_after,
        num_features_before=num_cols_before,
        num_features_after=num_cols_after,
        target_column=target,
        split_strategy='cutoff' if (cutoff_start or cutoff_mid) else 'ratio_time_order',
        split_params={
            'train_ratio': str(train_ratio),
            'val_ratio': str(val_ratio),
            'test_ratio': str(test_ratio),
            'cutoff_start': str(cutoff_start),
            'cutoff_mid': str(cutoff_mid),
            'warmup_rows': str(warmup_rows),
        },
        dropped_constant_columns=dropped_constants,
        dropped_na_rows=dropped_na_rows,
        split_timestamps={
            'train': _ts_list(train),
            'val': _ts_list(val),
            'test': _ts_list(test),
        },
        split_timestamp_ranges={
            'train': _ts_range(train),
            'val': _ts_range(val),
            'test': _ts_range(test),
        },
        selected_feature_columns=[c for c in selected_columns if c not in dropped_constants],
        excluded_feature_columns=list(exclude_features or []),
        extra_feature_sources=extra_sources or None,
    )
    with open(final_out_dir / 'prep_metadata.json', 'w') as f:
        json.dump(asdict(meta), f, indent=2, default=str)

    return final_out_dir


def _load_feature_store(features_path: Path, targets_path: Path, target_col: str) -> pd.DataFrame:
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)

    if features.empty or targets.empty:
        raise ValueError("Feature store files must not be empty")
    if "timestamp" not in features.columns or "timestamp" not in targets.columns:
        raise ValueError("Both features and targets files must include a 'timestamp' column")

    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    targets["timestamp"] = pd.to_datetime(targets["timestamp"], utc=True)

    merged = features.merge(targets, on="timestamp", how="inner", suffixes=("", "_target"))
    if target_col not in merged.columns:
        raise ValueError(f"Target column '{target_col}' not found after merging feature store data")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


def _load_extra_feature_file(entry: dict) -> pd.DataFrame:
    path = Path(entry.get("path"))
    if not path.exists():
        raise FileNotFoundError(f"Extra feature file not found: {path}")
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Extra feature file {path} missing 'timestamp' column")
    include = entry.get("include")
    include_patterns = entry.get("include_patterns")
    exclude = entry.get("exclude")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df, _ = _apply_feature_filters(
        df,
        target_col="__dummy__",
        include=include,
        include_patterns=include_patterns,
        exclude=exclude,
    )
    if "__dummy__" in df.columns:
        df = df.drop(columns="__dummy__")
    return df


def prepare_splits_from_feature_store(
    features_csv: Path,
    targets_csv: Path,
    output_dir: Path,
    target: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    cutoff_start: Optional[str] = None,
    cutoff_mid: Optional[str] = None,
    *,
    store_merged_csv: bool = True,
    include_features: Optional[Sequence[str]] = None,
    include_patterns: Optional[Sequence[str]] = None,
    exclude_features: Optional[Sequence[str]] = None,
    extra_feature_files: Optional[Sequence[dict]] = None,
    warmup_rows: int = 0,
) -> Path:
    merged = _load_feature_store(features_csv, targets_csv, target)

    num_rows_before, num_cols_before = merged.shape
    merged_with_extras = merged.copy()
    extra_sources: List[dict] = []
    if extra_feature_files:
        for entry in extra_feature_files:
            extra_df = _load_extra_feature_file(entry)
            merged_with_extras = merged_with_extras.merge(extra_df, on="timestamp", how="left")
            extra_sources.append({
                "path": str(entry.get("path")),
                "include": entry.get("include"),
                "exclude": entry.get("exclude"),
                "added_columns": [c for c in extra_df.columns if c != "timestamp"],
            })

    cleaned, dropped_constants, dropped_na_rows, selected_columns = _clean_dataframe(
        merged_with_extras,
        target_col=target,
        include_features=include_features,
        include_patterns=include_patterns,
        exclude_features=exclude_features,
        warmup_rows=warmup_rows,
    )
    num_rows_after, num_cols_after = cleaned.shape

    split_cfg = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        cutoff_dates=(cutoff_start, cutoff_mid),
    )
    train, val, test = _time_based_split(cleaned, split_cfg)

    final_out_dir = output_dir.parent / f"{output_dir.name}_{target}"

    _write_outputs(final_out_dir, target, train, val, test)

    def _ts_list(df: pd.DataFrame) -> List[str]:
        if "timestamp" in df.columns:
            return df["timestamp"].astype(str).tolist()
        return []

    def _ts_range(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        if "timestamp" in df.columns and len(df) > 0:
            return {
                "min": str(df["timestamp"].min()),
                "max": str(df["timestamp"].max()),
            }
        return {"min": None, "max": None}

    merged_csv_path: Optional[Path] = None
    if store_merged_csv:
        merged_csv_path = final_out_dir / "merged_features_targets.csv"
        merged_with_extras.to_csv(merged_csv_path, index=False)

    meta = PrepMetadata(
        input_path=f"features={features_csv};targets={targets_csv}",
        num_rows_before=num_rows_before,
        num_rows_after=num_rows_after,
        num_features_before=num_cols_before,
        num_features_after=num_cols_after,
        target_column=target,
        split_strategy="feature_store_ratio_time_order" if not (cutoff_start or cutoff_mid) else "feature_store_cutoff",
        split_params={
            "train_ratio": str(train_ratio),
            "val_ratio": str(val_ratio),
            "test_ratio": str(test_ratio),
            "cutoff_start": str(cutoff_start),
            "cutoff_mid": str(cutoff_mid),
            "warmup_rows": str(warmup_rows),
        },
        dropped_constant_columns=dropped_constants,
        dropped_na_rows=dropped_na_rows,
        split_timestamps={
            "train": _ts_list(train),
            "val": _ts_list(val),
            "test": _ts_list(test),
        },
        split_timestamp_ranges={
            "train": _ts_range(train),
            "val": _ts_range(val),
            "test": _ts_range(test),
        },
        merged_output_csv=str(merged_csv_path) if merged_csv_path else None,
        selected_feature_columns=[c for c in selected_columns if c not in dropped_constants],
        excluded_feature_columns=list(exclude_features or []),
        extra_feature_sources=extra_sources or None,
    )
    final_out_dir.mkdir(parents=True, exist_ok=True)
    with open(final_out_dir / "prep_metadata.json", "w") as f:
        json.dump(asdict(meta), f, indent=2, default=str)

    return final_out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare ML training data from merged features-targets')
    parser.add_argument('--input', type=Path, required=True, help='Path to merged_features_targets.csv')
    parser.add_argument('--output-dir', type=Path, required=False, default=Path('/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/prepared'), help='Base output directory for prepared splits (target suffix will be appended)')
    parser.add_argument('--target', type=str, required=False, default='y_logret_24h', help='Target column to predict')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--cutoff-start', type=str, default=None, help='Optional timestamp cutoff for train end (e.g., 2024-12-31)')
    parser.add_argument('--cutoff-mid', type=str, default=None, help='Optional timestamp cutoff to split val/test (e.g., 2025-06-01)')
    parser.add_argument('--warmup-rows', type=int, default=0, help='Drop the first N rows (after sorting by timestamp) before NA cleanup')
    args = parser.parse_args()

    final_out_dir = prepare_splits(
        input_path=args.input,
        output_dir=args.output_dir,
        target=args.target,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        cutoff_start=args.cutoff_start,
        cutoff_mid=args.cutoff_mid,
        warmup_rows=args.warmup_rows,
    )

    print(f"Prepared data written to: {final_out_dir}")
    # sizes are already available in metadata; re-load to print
    meta = json.load(open(final_out_dir / 'prep_metadata.json', 'r'))
    print(
        f"Train/Val/Test sizes: {len(meta['split_timestamps']['train'])}/"
        f"{len(meta['split_timestamps']['val'])}/"
        f"{len(meta['split_timestamps']['test'])}"
    )


if __name__ == '__main__':
    main()

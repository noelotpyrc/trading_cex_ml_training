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
        filtered_cols.extend([c for c in pattern_matches if c not in filtered_cols])

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
    # Create a copy to avoid modifying the input DataFrame
    df = df.copy()
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

    # Normalize timestamps: parse as UTC then convert to tz-naive for consistent comparisons
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    targets["timestamp"] = pd.to_datetime(targets["timestamp"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

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


def generate_walk_forward_splits(
    df: pd.DataFrame,
    output_dir: Path,
    target_col: str,
    # Window sizes
    test_days: int = 90,
    val_days: int = 30,
    min_train_months: int = 12,
    step_days: int = 30,
    # Mode
    mode: str = "expanding",  # or "rolling"
    rolling_train_months: Optional[int] = None,
    # Boundary handling
    allow_partial_test: bool = False,
    min_test_days: int = 30,
    extend_last_test: bool = True,
    # Date range filter
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    # Feature filtering
    include_features: Optional[Sequence[str]] = None,
    include_patterns: Optional[Sequence[str]] = None,
    exclude_features: Optional[Sequence[str]] = None,
    warmup_rows: int = 0,
) -> List[Path]:
    """Generate walk-forward folds with rolling or expanding training windows.

    Creates fold directories (fold_01, fold_02, ...) each containing train/val/test
    splits in the same format as prepare_splits(), compatible with downstream pipelines.

    Args:
        df: DataFrame with 'timestamp' column and features + target.
        output_dir: Base directory for output. Folds written to output_dir/fold_NN/.
        target_col: Name of the target column.
        test_days: Number of days in each test window.
        val_days: Number of days in each validation window.
        min_train_months: Minimum training data required (in months) for first fold.
        step_days: How many days to slide forward between folds.
        mode: 'expanding' (train grows over time) or 'rolling' (fixed train window).
        rolling_train_months: Train window size in months when mode='rolling'. Required if mode='rolling'.
        allow_partial_test: If True, allow last fold to have partial test data.
        min_test_days: Minimum test days required when allow_partial_test=True.
        include_features: Feature column names to include.
        include_patterns: Glob patterns to match feature columns.
        exclude_features: Feature column names or patterns to exclude.
        warmup_rows: Number of initial rows to drop after sorting.
        extend_last_test: If True, extend last fold's test to include all remaining data.
        date_start: Optional start date filter (inclusive) for input data.
        date_end: Optional end date filter (inclusive) for input data.

    Returns:
        List of fold directory paths created.
    """
    from dateutil.relativedelta import relativedelta

    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    if mode == "rolling" and rolling_train_months is None:
        raise ValueError("rolling_train_months is required when mode='rolling'")

    # Warn if step_days differs from test_days (causes overlapping or gaps in test periods)
    if step_days != test_days:
        import warnings
        if step_days < test_days:
            warnings.warn(
                f"step_days ({step_days}) < test_days ({test_days}): test periods will overlap by {test_days - step_days} days",
                UserWarning
            )
        else:
            warnings.warn(
                f"step_days ({step_days}) > test_days ({test_days}): there will be {step_days - test_days} day gaps between test periods",
                UserWarning
            )

    # Ensure sorted by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Normalize to tz-naive for consistent comparisons
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    # Apply date range filter if specified
    if date_start is not None:
        df = df[df["timestamp"] >= pd.to_datetime(date_start)]
    if date_end is not None:
        df = df[df["timestamp"] <= pd.to_datetime(date_end)]
    if len(df) == 0:
        raise ValueError(f"No data remaining after date filter: date_start={date_start}, date_end={date_end}")
    df = df.reset_index(drop=True)

    data_start = df["timestamp"].min()
    data_end = df["timestamp"].max()

    # Compute fold boundaries
    folds: List[Dict[str, pd.Timestamp]] = []

    # First fold: train ends after min_train_months from data_start
    first_train_end = data_start + relativedelta(months=min_train_months)

    current_test_start = first_train_end + pd.Timedelta(days=val_days)

    while True:
        test_end_target = current_test_start + pd.Timedelta(days=test_days)

        # Check if test window has enough data
        actual_test_end = min(test_end_target, data_end + pd.Timedelta(days=1))
        test_days_available = (actual_test_end - current_test_start).days

        if test_days_available < min_test_days:
            # Not enough test data, stop generating folds
            break

        if test_days_available < test_days and not allow_partial_test:
            # Partial test not allowed, stop
            break

        # Compute train and val boundaries
        val_end = current_test_start
        val_start = val_end - pd.Timedelta(days=val_days)

        if mode == "expanding":
            train_start = data_start
        else:  # rolling
            train_start_unclamped = val_start - relativedelta(months=rolling_train_months)
            train_start = max(train_start_unclamped, data_start)
            if train_start != train_start_unclamped:
                import warnings
                warnings.warn(
                    f"Rolling train_start clamped to data_start: requested {train_start_unclamped.date()} -> {train_start.date()}",
                    UserWarning
                )

        train_end = val_start

        # Validate train has enough data
        train_months_actual = (train_end.year - train_start.year) * 12 + (train_end.month - train_start.month)
        if train_months_actual < min_train_months:
            # Not enough training data, skip to next potential fold
            current_test_start += pd.Timedelta(days=step_days)
            continue

        folds.append({
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": current_test_start,
            "test_end": actual_test_end,
        })

        # Slide forward
        current_test_start += pd.Timedelta(days=step_days)

    if not folds:
        raise ValueError(
            f"Could not generate any folds. Data range: {data_start} to {data_end}, "
            f"min_train_months={min_train_months}, test_days={test_days}, val_days={val_days}"
        )

    # Extend last fold's test to include all remaining data
    if extend_last_test and folds:
        last_fold = folds[-1]
        leftover_end = data_end + pd.Timedelta(days=1)  # Make end exclusive boundary past last timestamp
        if leftover_end > last_fold["test_end"]:
            leftover_days = (leftover_end - last_fold["test_end"]).days
            import warnings
            warnings.warn(
                f"Extending last fold test by {leftover_days} days to include remaining data "
                f"(test_end: {last_fold['test_end']} -> {leftover_end})",
                UserWarning
            )
            last_fold["test_end"] = leftover_end

    # Clean DataFrame once (feature filtering, NA handling)
    num_rows_before, num_cols_before = df.shape
    df_cleaned, dropped_constants, dropped_na_rows, selected_columns = _clean_dataframe(
        df,
        target_col=target_col,
        include_features=include_features,
        include_patterns=include_patterns,
        exclude_features=exclude_features,
        warmup_rows=warmup_rows,
    )
    num_rows_after, num_cols_after = df_cleaned.shape

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs: List[Path] = []

    for i, fold in enumerate(folds, start=1):
        fold_name = f"fold_{i:02d}"
        fold_dir = output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Slice data for each split
        ts = df_cleaned["timestamp"]
        train_mask = (ts >= fold["train_start"]) & (ts < fold["train_end"])
        val_mask = (ts >= fold["val_start"]) & (ts < fold["val_end"])
        test_mask = (ts >= fold["test_start"]) & (ts < fold["test_end"])

        train_df = df_cleaned[train_mask].copy()
        val_df = df_cleaned[val_mask].copy()
        test_df = df_cleaned[test_mask].copy()

        # Write outputs using existing helper
        _write_outputs(fold_dir, target_col, train_df, val_df, test_df)

        # Helper functions for metadata
        def _ts_list(part_df: pd.DataFrame) -> List[str]:
            if "timestamp" in part_df.columns and len(part_df) > 0:
                return part_df["timestamp"].astype(str).tolist()
            return []

        def _ts_range(part_df: pd.DataFrame) -> Dict[str, Optional[str]]:
            if "timestamp" in part_df.columns and len(part_df) > 0:
                return {
                    "min": str(part_df["timestamp"].min()),
                    "max": str(part_df["timestamp"].max()),
                }
            return {"min": None, "max": None}

        # Write prep_metadata.json (same format as existing)
        fold_total_rows = len(train_df) + len(val_df) + len(test_df)
        meta = PrepMetadata(
            input_path="walk_forward_split",
            num_rows_before=num_rows_before,
            num_rows_after=num_rows_after,  # After NA drop, before slicing
            num_features_before=num_cols_before,
            num_features_after=num_cols_after,
            target_column=target_col,
            split_strategy=f"walk_forward_{mode}",
            split_params={
                "fold_index": str(i),
                "total_folds": str(len(folds)),
                "fold_total_rows": str(fold_total_rows),
                "mode": mode,
                "test_days": str(test_days),
                "val_days": str(val_days),
                "min_train_months": str(min_train_months),
                "step_days": str(step_days),
                "train_start": str(fold["train_start"]),
                "train_end": str(fold["train_end"]),
                "val_start": str(fold["val_start"]),
                "val_end": str(fold["val_end"]),
                "test_start": str(fold["test_start"]),
                "test_end": str(fold["test_end"]),
            },
            dropped_constant_columns=dropped_constants,
            dropped_na_rows=dropped_na_rows,
            split_timestamps={
                "train": _ts_list(train_df),
                "val": _ts_list(val_df),
                "test": _ts_list(test_df),
            },
            split_timestamp_ranges={
                "train": _ts_range(train_df),
                "val": _ts_range(val_df),
                "test": _ts_range(test_df),
            },
            selected_feature_columns=[c for c in selected_columns if c not in dropped_constants],
            excluded_feature_columns=list(exclude_features or []),
        )
        with open(fold_dir / "prep_metadata.json", "w") as f:
            json.dump(asdict(meta), f, indent=2, default=str)

        fold_dirs.append(fold_dir)

    # Collect fold stats for summary
    fold_stats = []
    for i, (fold, fold_dir) in enumerate(zip(folds, fold_dirs), start=1):
        with open(fold_dir / "prep_metadata.json") as f:
            meta = json.load(f)
        train_rows = len(meta["split_timestamps"]["train"])
        val_rows = len(meta["split_timestamps"]["val"])
        test_rows = len(meta["split_timestamps"]["test"])
        # Expected rows based on days (hourly data assumption)
        expected_train_days = (fold["train_end"] - fold["train_start"]).days
        expected_val_days = (fold["val_end"] - fold["val_start"]).days
        expected_test_days = (fold["test_end"] - fold["test_start"]).days
        fold_stats.append({
            "name": f"fold_{i:02d}",
            "train_rows": train_rows,
            "val_rows": val_rows,
            "test_rows": test_rows,
            "expected_train_days": expected_train_days,
            "expected_val_days": expected_val_days,
            "expected_test_days": expected_test_days,
        })

    # Write walk_forward_params.json (input parameters only)
    params = {
        "mode": mode,
        "test_days": test_days,
        "val_days": val_days,
        "min_train_months": min_train_months,
        "step_days": step_days,
        "rolling_train_months": rolling_train_months,
        "allow_partial_test": allow_partial_test,
        "min_test_days": min_test_days,
        "extend_last_test": extend_last_test,
        "date_start": date_start,
        "date_end": date_end,
        "warmup_rows": warmup_rows,
    }
    with open(output_dir / "walk_forward_params.json", "w") as f:
        json.dump(params, f, indent=2)

    # Write walk_forward_config.json (planned boundaries per fold)
    config = {
        "data_start": str(data_start),
        "data_end": str(data_end),
        "num_folds": len(folds),
        "folds": [
            {
                "name": f"fold_{i:02d}",
                "train_start": str(fold["train_start"]),
                "train_end": str(fold["train_end"]),
                "val_start": str(fold["val_start"]),
                "val_end": str(fold["val_end"]),
                "test_start": str(fold["test_start"]),
                "test_end": str(fold["test_end"]),
            }
            for i, fold in enumerate(folds, start=1)
        ],
    }
    with open(output_dir / "walk_forward_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Write walk_forward_summary.json (actual results)
    summary = {
        "data_start": str(data_start),
        "data_end": str(data_end),
        "num_rows_before_cleaning": num_rows_before,
        "num_rows_after_cleaning": num_rows_after,
        "dropped_na_rows": dropped_na_rows,
        "num_folds": len(folds),
        "folds": fold_stats,
    }
    with open(output_dir / "walk_forward_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return fold_dirs


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


def walk_forward_main() -> None:
    """CLI entry point for generating walk-forward splits."""
    parser = argparse.ArgumentParser(
        description='Generate walk-forward validation splits from merged features-targets'
    )
    parser.add_argument('--input', type=Path, required=True, help='Path to merged_features_targets.csv')
    parser.add_argument('--output-dir', type=Path, required=True, help='Base output directory for fold subdirectories')
    parser.add_argument('--target', type=str, required=True, help='Target column to predict')
    # Window sizes
    parser.add_argument('--test-days', type=int, default=90, help='Number of days in each test window')
    parser.add_argument('--val-days', type=int, default=30, help='Number of days in each validation window')
    parser.add_argument('--min-train-months', type=int, default=12, help='Minimum training data (months) for first fold')
    parser.add_argument('--step-days', type=int, default=30, help='Days to slide forward between folds')
    # Mode
    parser.add_argument('--mode', type=str, default='expanding', choices=['expanding', 'rolling'],
                        help='Window mode: expanding (train grows) or rolling (fixed train size)')
    parser.add_argument('--rolling-train-months', type=int, default=None,
                        help='Train window size in months when mode=rolling (required for rolling mode)')
    # Boundary handling
    parser.add_argument('--allow-partial-test', action='store_true',
                        help='Allow last fold to have partial test data')
    parser.add_argument('--min-test-days', type=int, default=30,
                        help='Minimum test days when partial test is allowed')
    parser.add_argument('--extend-last-test', action='store_true', default=True,
                        help='Extend last fold test to include remaining data (default: True)')
    parser.add_argument('--no-extend-last-test', dest='extend_last_test', action='store_false',
                        help='Do not extend last fold test')
    # Date range filter
    parser.add_argument('--date-start', type=str, default=None,
                        help='Optional start date filter (inclusive), e.g., 2022-01-01')
    parser.add_argument('--date-end', type=str, default=None,
                        help='Optional end date filter (inclusive), e.g., 2025-12-31')
    # Feature filtering
    parser.add_argument('--include-features', type=str, nargs='+', default=None,
                        help='Feature column names to include (space-separated)')
    parser.add_argument('--exclude-features', type=str, nargs='+', default=None,
                        help='Feature column names to exclude (space-separated)')
    parser.add_argument('--warmup-rows', type=int, default=0,
                        help='Drop the first N rows before processing')
    args = parser.parse_args()

    # Load input data
    df = _load_merged(args.input)

    fold_dirs = generate_walk_forward_splits(
        df=df,
        output_dir=args.output_dir,
        target_col=args.target,
        test_days=args.test_days,
        val_days=args.val_days,
        min_train_months=args.min_train_months,
        step_days=args.step_days,
        mode=args.mode,
        rolling_train_months=args.rolling_train_months,
        allow_partial_test=args.allow_partial_test,
        min_test_days=args.min_test_days,
        extend_last_test=args.extend_last_test,
        date_start=args.date_start,
        date_end=args.date_end,
        include_features=args.include_features,
        exclude_features=args.exclude_features,
        warmup_rows=args.warmup_rows,
    )

    print(f"Generated {len(fold_dirs)} walk-forward folds:")
    for fold_dir in fold_dirs:
        meta = json.load(open(fold_dir / 'prep_metadata.json', 'r'))
        train_range = meta['split_timestamp_ranges']['train']
        test_range = meta['split_timestamp_ranges']['test']
        print(f"  {fold_dir.name}: train {train_range['min'][:10]} to {train_range['max'][:10]}, "
              f"test {test_range['min'][:10]} to {test_range['max'][:10]}")


if __name__ == '__main__':
    main()


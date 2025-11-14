#!/usr/bin/env python3
"""
Generate rolling train/val/test splits as CSVs for use with `split.existing_dir`.

Per fold:
- Train: previous 12 full months (configurable)
- Val: N days immediately before the test month (configurable)
- Test: next 1 full month (configurable)

Outputs per fold directory:
  X_train.csv, y_train.csv, X_val.csv, y_val.csv, X_test.csv, y_test.csv, prep_metadata.json

The generated directories can be plugged into the existing LGBM pipeline by
setting `split.existing_dir` in the base config.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _ensure_repo_imports() -> None:
    """Ensure project root is on sys.path for module imports."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_imports()

# Import pipeline helpers for consistent cleaning/filters and config parsing
from training.run_lgbm_pipeline import load_config  # type: ignore
from training.prepare_training_data import (  # type: ignore
    _apply_feature_filters,
    _clean_dataframe,
    _load_feature_store,
    _load_extra_feature_file,
    _load_merged,
)


@dataclass
class RollingConfig:
    start_date: Optional[str]
    end_date: Optional[str]
    train_months: int = 12
    val_days: int = 7
    test_months: int = 1
    step_months: int = 1
    min_rows_train: int = 100
    min_rows_val: int = 50
    min_rows_test: int = 200
    require_full_train_window: bool = True


def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz=ts.tz).normalize()


def _next_month(ts: pd.Timestamp, n: int = 1) -> pd.Timestamp:
    return (ts + pd.offsets.MonthBegin(n)).normalize()


def _build_fold_ranges(
    test_month_start: pd.Timestamp,
    rcfg: RollingConfig,
) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return half-open intervals [start, end) for train/val/test for one fold.

    Semantics:
    - test: full calendar month starting at `test_month_start`
    - val: N days immediately preceding `test_month_start`
    - train: 12 months window ending at `val_start` (to-the-day), i.e.,
             [val_start - 12 months, val_start)
    """
    # Test month covers full month
    test_start = _month_start(test_month_start)
    test_end = _next_month(test_start, rcfg.test_months)

    # Validation: N days immediately before test_start
    val_end = test_start
    val_start = val_end - pd.Timedelta(days=rcfg.val_days)

    # Train: previous N months ending exactly at val_start (sliding window)
    train_end = val_start
    train_start = val_start - pd.DateOffset(months=rcfg.train_months)

    return {
        "train": (train_start, train_end),
        "val": (val_start, val_end),
        "test": (test_start, test_end),
    }


def _ceil_month_start(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the month start on or after ts (ceil to month boundary)."""
    ms = _month_start(ts)
    return ms if ms == ts.normalize() and ts.day == 1 else _next_month(ms, 1)


def _slice_by_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ts = df["timestamp"]
    return df[(ts >= start) & (ts < end)].copy()


def _split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    # Exclude timestamp and all y_* columns from X to avoid leakage
    x_cols = [c for c in df.columns if c != "timestamp" and not c.startswith("y_")]
    X = df[x_cols]
    y = df[target_col].astype(float)
    return X, y


def _write_split_dir(
    out_dir: Path,
    target_col: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    meta: Dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X, y = _split_xy(part, target_col)
        X.to_csv(out_dir / f"X_{name}.csv", index=False)
        y.to_csv(out_dir / f"y_{name}.csv", index=False, header=[target_col])
    with open(out_dir / "prep_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def _ts_list(df: pd.DataFrame) -> List[str]:
    if "timestamp" in df.columns and len(df) > 0:
        return df["timestamp"].astype(str).tolist()
    return []


def _ts_range(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    if "timestamp" in df.columns and len(df) > 0:
        return {"min": str(df["timestamp"].min()), "max": str(df["timestamp"].max())}
    return {"min": None, "max": None}


def _load_and_clean_dataset(config: Dict[str, object]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load merged data using the same helpers as training, apply filters and cleaning.

    Returns the cleaned DataFrame (with timestamp + features + target) and an info dict
    with details for metadata: selected_columns, excluded_columns, dropped_constants,
    dropped_na_rows, extra_sources, input_path.
    """
    target = config["target"]["variable"]  # type: ignore[index]
    split_cfg = (config.get("split") or {})  # type: ignore[assignment]
    warmup_rows = int((split_cfg or {}).get("warmup_rows", 0) or 0)  # type: ignore[arg-type]

    # Determine features/targets source
    features_path = (config.get("feature_store") or {}).get("features_csv")  # type: ignore[dict-item]
    targets_path = (config.get("feature_store") or {}).get("targets_csv")  # type: ignore[dict-item]
    input_path = config.get("input_data")  # type: ignore[assignment]

    # Collect feature selection rules
    fsel = config.get("feature_selection") or {}  # type: ignore[assignment]
    include_features = list(fsel.get("include") or [])  # type: ignore[call-arg]
    include_patterns = list(fsel.get("include_patterns") or [])  # type: ignore[call-arg]
    exclude_features = fsel.get("exclude")  # type: ignore[assignment]

    # Optional extra feature files
    extra_feature_files = list(config.get("extra_feature_files") or [])  # type: ignore[call-arg]

    # Load merged dataset
    if features_path and targets_path:
        merged = _load_feature_store(Path(features_path), Path(targets_path), target)  # type: ignore[arg-type]
        src_label = f"feature_store:{features_path}|{targets_path}"
    else:
        if not input_path:
            raise ValueError("Configuration must provide either feature_store paths or input_data")
        merged = _load_merged(Path(input_path))  # type: ignore[arg-type]
        src_label = str(input_path)

    # Merge extra features
    extra_sources: List[dict] = []
    if extra_feature_files:
        for entry in extra_feature_files:
            extra_df = _load_extra_feature_file(entry)
            merged = merged.merge(extra_df, on="timestamp", how="left")
            extra_sources.append(
                {
                    "path": str(entry.get("path")),
                    "include": entry.get("include"),
                    "exclude": entry.get("exclude"),
                    "added_columns": [c for c in extra_df.columns if c != "timestamp"],
                }
            )

    # Normalize timestamps to tz-naive UTC to avoid tz-aware/naive comparison issues
    if "timestamp" in merged.columns:
        ts = pd.to_datetime(merged["timestamp"], errors="coerce", utc=True)
        # Ensure UTC then drop timezone to make comparisons with naive anchors valid
        merged["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    # Clean and select features
    cleaned, dropped_constants, dropped_na_rows, selected_columns = _clean_dataframe(
        merged,
        target_col=target,
        include_features=include_features,
        include_patterns=include_patterns,
        exclude_features=exclude_features,
        warmup_rows=warmup_rows,
    )

    info = {
        "selected_feature_columns": [c for c in selected_columns if c not in dropped_constants],
        "excluded_feature_columns": list(exclude_features or []),
        "dropped_constant_columns": list(dropped_constants),
        "dropped_na_rows": int(dropped_na_rows),
        "extra_feature_sources": extra_sources or None,
        "input_path": src_label,
    }
    return cleaned, info


def generate_rolling_splits(
    base_config_path: Path,
    rolling: RollingConfig,
    output_root: Optional[Path] = None,
) -> Path:
    # Parse and normalize base config via pipeline loader for consistency
    cfg = load_config(base_config_path)
    target = cfg["target"]["variable"]  # type: ignore[index]

    # Decide output root
    splits_dir = cfg.get("training_splits_dir") or (cfg["output_dir"] / "training_splits")  # type: ignore[index]
    scheme = f"train{rolling.train_months}_val{rolling.val_days}d_test{rolling.test_months}m"
    final_root = Path(output_root) if output_root else (Path(splits_dir) / "rolling" / scheme)
    final_root.mkdir(parents=True, exist_ok=True)

    # Load and clean dataset
    df, info = _load_and_clean_dataset(cfg)
    if "timestamp" not in df.columns:
        raise ValueError("Dataset must include 'timestamp' column for time-based slicing")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Determine fold range based on available data and requested limits
    ts_min = pd.to_datetime(df["timestamp"].min())
    ts_max = pd.to_datetime(df["timestamp"].max())

    # Compute the earliest test month start candidate honoring full train window
    start_anchor = pd.to_datetime(rolling.start_date) if rolling.start_date else ts_min  # type: ignore[arg-type]
    # earliest S such that (S - val_days - train_months) >= ts_min and S >= start_anchor
    earliest_needed = ts_min + pd.DateOffset(months=rolling.train_months) + pd.Timedelta(days=rolling.val_days)
    earliest_test_start = _ceil_month_start(earliest_needed)
    start_anchor = max(_month_start(pd.to_datetime(start_anchor)), earliest_test_start)
    end_anchor = pd.to_datetime(rolling.end_date) if rolling.end_date else ts_max  # type: ignore[arg-type]
    end_anchor = _month_start(end_anchor)

    # Iterate test month anchors
    cur = start_anchor
    created = 0
    folds: List[str] = []
    while cur <= end_anchor:
        ranges = _build_fold_ranges(cur, rolling)
        tr_s, tr_e = ranges["train"]
        va_s, va_e = ranges["val"]
        te_s, te_e = ranges["test"]

        # Slice
        df_train = _slice_by_range(df, tr_s, tr_e)
        df_val = _slice_by_range(df, va_s, va_e)
        df_test = _slice_by_range(df, te_s, te_e)

        # Enforce: full train window available by default
        if rolling.require_full_train_window and tr_s < ts_min:
            cur = _next_month(cur, rolling.step_months)
            continue

        # Guards
        if (
            len(df_train) >= rolling.min_rows_train
            and len(df_val) >= rolling.min_rows_val
            and len(df_test) >= rolling.min_rows_test
        ):
            fold_id = f"{cur.year:04d}-{cur.month:02d}"
            fold_dir = final_root / fold_id

            # Effective durations for metadata/debug
            train_days = float((tr_e - tr_s).days)
            val_days = float((va_e - va_s).days)
            test_days = float((te_e - te_s).days)
            eff_train_months = round(train_days / 30.437, 2)  # approx months

            meta = {
                "input_path": info["input_path"],
                "num_rows_before": int(len(df)),
                "num_rows_after": int(len(df)),
                "num_features_before": int(df.drop(columns=[c for c in df.columns if c.startswith("y_") or c == "timestamp"]).shape[1]),
                "num_features_after": int(df.drop(columns=[c for c in df.columns if c.startswith("y_") or c == "timestamp"]).shape[1]),
                "target_column": target,
                "split_strategy": f"rolling_train{rolling.train_months}m_val{rolling.val_days}d_test{rolling.test_months}m",
                "split_params": {
                    "train_months": str(rolling.train_months),
                    "val_days": str(rolling.val_days),
                    "test_months": str(rolling.test_months),
                    "step_months": str(rolling.step_months),
                    "fold_id": fold_id,
                    "effective_train_days": train_days,
                    "effective_train_months_approx": eff_train_months,
                    "effective_val_days": val_days,
                    "effective_test_days": test_days,
                },
                "dropped_constant_columns": info["dropped_constant_columns"],
                "dropped_na_rows": info["dropped_na_rows"],
                "split_timestamps": {
                    "train": _ts_list(df_train),
                    "val": _ts_list(df_val),
                    "test": _ts_list(df_test),
                },
                "split_timestamp_ranges": {
                    "train": _ts_range(df_train),
                    "val": _ts_range(df_val),
                    "test": _ts_range(df_test),
                },
                "merged_output_csv": None,
                "selected_feature_columns": info["selected_feature_columns"],
                "excluded_feature_columns": info["excluded_feature_columns"],
                "extra_feature_sources": info["extra_feature_sources"],
            }

            _write_split_dir(fold_dir, target, df_train, df_val, df_test, meta)
            created += 1
            folds.append(fold_id)

        # Advance to next fold anchor
        cur = _next_month(cur, rolling.step_months)

    if created == 0:
        raise SystemExit("No fold created; check date bounds and min_rows guards.")

    print(f"Created {created} fold(s) under: {final_root}")
    print("Folds:", ", ".join(folds))
    return final_root


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate rolling splits for split.existing_dir")
    ap.add_argument("--config", required=True, type=Path, help="Path to base model config JSON")
    ap.add_argument("--start-date", default=None, help="YYYY-MM-DD; default: earliest timestamp")
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD; default: latest timestamp")
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--val-days", type=int, default=7)
    ap.add_argument("--test-months", type=int, default=1)
    ap.add_argument("--step-months", type=int, default=1)
    ap.add_argument("--min-rows-train", type=int, default=100)
    ap.add_argument("--min-rows-val", type=int, default=50)
    ap.add_argument("--min-rows-test", type=int, default=200)
    ap.add_argument("--output-root", type=Path, default=None, help="Override output root directory for folds")
    ap.add_argument("--allow-partial-train-window", action="store_true", help="Include folds even when full train history is not available")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rcfg = RollingConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        train_months=int(args.train_months),
        val_days=int(args.val_days),
        test_months=int(args.test_months),
        step_months=int(args.step_months),
        min_rows_train=int(args.min_rows_train),
        min_rows_val=int(args.min_rows_val),
        min_rows_test=int(args.min_rows_test),
        require_full_train_window=(not bool(args.allow_partial_train_window)),
    )
    generate_rolling_splits(args.config, rcfg, args.output_root)


if __name__ == "__main__":
    main()

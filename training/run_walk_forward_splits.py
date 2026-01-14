#!/usr/bin/env python3
"""
Generate walk-forward validation splits from feature store files.

Supports:
- Separate features and targets CSVs (merged on timestamp)
- Single merged file (use same path for both --features and --targets)
- Feature selection via --include-features
- All walk-forward parameters (test_days, val_days, step_days, mode, etc.)

Usage:
    python run_walk_forward_splits.py \
        --features /path/to/features.csv \
        --targets /path/to/targets.csv \
        --output-dir /path/to/output \
        --target y_logret_24h \
        --test-days 180 \
        --val-days 7 \
        --step-days 180 \
        --mode expanding
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add parent to path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
from training.prepare_training_data import (
    generate_walk_forward_splits,
    _load_feature_store,
    _load_merged,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate walk-forward validation splits from feature store files"
    )
    
    # Input files
    parser.add_argument(
        "--features", type=Path, required=True,
        help="Path to features CSV (must have 'timestamp' column)"
    )
    parser.add_argument(
        "--targets", type=Path, default=None,
        help="Path to targets CSV (if separate from features). If not provided, uses --features file."
    )
    parser.add_argument(
        "--target", type=str, required=True,
        help="Target column name to predict"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Base output directory for fold subdirectories"
    )
    
    # Feature selection
    parser.add_argument(
        "--include-features", type=str, nargs="+", default=None,
        help="Feature column names to include (space-separated)"
    )
    parser.add_argument(
        "--include-features-file", type=Path, default=None,
        help="JSON file containing list of feature names to include"
    )
    parser.add_argument(
        "--exclude-features", type=str, nargs="+", default=None,
        help="Feature column names to exclude (space-separated)"
    )
    
    # Window sizes
    parser.add_argument("--test-days", type=int, default=90, help="Days in each test window")
    parser.add_argument("--val-days", type=int, default=30, help="Days in each validation window")
    parser.add_argument("--min-train-months", type=int, default=12, help="Minimum training months")
    parser.add_argument("--step-days", type=int, default=30, help="Days to slide between folds")
    
    # Mode
    parser.add_argument(
        "--mode", type=str, default="expanding", choices=["expanding", "rolling"],
        help="Window mode: expanding (train grows) or rolling (fixed train size)"
    )
    parser.add_argument(
        "--rolling-train-months", type=int, default=None,
        help="Train window size in months for rolling mode"
    )
    
    # Boundary handling
    parser.add_argument("--allow-partial-test", action="store_true", help="Allow partial test data")
    parser.add_argument("--min-test-days", type=int, default=30, help="Minimum test days if partial allowed")
    parser.add_argument("--extend-last-test", action="store_true", default=True,
                        help="Extend last fold test to include remaining data")
    parser.add_argument("--no-extend-last-test", dest="extend_last_test", action="store_false")
    
    # Date range filter
    parser.add_argument("--date-start", type=str, default=None, help="Start date filter (inclusive)")
    parser.add_argument("--date-end", type=str, default=None, help="End date filter (inclusive)")
    
    # Data handling
    parser.add_argument("--warmup-rows", type=int, default=0, help="Drop first N rows after sorting")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading features from: {args.features}")
    if args.targets and args.targets != args.features:
        print(f"Loading targets from: {args.targets}")
        df = _load_feature_store(args.features, args.targets, args.target)
    else:
        df = _load_merged(args.features)
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Resolve include features
    include_features: Optional[List[str]] = args.include_features
    if args.include_features_file:
        with open(args.include_features_file) as f:
            include_features = json.load(f)
        print(f"Loaded {len(include_features)} features from {args.include_features_file}")
    
    # Generate splits
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
        include_features=include_features,
        exclude_features=args.exclude_features,
        warmup_rows=args.warmup_rows,
    )
    
    # Build summary with actual vs expected row counts
    fold_summary = []
    for fold_dir in fold_dirs:
        with open(fold_dir / "prep_metadata.json") as f:
            meta = json.load(f)
        
        train_rows = len(meta["split_timestamps"]["train"])
        val_rows = len(meta["split_timestamps"]["val"])
        test_rows = len(meta["split_timestamps"]["test"])
        
        # Get planned boundaries from split_params
        p = meta["split_params"]
        
        # Calculate expected rows (days * 24 for hourly data)
        def days_between(start_str: str, end_str: str) -> int:
            from datetime import datetime
            start = datetime.fromisoformat(start_str.replace(" ", "T").split("+")[0])
            end = datetime.fromisoformat(end_str.replace(" ", "T").split("+")[0])
            return (end - start).days
        
        expected_train = days_between(p["train_start"], p["train_end"]) * 24
        expected_val = days_between(p["val_start"], p["val_end"]) * 24
        expected_test = days_between(p["test_start"], p["test_end"]) * 24
        
        fold_summary.append({
            "name": fold_dir.name,
            "train_rows": train_rows,
            "train_expected": expected_train,
            "val_rows": val_rows,
            "val_expected": expected_val,
            "test_rows": test_rows,
            "test_expected": expected_test,
            "train_range": meta["split_timestamp_ranges"]["train"],
            "test_range": meta["split_timestamp_ranges"]["test"],
        })
    
    # Write walk_forward_summary.json
    summary = {
        "num_rows_before_cleaning": meta.get("num_rows_before"),
        "num_rows_after_cleaning": meta.get("num_rows_after"),
        "dropped_na_rows": meta.get("dropped_na_rows"),
        "num_folds": len(fold_dirs),
        "folds": fold_summary,
    }
    summary_path = args.output_dir / "walk_forward_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\nGenerated {len(fold_dirs)} folds:")
    print(f"  Rows before cleaning: {meta.get('num_rows_before')}")
    print(f"  Rows after cleaning: {meta.get('num_rows_after')}")
    print(f"  Dropped NA rows: {meta.get('dropped_na_rows')}")
    print()
    for fs in fold_summary:
        train_diff = fs['train_rows'] - fs['train_expected']
        test_diff = fs['test_rows'] - fs['test_expected']
        print(
            f"  {fs['name']}: "
            f"train={fs['train_rows']}/{fs['train_expected']} ({train_diff:+d}), "
            f"val={fs['val_rows']}/{fs['val_expected']}, "
            f"test={fs['test_rows']}/{fs['test_expected']} ({test_diff:+d})"
        )
    
    print(f"\nOutput written to: {args.output_dir}")
    print(f"  walk_forward_config.json (planned boundaries)")
    print(f"  walk_forward_summary.json (actual vs expected)")


if __name__ == "__main__":
    main()

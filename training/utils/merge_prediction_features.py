#!/usr/bin/env python3
"""
Merge walk-forward prediction features with the original feature file.

Uses the original feature file as base and left-joins prediction columns.
Includes sanity checks for timestamp alignment and coverage.

Usage:
    python training/utils/merge_prediction_features.py \
        --base-csv "/path/to/derived_features_full.csv" \
        --prediction-csvs "/path/to/pred_vol_model.csv" "/path/to/pred_drift_model.csv" \
        --output "/path/to/derived_features_with_stacked.csv"
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_validate_predictions(pred_path: Path) -> pd.DataFrame:
    """
    Load prediction CSV and validate format.
    Expected format: timestamp, <prediction_column>
    """
    df = pd.read_csv(pred_path, parse_dates=["timestamp"])
    
    if "timestamp" not in df.columns:
        raise ValueError(f"Prediction file {pred_path} missing 'timestamp' column")
    
    if len(df.columns) != 2:
        raise ValueError(f"Prediction file {pred_path} should have exactly 2 columns (timestamp, prediction), got {list(df.columns)}")
    
    pred_col = [c for c in df.columns if c != "timestamp"][0]
    
    # Check for duplicates
    dup_count = df.duplicated(subset=["timestamp"]).sum()
    if dup_count > 0:
        raise ValueError(f"Prediction file {pred_path} has {dup_count} duplicate timestamps")
    
    # Check for NaNs in predictions
    nan_count = df[pred_col].isna().sum()
    if nan_count > 0:
        logger.warning(f"Prediction file {pred_path} has {nan_count} NaN values in {pred_col}")
    
    return df, pred_col


def merge_prediction_features(
    base_csv: Path,
    prediction_csvs: List[Path],
    output_csv: Path,
) -> pd.DataFrame:
    """
    Merge prediction CSVs with base feature file.
    
    Uses base as left side, so all base rows are preserved.
    Prediction columns will have NaN where timestamps don't match.
    """
    logger.info(f"Base features: {base_csv}")
    logger.info(f"Prediction files: {len(prediction_csvs)}")
    
    # Load base features
    logger.info("Loading base features...")
    base_df = pd.read_csv(base_csv, parse_dates=["timestamp"])
    logger.info(f"Base features: {len(base_df)} rows, {len(base_df.columns)} columns")
    logger.info(f"Base date range: {base_df['timestamp'].min()} to {base_df['timestamp'].max()}")
    
    # Check for duplicate timestamps in base
    base_dups = base_df.duplicated(subset=["timestamp"]).sum()
    if base_dups > 0:
        logger.warning(f"Base file has {base_dups} duplicate timestamps")
    
    # Track merge statistics
    merge_stats = []
    
    # Merge each prediction file
    result_df = base_df.copy()
    
    for pred_path in prediction_csvs:
        logger.info(f"\nProcessing: {pred_path.name}")
        
        pred_df, pred_col = load_and_validate_predictions(pred_path)
        logger.info(f"  Prediction column: {pred_col}")
        logger.info(f"  Rows: {len(pred_df)}")
        logger.info(f"  Date range: {pred_df['timestamp'].min()} to {pred_df['timestamp'].max()}")
        
        # Check if column already exists
        if pred_col in result_df.columns:
            logger.warning(f"  Column {pred_col} already exists in base, will be overwritten")
            result_df = result_df.drop(columns=[pred_col])
        
        # Merge
        before_rows = len(result_df)
        result_df = result_df.merge(
            pred_df,
            on="timestamp",
            how="left"
        )
        after_rows = len(result_df)
        
        if after_rows != before_rows:
            logger.warning(f"  Row count changed during merge: {before_rows} -> {after_rows}")
        
        # Calculate coverage
        matched = result_df[pred_col].notna().sum()
        missing = result_df[pred_col].isna().sum()
        coverage_pct = matched / len(result_df) * 100
        
        logger.info(f"  Coverage: {matched}/{len(result_df)} ({coverage_pct:.1f}%)")
        logger.info(f"  Missing (NaN): {missing}")
        
        # Find timestamps in predictions but not in base
        pred_only = set(pred_df["timestamp"]) - set(base_df["timestamp"])
        if pred_only:
            logger.info(f"  Predictions not in base: {len(pred_only)} timestamps")
        
        merge_stats.append({
            "file": pred_path.name,
            "column": pred_col,
            "pred_rows": len(pred_df),
            "matched": matched,
            "missing": missing,
            "coverage_pct": coverage_pct,
            "pred_only": len(pred_only),
        })
    
    # Summary
    logger.info("\n=== Merge Summary ===")
    logger.info(f"Output rows: {len(result_df)}")
    logger.info(f"Output columns: {len(result_df.columns)}")
    
    new_cols = [s["column"] for s in merge_stats]
    logger.info(f"New prediction columns: {new_cols}")
    
    # Coverage table
    logger.info("\n=== Coverage by Prediction ===")
    logger.info(f"{'Column':<25} {'Matched':>10} {'Missing':>10} {'Coverage':>10}")
    logger.info("-" * 57)
    for s in merge_stats:
        logger.info(f"{s['column']:<25} {s['matched']:>10} {s['missing']:>10} {s['coverage_pct']:>9.1f}%")
    
    # Sanity checks
    logger.info("\n=== Sanity Checks ===")
    
    # Check row count preserved
    if len(result_df) != len(base_df):
        logger.error(f"✗ Row count mismatch: base={len(base_df)}, output={len(result_df)}")
    else:
        logger.info("✓ Row count preserved")
    
    # Check all base columns preserved
    missing_cols = set(base_df.columns) - set(result_df.columns)
    if missing_cols:
        logger.error(f"✗ Missing base columns: {missing_cols}")
    else:
        logger.info("✓ All base columns preserved")
    
    # Check timestamp order preserved
    if not result_df["timestamp"].is_monotonic_increasing:
        # Check if original was monotonic
        if base_df["timestamp"].is_monotonic_increasing:
            logger.warning("✗ Timestamp order not monotonic (but base wasn't either)")
        else:
            logger.info("○ Timestamp order not monotonic (same as base)")
    else:
        logger.info("✓ Timestamp order preserved")
    
    # Warn about low coverage
    low_coverage = [s for s in merge_stats if s["coverage_pct"] < 50]
    if low_coverage:
        logger.warning(f"⚠ Low coverage (<50%) for: {[s['column'] for s in low_coverage]}")
    
    # Save
    logger.info(f"\nSaving to: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(result_df)} rows, {len(result_df.columns)} columns")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Merge walk-forward prediction features with original feature file"
    )
    parser.add_argument(
        "--base-csv",
        type=Path,
        required=True,
        help="Path to base features CSV (all rows preserved)"
    )
    parser.add_argument(
        "--prediction-csvs",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to prediction CSV files to merge"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save merged output CSV"
    )
    
    args = parser.parse_args()
    
    if not args.base_csv.exists():
        logger.error(f"Base CSV not found: {args.base_csv}")
        sys.exit(1)
    
    for pred_path in args.prediction_csvs:
        if not pred_path.exists():
            logger.error(f"Prediction CSV not found: {pred_path}")
            sys.exit(1)
    
    merge_prediction_features(
        base_csv=args.base_csv,
        prediction_csvs=args.prediction_csvs,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()

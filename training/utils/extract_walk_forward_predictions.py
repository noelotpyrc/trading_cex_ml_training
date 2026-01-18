#!/usr/bin/env python3
"""
Extract and merge walk-forward test predictions from all fold runs.

Usage:
    python training/utils/extract_walk_forward_predictions.py \
        --experiment-dir "/path/to/walk_forward/experiment_name" \
        --output "/path/to/output/pred_model_name.csv" \
        --column-name "pred_vol_model" \
        --overlap-strategy earliest
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def discover_run_dirs(experiment_dir: Path) -> List[Path]:
    """
    Discover all run directories in the experiment folder.
    Returns sorted list of run directories that contain pred_test.csv.
    """
    run_dirs = []
    for d in experiment_dir.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            pred_file = d / "pred_test.csv"
            if pred_file.exists():
                run_dirs.append(d)
            else:
                logger.warning(f"Run dir {d.name} missing pred_test.csv, skipping")
    
    # Sort by run directory name (which includes timestamp)
    run_dirs.sort(key=lambda x: x.name)
    return run_dirs


def load_predictions(run_dir: Path) -> Tuple[pd.DataFrame, dict]:
    """
    Load pred_test.csv from a run directory.
    Returns DataFrame and metadata about the fold.
    """
    pred_file = run_dir / "pred_test.csv"
    df = pd.read_csv(pred_file, parse_dates=["timestamp"])
    
    metadata = {
        "run_name": run_dir.name,
        "row_count": len(df),
        "min_timestamp": df["timestamp"].min(),
        "max_timestamp": df["timestamp"].max(),
        "nan_count": df["y_pred"].isna().sum(),
    }
    
    return df, metadata


def check_timestamp_gaps(fold_metadata: List[dict]) -> List[dict]:
    """
    Check for gaps between consecutive folds.
    Returns list of gap info dicts.
    """
    gaps = []
    for i in range(1, len(fold_metadata)):
        prev = fold_metadata[i - 1]
        curr = fold_metadata[i]
        
        gap_hours = (curr["min_timestamp"] - prev["max_timestamp"]).total_seconds() / 3600
        
        if gap_hours > 1:  # More than 1 hour gap
            gaps.append({
                "after_fold": prev["run_name"],
                "before_fold": curr["run_name"],
                "gap_hours": gap_hours,
            })
    
    return gaps


def merge_predictions(
    all_predictions: List[pd.DataFrame],
    overlap_strategy: str
) -> pd.DataFrame:
    """
    Merge predictions from all folds, handling overlapping timestamps.
    
    Args:
        all_predictions: List of DataFrames with timestamp and y_pred columns
        overlap_strategy: How to handle overlapping timestamps
            - "earliest": Keep prediction from earliest fold
            - "latest": Keep prediction from latest fold  
            - "average": Average predictions for overlapping timestamps
    
    Returns:
        Merged DataFrame with unique timestamps
    """
    # Concatenate all predictions
    combined = pd.concat(all_predictions, ignore_index=True)
    
    # Check for duplicates
    dup_count = combined.duplicated(subset=["timestamp"], keep=False).sum()
    unique_timestamps = combined["timestamp"].nunique()
    total_rows = len(combined)
    
    logger.info(f"Total rows: {total_rows}, Unique timestamps: {unique_timestamps}")
    if dup_count > 0:
        logger.info(f"Overlapping predictions found: {dup_count} rows ({dup_count - unique_timestamps} duplicates)")
    
    if overlap_strategy == "earliest":
        # First occurrence is from earliest fold (since we sorted run_dirs)
        merged = combined.drop_duplicates(subset=["timestamp"], keep="first")
    elif overlap_strategy == "latest":
        merged = combined.drop_duplicates(subset=["timestamp"], keep="last")
    elif overlap_strategy == "average":
        merged = combined.groupby("timestamp", as_index=False).agg({
            "y_pred": "mean",
            "y_true": "first"  # y_true should be same for same timestamp
        })
    else:
        raise ValueError(f"Unknown overlap_strategy: {overlap_strategy}")
    
    # Sort by timestamp
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    
    return merged


def extract_walk_forward_predictions(
    experiment_dir: Path,
    prediction_column_name: str,
    output_csv: Path,
    overlap_strategy: str = "earliest"
) -> pd.DataFrame:
    """
    Extract and merge walk-forward test predictions from all fold runs.
    
    Args:
        experiment_dir: Path to experiment directory containing run_* folders
        prediction_column_name: Name for the prediction column in output
        output_csv: Path to save merged predictions
        overlap_strategy: How to handle overlapping timestamps (earliest|latest|average)
    
    Returns:
        DataFrame with timestamp and prediction column
    """
    logger.info(f"Extracting predictions from: {experiment_dir}")
    logger.info(f"Overlap strategy: {overlap_strategy}")
    
    # Discover run directories
    run_dirs = discover_run_dirs(experiment_dir)
    if not run_dirs:
        raise ValueError(f"No run directories with pred_test.csv found in {experiment_dir}")
    
    logger.info(f"Found {len(run_dirs)} run directories")
    
    # Load predictions from each fold
    all_predictions = []
    fold_metadata = []
    
    for run_dir in run_dirs:
        df, metadata = load_predictions(run_dir)
        all_predictions.append(df[["timestamp", "y_pred", "y_true"]])
        fold_metadata.append(metadata)
        
        logger.info(
            f"  {metadata['run_name']}: {metadata['row_count']} rows, "
            f"{metadata['min_timestamp'].date()} to {metadata['max_timestamp'].date()}, "
            f"NaNs: {metadata['nan_count']}"
        )
    
    # Sanity checks
    logger.info("\n=== Sanity Checks ===")
    
    # Check for gaps
    gaps = check_timestamp_gaps(fold_metadata)
    if gaps:
        logger.warning("Timestamp gaps detected between folds:")
        for g in gaps:
            logger.warning(f"  {g['after_fold']} -> {g['before_fold']}: {g['gap_hours']:.1f} hours")
    else:
        logger.info("✓ No significant gaps between folds")
    
    # Check for NaNs
    total_nans = sum(m["nan_count"] for m in fold_metadata)
    if total_nans > 0:
        logger.warning(f"Total NaN predictions: {total_nans}")
    else:
        logger.info("✓ No NaN predictions")
    
    # Merge predictions
    logger.info("\n=== Merging Predictions ===")
    merged = merge_predictions(all_predictions, overlap_strategy)
    
    # Rename column
    merged = merged.rename(columns={"y_pred": prediction_column_name})
    
    # Final output
    output_df = merged[["timestamp", prediction_column_name]]
    
    # Summary
    logger.info("\n=== Output Summary ===")
    logger.info(f"Output rows: {len(output_df)}")
    logger.info(f"Date range: {output_df['timestamp'].min().date()} to {output_df['timestamp'].max().date()}")
    logger.info(f"Prediction stats: min={output_df[prediction_column_name].min():.4f}, "
                f"max={output_df[prediction_column_name].max():.4f}, "
                f"mean={output_df[prediction_column_name].mean():.4f}")
    
    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Saved to: {output_csv}")
    
    return output_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract and merge walk-forward test predictions"
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to experiment directory containing run_* folders"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save merged predictions CSV"
    )
    parser.add_argument(
        "--column-name",
        type=str,
        required=True,
        help="Name for the prediction column (e.g., pred_vol_model)"
    )
    parser.add_argument(
        "--overlap-strategy",
        type=str,
        choices=["earliest", "latest", "average"],
        default="earliest",
        help="How to handle overlapping timestamps (default: earliest)"
    )
    
    args = parser.parse_args()
    
    if not args.experiment_dir.exists():
        logger.error(f"Experiment directory not found: {args.experiment_dir}")
        sys.exit(1)
    
    extract_walk_forward_predictions(
        experiment_dir=args.experiment_dir,
        prediction_column_name=args.column_name,
        output_csv=args.output,
        overlap_strategy=args.overlap_strategy,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Rescore walk-forward models on full timestamp range using original features.

This utility loads trained models from each fold and scores ALL timestamps 
in the test date range (not just those with non-NaN targets). This ensures
full coverage when using predictions as stacking features.

Usage:
    python training/utils/rescore_walk_forward_full_range.py \
        --experiment-dir "/path/to/walk_forward/experiment_name" \
        --features-csv "/path/to/derived_features_full.csv" \
        --output "/path/to/output/pred_model_name.csv" \
        --column-name "pred_drift_model" \
        --overlap-strategy earliest
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_timestamp_gaps(fold_metadata: List[Dict]) -> List[Dict]:
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
                "after_fold": prev["fold_name"],
                "before_fold": curr["fold_name"],
                "gap_hours": gap_hours,
            })
    
    return gaps


def load_original_predictions(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load pred_test.csv from run directory if it exists."""
    pred_file = run_dir / "pred_test.csv"
    if not pred_file.exists():
        return None
    return pd.read_csv(pred_file, parse_dates=["timestamp"])


def compare_with_original(rescore_preds: pd.DataFrame, original_preds: pd.DataFrame, fold_name: str) -> Dict:
    """
    Compare rescored predictions with original pred_test.csv.
    Returns comparison metrics.
    """
    # Merge on timestamp
    merged = rescore_preds.merge(
        original_preds[["timestamp", "y_pred"]], 
        on="timestamp", 
        suffixes=("_rescore", "_original"),
        how="outer",
        indicator=True
    )
    
    only_rescore = (merged["_merge"] == "left_only").sum()
    only_original = (merged["_merge"] == "right_only").sum()
    both = (merged["_merge"] == "both").sum()
    
    # Check prediction match on common timestamps
    common = merged[merged["_merge"] == "both"]
    if len(common) > 0:
        max_diff = (common["y_pred_rescore"] - common["y_pred_original"]).abs().max()
        predictions_match = max_diff < 1e-6
    else:
        max_diff = None
        predictions_match = None
    
    return {
        "fold_name": fold_name,
        "rescore_rows": len(rescore_preds),
        "original_rows": len(original_preds),
        "common_rows": both,
        "only_in_rescore": only_rescore,
        "only_in_original": only_original,
        "max_pred_diff": max_diff,
        "predictions_match": predictions_match,
    }


def load_walk_forward_config(splits_dir: Path) -> Dict:
    """Load walk_forward_config.json from splits directory."""
    config_path = splits_dir / "walk_forward_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"walk_forward_config.json not found in {splits_dir}")
    
    with open(config_path) as f:
        return json.load(f)


def discover_run_dirs(experiment_dir: Path) -> List[Path]:
    """Discover all run directories in the experiment folder."""
    run_dirs = []
    for d in experiment_dir.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            model_file = d / "model.cbm"
            if model_file.exists():
                run_dirs.append(d)
            else:
                logger.warning(f"Run dir {d.name} missing model.cbm, skipping")
    
    run_dirs.sort(key=lambda x: x.name)
    return run_dirs


def get_fold_info_from_config(experiment_dir: Path, fold_name: str) -> Optional[Dict]:
    """Get fold test boundaries from config_fold_XX.json."""
    config_path = experiment_dir / f"config_{fold_name}.json"
    if not config_path.exists():
        return None
    
    with open(config_path) as f:
        fold_config = json.load(f)
    
    # Get the splits directory
    training_splits_dir = fold_config.get("training_splits_dir")
    if not training_splits_dir:
        existing_dir = fold_config.get("split", {}).get("existing_dir", "")
        if existing_dir:
            training_splits_dir = str(Path(existing_dir).parent)
    
    if not training_splits_dir:
        return None
    
    return {
        "splits_dir": Path(training_splits_dir),
        "fold_name": fold_name,
        "features": fold_config.get("feature_selection", {}).get("include", []),
    }


def map_runs_to_folds(
    experiment_dir: Path, 
    run_dirs: List[Path]
) -> List[Tuple[Path, Dict]]:
    """
    Map run directories to their fold configurations.
    Returns list of (run_dir, fold_info) tuples.
    """
    # First, try to get splits_dir from any config_fold file
    splits_dir = None
    for i in range(1, 100):
        fold_name = f"fold_{i:02d}"
        fold_info = get_fold_info_from_config(experiment_dir, fold_name)
        if fold_info:
            splits_dir = fold_info["splits_dir"]
            break
    
    if not splits_dir:
        raise ValueError("Could not find splits directory from fold configs")
    
    # Load walk_forward_config.json to get test boundaries
    wf_config = load_walk_forward_config(splits_dir)
    folds_by_name = {f["name"]: f for f in wf_config["folds"]}
    
    # Map runs to folds by order (runs are sorted, same order as folds)
    mapped = []
    for i, run_dir in enumerate(run_dirs):
        fold_name = f"fold_{i+1:02d}"
        if fold_name not in folds_by_name:
            logger.warning(f"No fold config for {fold_name}, skipping run {run_dir.name}")
            continue
        
        fold_info = get_fold_info_from_config(experiment_dir, fold_name)
        if not fold_info:
            logger.warning(f"Could not load config for {fold_name}, skipping")
            continue
            
        fold_boundaries = folds_by_name[fold_name]
        fold_info["test_start"] = pd.Timestamp(fold_boundaries["test_start"])
        fold_info["test_end"] = pd.Timestamp(fold_boundaries["test_end"])
        
        mapped.append((run_dir, fold_info))
    
    return mapped


def load_model(run_dir: Path) -> Tuple:
    """Load CatBoost model and determine if it's classifier or regressor."""
    model_path = run_dir / "model.cbm"
    
    # Try loading as classifier first
    try:
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        is_classifier = True
    except Exception:
        model = CatBoostRegressor()
        model.load_model(str(model_path))
        is_classifier = False
    
    return model, is_classifier


def score_fold(
    run_dir: Path,
    fold_info: Dict,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score all timestamps in the fold's test range using the trained model.
    
    Returns DataFrame with timestamp and y_pred columns.
    """
    # Filter features to test date range
    test_start = fold_info["test_start"]
    test_end = fold_info["test_end"]
    
    mask = (features_df["timestamp"] >= test_start) & (features_df["timestamp"] < test_end)
    test_data = features_df[mask].copy()
    
    if len(test_data) == 0:
        logger.warning(f"No data in test range {test_start} to {test_end}")
        return pd.DataFrame(columns=["timestamp", "y_pred"])
    
    # Load model
    model, is_classifier = load_model(run_dir)
    
    # Get feature columns used by model
    model_features = model.feature_names_
    
    # Check for missing features
    missing = set(model_features) - set(test_data.columns)
    if missing:
        raise ValueError(f"Missing features in data: {missing}")
    
    # Prepare X
    X = test_data[model_features]
    
    # Score
    if is_classifier:
        y_pred = model.predict_proba(X)[:, 1]  # Probability of class 1
    else:
        y_pred = model.predict(X)
    
    result = pd.DataFrame({
        "timestamp": test_data["timestamp"].values,
        "y_pred": y_pred,
    })
    
    return result


def merge_predictions(
    all_predictions: List[pd.DataFrame],
    overlap_strategy: str
) -> pd.DataFrame:
    """Merge predictions from all folds, handling overlapping timestamps."""
    combined = pd.concat(all_predictions, ignore_index=True)
    
    dup_count = combined.duplicated(subset=["timestamp"], keep=False).sum()
    unique_timestamps = combined["timestamp"].nunique()
    total_rows = len(combined)
    
    logger.info(f"Total rows: {total_rows}, Unique timestamps: {unique_timestamps}")
    if dup_count > 0:
        logger.info(f"Overlapping predictions: {dup_count} rows")
    
    if overlap_strategy == "earliest":
        merged = combined.drop_duplicates(subset=["timestamp"], keep="first")
    elif overlap_strategy == "latest":
        merged = combined.drop_duplicates(subset=["timestamp"], keep="last")
    elif overlap_strategy == "average":
        merged = combined.groupby("timestamp", as_index=False).agg({"y_pred": "mean"})
    else:
        raise ValueError(f"Unknown overlap_strategy: {overlap_strategy}")
    
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


def rescore_walk_forward_full_range(
    experiment_dir: Path,
    features_csv: Path,
    prediction_column_name: str,
    output_csv: Path,
    overlap_strategy: str = "earliest"
) -> pd.DataFrame:
    """
    Rescore all walk-forward folds using the original features CSV.
    
    This ensures full timestamp coverage (no gaps from NaN targets).
    """
    logger.info(f"Rescoring experiment: {experiment_dir}")
    logger.info(f"Features CSV: {features_csv}")
    logger.info(f"Overlap strategy: {overlap_strategy}")
    
    # Discover run directories
    run_dirs = discover_run_dirs(experiment_dir)
    if not run_dirs:
        raise ValueError(f"No run directories with model.cbm found in {experiment_dir}")
    
    logger.info(f"Found {len(run_dirs)} run directories")
    
    # Map runs to folds
    mapped = map_runs_to_folds(experiment_dir, run_dirs)
    logger.info(f"Mapped {len(mapped)} runs to folds")
    
    # Load features (once)
    logger.info("Loading features CSV...")
    features_df = pd.read_csv(features_csv, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(features_df)} rows, {len(features_df.columns)} columns")
    
    # Score each fold and collect metadata
    all_predictions = []
    fold_metadata = []
    comparison_results = []
    
    for run_dir, fold_info in mapped:
        fold_name = fold_info["fold_name"]
        test_start = fold_info["test_start"]
        test_end = fold_info["test_end"]
        
        # Calculate expected rows
        expected_hours = int((test_end - test_start).total_seconds() / 3600)
        
        logger.info(f"Scoring {fold_name}: {test_start.date()} to {test_end.date()}")
        
        preds = score_fold(run_dir, fold_info, features_df)
        
        if len(preds) > 0:
            logger.info(f"  Generated {len(preds)} predictions (expected ~{expected_hours})")
            all_predictions.append(preds)
            
            # Collect metadata for gap detection
            fold_metadata.append({
                "fold_name": fold_name,
                "run_name": run_dir.name,
                "row_count": len(preds),
                "expected_rows": expected_hours,
                "min_timestamp": preds["timestamp"].min(),
                "max_timestamp": preds["timestamp"].max(),
            })
            
            # Compare with original pred_test.csv
            original_preds = load_original_predictions(run_dir)
            if original_preds is not None:
                comp = compare_with_original(preds, original_preds, fold_name)
                comparison_results.append(comp)
        else:
            logger.warning(f"  No predictions generated for {fold_name}")
    
    if not all_predictions:
        raise ValueError("No predictions generated from any fold")
    
    # === Sanity Checks ===
    logger.info("\n=== Sanity Checks ===")
    
    # Gap detection
    gaps = check_timestamp_gaps(fold_metadata)
    if gaps:
        logger.warning("Timestamp gaps detected between folds:")
        for g in gaps:
            logger.warning(f"  {g['after_fold']} -> {g['before_fold']}: {g['gap_hours']:.1f} hours")
    else:
        logger.info("✓ No significant gaps between folds")
    
    # Coverage report
    logger.info("\n=== Coverage Report ===")
    total_expected = sum(m["expected_rows"] for m in fold_metadata)
    total_actual = sum(m["row_count"] for m in fold_metadata)
    coverage_pct = (total_actual / total_expected * 100) if total_expected > 0 else 0
    logger.info(f"Total expected rows: {total_expected}")
    logger.info(f"Total actual rows:   {total_actual}")
    logger.info(f"Coverage:            {coverage_pct:.1f}%")
    
    # Per-fold coverage issues
    coverage_issues = [m for m in fold_metadata if abs(m["row_count"] - m["expected_rows"]) > 1]
    if coverage_issues:
        logger.info("Folds with coverage differences:")
        for m in coverage_issues:
            diff = m["row_count"] - m["expected_rows"]
            logger.info(f"  {m['fold_name']}: {m['row_count']}/{m['expected_rows']} ({diff:+d})")
    
    # Comparison with original pred_test.csv
    if comparison_results:
        logger.info("\n=== Comparison with Original pred_test.csv ===")
        all_match = all(c["predictions_match"] for c in comparison_results if c["predictions_match"] is not None)
        total_extra = sum(c["only_in_rescore"] for c in comparison_results)
        total_missing = sum(c["only_in_original"] for c in comparison_results)
        
        if all_match:
            logger.info("✓ All predictions match original pred_test.csv on common timestamps")
        else:
            mismatched = [c["fold_name"] for c in comparison_results if c["predictions_match"] is False]
            logger.warning(f"Prediction mismatches in folds: {mismatched}")
        
        logger.info(f"Extra rows in rescore (not in original):   {total_extra}")
        logger.info(f"Missing rows in rescore (only in original): {total_missing}")
    
    # Merge
    logger.info("\n=== Merging Predictions ===")
    merged = merge_predictions(all_predictions, overlap_strategy)
    
    # Rename column
    merged = merged.rename(columns={"y_pred": prediction_column_name})
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
        description="Rescore walk-forward models on full timestamp range"
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to experiment directory containing run_* folders"
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help="Path to original features CSV (full range)"
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
        help="Name for the prediction column (e.g., pred_drift_model)"
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
    
    if not args.features_csv.exists():
        logger.error(f"Features CSV not found: {args.features_csv}")
        sys.exit(1)
    
    rescore_walk_forward_full_range(
        experiment_dir=args.experiment_dir,
        features_csv=args.features_csv,
        prediction_column_name=args.column_name,
        output_csv=args.output,
        overlap_strategy=args.overlap_strategy,
    )


if __name__ == "__main__":
    main()

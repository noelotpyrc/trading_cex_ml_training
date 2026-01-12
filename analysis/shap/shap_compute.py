#!/usr/bin/env python3
"""
CLI tool to compute SHAP values for trained models.

Supports both LightGBM and CatBoost models, loading features from DuckDB or CSV.

Usage:
    python analysis/shap/shap_compute.py --run-name <mlflow_run_name> --start 2025-01-01 --end 2025-12-31
    
    # Or with explicit paths (DuckDB):
    python analysis/shap/shap_compute.py \\
        --model-path /path/to/model.cbm \\
        --model-type catboost \\
        --start 2025-01-01 --end 2025-12-31
    
    # Or with CSV features (for regression models):
    python analysis/shap/shap_compute.py \\
        --model-path /path/to/model.cbm \\
        --model-type catboost \\
        --features-csv /path/to/merged_features_targets.csv \\
        --start 2025-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from shap_utils import (
    FEATURE_DB_PATH,
    MLFLOW_TRACKING_URI,
    SHAP_RESULTS_DIR,
    ShapMetadata,
    load_features_from_duckdb,
    load_model_info_from_mlflow,
    save_shap_results,
)


logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_lightgbm_model(model_path: Path):
    """Load a LightGBM booster from model.txt."""
    import lightgbm as lgb
    return lgb.Booster(model_file=str(model_path))


def load_catboost_model(model_path: Path):
    """Load a CatBoost model from .cbm or .joblib file.
    
    Automatically detects if it's a classifier or regressor.
    """
    if model_path.suffix == ".cbm":
        # Try loading as classifier first, then regressor
        try:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(str(model_path))
            return model
        except Exception:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor()
            model.load_model(str(model_path))
            return model
    elif model_path.suffix == ".joblib":
        import joblib
        return joblib.load(model_path)
    else:
        raise ValueError(f"Unknown CatBoost model format: {model_path.suffix}")



def get_model_features(model, model_type: str) -> List[str]:
    """Get feature names from model."""
    if model_type == "lightgbm":
        return list(model.feature_name())
    elif model_type == "catboost":
        return list(model.feature_names_)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def is_classifier(model) -> bool:
    """Check if model is a classifier (has predict_proba)."""
    return hasattr(model, 'predict_proba')


def load_features_from_csv(
    csv_path: Path,
    start_ts: str,
    end_ts: str,
    feature_names: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load features from a CSV file.
    
    Args:
        csv_path: Path to CSV file with features
        start_ts: Start timestamp (ISO format)
        end_ts: End timestamp (ISO format)
        feature_names: List of feature columns to load
    
    Returns:
        (features_df, meta_df) where meta_df contains timestamp column
    """
    logger.info("Loading features from CSV: %s", csv_path)
    
    df = pd.read_csv(csv_path)
    
    # Parse timestamp column
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    else:
        raise ValueError("CSV must have a 'timestamp' column")
    
    # Filter by date range
    start_dt = pd.to_datetime(start_ts, utc=True)
    end_dt = pd.to_datetime(end_ts, utc=True)
    df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
    
    logger.info("Filtered to %d rows in date range [%s, %s]", len(df), start_ts, end_ts)
    
    if len(df) == 0:
        raise ValueError(f"No data found in date range {start_ts} to {end_ts}")
    
    # Check for missing features
    missing = set(feature_names) - set(df.columns)
    if missing:
        raise ValueError(f"Features missing from CSV: {missing}")
    
    meta_df = df[['timestamp']].copy()
    features_df = df[feature_names].copy()
    
    return features_df, meta_df


def compute_shap_values(
    model,
    features_df: pd.DataFrame,
    model_type: str,
    class_index: Optional[int] = 1,
    background_sample: Optional[int] = None,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SHAP values for the given model and features.
    
    Returns:
        Tuple of (shap_values, base_values, predictions)
    """
    import shap
    
    # Build explainer
    if background_sample and len(features_df) > background_sample:
        background = features_df.sample(n=background_sample, random_state=random_seed)
        explainer = shap.TreeExplainer(model, data=background)
    else:
        explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(features_df)
    expected_value = explainer.expected_value
    
    # Handle multi-class output (select positive class for binary classification)
    if isinstance(shap_values, list):
        if class_index is None:
            class_index = 1 if len(shap_values) > 1 else 0
        if class_index >= len(shap_values):
            raise IndexError(f"class_index {class_index} out of bounds for {len(shap_values)} classes")
        shap_array = np.asarray(shap_values[class_index])
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            base_val = np.asarray(expected_value[class_index])
        else:
            base_val = np.asarray(expected_value)
    else:
        shap_array = np.asarray(shap_values)
        base_val = np.asarray(expected_value)
    
    # Expand base values to match number of samples
    if base_val.ndim == 0:
        base_values = np.full(len(features_df), float(base_val))
    elif base_val.size == 1:
        base_values = np.full(len(features_df), float(base_val.ravel()[0]))
    else:
        base_values = base_val.astype(float)
    
    # Get predictions
    if model_type == "lightgbm":
        predictions = model.predict(features_df)
    else:  # catboost
        if is_classifier(model):
            predictions = model.predict_proba(features_df)[:, 1]
        else:
            predictions = model.predict(features_df)
    
    return shap_array, base_values, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SHAP values for a trained model")
    
    # Model identification (either via MLflow or explicit paths)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--run-name", type=str, help="MLflow run name to load model from")
    model_group.add_argument("--model-path", type=Path, help="Explicit path to model file")
    
    # Required when using --model-path
    parser.add_argument("--model-type", choices=["lightgbm", "catboost"], 
                        help="Model type (required with --model-path)")
    
    # Feature loading
    parser.add_argument("--features-csv", type=Path, default=None,
                        help="Path to CSV file with features (alternative to DuckDB)")
    parser.add_argument("--feature-db", type=Path, default=FEATURE_DB_PATH,
                        help="Path to feature DuckDB database")
    parser.add_argument("--feature-table", type=str, default=None,
                        help="Override table name (default: auto-select based on model type)")
    
    # Date range
    parser.add_argument("--start", type=str, required=True, 
                        help="Start date (YYYY-MM-DD or ISO timestamp)")
    parser.add_argument("--end", type=str, required=True,
                        help="End date (YYYY-MM-DD or ISO timestamp)")
    
    # SHAP options
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max rows to compute SHAP for (sample if more)")
    parser.add_argument("--background-sample", type=int, default=100,
                        help="Background sample size for SHAP explainer")
    parser.add_argument("--class-index", type=int, default=1,
                        help="Class index for binary classification")
    parser.add_argument("--random-seed", type=int, default=42)
    
    # Output
    parser.add_argument("--output-dir", type=Path, default=SHAP_RESULTS_DIR,
                        help="Directory to save SHAP results")
    
    # MLflow
    parser.add_argument("--tracking-uri", type=str, default=MLFLOW_TRACKING_URI,
                        help="MLflow tracking URI")
    
    parser.add_argument("--log-level", default="INFO")
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    start_time = time.time()
    
    # Resolve model info
    if args.run_name:
        logger.info("Loading model info from MLflow for run: %s", args.run_name)
        model_info = load_model_info_from_mlflow(args.run_name, args.tracking_uri)
        model_path = Path(model_info["model_path"])
        model_type = model_info["model_type"]
        run_name = args.run_name
        feature_names = model_info.get("feature_include") or None
    else:
        if not args.model_type:
            raise ValueError("--model-type is required when using --model-path")
        model_path = args.model_path
        model_type = args.model_type
        run_name = model_path.parent.name  # Use directory name, not model filename
        feature_names = None
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info("Model: %s (%s)", model_path, model_type)
    
    # Load model
    if model_type == "lightgbm":
        model = load_lightgbm_model(model_path)
    else:
        model = load_catboost_model(model_path)
    
    # Get feature names from model if not provided
    if feature_names is None or len(feature_names) == 0:
        feature_names = get_model_features(model, model_type)
    
    logger.info("Model has %d features", len(feature_names))
    
    # Parse date range
    start_ts = pd.to_datetime(args.start, utc=True).isoformat()
    end_ts = pd.to_datetime(args.end, utc=True).isoformat()
    
    # Load features (from CSV or DuckDB)
    if args.features_csv:
        if not args.features_csv.exists():
            raise FileNotFoundError(f"Features CSV not found: {args.features_csv}")
        logger.info("Loading features from CSV: %s", args.features_csv)
        features_df, meta_df = load_features_from_csv(
            csv_path=args.features_csv,
            start_ts=start_ts,
            end_ts=end_ts,
            feature_names=feature_names,
        )
        feature_source = str(args.features_csv)
    else:
        # Determine feature table
        if args.feature_table:
            feature_table = args.feature_table
        elif model_type == "lightgbm":
            feature_table = "features"
        else:  # catboost
            feature_table = "derived_features_inference"
        
        logger.info("Loading features from DuckDB: %s.%s", args.feature_db, feature_table)
        features_df, meta_df = load_features_from_duckdb(
            db_path=args.feature_db,
            table=feature_table,
            start_ts=start_ts,
            end_ts=end_ts,
            feature_names=feature_names,
        )
        feature_source = f"{args.feature_db}:{feature_table}"
    
    logger.info("Loaded %d samples with %d features", len(features_df), len(features_df.columns))
    
    # Sample if needed
    if args.max_rows and len(features_df) > args.max_rows:
        indices = features_df.sample(n=args.max_rows, random_state=args.random_seed).index
        features_df = features_df.loc[indices].reset_index(drop=True)
        meta_df = meta_df.loc[indices].reset_index(drop=True)
        logger.info("Sampled down to %d rows", len(features_df))
    
    # Align features to model
    model_feature_names = get_model_features(model, model_type)
    missing = set(model_feature_names) - set(features_df.columns)
    if missing:
        raise ValueError(f"Features missing from data: {missing}")
    
    # Reorder columns to match model
    features_df = features_df[model_feature_names]
    
    # Compute SHAP values
    logger.info("Computing SHAP values...")
    shap_values, base_values, predictions = compute_shap_values(
        model=model,
        features_df=features_df,
        model_type=model_type,
        class_index=args.class_index,
        background_sample=args.background_sample,
        random_seed=args.random_seed,
    )
    
    computation_time = time.time() - start_time
    logger.info("SHAP computation completed in %.1f seconds", computation_time)
    
    # Create metadata
    metadata = ShapMetadata(
        run_name=run_name,
        model_type=model_type,
        model_path=str(model_path),
        feature_table=feature_source,  # Can be CSV path or DB:table
        start_date=args.start,
        end_date=args.end,
        num_samples=len(features_df),
        num_features=len(model_feature_names),
        feature_names=model_feature_names,
        computation_time_seconds=computation_time,
        created_at=datetime.now().isoformat(),
    )
    
    # Save results
    result_dir = save_shap_results(
        shap_values=shap_values,
        base_values=base_values,
        predictions=predictions,
        feature_names=model_feature_names,
        timestamps=meta_df["timestamp"],
        metadata=metadata,
        output_dir=args.output_dir,
    )
    
    logger.info("Results saved to: %s", result_dir)


if __name__ == "__main__":
    main()

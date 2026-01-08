#!/usr/bin/env python3
"""Shared utilities for SHAP analysis: feature loading, MLflow integration, result storage."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# Default paths
FEATURE_DB_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb")
OHLCV_DB_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb")
SHAP_RESULTS_DIR = Path("/Volumes/Extreme SSD/trading_data/cex/shap_results")
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


@dataclass
class ShapMetadata:
    """Metadata for a SHAP computation run."""
    run_name: str
    model_type: str  # "lightgbm" or "catboost"
    model_path: str
    feature_table: str  # "features" or "derived_features_inference"
    start_date: str
    end_date: str
    num_samples: int
    num_features: int
    feature_names: List[str]
    computation_time_seconds: float
    created_at: str


def load_features_from_duckdb(
    db_path: Path,
    table: str,
    start_ts: str,
    end_ts: str,
    feature_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load features from DuckDB for SHAP computation.
    
    Args:
        db_path: Path to DuckDB database
        table: Table name ("features" or "derived_features_inference")
        start_ts: Start timestamp (ISO format)
        end_ts: End timestamp (ISO format)
        feature_names: Optional list of feature names to extract (if None, extract all)
    
    Returns:
        Tuple of (features_df, meta_df) where:
        - features_df: DataFrame with feature columns aligned for model
        - meta_df: DataFrame with timestamp column
    """
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("duckdb is required for feature loading") from exc
    
    query = f"""
        SELECT ts, features 
        FROM {table} 
        WHERE ts >= ?
          AND ts <= ?
        ORDER BY ts ASC
    """
    
    logger.info("Loading features from %s.%s, range=[%s, %s]", 
                db_path, table, start_ts, end_ts)
    
    with duckdb.connect(str(db_path)) as con:
        con.execute("SET TimeZone='UTC';")
        df = con.execute(query, [start_ts, end_ts]).fetch_df()
    
    if df.empty:
        raise ValueError(f"No features found in {table} for date range [{start_ts}, {end_ts}]")
    
    logger.info("Retrieved %d rows from DuckDB", len(df))
    
    # Parse JSON features
    feature_rows = []
    all_feature_names = None
    for payload in df["features"]:
        data = json.loads(payload) if isinstance(payload, str) else payload
        if all_feature_names is None:
            all_feature_names = list(data.keys())
        feature_rows.append(data)
    
    features_df = pd.DataFrame(feature_rows)
    
    # Filter to requested features if specified
    if feature_names is not None:
        missing = set(feature_names) - set(features_df.columns)
        if missing:
            logger.warning("Missing features in data: %s", missing)
        available = [f for f in feature_names if f in features_df.columns]
        features_df = features_df[available]
    
    meta_df = pd.DataFrame({
        "timestamp": pd.to_datetime(df["ts"], utc=True)
    })
    
    return features_df, meta_df


def load_model_info_from_mlflow(
    run_name: str,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> Dict[str, Any]:
    """
    Query MLflow for model information by run name.
    
    Returns dict with:
    - run_id: MLflow run ID
    - run_name: Run name
    - model_path: Path to model file
    - model_type: "lightgbm" or "catboost"
    - run_dir: Directory containing model artifacts
    - feature_include: List of features used in training (if available)
    """
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("mlflow is required for model discovery") from exc
    
    mlflow.set_tracking_uri(tracking_uri)
    
    # Search for run by name across all experiments
    experiments = mlflow.search_experiments()
    
    for exp in experiments:
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            max_results=1,
        )
        if not runs.empty:
            run = runs.iloc[0]
            break
    else:
        raise ValueError(f"Run not found in MLflow: {run_name}")
    
    # Extract model info from run params
    run_dir = run.get("params.run_dir", "")
    model_path = run.get("params.model_path", "")
    model_type = run.get("params.model_type", "")
    
    # Infer model type from path if not set
    if not model_type and model_path:
        if model_path.endswith(".txt"):
            model_type = "lightgbm"
        elif model_path.endswith(".cbm") or model_path.endswith(".joblib"):
            model_type = "catboost"
    
    # Get feature list
    feature_include = run.get("params.feature_selection.include", "")
    if feature_include:
        feature_include = [f.strip() for f in feature_include.split(",")]
    else:
        feature_include = []
    
    return {
        "run_id": run["run_id"],
        "run_name": run_name,
        "model_path": model_path,
        "model_type": model_type,
        "run_dir": run_dir,
        "feature_include": feature_include,
    }


def save_shap_results(
    shap_values: np.ndarray,
    base_values: np.ndarray,
    predictions: np.ndarray,
    feature_names: List[str],
    timestamps: pd.Series,
    metadata: ShapMetadata,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save SHAP computation results to disk.
    
    Creates directory: {output_dir}/{run_name}_{start}_{end}/
    With files:
    - shap_values.parquet: Full SHAP matrix with timestamps
    - shap_summary.csv: Feature importance ranking
    - shap_metadata.json: Computation metadata
    
    Returns the output directory path.
    """
    if output_dir is None:
        output_dir = SHAP_RESULTS_DIR
    
    # Create directory name from run name and date range
    dir_name = f"{metadata.run_name}_{metadata.start_date}_{metadata.end_date}"
    result_dir = output_dir / dir_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save SHAP values as parquet
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.insert(0, "timestamp", timestamps.values)
    shap_df.insert(1, "base_value", base_values)
    shap_df.insert(2, "prediction", predictions)
    
    shap_path = result_dir / "shap_values.parquet"
    shap_df.to_parquet(shap_path, index=False)
    logger.info("Saved SHAP values to %s", shap_path)
    
    # Save summary statistics
    summary = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "std_shap": shap_values.std(axis=0),
        "mean_shap": shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    
    summary_path = result_dir / "shap_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved SHAP summary to %s", summary_path)
    
    # Save metadata
    metadata_path = result_dir / "shap_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(asdict(metadata), f, indent=2)
    logger.info("Saved metadata to %s", metadata_path)
    
    return result_dir


def load_shap_results(result_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, ShapMetadata]:
    """
    Load previously computed SHAP results.
    
    Returns:
        Tuple of (shap_df, summary_df, metadata)
    """
    shap_path = result_dir / "shap_values.parquet"
    summary_path = result_dir / "shap_summary.csv"
    metadata_path = result_dir / "shap_metadata.json"
    
    if not shap_path.exists():
        raise FileNotFoundError(f"SHAP values not found: {shap_path}")
    
    shap_df = pd.read_parquet(shap_path)
    if "timestamp" in shap_df.columns:
        shap_df["timestamp"] = pd.to_datetime(shap_df["timestamp"], utc=True)
    
    summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    
    with open(metadata_path, "r") as f:
        meta_dict = json.load(f)
    metadata = ShapMetadata(**meta_dict)
    
    return shap_df, summary_df, metadata


def get_available_results(results_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    List all available SHAP result directories.
    
    Returns list of dicts with:
    - dir_name: Directory name
    - path: Full path
    - run_name: Model run name
    - start_date: Start date
    - end_date: End date
    - metadata: Full metadata if available
    """
    if results_dir is None:
        results_dir = SHAP_RESULTS_DIR
    
    if not results_dir.exists():
        return []
    
    results = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        
        metadata_path = d / "shap_metadata.json"
        if not metadata_path.exists():
            continue
        
        try:
            with open(metadata_path, "r") as f:
                meta_dict = json.load(f)
            
            results.append({
                "dir_name": d.name,
                "path": d,
                "run_name": meta_dict.get("run_name", ""),
                "start_date": meta_dict.get("start_date", ""),
                "end_date": meta_dict.get("end_date", ""),
                "model_type": meta_dict.get("model_type", ""),
                "num_samples": meta_dict.get("num_samples", 0),
                "metadata": meta_dict,
            })
        except Exception as e:
            logger.warning("Failed to read metadata from %s: %s", d, e)
    
    return results


def get_feature_columns(shap_df: pd.DataFrame) -> List[str]:
    """Get feature column names from SHAP DataFrame (excluding meta columns)."""
    meta_cols = {"timestamp", "base_value", "prediction", "y_true", "y_pred", "shap_sum", "computed_pred"}
    return [c for c in shap_df.columns if c not in meta_cols and np.issubdtype(shap_df[c].dtype, np.number)]


def load_targets_from_duckdb(
    start_ts: str,
    end_ts: str,
    target_key: str = "y_tp_before_sl_u0.04_d0.02_24h",
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load binary targets from the OHLCV database for segmentation.
    
    Args:
        start_ts: Start timestamp (ISO format)
        end_ts: End timestamp (ISO format)
        target_key: Target key to load (default: y_tp_before_sl_u0.04_d0.02_24h)
        db_path: Optional path to OHLCV DuckDB
    
    Returns:
        DataFrame with 'timestamp' and 'y_true' columns
    """
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("duckdb is required for target loading") from exc
    
    if db_path is None:
        db_path = OHLCV_DB_PATH
    
    query = """
        SELECT timestamp, target_value as y_true
        FROM targets
        WHERE target_key = ?
          AND timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp ASC
    """
    
    logger.info("Loading targets from %s for key=%s, range=[%s, %s]", 
                db_path, target_key, start_ts, end_ts)
    
    with duckdb.connect(str(db_path)) as con:
        con.execute("SET TimeZone='UTC';")
        df = con.execute(query, [target_key, start_ts, end_ts]).fetch_df()
    
    if df.empty:
        logger.warning("No targets found for key=%s in date range", target_key)
        return df
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["y_true"] = df["y_true"].astype(int)
    
    logger.info("Retrieved %d target rows", len(df))
    return df

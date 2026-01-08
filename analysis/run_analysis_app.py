#!/usr/bin/env python3
"""
Single Run Analysis App

Streamlit app to analyze a specific MLflow registered run:
- Show basic stats (metrics)
- Show AUC test in a monthly time series manner
- Adjustable month filtering (min days requirement)

Usage:
    streamlit run analysis/run_analysis_app.py
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import mlflow
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import skew, kurtosis, wasserstein_distance, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import duckdb

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

TRACKING_URI = "http://127.0.0.1:5001"

# DuckDB paths for Real-Time Analysis
TARGETS_DB_PATH = "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb"
PREDICTIONS_DB_PATH = "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction.duckdb"
PREDICTIONS_CLASSIFIER_DB_PATH = "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction_classifier.duckdb"
TARGET_KEY = "y_tp_before_sl_u0.04_d0.02_24h"  # tp4sl2 target
REALTIME_MIN_DATE = pd.Timestamp("2025-04-01")
BENCHMARK_RUN_NAME = "run_20251102_231428_lgbm_y_tp_before_sl_u0.04_d0.02_24h_binary"
BENCHMARK_RUN_DIR = "/Volumes/Extreme SSD/trading_data/cex/models/binance_btcusdt_perp_1h_original/run_20251102_231428_lgbm_y_tp_before_sl_u0.04_d0.02_24h_binary"

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading Functions
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def get_all_experiments() -> List[Dict[str, Any]]:
    """Get all experiments from MLflow."""
    mlflow.set_tracking_uri(TRACKING_URI)
    experiments = mlflow.search_experiments()
    return [
        {"experiment_id": exp.experiment_id, "name": exp.name}
        for exp in experiments
        if exp.name != "Default"
    ]


@st.cache_data
def get_runs_for_experiment(experiment_id: str) -> pd.DataFrame:
    """Get all runs for a given experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time ASC"],
    )
    return runs


def load_run_artifacts(run_dir: str) -> Dict[str, Any]:
    """Load artifacts from run directory."""
    run_path = Path(run_dir)
    artifacts = {}
    
    # Load metrics.json
    metrics_path = run_path / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            artifacts["metrics"] = json.load(f)
    
    # Load feature_importance.csv
    fi_path = run_path / "feature_importance.csv"
    if fi_path.exists():
        artifacts["feature_importance"] = pd.read_csv(fi_path)
    
    # Load best_params.json
    params_path = run_path / "best_params.json"
    if params_path.exists():
        with open(params_path, "r") as f:
            artifacts["best_params"] = json.load(f)
            
    # Load pred_train.csv
    pred_train_path = run_path / "pred_train.csv"
    if pred_train_path.exists():
        artifacts["pred_train"] = pd.read_csv(pred_train_path)
        
    # Load pred_val.csv
    pred_val_path = run_path / "pred_val.csv"
    if pred_val_path.exists():
        artifacts["pred_val"] = pd.read_csv(pred_val_path)

    # Load pred_test.csv
    pred_path = run_path / "pred_test.csv"
    if pred_path.exists():
        artifacts["pred_test"] = pd.read_csv(pred_path)
    
    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Functions
# ─────────────────────────────────────────────────────────────────────────────

def calculate_monthly_auc(pred_df: pd.DataFrame, min_days: int = 20) -> pd.DataFrame:
    """Calculate AUC and prAUC per month, filtering by minimum days of data."""
    if "timestamp" not in pred_df.columns or "y_true" not in pred_df.columns or "y_pred" not in pred_df.columns:
        return pd.DataFrame()

    df = pred_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["month"] = df["timestamp"].dt.to_period("M")
    
    results = []
    for month, group in df.groupby("month"):
        # Count unique days in this month
        unique_days = group["timestamp"].dt.date.nunique()
        
        if unique_days >= min_days:
            try:
                # Calculate metrics
                # Check if we have both classes for AUC
                if group["y_true"].nunique() > 1:
                    auc = roc_auc_score(group["y_true"], group["y_pred"])
                    
                    # prAUC for Class 1 (Positive)
                    prauc1 = average_precision_score(group["y_true"], group["y_pred"])
                    
                    # prAUC for Class 0 (Negative) - invert true labels and pred probabilities
                    prauc0 = average_precision_score(1 - group["y_true"], 1 - group["y_pred"])
                    
                    y_true_rate = group["y_true"].mean()  # Positive class rate
                    
                    results.append({
                        "Month": str(month),
                        "AUC": auc,
                        "prAUC_1": prauc1,
                        "prAUC_0": prauc0,
                        "Y_True_Rate": y_true_rate,
                        "Days": unique_days,
                        "Samples": len(group)
                    })
            except Exception:
                pass
                
    return pd.DataFrame(results)


def calculate_rolling_auc(df: pd.DataFrame, window_days: int = 90, step_days: int = 1) -> pd.DataFrame:
    """Calculate rolling AUC with a daily step."""
    if df.empty:
        return pd.DataFrame()
        
    df = df.sort_values("timestamp")
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    
    # Generate reference dates (daily)
    ref_dates = pd.date_range(start=min_ts, end=max_ts, freq=f"{step_days}D")
    
    results = []
    
    # Pre-compute date bounds for efficiency
    # But for simplicity and correctness with arbitrary gaps, we'll slice by time
    # Optimization: Filter df once? No, it's sliding.
    
    # We can use searchsorted if memory is an issue, but standard boolean masking 
    # might be slow for many steps. 
    # Let's iterate. To be faster, we can just use the subset.
    
    for current_date in ref_dates:
        start_date = current_date - pd.Timedelta(days=window_days)
        
        # Select window
        mask = (df["timestamp"] > start_date) & (df["timestamp"] <= current_date)
        window_df = df.loc[mask]
        
        if len(window_df) > 100 and window_df["y_true"].nunique() > 1:
            try:
                auc = roc_auc_score(window_df["y_true"], window_df["y_pred"])
                results.append({
                    "Date": current_date,
                    "AUC": auc,
                    "Samples": len(window_df)
                })
            except Exception:
                pass
                
    return pd.DataFrame(results)


def load_realtime_data(run_name: str, start_date: pd.Timestamp) -> pd.DataFrame:
    """Load real-time predictions and targets from DuckDB, join on timestamp."""
    # Search both prediction DBs
    pred_dbs = [PREDICTIONS_DB_PATH, PREDICTIONS_CLASSIFIER_DB_PATH]
    preds_df = pd.DataFrame()
    
    for db_path in pred_dbs:
        try:
            pred_conn = duckdb.connect(db_path, read_only=True)
            # Find model path containing run_name
            model_paths_df = pred_conn.execute(
                "SELECT DISTINCT model_path FROM predictions WHERE model_path LIKE ?",
                [f"%{run_name}%"]
            ).fetchdf()
            
            if not model_paths_df.empty:
                model_path = model_paths_df.iloc[0]["model_path"]
                preds_df = pred_conn.execute(
                    "SELECT ts AS timestamp, y_pred FROM predictions WHERE model_path = ? AND ts >= ?",
                    [model_path, start_date]
                ).fetchdf()
                pred_conn.close()
                if not preds_df.empty:
                    break
            else:
                pred_conn.close()
        except Exception:
            continue
    
    if preds_df.empty:
        return pd.DataFrame()
    
    # Get targets
    try:
        target_conn = duckdb.connect(TARGETS_DB_PATH, read_only=True)
        targets_df = target_conn.execute(
            "SELECT timestamp, target_value AS y_true FROM targets WHERE target_key = ? AND timestamp >= ?",
            [TARGET_KEY, start_date]
        ).fetchdf()
        target_conn.close()
    except Exception as e:
        st.warning(f"Error loading targets: {e}")
        return pd.DataFrame()
    
    if targets_df.empty:
        return pd.DataFrame()
    
    # Convert timestamps
    preds_df["timestamp"] = pd.to_datetime(preds_df["timestamp"])
    targets_df["timestamp"] = pd.to_datetime(targets_df["timestamp"])
    
    # Make tz-naive if needed
    if preds_df["timestamp"].dt.tz is not None:
        preds_df["timestamp"] = preds_df["timestamp"].dt.tz_localize(None)
    if targets_df["timestamp"].dt.tz is not None:
        targets_df["timestamp"] = targets_df["timestamp"].dt.tz_localize(None)
    
    # Inner join on timestamp
    merged_df = pd.merge(preds_df, targets_df, on="timestamp", how="inner")
    return merged_df


def load_predictions_only(run_name: str, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None) -> pd.DataFrame:
    """Load predictions only (no targets) from DuckDB for a given run."""
    pred_dbs = [PREDICTIONS_DB_PATH, PREDICTIONS_CLASSIFIER_DB_PATH]
    
    for db_path in pred_dbs:
        try:
            pred_conn = duckdb.connect(db_path, read_only=True)
            model_paths_df = pred_conn.execute(
                "SELECT DISTINCT model_path FROM predictions WHERE model_path LIKE ?",
                [f"%{run_name}%"]
            ).fetchdf()
            
            if not model_paths_df.empty:
                model_path = model_paths_df.iloc[0]["model_path"]
                
                # Build query with optional date filters
                query = "SELECT ts AS timestamp, y_pred FROM predictions WHERE model_path = ?"
                params = [model_path]
                
                if start_date is not None:
                    query += " AND ts >= ?"
                    params.append(start_date)
                if end_date is not None:
                    query += " AND ts <= ?"
                    params.append(end_date)
                
                preds_df = pred_conn.execute(query, params).fetchdf()
                pred_conn.close()
                
                if not preds_df.empty:
                    preds_df["timestamp"] = pd.to_datetime(preds_df["timestamp"])
                    if preds_df["timestamp"].dt.tz is not None:
                        preds_df["timestamp"] = preds_df["timestamp"].dt.tz_localize(None)
                    return preds_df
            else:
                pred_conn.close()
        except Exception:
            continue
    
    return pd.DataFrame()


@st.cache_data(ttl=300)
def get_available_models_from_db() -> List[str]:
    """Get all distinct model names from the classifier predictions DB."""
    try:
        conn = duckdb.connect(PREDICTIONS_CLASSIFIER_DB_PATH, read_only=True)
        result = conn.execute("SELECT DISTINCT model_path FROM predictions").fetchdf()
        conn.close()
        
        if result.empty:
            return []
        
        # Extract run name from full path
        model_names = []
        for path in result["model_path"]:
            # Extract the run folder name from the path
            parts = path.split("/")
            for part in parts:
                if part.startswith("run_"):
                    model_names.append(part)
                    break
        
        return sorted(set(model_names), reverse=True)  # Most recent first
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Run Analysis",
        page_icon="📊",
        layout="wide",
    )
    
    st.title("📊 Single Run Analysis")
    st.markdown("Detailed analysis of a single MLflow run using local artifacts.")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Sidebar: Run Selection
    # ─────────────────────────────────────────────────────────────────────────
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
            
        try:
            experiments = get_all_experiments()
            if not experiments:
                st.error("No experiments found.")
                return
        except Exception as e:
            st.error(f"Error: {e}")
            return
            
        exp_names = [exp["name"] for exp in experiments]
        selected_exp_name = st.selectbox("Select Experiment", options=exp_names)
        
        selected_exp = next(exp for exp in experiments if exp["name"] == selected_exp_name)
        runs_df = get_runs_for_experiment(selected_exp["experiment_id"])
        
        if runs_df.empty:
            st.warning("No runs found.")
            return
            
        # Create a display name for runs
        runs_df["display_name"] = runs_df["start_time"].dt.strftime("%Y-%m-%d %H:%M") + " - " + runs_df["run_id"].str[:8]
        
        # 1. Try tags.mlflow.runName (user-specified run name)
        if "tags.mlflow.runName" in runs_df.columns:
             # Use where to replace only non-null values
             runs_df["display_name"] = np.where(
                 runs_df["tags.mlflow.runName"].notnull(), 
                 runs_df["tags.mlflow.runName"], 
                 runs_df["display_name"]
             )
        # 2. Try run name column
        elif "name" in runs_df.columns:
             runs_df["display_name"] = np.where(
                 runs_df["name"].notnull(), 
                 runs_df["name"], 
                 runs_df["display_name"]
             )
        
        # Ensure name isn't empty string
        runs_df["display_name"] = runs_df.apply(
            lambda x: x["display_name"] if str(x["display_name"]).strip() != "" 
            else x["start_time"].strftime("%Y-%m-%d %H:%M") + " - " + str(x["run_id"])[:8],
            axis=1
        )
        
        # Set default index if target run exists
        default_index = 0
        target_default_run = "run_20260103_131046_4672993e_catboost_y_long_tp4sl2_24h_binary"
        
        display_names = runs_df["display_name"].tolist()
        if target_default_run in display_names:
            default_index = display_names.index(target_default_run)
        
        selected_run_display = st.selectbox(
            "Select Run", 
            options=display_names,
            index=default_index
        )
        
        selected_run_row = runs_df[runs_df["display_name"] == selected_run_display].iloc[0]
        run_id = selected_run_row["run_id"]
        run_dir = selected_run_row.get("params.run_dir")
        
        st.divider()
        st.caption(f"Run ID: {run_id}")
        if run_dir:
            st.caption(f"Path: {run_dir}")
        else:
            st.error("No run_dir parameter found for this run.")
            return

    # ─────────────────────────────────────────────────────────────────────────
    # Load Data
    # ─────────────────────────────────────────────────────────────────────────
    
    if not Path(run_dir).exists():
        st.error(f"Run directory not found locally: {run_dir}")
        return
        
    artifacts = load_run_artifacts(run_dir)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tabs
    # ─────────────────────────────────────────────────────────────────────────
    
    tab_stats, tab_ts, tab_rt, tab_dist, tab_comp = st.tabs(["📊 General Stats", "📈 Time Series Analysis", "🔴 Real-Time Analysis", "📉 Score Distributions", "⚖️ Dist. Comparison"])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1: General Stats
    # ─────────────────────────────────────────────────────────────────────────
    
    with tab_stats:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Key Metrics")
            metrics = artifacts.get("metrics", {})
            if metrics:
                metric_items = []
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        metric_items.append({"Metric": k, "Value": f"{v:.4f}"})
                st.table(pd.DataFrame(metric_items))
            else:
                st.warning("No metrics.json found.")
                
        with col2:
            st.subheader("Feature Importance")
            fi_df = artifacts.get("feature_importance")
            if fi_df is not None:
                # Normalize column names
                val_col = None
                if "importance_gain" in fi_df.columns:
                    val_col = "importance_gain"
                elif "importance" in fi_df.columns:
                    val_col = "importance"
                    
                if val_col:
                    top_fi = fi_df.sort_values(val_col, ascending=False).head(20)
                    fig_fi = px.bar(
                        top_fi,
                        x=val_col,
                        y="feature",
                        orientation="h",
                        title=f"Top 20 Features ({val_col})"
                    )
                    fig_fi.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.info("Feature importance data format not recognized (expected 'importance_gain' or 'importance').")
            else:
                st.info("No feature_importance.csv found.")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 2: Time Series Analysis
    # ─────────────────────────────────────────────────────────────────────────
    
    with tab_ts:
        st.subheader("Time Series Analysis")
        
        # ─────────────────────────────────────────────────────────────────────
        # Comparison Model
        # ─────────────────────────────────────────────────────────────────────
        
        st.markdown(f"**Model 1 (Primary)**: `{selected_run_display}`")
        compare_benchmark = st.checkbox(f"Compare with Benchmark '{BENCHMARK_RUN_NAME}'", value=True)
        
        compare_run_display = "None"
        compare_run_dir = None
        
        if compare_benchmark:
            # Check if primary model is the benchmark itself
            if selected_run_display == BENCHMARK_RUN_NAME:
                st.info("Primary model is the benchmark. Comparison disabled.")
            else:
                compare_run_display = BENCHMARK_RUN_NAME
                compare_run_dir = BENCHMARK_RUN_DIR
        
        # Load primary model data
        dfs = []
        for p in ["pred_train", "pred_val", "pred_test"]:
            if p in artifacts:
                d = artifacts[p].copy()
                d["split"] = p.replace("pred_", "")
                dfs.append(d)
        
        if not dfs:
            st.warning("No prediction files found (pred_train.csv, pred_test.csv).")
        else:
            combined_df = pd.concat(dfs, ignore_index=True)
            if "timestamp" in combined_df.columns:
                combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
                if combined_df["timestamp"].dt.tz is not None:
                    combined_df["timestamp"] = combined_df["timestamp"].dt.tz_localize(None)
            else:
                st.error("Timestamp column missing in predictions.")
                return
            
            # Load comparison model data if selected
            compare_df = None
            if compare_run_dir and compare_run_display != "None":
                if Path(compare_run_dir).exists():
                    compare_artifacts = load_run_artifacts(compare_run_dir)
                    compare_dfs = []
                    for p in ["pred_train", "pred_val", "pred_test"]:
                        if p in compare_artifacts:
                            d = compare_artifacts[p].copy()
                            d["split"] = p.replace("pred_", "")
                            compare_dfs.append(d)
                    if compare_dfs:
                        compare_df = pd.concat(compare_dfs, ignore_index=True)
                        if "timestamp" in compare_df.columns:
                            compare_df["timestamp"] = pd.to_datetime(compare_df["timestamp"])
                            if compare_df["timestamp"].dt.tz is not None:
                                compare_df["timestamp"] = compare_df["timestamp"].dt.tz_localize(None)

            # Controls
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            with col_ctrl1:
                start_date = st.date_input(
                    "Start Date", 
                    value=pd.Timestamp("2025-05-01")
                )
            with col_ctrl2:
                min_days = st.slider("Min Days / Month (Monthly Chart)", 1, 31, 20)
            with col_ctrl3:
                rolling_window = st.slider("Rolling Window (Days)", 7, 365, 365)

            # Filter Data for Monthly (needs only display range)
            filtered_df_monthly = combined_df[combined_df["timestamp"] >= pd.to_datetime(start_date)].copy()
            
            if filtered_df_monthly.empty:
                st.warning(f"No data found after {start_date}.")
            else:
                st.info(f"Model 1: {len(filtered_df_monthly)} samples from {start_date} to {filtered_df_monthly['timestamp'].max()}")
                
                # 1. Monthly AUC
                st.markdown("### 📅 Monthly AUC")
                monthly_auc = calculate_monthly_auc(filtered_df_monthly, min_days)
                monthly_auc["Model"] = selected_run_display[:30] + "..." if len(selected_run_display) > 30 else selected_run_display
                
                # Add comparison model monthly AUC
                if compare_df is not None:
                    compare_filtered = compare_df[compare_df["timestamp"] >= pd.to_datetime(start_date)].copy()
                    if not compare_filtered.empty:
                        compare_monthly = calculate_monthly_auc(compare_filtered, min_days)
                        if not compare_monthly.empty:
                            compare_monthly["Model"] = compare_run_display[:30] + "..." if len(compare_run_display) > 30 else compare_run_display
                            monthly_auc = pd.concat([monthly_auc, compare_monthly], ignore_index=True)
                
                if not monthly_auc.empty:
                    fig_monthly = px.line(
                        monthly_auc, 
                        x="Month", 
                        y="AUC", 
                        color="Model",
                        markers=True,
                        title=f"Monthly AUC (Min Days: {min_days})"
                    )
                    fig_monthly.update_layout(yaxis_range=[0.45, max(0.6, monthly_auc["AUC"].max() + 0.05)])
                    st.plotly_chart(fig_monthly, use_container_width=True, key="chart_monthly")
                else:
                    st.warning("Not enough data for monthly AUC.")
                    
                st.divider()
                
                # 2. Rolling AUC
                st.markdown(f"### 🔄 Rolling {rolling_window}-Day AUC")
                
                with st.spinner("Calculating rolling AUC..."):
                    # Include lookback period for calculation
                    lookback_date = pd.to_datetime(start_date) - pd.Timedelta(days=rolling_window)
                    # Filter minimal columns for speed
                    calc_df = combined_df[
                        (combined_df["timestamp"] >= lookback_date)
                    ][["timestamp", "y_true", "y_pred"]].copy()
                    
                    rolling_auc_df = calculate_rolling_auc(calc_df, window_days=rolling_window)
                    
                    # Filter results for display range
                    if not rolling_auc_df.empty:
                        rolling_auc_df = rolling_auc_df[rolling_auc_df["Date"] >= pd.to_datetime(start_date)]
                        rolling_auc_df["Model"] = selected_run_display[:30] + "..." if len(selected_run_display) > 30 else selected_run_display
                    
                    # Add comparison model rolling AUC
                    if compare_df is not None:
                        compare_calc_df = compare_df[
                            (compare_df["timestamp"] >= lookback_date)
                        ][["timestamp", "y_true", "y_pred"]].copy()
                        compare_rolling = calculate_rolling_auc(compare_calc_df, window_days=rolling_window)
                        if not compare_rolling.empty:
                            compare_rolling = compare_rolling[compare_rolling["Date"] >= pd.to_datetime(start_date)]
                            compare_rolling["Model"] = compare_run_display[:30] + "..." if len(compare_run_display) > 30 else compare_run_display
                            rolling_auc_df = pd.concat([rolling_auc_df, compare_rolling], ignore_index=True)
                
                if not rolling_auc_df.empty:
                    fig_roll = px.line(
                        rolling_auc_df,
                        x="Date",
                        y="AUC",
                        color="Model",
                        title=f"Rolling AUC ({rolling_window}-Day Window)",
                        labels={"Date": "Date", "AUC": "AUC"}
                    )
                    # Add reference line for 0.5
                    fig_roll.add_hline(y=0.5, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_roll, use_container_width=True, key="chart_rolling")
                else:
                    st.warning("Not enough data to calculate rolling AUC.")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3: Real-Time Analysis
    # ─────────────────────────────────────────────────────────────────────────
    
    with tab_rt:
        st.subheader("Real-Time Analysis")
        st.markdown(f"**Target**: `{TARGET_KEY}`")
        
        # ─────────────────────────────────────────────────────────────────────
        # Model Comparison Selector (from DB)
        # ─────────────────────────────────────────────────────────────────────
        
        # Get available models from classifier prediction DB
        rt_available_models = get_available_models_from_db()
        
        if not rt_available_models:
            st.warning("No models found in the classifier prediction DB.")
            return
        
        rt_col1, rt_col2 = st.columns(2)
        with rt_col1:
            rt_model1 = st.selectbox(
                "Model 1",
                options=rt_available_models,
                index=0,
                key="rt_model1"
            )
        with rt_col2:
            st.markdown("### ") # Spacing
            compare_benchmark_rt = st.checkbox("Compare with Benchmark", value=True, key="rt_compare_bench")
            if compare_benchmark_rt:
                rt_model2 = BENCHMARK_RUN_NAME
                st.caption(f"Benchmark: {rt_model2}")
                if rt_model1 == rt_model2:
                    st.info("Primary model is the benchmark.")
                    rt_model2 = "None"
            else:
                rt_model2 = "None"
        
        # Controls
        col_rt1, col_rt2, col_rt3 = st.columns(3)
        with col_rt1:
            rt_start_date = st.date_input(
                "Start Date (RT)", 
                value=pd.Timestamp("2025-05-01"),
                key="rt_start_date"
            )
        with col_rt2:
            rt_min_days = st.slider("Min Days / Month (RT)", 1, 31, 20, key="rt_min_days")
        with col_rt3:
            rt_rolling_window = st.slider("Rolling Window Days (RT)", 7, 365, 365, key="rt_rolling")
        
        # Load Data for Model 1
        with st.spinner(f"Loading data for {rt_model1[:30]}..."):
            rt_df1 = load_realtime_data(rt_model1, pd.to_datetime(rt_start_date))
        
        # Load Data for Model 2 if selected
        rt_df2 = pd.DataFrame()
        if rt_model2 != "None":
            with st.spinner(f"Loading data for {rt_model2[:30]}..."):
                rt_df2 = load_realtime_data(rt_model2, pd.to_datetime(rt_start_date))
        
        if rt_df1.empty:
            st.warning(f"No data found for '{rt_model1}'.")
        else:
            st.success(f"Model 1: {len(rt_df1)} samples ({rt_df1['timestamp'].min()} to {rt_df1['timestamp'].max()})")
            if not rt_df2.empty:
                st.success(f"Model 2: {len(rt_df2)} samples ({rt_df2['timestamp'].min()} to {rt_df2['timestamp'].max()})")
            
            # Overall AUC Summary
            st.markdown("### 📊 Overall AUC Summary (After Start Date)")
            
            # Calculate overall AUC for each model
            try:
                auc1 = roc_auc_score(rt_df1["y_true"], rt_df1["y_pred"])
                auc1_str = f"{auc1:.4f}"
            except Exception:
                auc1_str = "N/A"
            
            auc2_str = "N/A"
            if not rt_df2.empty:
                try:
                    auc2 = roc_auc_score(rt_df2["y_true"], rt_df2["y_pred"])
                    auc2_str = f"{auc2:.4f}"
                except Exception:
                    pass
            
            # Display as metrics
            auc_col1, auc_col2 = st.columns(2)
            with auc_col1:
                st.metric(
                    label=f"Model 1: {rt_model1[:35]}...",
                    value=auc1_str
                )
            with auc_col2:
                if rt_model2 != "None":
                    st.metric(
                        label=f"Model 2: {rt_model2[:35]}...",
                        value=auc2_str
                    )
            
            st.divider()
            
            # 1. Monthly Performance
            st.markdown("### 📅 Monthly Performance Metrics")
            rt_monthly_auc1 = calculate_monthly_auc(rt_df1, rt_min_days)
            
            if not rt_monthly_auc1.empty:
                # Rename for Model 1
                rt_monthly_auc1 = rt_monthly_auc1.rename(columns={
                    "AUC": "AUC_M1", "prAUC_1": "prAUC1_M1", "prAUC_0": "prAUC0_M1",
                    "Y_True_Rate": "Y_Rate", "Days": "Days", "Samples": "Samples"
                })
            
            # Add Model 2 metrics if selected
            full_monthly_df = rt_monthly_auc1.copy()
            if not rt_df2.empty and not rt_monthly_auc1.empty:
                rt_monthly_auc2 = calculate_monthly_auc(rt_df2, rt_min_days)
                if not rt_monthly_auc2.empty:
                    # Rename for Model 2 (only metrics needed)
                    rt_monthly_auc2 = rt_monthly_auc2.rename(columns={
                        "AUC": "AUC_M2", "prAUC_1": "prAUC1_M2", "prAUC_0": "prAUC0_M2"
                    })
                    # Merge - assuming same months due to same min_date likely, but outer join to be safe
                    # Note: Y_Rate, Days, Samples should be identical for same target/period
                    full_monthly_df = pd.merge(full_monthly_df, rt_monthly_auc2[["Month", "AUC_M2", "prAUC1_M2", "prAUC0_M2"]], on="Month", how="outer")
            
            if not full_monthly_df.empty:
                full_monthly_df = full_monthly_df.sort_values("Month", ascending=False)
                
                # Table 1: Metrics (AUC, prAUC)
                st.markdown("**1. Model Performance**")
                metrics_cols = ["Month", "AUC_M1"]
                if "AUC_M2" in full_monthly_df.columns:
                    metrics_cols.append("AUC_M2")
                metrics_cols.append("prAUC1_M1")
                if "prAUC1_M2" in full_monthly_df.columns:
                    metrics_cols.append("prAUC1_M2")
                metrics_cols.append("prAUC0_M1")
                if "prAUC0_M2" in full_monthly_df.columns:
                    metrics_cols.append("prAUC0_M2")
                    
                metrics_df = full_monthly_df[[c for c in metrics_cols if c in full_monthly_df.columns]]
                
                # Format Metrics Table
                st.dataframe(metrics_df.style.format("{:.4f}", subset=metrics_df.columns.drop("Month")), use_container_width=True)
                
                # Table 2: Stats (Rate, Days, Samples) - Only need once since it's same target
                st.markdown("**2. Common Statistics**")
                stats_cols = ["Month", "Y_Rate", "Days", "Samples"]
                stats_df = full_monthly_df[[c for c in stats_cols if c in full_monthly_df.columns]]
                
                # Format Stats Table
                format_dict = {"Y_Rate": "{:.2%}"}
                st.dataframe(stats_df.style.format(format_dict), use_container_width=True)
                
            else:
                st.warning("Not enough data for monthly metrics in real-time data.")
            
            st.divider()
            
            # 2. Rolling AUC
            st.markdown(f"### 🔄 Rolling {rt_rolling_window}-Day AUC (Real-Time)")
            
            with st.spinner("Calculating rolling AUC..."):
                # Enforce minimum date for rolling calc data
                lookback_start = pd.to_datetime(rt_start_date) - pd.Timedelta(days=rt_rolling_window)
                if lookback_start < REALTIME_MIN_DATE:
                    lookback_start = REALTIME_MIN_DATE
                
                rt_df1_full = load_realtime_data(rt_model1, lookback_start)
                
                rt_rolling_df = pd.DataFrame()
                if not rt_df1_full.empty:
                    rt_rolling_df = calculate_rolling_auc(rt_df1_full, window_days=rt_rolling_window)
                    if not rt_rolling_df.empty:
                        rt_rolling_df = rt_rolling_df[rt_rolling_df["Date"] >= pd.to_datetime(rt_start_date)]
                        rt_rolling_df["Model"] = rt_model1[:40] + "..." if len(rt_model1) > 40 else rt_model1
                
                # Add Model 2 rolling AUC
                if not rt_df2.empty:
                    rt_df2_full = load_realtime_data(rt_model2, lookback_start)
                    if not rt_df2_full.empty:
                        rt_rolling_df2 = calculate_rolling_auc(rt_df2_full, window_days=rt_rolling_window)
                        if not rt_rolling_df2.empty:
                            rt_rolling_df2 = rt_rolling_df2[rt_rolling_df2["Date"] >= pd.to_datetime(rt_start_date)]
                            rt_rolling_df2["Model"] = rt_model2[:40] + "..." if len(rt_model2) > 40 else rt_model2
                            rt_rolling_df = pd.concat([rt_rolling_df, rt_rolling_df2], ignore_index=True)
            
            if not rt_rolling_df.empty:
                fig_rt_roll = px.line(
                    rt_rolling_df,
                    x="Date",
                    y="AUC",
                    color="Model",
                    title=f"Rolling AUC ({rt_rolling_window}-Day Window)",
                    labels={"Date": "Date", "AUC": "AUC"}
                )
                fig_rt_roll.add_hline(y=0.5, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_rt_roll, use_container_width=True, key="chart_rt_rolling")
            else:
                st.warning("Not enough data to calculate rolling AUC in real-time data.")

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 4: Score Distributions
    # ─────────────────────────────────────────────────────────────────────────
    
    with tab_dist:
        st.subheader("Score Distributions")
        st.markdown("Compare prediction score distributions across **Train**, **Test**, and **Post-Test** periods.")
        
        # Get available models from classifier prediction DB
        dist_available_models = get_available_models_from_db()
        
        if not dist_available_models:
            st.warning("No models found in the classifier prediction DB.")
        else:
            # Base directory for model runs
            MODEL_BASE_DIR = "/Volumes/Extreme SSD/trading_data/cex/models/binance_btcusdt_perp_1h_original"
            
            dist_model_selected = st.selectbox(
                "Select Model",
                options=dist_available_models,
                index=0,
                key="dist_model"
            )
            
            # Infer run directory from model name
            model_run_dir = f"{MODEL_BASE_DIR}/{dist_model_selected}"
            
            # Try to load local predictions (train/test)
            train_scores = pd.DataFrame()
            test_scores = pd.DataFrame()
            test_max_ts = None
            
            if Path(model_run_dir).exists():
                # Load train predictions
                train_path = Path(model_run_dir) / "pred_train.csv"
                if train_path.exists():
                    train_df = pd.read_csv(train_path)
                    if "y_pred" in train_df.columns:
                        train_scores = train_df[["y_pred"]].copy()
                        train_scores["Period"] = "Train"
                
                # Load test predictions
                test_path = Path(model_run_dir) / "pred_test.csv"
                if test_path.exists():
                    test_df = pd.read_csv(test_path)
                    if "y_pred" in test_df.columns:
                        test_scores = test_df[["y_pred"]].copy()
                        test_scores["Period"] = "Test"
                        # Get max timestamp from test to filter post-test
                        if "timestamp" in test_df.columns:
                            test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
                            test_max_ts = test_df["timestamp"].max()
            
            # Load post-test predictions from DB
            post_test_scores = pd.DataFrame()
            if test_max_ts is not None:
                post_test_df = load_predictions_only(dist_model_selected, start_date=test_max_ts + pd.Timedelta(hours=1))
                if not post_test_df.empty:
                    post_test_scores = post_test_df[["y_pred"]].copy()
                    post_test_scores["Period"] = "Post-Test"
            else:
                # If no test timestamp, try loading all DB predictions
                post_test_df = load_predictions_only(dist_model_selected)
                if not post_test_df.empty:
                    post_test_scores = post_test_df[["y_pred"]].copy()
                    post_test_scores["Period"] = "Post-Test (All DB)"
            
            # Combine all scores
            all_scores = pd.concat([train_scores, test_scores, post_test_scores], ignore_index=True)
            
            if all_scores.empty:
                st.warning(f"No prediction data found for model '{dist_model_selected}'.")
            else:
                # Display counts
                st.markdown("#### Sample Counts")
                counts = all_scores.groupby("Period").size().reset_index(name="Count")
                st.dataframe(counts, use_container_width=True)
                
                # Density plot
                st.markdown("#### Score Distribution (Density)")
                fig_hist = px.histogram(
                    all_scores,
                    x="y_pred",
                    color="Period",
                    nbins=50,
                    barmode="overlay",
                    opacity=0.5,
                    histnorm="probability density",
                    title=f"Score Distribution: {dist_model_selected[:50]}..."
                )
                fig_hist.update_layout(xaxis_title="Prediction Score (y_pred)", yaxis_title="Density")
                st.plotly_chart(fig_hist, use_container_width=True, key="chart_dist_hist")
                
                # Box plot
                st.markdown("#### Score Distribution Box Plot")
                fig_box = px.box(
                    all_scores,
                    x="Period",
                    y="y_pred",
                    color="Period",
                    title="Score Distribution by Period"
                )
                fig_box.update_layout(yaxis_title="Prediction Score (y_pred)")
                st.plotly_chart(fig_box, use_container_width=True, key="chart_dist_box")
                
                # Stats table
                st.markdown("#### Summary Statistics")
                stats_df = all_scores.groupby("Period")["y_pred"].agg(["mean", "std", "min", "max", "median"]).reset_index()
                stats_df.columns = ["Period", "Mean", "Std", "Min", "Max", "Median"]
                st.dataframe(stats_df.style.format({"Mean": "{:.4f}", "Std": "{:.4f}", "Min": "{:.4f}", "Max": "{:.4f}", "Median": "{:.4f}"}), use_container_width=True)

                st.dataframe(stats_df.style.format({"Mean": "{:.4f}", "Std": "{:.4f}", "Min": "{:.4f}", "Max": "{:.4f}", "Median": "{:.4f}"}), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 5: Distribution Comparison
    # ─────────────────────────────────────────────────────────────────────────

    with tab_comp:
        st.subheader("Score Distribution Comparison")
        st.markdown(f"**Benchmark Model**: `{BENCHMARK_RUN_NAME}`")
        
        # Candidate Selector
        comp_available_models = get_available_models_from_db()
        # Filter benchmark out of options if present to avoid self-comparison default
        comp_options = [m for m in comp_available_models if m != BENCHMARK_RUN_NAME]
        
        comp_candidate = st.selectbox(
            "Select Candidate Model",
            options=comp_options,
            index=0 if comp_options else None,
            key="comp_candidate"
        )
        
        if not comp_candidate:
            st.warning("No candidate models available.")
        else:
            with st.spinner("Loading distributions and targets for calibration..."):
                # Load Benchmark Data (from DB)
                bench_df = load_predictions_only(BENCHMARK_RUN_NAME)
                # Load Candidate Data (from DB)
                cand_df = load_predictions_only(comp_candidate)
                
                # Load Targets (Shared)
                targets_df = pd.DataFrame()
                try:
                    conn = duckdb.connect(TARGETS_DB_PATH, read_only=True)
                    # Fetch timestamps and targets (long format table)
                    query = f"SELECT timestamp, target_value as y_true FROM targets WHERE target_key = '{TARGET_KEY}'"
                    targets_df = conn.execute(query).df()
                    conn.close()
                    targets_df["timestamp"] = pd.to_datetime(targets_df["timestamp"])
                    if targets_df["timestamp"].dt.tz is not None:
                        targets_df["timestamp"] = targets_df["timestamp"].dt.tz_localize(None)
                except Exception as e:
                    st.error(f"Error loading targets: {e}")

                if bench_df.empty:
                    st.error(f"Benchmark data not found for {BENCHMARK_RUN_NAME}")
                elif cand_df.empty:
                    st.warning(f"No data found for candidate {comp_candidate}")
                elif targets_df.empty:
                    st.error("Targets data could not be loaded.")
                else:
                    # 1. Align Data (Inner Join on Timestamp) for Paired Metrics
                    # Join Benchmark + Candidate
                    merged_preds = pd.merge(
                        bench_df[["timestamp", "y_pred"]].rename(columns={"y_pred": "Benchmark_Raw"}),
                        cand_df[["timestamp", "y_pred"]].rename(columns={"y_pred": "Candidate_Raw"}),
                        on="timestamp",
                        how="inner"
                    )
                    
                    # Join with Targets
                    merged_df = pd.merge(merged_preds, targets_df, on="timestamp", how="inner")
                    
                    if merged_df.empty:
                        st.warning("No overlap between predictions and targets.")
                    else:
                        st.success(f"Common samples: {len(merged_df)} (from {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()})")
                        
                        # 2. Platt Calibration
                        # Define Calibration Period
                        calib_start = pd.Timestamp("2025-04-01")
                        calib_end = pd.Timestamp("2025-12-31")
                        
                        calib_mask = (merged_df["timestamp"] >= calib_start) & (merged_df["timestamp"] <= calib_end)
                        calib_data = merged_df[calib_mask]
                        
                        if calib_data.empty or len(calib_data) < 100:
                            st.warning(f"Not enough data in calibration period ({calib_start} to {calib_end}). Showing raw scores.")
                            merged_df["Benchmark"] = merged_df["Benchmark_Raw"]
                            merged_df["Candidate"] = merged_df["Candidate_Raw"]
                        else:
                            st.info(f"Calibrating on {len(calib_data)} samples ({calib_start.date()} to {calib_end.date()}) using Platt Scaling (Logistic Regression)")
                            
                            # Fit Platt Scaling (Logistic Regression)
                            lr_bench = LogisticRegression(solver='lbfgs', C=1e5) 
                            lr_bench.fit(calib_data[["Benchmark_Raw"]], calib_data["y_true"])
                            
                            lr_cand = LogisticRegression(solver='lbfgs', C=1e5)
                            lr_cand.fit(calib_data[["Candidate_Raw"]], calib_data["y_true"])
                            
                            # Apply Calibration to ALL data
                            merged_df["Benchmark"] = lr_bench.predict_proba(merged_df[["Benchmark_Raw"]])[:, 1]
                            merged_df["Candidate"] = lr_cand.predict_proba(merged_df[["Candidate_Raw"]])[:, 1]
                            
                            # Calibration Diagnostics
                            st.write("### 🔧 Calibration Diagnostics")
                            diag_cols = st.columns(4)
                            diag_cols[0].metric("Bench Raw Min/Max", f"{merged_df['Benchmark_Raw'].min():.4f} / {merged_df['Benchmark_Raw'].max():.4f}")
                            diag_cols[1].metric("Bench Calib Min/Max", f"{merged_df['Benchmark'].min():.4f} / {merged_df['Benchmark'].max():.4f}")
                            diag_cols[2].metric("Cand Raw Min/Max", f"{merged_df['Candidate_Raw'].min():.4f} / {merged_df['Candidate_Raw'].max():.4f}")
                            diag_cols[3].metric("Cand Calib Min/Max", f"{merged_df['Candidate'].min():.4f} / {merged_df['Candidate'].max():.4f}")
                        
                        # 3. Calculate Uniqueness/Similarity Metrics on CALIBRATED scores
                        # Helper for stats
                        def get_dist_stats(series):
                            return {
                                "Mean": series.mean(),
                                "Median": series.median(),
                                "Std": series.std(),
                                "Skew": skew(series),
                                "Kurtosis": kurtosis(series),
                                "Min": series.min(),
                                "Max": series.max(),
                                "Q1": series.quantile(0.25),
                                "Q3": series.quantile(0.75)
                            }
                            
                        bench_stats = get_dist_stats(merged_df["Benchmark"])
                        cand_stats = get_dist_stats(merged_df["Candidate"])
                        
                        # Paired Metrics
                        pearson_corr, _ = pearsonr(merged_df["Benchmark"], merged_df["Candidate"])
                        spearman_corr, _ = spearmanr(merged_df["Benchmark"], merged_df["Candidate"])
                        w_dist = wasserstein_distance(merged_df["Benchmark"], merged_df["Candidate"])
                        
                        # Build Table
                        comp_data = []
                        metrics_list = ["Mean", "Median", "Std", "Skew", "Kurtosis", "Min", "Max", "Q1", "Q3"]
                        
                        for m in metrics_list:
                            comp_data.append({
                                "Metric": m,
                                "Benchmark": bench_stats[m],
                                "Candidate": cand_stats[m],
                                "Delta": cand_stats[m] - bench_stats[m]
                            })
                            
                        # Add Similarity Metrics Section
                        comp_data.append({"Metric": "── Similarity ──", "Benchmark": None, "Candidate": None, "Delta": None})
                        comp_data.append({"Metric": "Pearson Corr (Linear)", "Benchmark": 1.0, "Candidate": pearson_corr, "Delta": pearson_corr - 1.0})
                        comp_data.append({"Metric": "Spearman Corr (Rank)", "Benchmark": 1.0, "Candidate": spearman_corr, "Delta": spearman_corr - 1.0})
                        comp_data.append({"Metric": "Wasserstein Dist.", "Benchmark": 0.0, "Candidate": w_dist, "Delta": w_dist})
                        
                        comp_table_df = pd.DataFrame(comp_data)
                        
                        # Display Table
                        st.markdown("### 📊 Calibrated Distribution Statistics & Uniqueness")
                        st.dataframe(
                            comp_table_df.style.format({
                                "Benchmark": "{:.4f}", 
                                "Candidate": "{:.4f}", 
                                "Delta": "{:.4f}"
                            }, na_rep=""),
                            use_container_width=True
                        )
                        
                        # Visual Comparison (Density)
                        st.markdown("### 📈 Calibrated Density Comparison")
                        plot_df = pd.melt(merged_df, id_vars=["timestamp"], value_vars=["Benchmark", "Candidate"], 
                                         var_name="Model", value_name="Score")
                        
                        fig_dens = px.histogram(
                            plot_df,
                            x="Score",
                            color="Model",
                            nbins=100,
                            barmode="overlay",
                            opacity=0.5,
                            histnorm="probability density",
                            title="Calibrated Score Density Overlay"
                        )
                        fig_dens.update_layout(xaxis_title="Calibrated Probability", yaxis_title="Density")
                        st.plotly_chart(fig_dens, use_container_width=True)

                        # Quantile Shift Analysis (20 Equal Bins)
                        st.markdown("### 📊 Quantile Boundary Shift (20 Equal Bins)")
                        st.caption("Difference in score boundaries (Candidate - Benchmark) for each 5% quantile step.")
                        
                        # Calculate 20 quantiles (5% steps from 5% to 100%)
                        quantiles = np.linspace(0.05, 1.0, 20)
                        bench_qs = merged_df["Benchmark"].quantile(quantiles)
                        cand_qs = merged_df["Candidate"].quantile(quantiles)
                        
                        shift_data = []
                        for q in quantiles:
                            b_val = bench_qs[q]
                            c_val = cand_qs[q]
                            shift_data.append({
                                "Quantile": f"{int(q*100)}%",
                                "Quantile_Val": q,
                                "Benchmark Boundary": b_val,
                                "Candidate Boundary": c_val,
                                "Difference": c_val - b_val,
                                "Color": "Positive" if (c_val - b_val) >= 0 else "Negative"
                            })
                        
                        shift_df = pd.DataFrame(shift_data)
                        
                        # Display as Table
                        st.markdown("### 📋 Quantile Boundaries Table")
                        st.dataframe(
                            shift_df[["Quantile", "Benchmark Boundary", "Candidate Boundary", "Difference"]].style.format({
                                "Benchmark Boundary": "{:.4f}",
                                "Candidate Boundary": "{:.4f}",
                                "Difference": "{:.4f}"
                            }).background_gradient(subset=["Difference"], cmap="RdYlGn", vmin=-0.1, vmax=0.1),
                            use_container_width=True
                        )

                        # Target Rate by Quantile (Calibrated Scores)
                        st.markdown("### 🎯 Calibration Quality: Predicted vs Actual (Equal-Frequency Bins)")
                        st.caption("20 Equal-Frequency Bins (Quantiles) of Calibrated Scores. Checks if the average calibrated probability in each bin matches the actual positive rate.")

                        # Bin Calibrated Scores by Quantile (Equal Frequency)
                        # We use duplicates='drop' just in case scores are extremely degenerate
                        merged_df["Bin_Bench"] = pd.qcut(merged_df["Benchmark"], q=20, labels=False, duplicates='drop')
                        merged_df["Bin_Cand"] = pd.qcut(merged_df["Candidate"], q=20, labels=False, duplicates='drop')
                        
                        # Helper to aggregate
                        def get_reliability_stats(df, group_col, prob_col, model_name):
                            stats = df.groupby(group_col) \
                                .agg(
                                    Target_Rate=("y_true", "mean"),
                                    Avg_Prob=(prob_col, "mean"),
                                    Count=("y_true", "count")
                                ).reset_index()
                            stats.columns = ["Bin_Index", "Actual Rate", "Predicted Prob", "Count"]
                            stats["Model"] = model_name
                            stats["Quantile"] = stats["Bin_Index"].apply(lambda x: f"{(x)*5}-{(x+1)*5}%")
                            return stats

                        bench_rel = get_reliability_stats(merged_df, "Bin_Bench", "Benchmark", "Benchmark")
                        cand_rel = get_reliability_stats(merged_df, "Bin_Cand", "Candidate", "Candidate")
                        
                        rel_df = pd.concat([bench_rel, cand_rel], ignore_index=True)
                        
                        # Melt for Grouped Bar Chart (comparing Actual vs Predicted side-by-side or difference)
                        # A better vis: Scatter plot with x=Predicted, y=Actual (Reliability Diagram)
                        # But user liked the bar chart of rates. Let's do a dual-line or overaly.
                        # Actually, keeping it simple: Plot Actual Rate bars, but put avg predicted prob in tooltip
                        
                        fig_rel = px.bar(
                            rel_df,
                            x="Quantile",
                            y="Actual Rate",
                            color="Model",
                            barmode="group",
                            title="Actual Target Rate by Calibrated Quantile",
                            color_discrete_map={"Benchmark": "#636EFA", "Candidate": "#EF553B"},
                            hover_data=["Predicted Prob", "Count"]
                        )
                        
                        # Add line for Predicted Prob? It might be messy for grouped.
                        # Instead, let's create a Difference metric: (Actual - Predicted)
                        # But simpler is just showing the Rates as requested, ensuring bins are correct.
                        
                        fig_rel.update_layout(
                            yaxis_title="Actual Positive Rate",
                            xaxis_title="Calibrated Score Percentile"
                        )
                        st.plotly_chart(fig_rel, use_container_width=True)
                        
                        # Reliability Diagram (Predicted vs Actual)
                        st.markdown("### 📉 Reliability Diagram")
                        # fig_diag = transform_to_reliability_diagram(rel_df) # Defining inline below
                        fig_diag = px.line(
                            rel_df,
                            x="Predicted Prob",
                            y="Actual Rate",
                            color="Model",
                            markers=True,
                            title="Reliability Diagram (Perfect Calibration = Diagonal)",
                            color_discrete_map={"Benchmark": "#636EFA", "Candidate": "#EF553B"}
                        )
                        # Add diagonal line
                        fig_diag.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="Gray", dash="dash"))
                        fig_diag.update_layout(xaxis_title="Average Predicted Probability", yaxis_title="Actual Positive Rate", xaxis_range=[0,0.2], yaxis_range=[0,0.2], width=600, height=600)
                        st.plotly_chart(fig_diag, use_container_width=True)




if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Experiment Diagnosis App

Streamlit app to analyze MLflow experiments:
- Select experiment and view all runs
- Display metrics summary
- Aggregate feature importance across runs

Usage:
    streamlit run analysis/experiment_diagnosis.py
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

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

TRACKING_URI = "http://127.0.0.1:5000"

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
    
    # Load run_metadata.json
    meta_path = run_path / "run_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            artifacts["metadata"] = json.load(f)
    
    return artifacts


def aggregate_feature_importance(run_dirs: List[str]) -> Optional[pd.DataFrame]:
    """Aggregate feature importance across multiple runs."""
    all_fi = []
    
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        fi_path = run_path / "feature_importance.csv"
        if fi_path.exists():
            try:
                fi_df = pd.read_csv(fi_path)
                fi_df["run_dir"] = run_dir
                all_fi.append(fi_df)
            except Exception:
                pass
    
    if not all_fi:
        return None
    
    combined = pd.concat(all_fi, ignore_index=True)
    agg = combined.groupby("feature")["importance"].agg(
        ["mean", "std", "min", "max", "count"]
    ).reset_index()
    agg = agg.sort_values("mean", ascending=False)
    agg.columns = ["Feature", "Mean", "Std", "Min", "Max", "Runs"]
    
    return agg


def aggregate_metrics(run_dirs: List[str]) -> Optional[pd.DataFrame]:
    """Aggregate metrics across multiple runs."""
    all_metrics = []
    
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        metrics_path = run_path / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                metrics["run_dir"] = run_dir
                all_metrics.append(metrics)
            except Exception:
                pass
    
    if not all_metrics:
        return None
    
    return pd.DataFrame(all_metrics)


def aggregate_best_params(run_dirs: List[str]) -> Optional[pd.DataFrame]:
    """Aggregate best params across multiple runs."""
    all_params = []
    
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        params_path = run_path / "best_params.json"
        if params_path.exists():
            try:
                with open(params_path, "r") as f:
                    params = json.load(f)
                params["run_dir"] = run_dir
                all_params.append(params)
            except Exception:
                pass
    
    if not all_params:
        return None
    
    return pd.DataFrame(all_params)


def get_param_summary(params_df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics for each parameter."""
    param_cols = [c for c in params_df.columns if c != "run_dir"]
    summary_data = []
    
    for col in param_cols:
        values = params_df[col].dropna()
        if len(values) == 0:
            continue
        
        # Mode (most common value)
        mode = values.mode().iloc[0] if len(values.mode()) > 0 else values.iloc[0]
        mode_count = (values == mode).sum()
        mode_pct = mode_count / len(values) * 100
        
        summary_data.append({
            "Parameter": col,
            "Mode (Most Common)": mode,
            "Mode Count": f"{mode_count}/{len(values)}",
            "Mode %": f"{mode_pct:.0f}%",
            "Unique Values": values.nunique(),
        })
    
    return pd.DataFrame(summary_data)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────────────────────


def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 15):
    """Plot feature importance bar chart."""
    top_features = fi_df.head(top_n).copy()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_features["Mean"],
        y=top_features["Feature"],
        orientation="h",
        error_x=dict(type="data", array=top_features["Std"]),
        marker_color="steelblue",
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Features by Mean Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
        height=max(400, top_n * 25),
    )
    return fig


def plot_param_histogram(params_df: pd.DataFrame, param: str):
    """Plot histogram for a single parameter."""
    values = params_df[param].dropna()
    
    # Count occurrences
    value_counts = values.value_counts().sort_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(v) for v in value_counts.index],
        y=value_counts.values,
        marker_color="steelblue",
        text=value_counts.values,
        textposition="outside",
    ))
    
    fig.update_layout(
        title=f"{param} Distribution",
        xaxis_title="Value",
        yaxis_title="Count",
        height=250,
        margin=dict(t=40, b=40),
    )
    return fig


def plot_param_vs_auc(merged_df: pd.DataFrame, param: str):
    """Plot scatter of AUC train vs AUC test, colored by param value."""
    required_cols = [param, "auc_test"]
    # Check for train AUC - could be auc_train or auc_val
    train_col = None
    for col in ["auc_train", "auc_val"]:
        if col in merged_df.columns:
            train_col = col
            break
    
    if train_col is None or param not in merged_df.columns or "auc_test" not in merged_df.columns:
        return None
    
    df = merged_df[[param, train_col, "auc_test"]].dropna()
    if df.empty:
        return None
    
    # Use numeric param value for continuous color scale (heatmap style)
    try:
        color_values = pd.to_numeric(df[param])
    except (ValueError, TypeError):
        color_values = df[param].astype("category").cat.codes
    
    fig = px.scatter(
        df,
        x=train_col,
        y="auc_test",
        color=color_values,
        color_continuous_scale="Viridis",
        title=f"{train_col} vs AUC Test by {param}",
        labels={train_col: train_col.replace('_', ' ').title(), "auc_test": "AUC Test"},
    )
    
    # Add diagonal line (perfect generalization)
    min_val = min(df[train_col].min(), df["auc_test"].min())
    max_val = max(df[train_col].max(), df["auc_test"].max())
    fig.add_shape(
        type="line",
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color="gray", dash="dash"),
    )
    
    fig.update_layout(
        height=300,
        margin=dict(t=40, b=40),
        coloraxis_colorbar_title_text=param,
    )
    return fig


def plot_metrics_over_runs(metrics_df: pd.DataFrame, metric: str):
    """Plot metric values across runs."""
    if metric not in metrics_df.columns:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(metrics_df) + 1)),
        y=metrics_df[metric],
        mode="lines+markers",
        name=metric,
        marker=dict(size=8),
    ))
    
    mean_val = metrics_df[metric].mean()
    fig.add_hline(y=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_val:.4f}")
    
    fig.update_layout(
        title=f"{metric} Across Runs",
        xaxis_title="Run #",
        yaxis_title=metric,
        height=350,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title="Experiment Diagnosis",
        page_icon="🔬",
        layout="wide",
    )
    
    st.title("🔬 Experiment Diagnosis")
    st.markdown("Analyze MLflow experiments: metrics, feature importance, and stability.")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Sidebar: Experiment Selection
    # ─────────────────────────────────────────────────────────────────────────
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        try:
            experiments = get_all_experiments()
            if not experiments:
                st.error("No experiments found in MLflow.")
                return
        except Exception as e:
            st.error(f"Could not connect to MLflow: {e}")
            st.info(f"Make sure MLflow server is running at {TRACKING_URI}")
            return
        
        exp_names = [exp["name"] for exp in experiments]
        selected_exp_name = st.selectbox(
            "Select Experiment",
            options=exp_names,
            index=0,
        )
        
        selected_exp = next(
            (exp for exp in experiments if exp["name"] == selected_exp_name),
            None
        )
        
        if selected_exp is None:
            st.error("Experiment not found.")
            return
        
        st.divider()
        st.caption(f"Experiment ID: {selected_exp['experiment_id']}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Load Runs
    # ─────────────────────────────────────────────────────────────────────────
    
    runs_df = get_runs_for_experiment(selected_exp["experiment_id"])
    
    if runs_df.empty:
        st.warning("No runs found for this experiment.")
        return
    
    st.info(f"📊 Found **{len(runs_df)}** runs in experiment **{selected_exp_name}**")
    
    # Extract run_dir from params
    run_dirs = []
    for _, row in runs_df.iterrows():
        run_dir = row.get("params.run_dir")
        if run_dir and Path(run_dir).exists():
            run_dirs.append(run_dir)
    
    if not run_dirs:
        st.warning("No valid run directories found. Runs may not have logged 'run_dir' parameter.")
        st.write("Available parameters:", [c for c in runs_df.columns if c.startswith("params.")])
        return
    
    st.success(f"✅ Found **{len(run_dirs)}** runs with valid artifacts")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tabs
    # ─────────────────────────────────────────────────────────────────────────
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Metrics Summary",
        "🎯 Feature Importance",
        "⚙️ Best Params",
        "📉 Stability Analysis",
        "🔍 Run Details",
    ])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1: Metrics Summary
    # ─────────────────────────────────────────────────────────────────────────
    
    with tab1:
        st.header("📈 Metrics Summary")
        
        metrics_df = aggregate_metrics(run_dirs)
        
        if metrics_df is None or metrics_df.empty:
            st.warning("No metrics found in run directories.")
        else:
            # Identify numeric metric columns
            metric_cols = [c for c in metrics_df.columns 
                          if c != "run_dir" and pd.api.types.is_numeric_dtype(metrics_df[c])]
            
            if metric_cols:
                # Summary statistics
                st.subheader("Summary Statistics")
                summary = metrics_df[metric_cols].describe().T
                summary["cv"] = summary["std"] / summary["mean"]  # Coefficient of variation
                st.dataframe(summary.style.format("{:.4f}"), use_container_width=True)
                
                # Highlight key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                if "auc_test" in metric_cols:
                    with col1:
                        st.metric(
                            "AUC Test (Mean)",
                            f"{metrics_df['auc_test'].mean():.4f}",
                            f"±{metrics_df['auc_test'].std():.4f}"
                        )
                
                if "logloss_test" in metric_cols:
                    with col2:
                        st.metric(
                            "LogLoss Test (Mean)",
                            f"{metrics_df['logloss_test'].mean():.4f}",
                            f"±{metrics_df['logloss_test'].std():.4f}"
                        )
                
                if "auc_val" in metric_cols:
                    with col3:
                        st.metric(
                            "AUC Val (Mean)",
                            f"{metrics_df['auc_val'].mean():.4f}",
                            f"±{metrics_df['auc_val'].std():.4f}"
                        )
                
                if "rmse_test" in metric_cols:
                    with col4:
                        st.metric(
                            "RMSE Test (Mean)",
                            f"{metrics_df['rmse_test'].mean():.4f}",
                            f"±{metrics_df['rmse_test'].std():.4f}"
                        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tab 2: Feature Importance
    # ─────────────────────────────────────────────────────────────────────────
    
    with tab2:
        st.header("🎯 Feature Importance")
        
        fi_df = aggregate_feature_importance(run_dirs)
        
        if fi_df is None or fi_df.empty:
            st.warning("No feature importance data found.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                top_n = st.slider("Top N Features", min_value=5, max_value=len(fi_df), value=min(15, len(fi_df)))
                fig = plot_feature_importance(fi_df, top_n=top_n)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Full Ranking")
                st.dataframe(
                    fi_df.style.format({
                        "Mean": "{:.2f}",
                        "Std": "{:.2f}",
                        "Min": "{:.2f}",
                        "Max": "{:.2f}",
                    }),
                    use_container_width=True,
                    height=500,
                )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3: Best Params Analysis
    # ─────────────────────────────────────────────────────────────────────────
    
    with tab3:
        st.header("⚙️ Best Params Analysis")
        
        params_df = aggregate_best_params(run_dirs)
        
        if params_df is None or params_df.empty:
            st.warning("No best params data found in run directories.")
        else:
            # Filter out non-tunable params
            excluded_params = {"od_wait", "random_seed", "run_dir"}
            param_cols = [c for c in params_df.columns if c not in excluded_params]
            
            # Summary table
            st.subheader("Parameter Summary")
            summary_df = get_param_summary(params_df)
            st.dataframe(summary_df, use_container_width=True)
            
            # Merge params with metrics for scatter plots
            metrics_df = aggregate_metrics(run_dirs)
            if metrics_df is not None:
                # Join on run_dir
                merged_df = params_df.merge(metrics_df, on="run_dir", how="left")
            else:
                merged_df = params_df.copy()
            
            # Parameter analysis (one per row with histogram + scatter)
            st.subheader("Parameter Analysis")
            
            for param in param_cols:
                st.markdown(f"### `{param}`")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = plot_param_histogram(params_df, param)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = plot_param_vs_auc(merged_df, param)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No AUC data available")
            
            # Raw data expander
            with st.expander("View Raw Parameters Table"):
                st.dataframe(params_df.drop(columns=["run_dir"]), use_container_width=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tab 4: Stability Analysis
    # ─────────────────────────────────────────────────────────────────────────
    
    with tab4:
        st.header("📉 Stability Analysis")
        
        metrics_df = aggregate_metrics(run_dirs)
        
        if metrics_df is None or metrics_df.empty:
            st.warning("No metrics found for stability analysis.")
        else:
            metric_cols = [c for c in metrics_df.columns 
                          if c != "run_dir" and pd.api.types.is_numeric_dtype(metrics_df[c])]
            
            if metric_cols:
                selected_metric = st.selectbox("Select Metric", metric_cols)
                
                fig = plot_metrics_over_runs(metrics_df, selected_metric)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Stability verdict
                if selected_metric in metrics_df.columns:
                    values = metrics_df[selected_metric].dropna()
                    if len(values) > 1:
                        value_range = values.max() - values.min()
                        cv = values.std() / values.mean() if values.mean() != 0 else 0
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Range", f"{value_range:.4f}")
                        with col2:
                            st.metric("CV", f"{cv:.2%}")
                        with col3:
                            st.metric("Min", f"{values.min():.4f}")
                        with col4:
                            st.metric("Max", f"{values.max():.4f}")
                        
                        # Stability verdict
                        if selected_metric.startswith("auc"):
                            threshold = 0.02
                            if value_range < threshold:
                                st.success(f"✅ **STABLE**: Range ({value_range:.4f}) < {threshold}")
                            else:
                                st.error(f"⚠️ **UNSTABLE**: Range ({value_range:.4f}) >= {threshold}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tab 5: Run Details
    # ─────────────────────────────────────────────────────────────────────────
    
    with tab5:
        st.header("🔍 Run Details")
        
        run_idx = st.selectbox(
            "Select Run",
            options=range(len(run_dirs)),
            format_func=lambda i: f"Run {i+1}: {Path(run_dirs[i]).name}"
        )
        
        if run_idx is not None:
            artifacts = load_run_artifacts(run_dirs[run_idx])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Metrics")
                if "metrics" in artifacts:
                    st.json(artifacts["metrics"])
                else:
                    st.warning("No metrics found.")
            
            with col2:
                st.subheader("Best Parameters")
                if "best_params" in artifacts:
                    st.json(artifacts["best_params"])
                else:
                    st.warning("No parameters found.")
            
            st.subheader("Feature Importance")
            if "feature_importance" in artifacts:
                st.dataframe(
                    artifacts["feature_importance"].style.format({"importance": "{:.4f}"}),
                    use_container_width=True,
                )
            else:
                st.warning("No feature importance found.")
            
            st.subheader("Run Directory")
            st.code(run_dirs[run_idx])


if __name__ == "__main__":
    main()

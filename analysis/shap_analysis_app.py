#!/usr/bin/env python3
"""
SHAP Analysis Streamlit App

Visualizes pre-computed SHAP results for model comparison and feature contribution analysis.

Usage:
    streamlit run analysis/shap_analysis_app.py

Tabs:
1. Global Importance - Bar charts of mean |SHAP| per feature
2. Monthly Importance - Feature importance breakdown by month
3. Time Series - SHAP contribution over time for selected features
4. Feature Dependence - Scatter plots of feature value vs SHAP contribution
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from shap.shap_utils import (
    SHAP_RESULTS_DIR,
    get_available_results,
    load_shap_results,
    load_targets_from_duckdb,
    get_feature_columns,
    ShapMetadata,
)


st.set_page_config(
    page_title="SHAP Analysis",
    page_icon="📊",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_results_list() -> List[Dict[str, Any]]:
    """Get list of available SHAP result directories."""
    return get_available_results(SHAP_RESULTS_DIR)


@st.cache_data(ttl=300)
def load_result_data(result_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Load SHAP data from a result directory."""
    path = Path(result_path)
    shap_df, summary_df, metadata = load_shap_results(path)
    return shap_df, summary_df, metadata.__dict__ if hasattr(metadata, '__dict__') else metadata


@st.cache_data(ttl=300)
def get_targets(start_date: str, end_date: str) -> pd.DataFrame:
    """Load targets for the given date range."""
    try:
        return load_targets_from_duckdb(start_date, end_date)
    except Exception as e:
        st.warning(f"Failed to load targets: {e}")
        return pd.DataFrame()


def compute_monthly_importance(
    shap_df: pd.DataFrame, 
    feature_cols: List[str],
    segment_by_target: bool = False,
) -> pd.DataFrame:
    """
    Compute mean |SHAP| per feature per month.
    
    If segment_by_target is True, also computes separate stats for y_true=0 and y_true=1.
    """
    df = shap_df.copy()
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    
    records = []
    
    # Define segments
    if segment_by_target and "y_true" in df.columns:
        segments = [("all", df), ("y_true=0", df[df["y_true"] == 0]), ("y_true=1", df[df["y_true"] == 1])]
    else:
        segments = [("all", df)]
    
    for segment_name, segment_df in segments:
        for month in sorted(df["month"].unique()):
            month_df = segment_df[segment_df["month"] == month]
            if len(month_df) == 0:
                continue
            for feat in feature_cols:
                records.append({
                    "segment": segment_name,
                    "month": month,
                    "feature": feat,
                    "mean_abs_shap": np.abs(month_df[feat]).mean(),
                    "mean_shap": month_df[feat].mean(),
                    "std_shap": month_df[feat].std(),
                    "num_samples": len(month_df),
                })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("📊 SHAP Feature Analysis")
    
    # Sidebar: Result selection
    st.sidebar.header("Select SHAP Results")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Results", help="Reload available SHAP results"):
        st.cache_data.clear()
        st.rerun()
    
    results = get_results_list()
    
    if not results:
        st.warning(f"No SHAP results found in {SHAP_RESULTS_DIR}")
        st.info("Run `python analysis/shap/shap_compute.py` to generate SHAP values first.")
        return
    
    # Format results for selection
    result_options = {
        f"{r['run_name']} ({r['start_date']} to {r['end_date']}) - {r['model_type']}": r['path']
        for r in results
    }
    
    selected_results = st.sidebar.multiselect(
        "Select Results to Analyze",
        options=list(result_options.keys()),
        default=[list(result_options.keys())[0]] if result_options else [],
    )
    
    if not selected_results:
        st.info("Please select at least one SHAP result to analyze.")
        return
    
    # Load selected results
    loaded_results = {}
    for result_label in selected_results:
        result_path = result_options[result_label]
        try:
            shap_df, summary_df, metadata = load_result_data(str(result_path))
            loaded_results[result_label] = {
                "shap_df": shap_df,
                "summary_df": summary_df,
                "metadata": metadata,
                "path": result_path,
            }
        except Exception as e:
            st.error(f"Failed to load {result_label}: {e}")
    
    if not loaded_results:
        st.error("Failed to load any results.")
        return
    
    # Show metadata summary
    with st.sidebar.expander("Result Details", expanded=False):
        for label, data in loaded_results.items():
            meta = data["metadata"]
            st.markdown(f"**{meta.get('run_name', 'Unknown')}**")
            st.write(f"- Model: {meta.get('model_type', '-')}")
            st.write(f"- Features: {meta.get('num_features', '-')}")
            st.write(f"- Samples: {meta.get('num_samples', '-')}")
            st.write(f"- Date: {meta.get('start_date', '')} to {meta.get('end_date', '')}")
            st.divider()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Global Importance",
        "📅 Monthly Importance",
        "📈 Time Series",
        "🔍 Feature Dependence",
        "🎯 Feature Alignment",
    ])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1: Global Importance
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.header("Global Feature Importance")
        st.markdown("Ranking features by mean absolute SHAP value across all samples.")
        
        top_n = st.slider("Top N Features", 5, 50, 20, key="global_top_n")
        
        if len(loaded_results) == 1:
            # Single model view
            label, data = list(loaded_results.items())[0]
            summary_df = data["summary_df"]
            
            if summary_df.empty:
                # Compute from shap_df
                shap_df = data["shap_df"]
                feature_cols = get_feature_columns(shap_df)
                summary_df = pd.DataFrame({
                    "feature": feature_cols,
                    "mean_abs_shap": [np.abs(shap_df[f]).mean() for f in feature_cols],
                }).sort_values("mean_abs_shap", ascending=False)
            
            top_features = summary_df.head(top_n)
            
            fig = px.bar(
                top_features,
                x="feature",
                y="mean_abs_shap",
                title=f"Top {top_n} Features by Mean |SHAP| - {data['metadata'].get('run_name', '')}",
                labels={"mean_abs_shap": "Mean |SHAP|", "feature": "Feature"},
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(top_features.reset_index(drop=True), use_container_width=True)
            
        else:
            # Multi-model comparison
            comparison_data = []
            for label, data in loaded_results.items():
                summary_df = data["summary_df"]
                if summary_df.empty:
                    shap_df = data["shap_df"]
                    feature_cols = get_feature_columns(shap_df)
                    summary_df = pd.DataFrame({
                        "feature": feature_cols,
                        "mean_abs_shap": [np.abs(shap_df[f]).mean() for f in feature_cols],
                    })
                
                for _, row in summary_df.iterrows():
                    comparison_data.append({
                        "model": data["metadata"].get("run_name", label),
                        "feature": row["feature"],
                        "mean_abs_shap": row["mean_abs_shap"],
                    })
            
            comp_df = pd.DataFrame(comparison_data)
            
            # Get top features from first model
            first_model = comp_df["model"].unique()[0]
            top_features = comp_df[comp_df["model"] == first_model].nlargest(top_n, "mean_abs_shap")["feature"].tolist()
            
            plot_df = comp_df[comp_df["feature"].isin(top_features)]
            
            fig = px.bar(
                plot_df,
                x="feature",
                y="mean_abs_shap",
                color="model",
                barmode="group",
                title=f"Feature Importance Comparison - Top {top_n}",
                labels={"mean_abs_shap": "Mean |SHAP|", "feature": "Feature"},
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tab 2: Monthly Importance
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        st.header("Monthly Feature Importance")
        st.markdown("Track how feature importance changes month by month.")
        
        # Use first selected result for monthly analysis
        label, data = list(loaded_results.items())[0]
        shap_df = data["shap_df"].copy()
        metadata = data["metadata"]
        feature_cols = get_feature_columns(shap_df)
        
        # Load targets and merge
        segment_by_target = st.checkbox("Segment by Target (y_true)", value=False, key="monthly_segment")
        
        if segment_by_target:
            targets_df = get_targets(metadata.get("start_date", "2025-04-01"), metadata.get("end_date", "2025-12-31"))
            if not targets_df.empty:
                shap_df = shap_df.merge(targets_df, on="timestamp", how="left")
                st.info(f"Merged {len(targets_df)} targets. {shap_df['y_true'].notna().sum()} matched.")
            else:
                st.warning("No targets available for segmentation.")
                segment_by_target = False
        
        # Compute monthly importance
        monthly_df = compute_monthly_importance(shap_df, feature_cols, segment_by_target=segment_by_target)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            top_n_monthly = st.slider("Top N Features", 5, 30, 10, key="monthly_top_n")
            
            # Segment filter (if segmented)
            if segment_by_target and "segment" in monthly_df.columns:
                segment_options = monthly_df["segment"].unique().tolist()
                selected_segment = st.selectbox("Segment", options=segment_options, index=0)
                plot_monthly_df = monthly_df[monthly_df["segment"] == selected_segment]
            else:
                plot_monthly_df = monthly_df
            
            # Get overall top features
            overall_importance = plot_monthly_df.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False)
            top_features_monthly = overall_importance.head(top_n_monthly).index.tolist()
            
            selected_features = st.multiselect(
                "Features to Display",
                options=feature_cols,
                default=top_features_monthly,
            )
        
        with col2:
            if selected_features:
                plot_df = plot_monthly_df[plot_monthly_df["feature"].isin(selected_features)]
                
                fig = px.line(
                    plot_df,
                    x="month",
                    y="mean_abs_shap",
                    color="feature",
                    title=f"Feature Importance Over Time (Monthly){' - ' + selected_segment if segment_by_target else ''}",
                    labels={"mean_abs_shap": "Mean |SHAP|", "month": "Month"},
                    markers=True,
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Monthly heatmap
        st.subheader("Monthly Importance Heatmap")
        if selected_features:
            pivot_df = plot_monthly_df[plot_monthly_df["feature"].isin(selected_features)].pivot(
                index="feature", columns="month", values="mean_abs_shap"
            )
            
            fig = px.imshow(
                pivot_df,
                title="Feature Importance Heatmap by Month",
                labels={"color": "Mean |SHAP|"},
                aspect="auto",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly table
        st.subheader("Monthly Ranking Table")
        month_select = st.selectbox(
            "Select Month",
            options=sorted(plot_monthly_df["month"].unique()),
            index=len(plot_monthly_df["month"].unique()) - 1 if len(plot_monthly_df["month"].unique()) > 0 else 0,
        )
        
        month_ranking = plot_monthly_df[plot_monthly_df["month"] == month_select].sort_values(
            "mean_abs_shap", ascending=False
        ).head(20)
        st.dataframe(month_ranking.reset_index(drop=True), use_container_width=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3: Time Series
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        st.header("SHAP Contribution Time Series")
        st.markdown("View how SHAP contributions evolve over time for specific features.")
        
        label, data = list(loaded_results.items())[0]
        shap_df_ts = data["shap_df"].copy()
        metadata = data["metadata"]
        feature_cols = get_feature_columns(shap_df_ts)
        
        # Target segmentation option
        color_by_target = st.checkbox("Color by Target (y_true)", value=False, key="ts_segment")
        
        if color_by_target:
            targets_df = get_targets(metadata.get("start_date", "2025-04-01"), metadata.get("end_date", "2025-12-31"))
            if not targets_df.empty:
                shap_df_ts = shap_df_ts.merge(targets_df, on="timestamp", how="left")
                shap_df_ts["y_true"] = shap_df_ts["y_true"].fillna(-1).astype(int)
                st.info(f"Merged targets. y_true=1: {(shap_df_ts['y_true'] == 1).sum()}, y_true=0: {(shap_df_ts['y_true'] == 0).sum()}")
            else:
                st.warning("No targets available.")
                color_by_target = False
        
        # Feature selection
        col1, col2 = st.columns([1, 3])
        
        with col1:
            overall_importance = pd.DataFrame({
                "feature": feature_cols,
                "importance": [np.abs(shap_df_ts[f]).mean() for f in feature_cols],
            }).sort_values("importance", ascending=False)
            
            default_feature = overall_importance.iloc[0]["feature"] if len(overall_importance) > 0 else None
            
            selected_feature = st.selectbox(
                "Select Feature",
                options=feature_cols,
                index=0 if default_feature else 0,
            )
            
            show_rolling = st.checkbox("Show Rolling Average", value=True)
            if show_rolling:
                rolling_window = st.slider("Rolling Window (hours)", 24, 168, 72)
        
        with col2:
            if selected_feature:
                plot_df = shap_df_ts[["timestamp", selected_feature]].copy()
                plot_df["shap_value"] = plot_df[selected_feature]
                
                fig = go.Figure()
                
                if color_by_target and "y_true" in shap_df_ts.columns:
                    # Color-coded scatter by target
                    for target_val, color, name in [(0, "blue", "y_true=0"), (1, "red", "y_true=1")]:
                        mask = shap_df_ts["y_true"] == target_val
                        fig.add_trace(go.Scatter(
                            x=shap_df_ts.loc[mask, "timestamp"],
                            y=shap_df_ts.loc[mask, selected_feature],
                            mode="markers",
                            name=name,
                            marker=dict(size=4, color=color, opacity=0.5),
                        ))
                else:
                    # Raw values line
                    fig.add_trace(go.Scatter(
                        x=plot_df["timestamp"],
                        y=plot_df["shap_value"],
                        mode="lines",
                        name="SHAP Value",
                        line=dict(width=1, color="rgba(100, 100, 255, 0.5)"),
                    ))
                
                if show_rolling:
                    plot_df["rolling_shap"] = plot_df["shap_value"].rolling(rolling_window, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=plot_df["timestamp"],
                        y=plot_df["rolling_shap"],
                        mode="lines",
                        name=f"{rolling_window}h Rolling Avg",
                        line=dict(width=2, color="black"),
                    ))
                
                # Zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                
                fig.update_layout(
                    title=f"SHAP Contribution: {selected_feature}",
                    xaxis_title="Time",
                    yaxis_title="SHAP Contribution",
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Multi-feature comparison
        st.subheader("Compare Multiple Features")
        compare_features = st.multiselect(
            "Select Features to Compare",
            options=feature_cols,
            default=overall_importance.head(3)["feature"].tolist() if len(overall_importance) >= 3 else [],
        )
        
        if compare_features:
            fig = go.Figure()
            for feat in compare_features:
                rolling_vals = shap_df_ts[feat].rolling(72, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=shap_df_ts["timestamp"],
                    y=rolling_vals,
                    mode="lines",
                    name=feat,
                ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="SHAP Contribution Comparison (72h Rolling Avg)",
                xaxis_title="Time",
                yaxis_title="SHAP Contribution",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tab 4: Feature Dependence
    # ─────────────────────────────────────────────────────────────────────────
    with tab4:
        st.header("Feature Dependence Plots")
        st.markdown("Explore the relationship between feature values and SHAP contributions.")
        
        st.warning("Feature dependence plots require loading feature values from the database. "
                   "This is a placeholder - implement feature value loading if needed.")
        
        # For now, show prediction vs SHAP sum
        label, data = list(loaded_results.items())[0]
        shap_df = data["shap_df"]
        feature_cols = get_feature_columns(shap_df)
        
        if "prediction" in shap_df.columns and "base_value" in shap_df.columns:
            st.subheader("Prediction Decomposition")
            
            # Compute SHAP sum
            shap_df["shap_sum"] = shap_df[feature_cols].sum(axis=1)
            shap_df["computed_pred"] = shap_df["base_value"] + shap_df["shap_sum"]
            
            fig = px.scatter(
                shap_df,
                x="computed_pred",
                y="prediction",
                title="Predicted vs SHAP Sum + Base Value",
                labels={"computed_pred": "Base + SHAP Sum", "prediction": "Model Prediction"},
                opacity=0.5,
            )
            
            # Add identity line
            min_val = min(shap_df["computed_pred"].min(), shap_df["prediction"].min())
            max_val = max(shap_df["computed_pred"].max(), shap_df["prediction"].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Identity",
                line=dict(dash="dash", color="red"),
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction distribution
            st.subheader("Prediction Distribution")
            fig = px.histogram(
                shap_df,
                x="prediction",
                nbins=50,
                title="Distribution of Model Predictions",
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tab 5: Feature Alignment
    # ─────────────────────────────────────────────────────────────────────────
    with tab5:
        st.header("Feature Alignment Analysis")
        st.markdown("""
        Measures whether features contribute in the **correct direction** relative to the target:
        - **Correct direction**: SHAP > 0 when y=1, or SHAP < 0 when y=0
        """)
        
        # Use first selected result
        label, data = list(loaded_results.items())[0]
        shap_df_align = data["shap_df"].copy()
        metadata = data["metadata"]
        feature_cols = get_feature_columns(shap_df_align)
        
        # Load and merge targets
        targets_df = get_targets(metadata.get("start_date", "2025-04-01"), metadata.get("end_date", "2025-12-31"))
        
        if targets_df.empty:
            st.error("No targets available for alignment analysis.")
        else:
            shap_df_align = shap_df_align.merge(targets_df, on="timestamp", how="inner")
            st.info(f"Loaded {len(shap_df_align)} samples with targets. y=1: {(shap_df_align['y_true'] == 1).sum()}, y=0: {(shap_df_align['y_true'] == 0).sum()}")
            
            # Compute alignment stats per feature
            alignment_records = []
            for feat in feature_cols:
                shap_vals = shap_df_align[feat]
                y_true = shap_df_align["y_true"]
                
                # Directional Accuracy: % where (y=1 & SHAP>0) or (y=0 & SHAP<0)
                correct_direction = ((y_true == 1) & (shap_vals > 0)) | ((y_true == 0) & (shap_vals < 0))
                directional_accuracy = correct_direction.mean() * 100
                
                # Target Separation: mean(SHAP|y=1) - mean(SHAP|y=0)
                mean_shap_y1 = shap_vals[y_true == 1].mean()
                mean_shap_y0 = shap_vals[y_true == 0].mean()
                target_separation = mean_shap_y1 - mean_shap_y0
                
                alignment_records.append({
                    "feature": feat,
                    "directional_accuracy": directional_accuracy,
                    "mean_shap_y1": mean_shap_y1,
                    "mean_shap_y0": mean_shap_y0,
                    "target_separation": target_separation,
                    "mean_abs_shap": np.abs(shap_vals).mean(),
                })
            
            align_df = pd.DataFrame(alignment_records).sort_values("target_separation", ascending=False)
            
            # Display overall stats
            st.subheader("Alignment Summary (All Data)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Target Separation bar chart
                fig = px.bar(
                    align_df.head(20),
                    x="feature",
                    y="target_separation",
                    title="Target Separation (mean SHAP y=1 - mean SHAP y=0)",
                    labels={"target_separation": "Target Separation", "feature": "Feature"},
                    color="target_separation",
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Directional Accuracy bar chart
                fig = px.bar(
                    align_df.sort_values("directional_accuracy", ascending=False).head(20),
                    x="feature",
                    y="directional_accuracy",
                    title="Directional Accuracy (%)",
                    labels={"directional_accuracy": "Accuracy (%)", "feature": "Feature"},
                    color="directional_accuracy",
                    color_continuous_scale="Blues",
                )
                fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Random (50%)")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Full table
            st.dataframe(
                align_df.style.format({
                    "directional_accuracy": "{:.1f}%",
                    "mean_shap_y1": "{:.4f}",
                    "mean_shap_y0": "{:.4f}",
                    "target_separation": "{:.4f}",
                    "mean_abs_shap": "{:.4f}",
                }),
                use_container_width=True,
            )
            
            # ─────────────────────────────────────────────────────────────────────
            # Monthly Time Series
            # ─────────────────────────────────────────────────────────────────────
            st.subheader("Monthly Alignment Trend")
            
            shap_df_align["month"] = shap_df_align["timestamp"].dt.to_period("M").astype(str)
            
            # Compute monthly stats
            monthly_align_records = []
            for month in sorted(shap_df_align["month"].unique()):
                month_df = shap_df_align[shap_df_align["month"] == month]
                for feat in feature_cols:
                    shap_vals = month_df[feat]
                    y_true = month_df["y_true"]
                    
                    correct_direction = ((y_true == 1) & (shap_vals > 0)) | ((y_true == 0) & (shap_vals < 0))
                    directional_accuracy = correct_direction.mean() * 100 if len(month_df) > 0 else 50
                    
                    mean_shap_y1 = shap_vals[y_true == 1].mean() if (y_true == 1).sum() > 0 else 0
                    mean_shap_y0 = shap_vals[y_true == 0].mean() if (y_true == 0).sum() > 0 else 0
                    target_separation = mean_shap_y1 - mean_shap_y0
                    
                    monthly_align_records.append({
                        "month": month,
                        "feature": feat,
                        "directional_accuracy": directional_accuracy,
                        "target_separation": target_separation,
                    })
            
            monthly_align_df = pd.DataFrame(monthly_align_records)
            
            # Feature selection for time series
            top_features_align = feature_cols  # Default to all features as requested
            selected_features_align = st.multiselect(
                "Features to Plot",
                options=feature_cols,
                default=top_features_align,
                key="align_features",
            )
            
            if selected_features_align:
                plot_df = monthly_align_df[monthly_align_df["feature"].isin(selected_features_align)]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.line(
                        plot_df,
                        x="month",
                        y="target_separation",
                        color="feature",
                        title="Target Separation Over Time",
                        labels={"target_separation": "Target Separation", "month": "Month"},
                        markers=True,
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.line(
                        plot_df,
                        x="month",
                        y="directional_accuracy",
                        color="feature",
                        title="Directional Accuracy Over Time",
                        labels={"directional_accuracy": "Accuracy (%)", "month": "Month"},
                        markers=True,
                    )
                    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Random")
                    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

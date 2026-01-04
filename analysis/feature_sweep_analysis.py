#!/usr/bin/env python3
"""
Feature Sweep Analysis App

Streamlit app to analyze feature-family sweep runs:
- Metrics and feature importance aggregation
- Family vs without-family performance comparison
- In-family model comparison

Usage:
    streamlit run analysis/feature_sweep_analysis.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

TRACKING_URI = "http://127.0.0.1:5001"

DEFAULT_FAMILIES: Dict[str, List[str]] = {
    "adx": ["adx_168", "adx_24", "adx_720"],
    "cvar_var_ratio": ["cvar_var_ratio_168", "cvar_var_ratio_720"],
    "efficiency_avg": ["efficiency_avg_168", "efficiency_avg_24"],
    "mean_cross_rate_ema": [
        "mean_cross_rate_ema_168_168",
        "mean_cross_rate_ema_168_48",
        "mean_cross_rate_ema_24_168",
        "mean_cross_rate_ema_24_48",
        "mean_cross_rate_ema_720_168",
        "mean_cross_rate_ema_720_48",
    ],
    "parkinson_volatility": ["parkinson_volatility_168", "parkinson_volatility_24"],
    "price_vwap_distance_zscore": [
        "price_vwap_distance_zscore_168_168",
        "price_vwap_distance_zscore_24_168",
        "price_vwap_distance_zscore_720_168",
    ],
    "pullback_slope_vwap": [
        "pullback_slope_vwap_168_168",
        "pullback_slope_vwap_168_48",
        "pullback_slope_vwap_24_168",
        "pullback_slope_vwap_24_48",
        "pullback_slope_vwap_720_168",
        "pullback_slope_vwap_720_48",
    ],
    "range_stretch_interaction": [
        "range_stretch_interaction_168_168",
        "range_stretch_interaction_168_24",
        "range_stretch_interaction_720_168",
        "range_stretch_interaction_720_24",
    ],
    "relative_volume": ["relative_volume_30", "relative_volume_7"],
    "return_autocorr": ["return_autocorr_168", "return_autocorr_48"],
    "rsi": ["rsi_168", "rsi_24", "rsi_720"],
    "scaled_acceleration": ["scaled_acceleration_168"],
    "variance_ratio": ["variance_ratio_24_168", "variance_ratio_24_48", "variance_ratio_24_720"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading Helpers
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data
def get_all_experiments(tracking_uri: str) -> List[Dict[str, Any]]:
    mlflow.set_tracking_uri(tracking_uri)
    experiments = mlflow.search_experiments()
    return [
        {"experiment_id": exp.experiment_id, "name": exp.name}
        for exp in experiments
        if exp.name != "Default"
    ]


@st.cache_data
def get_runs_for_experiment(tracking_uri: str, experiment_id: str) -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time ASC"],
    )


@st.cache_data
def load_metrics(run_dir: str) -> Dict[str, Any]:
    metrics_path = Path(run_dir) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_feature_importance(run_dir: str) -> Optional[pd.DataFrame]:
    fi_path = Path(run_dir) / "feature_importance.csv"
    if fi_path.exists():
        return pd.read_csv(fi_path)
    return None


def aggregate_feature_importance(run_dirs: Sequence[str]) -> Optional[pd.DataFrame]:
    all_fi = []
    for run_dir in run_dirs:
        fi_df = load_feature_importance(run_dir)
        if fi_df is None or fi_df.empty:
            continue
        if "feature" not in fi_df.columns:
            continue
        # Normalize importance column name
        if "importance" in fi_df.columns:
            imp_col = "importance"
        elif "importance_gain" in fi_df.columns:
            imp_col = "importance_gain"
        else:
            continue
        tmp = fi_df[["feature", imp_col]].copy()
        tmp.columns = ["feature", "importance"]
        all_fi.append(tmp)
    if not all_fi:
        return None
    combined = pd.concat(all_fi, ignore_index=True)
    agg = combined.groupby("feature")["importance"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    agg = agg.sort_values("mean", ascending=False)
    agg.columns = ["Feature", "Mean", "Std", "Min", "Max", "Runs"]
    return agg


def load_family_map(path: Optional[str]) -> Dict[str, List[str]]:
    default_path = Path("configs/feature_lists/feature_families_default_plus_time_vol_ratio.json")
    if not path:
        path = str(default_path)
    p = Path(path)
    if not p.exists():
        st.warning(f"Family map not found: {p}")
        return DEFAULT_FAMILIES
    with open(p, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        st.warning("Family map must be a JSON object: {family: [features...]}")
        return DEFAULT_FAMILIES
    cleaned: Dict[str, List[str]] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, list) and all(isinstance(x, str) for x in v):
            cleaned[k] = v
    return cleaned or DEFAULT_FAMILIES


def build_family_lookup(family_map: Dict[str, List[str]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for fam, feats in family_map.items():
        for feat in feats:
            lookup[feat] = fam
    return lookup


def parse_combo_families(raw: Any) -> List[str]:
    if not raw or not isinstance(raw, str):
        return []
    return [f.strip() for f in raw.split(",") if f.strip()]


def parse_combo_features(raw: Any) -> List[str]:
    if not raw or not isinstance(raw, str):
        return []
    return [f.strip() for f in raw.split(",") if f.strip()]


def build_run_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for _, row in runs_df.iterrows():
        run_dir = row.get("params.run_dir")
        if not run_dir or not Path(run_dir).exists():
            continue
        combo_str = row.get("params.combo_families")
        combo_family = row.get("params.combo_family")
        combo_features_str = row.get("params.combo_features")
        families = parse_combo_families(combo_str)
        if not families and isinstance(combo_family, str) and combo_family:
            families = [combo_family]
        combo_features = parse_combo_features(combo_features_str)
        metrics = load_metrics(run_dir)
        combo_size = row.get("params.combo_size")
        if combo_size is None and combo_features:
            combo_size = len(combo_features)
        record: Dict[str, Any] = {
            "run_id": row.get("run_id"),
            "run_dir": run_dir,
            "combo_families": combo_str or "",
            "combo_family": combo_family or "",
            "combo_features": combo_features_str or "",
            "combo_size": len(families),
            "families": families,
        }
        if combo_size is not None:
            try:
                record["combo_size"] = int(combo_size)
            except Exception:
                record["combo_size"] = combo_size
        record.update(metrics)
        records.append(record)
    df = pd.DataFrame(records)
    if "auc_train" in df.columns and "auc_test" in df.columns:
        denom = df["auc_test"].replace(0, np.nan)
        df["auc_train/auc_test"] = df["auc_train"] / denom
    return df


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in metrics_df.columns
        if c not in ("run_id", "run_dir", "combo_families", "combo_size", "families")
        and pd.api.types.is_numeric_dtype(metrics_df[c])
    ]
    if not metric_cols:
        return pd.DataFrame()
    summary = metrics_df[metric_cols].describe().T
    summary["cv"] = summary["std"] / summary["mean"]
    return summary


def family_vs_without(metrics_df: pd.DataFrame, families: Sequence[str], metric: str) -> pd.DataFrame:
    rows = []
    for fam in families:
        mask_with = metrics_df["families"].apply(lambda x: fam in x)
        with_vals = metrics_df.loc[mask_with, metric].dropna()
        without_vals = metrics_df.loc[~mask_with, metric].dropna()
        if with_vals.empty or without_vals.empty:
            continue
        rows.append({
            "family": fam,
            "with_mean": with_vals.mean(),
            "without_mean": without_vals.mean(),
            "delta": with_vals.mean() - without_vals.mean(),
            "with_n": len(with_vals),
            "without_n": len(without_vals),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("delta", ascending=False)
    return out


def cooccurrence_effect(metrics_df: pd.DataFrame, target_family: str, metric: str) -> pd.DataFrame:
    rows = []
    base = metrics_df[metrics_df["families"].apply(lambda x: target_family in x)].copy()
    for fam in sorted({f for fs in base["families"] for f in fs if f != target_family}):
        mask = base["families"].apply(lambda x: fam in x)
        vals = base.loc[mask, metric].dropna()
        if vals.empty:
            continue
        rows.append({"family": fam, "mean": vals.mean(), "n": len(vals)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("mean", ascending=False)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────


def plot_family_delta(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["delta"],
        y=df["family"],
        orientation="h",
        marker_color="steelblue",
    ))
    fig.update_layout(
        title="Family Impact (With - Without)",
        xaxis_title="Metric Delta",
        yaxis_title="Family",
        yaxis=dict(autorange="reversed"),
        height=max(400, 20 * len(df)),
    )
    return fig


def plot_feature_importance(fi_df: pd.DataFrame, top_n: int):
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
        height=max(400, top_n * 22),
    )
    return fig


def plot_box_by_group(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    fig = px.box(df, x=x_col, y=y_col, points="all", title=title)
    fig.update_layout(height=350, margin=dict(t=40, b=40))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Feature Sweep Analysis",
        page_icon="🧪",
        layout="wide",
    )
    st.title("🧪 Feature Sweep Analysis")
    st.markdown("Analyze MLflow feature-family sweep experiments.")

    with st.sidebar:
        st.header("⚙️ Configuration")
        tracking_uri = st.text_input("MLflow Tracking URI", value=TRACKING_URI)
        family_map_path = st.text_input("Family Map JSON (optional)", value="")
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

        try:
            experiments = get_all_experiments(tracking_uri)
            if not experiments:
                st.error("No experiments found in MLflow.")
                return
        except Exception as e:
            st.error(f"Could not connect to MLflow: {e}")
            st.info(f"Make sure MLflow server is running at {tracking_uri}")
            return

        exp_names = [exp["name"] for exp in experiments]
        selected_exp_name = st.selectbox("Select Experiment", options=exp_names, index=0)
        selected_exp = next((exp for exp in experiments if exp["name"] == selected_exp_name), None)
        if selected_exp is None:
            st.error("Experiment not found.")
            return

        st.caption(f"Experiment ID: {selected_exp['experiment_id']}")

    runs_df = get_runs_for_experiment(tracking_uri, selected_exp["experiment_id"])
    if runs_df.empty:
        st.warning("No runs found for this experiment.")
        return

    run_table = build_run_table(runs_df)
    if run_table.empty:
        st.warning("No valid run directories found. Ensure params.run_dir is logged.")
        return

    st.info(f"📊 Found **{len(run_table)}** runs with valid artifacts")

    family_map = load_family_map(family_map_path)
    family_lookup = build_family_lookup(family_map)
    observed_families = sorted({f for fs in run_table["families"] for f in fs})

    metric_cols = [
        c for c in run_table.columns
        if c not in (
            "run_id",
            "run_dir",
            "combo_families",
            "combo_family",
            "combo_features",
            "combo_size",
            "families",
        )
        and pd.api.types.is_numeric_dtype(run_table[c])
    ]
    default_metric = "auc_test" if "auc_test" in metric_cols else (metric_cols[0] if metric_cols else None)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Metrics",
        "🎯 Feature Importance",
        "🧭 Family vs Without",
        "🔬 In-Family Comparison",
        "🧩 Combo Comparison",
    ])

    with tab1:
        st.header("📈 Metrics Summary")
        summary = summarize_metrics(run_table)
        if summary.empty:
            st.warning("No numeric metrics found.")
        else:
            st.dataframe(summary.style.format("{:.4f}"), use_container_width=True)

        if "combo_size" in run_table.columns:
            st.subheader("Runs by Combo Size")
            counts = run_table["combo_size"].value_counts().sort_index()
            fig = px.bar(x=counts.index, y=counts.values, labels={"x": "Combo Size", "y": "Runs"})
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("🎯 Feature Importance (Aggregated)")
        run_dirs = run_table["run_dir"].tolist()
        fi_agg = aggregate_feature_importance(run_dirs)
        if fi_agg is None or fi_agg.empty:
            st.warning("No feature importance data found.")
        else:
            top_n = st.slider("Top N Features", min_value=5, max_value=max(10, len(fi_agg)), value=20)
            st.plotly_chart(plot_feature_importance(fi_agg, top_n), use_container_width=True)
            st.dataframe(fi_agg.head(top_n), use_container_width=True)

            # Family-level aggregation
            st.subheader("Family-Level Importance")
            tmp = fi_agg.copy()
            tmp["Family"] = tmp["Feature"].map(family_lookup).fillna("other")
            fam_agg = tmp.groupby("Family")["Mean"].sum().reset_index().sort_values("Mean", ascending=False)
            fig = px.bar(fam_agg, x="Mean", y="Family", orientation="h")
            fig.update_layout(height=max(350, 20 * len(fam_agg)))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("🧭 Family vs Without")
        if not metric_cols:
            st.warning("No numeric metrics available for comparison.")
        else:
            metric = st.selectbox("Metric", options=metric_cols, index=metric_cols.index(default_metric))
            if not observed_families:
                st.warning("No combo_families found in this experiment.")
            else:
                comp_df = family_vs_without(run_table, observed_families, metric)
                if comp_df.empty:
                    st.warning("Not enough runs to compare with/without.")
                else:
                    st.dataframe(
                        comp_df.style.format({
                            "with_mean": "{:.4f}",
                            "without_mean": "{:.4f}",
                            "delta": "{:+.4f}",
                        }),
                        use_container_width=True,
                    )
                    st.plotly_chart(plot_family_delta(comp_df), use_container_width=True)

                st.subheader("Distribution for Selected Family")
                selected_family = st.selectbox("Family", options=observed_families)
                selected_metrics = st.multiselect(
                    "Metrics",
                    options=metric_cols,
                    default=[metric] if metric in metric_cols else metric_cols[:1],
                    help="Show distribution tables for each selected metric",
                )
                mask = run_table["families"].apply(lambda x: selected_family in x)
                for m in selected_metrics:
                    dist_df = run_table.loc[:, [m]].copy()
                    dist_df["group"] = np.where(mask, "with", "without")
                    dist_df = dist_df.dropna()
                    if dist_df.empty:
                        st.info(f"No data for {m}.")
                        continue
                    summary_rows = []
                    for grp, grp_df in dist_df.groupby("group"):
                        values = grp_df[m].dropna()
                        if values.empty:
                            continue
                        summary_rows.append({
                            "group": grp,
                            "count": int(values.count()),
                            "mean": float(values.mean()),
                            "std": float(values.std(ddof=0)),
                            "min": float(values.min()),
                            "p25": float(values.quantile(0.25)),
                            "median": float(values.median()),
                            "p75": float(values.quantile(0.75)),
                            "max": float(values.max()),
                        })
                    if summary_rows:
                        st.markdown(f"**{m}**")
                        summary_df = pd.DataFrame(summary_rows).sort_values("group")
                        st.dataframe(
                            summary_df.style.format({
                                "mean": "{:.4f}",
                                "std": "{:.4f}",
                                "min": "{:.4f}",
                                "p25": "{:.4f}",
                                "median": "{:.4f}",
                                "p75": "{:.4f}",
                                "max": "{:.4f}",
                            }),
                            use_container_width=True,
                        )

    with tab4:
        st.header("🔬 In-Family Model Comparison")
        if not metric_cols:
            st.warning("No numeric metrics available.")
        elif not observed_families:
            st.warning("No combo_families found in this experiment.")
        else:
            metrics_selected = st.multiselect(
                "Metrics",
                options=metric_cols,
                default=[default_metric] if default_metric in metric_cols else metric_cols[:1],
                key="in_family_metrics",
            )
            target_family = st.selectbox("Family", options=observed_families, key="in_family_target")

            family_runs = run_table[run_table["families"].apply(lambda x: target_family in x)].copy()
            if family_runs.empty:
                st.warning("No runs contain the selected family.")
            else:
                st.subheader("Top Runs Containing Family")
                max_runs = min(50, len(family_runs))
                if max_runs <= 5:
                    top_n = max_runs
                    st.caption(f"Top N Runs fixed at {top_n} (only {max_runs} runs available).")
                else:
                    top_n = st.slider(
                        "Top N Runs",
                        min_value=5,
                        max_value=max_runs,
                        value=min(15, max_runs),
                    )
                if not metrics_selected:
                    st.info("Select at least one metric.")
                    return
                sort_metric = metrics_selected[0]
                top = family_runs.sort_values(sort_metric, ascending=False).head(top_n)
                display_cols = ["combo_families", "combo_family", "combo_features", "combo_size"] + metrics_selected
                display_cols = [c for c in display_cols if c in top.columns]
                fmt_map = {m: "{:.4f}" for m in metrics_selected if m in top.columns}
                st.dataframe(top[display_cols].style.format(fmt_map), use_container_width=True)

                st.subheader("Metric by Combo Size")
                if "combo_size" in family_runs.columns:
                    for m in metrics_selected:
                        if m not in family_runs.columns:
                            continue
                        fig = plot_box_by_group(family_runs, "combo_size", m, f"{m} by combo size")
                        st.plotly_chart(fig, use_container_width=True)

                st.subheader("Co-occurring Family Effect")
                for m in metrics_selected:
                    if m not in run_table.columns:
                        continue
                    co_df = cooccurrence_effect(run_table, target_family, m)
                    if co_df.empty:
                        st.info(f"No co-occurring families found for {m}.")
                        continue
                    fig = px.bar(co_df, x="mean", y="family", orientation="h")
                    fig.update_layout(height=max(300, 20 * len(co_df)))
                    fig.update_layout(title=f"{m} by co-occurring family")
                    st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.header("🧩 Combo Comparison")
        st.markdown("Compare a base family set vs base + additional family.")
        if not metric_cols:
            st.warning("No numeric metrics available.")
        elif not observed_families:
            st.warning("No combo_families found in this experiment.")
        else:
            base_families = st.multiselect(
                "Base families",
                options=observed_families,
                default=[],
                help="Select the base family set to compare against",
                key="combo_base_families",
            )
            remaining = [f for f in observed_families if f not in base_families]
            add_families = st.multiselect(
                "Add families",
                options=remaining,
                default=remaining[:1],
                help="Compare base vs base + each selected family",
                key="combo_add_families",
            )
            match_mode = st.radio(
                "Match Mode",
                options=["Exact combo", "Superset (contains base)"],
                index=0,
                help="Exact combo requires the family set to match exactly; superset allows larger combos",
                horizontal=True,
            )
            selected_metrics = st.multiselect(
                "Metrics",
                options=metric_cols,
                default=[default_metric] if default_metric in metric_cols else metric_cols[:1],
                key="combo_metrics",
            )

            if not base_families:
                st.info("Pick at least one base family to compare.")
            elif not add_families:
                st.info("Pick at least one family to add.")
            elif not selected_metrics:
                st.info("Select at least one metric.")
            else:
                base_set = set(base_families)
                mode_exact = match_mode == "Exact combo"

                def _match(fams: Sequence[str], target: set) -> bool:
                    fam_set = set(fams)
                    return fam_set == target if mode_exact else target.issubset(fam_set)

                base_mask = run_table["families"].apply(lambda x: _match(x, base_set))
                base_runs = run_table[base_mask]
                if base_runs.empty:
                    st.warning("No runs matched the base family set with the selected mode.")
                else:
                    st.caption(f"Base runs matched: {len(base_runs)}")

                for metric in selected_metrics:
                    rows = []
                    for fam in add_families:
                        target_set = set(base_set)
                        target_set.add(fam)
                        add_mask = run_table["families"].apply(lambda x: _match(x, target_set))
                        add_runs = run_table[add_mask]
                        base_vals = base_runs[metric].dropna()
                        add_vals = add_runs[metric].dropna()
                        if base_vals.empty or add_vals.empty:
                            rows.append({
                                "add_family": fam,
                                "base_n": len(base_vals),
                                "add_n": len(add_vals),
                                "base_mean": np.nan,
                                "add_mean": np.nan,
                                "delta": np.nan,
                            })
                            continue
                        base_mean = float(base_vals.mean())
                        add_mean = float(add_vals.mean())
                        rows.append({
                            "add_family": fam,
                            "base_n": len(base_vals),
                            "add_n": len(add_vals),
                            "base_mean": base_mean,
                            "add_mean": add_mean,
                            "delta": add_mean - base_mean,
                        })

                    out = pd.DataFrame(rows)
                    if not out.empty:
                        out = out.sort_values("delta", ascending=False, na_position="last")
                        st.markdown(f"**{metric}**")
                        st.dataframe(
                            out.style.format({
                                "base_mean": "{:.4f}",
                                "add_mean": "{:.4f}",
                                "delta": "{:+.4f}",
                            }),
                            use_container_width=True,
                        )


if __name__ == "__main__":
    main()

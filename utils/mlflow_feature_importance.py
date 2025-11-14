"""
Helpers to aggregate top-N feature names from MLflow artifacts.

Inputs
- experiment_name: MLflow experiment name
- metric_name, metric_threshold: filter runs by metric (>= threshold by default)
- top_n: number of top features to take from each run's feature_importance.csv

Output
- Dict[str, int]: feature -> number of runs where it appears in the top-N

Notes
- Requires access to the configured MLflow tracking server/local store.
- Looks for an artifact named 'feature_importance.csv' under each matching run.
- Attempts to parse flexible CSV schemas by locating columns containing
  'feature' and one of {'importance','gain','split','weight'} (case-insensitive).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class RunRef:
    run_id: str
    metrics: Dict[str, float]
    tags: Dict[str, str]


def _load_mlflow_client(tracking_uri: Optional[str] = None):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as e:
        raise RuntimeError(
            "mlflow is required for utils.mlflow_feature_importance. Install mlflow and configure tracking URI."
        ) from e

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    # Registry follows tracking by default; not strictly needed here.
    return MlflowClient()


def _get_experiment_id(client, experiment_name: str) -> str:
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")
    return exp.experiment_id


def _search_runs(
    client,
    experiment_id: str,
    metric_name: str,
    metric_threshold: float,
    comparator: str = ">=",
    max_results: int = 5000,
    order_by: Optional[str] = None,
) -> List[RunRef]:
    # MLflow filter syntax: metrics.<name> >= value AND attributes.status = 'FINISHED'
    cmp = comparator.strip()
    if cmp not in {">=", ">", "<=", "<", "=", "=="}:
        raise ValueError("comparator must be one of >=, >, <=, <, =, ==")
    # Use single '=' for MLflow syntax
    if cmp == "==":
        cmp = "="
    flt = f"metrics.{metric_name} {cmp} {metric_threshold} and attributes.status = 'FINISHED'"
    runs = client.search_runs(
        [experiment_id], filter_string=flt, max_results=max_results, order_by=[order_by] if order_by else None
    )
    out: List[RunRef] = []
    for r in runs:
        out.append(RunRef(run_id=r.info.run_id, metrics=dict(r.data.metrics), tags=dict(r.data.tags)))
    return out


def _list_all_artifacts(client, run_id: str, path: str = "") -> List[str]:
    """Return a list of artifact relative paths (files only)."""
    stack = [path]
    files: List[str] = []
    while stack:
        p = stack.pop()
        for fi in client.list_artifacts(run_id, p):
            if fi.is_dir:
                stack.append(fi.path)
            else:
                files.append(fi.path)
    return files


def _find_feature_importance_paths(client, run_id: str) -> List[str]:
    # Collect all files and filter by basename match
    try:
        all_files = _list_all_artifacts(client, run_id, path="")
    except Exception:
        # Some stores may not support listing; try common path directly
        all_files = []
    if not all_files:
        # Best-effort guess
        candidates = ["feature_importance.csv"]
    else:
        candidates = [p for p in all_files if p.lower().endswith("feature_importance.csv")]
        if not candidates:
            candidates = [p for p in all_files if Path(p).name.lower() == "feature_importance.csv"]
        if not candidates:
            candidates = ["feature_importance.csv"]
    return candidates


def _read_importance_csv(client, run_id: str, rel_path: str) -> pd.DataFrame:
    # Download to a temp dir and read
    import tempfile
    import os

    local_dir = tempfile.mkdtemp(prefix=f"mlflow_imp_{run_id[:8]}_")
    local_path = client.download_artifacts(run_id, rel_path, local_dir)
    # download_artifacts may return a directory if rel_path is a directory.
    if os.path.isdir(local_path):
        # Try to find a CSV in it
        for root, _, files in os.walk(local_path):
            for f in files:
                if f.lower().endswith(".csv") and f.lower() == "feature_importance.csv":
                    return pd.read_csv(os.path.join(root, f))
        # fallback to any csv
        for root, _, files in os.walk(local_path):
            for f in files:
                if f.lower().endswith(".csv"):
                    return pd.read_csv(os.path.join(root, f))
        raise FileNotFoundError(f"No CSV found under downloaded artifacts: {local_path}")
    else:
        return pd.read_csv(local_path)


def _extract_top_features(df: pd.DataFrame, top_n: int) -> List[str]:
    if df is None or df.empty:
        return []
    cols = {c.lower(): c for c in df.columns}
    # Find feature col
    feature_col = cols.get("feature") or next((c for c in df.columns if "feature" in c.lower()), None)
    if not feature_col:
        # Heuristic: first column
        feature_col = df.columns[0]

    # Find importance-like column
    importance_col = None
    for key in ("importance", "gain", "gain_importance", "weight", "split"):
        importance_col = cols.get(key) or next((c for c in df.columns if key in c.lower()), None)
        if importance_col:
            break

    if importance_col:
        tmp = df[[feature_col, importance_col]].copy()
        tmp[importance_col] = pd.to_numeric(tmp[importance_col], errors="coerce").fillna(0.0)
        tmp = tmp.sort_values(importance_col, ascending=False)
        feats = tmp[feature_col].astype(str).head(int(top_n)).tolist()
    else:
        feats = df[feature_col].astype(str).head(int(top_n)).tolist()

    # Deduplicate while preserving order
    seen = set()
    out = []
    for f in feats:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def get_feature_occurrence_counts(
    experiment_name: str,
    metric_name: str,
    metric_threshold: float,
    top_n: int,
    *,
    tracking_uri: Optional[str] = None,
    comparator: str = ">=",
    order_by: Optional[str] = None,
) -> Dict[str, int]:
    """Return a dict feature -> count across runs.

    - Filters MLflow runs in `experiment_name` with `metrics.<metric_name> comparator metric_threshold`.
    - For each run, searches for 'feature_importance.csv' in artifacts, extracts top-N features,
      and increments counts.
    - comparator can be one of ">=, >, <=, <, =, ==" (default ">=").
    - order_by is optional MLflow order string (e.g., "metrics.auc DESC").
    """
    client = _load_mlflow_client(tracking_uri)
    exp_id = _get_experiment_id(client, experiment_name)
    runs = _search_runs(client, exp_id, metric_name, metric_threshold, comparator=comparator, order_by=order_by)

    counts: Dict[str, int] = {}
    for r in runs:
        paths = _find_feature_importance_paths(client, r.run_id)
        csv_df = None
        for p in paths:
            try:
                csv_df = _read_importance_csv(client, r.run_id, p)
                if not csv_df.empty:
                    break
            except Exception:
                continue
        if csv_df is None or csv_df.empty:
            continue
        top_feats = _extract_top_features(csv_df, top_n)
        for f in top_feats:
            counts[f] = counts.get(f, 0) + 1
    return counts


def get_feature_occurrence_series(
    experiment_name: str,
    metric_name: str,
    metric_threshold: float,
    top_n: int,
    **kwargs,
) -> pd.Series:
    """Same as get_feature_occurrence_counts but returns a sorted pandas Series."""
    counts = get_feature_occurrence_counts(
        experiment_name, metric_name, metric_threshold, top_n, **kwargs
    )
    if not counts:
        return pd.Series(dtype=int)
    return pd.Series(counts, dtype=int).sort_values(ascending=False)


# Example usage (documentation only):
# counts = get_feature_occurrence_counts(
#     experiment_name="cex-btcusdt-p60",
#     metric_name="auc",
#     metric_threshold=0.70,
#     top_n=100,
#     comparator=">=",
# )


def _cli_main() -> None:  # pragma: no cover - manual test CLI
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Aggregate top-N feature names from MLflow runs' feature_importance.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", required=True, help="MLflow experiment name")
    parser.add_argument("--metric", required=True, help="Metric name to filter (e.g., auc, rmse)")
    parser.add_argument("--threshold", required=True, type=float, help="Metric threshold for filtering runs")
    parser.add_argument("--top-n", dest="top_n", type=int, default=100, help="Top-N features per run to count")
    parser.add_argument(
        "--comparator",
        choices=[">=", ">", "<=", "<", "=", "=="],
        default=">=",
        help="Comparator for metric threshold",
    )
    parser.add_argument("--tracking-uri", default=None, help="Override MLflow tracking URI")
    parser.add_argument("--order-by", default=None, help='Optional MLflow order string, e.g., "metrics.auc DESC"')
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows printed")

    args = parser.parse_args()

    try:
        series = get_feature_occurrence_series(
            experiment_name=args.experiment,
            metric_name=args.metric,
            metric_threshold=args.threshold,
            top_n=args.top_n,
            tracking_uri=args.tracking_uri,
            comparator=args.comparator,
            order_by=args.order_by,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    if args.limit is not None and not series.empty:
        series = series.head(int(args.limit))

    if args.format == "json":
        print(json.dumps(series.to_dict(), indent=2))
    else:
        if series.empty:
            print("No features found (no matching runs or artifacts).")
            return
        width = max(4, len(str(int(series.max()))))
        for feat, cnt in series.items():
            print(f"{cnt:>{width}d}  {feat}")


if __name__ == "__main__":  # pragma: no cover - manual test entry
    _cli_main()

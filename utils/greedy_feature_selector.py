"""
Greedy correlation-based feature selector (MLflow-first simplified).

Workflow
- Build candidate features from MLflow importance counts (top-N per run).
- Order candidates by occurrence count, then tie-break by NaN rate, rank-std, shorter TF.
- Greedy prune by absolute Spearman correlation threshold and per-family cap.

Notes
- Target-free; uses Spearman with pairwise-complete observations.
- Families are derived by stripping a trailing timeframe suffix (e.g., _1H/_4H/_12H/_1D).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .mlflow_feature_importance import get_feature_occurrence_series


_TF_UNIT_TO_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}


def _parse_tf_suffix(col: str, known_tfs: Optional[Sequence[str]] = None) -> Tuple[Optional[str], Optional[int]]:
    if known_tfs:
        for tf in known_tfs:
            if col.endswith("_" + tf):
                num = None
                unit = None
                for i, ch in enumerate(tf):
                    if not ch.isdigit():
                        num = tf[:i]
                        unit = tf[i:].lower()
                        break
                if num and unit and unit[0] in _TF_UNIT_TO_SECONDS:
                    return tf, int(num) * _TF_UNIT_TO_SECONDS[unit[0]]
                return tf, None
    import re
    m = re.search(r"_(\d+)([smhdwSMHDW])$", col)
    if not m:
        return None, None
    num, unit = int(m.group(1)), m.group(2).lower()
    return f"{num}{unit}", num * _TF_UNIT_TO_SECONDS.get(unit, 1)


def _family_from_name(col: str, known_tfs: Optional[Sequence[str]] = None) -> str:
    tf, _ = _parse_tf_suffix(col, known_tfs)
    if tf is None:
        return col
    # Remove trailing _{tf}
    return col[: -(len(tf) + 1)]


def _is_excluded(col: str, exclude_suffixes: Optional[Sequence[str]], known_tfs: Optional[Sequence[str]]) -> bool:
    if not exclude_suffixes:
        return False
    base = _family_from_name(col, known_tfs)
    for suf in exclude_suffixes:
        if base.endswith(str(suf)):
            return True
    return False


def _rank_std(series: pd.Series) -> float:
    r = series.rank(method="average", pct=True)
    return float(r.std(skipna=True)) if r.notna().any() else 0.0


def _nan_rate(series: pd.Series) -> float:
    n = series.isna().mean()
    return float(n) if not np.isnan(n) else 1.0


def _tf_pref_weight(feature: str, known_tfs: Optional[Sequence[str]]) -> float:
    tf, secs = _parse_tf_suffix(feature, known_tfs)
    if secs is None or secs <= 0:
        return 1.0
    w = secs ** -0.3
    return float(w)


def _quality_score(
    X: pd.DataFrame,
    *,
    known_tfs: Optional[Sequence[str]] = None,
    nan_penalty: float = 1.0,
) -> pd.Series:
    """Unsupervised feature quality ranking used to order non-candidate columns.

    Score = rank-std × (1 − NaN_rate)^nan_penalty × timeframe_weight.
    - rank-std: std of rank-transformed values (robust variability).
    - timeframe_weight: secs^-0.3 for known timeframe suffixes; 1.0 otherwise.
    """
    if X.empty:
        return pd.Series(dtype=float)
    Xr = X.rank(method="average", pct=True).astype("float32")
    var = Xr.std(axis=0).fillna(0.0)
    miss = X.isna().mean().fillna(1.0)
    q = var * (1.0 - miss) ** float(nan_penalty)
    tf_w = pd.Series({c: _tf_pref_weight(c, known_tfs) for c in X.columns}, index=X.columns, dtype="float64")
    q = q * tf_w.fillna(1.0)
    return q.sort_values(ascending=False)


def _order_candidates(
    X: pd.DataFrame,
    candidates: Iterable[str],
    counts: pd.Series,
    known_tfs: Optional[Sequence[str]] = None,
) -> List[str]:
    cols = [c for c in candidates if c in X.columns]
    if not cols:
        return []

    # Precompute tie-break features
    nan_rates = {c: _nan_rate(X[c]) for c in cols}
    rank_stds = {c: _rank_std(X[c]) for c in cols}
    tf_w = {c: _tf_pref_weight(c, known_tfs) for c in cols}

    # Build sort keys: (-count, nan_rate, -rank_std, -tf_weight, name)
    def key(c: str):
        cnt = float(counts.get(c, 0.0))
        return (-cnt, nan_rates[c], -rank_stds[c], -tf_w[c], c)

    cols.sort(key=key)
    return cols


def _spearman_abs(
    s1: pd.Series,
    s2: pd.Series,
    *,
    min_overlap: int = 200,
) -> float:
    r1 = s1.rank(method="average", pct=True)
    r2 = s2.rank(method="average", pct=True)
    a = pd.concat([r1, r2], axis=1, join="inner").dropna()
    if len(a) < int(min_overlap):
        return 0.0
    v = np.corrcoef(a.iloc[:, 0].values, a.iloc[:, 1].values)[0, 1]
    if np.isnan(v):
        return 0.0
    return float(abs(v))


@dataclass
class GreedyParams:
    tau: float = 0.90
    cap_per_family: int = 2
    min_overlap: int = 200
    known_tfs: Optional[Sequence[str]] = None
    exclude_suffixes: Optional[Sequence[str]] = None


def greedy_select(
    X: pd.DataFrame,
    ordered_candidates: Sequence[str],
    params: GreedyParams,
) -> List[str]:
    keep: List[str] = []
    fam_counts: Dict[str, int] = {}

    for f in ordered_candidates:
        if f not in X.columns:
            continue
        if _is_excluded(f, params.exclude_suffixes, params.known_tfs):
            continue
        fam = _family_from_name(f, params.known_tfs)
        cap = params.cap_per_family
        if cap is not None and fam_counts.get(fam, 0) >= int(cap):
            continue

        if not keep:
            keep.append(f)
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
            continue

        s = X[f]
        max_r = 0.0
        for k in keep:
            r = _spearman_abs(s, X[k], min_overlap=params.min_overlap)
            if r > max_r:
                max_r = r
            if max_r >= params.tau:
                break
        if max_r < params.tau:
            keep.append(f)
            fam_counts[fam] = fam_counts.get(fam, 0) + 1

    return keep


def greedy_expand_from_rest(
    X: pd.DataFrame,
    keep: List[str],
    params: GreedyParams,
    *,
    order_known_tfs: Optional[Sequence[str]] = None,
    nan_penalty: float = 1.0,
    target_total: Optional[int] = None,
    restrict_features: Optional[Sequence[str]] = None,
) -> List[str]:
    """Greedily add non-redundant features from the rest of X (beyond the seed set).

    - Orders remaining columns by an unsupervised quality score.
    - Applies the same tau/cap_per_family constraints as for candidates.
    - Stops when no more can be added or when `target_total` is reached.
    """
    keep = [c for c in keep if c in X.columns]
    fam_counts: Dict[str, int] = {}
    for f in keep:
        fam = _family_from_name(f, params.known_tfs)
        fam_counts[fam] = fam_counts.get(fam, 0) + 1

    allow_set = set(restrict_features) if restrict_features else None
    rest_cols = [c for c in X.columns if c not in keep and (allow_set is None or c in allow_set)]
    # Apply exclusion filter to rest
    rest_cols = [c for c in rest_cols if not _is_excluded(c, params.exclude_suffixes, params.known_tfs)]
    if not rest_cols:
        return keep

    score = _quality_score(X[rest_cols], known_tfs=order_known_tfs, nan_penalty=nan_penalty)
    ordered_rest = [c for c in score.index if c in X.columns]

    for f in ordered_rest:
        fam = _family_from_name(f, params.known_tfs)
        cap = params.cap_per_family
        if cap is not None and fam_counts.get(fam, 0) >= int(cap):
            continue
        s = X[f]
        max_r = 0.0
        for k in keep:
            r = _spearman_abs(s, X[k], min_overlap=params.min_overlap)
            if r > max_r:
                max_r = r
            if max_r >= params.tau:
                break
        if max_r < params.tau:
            keep.append(f)
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
            if target_total is not None and len(keep) >= int(target_total):
                break
    return keep


def select_mlflow_first_greedy(
    X: pd.DataFrame,
    *,
    experiment_name: str,
    metric_name: str,
    metric_threshold: float,
    top_n_per_run: int = 100,
    comparator: str = ">=",
    tracking_uri: Optional[str] = None,
    min_occurrence: int = 2,
    topK_overall: Optional[int] = 300,
    params: Optional[GreedyParams] = None,
    order_known_tfs: Optional[Sequence[str]] = None,
    include_rest: bool = False,
    target_total: Optional[int] = None,
    nan_penalty: float = 1.0,
    exclude_suffixes: Optional[Sequence[str]] = ("_all_tf_normalized",),
    restrict_features: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    counts = get_feature_occurrence_series(
        experiment_name=experiment_name,
        metric_name=metric_name,
        metric_threshold=metric_threshold,
        top_n=top_n_per_run,
        comparator=comparator,
        tracking_uri=tracking_uri,
    )

    # Filter candidates by occurrence and intersect with available X columns
    if counts is None or counts.empty:
        return {"selected": [], "counts": pd.Series(dtype=int), "ordered": []}

    mask = counts >= int(min_occurrence)
    filtered = counts[mask]
    # Apply exclusion filter to candidate names
    if exclude_suffixes:
        idx = pd.Index([c for c in filtered.index if not _is_excluded(c, exclude_suffixes, order_known_tfs)])
        filtered = filtered.reindex(idx)
    # Apply restrict list if provided
    if restrict_features is not None:
        allow = set(restrict_features)
        idx = pd.Index([c for c in filtered.index if c in allow])
        filtered = filtered.reindex(idx)
    if topK_overall is not None and len(filtered) > int(topK_overall):
        filtered = filtered.head(int(topK_overall))

    candidates = [c for c in filtered.index if c in X.columns]
    if not candidates:
        return {"selected": [], "counts": filtered, "ordered": []}

    ordered = _order_candidates(X, candidates, filtered, known_tfs=order_known_tfs)
    gp = params or GreedyParams(known_tfs=order_known_tfs)
    gp.exclude_suffixes = exclude_suffixes
    selected = greedy_select(X, ordered, gp)

    if include_rest:
        selected = greedy_expand_from_rest(
            X,
            selected,
            gp,
            order_known_tfs=order_known_tfs,
            nan_penalty=nan_penalty,
            target_total=target_total,
            restrict_features=restrict_features,
        )

    return {"selected": selected, "counts": filtered, "ordered": ordered}


def _cli_main() -> None:  # pragma: no cover
    import argparse
    import json
    import sys

    p = argparse.ArgumentParser(
        description="MLflow-first greedy correlation selector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--features-csv", required=True, help="Path to features CSV (columns=features)")
    p.add_argument("--experiment", required=True)
    p.add_argument("--metric", required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--top-n-per-run", type=int, default=100)
    p.add_argument("--comparator", choices=[">=", ">", "<=", "<", "=", "=="], default=">=")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--min-occurrence", type=int, default=2)
    p.add_argument("--topK-overall", type=int, default=300)
    p.add_argument("--tau", type=float, default=0.90)
    p.add_argument("--cap-per-family", type=int, default=2)
    p.add_argument("--min-overlap", type=int, default=200)
    p.add_argument("--known-tfs", nargs="*", default=["1H", "4H", "12H", "1D"])
    p.add_argument("--include-rest", action="store_true", help="After seeding with candidates, greedily add from all X")
    p.add_argument("--target-total", type=int, default=None, help="Optional total feature target when including rest")
    p.add_argument("--nan-penalty", type=float, default=1.0, help="Exponent for (1-NaN_rate) in quality score")
    p.add_argument("--output", default=None, help="Path to write selected features JSON list")
    p.add_argument("--limit", type=int, default=None, help="Print only first N selected")

    args = p.parse_args()

    try:
        X = pd.read_csv(args.features_csv)
    except Exception as e:
        print(f"Failed to read features CSV: {e}", file=sys.stderr)
        sys.exit(2)

    # Drop obvious non-feature columns
    drop_cols = [c for c in ("timestamp",) if c in X.columns]
    X = X.drop(columns=drop_cols)

    res = select_mlflow_first_greedy(
        X,
        experiment_name=args.experiment,
        metric_name=args.metric,
        metric_threshold=args.threshold,
        top_n_per_run=args.top_n_per_run,
        comparator=args.comparator,
        tracking_uri=args.tracking_uri,
        min_occurrence=args.min_occurrence,
        topK_overall=args.topK_overall,
        params=GreedyParams(
            tau=args.tau,
            cap_per_family=args.cap_per_family,
            min_overlap=args.min_overlap,
            known_tfs=args.known_tfs,
        ),
        order_known_tfs=args.known_tfs,
        include_rest=bool(args.include_rest),
        target_total=args.target_total,
        nan_penalty=args.nan_penalty,
    )

    selected = res.get("selected", [])
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(selected, f, indent=2)

    n = len(selected)
    print(f"Selected features: {n}")
    if n:
        lim = n if args.limit is None else min(n, int(args.limit))
        for i in range(lim):
            print(selected[i])


if __name__ == "__main__":  # pragma: no cover
    _cli_main()

#!/usr/bin/env python3
"""
HMM training pipeline runner (lean, config-driven), similar style to run_lgbm_pipeline.py.

Usage:
  python model/run_hmm_pipeline.py --config configs/model_configs/hmm_v1_1h.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def setup_logging(log_level: str = "INFO", logfile: Path | None = None) -> None:
    """Initialize logging to stdout and optional logfile."""
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(logfile)))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def load_config(path: Path) -> Dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, 'r') as f:
        cfg = json.load(f)

    for key in ['input_data', 'output_dir', 'split', 'model']:
        if key not in cfg:
            raise KeyError(f"Missing config key: {key}")

    cfg['input_data'] = Path(cfg['input_data'])
    cfg['output_dir'] = Path(cfg['output_dir'])

    # Resolve feature selection (unified with LGBM pipeline)
    feature_cfg = cfg.get('feature_selection') or cfg.get('features')  # Support both for backward compatibility
    include: List[str] = []
    exclude: Optional[List[str]] = None
    include_patterns: List[str] = []

    if isinstance(feature_cfg, dict):
        for list_path in feature_cfg.get('include_files', []) or []:
            p = Path(list_path)
            if not p.is_absolute():
                p = Path(__file__).resolve().parent.parent / p
            with open(p, 'r') as f:
                include.extend(json.load(f))
        include.extend(feature_cfg.get('include', []) or [])
        include_patterns = feature_cfg.get('include_patterns', []) or []
        exclude = feature_cfg.get('exclude')
    elif isinstance(feature_cfg, str):
        p = Path(feature_cfg)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / p
        with open(p, 'r') as f:
            include.extend(json.load(f))
    elif feature_cfg is None:
        default_feature_list = Path(__file__).resolve().parent.parent / 'configs' / 'feature_lists' / 'binance_btcusdt_p60_hmm_1h.json'
        with open(default_feature_list, 'r') as f:
            include.extend(json.load(f))
    else:
        raise ValueError("feature_selection/features must be dict or path to JSON list")

    cfg['features'] = {
        'include': include,
        'include_patterns': include_patterns,
        'exclude': exclude,
    }

    split = cfg['split']
    split.setdefault('train_ratio', 0.7)
    split.setdefault('val_ratio', 0.15)
    split.setdefault('test_ratio', 0.15)
    model = cfg['model']
    model.setdefault('covariance_type', 'diag')
    model.setdefault('n_iter', 200)
    model.setdefault('tol', 1e-3)
    model.setdefault('random_state', 42)
    # Optional numeric stability and sticky init
    model.setdefault('reg_covar', None)
    model.setdefault('sticky_diag', None)
    # Normalize state_grid from min/max range if provided
    sg = model.get('state_grid')
    if isinstance(sg, dict):
        kmin = int(sg.get('min'))
        kmax = int(sg.get('max'))
        if kmin < 1 or kmax < kmin:
            raise ValueError(f"Invalid state_grid range: {sg}")
        model['state_grid'] = list(range(kmin, kmax + 1))

    # Selection defaults (HMM-appropriate)
    sel = cfg.get('selection') or {}
    sel.setdefault('criterion', 'bic')  # 'icl' preferred for stricter selection; default 'bic' for backward comp
    sel.setdefault('restarts', 1)
    sel.setdefault('delta_threshold', None)  # e.g., 10.0 for elbow rule
    sel.setdefault('min_state_occupancy_pct', 0.0)  # e.g., 0.02 to require >=2% per state
    sel.setdefault('cv_folds', 0)  # blocked CV folds on train (>=2 to enable)
    sel.setdefault('one_std_rule', False)  # if cv enabled, pick smallest K within 1 std err of best
    cfg['selection'] = sel
    # Ensure model_type is recorded in artifacts for registrar consumption
    try:
        if isinstance(cfg.get('model'), dict):
            cfg['model']['type'] = 'hmm'
    except Exception:
        pass
    return cfg


def select_feature_columns(df_cols: List[str], cfg: Dict[str, Any]) -> List[str]:
    """Select feature columns using unified feature selection logic (same as LGBM pipeline)."""
    import fnmatch
    
    feats = cfg['features']
    include = feats.get('include', [])
    include_patterns = feats.get('include_patterns', [])
    exclude = feats.get('exclude', [])
    
    if not include and not include_patterns:
        raise ValueError("features.include or features.include_patterns must provide a non-empty list of columns/patterns")
    
    # Apply include filters
    filtered_cols: List[str] = []
    if include:
        matches: set[str] = set()
        for pattern in include:
            pattern_matches = fnmatch.filter(df_cols, pattern)
            matches.update(pattern_matches)
        missing = [p for p in include if not fnmatch.filter(df_cols, p)]
        if missing:
            raise KeyError(f"Included feature columns not found for patterns: {missing}")
        filtered_cols = [c for c in df_cols if c in matches]
    else:
        filtered_cols = list(df_cols)

    # Apply include_patterns (additive)
    if include_patterns:
        pattern_matches: set[str] = set()
        for pattern in include_patterns:
            pattern_matches.update(fnmatch.filter(df_cols, pattern))
        filtered_cols.extend([c for c in pattern_matches if c not in filtered_cols])

    # Apply exclude filters
    if exclude:
        exclude_matches: set[str] = set()
        for pattern in exclude:
            exclude_matches.update(fnmatch.filter(filtered_cols, pattern))
        filtered_cols = [c for c in filtered_cols if c not in exclude_matches]
    
    if not filtered_cols:
        raise ValueError("No feature columns remain after applying feature selection filters")
    
    return filtered_cols


def load_features(path: Path, selected_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns:
        raise ValueError("Features CSV must include 'timestamp'")
    ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    keep = ['timestamp'] + [c for c in selected_cols if c in df.columns]
    selected_df = df.loc[:, keep].sort_values('timestamp').reset_index(drop=True)
    rows_after_selection = len(selected_df)
    metadata = {
        'input_path': str(path),
        'rows_total': int(len(df)),
        'rows_after_selection': int(rows_after_selection),
        'selected_feature_columns': [c for c in selected_cols if c in selected_df.columns],
        'missing_feature_columns': [c for c in selected_cols if c not in selected_df.columns],
    }
    return selected_df, metadata


def time_split_indices(ts: pd.Series, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("ratios must sum to 1.0")
    n = len(ts)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))
    train_idx = np.arange(0, i1)
    val_idx = np.arange(i1, i2)
    test_idx = np.arange(i2, n)
    return train_idx, val_idx, test_idx


def _normalize_to_utc_naive_strings(values: List[Any]) -> set[str]:
    """Normalize an iterable of timestamp-like values to UTC-naive 'YYYY-MM-DD HH:MM:SS' strings.

    Accepts ISO strings with/without timezone, epoch ints/floats, or pandas Timestamps.
    Returns a set of canonicalized strings for robust membership comparison.
    """
    try:
        s = pd.to_datetime(pd.Series(list(values)), utc=True, errors='coerce')
        # Ensure UTC then drop tz to naive
        s = s.dt.tz_convert('UTC').dt.tz_localize(None)
        return set(s.astype(str).tolist())
    except Exception:
        return set()


def load_split_indices_from_meta(ts: pd.Series, meta_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load split indices from a LightGBM prepare_splits prep_metadata.json file.

    Robustly matches by normalizing both sides to UTC-naive timestamp strings.
    """
    meta = json.load(open(meta_path, 'r'))
    split_ts = meta.get('split_timestamps', {}) or {}
    train_set = _normalize_to_utc_naive_strings(split_ts.get('train', []) or [])
    val_set = _normalize_to_utc_naive_strings(split_ts.get('val', []) or [])
    test_set = _normalize_to_utc_naive_strings(split_ts.get('test', []) or [])
    ts_norm = pd.to_datetime(ts, errors='coerce', utc=True)
    ts_norm = ts_norm.dt.tz_convert('UTC').dt.tz_localize(None)
    ts_str = ts_norm.astype(str).tolist()
    train_idx = np.array([i for i, s in enumerate(ts_str) if s in train_set], dtype=int)
    val_idx = np.array([i for i, s in enumerate(ts_str) if s in val_set], dtype=int)
    test_idx = np.array([i for i, s in enumerate(ts_str) if s in test_set], dtype=int)
    return train_idx, val_idx, test_idx


def _debug_log_timestamp_mismatch(ts: pd.Series, meta_path: Path) -> None:
    """Log helpful diagnostics when split timestamp matching yields zero indices."""
    try:
        logging.error('Timestamp matching yielded 0/0/0. Dumping diagnostics:')
        # Features side
        ts_norm = pd.to_datetime(ts, errors='coerce', utc=True)
        ts_norm = ts_norm.dt.tz_convert('UTC').dt.tz_localize(None)
        features_count = int(len(ts_norm))
        features_min = str(ts_norm.min()) if features_count else 'NA'
        features_max = str(ts_norm.max()) if features_count else 'NA'
        features_head = [str(x) for x in ts_norm.head(3).tolist()]
        features_tail = [str(x) for x in ts_norm.tail(3).tolist()]
        try:
            freq_top = ts_norm.diff().dropna().value_counts().head(3).to_dict()
            freq_top = {str(k): int(v) for k, v in freq_top.items()}
        except Exception:
            freq_top = {}
        logging.error('features: count=%d min=%s max=%s head=%s tail=%s freq_top=%s',
                      features_count, features_min, features_max, features_head, features_tail, freq_top)

        # Metadata side
        meta = json.load(open(meta_path, 'r'))
        split_ts = meta.get('split_timestamps', {}) or {}
        train_set = _normalize_to_utc_naive_strings(split_ts.get('train', []) or [])
        val_set = _normalize_to_utc_naive_strings(split_ts.get('val', []) or [])
        test_set = _normalize_to_utc_naive_strings(split_ts.get('test', []) or [])

        def _set_stats(name: str, s: set[str]) -> None:
            if not s:
                logging.error('%s: count=0', name)
                return
            lst = sorted(list(s))
            series = pd.to_datetime(pd.Series(lst), errors='coerce')
            vmin = str(series.min())
            vmax = str(series.max())
            head = lst[:3]
            tail = lst[-3:]
            logging.error('%s: count=%d min=%s max=%s head=%s tail=%s', name, len(s), vmin, vmax, head, tail)

        _set_stats('meta.train', train_set)
        _set_stats('meta.val', val_set)
        _set_stats('meta.test', test_set)
    except Exception as e:
        logging.error('Failed to dump timestamp mismatch diagnostics: %s', e)


def hmm_param_count(n_states: int, n_features: int, covariance_type: str = 'diag') -> int:
    startprob = n_states - 1
    transmat = n_states * (n_states - 1)
    means = n_states * n_features
    if covariance_type == 'full':
        cov = n_states * (n_features * (n_features + 1) // 2)
    else:
        cov = n_states * n_features
    return startprob + transmat + means + cov


def _sticky_transmat(n_states: int, diag_weight: float) -> np.ndarray:
    # Create a transition matrix with high self-transition probability
    diag_weight = float(min(max(diag_weight, 0.0), 0.999))
    off = (1.0 - diag_weight)
    if n_states > 1:
        off_each = off / (n_states - 1)
    else:
        off_each = 0.0
    T = np.full((n_states, n_states), off_each, dtype=float)
    np.fill_diagonal(T, diag_weight)
    return T


def fit_and_score(
    X: np.ndarray,
    n_states: int,
    covariance_type: str,
    n_iter: int,
    tol: float,
    random_state: int,
    *,
    reg_covar: float | None = None,
    sticky_diag: float | None = None,
) -> Tuple[GaussianHMM, float, float, float, float, float, List[float]]:
    # Configure model
    kwargs: Dict[str, Any] = dict(
        n_components=int(n_states),
        covariance_type=covariance_type,
        n_iter=int(n_iter),
        tol=float(tol),
        random_state=int(random_state),
    )
    if reg_covar is not None:
        # hmmlearn uses min_covar
        kwargs['min_covar'] = float(reg_covar)
    # If we want to preserve custom transmat_, exclude 't' from init_params
    init_params = 'stmc'
    if sticky_diag is not None:
        init_params = 'smc'
    model = GaussianHMM(init_params=init_params, **kwargs)
    if sticky_diag is not None:
        model.transmat_ = _sticky_transmat(int(n_states), float(sticky_diag))
    # Fit
    model.fit(X)
    # Train log-likelihood
    ll = float(model.score(X))
    # Information criteria
    p = hmm_param_count(int(n_states), X.shape[1], covariance_type)
    n = X.shape[0]
    bic = -2.0 * ll + p * np.log(n)
    aic = -2.0 * ll + 2.0 * p
    # Classification entropy and ICL
    gamma = model.predict_proba(X)
    eps = 1e-12
    H = float(-np.sum(gamma * np.log(np.clip(gamma, eps, 1.0))))
    icl = bic - 2.0 * H
    # Expected state occupancy (fraction)
    occ = list(np.sum(gamma, axis=0) / float(n))
    return model, ll, bic, aic, icl, H, occ


def select_n_states(
    X_train: np.ndarray,
    X_val: np.ndarray,
    model_cfg: Dict[str, Any],
    sel_cfg: Dict[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    # If CV is enabled, use blocked CV over the train window to choose K
    cv_folds = int(sel_cfg.get('cv_folds') or 0)
    one_se = bool(sel_cfg.get('one_std_rule') or False)
    grid = model_cfg.get('state_grid') or [model_cfg.get('n_states', 3)]
    cov = model_cfg.get('covariance_type', 'diag')
    n_iter = model_cfg.get('n_iter', 200)
    tol = model_cfg.get('tol', 1e-3)
    rs = model_cfg.get('random_state', 42)
    reg_covar = model_cfg.get('reg_covar')
    sticky_diag = model_cfg.get('sticky_diag')

    restarts = int(sel_cfg.get('restarts', 1))
    crit_name = str(sel_cfg.get('criterion', 'bic')).lower()
    delta_thr = sel_cfg.get('delta_threshold')
    min_occ = float(sel_cfg.get('min_state_occupancy_pct') or 0.0)

    if cv_folds and cv_folds > 1 and X_train.shape[0] > cv_folds:
        N = X_train.shape[0]
        fold_size = N // cv_folds
        folds = []
        for i in range(cv_folds):
            start = i * fold_size
            end = (i + 1) * fold_size if i < cv_folds - 1 else N
            val_idx = np.arange(start, end)
            train_idx = np.concatenate([np.arange(0, start), np.arange(end, N)]) if start > 0 else np.arange(end, N)
            folds.append((train_idx, val_idx))

        per_k_cv: List[Dict[str, Any]] = []
        for k in grid:
            fold_lls: List[float] = []
            fold_recs: List[Dict[str, Any]] = []
            for (tr_idx, va_idx) in folds:
                Xtr = X_train[tr_idx]
                Xva = X_train[va_idx]
                best_for_fold = None
                for r in range(max(1, restarts)):
                    seed = int(rs) + r
                    m, ll_tr, bic, aic, icl, H, occ = fit_and_score(
                        Xtr, k, cov, n_iter, tol, seed, reg_covar=reg_covar, sticky_diag=sticky_diag,
                    )
                    if min_occ > 0.0 and any(o < min_occ for o in occ):
                        continue
                    ll_val = float(m.score(Xva)) if len(Xva) else float('nan')
                    rec = {'n_states': int(k), 'train_ll': ll_tr, 'val_ll': ll_val, 'bic': bic, 'aic': aic, 'icl': icl, 'entropy': H, 'occupancy': occ, 'seed': seed}
                    # Select restart by validation likelihood (higher is better); tie-break by train LL
                    if best_for_fold is None:
                        best_for_fold = rec
                    else:
                        curr_v = rec['val_ll']
                        best_v = best_for_fold['val_ll']
                        curr_v = curr_v if np.isfinite(curr_v) else -np.inf
                        best_v = best_v if np.isfinite(best_v) else -np.inf
                        if (curr_v > best_v) or (curr_v == best_v and rec['train_ll'] > best_for_fold['train_ll']):
                            best_for_fold = rec
                if best_for_fold is not None:
                    fold_lls.append(best_for_fold['val_ll'])
                    fold_recs.append(best_for_fold)
            if fold_lls:
                mean_ll = float(np.mean(fold_lls))
                std_ll = float(np.std(fold_lls, ddof=1)) if len(fold_lls) > 1 else 0.0
                per_k_cv.append({'n_states': int(k), 'cv_mean_ll': mean_ll, 'cv_std_ll': std_ll, 'folds': len(fold_lls)})

        if per_k_cv:
            # Select by max cv_mean_ll; apply one-std-error rule if enabled
            best = max(per_k_cv, key=lambda r: r['cv_mean_ll'])
            if one_se:
                se = best['cv_std_ll'] / np.sqrt(best['folds']) if best['folds'] > 0 else 0.0
                threshold = best['cv_mean_ll'] - se
                # choose smallest K with mean >= threshold
                candidates = sorted([r for r in per_k_cv if r['cv_mean_ll'] >= threshold], key=lambda r: int(r['n_states']))
                if candidates:
                    chosen_cv = candidates[0]
                else:
                    chosen_cv = best
            else:
                chosen_cv = best
            return int(chosen_cv['n_states']), {'grid': per_k_cv, 'chosen': chosen_cv, 'criterion': 'cv_ll', 'cv_folds': cv_folds, 'one_std_rule': one_se, 'restarts': restarts}

    per_k: List[Dict[str, Any]] = []
    for k in grid:
        best_for_k = None
        for r in range(max(1, restarts)):
            seed = int(rs) + r
            m, ll_tr, bic, aic, icl, H, occ = fit_and_score(
                X_train, k, cov, n_iter, tol, seed, reg_covar=reg_covar, sticky_diag=sticky_diag,
            )
            # Reject by occupancy threshold
            if min_occ > 0.0 and any(o < min_occ for o in occ):
                continue
            ll_val = float(m.score(X_val)) if len(X_val) > 0 else float('nan')
            rec = {
                'n_states': int(k), 'train_ll': ll_tr, 'val_ll': ll_val,
                'bic': bic, 'aic': aic, 'icl': icl, 'entropy': H, 'occupancy': occ,
                'seed': seed,
            }
            # Keep best per K by primary criterion then by train_ll
            def _crit(r):
                return r.get(crit_name, np.inf)
            if (best_for_k is None) or (_crit(rec) < _crit(best_for_k)) or (_crit(rec) == _crit(best_for_k) and rec['train_ll'] > best_for_k['train_ll']):
                best_for_k = rec
        if best_for_k is not None:
            per_k.append(best_for_k)

    if not per_k:
        # Fallback: try without occupancy filter
        for k in grid:
            m, ll_tr, bic, aic, icl, H, occ = fit_and_score(X_train, k, cov, n_iter, tol, rs, reg_covar=reg_covar, sticky_diag=sticky_diag)
            ll_val = float(m.score(X_val)) if len(X_val) > 0 else float('nan')
            per_k.append({'n_states': int(k), 'train_ll': ll_tr, 'val_ll': ll_val, 'bic': bic, 'aic': aic, 'icl': icl, 'entropy': H, 'occupancy': occ, 'seed': rs})

    # Apply delta-threshold elbow if configured
    chosen = None
    if delta_thr is not None and len(per_k) >= 2:
        per_k_sorted = sorted(per_k, key=lambda r: int(r['n_states']))
        crit_vals = [r.get(crit_name, np.inf) for r in per_k_sorted]
        ks = [int(r['n_states']) for r in per_k_sorted]
        # improvements: previous - current (positive means improvement)
        improvements = [np.inf]
        for i in range(1, len(crit_vals)):
            improvements.append(crit_vals[i-1] - crit_vals[i])
        elbow_idx = None
        for i in range(1, len(improvements)):
            if improvements[i] < float(delta_thr):
                elbow_idx = i - 1
                break
        if elbow_idx is not None and elbow_idx >= 0:
            chosen_k = ks[elbow_idx]
            chosen = next(r for r in per_k_sorted if int(r['n_states']) == chosen_k)

    # If not chosen by elbow, choose by primary criterion then by train_ll
    if chosen is None:
        chosen = sorted(per_k, key=lambda r: (r.get(crit_name, np.inf), -r['train_ll']))[0]

    return int(chosen['n_states']), {'grid': per_k, 'chosen': chosen, 'criterion': crit_name, 'delta_threshold': delta_thr, 'min_state_occupancy_pct': min_occ, 'restarts': restarts}


def evaluate(model: GaussianHMM, X: np.ndarray, ts: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    states = model.predict(X)
    post = model.predict_proba(X)
    out = pd.DataFrame({'timestamp': pd.to_datetime(ts), 'state': states})
    for k in range(post.shape[1]):
        out[f'p_state_{k}'] = post[:, k]
    # Minimal diagnostics
    diag = {
        'n_states': int(model.n_components),
        'covariance_type': str(model.covariance_type),
        'transmat': model.transmat_.tolist(),
        'startprob': model.startprob_.tolist(),
        'means': model.means_.tolist(),
    }
    return out, diag


def save_artifacts(out_dir: Path, model: GaussianHMM, scaler: StandardScaler, config: Dict[str, Any], metrics: Dict[str, Any], regimes_df: pd.DataFrame, diagnostics: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / 'model.joblib')
    joblib.dump(scaler, out_dir / 'scaler.joblib')
    def _json_safe(x):
        from pathlib import Path as _P
        if isinstance(x, _P):
            return str(x)
        if isinstance(x, dict):
            return {k: _json_safe(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_json_safe(v) for v in x]
        return x
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(_json_safe(config), f, indent=2)
    # Also emit pipeline_config.json for alignment with registrar expectations
    try:
        with open(out_dir / 'pipeline_config.json', 'w') as f:
            json.dump(_json_safe(config), f, indent=2)
    except Exception:
        pass
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    regimes_df.to_csv(out_dir / 'regimes.csv', index=False)
    with open(out_dir / 'diagnostics.json', 'w') as f:
        json.dump(diagnostics, f, indent=2)


def run_hmm_pipeline(cfg: Dict[str, Any], log_level: str = "INFO") -> Path:
    # First-stage logging to stdout only; will add file handler once run_dir is known
    setup_logging(log_level)

    # Load and select columns
    tmp_df = pd.read_csv(cfg['input_data'], nrows=1)
    feature_cols = select_feature_columns(list(tmp_df.columns), cfg)
    raw_df, feature_meta = load_features(cfg['input_data'], feature_cols)
    rows_with_na_total = int(raw_df[feature_cols].isna().any(axis=1).sum())
    feature_meta['rows_with_na_total'] = rows_with_na_total
    logging.info(
        'Loaded features: rows=%d cols=%d (rows_with_na=%d)',
        len(raw_df),
        len(feature_cols),
        rows_with_na_total,
    )

    # Prepare run directory and attach file logging (standardized name)
    ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = cfg['output_dir'] / f"run_hmm_{ts_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Reconfigure logging to also write to file
    file_handler = logging.FileHandler(str(run_dir / 'hmm_pipeline.log'))
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.info('Run directory: %s', run_dir)

    # Split: prefer prepared metadata (LightGBM) if provided
    split_cfg = cfg.get('split', {}) or {}
    meta_path = None
    existing_dir = split_cfg.get('existing_dir')
    if existing_dir:
        meta_path = Path(existing_dir) / 'prep_metadata.json'
    meta_path = split_cfg.get('prep_metadata') or meta_path

    if meta_path:
        meta_path = Path(meta_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"prep_metadata.json not found at: {meta_path}")
        tr_raw, va_raw, te_raw = load_split_indices_from_meta(raw_df['timestamp'], meta_path)
        if (len(tr_raw) == 0) and (len(va_raw) == 0) and (len(te_raw) == 0):
            _debug_log_timestamp_mismatch(raw_df['timestamp'], meta_path)
            raise ValueError(
                f"No matching timestamps found between features and prep_metadata: {meta_path}. "
                f"Ensure both are normalized to UTC and share the same sampling schedule."
            )
        logging.info(
            'Loaded split indices from %s | sizes train/val/test: %d/%d/%d',
            meta_path,
            len(tr_raw),
            len(va_raw),
            len(te_raw),
        )
    else:
        tr_raw, va_raw, te_raw = time_split_indices(
            raw_df['timestamp'],
            float(split_cfg.get('train_ratio', 0.7)),
            float(split_cfg.get('val_ratio', 0.15)),
            float(split_cfg.get('test_ratio', 0.15)),
        )

    combined_indices_raw = [arr for arr in (tr_raw, va_raw, te_raw) if len(arr)]
    union_raw = np.unique(np.concatenate(combined_indices_raw)) if combined_indices_raw else np.array([], dtype=int)
    if union_raw.size == 0:
        raise ValueError('Split configuration yielded zero rows across train/val/test')

    trimmed_df = raw_df.iloc[union_raw].reset_index(drop=True)
    trimmed_outside = int(len(raw_df) - len(trimmed_df))
    if trimmed_outside > 0:
        logging.info('Trimmed %d rows outside split ranges prior to modeling', trimmed_outside)

    na_mask = trimmed_df[feature_cols].isna().any(axis=1)
    if na_mask.any():
        bad_ts = trimmed_df.loc[na_mask, 'timestamp'].astype(str).head(5).tolist()
        raise ValueError(f'Encountered NaNs within split-aligned data; example timestamps: {bad_ts}')

    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(union_raw)}

    def _map_indices(indices: np.ndarray) -> np.ndarray:
        if len(indices) == 0:
            return np.zeros(0, dtype=int)
        return np.array([index_map[i] for i in indices], dtype=int)

    tr = _map_indices(tr_raw)
    va = _map_indices(va_raw)
    te = _map_indices(te_raw)

    def _assert_split_no_nan(name: str, indices: np.ndarray) -> None:
        if len(indices) == 0:
            return
        subset = trimmed_df.iloc[indices][feature_cols]
        if subset.isna().any().any():
            bad_mask = subset.isna().any(axis=1)
            bad_ts = trimmed_df.iloc[indices].loc[bad_mask, 'timestamp'].astype(str).head(5).tolist()
            raise ValueError(f'{name} split contains NaNs after trimming; example timestamps: {bad_ts}')

    _assert_split_no_nan('train', tr)
    _assert_split_no_nan('val', va)
    _assert_split_no_nan('test', te)

    final_rows = int(len(trimmed_df))
    logging.info(
        'Row tally | original=%d trimmed_to_splits=%d train=%d val=%d test=%d final=%d',
        len(raw_df),
        final_rows,
        len(tr),
        len(va),
        len(te),
        final_rows,
    )

    feature_meta['rows_trimmed_outside_splits'] = trimmed_outside
    feature_meta['rows_after_trim_to_splits'] = final_rows
    feature_meta['rows_by_split'] = {'train': int(len(tr)), 'val': int(len(va)), 'test': int(len(te))}
    feature_meta['final_model_rows'] = final_rows

    df = trimmed_df
    ts_all = df['timestamp']
    X_train = df.iloc[tr][feature_cols].values
    X_val = df.iloc[va][feature_cols].values
    X_test = df.iloc[te][feature_cols].values

    # Scale: fit on train only to prevent leakage
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if len(X_val) else X_val
    X_test_s = scaler.transform(X_test) if len(X_test) else X_test

    # Select n_states
    n_states_cfg = cfg['model'].get('n_states')
    if n_states_cfg is None:
        # If using prepared splits, default to train-only selection by passing empty val unless selection one-std is enabled
        use_val = False if meta_path else True
        X_val_for_sel = X_val_s if (use_val and len(X_val_s)) else np.zeros((0, X_train_s.shape[1]))
        n_best, sel = select_n_states(X_train_s, X_val_for_sel, cfg['model'], cfg.get('selection', {}))
        cfg['model']['n_states'] = int(n_best)
        logging.info('Selected n_states=%d using %s over grid (restarts=%s, delta_thr=%s, min_occ=%.4f)', n_best, sel.get('criterion'), sel.get('restarts'), sel.get('delta_threshold'), sel.get('min_state_occupancy_pct') or 0.0)
        # Persist selection grid for diagnosis
        try:
            import pandas as _pd
            import numpy as _np
            grid = sel.get('grid', []) or []
            if grid:
                grid_df = _pd.DataFrame(grid)
                crit = sel.get('criterion')
                if crit == 'cv_ll':
                    # Higher is better
                    grid_df['criterion_value'] = grid_df.get('cv_mean_ll', _pd.Series([np.nan]*len(grid_df)))
                    grid_df = grid_df.sort_values('n_states')
                    grid_df['delta'] = grid_df['criterion_value'].diff()
                else:
                    # Lower is better (e.g., bic, icl)
                    if crit in grid_df.columns:
                        grid_df['criterion_value'] = grid_df[crit]
                    else:
                        grid_df['criterion_value'] = np.nan
                    grid_df = grid_df.sort_values('n_states')
                    # improvement = prev - current (positive is better)
                    grid_df['delta'] = grid_df['criterion_value'].shift(1) - grid_df['criterion_value']
                # Expand occupancy list if present
                if 'occupancy' in grid_df.columns:
                    grid_df['occupancy_min'] = grid_df['occupancy'].apply(lambda x: float(min(x)) if isinstance(x, (list, tuple)) and len(x) else np.nan)
                grid_df.to_csv(run_dir / 'selection_grid.csv', index=False)
                logging.info('Saved selection grid to %s', run_dir / 'selection_grid.csv')
        except Exception as e:
            logging.warning('Failed to persist selection grid: %s', e)
    else:
        sel = {'grid': [{'n_states': int(n_states_cfg)}], 'chosen': {'n_states': int(n_states_cfg)}}

    # Final fit set: default train+val if val exists, else train; can be overridden
    final_fit_on = (cfg.get('final_fit') or {}).get('on') or cfg['model'].get('final_fit_on')
    if final_fit_on is None:
        final_fit_on = 'train_val' if len(X_val_s) else 'train'
    if str(final_fit_on).lower() == 'train':
        X_final = X_train_s
    elif str(final_fit_on).lower() in ('train_val', 'train+val'):
        X_final = np.vstack([X_train_s, X_val_s]) if len(X_val_s) else X_train_s
    else:
        # Fallback to train_val for unknown values
        X_final = np.vstack([X_train_s, X_val_s]) if len(X_val_s) else X_train_s
    logging.info('Final fit on: %s (rows=%d)', final_fit_on, X_final.shape[0])
    reg_covar = cfg['model'].get('reg_covar')
    sticky_diag = cfg['model'].get('sticky_diag')
    model, ll_train_final, bic_final, aic_final, icl_final, H_final, occ_final = fit_and_score(
        X_final,
        cfg['model']['n_states'],
        cfg['model']['covariance_type'],
        cfg['model']['n_iter'],
        cfg['model']['tol'],
        cfg['model']['random_state'],
        reg_covar=reg_covar,
        sticky_diag=sticky_diag,
    )
    ll_test = float(model.score(X_test_s)) if len(X_test_s) else float('nan')

    # Evaluate on full series for regimes (apply-only on val/test to avoid leakage)
    X_all_s = scaler.transform(df[feature_cols].values)
    regimes_df, diagnostics = evaluate(model, X_all_s, ts_all)

    # Persist prep metadata describing the split and feature processing
    def _sorted_indices(indices: np.ndarray) -> np.ndarray:
        if isinstance(indices, np.ndarray):
            return np.sort(indices.astype(int))
        return np.sort(np.asarray(indices, dtype=int))

    def _ts_list(indices: np.ndarray) -> List[str]:
        if len(indices) == 0:
            return []
        subset = df.iloc[_sorted_indices(indices)]
        return subset['timestamp'].astype(str).tolist()

    def _ts_range(indices: np.ndarray) -> Dict[str, Any]:
        if len(indices) == 0:
            return {'min': None, 'max': None}
        subset = df.iloc[_sorted_indices(indices)]
        series = subset['timestamp']
        return {'min': str(series.min()), 'max': str(series.max())}

    split_timestamps = {
        'train': _ts_list(tr),
        'val': _ts_list(va),
        'test': _ts_list(te),
    }
    split_ranges = {
        'train': _ts_range(tr),
        'val': _ts_range(va),
        'test': _ts_range(te),
    }
    split_counts = {
        'train': int(len(tr)),
        'val': int(len(va)),
        'test': int(len(te)),
    }
    assigned_idx = set(np.concatenate([_sorted_indices(tr), _sorted_indices(va), _sorted_indices(te)]) if len(df) else [])
    all_idx = set(range(len(df)))
    unknown_idx = sorted(all_idx - assigned_idx)
    if unknown_idx:
        unknown_array = np.asarray(unknown_idx, dtype=int)
        split_timestamps['unknown'] = _ts_list(unknown_array)
        split_ranges['unknown'] = _ts_range(unknown_array)
        split_counts['unknown'] = int(len(unknown_idx))

    prep_metadata = {
        'pipeline': 'run_hmm_pipeline',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_data': str(cfg['input_data']),
        'feature_summary': {
            'selected_feature_columns': feature_meta.get('selected_feature_columns', feature_cols),
            'missing_feature_columns': feature_meta.get('missing_feature_columns', []),
            'n_features': len(feature_cols),
        },
        'row_counts': {
            'rows_total': feature_meta.get('rows_total', len(raw_df)),
            'rows_after_selection': feature_meta.get('rows_after_selection', len(raw_df)),
            'rows_with_na_total': feature_meta.get('rows_with_na_total', 0),
            'rows_trimmed_outside_splits': feature_meta.get('rows_trimmed_outside_splits', 0),
            'rows_after_trim_to_splits': feature_meta.get('rows_after_trim_to_splits', final_rows),
            'rows_by_split': feature_meta.get('rows_by_split', {'train': len(tr), 'val': len(va), 'test': len(te)}),
            'final_model_rows': feature_meta.get('final_model_rows', final_rows),
        },
        'split': {
            'source': 'prep_metadata' if meta_path else 'ratio_time_order',
            'source_path': str(meta_path) if meta_path else None,
            'counts': split_counts,
            'timestamp_ranges': split_ranges,
            'timestamps': split_timestamps,
            'rows_accounted_for': int(
                sum(split_counts.get(k, 0) for k in ('train', 'val', 'test'))
            ),
            'final_fit_on': str(final_fit_on),
        },
    }
    prep_meta_path = run_dir / 'prep_metadata.json'
    with open(prep_meta_path, 'w') as f:
        json.dump(prep_metadata, f, indent=2)
    logging.info('Saved prep metadata to %s', prep_meta_path)

    split_cfg_meta = cfg.get('split', {}) or {}
    split_cfg_meta['source'] = 'prep_metadata' if meta_path else 'ratio_time_order'
    if meta_path:
        split_cfg_meta['prep_metadata'] = str(meta_path)
    split_cfg_meta['hmm_prep_metadata'] = str(prep_meta_path)
    split_cfg_meta['counts'] = {k: int(v) for k, v in split_counts.items()}
    split_cfg_meta['timestamp_ranges'] = split_ranges
    split_cfg_meta['timestamps'] = split_timestamps
    split_cfg_meta['rows_total'] = int(feature_meta.get('rows_total', len(raw_df)))
    split_cfg_meta['rows_with_na_total'] = int(feature_meta.get('rows_with_na_total', 0))
    split_cfg_meta['rows_trimmed_outside_splits'] = int(feature_meta.get('rows_trimmed_outside_splits', 0))
    split_cfg_meta['rows_after_trim_to_splits'] = int(feature_meta.get('rows_after_trim_to_splits', final_rows))
    split_cfg_meta['rows_by_split'] = {k: int(v) for k, v in (feature_meta.get('rows_by_split') or {'train': len(tr), 'val': len(va), 'test': len(te)}).items()}
    split_cfg_meta['final_model_rows'] = int(feature_meta.get('final_model_rows', final_rows))
    cfg['split'] = split_cfg_meta

    # Metrics
    metrics = {
        'n_states': int(cfg['model']['n_states']),
        'bic_final': bic_final,
        'aic_final': aic_final,
        'icl_final': icl_final,
        'entropy_final': H_final,
        'train_final_ll': ll_train_final,
        'test_ll': ll_test,
        'selection': sel,
        'rows': len(df),
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'final_occupancy': occ_final,
        'split_counts': {k: int(v) for k, v in split_counts.items()},
        'rows_with_na_total': int(feature_meta.get('rows_with_na_total', 0)),
        'rows_trimmed_outside_splits': int(feature_meta.get('rows_trimmed_outside_splits', 0)),
        'rows_after_trim_to_splits': int(feature_meta.get('rows_after_trim_to_splits', final_rows)),
        'final_model_rows': int(feature_meta.get('final_model_rows', final_rows)),
    }

    # Save
    save_artifacts(run_dir, model, scaler, cfg, metrics, regimes_df, diagnostics)
    logging.info('Saved artifacts to %s', run_dir)

    # No longer duplicate logs outside run_dir; artifacts remain only under run_dir

    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description='Train a Gaussian HMM on v1/v2 features (config-driven)')
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--log-level', default='INFO')
    args = ap.parse_args()

    # First-stage logging to stdout only; will add file handler once run_dir is known
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    logging.info('Loaded config: %s', args.config)

    run_hmm_pipeline(cfg, args.log_level)
    logging.info("HMM pipeline completed successfully")


if __name__ == '__main__':
    main()

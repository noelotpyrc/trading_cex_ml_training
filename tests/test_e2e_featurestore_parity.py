#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Prepare synthetic features/targets with shared timestamp
    data_dir = root / '.tmp' / 'featurestore_parity'
    data_dir.mkdir(parents=True, exist_ok=True)
    features_csv = data_dir / 'features.csv'
    targets_csv = data_dir / 'targets.csv'
    merged_csv = data_dir / 'merged.csv'

    import pandas as pd
    import numpy as np

    n = 240
    ts = pd.date_range('2024-01-01', periods=n, freq='H')
    rng = np.random.default_rng(42)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    f3 = rng.normal(size=n)
    y = 0.3 * f1 - 0.2 * f2 + 0.1 * f3 + 0.05 * rng.normal(size=n)

    feats = pd.DataFrame({'timestamp': ts, 'f1': f1, 'f2': f2, 'f3': f3})
    targs = pd.DataFrame({'timestamp': ts, 'y_logret_24h': y})
    feats.to_csv(features_csv, index=False)
    targs.to_csv(targets_csv, index=False)
    merged = feats.merge(targs, on='timestamp', how='inner')
    merged.to_csv(merged_csv, index=False)

    out_root = data_dir / 'out'
    out_root.mkdir(parents=True, exist_ok=True)

    # Legacy run using feature_store
    from model.run_lgbm_pipeline import prepare_training_data, tune_hyperparameters, train_model, persist_results

    cfg_fs = {
        'output_dir': out_root / 'fs',
        'training_splits_dir': out_root / 'fs' / 'splits',
        'target': {'variable': 'y_logret_24h', 'objective': {'name': 'regression', 'params': {}}},
        'split': {'train_ratio': 0.6, 'val_ratio': 0.2, 'test_ratio': 0.2},
        'model': {
            'type': 'lgbm',
            'params': {'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': 6, 'min_data_in_leaf': 20, 'num_boost_round': 50, 'early_stopping_rounds': 10, 'seed': 42},
            'eval_metrics': ['rmse']
        },
        'feature_store': {'features_csv': str(features_csv), 'targets_csv': str(targets_csv)},
    }

    dir_fs = prepare_training_data(cfg_fs)
    best_fs = tune_hyperparameters(cfg_fs, dir_fs)
    final_fs = best_fs if cfg_fs['model'].get('use_best_params_for_final', True) else cfg_fs['model']['params']
    _, m_fs, run_fs = train_model(cfg_fs, dir_fs, final_fs)
    persist_results(cfg_fs, run_fs, m_fs, final_fs, dir_fs)

    # Legacy run using merged CSV
    cfg_m = {
        'output_dir': out_root / 'merged',
        'training_splits_dir': out_root / 'merged' / 'splits',
        'target': {'variable': 'y_logret_24h', 'objective': {'name': 'regression', 'params': {}}},
        'split': {'train_ratio': 0.6, 'val_ratio': 0.2, 'test_ratio': 0.2},
        'model': cfg_fs['model'],
        'input_data': str(merged_csv),
    }

    dir_m = prepare_training_data(cfg_m)
    best_m = tune_hyperparameters(cfg_m, dir_m)
    final_m = best_m if cfg_m['model'].get('use_best_params_for_final', True) else cfg_m['model']['params']
    _, m_m, run_m = train_model(cfg_m, dir_m, final_m)
    persist_results(cfg_m, run_m, m_m, final_m, dir_m)

    # Compare metrics parity
    import json as _json
    rm_fs = _json.load(open(run_fs / 'metrics.json', 'r'))
    rm_m = _json.load(open(run_m / 'metrics.json', 'r'))
    key = 'rmse_test'
    assert key in rm_fs and key in rm_m
    diff = abs(float(rm_fs[key]) - float(rm_m[key]))
    assert diff < 1e-6, f'metric mismatch: {diff} >= tol'

    print('OK: feature_store vs merged CSV parity holds')


if __name__ == '__main__':
    main()

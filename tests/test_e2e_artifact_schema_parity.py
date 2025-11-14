#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import pandas as pd
    import numpy as np
    from model.run_lgbm_pipeline import prepare_training_data, train_model, persist_results

    # Synthetic merged dataset
    tmp = root / '.tmp' / 'artifact_parity'
    tmp.mkdir(parents=True, exist_ok=True)
    merged = tmp / 'merged.csv'
    n = 240
    ts = pd.date_range('2024-01-01', periods=n, freq='H')
    rng = np.random.default_rng(42)
    f1, f2, f3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = 0.3 * f1 - 0.2 * f2 + 0.1 * f3 + 0.05 * rng.normal(size=n)
    pd.DataFrame({'timestamp': ts, 'f1': f1, 'f2': f2, 'f3': f3, 'y_logret_24h': y}).to_csv(merged, index=False)

    cfg = {
        'input_data': str(merged),
        'output_dir': tmp / 'models',
        'training_splits_dir': tmp / 'models' / 'splits',
        'target': {'variable': 'y_logret_24h', 'objective': {'name': 'regression', 'params': {}}},
        'split': {'train_ratio': 0.6, 'val_ratio': 0.2, 'test_ratio': 0.2},
        'model': {
            'type': 'lgbm',
            'params': {'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': 6, 'min_data_in_leaf': 20, 'num_boost_round': 50, 'early_stopping_rounds': 10, 'seed': 42},
            'eval_metrics': ['rmse'],
        },
    }

    data_dir = prepare_training_data(cfg)
    _, metrics, run_dir = train_model(cfg, data_dir, cfg['model']['params'])
    # Persist run metadata and convenience artifacts
    persist_results(cfg, run_dir, metrics, cfg['model']['params'], data_dir)

    # Required artifacts
    required = ['model.txt', 'metrics.json', 'feature_importance.csv', 'pred_train.csv', 'pred_val.csv', 'pred_test.csv', 'best_params.json', 'pipeline_config.json', 'paths.json']
    for name in required:
        p = Path(run_dir) / name
        assert p.exists(), f'missing artifact: {name}'

    # Basic schema checks
    fi = pd.read_csv(Path(run_dir) / 'feature_importance.csv')
    assert set(['feature', 'importance_gain', 'importance_split']).issubset(set(fi.columns))

    pt = pd.read_csv(Path(run_dir) / 'pred_test.csv')
    assert 'y_true' in pt.columns and 'y_pred' in pt.columns

    print('OK: artifact presence and schema checks passed')


if __name__ == '__main__':
    main()

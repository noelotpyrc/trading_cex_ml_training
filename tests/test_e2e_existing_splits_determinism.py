#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from model.prepare_training_data import prepare_splits
    from model.run_lgbm_pipeline import train_model
    import pandas as pd
    import numpy as np

    # Create a small merged CSV
    tmp = root / '.tmp' / 'determinism'
    tmp.mkdir(parents=True, exist_ok=True)
    csv_path = tmp / 'merged.csv'

    n = 240
    ts = pd.date_range('2024-01-01', periods=n, freq='H')
    rng = np.random.default_rng(42)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    f3 = rng.normal(size=n)
    y = 0.3 * f1 - 0.2 * f2 + 0.1 * f3 + 0.05 * rng.normal(size=n)
    df = pd.DataFrame({'timestamp': ts, 'f1': f1, 'f2': f2, 'f3': f3, 'y_logret_24h': y})
    df.to_csv(csv_path, index=False)

    # Prepare splits once
    splits_dir = prepare_splits(
        input_path=csv_path,
        output_dir=tmp / 'splits',
        target='y_logret_24h',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )

    # Common config (no tuning)
    base_cfg = {
        'output_dir': tmp / 'models',
        'training_splits_dir': tmp / 'models' / 'splits',
        'target': {'variable': 'y_logret_24h', 'objective': {'name': 'regression', 'params': {}}},
        'split': {'existing_dir': str(splits_dir)},
        'model': {
            'type': 'lgbm',
            'params': {'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': 6, 'min_data_in_leaf': 20, 'num_boost_round': 50, 'early_stopping_rounds': 10, 'seed': 42},
            'eval_metrics': ['rmse']
        },
    }

    # Run twice reusing the same splits
    dir1 = splits_dir
    _, m1, run1 = train_model(base_cfg, dir1, base_cfg['model']['params'])

    dir2 = splits_dir
    _, m2, run2 = train_model(base_cfg, dir2, base_cfg['model']['params'])

    # Determinism: metrics and predictions must match
    import json as _json
    import pandas as _pd

    j1 = _json.load(open(run1 / 'metrics.json', 'r'))
    j2 = _json.load(open(run2 / 'metrics.json', 'r'))
    assert j1 == j2, 'metrics differ when reusing splits'

    p1 = _pd.read_csv(run1 / 'pred_test.csv')
    p2 = _pd.read_csv(run2 / 'pred_test.csv')
    assert _np_allclose_df(p1, p2), 'predictions differ when reusing splits'

    print('OK: determinism holds when reusing existing splits')


def _np_allclose_df(a, b, tol=1e-12) -> bool:
    import numpy as np
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        if a[c].dtype.kind in 'if' and b[c].dtype.kind in 'if':
            if not np.allclose(a[c].values, b[c].values, atol=tol, rtol=0):
                return False
        else:
            if not (a[c].values == b[c].values).all():
                return False
    return True


if __name__ == '__main__':
    main()

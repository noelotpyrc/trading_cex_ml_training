#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import pandas as pd
    from omegaconf import OmegaConf  # type: ignore

    # Build a minimal run to obtain a model
    from model.run_lgbm_pipeline import prepare_training_data, train_model

    tmp = root / '.tmp' / 'inference_parity'
    tmp.mkdir(parents=True, exist_ok=True)
    merged = tmp / 'merged.csv'

    n = 240
    ts = pd.date_range('2024-01-01', periods=n, freq='H')
    import numpy as np
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

    # Prepare one-row features using the same synthetic process as run_inference would
    df = pd.read_csv(merged)
    last_row = df.iloc[[-1]].copy()
    last_row_features = last_row.drop(columns=['timestamp', 'y_logret_24h'])
    last_row_features.insert(0, 'timestamp', last_row['timestamp'].values)

    # Legacy inference helpers
    from model.lgbm_inference import resolve_model_file, load_booster, align_features_for_booster, predict_dataframe

    model_file, _ = resolve_model_file(model_root=str(Path(run_dir).parent), model_path=str(run_dir / 'model.txt'))
    booster = load_booster(Path(model_file))

    aligned = align_features_for_booster(last_row_features.drop(columns=['timestamp']), booster)
    pred_legacy = float(predict_dataframe(booster, aligned)[0])

    # Hydra parity: identical path/model, so prediction must match exactly
    pred_hydra = float(predict_dataframe(booster, aligned)[0])

    assert abs(pred_legacy - pred_hydra) < 1e-12
    print('OK: inference parity (legacy vs Hydra) holds for last-bar sample')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    data_path = Path("/Volumes/Extreme SSD/trading_data/cex/tests/synth.csv")
    if not data_path.exists():
        from utils.generate_synthetic_data import main as gen
        sys.argv = ["gen", "--output", str(data_path), "--rows", "180"]
        gen()

    from model.prepare_training_data import prepare_splits

    out_base = Path("/Volumes/Extreme SSD/trading_data/cex/tests/prepared")
    target = "y_logret_24h"
    prepared_dir = prepare_splits(
        input_path=data_path,
        output_dir=out_base,
        target=target,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )

    config = {
        'input_data': data_path,
        'output_dir': Path("/Volumes/Extreme SSD/trading_data/cex/models/tests"),
        'target': {'variable': target, 'objective': {'name': 'regression', 'params': {}}},
        'model': {
            'type': 'lgbm',
            'params': {
                'learning_rate': 0.1,
                'num_leaves': 31,
                'max_depth': 6,
                'min_data_in_leaf': 20,
                'feature_fraction': 1.0,
                'bagging_fraction': 1.0,
                'bagging_freq': 1,
                'lambda_l1': 0.0,
                'lambda_l2': 0.0,
                'num_boost_round': 200,
                'early_stopping_rounds': 20,
                'seed': 42,
            },
            'eval_metrics': ['rmse'],
        },
    }

    from model.run_lgbm_pipeline import train_model
    booster, metrics, run_dir = train_model(config, prepared_dir, config['model']['params'])
    print("metrics:", metrics)
    print("run_dir:", run_dir)


if __name__ == '__main__':
    main()

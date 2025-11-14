#!/usr/bin/env python3
"""
Test script to validate that the pipeline uses CV best iteration for final training
and avoids validation leakage by disabling early stopping in the final fit.

Usage:
  python model/tests/test_synth_cv_best_iter.py \
    --config /Users/noel/projects/trading_cex/configs/model_configs/examples/synth_regression_rmse.json \
    --generate-if-missing

The script will:
- Ensure synthetic input CSV exists (generate if missing, with workspace fallbacks)
- Run data preparation, hyperparameter tuning, and model training
- Assert tuned params include num_boost_round and early_stopping_rounds=0
- Assert training used the fixed number of rounds (no early stopping)
- Persist results and verify artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `model` package is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Local imports
from model import run_lgbm_pipeline as pipeline


def _ensure_synth_csv(csv_path: Path, num_rows: int = 1200, seed: int = 42) -> None:
    """Create a simple synthetic regression dataset compatible with the pipeline.

    Columns: timestamp, x1, x2, x3, y_logret_24h
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range('2024-01-01', periods=num_rows, freq='h')
    feature_x1 = rng.normal(loc=0.0, scale=1.0, size=num_rows)
    feature_x2 = rng.normal(loc=0.0, scale=1.0, size=num_rows)
    feature_x3 = rng.normal(loc=0.0, scale=1.0, size=num_rows)
    target = 0.3 * feature_x1 - 0.2 * feature_x2 + 0.1 * feature_x3 + rng.normal(scale=0.1, size=num_rows)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'x1': feature_x1,
        'x2': feature_x2,
        'x3': feature_x3,
        'y_logret_24h': target,
    })
    df.to_csv(csv_path, index=False)


def _resolve_paths_with_fallbacks(config: dict, workspace_root: Path) -> Tuple[Path, Path]:
    """Return (input_csv, output_dir) ensuring they exist or creating fallbacks.

    If the configured volume (e.g., /Volumes/Extreme SSD/...) does not exist,
    fall back to workspace paths under results/tests/.
    """
    input_csv: Path = config['input_data']
    output_dir: Path = config['output_dir']

    # If the volume path root doesn't exist, fall back inside the repo
    if not input_csv.parent.exists():
        fallback_input = workspace_root / 'data' / 'tests' / 'synth.csv'
        input_csv = fallback_input
        config['input_data'] = input_csv

    if not output_dir.exists() and not output_dir.parent.exists():
        fallback_output = workspace_root / 'results' / 'tests'
        fallback_output.mkdir(parents=True, exist_ok=True)
        output_dir = fallback_output
        config['output_dir'] = output_dir

    return input_csv, output_dir


def run_test(config_path: Path, generate_if_missing: bool) -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    # Initialize pipeline logging to produce a pipeline_{timestamp}.log like production runs
    pipeline.setup_logging('INFO')
    config = pipeline.load_config(config_path)

    # Normalize paths and ensure existence (with fallbacks under workspace if needed)
    input_csv, output_dir = _resolve_paths_with_fallbacks(config, workspace_root)

    if not input_csv.exists():
        if generate_if_missing:
            _ensure_synth_csv(input_csv)
        else:
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Use a stable run timestamp suffix to isolate artifacts for this test
    config['_run_ts'] = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Step 1: Prepare data
    prepared_dir = pipeline.prepare_training_data(config)
    assert prepared_dir.exists(), f"Prepared data dir not found: {prepared_dir}"

    # Step 2: Tuning - should inject num_boost_round and early_stopping_rounds=0
    tuned_params = pipeline.tune_hyperparameters(config, prepared_dir)
    assert 'num_boost_round' in tuned_params and int(tuned_params['num_boost_round']) > 0, \
        "tuned params must include a positive num_boost_round captured from CV"
    assert int(tuned_params.get('early_stopping_rounds', -1)) == 0, \
        "final training must disable early stopping to avoid validation leakage"

    # Step 3: Train - must honor fixed num_boost_round (no early stopping)
    booster, metrics, run_dir = pipeline.train_model(config, prepared_dir, tuned_params)
    # Prefer current_iteration() when available; fallback to best_iteration
    trained_rounds = None
    if hasattr(booster, 'current_iteration'):
        try:
            trained_rounds = int(booster.current_iteration())
        except Exception:
            trained_rounds = None
    if trained_rounds is None or trained_rounds == 0:
        trained_rounds = int(getattr(booster, 'best_iteration', 0))
    assert trained_rounds == int(tuned_params['num_boost_round']), \
        f"Model trained rounds ({trained_rounds}) must equal tuned num_boost_round ({tuned_params['num_boost_round']})"

    # Step 4: Persist and verify artifacts
    run_dir = pipeline.persist_results(config, run_dir, metrics, tuned_params, prepared_dir)
    assert (run_dir / 'model.txt').exists(), 'model.txt not found in run dir'
    assert (run_dir / 'metrics.json').exists(), 'metrics.json not found in run dir'
    assert (run_dir / 'best_params.json').exists(), 'best_params.json not found in run dir'

    saved_best = json.load(open(run_dir / 'best_params.json', 'r'))
    assert int(saved_best.get('num_boost_round', -1)) == int(tuned_params['num_boost_round'])
    assert int(saved_best.get('early_stopping_rounds', -1)) == 0

    print('OK: CV best iteration propagated and early stopping disabled in final training.')
    print(f"Run directory: {run_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test CV best-iter propagation on synth RMSE config')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('/Users/noel/projects/trading_cex/configs/model_configs/examples/synth_regression_rmse.json'),
        help='Path to synth_regression_rmse.json config',
    )
    parser.add_argument('--generate-if-missing', action='store_true', help='Generate synthetic CSV if missing')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_test(args.config.resolve(), args.generate_if_missing)


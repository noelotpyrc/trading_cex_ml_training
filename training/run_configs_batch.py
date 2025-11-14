#!/usr/bin/env python3
"""
Batch runner for LightGBM pipeline configs.

Filters configs by input_data and target.variable, then runs each via
model/run_lgbm_pipeline.py sequentially.

Examples:
  python model/run_configs_batch.py \
    --config-dir configs/model_configs \
    --input-data \
      "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets.csv" \
    --target y_logret_24h \
    --log-level INFO

  # Or run explicitly-listed configs (skips filtering):
  python model/run_configs_batch.py \
    --configs \
      configs/model_configs/binance_btcusdt_p60_quantile_y_logret_24h_q05.json \
      configs/model_configs/binance_btcusdt_p60_quantile_y_logret_24h_q50.json \
    --log-level INFO
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List


def _load_json_if_valid(path: Path) -> dict | None:
    try:
        with path.open('r') as f:
            return json.load(f)
    except Exception:
        return None


def find_matching_configs(config_dir: Path, input_data: str, target_var: str) -> List[Path]:
    matches: List[Path] = []
    for p in sorted(config_dir.glob('*.json')):
        cfg = _load_json_if_valid(p)
        if not cfg:
            continue
        try:
            if str(cfg.get('input_data')) != input_data:
                continue
            tgt = cfg.get('target', {})
            if str(tgt.get('variable')) != target_var:
                continue
            matches.append(p)
        except Exception:
            # Skip malformed configs
            continue
    return matches


def run_pipeline_for_configs(configs: List[Path], log_level: str, stop_on_error: bool) -> int:
    if not configs:
        print('No matching configs found.')
        return 1

    pipeline_script = Path(__file__).resolve().parent / 'run_lgbm_pipeline.py'
    python_exe = sys.executable  # use current interpreter (ideally venv)

    print(f'Running {len(configs)} config(s) via {pipeline_script} ...')
    failures = 0
    for idx, cfg in enumerate(configs, start=1):
        print(f'[{idx}/{len(configs)}] {cfg}')
        cmd = [
            str(python_exe),
            str(pipeline_script),
            '--config', str(cfg.resolve()),
            '--log-level', log_level,
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            failures += 1
            print(f'FAILED ({cfg}): {e}')
            if stop_on_error:
                return failures
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch run LightGBM pipeline for matching configs.')
    parser.add_argument('--config-dir', type=Path, default=Path('configs/model_configs'),
                        help='Directory containing JSON configs (no JSONC).')
    parser.add_argument('--input-data', required=False,
                        help='Exact input_data path to match in configs (ignored if --configs is provided).')
    parser.add_argument('--target', required=False,
                        help='Target variable to match (e.g., y_logret_24h). Ignored if --configs is provided.')
    parser.add_argument('--configs', nargs='+', type=Path, default=None,
                        help='Explicit list of config files to run (bypasses filtering).')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--stop-on-error', action='store_true', help='Stop on first failure.')

    args = parser.parse_args()
    # If explicit configs provided, use them; otherwise filter by input/target
    if args.configs:
        configs = [p.resolve() for p in args.configs if p.exists()]
        missing = [str(p) for p in args.configs if not p.exists()]
        if missing:
            print(f"Warning: missing config files skipped: {', '.join(missing)}")
    else:
        if not args.input_data or not args.target:
            print('--input-data and --target are required when --configs is not provided', file=sys.stderr)
            sys.exit(2)
        config_dir = args.config_dir.resolve()
        input_data = str(args.input_data)
        target_var = str(args.target)
        configs = find_matching_configs(config_dir, input_data, target_var)
    failures = run_pipeline_for_configs(configs, args.log_level, args.stop_on_error)
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()



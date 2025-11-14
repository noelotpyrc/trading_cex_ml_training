#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Ensure project root on path
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Prepare synthetic dataset (small) if missing
    data_path = Path("/Volumes/Extreme SSD/trading_data/cex/tests/synth.csv")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        from utils.generate_synthetic_data import main as gen
        sys.argv = ["gen", "--output", str(data_path), "--rows", "180"]
        gen()

    # Build a minimal config JSON for full pipeline run
    out_base = Path("/Volumes/Extreme SSD/trading_data/cex/models/e2e")
    out_base.mkdir(parents=True, exist_ok=True)
    splits_root = out_base / "shared_splits"

    config = {
        "input_data": str(data_path),
        "output_dir": str(out_base),
        "training_splits_dir": str(splits_root),
        "target": {"variable": "y_logret_24h", "objective": {"name": "regression", "params": {}}},
        "split": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
        "model": {
            "type": "lgbm",
            # Use grid tuning to exercise CV + training; small space for speed
            "hyperparameter_tuning_method": "grid",
            "hyperparameter_search_space": {
                "learning_rate": [0.05, 0.1],
                "num_leaves": [15, 31]
            },
            "cv": {"method": "expanding", "n_folds": 3, "fold_val_size": 0.2, "gap": 0},
            "params": {
                "max_depth": 6,
                "min_data_in_leaf": 20,
                "feature_fraction": 1.0,
                "bagging_fraction": 1.0,
                "bagging_freq": 1,
                "lambda_l1": 0.0,
                "lambda_l2": 0.0,
                "num_boost_round": 200,
                "early_stopping_rounds": 20,
                "seed": 42
            },
            "eval_metrics": ["rmse"]
        }
    }

    tmp_cfg_dir = root / ".tmp" / "tests" / "configs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_cfg_dir / "e2e_config.json"
    cfg_path.write_text(json.dumps(config, indent=2))

    # Invoke pipeline CLI via venv python
    venv_python = root / "venv" / "bin" / "python"
    cmd = [str(venv_python), str(root / "model" / "run_lgbm_pipeline.py"), "--config", str(cfg_path)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Locate latest run directory and validate artifacts
    # New layout: output_dir/run_<ts>_lgbm_<target>_<objective>
    runs = sorted(out_base.glob("run_*_lgbm_*"))
    assert runs, f"No run_* folders under {out_base}"
    run_dir = runs[-1]
    print("Run dir:", run_dir)

    # Expected files
    expected = [
        "model.txt",
        "metrics.json",
        "feature_importance.csv",
        "pred_train.csv",
        "pred_val.csv",
        "pred_test.csv",
        "run_metadata.json",
        "pipeline_config.json",
        "best_params.json",
        "paths.json",
        "prep_metadata.json",
        "tuning_trials.csv",
    ]
    missing = [f for f in expected if not (run_dir / f).exists()]
    assert not missing, f"Missing artifacts: {missing}"

    # Quick sanity check: metrics file contains rmse_* keys
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert any(k.startswith("rmse_") for k in metrics.keys()), "Expected rmse_* metrics in metrics.json"

    # paths.json should point to training_splits_dir and prepared_data_dir
    paths = json.loads((run_dir / "paths.json").read_text())
    assert Path(paths["training_splits_dir"]).exists(), "training_splits_dir path missing"
    assert Path(paths["prepared_data_dir"]).exists(), "prepared_data_dir path missing"
    print("E2E OK. Artifacts present and metrics recorded.")


if __name__ == "__main__":
    main()



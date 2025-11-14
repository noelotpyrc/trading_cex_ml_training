#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
import subprocess


def ensure_splits(splits_root: Path, data_path: Path, target: str) -> Path:
    """Ensure at least one prepared_<ts>_<target> splits folder exists and return it."""
    # Prefer most recent existing
    candidates = sorted(splits_root.glob(f"prepared_*_{target}"))
    if candidates:
        return candidates[-1]

    # Otherwise, generate one using programmatic API
    splits_root.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        from utils.generate_synthetic_data import main as gen
        sys.argv = ["gen", "--output", str(data_path), "--rows", "180"]
        gen()

    # Use prepare_splits directly
    from model.prepare_training_data import prepare_splits
    ts_dir = splits_root / "prepared_manual"
    prepared_dir = prepare_splits(
        input_path=data_path,
        output_dir=ts_dir,
        target=target,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )
    return prepared_dir


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    data_path = Path("/Volumes/Extreme SSD/trading_data/cex/tests/synth.csv")
    out_base = Path("/Volumes/Extreme SSD/trading_data/cex/models/e2e")
    splits_root = out_base / "shared_splits"
    target = "y_logret_24h"

    # Ensure an existing splits folder
    prepared_dir = ensure_splits(splits_root, data_path, target)

    # Snapshot prepared_* folders to confirm no new splits are created during reuse
    before_dirs = set(p.name for p in splits_root.glob(f"prepared_*_{target}"))

    # Build config that reuses existing splits
    config = {
        "input_data": str(data_path),
        "output_dir": str(out_base),
        "training_splits_dir": str(splits_root),
        "target": {"variable": target, "objective": {"name": "regression", "params": {}}},
        "split": {"existing_dir": str(prepared_dir)},
        "model": {
            "type": "lgbm",
            "params": {
                "learning_rate": 0.1,
                "num_leaves": 31,
                "max_depth": 6,
                "min_data_in_leaf": 20,
                "feature_fraction": 1.0,
                "bagging_fraction": 1.0,
                "bagging_freq": 1,
                "lambda_l1": 0.0,
                "lambda_l2": 0.0,
                "num_boost_round": 120,
                "early_stopping_rounds": 20,
                "seed": 42
            },
            "eval_metrics": ["rmse"]
        }
    }

    tmp_cfg_dir = root / ".tmp" / "tests" / "configs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_cfg_dir / "e2e_existing_splits.json"
    cfg_path.write_text(json.dumps(config, indent=2))

    # Run pipeline via CLI
    venv_python = root / "venv" / "bin" / "python"
    cmd = [str(venv_python), str(root / "model" / "run_lgbm_pipeline.py"), "--config", str(cfg_path)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Verify outputs
    runs = sorted(out_base.glob("run_*_lgbm_*"))
    assert runs, f"No run_* folders under {out_base}"
    run_dir = runs[-1]
    print("Run dir:", run_dir)

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
    ]
    missing = [f for f in expected if not (run_dir / f).exists()]
    assert not missing, f"Missing artifacts: {missing}"

    # paths.json should reference the reused prepared_dir
    paths = json.loads((run_dir / "paths.json").read_text())
    assert Path(paths["prepared_data_dir"]) == prepared_dir, "prepared_data_dir should match existing_dir"
    assert Path(paths["training_splits_dir"]) == splits_root, "training_splits_dir mismatch"

    # Confirm no new prepared_* directory created
    after_dirs = set(p.name for p in splits_root.glob(f"prepared_*_{target}"))
    assert after_dirs == before_dirs, "New splits were created unexpectedly while reusing existing_dir"
    print("Existing-splits E2E OK. Reused splits and produced outputs.")


if __name__ == "__main__":
    main()



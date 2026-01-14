#!/usr/bin/env python3
"""
Walk-Forward Training Runner (Wrapper Pattern).

Orchestrates training across walk-forward folds by invoking `run_catboost_pipeline.py`
for each fold via subprocess. This ensures isolation and reuses existing 
training/logging logic.

Usage:
    python training/run_walk_forward_training.py --config configs/walk_forward/my_config.json
"""

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration file."""
    with open(config_path) as f:
        return json.load(f)


def discover_folds(splits_dir: Path) -> List[Dict[str, Any]]:
    """Discover available folds from walk_forward_config.json."""
    config_path = splits_dir / "walk_forward_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"walk_forward_config.json not found in {splits_dir}")
    
    with open(config_path) as f:
        wf_config = json.load(f)
    
    folds = []
    for fold_info in wf_config.get("folds", []):
        fold_name = fold_info["name"]
        fold_dir = splits_dir / fold_name
        if fold_dir.exists():
            folds.append({
                "name": fold_name,
                "dir": fold_dir,
                "info": fold_info,  # Store full info
            })
    return folds


def filter_folds(
    folds: List[Dict[str, Any]],
    start_fold: Optional[int] = None,
    end_fold: Optional[int] = None,
    fold_range: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter folds based on CLI arguments."""
    if fold_range:
        parts = fold_range.split("-")
        start_fold = int(parts[0])
        end_fold = int(parts[1]) if len(parts) > 1 else start_fold
    
    if start_fold is not None or end_fold is not None:
        start = start_fold or 1
        end = end_fold or len(folds)
        folds = [f for f in folds if start <= int(f["name"].split("_")[1]) <= end]
    
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward training wrapper")
    parser.add_argument("--config", type=Path, required=True, help="Walk-forward config JSON")
    parser.add_argument("--folds", type=str, default=None, help="Fold range (e.g., '1-10')")
    parser.add_argument("--start-fold", type=int, default=None, help="Start fold")
    parser.add_argument("--end-fold", type=int, default=None, help="End fold")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--log-level", type=str, default="INFO")
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    # Load config
    wf_config = load_config(args.config)
    splits_dir = Path(wf_config["walk_forward"]["splits_dir"])
    experiment_name = wf_config["walk_forward"]["experiment_name"]
    run_prefix = wf_config["walk_forward"].get("run_name_prefix", "wf")
    root_out_dir = Path(wf_config["output"]["root_dir"]) / experiment_name
    
    # Discover folds
    folds = discover_folds(splits_dir)
    folds = filter_folds(folds, args.start_fold, args.end_fold, args.folds)
    logging.info(f"Running {len(folds)} folds: {[f['name'] for f in folds]}")
    
    # Create experiment dir
    root_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save a copy of the master walk-forward config for reproducibility
    import shutil
    config_copy_path = root_out_dir / args.config.name
    if not config_copy_path.exists():
        shutil.copy(args.config, config_copy_path)
        logging.info(f"Saved walk-forward config to: {config_copy_path}")
    
    # Path to the actual trainer script (train_and_register.py handles MLflow)
    trainer_script = _project_root / "mlflow_scripts" / "train_and_register.py"
    if not trainer_script.exists():
        logging.error(f"Trainer script not found: {trainer_script}")
        sys.exit(1)
    
    # Get MLflow settings
    mlflow_config = wf_config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "http://127.0.0.1:5001")

    successful = 0
    failed = 0
    
    for fold in folds:
        fold_name = fold["name"]
        fold_dir = fold["dir"]
        fold_idx = int(fold_name.split("_")[1])
        
        # Construct run-specific config based on base config + overrides
        # We start with the 'model', 'target', 'feature_selection' from wf_config
        # And construct a full config object expected by run_catboost_pipeline.py
        
        run_config = {
            "model": wf_config["model"],
            "target": wf_config["target"],
            "feature_selection": wf_config.get("feature_selection", {}),
            "split": {
                "type": "existing",
                "existing_dir": str(fold_dir),
            },
            # Explicit output dir for this run
            "output_dir": str(root_out_dir),
            # Use existing splits dir to avoid creating empty 'splits' folder
            "training_splits_dir": str(fold_dir.parent),
        }
        
        # Force usage of provided params (skip tuning logic)
        run_config["model"]["use_best_params_for_final"] = False
        
        # Let's create a temporary config file for this run
        temp_config_path = root_out_dir / f"config_{fold_name}.json"
        
        with open(temp_config_path, "w") as f:
            json.dump(run_config, f, indent=2)
            
        cmd = [
            sys.executable,
            str(trainer_script),
            "--config", str(temp_config_path),
            "--tracking-uri", tracking_uri,
            "--experiment", experiment_name,
            "--log-run-only",  # Log to MLflow but don't register model
            "--sanitize-paths",
        ]
        
        logging.info(f"Starting {fold_name}...")
        if args.dry_run:
            logging.info(f"Dry run: {' '.join(cmd)}")
            successful += 1
            continue
            
        try:
            # Run subprocess
            subprocess.run(cmd, check=True)
            successful += 1
            logging.info(f"Finished {fold_name} successfully.")
        except subprocess.CalledProcessError as e:
            failed += 1
            logging.error(f"Failed {fold_name}: {e}")
    logging.info(f"Walk-forward complete. Success: {successful}, Failed: {failed}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Feature Family Sweep Runner for Single Model Configs.

Runs training multiple times, modifying features one family at a time.
All runs are logged to a SINGLE experiment for easy comparison.

For walk-forward configs, use run_feature_sweep_walkforward.py instead.

Two modes:
  - exclude: Remove one family at a time (ablation study)
  - include: Add one family at a time to baseline features

Usage:
    # Exclude mode (remove one family at a time):
    python training/run_feature_sweep_single.py \
        --config configs/model_configs/my_config.json \
        --feature-families configs/feature_lists/feature_families.json \
        --experiment my-experiment-name \
        --mode exclude

    # Include mode (add one family at a time):
    python training/run_feature_sweep_single.py \
        --config configs/model_configs/my_config.json \
        --feature-families configs/feature_lists/feature_families.json \
        --experiment my-experiment-name \
        --mode include

    # Dry run:
    python training/run_feature_sweep_single.py \
        --config configs/model_configs/my_config.json \
        --feature-families configs/feature_lists/feature_families.json \
        --experiment my-experiment-name \
        --mode include --dry-run
"""

import argparse
import copy
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

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


def load_feature_families(families_path: Path) -> Dict[str, List[str]]:
    """Load feature families from a JSON config file."""
    with open(families_path) as f:
        return json.load(f)


def get_features_excluding_family(
    all_features: List[str], 
    family_features: List[str]
) -> List[str]:
    """Return all features except those in the specified family."""
    excluded_set = set(family_features)
    return [f for f in all_features if f not in excluded_set]


def get_features_including_family(
    base_features: List[str],
    family_features: List[str]
) -> List[str]:
    """Return base features plus the specified family."""
    return base_features + family_features


def create_modified_config(
    base_config: Dict[str, Any],
    features: List[str],
    family_name: str,
    mode: str,
) -> Dict[str, Any]:
    """Create a modified config with specific features."""
    config = copy.deepcopy(base_config)
    
    # Update feature selection
    config["feature_selection"]["include"] = features
    
    # Add metadata for MLflow tracking
    config["feature_sweep"] = {
        "mode": mode,
        "family": family_name,
    }
    
    return config


def run_catboost_pipeline(
    config_path: Path,
    experiment: str,
    tracking_uri: str,
    mode: str,
    family_name: str,
    dry_run: bool = False,
) -> bool:
    """Run catboost pipeline with the given config."""
    cmd = [
        sys.executable,
        str(_project_root / "mlflow_scripts" / "train_and_register.py"),
        "--config", str(config_path),
        "--experiment", experiment,
        "--tracking-uri", tracking_uri,
        "--log-run-only",
        "--sanitize-paths",
        # Pass extra params for MLflow tracking
        "--extra-params", json.dumps({
            "sweep_mode": mode,
            "sweep_family": family_name,
        }),
    ]
    
    logging.info(f"Executing: {' '.join(cmd)}")
    
    if dry_run:
        logging.info("[DRY RUN] Would execute above command")
        return True
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Training failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run feature family sweep using standard model config"
    )
    parser.add_argument(
        "--config", 
        type=Path, 
        required=True, 
        help="Base model config JSON"
    )
    parser.add_argument(
        "--feature-families",
        type=Path,
        required=True,
        help="JSON file mapping family names to lists of features"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="MLflow experiment name (all runs go to this single experiment)"
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://127.0.0.1:5001",
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["exclude", "include"],
        default="exclude",
        help="'exclude' removes one family at a time, 'include' adds one family at a time"
    )
    parser.add_argument(
        "--families",
        type=str,
        default=None,
        help="Comma-separated list of specific families to process (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be executed without running"
    )
    parser.add_argument(
        "--list-families",
        action="store_true",
        help="List feature families and exit"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also run baseline before feature modifications"
    )
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    # Load base config
    base_config = load_config(args.config)
    base_features = base_config.get("feature_selection", {}).get("include", [])
    
    if not base_features:
        logging.error("No features found in config under feature_selection.include")
        sys.exit(1)
    
    logging.info(f"Found {len(base_features)} features in config")
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Experiment: {args.experiment}")
    
    # Load feature families
    feature_families = load_feature_families(args.feature_families)
    logging.info(f"Loaded {len(feature_families)} feature families from {args.feature_families}")
    
    # For exclude mode, filter to families that exist in base features
    if args.mode == "exclude":
        base_features_set = set(base_features)
        filtered_families = {}
        for family, features in feature_families.items():
            matching = [f for f in features if f in base_features_set]
            if matching:
                filtered_families[family] = matching
        feature_families = filtered_families
    
    # Print families
    logging.info("Feature families:")
    for family, features in sorted(feature_families.items()):
        logging.info(f"  {family}: {len(features)} features - {features}")
    
    if args.list_families:
        print(f"\nFeature Families Summary (mode={args.mode}):")
        print("-" * 50)
        for family, features in sorted(feature_families.items()):
            print(f"\n{family}:")
            for f in features:
                print(f"  - {f}")
        sys.exit(0)
    
    # Determine which families to process
    if args.families:
        families_to_process = [f.strip() for f in args.families.split(",")]
        for family in families_to_process:
            if family not in feature_families:
                logging.error(f"Unknown family: {family}")
                logging.error(f"Available families: {list(feature_families.keys())}")
                sys.exit(1)
    else:
        families_to_process = list(feature_families.keys())
    
    logging.info(f"Will process {len(families_to_process)} families: {families_to_process}")
    
    # Create temp directory for configs
    temp_config_dir = args.config.parent / "sweep_temp"
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Optionally run baseline first
    if args.include_baseline:
        logging.info("=" * 60)
        logging.info("Running BASELINE")
        logging.info("=" * 60)
        
        baseline_config = copy.deepcopy(base_config)
        baseline_config["feature_sweep"] = {
            "mode": "baseline",
            "family": "none",
        }
        
        baseline_config_path = temp_config_dir / "config_baseline.json"
        with open(baseline_config_path, "w") as f:
            json.dump(baseline_config, f, indent=2)
        
        if args.dry_run:
            logging.info(f"[DRY RUN] Would run baseline with {len(base_features)} features")
            results["baseline"] = "dry_run"
        else:
            success = run_catboost_pipeline(
                baseline_config_path,
                args.experiment,
                args.tracking_uri,
                mode="baseline",
                family_name="none",
                dry_run=False,
            )
            results["baseline"] = "success" if success else "failed"
    
    # Process each family
    for family in families_to_process:
        family_features = feature_families[family]
        
        if args.mode == "exclude":
            modified_features = get_features_excluding_family(base_features, family_features)
            action = "Excluding"
        else:
            modified_features = get_features_including_family(base_features, family_features)
            action = "Including"
        
        logging.info("=" * 60)
        logging.info(f"{action}: {family}")
        logging.info(f"  Family has {len(family_features)} features: {family_features}")
        logging.info(f"  Result: {len(modified_features)} total features")
        logging.info("=" * 60)
        
        # Create modified config
        modified_config = create_modified_config(
            base_config,
            modified_features,
            family,
            args.mode,
        )
        
        # Save temporary config
        config_filename = f"config_{args.mode}_{family}.json"
        temp_config_path = temp_config_dir / config_filename
        
        with open(temp_config_path, "w") as f:
            json.dump(modified_config, f, indent=2)
        
        logging.info(f"Created config: {temp_config_path}")
        
        if args.dry_run:
            logging.info(f"[DRY RUN] Would run with {len(modified_features)} features")
            results[family] = "dry_run"
        else:
            success = run_catboost_pipeline(
                temp_config_path,
                args.experiment,
                args.tracking_uri,
                mode=args.mode,
                family_name=family,
                dry_run=False,
            )
            results[family] = "success" if success else "failed"
    
    # Summary
    logging.info("\n" + "=" * 60)
    logging.info(f"FEATURE SWEEP SUMMARY (mode={args.mode})")
    logging.info("=" * 60)
    for family, status in results.items():
        logging.info(f"  {family}: {status}")
    
    success_count = sum(1 for s in results.values() if s == "success")
    fail_count = sum(1 for s in results.values() if s == "failed")
    dry_run_count = sum(1 for s in results.values() if s == "dry_run")
    
    if dry_run_count > 0:
        logging.info(f"\nDry run completed. {dry_run_count} runs would be executed.")
    else:
        logging.info(f"\nCompleted: {success_count} success, {fail_count} failed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Feature Family Sweep Runner for Walk-Forward Configs.

Runs walk-forward training multiple times, modifying features one family at a time.
Each family gets its own experiment for separate tracking.

For single model configs, use run_feature_sweep_single.py instead.

Two modes:
  - exclude: Remove one family at a time (ablation study)
  - include: Add one family at a time to baseline features

Usage:
    # Exclude mode (remove one family at a time):
    python training/run_feature_sweep_walkforward.py \
        --config configs/walk_forward/aux_momentum_drift_rolling_v2.json \
        --feature-families configs/feature_lists/feature_families_momentum_drift.json \
        --mode exclude

    # Include mode (add one family at a time):
    python training/run_feature_sweep_walkforward.py \
        --config configs/walk_forward/aux_momentum_drift_rolling_v2.json \
        --feature-families configs/feature_lists/feature_families_momentum_drift_addon.json \
        --mode include
"""

import argparse
import copy
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

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
    """
    Load feature families from a JSON config file.
    
    Expected format:
    {
        "family_name": ["feature1", "feature2", ...],
        ...
    }
    """
    with open(families_path) as f:
        return json.load(f)


def categorize_features(
    features: List[str], 
    feature_families: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Categorize features based on the provided feature families config.
    
    Only includes features that are both in the families config AND in the 
    provided features list (intersection).
    
    Returns a dict mapping family names to lists of features in that family.
    """
    features_set = set(features)
    categorized = {}
    
    for family, family_features in feature_families.items():
        # Only include features that are actually in the config's feature list
        matching = [f for f in family_features if f in features_set]
        if matching:
            categorized[family] = matching
    
    # Find uncategorized features (in config but not in any family)
    all_family_features = set()
    for family_features in feature_families.values():
        all_family_features.update(family_features)
    
    uncategorized = [f for f in features if f not in all_family_features]
    if uncategorized:
        categorized["uncategorized"] = uncategorized
    
    return categorized


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
    mode: str,  # "exclude" or "include"
    experiment_suffix: str = None,
) -> Dict[str, Any]:
    """Create a modified config with specific features."""
    config = copy.deepcopy(base_config)
    
    # Update feature selection
    config["feature_selection"]["include"] = features
    
    # Update experiment name - each run gets its own experiment
    base_experiment = config["walk_forward"]["experiment_name"]
    mode_label = "exclude" if mode == "exclude" else "include"
    
    if experiment_suffix:
        config["walk_forward"]["experiment_name"] = f"{base_experiment}-{experiment_suffix}-{mode_label}-{family_name}"
    else:
        config["walk_forward"]["experiment_name"] = f"{base_experiment}-{mode_label}-{family_name}"
    
    # Add metadata
    config["feature_study"] = {
        "mode": mode,
        "family": family_name,
        "original_experiment": base_experiment,
    }
    
    return config


def run_walk_forward_training(
    config_path: Path,
    dry_run: bool = False,
    folds: str = None,
) -> bool:
    """Run walk-forward training with the given config."""
    cmd = [
        sys.executable,
        str(_project_root / "training" / "run_walk_forward_training.py"),
        "--config", str(config_path),
    ]
    
    if folds:
        cmd.extend(["--folds", folds])
    
    if dry_run:
        cmd.append("--dry-run")
    
    logging.info(f"Executing: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Walk-forward training failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run feature ablation/addition study using walk-forward training"
    )
    parser.add_argument(
        "--config", 
        type=Path, 
        required=True, 
        help="Base walk-forward config JSON"
    )
    parser.add_argument(
        "--feature-families",
        type=Path,
        required=True,
        help="JSON file mapping family names to lists of features"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["exclude", "include"],
        default="exclude",
        help="'exclude' removes one family at a time (ablation), 'include' adds one family at a time"
    )
    parser.add_argument(
        "--experiment-suffix",
        type=str,
        default=None,
        help="Optional suffix for experiment name (e.g., 'v1')"
    )
    parser.add_argument(
        "--families",
        type=str,
        default=None,
        help="Comma-separated list of specific families to process (default: all)"
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Fold range to run (e.g., '1-3'), passed to walk-forward runner"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be executed without running"
    )
    parser.add_argument(
        "--list-families",
        action="store_true",
        help="List feature families found in config and exit"
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
    
    logging.info(f"Found {len(base_features)} features in walk-forward config")
    logging.info(f"Mode: {args.mode}")
    
    # Load feature families from config file
    feature_families = load_feature_families(args.feature_families)
    logging.info(f"Loaded {len(feature_families)} feature families from {args.feature_families}")
    
    # Categorize features based on mode
    if args.mode == "exclude":
        # For exclude mode, we want families that ARE in the base config
        categorized = categorize_features(base_features, feature_families)
    else:
        # For include mode, the families file contains NEW features to add
        # We don't filter by base_features - we use the families as-is
        categorized = {k: v for k, v in feature_families.items()}
    
    # Print feature families
    logging.info("Feature families:")
    for family, features in sorted(categorized.items()):
        logging.info(f"  {family}: {len(features)} features - {features}")
    
    if args.list_families:
        print(f"\nFeature Families Summary (mode={args.mode}):")
        print("-" * 50)
        for family, features in sorted(categorized.items()):
            print(f"\n{family}:")
            for f in features:
                print(f"  - {f}")
        sys.exit(0)
    
    # Determine which families to process
    if args.families:
        families_to_process = [f.strip() for f in args.families.split(",")]
        # Validate
        for family in families_to_process:
            if family not in categorized:
                logging.error(f"Unknown family: {family}")
                logging.error(f"Available families: {list(categorized.keys())}")
                sys.exit(1)
    else:
        # Process all families (except uncategorized for exclude mode)
        if args.mode == "exclude":
            families_to_process = [f for f in categorized.keys() if f != "uncategorized"]
        else:
            families_to_process = list(categorized.keys())
    
    logging.info(f"Will process {len(families_to_process)} families: {families_to_process}")
    
    # Create temp directory for configs
    temp_config_dir = args.config.parent / "ablation_temp"
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Optionally run baseline first
    if args.include_baseline:
        logging.info("=" * 60)
        logging.info("Running BASELINE")
        logging.info("=" * 60)
        
        baseline_config = copy.deepcopy(base_config)
        base_experiment = baseline_config["walk_forward"]["experiment_name"]
        if args.experiment_suffix:
            baseline_config["walk_forward"]["experiment_name"] = f"{base_experiment}-{args.experiment_suffix}-baseline"
        else:
            baseline_config["walk_forward"]["experiment_name"] = f"{base_experiment}-baseline"
        baseline_config["feature_study"] = {
            "mode": "baseline",
            "family": "none",
            "original_experiment": base_experiment,
        }
        
        baseline_config_path = temp_config_dir / "config_baseline.json"
        with open(baseline_config_path, "w") as f:
            json.dump(baseline_config, f, indent=2)
        
        if args.dry_run:
            logging.info(f"[DRY RUN] Would run baseline with {len(base_features)} features")
            logging.info(f"[DRY RUN] Config written to: {baseline_config_path}")
            results["baseline"] = "dry_run"
        else:
            success = run_walk_forward_training(
                baseline_config_path, 
                dry_run=False, 
                folds=args.folds
            )
            results["baseline"] = "success" if success else "failed"
    
    # Process each family
    for family in families_to_process:
        family_features = categorized[family]
        
        if args.mode == "exclude":
            # Remove this family from base features
            modified_features = get_features_excluding_family(base_features, family_features)
            action = "Excluding"
        else:
            # Add this family to base features
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
            args.experiment_suffix,
        )
        
        # Save temporary config
        config_filename = f"config_{args.mode}_{family}.json"
        temp_config_path = temp_config_dir / config_filename
        
        with open(temp_config_path, "w") as f:
            json.dump(modified_config, f, indent=2)
        
        logging.info(f"Created config: {temp_config_path}")
        
        if args.dry_run:
            logging.info(f"[DRY RUN] Would run walk-forward with {len(modified_features)} features")
            results[family] = "dry_run"
        else:
            success = run_walk_forward_training(
                temp_config_path, 
                dry_run=False, 
                folds=args.folds
            )
            results[family] = "success" if success else "failed"
    
    # Summary
    logging.info("\n" + "=" * 60)
    logging.info(f"FEATURE STUDY SUMMARY (mode={args.mode})")
    logging.info("=" * 60)
    for family, status in results.items():
        logging.info(f"  {family}: {status}")
    
    # Count successes/failures
    success_count = sum(1 for s in results.values() if s == "success")
    fail_count = sum(1 for s in results.values() if s == "failed")
    dry_run_count = sum(1 for s in results.values() if s == "dry_run")
    
    if dry_run_count > 0:
        logging.info(f"\nDry run completed. {dry_run_count} runs would be executed.")
    else:
        logging.info(f"\nCompleted: {success_count} success, {fail_count} failed")


if __name__ == "__main__":
    main()

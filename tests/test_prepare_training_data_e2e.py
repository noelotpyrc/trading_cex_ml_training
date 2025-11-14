#!/usr/bin/env python3
"""End-to-end regression check for prepare_training_data with feature_store inputs.

This script:
  1. Loads the canonical feature_store features/targets CSVs.
  2. Randomly selects one target column from targets.csv (excluding timestamp).
  3. Samples a random subset of feature columns from features.csv.
  4. Writes temporary trimmed CSVs and invokes prepare_splits_from_feature_store.
  5. Verifies that split artifacts and metadata are produced, then prints a summary.

Run with the project virtualenv python, e.g.:
    ./venv/bin/python model/tests/test_prepare_training_data_e2e.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

FEATURES_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/feature_store/features.csv")
TARGETS_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/feature_store/targets.csv")
HMM_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/feature_store/hmm_features.csv")
OUTPUT_ROOT = Path("/Volumes/Extreme SSD/trading_data/cex/tests/tmp_prepare_training_data_e2e")
RNG_SEED = 42
MAX_FEATURES = 32


def _ensure_project_root_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _cleanup_output_root() -> None:
    if OUTPUT_ROOT.exists():
        for child in OUTPUT_ROOT.glob("._*"):
            try:
                child.unlink()
            except FileNotFoundError:
                pass
        shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _select_random_columns(rng: np.random.Generator) -> tuple[str, List[str]]:
    targets_df_head = pd.read_csv(TARGETS_PATH, nrows=1)
    target_candidates = [c for c in targets_df_head.columns if c != "timestamp"]
    if not target_candidates:
        raise RuntimeError("No target columns available in targets.csv")
    target_col = rng.choice(target_candidates)

    features_df_head = pd.read_csv(FEATURES_PATH, nrows=1)
    feature_candidates = [c for c in features_df_head.columns if c != "timestamp"]
    if not feature_candidates:
        raise RuntimeError("No feature columns available in features.csv")
    num_features = min(MAX_FEATURES, max(5, len(feature_candidates) // 4))
    selected_features = rng.choice(feature_candidates, size=num_features, replace=False)
    return target_col, sorted(selected_features.tolist())


def _write_trimmed_csvs(target_col: str, feature_cols: List[str]) -> tuple[Path, Path]:
    trimmed_dir = OUTPUT_ROOT / "trimmed"
    trimmed_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(FEATURES_PATH, usecols=["timestamp"] + feature_cols)
    targets = pd.read_csv(TARGETS_PATH, usecols=["timestamp", target_col])

    features_out = trimmed_dir / "features_subset.csv"
    targets_out = trimmed_dir / "targets_subset.csv"
    features.to_csv(features_out, index=False)
    targets.to_csv(targets_out, index=False)
    return features_out, targets_out


def _run_prepare_splits(features_csv: Path, targets_csv: Path, target_col: str, *, include_hmm: bool) -> Path:
    from model.prepare_training_data import prepare_splits_from_feature_store

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"prepared_{ts}_{target_col}"
    extra_feature_files = []
    if include_hmm:
        if not HMM_PATH.exists():
            raise FileNotFoundError(f"hmm_features.csv not found at {HMM_PATH}")
        extra_feature_files.append({
            "path": str(HMM_PATH),
            "include": ["hmm_regime_*"],
        })

    prepared_dir = prepare_splits_from_feature_store(
        features_csv=features_csv,
        targets_csv=targets_csv,
        output_dir=out_dir,
        target=target_col,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        cutoff_start=None,
        cutoff_mid=None,
        extra_feature_files=extra_feature_files if include_hmm else None,
    )
    return prepared_dir


def _validate_outputs(prepared_dir: Path, target_col: str, feature_cols: List[str], include_hmm: bool) -> None:
    expected_files = {f"X_{split}.csv" for split in ("train", "val", "test")} | {
        f"y_{split}.csv" for split in ("train", "val", "test")
    } | {"prep_metadata.json"}
    actual = {p.name for p in prepared_dir.glob("*")}
    missing = expected_files - actual
    if missing:
        raise RuntimeError(f"Missing expected files in {prepared_dir}: {missing}")

    meta = json.loads((prepared_dir / "prep_metadata.json").read_text())
    if meta.get("target_column") != target_col:
        raise AssertionError(f"Metadata target mismatch: {meta.get('target_column')} vs {target_col}")

    X_train = pd.read_csv(prepared_dir / "X_train.csv")
    retained_cols = {c for c in feature_cols if c in X_train.columns}
    if not retained_cols:
        raise AssertionError("None of the sampled feature columns survived in X_train.csv")
    if "timestamp" in X_train.columns:
        raise AssertionError("timestamp column should be excluded from X splits")

    if include_hmm:
        hmm_cols = [c for c in X_train.columns if c.startswith("hmm_regime_")]
        if not hmm_cols:
            raise AssertionError("Expected HMM features in X_train.csv but none found")

    y_train = pd.read_csv(prepared_dir / "y_train.csv")
    if target_col not in y_train.columns:
        raise AssertionError(f"Target column '{target_col}' missing from y_train.csv")


def main() -> None:
    _ensure_project_root_on_path()
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"features.csv not found at {FEATURES_PATH}")
    if not TARGETS_PATH.exists():
        raise FileNotFoundError(f"targets.csv not found at {TARGETS_PATH}")

    _cleanup_output_root()

    rng = np.random.default_rng(RNG_SEED)
    target_col, feature_cols = _select_random_columns(rng)
    include_hmm = HMM_PATH.exists()
    print(f"[e2e] Selected target: {target_col}")
    print(f"[e2e] Selected {len(feature_cols)} feature columns (sample): {feature_cols[:8]}{ '...' if len(feature_cols) > 8 else ''}")
    print(f"[e2e] Including HMM features: {include_hmm}")

    features_subset, targets_subset = _write_trimmed_csvs(target_col, feature_cols)
    print(f"[e2e] Trimmed features -> {features_subset}")
    print(f"[e2e] Trimmed targets -> {targets_subset}")

    prepared_dir = _run_prepare_splits(features_subset, targets_subset, target_col, include_hmm=include_hmm)
    print(f"[e2e] prepare_splits_from_feature_store output -> {prepared_dir}")

    _validate_outputs(prepared_dir, target_col, feature_cols, include_hmm=include_hmm)
    print("[e2e] Validation passed. Split artifacts and metadata verified.")


if __name__ == "__main__":
    main()

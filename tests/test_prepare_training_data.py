#!/usr/bin/env python3
"""Lightweight regression checks for prepare_training_data helpers.

This script exercises both prepare_splits (merged CSV) and
prepare_splits_from_feature_store (separate features/targets CSVs) without
relying on pytest. Run directly with the project venv python:

    ./venv/bin/python model/tests/test_prepare_training_data.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def _ensure_project_root_on_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _make_tmp_dir() -> Path:
    root_tmp = Path("/Volumes/Extreme SSD/trading_data/cex/tests/tmp_prepare_training_data")
    if root_tmp.exists():
        # Guard against macOS dot-underscore artifacts left behind by Finder/Spotlight
        for child in root_tmp.glob("._*"):
            try:
                child.unlink()
            except FileNotFoundError:
                pass
        shutil.rmtree(root_tmp, ignore_errors=True)
    root_tmp.mkdir(parents=True, exist_ok=True)
    return root_tmp


def _build_merged_csv(path: Path, rows: int = 120) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "close": pd.Series(range(rows), dtype=float),
            "volume": pd.Series(range(rows), dtype=float) * 3.0,
            "y_logret_24h": pd.Series(range(rows), dtype=float) / 100.0,
        }
    )
    csv_path = path / "merged.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _build_feature_store_csvs(path: Path, rows: int = 120) -> tuple[Path, Path]:
    path.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h")
    features = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": pd.Series(range(rows), dtype=float),
            "volume": pd.Series(range(rows), dtype=float) * 2.0,
            "rsi": pd.Series(range(rows), dtype=float) / 10.0,
        }
    )
    targets = pd.DataFrame(
        {
            "timestamp": timestamps,
            "y_logret_24h": pd.Series(range(rows), dtype=float) / 100.0,
            "y_mfe_24h": pd.Series(range(rows), dtype=float) / 50.0,
        }
    )
    features_path = path / "features.csv"
    targets_path = path / "targets.csv"
    features.to_csv(features_path, index=False)
    targets.to_csv(targets_path, index=False)
    return features_path, targets_path


def _assert_basic_structure(prepared_dir: Path, target: str) -> None:
    expected_files = {f"X_{split}.csv" for split in ("train", "val", "test")} | {
        f"y_{split}.csv" for split in ("train", "val", "test")
    } | {"prep_metadata.json"}
    actual = {p.name for p in prepared_dir.glob("*")}
    missing = expected_files - actual
    assert not missing, f"Missing files in {prepared_dir}: {missing}"

    meta = json.loads((prepared_dir / "prep_metadata.json").read_text())
    assert meta["target_column"] == target, f"Unexpected target_column: {meta['target_column']}"
    assert meta["num_rows_after"] > 0, "Expected some rows after cleaning"


def _run_prepare_splits_checks(tmp_dir: Path) -> None:
    from model.prepare_training_data import prepare_splits

    merged_csv = _build_merged_csv(tmp_dir)
    out_dir = tmp_dir / "prepared_ratio"
    prepared_dir = prepare_splits(
        input_path=merged_csv,
        output_dir=out_dir,
        target="y_logret_24h",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        cutoff_start=None,
        cutoff_mid=None,
    )

    _assert_basic_structure(prepared_dir, "y_logret_24h")

    X_train = pd.read_csv(prepared_dir / "X_train.csv")
    assert "timestamp" not in X_train.columns
    assert "y_logret_24h" not in X_train.columns
    assert not any(col.startswith("y_") for col in X_train.columns), "Leakage columns in X_train"

    # Cutoff variant
    cutoff_dir = tmp_dir / "prepared_cutoff"
    cutoff_prepared = prepare_splits(
        input_path=merged_csv,
        output_dir=cutoff_dir,
        target="y_logret_24h",
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        cutoff_start="2024-01-03",
        cutoff_mid="2024-01-04",
    )
    meta = json.loads((cutoff_prepared / "prep_metadata.json").read_text())
    assert meta["split_strategy"].startswith("cutoff"), "Expected cutoff strategy metadata"


def _run_feature_store_checks(tmp_dir: Path) -> None:
    from model.prepare_training_data import prepare_splits_from_feature_store

    features_csv, targets_csv = _build_feature_store_csvs(tmp_dir)
    out_dir = tmp_dir / "prepared_feature_store"
    prepared_dir = prepare_splits_from_feature_store(
        features_csv=features_csv,
        targets_csv=targets_csv,
        output_dir=out_dir,
        target="y_logret_24h",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        cutoff_start=None,
        cutoff_mid=None,
    )

    _assert_basic_structure(prepared_dir, "y_logret_24h")

    X_train = pd.read_csv(prepared_dir / "X_train.csv")
    assert "timestamp" not in X_train.columns
    for col in ("close", "volume", "rsi"):
        assert col in X_train.columns, f"Missing feature column {col}"

    y_train = pd.read_csv(prepared_dir / "y_train.csv")
    assert "y_logret_24h" in y_train.columns
    assert y_train["y_logret_24h"].notna().all(), "Target contains NaNs"

    # Negative case: missing target column
    bad_targets = tmp_dir / "bad_targets.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=120, freq="h"),
            "y_other": 0.0,
        }
    ).to_csv(bad_targets, index=False)

    try:
        prepare_splits_from_feature_store(
            features_csv=features_csv,
            targets_csv=bad_targets,
            output_dir=out_dir,
            target="y_logret_24h",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
    except ValueError as exc:
        assert "Target column" in str(exc), f"Unexpected error message: {exc}"
    else:
        raise AssertionError("Expected ValueError for missing target column")


def main() -> None:
    _ensure_project_root_on_path()
    tmp_dir = _make_tmp_dir() / datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("[prepare_training_data] Running ratio/cutoff checks...")
    _run_prepare_splits_checks(tmp_dir / "merged")
    print("[prepare_training_data] Running feature store checks...")
    _run_feature_store_checks(tmp_dir / "feature_store")

    print("All prepare_training_data checks passed.")


if __name__ == "__main__":
    main()

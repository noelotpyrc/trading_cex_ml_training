#!/usr/bin/env python3
"""
Rolling training loop for pre-generated split.existing_dir folds.

Iterates fold directories under a rolling splits root, trains the model using
the existing LGBM pipeline, and logs each run to MLflow (optionally registers
model versions). No aggregation is performed here.

Usage example:

  python scripts/run_rolling_training.py \
    --config configs/model_configs/binance_btcusdt_perp_1h_since_2020_lgbm_y_binary4u2d_24h_tuning_selected_feature_rolling.json \
    --splits-root "/Volumes/Extreme SSD/trading_data/cex/training/binance_btcusdt_perp_1h_original/rolling/train48_val7d_test1m" \
    --tracking-uri http://127.0.0.1:5000 \
    --experiment btcusdt-1h-binary-rolling-48m-test1m-val7d

Register model versions (optional):

  python scripts/run_rolling_training.py \
    --config ... --splits-root ... --tracking-uri ... --experiment ... \
    --model-name lgbm-btcusdt-p60-rolling --register-model
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import List, Optional


def _ensure_repo_imports() -> None:
    import sys
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_repo_imports()

import importlib.util
import sys


def _load_local_train_register_module():
    """Load our repo's mlflow/train_and_register.py avoiding the 'mlflow' package name collision."""
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / "mlflow" / "train_and_register.py"
    spec = importlib.util.spec_from_file_location("local_train_and_register", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_train_and_register"] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _discover_folds(root: Path, start: Optional[str], end: Optional[str], reverse: bool) -> List[Path]:
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    # Keep only YYYY-MM dirs
    dirs = [p for p in dirs if len(p.name) == 7 and p.name[:4].isdigit() and p.name[4] == '-' and p.name[5:7].isdigit()]
    if start:
        dirs = [p for p in dirs if p.name >= start]
    if end:
        dirs = [p for p in dirs if p.name <= end]
    if reverse:
        dirs = list(reversed(dirs))
    return dirs


def _make_fold_config(base_cfg_path: Path, fold_dir: Path) -> Path:
    cfg = _load_json(base_cfg_path)
    # Ensure split.existing_dir is present and points to this fold
    split = cfg.get("split", {}) or {}
    split["existing_dir"] = str(fold_dir)
    cfg["split"] = split
    # Optional: embed fold context for traceability
    cfg.setdefault("_rolling", {})
    cfg["_rolling"]["fold_id"] = fold_dir.name
    cfg["_rolling"]["fold_dir"] = str(fold_dir)
    # Write to a temp file
    tmp = Path(tempfile.mkdtemp(prefix="rolling_cfg_")) / f"config_{fold_dir.name}.json"
    _write_json(tmp, cfg)
    return tmp


def run(args: argparse.Namespace) -> None:
    base_cfg_path = Path(args.config).resolve()
    splits_root = Path(args.splits_root).resolve()

    # Identify folds to run
    folds = _discover_folds(splits_root, args.start_fold, args.end_fold, args.reverse)
    if not folds:
        raise SystemExit(f"No fold dirs found under: {splits_root}")

    for i, fold_dir in enumerate(folds, 1):
        print(f"[{i}/{len(folds)}] Training fold {fold_dir.name} …")
        fold_cfg_path = _make_fold_config(base_cfg_path, fold_dir)

        # Train using existing pipeline
        tr_mod = _load_local_train_register_module()
        run_dir = tr_mod.train_lgbm_from_config(fold_cfg_path)

        # Attach rolling context into the run directory for MLflow artifacts
        context = {
            "fold_id": fold_dir.name,
            "fold_dir": str(fold_dir),
            "splits_root": str(splits_root),
        }
        _write_json(Path(run_dir) / "rolling_context.json", context)

        # MLflow logging / registration
        if args.register_model:
            if not args.model_name:
                raise SystemExit("--model-name is required when --register-model is set")
            # Use the registrar to register a model version
            tr_mod = _load_local_train_register_module()
            tr_mod.register_with_mlflow(
                run_dir,
                tracking_uri=args.tracking_uri,
                experiment=args.experiment,
                model_name=args.model_name,
                alias=None if not args.alias else args.alias,
                stage=(None if args.stage == "None" else args.stage),
                artifact_mode=args.artifact_mode,
                copy_model=bool(args.copy_model),
                sanitize_paths=bool(args.sanitize_paths),
            )
        else:
            # Log run only (no model registry)
            import subprocess, sys
            repo_root = Path(__file__).resolve().parent.parent
            registrar = repo_root / "mlflow" / "mlflow_register.py"
            cmd = [
                sys.executable,
                str(registrar),
                "--tracking-uri",
                args.tracking_uri,
                "--experiment",
                args.experiment,
                "--run-dir",
                str(run_dir),
                "--artifact-mode",
                args.artifact_mode,
                "--no-model-register",
            ]
            if args.copy_model:
                cmd.append("--copy-model")
            if args.sanitize_paths:
                cmd.append("--sanitize-paths")
            subprocess.run(cmd, check=True)

        # No aggregation in this script by design


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run rolling training across split.existing_dir folds")
    ap.add_argument("--config", required=True, type=Path, help="Base model config JSON")
    ap.add_argument("--splits-root", required=True, type=Path, help="Root directory containing YYYY-MM fold subdirectories")
    ap.add_argument("--start-fold", default=None, help="Earliest fold to include (YYYY-MM)")
    ap.add_argument("--end-fold", default=None, help="Latest fold to include (YYYY-MM)")
    ap.add_argument("--reverse", action="store_true", help="Run folds in reverse order")

    # MLflow
    ap.add_argument("--tracking-uri", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--model-name", default=None, help="Registered model name (optional if --register-model not set)")
    ap.add_argument("--register-model", action="store_true", help="Register a model version per fold")
    ap.add_argument("--alias", default=None)
    ap.add_argument("--stage", default="None", choices=["None", "Staging", "Production"])
    ap.add_argument("--artifact-mode", default="all", choices=["all", "subset"])
    ap.add_argument("--copy-model", action="store_true")
    ap.add_argument("--sanitize-paths", action="store_true")

    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())

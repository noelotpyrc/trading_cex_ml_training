#!/usr/bin/env python3
"""
End-to-end wrapper: Train a model (LGBM or HMM) from a config and register it to MLflow.

Usage (LGBM example):
  /Users/noel/projects/trading_cex/venv/bin/python scripts/train_and_register.py \
    --config /Users/noel/projects/trading_cex/configs/model_configs/binance_btcusdt_p60_huber_y_logret_168h.json \
    --tracking-uri http://127.0.0.1:5000 \
    --experiment cex-btcusdt-p60-lgbm \
    --model-name lgbm-btcusdt-p60 \
    --alias Staging

Usage (HMM example):
  /Users/noel/projects/trading_cex/venv/bin/python scripts/train_and_register.py \
    --config /Users/noel/projects/trading_cex/configs/model_configs/hmm_1d_all_features.json \
    --tracking-uri http://127.0.0.1:5000 \
    --experiment test-hmm \
    --model-name hmm-btcusdt-p60 \
    --alias Staging
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _read_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _detect_pipeline(config: Dict[str, Any]) -> str:
    model_cfg = config.get("model") or {}
    model_type = str(model_cfg.get("type") or "").lower()
    if model_type == "lgbm":
        return "lgbm"
    # HMM heuristic: common keys
    if any(k in model_cfg for k in ("n_states", "covariance_type", "n_iter")):
        return "hmm"
    # Fallback to lgbm if target/objective present
    if isinstance(config.get("target"), dict) and config["target"].get("objective") is not None:
        return "lgbm"
    return "hmm"


def train_lgbm_from_config(config_path: Path) -> Path:
    # Ensure project root is on sys.path for package imports
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from training import run_lgbm_pipeline as lgbm

    config = lgbm.load_config(config_path)
    # Record the original config file path so the registrar can log it as a param
    try:
        config["_config_path"] = str(Path(config_path).resolve())
    except Exception:
        config["_config_path"] = str(config_path)
    data_dir = lgbm.prepare_training_data(config)
    tuned_best_params = lgbm.tune_hyperparameters(config, data_dir)
    config["_tuned_best_params"] = tuned_best_params
    use_best_for_final = bool(config["model"].get("use_best_params_for_final", True))
    final_params = tuned_best_params if use_best_for_final else config["model"].get("params", {})
    _, metrics, run_dir = lgbm.train_model(config, data_dir, final_params)
    run_dir = lgbm.persist_results(config, run_dir, metrics, final_params, data_dir)
    return run_dir


def train_hmm_from_config(config_path: Path, log_level: str = "INFO") -> Path:
    # Ensure project root is on sys.path for package imports
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from training import run_hmm_pipeline as hmm

    cfg = hmm.load_config(config_path)
    # Record the original config file path so the registrar can log it as a param
    try:
        cfg["_config_path"] = str(Path(config_path).resolve())
    except Exception:
        cfg["_config_path"] = str(config_path)
    run_dir = hmm.run_hmm_pipeline(cfg, log_level)
    return run_dir


def register_with_mlflow(
    run_dir: Path,
    *,
    tracking_uri: str,
    experiment: str,
    model_name: str,
    alias: Optional[str] = None,
    stage: Optional[str] = None,
    artifact_mode: str = "all",
    copy_model: bool = True,
    sanitize_paths: bool = False,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    registrar = repo_root / "mlflow" / "mlflow_register.py"
    cmd = [
        sys.executable,
        str(registrar),
        "--tracking-uri",
        tracking_uri,
        "--experiment",
        experiment,
        "--run-dir",
        str(run_dir),
        "--model-name",
        model_name,
        "--artifact-mode",
        artifact_mode,
    ]
    if alias:
        cmd.extend(["--alias", alias])
    if stage:
        cmd.extend(["--stage", stage])
    if copy_model:
        cmd.append("--copy-model")
    if sanitize_paths:
        cmd.append("--sanitize-paths")

    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train model from config and register to MLflow")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--pipeline", choices=["auto", "lgbm", "hmm"], default="auto")
    ap.add_argument("--tracking-uri", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--alias", default=None)
    ap.add_argument("--stage", default=None, choices=["Staging", "Production", "None"])
    ap.add_argument("--artifact-mode", default="all", choices=["all", "subset"])
    ap.add_argument("--copy-model", action="store_true")
    ap.add_argument("--sanitize-paths", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--no-register", action="store_true", help="If set, only train; skip MLflow run logging and model registry")
    ap.add_argument("--log-run-only", action="store_true", help="Log run to MLflow tracking (params/metrics/artifacts) but do not register a model")
    args = ap.parse_args()

    config = _read_config(args.config)
    pipeline = args.pipeline if args.pipeline != "auto" else _detect_pipeline(config)

    if pipeline == "lgbm":
        run_dir = train_lgbm_from_config(args.config)
    elif pipeline == "hmm":
        run_dir = train_hmm_from_config(args.config, args.log_level)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline}")

    if not args.no_register and not args.log_run_only:
        if not args.model_name:
            raise SystemExit("--model-name is required unless --no-register or --log-run-only is set")
        register_with_mlflow(
            run_dir,
            tracking_uri=args.tracking_uri,
            experiment=args.experiment,
            model_name=args.model_name,
            alias=args.alias,
            stage=(None if args.stage == "None" else args.stage),
            artifact_mode=args.artifact_mode,
            copy_model=bool(args.copy_model),
            sanitize_paths=bool(args.sanitize_paths),
        )
    else:
        if args.no_register:
            print("Skipping MLflow run logging and model registry due to --no-register")
        else:
            # Log run only, no model registry
            repo_root = Path(__file__).resolve().parent.parent
            registrar = repo_root / "mlflow" / "mlflow_register.py"
            import subprocess, sys
            cmd = [
                sys.executable, str(registrar),
                "--tracking-uri", args.tracking_uri,
                "--experiment", args.experiment,
                "--run-dir", str(run_dir),
                "--artifact-mode", args.artifact_mode,
            ]
            if args.copy_model:
                cmd.append("--copy-model")
            if args.sanitize_paths:
                cmd.append("--sanitize-paths")
            cmd.append("--no-model-register")
            subprocess.run(cmd, check=True)
            print("Logged run only (no model registry) via registrar")

    print(f"Completed train+register: run_dir={run_dir}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Post-train MLflow registrar.

Reads a completed run directory, logs params/metrics/artifacts to MLflow, and
optionally registers/updates a model version and stage.

Requires an MLflow tracking server (local is fine):
  mlflow server --backend-store-uri sqlite:////PATH/mlflow.db \
    --default-artifact-root /PATH/mlflow_artifacts --host 127.0.0.1 --port 5000

Usage example:
  /Users/noel/projects/trading_cex/venv/bin/python scripts/mlflow_register.py \
    --tracking-uri http://127.0.0.1:5000 \
    --experiment cex-btcusdt-p60 \
    --run-dir "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/run_20251008_140235_lgbm_y_logret_168h_huber" \
    --model-name lgbm-btcusdt-p60 \
    --stage Staging
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a completed run to MLflow")
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--stage", default=None, choices=["Staging", "Production", "None"])
    parser.add_argument("--alias", default=None, help="Registered Model alias to set for this version (e.g., Staging, Production)")
    parser.add_argument("--copy-model", action="store_true", help="Log model artifact file into MLflow")
    parser.add_argument("--artifact-mode", default="all", choices=["all", "subset"], help="What to log as artifacts from run_dir")
    parser.add_argument("--sanitize-paths", action="store_true", help="If set, log artifacts from a sanitized path symlink (spaces/commas replaced)")
    parser.add_argument("--no-model-register", action="store_true", help="Log run only; skip creating a Registered Model version")
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    metrics_path = run_dir / "metrics.json"
    # Prefer unified pipeline_config.json; fallback to config.json (HMM)
    config_path = run_dir / "pipeline_config.json"
    if not config_path.exists():
        alt_cfg = run_dir / "config.json"
        config_path = alt_cfg if alt_cfg.exists() else config_path
    paths_path = run_dir / "paths.json"

    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    cfg = read_json(config_path) if config_path.exists() else {}
    paths = read_json(paths_path) if paths_path.exists() else {}

    # Prefer model.txt (LightGBM) then model.joblib
    model_file = None
    for cand in (run_dir / "model.txt", run_dir / "model.joblib"):
        if cand.exists():
            model_file = cand
            break

    # Derive tags and params
    tags: Dict[str, str] = {}
    params: Dict[str, Any] = {}
    # Record experiment name explicitly as a parameter for search/filter in UI
    # (helps when browsing runs across multiple experiments)
    params["experiment_name"] = str(args.experiment)
    # If the training wrapper embedded the original config path, record it
    cfg_path = cfg.get("_config_path")
    if cfg_path:
        params["config_path"] = str(cfg_path)
    # dataset slug from output_dir leaf if available; sanitize spaces/commas -> _
    out_dir = cfg.get("output_dir") or run_dir.parent
    dataset_leaf = Path(out_dir).name if out_dir else run_dir.parent.name
    dataset_slug = str(dataset_leaf).replace(",", "_").replace(" ", "_")
    params["dataset_slug"] = dataset_slug
    target = cfg.get("target", {}).get("variable") if isinstance(cfg.get("target"), dict) else None
    if target:
        params["target"] = str(target)
    else:
        params["target"] = "unsupervised target"
    model_type = (cfg.get("model", {}) or {}).get("type")
    if not model_type:
        m = (cfg.get("model", {}) or {})
        if any(k in m for k in ("n_states", "covariance_type", "n_iter")):
            model_type = "hmm"
    if model_type:
        params["model_type"] = str(model_type)
    # objective + primary_metric
    objective_name = None
    obj_cfg = None
    if isinstance(cfg.get("target"), dict):
        obj_cfg = cfg.get("target", {}).get("objective")
    if isinstance(obj_cfg, dict):
        objective_name = obj_cfg.get("name")
    elif isinstance(obj_cfg, str):
        objective_name = obj_cfg
    if objective_name:
        params["objective"] = str(objective_name)
        # Select primary_metric per objective family
        on = str(objective_name).lower()
        if on == "binary":
            params["primary_metric"] = "auc"
        elif on == "quantile":
            params["primary_metric"] = "pinball_loss"
        else:
            params["primary_metric"] = "rmse"
    # If no objective (e.g., HMM), prefer selection.criterion as primary_metric
    if "primary_metric" not in params:
        sel = (cfg.get("selection") or {}) if isinstance(cfg.get("selection"), dict) else {}
        if sel:
            crit = str(sel.get("criterion", "icl")).lower()
            params["primary_metric"] = crit
    # feature_selection params: include all fields if present
    feature_selection = cfg.get("feature_selection")
    if isinstance(feature_selection, dict):
        if feature_selection.get("include_files"):
            # Store as comma-separated list for readability
            include_files = feature_selection.get("include_files")
            params["feature_selection.include_files"] = ",".join(include_files) if isinstance(include_files, list) else str(include_files)
        if feature_selection.get("include"):
            # Store as comma-separated list for readability
            include = feature_selection.get("include")
            params["feature_selection.include"] = ",".join(include) if isinstance(include, list) else str(include)
        if feature_selection.get("include_patterns"):
            # Store as comma-separated list for readability
            include_patterns = feature_selection.get("include_patterns")
            params["feature_selection.include_patterns"] = ",".join(include_patterns) if isinstance(include_patterns, list) else str(include_patterns)
        if feature_selection.get("exclude"):
            # Store as comma-separated list for readability
            exclude = feature_selection.get("exclude")
            params["feature_selection.exclude"] = ",".join(exclude) if isinstance(exclude, list) else str(exclude)
    
    # attach run_dir and model_path as params for auditability
    params["run_dir"] = str(run_dir)
    if model_file is not None:
        params["model_path"] = str(model_file)
        # If user doesn't copy model into MLflow, expose external path
        if not args.copy_model:
            params["external_model_path"] = str(model_file)

    # Paths params (if available)
    if paths:
        if paths.get("prepared_data_dir"):
            params["prepared_data_dir"] = str(paths.get("prepared_data_dir"))
        if paths.get("output_dir"):
            params["output_dir"] = str(paths.get("output_dir"))
        if paths.get("input_data"):
            params["input_data"] = str(paths.get("input_data"))
        fs = paths.get("feature_store")
        if isinstance(fs, dict):
            if fs.get("features_csv"):
                params["feature_store.features_csv"] = str(fs.get("features_csv"))
            if fs.get("targets_csv"):
                params["feature_store.targets_csv"] = str(fs.get("targets_csv"))
        eff = paths.get("extra_feature_files")
        if isinstance(eff, list):
            for i, item in enumerate(eff):
                if isinstance(item, dict):
                    if item.get("path"):
                        params[f"extra_feature_files.{i}.path"] = str(item.get("path"))
                    if item.get("include") is not None:
                        # Store include patterns as a comma-separated list for readability
                        inc = item.get("include")
                        params[f"extra_feature_files.{i}.include"] = ",".join(inc) if isinstance(inc, list) else str(inc)
    else:
        # Fallback to config if paths.json was not produced
        if cfg.get("input_data"):
            params["input_data"] = str(cfg.get("input_data"))
        if cfg.get("output_dir"):
            params["output_dir"] = str(cfg.get("output_dir"))

    # hyperparam_tuning flag: LGBM via hyperparameter_tuning_method; HMM via state_grid
    mcfg = (cfg.get("model", {}) or {})
    hyperparam_tuning = bool(mcfg.get("hyperparameter_tuning_method")) or (mcfg.get("state_grid") is not None)
    params["hyperparam_tuning"] = int(hyperparam_tuning)

    # Import mlflow lazily
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import RestException

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    # Ensure registry uses same server
    mlflow.set_registry_uri(args.tracking_uri)

    with mlflow.start_run(run_name=run_dir.name) as run:
        # Log params/metrics
        # Flatten metrics (Dict[str, float]) and a few config params
        if metrics:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        # Log structured params
        if params:
            # log_params expects flat dict[str, str|int|float]
            flat_params = {k: (v if isinstance(v, (str, int, float)) else json.dumps(v)) for k, v in params.items()}
            mlflow.log_params(flat_params)
        # Persist tags
        mlflow.set_tags(tags)

        # Optionally log the model artifact into MLflow (copy)
        if args.copy_model and model_file is not None:
            mlflow.log_artifact(str(model_file), artifact_path="model")

        # Log artifacts from the run directory per mode (default: all)
        try:
            if args.artifact_mode == "all":
                src_dir = run_dir
                if args.sanitize_paths:
                    try:
                        sanitized_name = run_dir.name.replace(",", "_").replace(" ", "_")
                        san_dir = run_dir.parent / sanitized_name
                        # Create symlink if not exists
                        if not san_dir.exists():
                            os.symlink(run_dir, san_dir)
                        src_dir = san_dir
                    except Exception:
                        src_dir = run_dir
                mlflow.log_artifacts(str(src_dir))
            else:
                # subset: log a minimal set for discoverability
                for fname in ("metrics.json", "pipeline_config.json", "config.json", "run_metadata.json", "feature_importance.csv", "regimes.csv"):
                    fp = run_dir / fname
                    if fp.exists():
                        mlflow.log_artifact(str(fp))
        except Exception:
            # Artifacts logging is best-effort; continue
            pass

        run_id = run.info.run_id

    # Optionally skip model registry
    if args.no_model_register:
        print(f"Logged MLflow run only (no model registry). run_id={run_id}")
        return

    # Register model version
    if not args.model_name:
        raise SystemExit("--model-name is required unless --no-model-register is set")
    client = MlflowClient(tracking_uri=args.tracking_uri)

    # Ensure the Registered Model exists
    try:
        client.get_registered_model(name=args.model_name)
    except Exception:
        try:
            client.create_registered_model(name=args.model_name)
        except RestException:
            # Race or permissions; proceed and let create_model_version fail if still missing
            pass

    # Prefer artifact model if copied, else just register a dummy with params/tags
    model_uri = None
    if args.copy_model and model_file is not None:
        model_uri = f"runs:/{run_id}/model"
    else:
        # Register from run itself (no model directory). Still creates a version with metadata.
        model_uri = f"runs:/{run_id}"

    mv = client.create_model_version(name=args.model_name, source=model_uri, run_id=run_id)

    # Set alias if provided; otherwise optionally transition stage (legacy)
    if args.alias:
        try:
            client.set_registered_model_alias(name=args.model_name, alias=str(args.alias), version=mv.version)
        except Exception:
            # If alias API not available, ignore silently
            pass
    elif args.stage in {"Staging", "Production"}:
        try:
            client.transition_model_version_stage(
                name=args.model_name, version=mv.version, stage=args.stage
            )
        except Exception:
            pass

    print(f"Registered MLflow model '{args.model_name}' version {mv.version} from run {run_id}")


if __name__ == "__main__":
    main()

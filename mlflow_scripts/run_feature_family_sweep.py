#!/usr/bin/env python3
"""
Run an exhaustive feature-family sweep (all combinations) and log runs to MLflow.

This script:
- Loads a base config
- Adds one or more feature families to feature_selection.include
- Logs a `combo_families` param for MLflow filtering
- Runs train_and_register.py with --log-run-only
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_FAMILIES: Dict[str, List[str]] = {
    "adx": ["adx_168", "adx_24", "adx_720"],
    "cvar_var_ratio": ["cvar_var_ratio_168", "cvar_var_ratio_720"],
    "efficiency_avg": ["efficiency_avg_168", "efficiency_avg_24"],
    "mean_cross_rate_ema": [
        "mean_cross_rate_ema_168_168",
        "mean_cross_rate_ema_168_48",
        "mean_cross_rate_ema_24_168",
        "mean_cross_rate_ema_24_48",
        "mean_cross_rate_ema_720_168",
        "mean_cross_rate_ema_720_48",
    ],
    "parkinson_volatility": ["parkinson_volatility_168", "parkinson_volatility_24"],
    "price_vwap_distance_zscore": [
        "price_vwap_distance_zscore_168_168",
        "price_vwap_distance_zscore_24_168",
        "price_vwap_distance_zscore_720_168",
    ],
    "pullback_slope_vwap": [
        "pullback_slope_vwap_168_168",
        "pullback_slope_vwap_168_48",
        "pullback_slope_vwap_24_168",
        "pullback_slope_vwap_24_48",
        "pullback_slope_vwap_720_168",
        "pullback_slope_vwap_720_48",
    ],
    "range_stretch_interaction": [
        "range_stretch_interaction_168_168",
        "range_stretch_interaction_168_24",
        "range_stretch_interaction_720_168",
        "range_stretch_interaction_720_24",
    ],
    "relative_volume": ["relative_volume_30", "relative_volume_7"],
    "return_autocorr": ["return_autocorr_168", "return_autocorr_48"],
    "rsi": ["rsi_168", "rsi_24", "rsi_720"],
    "scaled_acceleration": ["scaled_acceleration_168"],
    "variance_ratio": ["variance_ratio_24_168", "variance_ratio_24_48", "variance_ratio_24_720"],
}


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _load_family_map(path: Path | None) -> Dict[str, List[str]]:
    if path is None:
        return DEFAULT_FAMILIES
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError("families JSON must be an object: {family: [features...]}")
    fams: Dict[str, List[str]] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            raise ValueError("family name must be a string")
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ValueError(f"family '{k}' must be a list of strings")
        fams[k] = v
    return fams


def _combo_count(n: int, min_k: int, max_k: int) -> int:
    return sum(math.comb(n, k) for k in range(min_k, max_k + 1))


def _apply_combo(
    base_cfg: dict,
    base_features: Sequence[str],
    families: Dict[str, List[str]],
    combo: Sequence[str],
) -> Tuple[dict, List[str], str]:
    cfg = copy.deepcopy(base_cfg)
    feature_selection = cfg.get("feature_selection")
    if not isinstance(feature_selection, dict):
        feature_selection = {}
        cfg["feature_selection"] = feature_selection

    include = list(base_features)
    seen = set(include)
    for fam in combo:
        for feat in families[fam]:
            if feat not in seen:
                include.append(feat)
                seen.add(feat)
    feature_selection["include"] = include
    combo_str = ",".join(combo)
    cfg["combo_families"] = combo_str
    cfg["combo_size"] = len(combo)
    return cfg, include, combo_str


def _iter_combos(names: Sequence[str], min_k: int, max_k: int) -> Iterable[Tuple[str, ...]]:
    for k in range(min_k, max_k + 1):
        for combo in combinations(names, k):
            yield combo


def _iter_feature_combos(features: Sequence[str], min_k: int, max_k: int) -> Iterable[Tuple[str, ...]]:
    for k in range(min_k, max_k + 1):
        for combo in combinations(features, k):
            yield combo


def _apply_single_family_combo(
    base_cfg: dict,
    base_features: Sequence[str],
    family: str,
    family_features: Sequence[str],
    combo_features: Sequence[str],
) -> Tuple[dict, List[str], str]:
    cfg = copy.deepcopy(base_cfg)
    feature_selection = cfg.get("feature_selection")
    if not isinstance(feature_selection, dict):
        feature_selection = {}
        cfg["feature_selection"] = feature_selection

    base_without = [f for f in base_features if f not in set(family_features)]
    include = list(base_without)
    seen = set(include)
    for feat in combo_features:
        if feat not in seen:
            include.append(feat)
            seen.add(feat)

    feature_selection["include"] = include
    combo_str = ",".join(combo_features)
    cfg["combo_families"] = family
    cfg["combo_family"] = family
    cfg["combo_features"] = list(combo_features)
    cfg["combo_size"] = len(combo_features)
    return cfg, include, combo_str


def main() -> None:
    ap = argparse.ArgumentParser(description="Run exhaustive feature-family sweep")
    ap.add_argument("--config", type=Path, required=True, help="Base config JSON path")
    ap.add_argument("--experiment", required=True, help="MLflow experiment name")
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    ap.add_argument("--families-json", type=Path, default=None, help="Optional JSON: {family: [features...]}")
    ap.add_argument("--min-k", type=int, default=1, help="Minimum family count")
    ap.add_argument("--max-k", type=int, default=None, help="Maximum family count")
    ap.add_argument("--sleep-secs", type=float, default=0.0, help="Sleep between runs")
    ap.add_argument("--existing-dir", type=Path, default=None, help="Reuse prepared splits dir for all runs")
    ap.add_argument("--single-family-combos", action="store_true", help="Run full feature combos within each family")
    ap.add_argument("--sweep-id", default=None, help="Optional sweep identifier for output folder")
    ap.add_argument("--output-dir", type=Path, default=None, help="Where to write generated configs")
    ap.add_argument("--dry-run", action="store_true", help="Print plan only, no training")
    ap.add_argument("--skip", type=int, default=0, help="Skip first N combos")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N combos")
    args = ap.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    families = _load_family_map(args.families_json)
    family_names = list(families.keys())
    if not family_names:
        raise ValueError("No families defined")

    min_k = max(1, args.min_k)
    max_k = args.max_k if args.max_k is not None else len(family_names)
    if max_k < min_k:
        raise ValueError("--max-k must be >= --min-k")
    if not args.single_family_combos and max_k > len(family_names):
        raise ValueError("--max-k cannot exceed number of families")

    base_cfg = _load_json(args.config)
    if args.existing_dir is not None:
        if not args.existing_dir.exists():
            raise FileNotFoundError(f"--existing-dir not found: {args.existing_dir}")
        split_cfg = base_cfg.get("split")
        if not isinstance(split_cfg, dict):
            split_cfg = {}
            base_cfg["split"] = split_cfg
        split_cfg["existing_dir"] = str(args.existing_dir)
    fs = base_cfg.get("feature_selection") or {}
    base_features = fs.get("include") or []
    if not isinstance(base_features, list):
        raise ValueError("feature_selection.include must be a list in base config")

    if args.single_family_combos:
        total = 0
        for fam in family_names:
            fam_size = len(families[fam])
            fam_max_k = min(max_k, fam_size)
            total += _combo_count(fam_size, min_k, fam_max_k)
    else:
        total = _combo_count(len(family_names), min_k, max_k)
    sweep_id = args.sweep_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (Path(__file__).resolve().parent / "_sweep_configs" / sweep_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep: families={len(family_names)} combos={total} min_k={min_k} max_k={max_k}")
    print(f"Configs: {out_dir}")
    if args.dry_run:
        return

    wrapper = Path(__file__).resolve().parent / "train_and_register.py"
    run_count = 0

    if args.single_family_combos:
        combo_iter: Iterable[Tuple[str, Tuple[str, ...]]] = (
            (fam, combo)
            for fam in family_names
            for combo in _iter_feature_combos(
                families[fam],
                min_k,
                min(max_k, len(families[fam])),
            )
        )
    else:
        combo_iter = ((None, combo) for combo in _iter_combos(family_names, min_k, max_k))

    for idx, (fam, combo) in enumerate(combo_iter, start=1):
        if args.skip and idx <= args.skip:
            continue
        if args.limit is not None and run_count >= args.limit:
            break

        if args.single_family_combos:
            cfg, _, combo_str = _apply_single_family_combo(
                base_cfg,
                base_features,
                fam,
                families[fam],
                combo,
            )
            combo_hash = hashlib.sha1(f"{fam}:{combo_str}".encode("utf-8")).hexdigest()[:10]
            safe_fam = fam.replace(" ", "_")
            cfg_path = out_dir / f"combo_{idx:05d}_{safe_fam}_k{len(combo)}_{combo_hash}.json"
            print(f"[{idx}/{total}] combo_family={fam} combo_features={combo_str}")
        else:
            cfg, _, combo_str = _apply_combo(base_cfg, base_features, families, combo)
            combo_hash = hashlib.sha1(combo_str.encode("utf-8")).hexdigest()[:10]
            cfg_path = out_dir / f"combo_{idx:05d}_k{len(combo)}_{combo_hash}.json"
            print(f"[{idx}/{total}] combo_families={combo_str}")
        _write_json(cfg_path, cfg)
        cmd = [
            sys.executable,
            str(wrapper),
            "--config",
            str(cfg_path),
            "--tracking-uri",
            args.tracking_uri,
            "--experiment",
            args.experiment,
            "--log-run-only",
            "--sanitize-paths",
        ]
        subprocess.run(cmd, check=True)
        run_count += 1

        if args.sleep_secs > 0:
            time.sleep(args.sleep_secs)

    print(f"Completed {run_count} run(s).")


if __name__ == "__main__":
    main()

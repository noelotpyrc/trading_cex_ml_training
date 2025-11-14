#!/usr/bin/env python3
import json
from pathlib import Path
import sys


def main() -> None:
    # Ensure project root on path
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from model.run_lgbm_pipeline import load_config

    tmp_dir = root / ".tmp" / "tests" / "configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Legacy quantiles format -> normalized to {name: quantile, params: {alpha}}
    cfg_legacy = {
        "input_data": str(root / "data" / "BINANCE_BTCUSDT.P, 60.csv"),
        "output_dir": str(root / ".tmp" / "tests" / "models"),
        "target": {
            "variable": "y_logret_24h",
            "objective": {"type": "quantiles", "quantiles": [0.05]}
        },
        "split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
        "model": {"type": "lgbm", "params": {}}
    }
    p1 = tmp_dir / "legacy_quantile.json"
    p1.write_text(json.dumps(cfg_legacy))
    cfg1 = load_config(p1)
    print("legacy->normalized objective:", cfg1["target"]["objective"])  # expect name=quantile, params.alpha=0.05

    # Native string objective
    cfg_str = {
        "input_data": str(root / "data" / "BINANCE_BTCUSDT.P, 60.csv"),
        "output_dir": str(root / ".tmp" / "tests" / "models"),
        "target": {"variable": "y_logret_24h", "objective": "regression"},
        "split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
        "model": {"type": "lgbm", "params": {}}
    }
    p2 = tmp_dir / "string_regression.json"
    p2.write_text(json.dumps(cfg_str))
    cfg2 = load_config(p2)
    print("string objective normalized:", cfg2["target"]["objective"])  # expect name=regression

    # Invalid legacy with multiple quantiles should raise
    cfg_bad = cfg_legacy.copy()
    cfg_bad = json.loads(json.dumps(cfg_bad))
    cfg_bad["target"]["objective"]["quantiles"] = [0.05, 0.95]
    p3 = tmp_dir / "bad_legacy.json"
    p3.write_text(json.dumps(cfg_bad))
    try:
        _ = load_config(p3)
        print("ERROR: expected ValueError for multiple quantiles")
    except ValueError as e:
        print("ok: raised ValueError for multiple quantiles ->", str(e))


if __name__ == "__main__":
    main()



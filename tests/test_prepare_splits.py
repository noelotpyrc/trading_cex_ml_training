#!/usr/bin/env python3
from pathlib import Path
import sys
import json


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Generate small synthetic dataset if missing
    data_path = Path("/Volumes/Extreme SSD/trading_data/cex/tests/synth.csv")
    if not data_path.exists():
        from utils.generate_synthetic_data import main as gen
        sys.argv = ["gen", "--output", str(data_path), "--rows", "120"]
        gen()

    from model.prepare_training_data import prepare_splits

    out_base = Path("/Volumes/Extreme SSD/trading_data/cex/tests/prepared")
    target = "y_logret_24h"
    prepared_dir = prepare_splits(
        input_path=data_path,
        output_dir=out_base,
        target=target,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        cutoff_start=None,
        cutoff_mid=None,
    )
    print("prepared_dir:", prepared_dir)
    meta = json.load(open(prepared_dir / "prep_metadata.json", "r"))
    print("meta target:", meta["target_column"])  # expect y_logret_24h
    print("files:", sorted([p.name for p in prepared_dir.glob("*.csv")]))


if __name__ == "__main__":
    main()



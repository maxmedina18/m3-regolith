from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lunar_m3.data_loading import load_m3_cube
from lunar_m3.features import extract_feature_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an M3 cube (.IMG/.HDR).",
    )
    parser.add_argument("--output", type=str, default="data/processed/features.parquet")
    args = parser.parse_args()

    cube = load_m3_cube(args.input)
    features = extract_feature_table(cube.data, cube.wavelengths)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        features.to_parquet(out_path, index=False)
    except Exception:
        features.to_csv(out_path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()

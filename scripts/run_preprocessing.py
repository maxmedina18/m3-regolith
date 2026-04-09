from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lunar_m3.data_loading import load_m3_cube
from lunar_m3.preprocessing import clip_invalid_reflectance, normalize_by_reference_window, savgol_smooth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an M3 cube (.IMG/.HDR).",
    )
    parser.add_argument("--output", type=str, default="data/processed/processed_cube.npz")
    args = parser.parse_args()

    cube = load_m3_cube(args.input)
    rows, cols, bands = cube.data.shape
    processed = np.zeros((rows, cols, bands), dtype=np.float32)

    for y in range(rows):
        for x in range(cols):
            r = clip_invalid_reflectance(cube.data[y, x, :])
            r = normalize_by_reference_window(cube.wavelengths, r)
            r = savgol_smooth(r)
            processed[y, x, :] = r.astype(np.float32)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, data=processed, wavelengths=cube.wavelengths)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from lunar_m3.data_loading import load_m3_cube
from lunar_m3.features import extract_feature_table
from lunar_m3.preprocessing import (
    clip_invalid_reflectance,
    normalize_by_reference_window,
    savgol_smooth,
)
from lunar_m3.validation import check_band_centers_plausible
from lunar_m3.visualization import plot_spectrum_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an M3 cube (.IMG/.HDR).",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args()

    cube = load_m3_cube(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = extract_feature_table(cube.data, cube.wavelengths)
    check_band_centers_plausible(features)

    rows, cols, _ = cube.data.shape

    x0, y0 = cols // 2, rows // 2
    raw = cube.get_pixel_spectrum(x0, y0)
    proc = clip_invalid_reflectance(raw)
    proc = normalize_by_reference_window(cube.wavelengths, proc)
    proc = savgol_smooth(proc)
    plot_spectrum_comparison(
        cube.wavelengths,
        raw,
        proc,
        title=f"Pixel spectrum (x={x0}, y={y0})",
        output_path=out_dir / "spectrum_example.png",
    )

    features.to_csv(out_dir / "features.csv", index=False)


if __name__ == "__main__":
    main()

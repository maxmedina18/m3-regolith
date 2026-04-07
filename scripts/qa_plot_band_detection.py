from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lunar_m3.data_loading import load_m3_cube
from lunar_m3.features import band_depth, detect_absorption_band
from lunar_m3.preprocessing import (
    clip_invalid_reflectance,
    continuum_remove_linear,
    mitigate_instrument_join,
    normalize_by_reference_window,
    savgol_smooth,
)


def _prep_spectrum(w_um: np.ndarray, r_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = clip_invalid_reflectance(r_raw)
    r = normalize_by_reference_window(w_um, r, window_um=(1.45, 1.55))
    r = savgol_smooth(r)
    r, join_mask = mitigate_instrument_join(w_um, r, join_region_um=(1.325, 1.355), mode="interpolate")
    return r_raw, r, join_mask


def _analyze_band(
    w_um: np.ndarray,
    r_proc: np.ndarray,
    *,
    search_um: tuple[float, float],
    join_region_um: tuple[float, float],
) -> dict[str, float]:
    r_cr_seed, _ = continuum_remove_linear(w_um, r_proc, region_um=search_um)
    det = detect_absorption_band(w_um, r_cr_seed, search_region_um=search_um, join_region_um=join_region_um)

    left = float(det["band_left_um"]) if np.isfinite(float(det["band_left_um"])) else float(search_um[0])
    right = float(det["band_right_um"]) if np.isfinite(float(det["band_right_um"])) else float(search_um[1])
    region = (left, right)

    r_cr, _ = continuum_remove_linear(w_um, r_proc, region_um=region)
    det["bd_final"] = band_depth(w_um, r_cr, region_um=region)
    det["region_left"] = region[0]
    det["region_right"] = region[1]
    return det


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "QA tool: sample random pixels, plot spectra, and overlay detected ~1 µm and ~2 µm band geometry. "
            "Use this for visual inspection on synthetic or real M3 cubes."
        )
    )
    parser.add_argument("--input", type=str, default="synthetic", help="Cube path (.npz or ENVI .IMG/.HDR) or 'synthetic'.")
    parser.add_argument("--output-dir", type=str, default="artifacts_qa", help="Directory where plots are saved.")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of pixels to sample.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cube = load_m3_cube(args.input)
    rows, cols, _ = cube.data.shape

    rng = np.random.default_rng(args.seed)
    ys = rng.integers(0, rows, size=args.num_samples)
    xs = rng.integers(0, cols, size=args.num_samples)

    w_um = np.asarray(cube.wavelengths, dtype=float)
    join_region_um = (1.325, 1.355)
    band1_search_um = (0.85, 1.30)
    band2_search_um = (1.70, 2.50)

    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        r_raw = cube.get_pixel_spectrum(int(x), int(y))
        raw, proc, join_mask = _prep_spectrum(w_um, r_raw)
        b1 = _analyze_band(w_um, proc, search_um=band1_search_um, join_region_um=join_region_um)
        b2 = _analyze_band(w_um, proc, search_um=band2_search_um, join_region_um=join_region_um)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(w_um, raw, lw=1.0, alpha=0.6, label="raw")
        ax.plot(w_um, proc, lw=1.5, label="processed")

        ax.axvspan(join_region_um[0], join_region_um[1], color="gray", alpha=0.15, label="~1.34 µm join")

        for det, color, name in [(b1, "tab:blue", "~1 µm"), (b2, "tab:orange", "~2 µm")]:
            ax.axvline(det["band_center_um"], color=color, ls="--", lw=1.5)
            ax.axvspan(det["region_left"], det["region_right"], color=color, alpha=0.10)
            ax.text(
                det["band_center_um"],
                float(np.nanmax(proc)),
                f"{name} center={det['band_center_um']:.3f}µm\nBD={det['bd_final']:.3f}",
                color=color,
                fontsize=9,
                va="top",
                ha="center",
            )

        ax.set_title(f"Band detection QA (x={int(x)}, y={int(y)})")
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Reflectance (normalized)")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")

        out_path = out_dir / f"qa_band_detection_{i:03d}_x{int(x)}_y{int(y)}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()


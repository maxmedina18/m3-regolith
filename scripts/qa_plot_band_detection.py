from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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


def _finite_percentile(values: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.percentile(v, q))


def _choose_ylim(raw: np.ndarray, processed: np.ndarray) -> tuple[float, float]:
    lo = min(_finite_percentile(raw, 1.0), _finite_percentile(processed, 1.0))
    hi = max(_finite_percentile(raw, 99.0), _finite_percentile(processed, 99.0))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return (0.0, 1.0)
    if hi <= lo:
        return (lo - 1.0, lo + 1.0)
    pad = 0.08 * (hi - lo)
    return (lo - pad, hi + pad)


def _annotate_band(
    ax: plt.Axes,
    *,
    center_um: float,
    depth: float,
    color: str,
    label: str,
    y_text: float,
) -> None:
    ax.axvline(center_um, color=color, ls="--", lw=1.7, alpha=0.9)
    ax.text(
        center_um,
        y_text,
        f"{label} center={center_um:.3f} µm\nBD={depth:.3f}",
        fontsize=10,
        color=color,
        ha="center",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": color,
            "alpha": 0.85,
        },
    )


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
    parser.add_argument("--input", type=str, required=True, help="Cube path (.npz or ENVI .IMG/.HDR).")
    parser.add_argument("--output-dir", type=str, default="artifacts_qa", help="Directory where plots are saved.")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of pixels to sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x", type=int, default=None, help="Optional fixed x pixel index.")
    parser.add_argument("--y", type=int, default=None, help="Optional fixed y pixel index.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cube = load_m3_cube(args.input)
    rows, cols, _ = cube.data.shape

    if args.x is not None or args.y is not None:
        if args.x is None or args.y is None:
            raise ValueError("If specifying a fixed pixel, both --x and --y must be provided")
        xs = np.array([int(args.x)], dtype=int)
        ys = np.array([int(args.y)], dtype=int)
    else:
        rng = np.random.default_rng(args.seed)
        ys = rng.integers(0, rows, size=args.num_samples)
        xs = rng.integers(0, cols, size=args.num_samples)

    w_um = np.asarray(cube.wavelengths, dtype=float)
    join_region_um = (1.325, 1.355)
    band1_search_um = (0.85, 1.30)
    band2_search_um = (1.70, 2.50)

    input_label = Path(args.input).name

    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        if not (0 <= int(x) < cols and 0 <= int(y) < rows):
            raise ValueError(f"Pixel (x={int(x)}, y={int(y)}) out of bounds for cube shape (rows={rows}, cols={cols})")
        r_raw = cube.get_pixel_spectrum(int(x), int(y))
        raw, proc, join_mask = _prep_spectrum(w_um, r_raw)
        b1 = _analyze_band(w_um, proc, search_um=band1_search_um, join_region_um=join_region_um)
        b2 = _analyze_band(w_um, proc, search_um=band2_search_um, join_region_um=join_region_um)

        fig, ax = plt.subplots(figsize=(12, 6))

        raw_plot = np.asarray(raw, dtype=float).copy()
        raw_plot[raw_plot <= -900.0] = np.nan

        ax.plot(w_um, raw_plot, lw=1.0, alpha=0.35, color="0.35")
        ax.plot(w_um, proc, lw=2.2, color="black")

        ax.axvspan(join_region_um[0], join_region_um[1], color="gray", alpha=0.12)
        ax.axvspan(band1_search_um[0], band1_search_um[1], color="tab:blue", alpha=0.06)
        ax.axvspan(band2_search_um[0], band2_search_um[1], color="tab:orange", alpha=0.06)

        ax.text(
            0.5 * (band1_search_um[0] + band1_search_um[1]),
            0.98,
            "~1 µm search",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            color="tab:blue",
        )
        ax.text(
            0.5 * (band2_search_um[0] + band2_search_um[1]),
            0.98,
            "~2 µm search",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            color="tab:orange",
        )

        ax.axvspan(b1["region_left"], b1["region_right"], color="tab:blue", alpha=0.12)
        ax.axvspan(b2["region_left"], b2["region_right"], color="tab:orange", alpha=0.12)

        y0, y1 = _choose_ylim(raw_plot, proc)
        ax.set_ylim(y0, y1)
        y_text = y1 - 0.04 * (y1 - y0)
        _annotate_band(
            ax,
            center_um=float(b1["band_center_um"]),
            depth=float(b1["bd_final"]),
            color="tab:blue",
            label="~1 µm",
            y_text=y_text,
        )
        _annotate_band(
            ax,
            center_um=float(b2["band_center_um"]),
            depth=float(b2["bd_final"]),
            color="tab:orange",
            label="~2 µm",
            y_text=y_text,
        )

        ax.set_title(f"M³ band detection QA — {input_label} — pixel (x={int(x)}, y={int(y)})")
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Reflectance (normalized)")
        ax.grid(True, alpha=0.18)

        legend_handles = [
            Line2D([0], [0], color="0.35", lw=1.2, alpha=0.6, label="raw (masked)") ,
            Line2D([0], [0], color="black", lw=2.4, label="processed"),
            Patch(facecolor="gray", alpha=0.12, label="~1.34 µm join"),
            Patch(facecolor="tab:blue", alpha=0.12, label="~1 µm detected band"),
            Patch(facecolor="tab:orange", alpha=0.12, label="~2 µm detected band"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

        out_path = out_dir / f"qa_band_detection_{i:03d}_x{int(x)}_y{int(y)}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()

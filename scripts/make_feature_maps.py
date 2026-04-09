from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _grid_from_feature_table(df: pd.DataFrame, *, value_col: str, rows: int, cols: int) -> np.ndarray:
    grid = np.full((rows, cols), np.nan, dtype=float)
    xy = df[["x", "y", value_col]].copy()
    xy = xy.dropna(subset=["x", "y"])
    x = xy["x"].astype(int).to_numpy()
    y = xy["y"].astype(int).to_numpy()
    v = xy[value_col].astype(float).to_numpy()
    ok = (x >= 0) & (x < cols) & (y >= 0) & (y < rows)
    grid[y[ok], x[ok]] = v[ok]
    return grid


def _finite_percentile(arr: np.ndarray, q: float) -> float:
    v = np.asarray(arr, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.percentile(v, q))


def _plot_map(
    grid: np.ndarray,
    *,
    title: str,
    output_path: Path,
    cmap: str,
    cbar_label: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(grid, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("x (sample)")
    ax.set_ylabel("y (line)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 2D spatial maps from an extracted feature table (features.csv). "
            "This reconstructs features back onto the pixel grid and saves publication-style PNGs."
        )
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        required=True,
        help="Path to a features table produced by run_pipeline.py (must include x,y columns).",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts_feature_maps")
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Optional number of rows (y). If omitted, inferred from max(y)+1.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Optional number of cols (x). If omitted, inferred from max(x)+1.",
    )
    parser.add_argument(
        "--clip-percentiles",
        type=str,
        default="1,99",
        help="Percentile clipping for color scales, e.g. '1,99' or '0,100' to disable.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    df = pd.read_csv(args.features_csv)
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError("features table must include 'x' and 'y' columns")

    rows = int(args.rows) if args.rows is not None else int(df["y"].max()) + 1
    cols = int(args.cols) if args.cols is not None else int(df["x"].max()) + 1

    p_lo, p_hi = [float(x.strip()) for x in args.clip_percentiles.split(",")]

    specs: list[dict[str, str]] = [
        {"col": "bd_1um", "title": "Band depth (~1 µm)", "cmap": "magma", "label": "BD (unitless)"},
        {"col": "bc_1um", "title": "Band center (~1 µm)", "cmap": "viridis", "label": "Center (µm)"},
        {"col": "bd_2um", "title": "Band depth (~2 µm)", "cmap": "magma", "label": "BD (unitless)"},
        {"col": "bc_2um", "title": "Band center (~2 µm)", "cmap": "viridis", "label": "Center (µm)"},
        {"col": "slope_global", "title": "Spectral slope (0.70–2.50 µm)", "cmap": "coolwarm", "label": "dR/dλ"},
        {"col": "brightness_ref", "title": "Brightness reference (1.45–1.55 µm)", "cmap": "gray", "label": "Reflectance"},
    ]

    for spec in specs:
        col = spec["col"]
        if col not in df.columns:
            raise ValueError(f"Missing expected feature column: {col}")

        grid = _grid_from_feature_table(df, value_col=col, rows=rows, cols=cols)

        if col == "slope_global":
            vmax0 = _finite_percentile(np.abs(grid), p_hi)
            vmin = -vmax0 if np.isfinite(vmax0) else None
            vmax = vmax0 if np.isfinite(vmax0) else None
        else:
            vmin0 = _finite_percentile(grid, p_lo)
            vmax0 = _finite_percentile(grid, p_hi)
            vmin = vmin0 if np.isfinite(vmin0) else None
            vmax = vmax0 if np.isfinite(vmax0) else None

        out_path = out_dir / f"map_{col}.png"
        _plot_map(
            grid,
            title=spec["title"],
            output_path=out_path,
            cmap=spec["cmap"],
            cbar_label=spec["label"],
            vmin=vmin,
            vmax=vmax,
        )


if __name__ == "__main__":
    main()


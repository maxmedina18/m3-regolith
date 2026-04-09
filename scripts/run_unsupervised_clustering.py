from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lunar_m3.models.gmm_clustering import fit_gmm_clusters, interpretation_scaffold, summarize_clusters


def _grid_shape_from_xy(df: pd.DataFrame, *, rows: int | None, cols: int | None) -> tuple[int, int]:
    if rows is not None and cols is not None:
        return int(rows), int(cols)
    inferred_rows = int(df["y"].max()) + 1
    inferred_cols = int(df["x"].max()) + 1
    return inferred_rows, inferred_cols


def _grid_from_xy_values(df: pd.DataFrame, *, value_col: str, rows: int, cols: int) -> np.ndarray:
    grid = np.full((rows, cols), np.nan, dtype=float)
    x = df["x"].astype(int).to_numpy()
    y = df["y"].astype(int).to_numpy()
    v = df[value_col].astype(float).to_numpy()
    ok = (x >= 0) & (x < cols) & (y >= 0) & (y < rows)
    grid[y[ok], x[ok]] = v[ok]
    return grid


def _finite_percentile(arr: np.ndarray, q: float) -> float:
    v = np.asarray(arr, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.percentile(v, q))


def _save_cluster_map(labels_grid: np.ndarray, *, out_path: Path, title: str) -> None:
    plt.rcParams.update({"savefig.dpi": 300, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(labels_grid, interpolation="nearest", cmap="tab20", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("x (sample)")
    ax.set_ylabel("y (line)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cluster")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _save_probability_panels(
    prob_grids: list[np.ndarray],
    *,
    out_path: Path,
    title: str,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> None:
    k = len(prob_grids)
    ncols = int(np.ceil(np.sqrt(k)))
    nrows = int(np.ceil(k / ncols))

    plt.rcParams.update({"savefig.dpi": 300, "figure.dpi": 150})
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3.5 * nrows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)

    for i in range(nrows * ncols):
        ax = axes_arr[i]
        if i >= k:
            ax.axis("off")
            continue

        grid = prob_grids[i]
        vmin = _finite_percentile(grid, clip_percentiles[0])
        vmax = _finite_percentile(grid, clip_percentiles[1])
        im = ax.imshow(grid, interpolation="nearest", cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"P(cluster={i})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unsupervised clustering (GMM) on physics-informed spectral features. "
            "Reads a features.csv (with x,y), fits a Gaussian Mixture Model, and outputs cluster maps, probability maps, and summaries."
        )
    )
    parser.add_argument("--features-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="artifacts_clusters")
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--feature-cols",
        type=str,
        default="bd_1um,bc_1um,bd_2um,bc_2um,slope_global,brightness_ref",
        help="Comma-separated feature columns to use for clustering.",
    )
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv)
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError("features table must include x and y columns")

    feature_columns = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    valid = df[["x", "y"] + feature_columns].copy()
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if valid.shape[0] == 0:
        raise ValueError("No finite rows remain after dropping NaNs")

    rows, cols = _grid_shape_from_xy(valid, rows=args.rows, cols=args.cols)

    fit = fit_gmm_clusters(
        valid,
        feature_columns=feature_columns,
        n_clusters=int(args.n_clusters),
        random_state=int(args.seed),
    )

    clustered = valid.copy()
    clustered["cluster"] = fit.labels
    for k in range(fit.probabilities.shape[1]):
        clustered[f"p_cluster_{k}"] = fit.probabilities[:, k]

    labels_grid = _grid_from_xy_values(clustered, value_col="cluster", rows=rows, cols=cols)
    _save_cluster_map(
        labels_grid,
        out_path=out_dir / "cluster_map.png",
        title=f"GMM clusters (K={args.n_clusters})",
    )

    prob_grids: list[np.ndarray] = []
    for k in range(fit.probabilities.shape[1]):
        prob_grids.append(_grid_from_xy_values(clustered, value_col=f"p_cluster_{k}", rows=rows, cols=cols))

    _save_probability_panels(
        prob_grids,
        out_path=out_dir / "cluster_prob_maps.png",
        title=f"Cluster probabilities (K={args.n_clusters})",
    )

    summary = summarize_clusters(clustered, feature_columns=feature_columns, labels=fit.labels)
    summary["interpretation_scaffold"] = summary.apply(interpretation_scaffold, axis=1)
    summary.to_csv(out_dir / "cluster_summary.csv", index=False)

    means_cols = ["cluster", "n"] + [f"mean_{c}" for c in feature_columns]
    summary[means_cols].to_csv(out_dir / "cluster_feature_means.csv", index=False)


if __name__ == "__main__":
    main()


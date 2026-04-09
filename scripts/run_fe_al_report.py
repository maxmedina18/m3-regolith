from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from lunar_m3.data_loading import load_m3_cube
from lunar_m3.features import extract_feature_table
from lunar_m3.validation import check_band_centers_plausible
from lunar_m3.visualization import plot_scalar_map


@dataclass(frozen=True)
class ProxyConfig:
    fe_weight_1um: float = 0.6
    fe_weight_2um: float = 0.4
    al_brightness_weight: float = 1.0
    al_fe_penalty: float = 3.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an M3 cube (.IMG/.HDR or .npz).",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts_fe_al")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cube = load_m3_cube(args.input)
    features = extract_feature_table(cube.data, cube.wavelengths)
    check_band_centers_plausible(features)

    cfg = ProxyConfig()

    fe_index = cfg.fe_weight_1um * features["bd_1um"].astype(float) + cfg.fe_weight_2um * features["bd_2um"].astype(float)
    fe_index = fe_index.clip(lower=0.0)

    brightness = features["brightness_ref"].astype(float)

    al_proxy = cfg.al_brightness_weight * brightness * np.exp(-cfg.al_fe_penalty * fe_index)

    rows, cols, _ = cube.data.shape
    fe_grid = fe_index.to_numpy().reshape(rows, cols)
    al_grid = al_proxy.to_numpy().reshape(rows, cols)

    plot_scalar_map(
        fe_grid,
        title="Fe proxy (band-depth based)",
        cmap="magma",
        output_path=out_dir / "fe_proxy_map.png",
        cbar_label="Fe proxy",
    )
    plot_scalar_map(
        al_grid,
        title="Plagioclase/highlands proxy (" "NOT elemental Al)",
        cmap="viridis",
        output_path=out_dir / "al_proxy_map.png",
        cbar_label="Al proxy",
    )

    pixel_out = features[["x", "y", "brightness_ref", "bd_1um", "bd_2um", "bc_1um", "bc_2um"]].copy()
    pixel_out["fe_proxy"] = fe_index
    pixel_out["al_proxy"] = al_proxy
    pixel_out.to_csv(out_dir / "pixel_proxies.csv", index=False)

    metrics = {
        "n_pixels": int(features.shape[0]),
        "fe_proxy_mean": float(np.mean(fe_index)),
        "fe_proxy_p95": float(np.quantile(fe_index, 0.95)),
        "al_proxy_mean": float(np.mean(al_proxy)),
        "al_proxy_p95": float(np.quantile(al_proxy, 0.95)),
    }
    pd.DataFrame([metrics]).to_csv(out_dir / "summary_metrics.csv", index=False)

    _write_pdf_report(out_dir, metrics, fe_index.to_numpy(), al_proxy.to_numpy())


def _write_pdf_report(out_dir: Path, metrics: dict[str, object], fe: np.ndarray, al: np.ndarray) -> None:
    pdf_path = out_dir / "report.pdf"

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle("Lunar M³ — Fe and Al-Proxy Report", fontsize=16)

        text = (
            "Notes:\n"
            "- Fe proxy is computed from ~1 µm and ~2 µm band depths after linear continuum removal.\n"
            "- 'Al proxy' here is a plagioclase/highlands proxy (high brightness + weak Fe absorptions), "
            "not a direct elemental aluminum measurement.\n"
        )
        fig.text(0.08, 0.90, text, va="top", fontsize=10)

        lines = [f"{k}: {v}" for k, v in metrics.items()]
        fig.text(0.08, 0.80, "Summary:\n" + "\n".join(lines), va="top", fontsize=10)

        plt.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.hist(fe[np.isfinite(fe)], bins=60, alpha=0.8)
        ax.set_title("Fe proxy distribution")
        ax.set_xlabel("Fe proxy")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.hist(al[np.isfinite(al)], bins=60, alpha=0.8)
        ax.set_title("Al proxy distribution (plagioclase/highlands proxy)")
        ax.set_xlabel("Al proxy")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()

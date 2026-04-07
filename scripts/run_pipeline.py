from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lunar_m3.data_loading import load_m3_cube
from lunar_m3.features import extract_feature_table
from lunar_m3.models import predict_labels, train_baseline_classifier
from lunar_m3.preprocessing import (
    clip_invalid_reflectance,
    normalize_by_reference_window,
    savgol_smooth,
)
from lunar_m3.validation import check_band_centers_plausible
from lunar_m3.visualization import plot_label_map, plot_spectrum_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="synthetic")
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "svm", "rf"])
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Extract features only (recommended for real M3 cubes without labels).",
    )
    args = parser.parse_args()

    cube = load_m3_cube(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = extract_feature_table(cube.data, cube.wavelengths)
    check_band_centers_plausible(features)

    if args.skip_classification:
        y_pred = None
    elif hasattr(cube, "synthetic_labels"):
        y = np.asarray(getattr(cube, "synthetic_labels")).reshape(-1)
        train_result = train_baseline_classifier(features, y, model=args.model, random_state=args.seed)
        (out_dir / "classification_report.txt").write_text(train_result.report)
        y_pred = predict_labels(train_result.pipeline, features)
    else:
        y_pred = None

    rows, cols, _ = cube.data.shape
    if y_pred is not None:
        pred_grid = y_pred.reshape(rows, cols)
        plot_label_map(pred_grid, title="Predicted classes", output_path=out_dir / "predicted_map.png")

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

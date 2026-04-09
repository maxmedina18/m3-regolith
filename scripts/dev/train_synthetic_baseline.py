from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from lunar_m3.dev.synthetic import generate_synthetic_cube
from lunar_m3.features import extract_feature_table
from lunar_m3.models import train_baseline_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline classifier on synthetic data (dev-only).")
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "svm", "rf"])
    parser.add_argument("--output-dir", type=str, default="artifacts_dev")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cube = generate_synthetic_cube(seed=args.seed)
    features = extract_feature_table(cube.data, cube.wavelengths)
    labels = np.asarray(getattr(cube, "synthetic_labels")).reshape(-1)

    result = train_baseline_classifier(features, labels, model=args.model, random_state=args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, out_dir / "model.joblib")
    (out_dir / "classification_report.txt").write_text(result.report)


if __name__ == "__main__":
    main()


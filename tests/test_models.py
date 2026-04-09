import numpy as np

from lunar_m3.dev.synthetic import generate_synthetic_cube
from lunar_m3.features import extract_feature_table
from lunar_m3.models import train_baseline_classifier


def test_train_baseline_classifier_runs() -> None:
    cube = generate_synthetic_cube(rows=12, cols=10, bands=60, seed=2)
    df = extract_feature_table(cube.data, cube.wavelengths)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=df.shape[0])
    result = train_baseline_classifier(df, y, model="logreg")
    assert "precision" in result.report

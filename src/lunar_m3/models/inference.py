from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def predict_labels(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Predict class labels for a feature table."""

    feature_columns = [c for c in features.columns if c not in {"x", "y"}]
    X = features[feature_columns]
    return np.asarray(pipeline.predict(X), dtype=int)

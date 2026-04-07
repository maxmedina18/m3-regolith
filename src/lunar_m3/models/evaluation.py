from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    """Return basic evaluation metrics for predictions."""

    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    return {
        "accuracy": float(accuracy_score(y_t, y_p)),
        "confusion_matrix": confusion_matrix(y_t, y_p).tolist(),
    }

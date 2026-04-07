from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class TrainResult:
    pipeline: Pipeline
    report: str
    feature_columns: list[str]


ModelName = Literal["logreg", "svm", "rf"]


def train_baseline_classifier(
    features: pd.DataFrame,
    labels: np.ndarray,
    *,
    model: ModelName = "logreg",
    test_size: float = 0.25,
    random_state: int = 0,
) -> TrainResult:
    """Train a baseline classifier on extracted features.

    Args:
        features: DataFrame containing feature columns and optional `x`, `y`.
        labels: Array shaped (n_samples,) of integer class labels.
        model: Model selection.

    Returns:
        TrainResult including fitted pipeline and text report.
    """

    feature_columns = [c for c in features.columns if c not in {"x", "y"}]
    X = features[feature_columns]
    y = np.asarray(labels, dtype=int)

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, feature_columns)])

    if model == "logreg":
        estimator = LogisticRegression(max_iter=2000, random_state=random_state)
    elif model == "svm":
        estimator = SVC(kernel="rbf", probability=False, random_state=random_state)
    elif model == "rf":
        estimator = RandomForestClassifier(n_estimators=300, random_state=random_state)
    else:
        raise ValueError(f"Unknown model: {model}")

    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)

    return TrainResult(pipeline=clf, report=report, feature_columns=feature_columns)

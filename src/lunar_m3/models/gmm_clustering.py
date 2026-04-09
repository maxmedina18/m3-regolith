from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GmmClusterResult:
    labels: np.ndarray
    probabilities: np.ndarray
    feature_columns: list[str]
    scaler: StandardScaler
    model: GaussianMixture


def fit_gmm_clusters(
    features: pd.DataFrame,
    *,
    feature_columns: list[str],
    n_clusters: int,
    random_state: int = 0,
    covariance_type: str = "full",
) -> GmmClusterResult:
    x = features[feature_columns].astype(float)
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(axis=0, how="any")
    if x.shape[0] == 0:
        raise ValueError("No finite rows remain after dropping NaNs for clustering")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.to_numpy())

    gmm = GaussianMixture(
        n_components=int(n_clusters),
        covariance_type=str(covariance_type),
        random_state=int(random_state),
        reg_covar=1e-6,
    )
    gmm.fit(x_scaled)
    probs = gmm.predict_proba(x_scaled)
    labels = np.asarray(np.argmax(probs, axis=1), dtype=int)

    return GmmClusterResult(
        labels=labels,
        probabilities=probs,
        feature_columns=list(feature_columns),
        scaler=scaler,
        model=gmm,
    )


def summarize_clusters(
    features: pd.DataFrame,
    *,
    feature_columns: list[str],
    labels: np.ndarray,
) -> pd.DataFrame:
    df = features[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    labs = np.asarray(labels, dtype=int)
    if df.shape[0] != labs.shape[0]:
        raise ValueError("labels length must match number of valid feature rows")

    out_rows: list[dict[str, object]] = []
    for k in sorted(set(int(x) for x in labs)):
        mask = labs == k
        row: dict[str, object] = {"cluster": int(k), "n": int(np.count_nonzero(mask))}
        for col in feature_columns:
            vals = df[col].to_numpy()[mask]
            row[f"mean_{col}"] = float(np.mean(vals))
            row[f"std_{col}"] = float(np.std(vals))
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def interpretation_scaffold(summary_row: pd.Series) -> str:
    bd1 = float(summary_row.get("mean_bd_1um", np.nan))
    bd2 = float(summary_row.get("mean_bd_2um", np.nan))
    slope = float(summary_row.get("mean_slope_global", np.nan))
    bright = float(summary_row.get("mean_brightness_ref", np.nan))

    tags: list[str] = []

    if np.isfinite(bd1) and np.isfinite(bd2):
        if bd1 > 0.05 and bd2 > 0.03:
            tags.append("stronger mafic absorptions")
        elif bd1 < 0.02 and bd2 < 0.02:
            tags.append("weak mafic absorptions")

    if np.isfinite(bright):
        if bright > 0.12:
            tags.append("higher brightness")
        elif bright < 0.06:
            tags.append("lower brightness")

    if np.isfinite(slope):
        if slope > 0.05:
            tags.append("redder slope (maturity proxy)")
        elif slope < 0.0:
            tags.append("bluer/flat slope")

    if not tags:
        return "no strong heuristic tag"
    return "; ".join(tags)


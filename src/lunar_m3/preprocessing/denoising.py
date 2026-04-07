from __future__ import annotations

import numpy as np


def clip_invalid_reflectance(
    reflectance: np.ndarray,
    *,
    min_reflectance: float = 0.0,
    max_reflectance: float = 2.0,
) -> np.ndarray:
    """Clip reflectance values and replace non-finite samples.

    Args:
        reflectance: Reflectance array shaped (bands,).
        min_reflectance: Lower bound.
        max_reflectance: Upper bound.

    Returns:
        Cleaned reflectance.
    """

    r = np.asarray(reflectance, dtype=float)
    finite = np.isfinite(r)
    if np.any(finite):
        fill_value = float(np.nanmedian(r[finite]))
    else:
        fill_value = 0.0

    r2 = np.where(finite, r, fill_value)
    return np.clip(r2, min_reflectance, max_reflectance)

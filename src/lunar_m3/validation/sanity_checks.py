from __future__ import annotations

import numpy as np


def require_finite(arr: np.ndarray, *, name: str) -> None:
    """Raise if array contains non-finite values."""

    a = np.asarray(arr)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")


def require_shape(arr: np.ndarray, *, ndim: int, name: str) -> None:
    """Raise if array does not have the expected ndim."""

    a = np.asarray(arr)
    if a.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}; got ndim={a.ndim}")

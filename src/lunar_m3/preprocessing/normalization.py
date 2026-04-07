from __future__ import annotations

import numpy as np


def normalize_by_reference_window(
    wavelengths_um: np.ndarray,
    reflectance: np.ndarray,
    *,
    window_um: tuple[float, float] = (1.45, 1.55),
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize reflectance by the median value within a reference window.

    This reduces brightness scaling effects while preserving band shapes.

    Args:
        wavelengths_um: Wavelengths in microns.
        reflectance: Reflectance array shaped (bands,).
        window_um: Reference wavelength interval.
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized reflectance.
    """

    w = np.asarray(wavelengths_um)
    r = np.asarray(reflectance, dtype=float)
    mask = (w >= window_um[0]) & (w <= window_um[1]) & np.isfinite(r)
    if not np.any(mask):
        scale = np.nanmedian(r[np.isfinite(r)])
    else:
        scale = np.nanmedian(r[mask])

    if not np.isfinite(scale) or abs(scale) < eps:
        return r.copy()

    return r / float(scale)

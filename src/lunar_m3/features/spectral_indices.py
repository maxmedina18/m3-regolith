from __future__ import annotations

import numpy as np


def spectral_slope(
    wavelengths_um: np.ndarray,
    reflectance: np.ndarray,
    *,
    region_um: tuple[float, float],
) -> float:
    """Compute a simple spectral slope using linear least squares in a region."""

    w = np.asarray(wavelengths_um, dtype=float)
    r = np.asarray(reflectance, dtype=float)
    mask = (w >= region_um[0]) & (w <= region_um[1]) & np.isfinite(r)
    if np.count_nonzero(mask) < 3:
        return float("nan")

    x = w[mask]
    y = r[mask]
    x0 = x.mean()
    denom = float(np.sum((x - x0) ** 2))
    if denom <= 0.0:
        return float("nan")

    m = float(np.sum((x - x0) * (y - y.mean())) / denom)
    return m


def reflectance_ratio(
    wavelengths_um: np.ndarray,
    reflectance: np.ndarray,
    *,
    numerator_um: float,
    denominator_um: float,
    eps: float = 1e-8,
) -> float:
    """Compute reflectance ratio R(λ_num) / R(λ_den) using interpolation."""

    w = np.asarray(wavelengths_um, dtype=float)
    r = np.asarray(reflectance, dtype=float)
    if not np.any(np.isfinite(r)):
        return float("nan")

    r_filled = np.where(np.isfinite(r), r, np.nanmedian(r[np.isfinite(r)]))

    r_num = float(np.interp(float(numerator_um), w, r_filled))
    r_den = float(np.interp(float(denominator_um), w, r_filled))
    if abs(r_den) < eps:
        return float("nan")

    return r_num / r_den

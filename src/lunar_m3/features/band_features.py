from __future__ import annotations

import numpy as np


def band_center(
    wavelengths_um: np.ndarray,
    cont_removed_reflectance: np.ndarray,
    *,
    region_um: tuple[float, float],
) -> float:
    """Estimate band center as the wavelength of minimum continuum-removed reflectance."""

    w = np.asarray(wavelengths_um, dtype=float)
    r = np.asarray(cont_removed_reflectance, dtype=float)
    mask = (w >= region_um[0]) & (w <= region_um[1]) & np.isfinite(r)
    if not np.any(mask):
        return float("nan")

    idx = np.argmin(r[mask])
    return float(w[mask][idx])


def band_depth(
    wavelengths_um: np.ndarray,
    cont_removed_reflectance: np.ndarray,
    *,
    region_um: tuple[float, float],
) -> float:
    """Compute band depth as `1 - min(R_c)` within a region."""

    w = np.asarray(wavelengths_um, dtype=float)
    r = np.asarray(cont_removed_reflectance, dtype=float)
    mask = (w >= region_um[0]) & (w <= region_um[1]) & np.isfinite(r)
    if not np.any(mask):
        return float("nan")

    return float(max(0.0, 1.0 - np.min(r[mask])))


def band_area(
    wavelengths_um: np.ndarray,
    cont_removed_reflectance: np.ndarray,
    *,
    region_um: tuple[float, float],
) -> float:
    """Compute band area as the integral of `(1 - R_c)` across a region."""

    w = np.asarray(wavelengths_um, dtype=float)
    r = np.asarray(cont_removed_reflectance, dtype=float)
    mask = (w >= region_um[0]) & (w <= region_um[1]) & np.isfinite(r)
    if not np.any(mask):
        return float("nan")

    w2 = w[mask]
    y = np.clip(1.0 - r[mask], 0.0, np.inf)
    return float(np.trapezoid(y, w2))

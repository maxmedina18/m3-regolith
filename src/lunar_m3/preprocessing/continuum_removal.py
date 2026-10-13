from __future__ import annotations

import numpy as np


def continuum_remove_linear(
    wavelengths_um: np.ndarray,
    reflectance: np.ndarray,
    *,
    region_um: tuple[float, float],
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear continuum removal over a wavelength interval.

    The continuum is a straight line between the reflectance at the region
    endpoints (interpolated if needed). Continuum-removed reflectance is `R / C`.
    Any nonfinite reflectance invalidates both complete outputs (all NaN),
    including nonfinite samples outside the requested region. No imputation.

    Args:
        wavelengths_um: Wavelength array in microns.
        reflectance: Reflectance array shaped (bands,).
        region_um: (start, end) wavelength interval.
        eps: Small constant to avoid division by zero.

    Returns:
        (reflectance_cont_removed, continuum)
    """

    w = np.asarray(wavelengths_um, dtype=float)
    r = np.asarray(reflectance, dtype=float)
    if w.ndim != 1 or r.shape != w.shape or len(w) < 2 or not np.all(np.isfinite(w)) or not np.all(np.diff(w) > 0):
        raise ValueError("Expected matching spectrum and finite increasing wavelengths")

    start, end = float(region_um[0]), float(region_um[1])
    if not np.isfinite(start + end) or start >= end:
        raise ValueError(f"Invalid region_um: {region_um}")

    if not np.all(np.isfinite(r)):
        return np.full_like(r, np.nan), np.full_like(r, np.nan)

    region_mask = (w >= start) & (w <= end)
    if start < w[0] or end > w[-1] or not np.any(region_mask):
        return np.full_like(r, np.nan), np.full_like(r, np.nan)

    w_region = w[region_mask]
    r_region = r[region_mask]

    # Use the full grid to bracket off-grid endpoints. Never impute missing data.
    r_start = float(np.interp(start, w, r))
    r_end = float(np.interp(end, w, r))

    continuum_region = r_start + (r_end - r_start) * (w_region - start) / (end - start)
    continuum_region = np.where(continuum_region > eps, continuum_region, np.nan)

    r_cr_region = r_region / continuum_region

    r_cr = r.copy()
    continuum = np.ones_like(r)
    r_cr[region_mask] = r_cr_region
    continuum[region_mask] = continuum_region

    return r_cr, continuum

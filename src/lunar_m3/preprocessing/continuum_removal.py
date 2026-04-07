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

    start, end = float(region_um[0]), float(region_um[1])
    if start >= end:
        raise ValueError(f"Invalid region_um: {region_um}")

    region_mask = (w >= start) & (w <= end)
    if not np.any(region_mask):
        return r.copy(), np.ones_like(r)

    w_region = w[region_mask]
    r_region = r[region_mask]

    if not np.all(np.isfinite(r_region)):
        median = np.nanmedian(r_region[np.isfinite(r_region)])
        r_region = np.where(np.isfinite(r_region), r_region, median)

    r_start = float(np.interp(start, w_region, r_region))
    r_end = float(np.interp(end, w_region, r_region))

    continuum_region = r_start + (r_end - r_start) * (w_region - start) / (end - start)
    continuum_region = np.clip(continuum_region, eps, np.inf)

    r_cr_region = r_region / continuum_region

    r_cr = r.copy()
    continuum = np.ones_like(r)
    r_cr[region_mask] = r_cr_region
    continuum[region_mask] = continuum_region

    return r_cr, continuum

from __future__ import annotations

import numpy as np


def spectral_slope(
    wavelengths_um: np.ndarray,
    reflectance: np.ndarray,
    *,
    region_um: tuple[float, float],
    weights: np.ndarray | None = None,
    downweight_regions_um: list[tuple[float, float]] | None = None,
    downweight_factor: float = 0.2,
) -> float:
    """Compute a simple spectral slope using weighted linear least squares.

    This is used as a compact proxy for spectral reddening (space weathering).
    The implementation supports downweighting known-problem wavelength regions
    (e.g., the ~1.34 µm instrument join).

    Args:
        wavelengths_um: Wavelength array in microns.
        reflectance: Reflectance array shaped (bands,).
        region_um: Inclusive wavelength region used for the fit.
        weights: Optional per-band weights (same shape as reflectance).
        downweight_regions_um: Optional list of wavelength intervals to downweight.
        downweight_factor: Multiplicative factor applied inside downweight intervals.

    Returns:
        Slope (reflectance per micron).
    """

    w = np.asarray(wavelengths_um, dtype=float)
    r = np.asarray(reflectance, dtype=float)
    mask = (w >= region_um[0]) & (w <= region_um[1]) & np.isfinite(r)
    if np.count_nonzero(mask) < 3:
        return float("nan")

    wts = np.ones_like(r, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if wts.shape != r.shape:
        raise ValueError("weights must have same shape as reflectance")

    if downweight_regions_um:
        for lo, hi in downweight_regions_um:
            wts[(w >= float(lo)) & (w <= float(hi))] *= float(downweight_factor)

    wts = np.where(np.isfinite(wts), wts, 0.0)
    wts = np.clip(wts, 0.0, np.inf)
    mask = mask & (wts > 0.0)
    if np.count_nonzero(mask) < 3:
        return float("nan")

    x = w[mask]
    y = r[mask]
    wt = wts[mask]

    wt_sum = float(np.sum(wt))
    if wt_sum <= 0.0:
        return float("nan")

    x0 = float(np.sum(wt * x) / wt_sum)
    y0 = float(np.sum(wt * y) / wt_sum)

    denom = float(np.sum(wt * (x - x0) ** 2))
    if denom <= 0.0:
        return float("nan")

    m = float(np.sum(wt * (x - x0) * (y - y0)) / denom)
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

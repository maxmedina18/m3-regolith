from __future__ import annotations

import numpy as np


def _parabolic_minimum(x: np.ndarray, y: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(x) - 1:
        return float(x[idx])

    x0, x1, x2 = float(x[idx - 1]), float(x[idx]), float(x[idx + 1])
    y0, y1, y2 = float(y[idx - 1]), float(y[idx]), float(y[idx + 1])

    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0.0:
        return float(x1)

    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    if a == 0.0:
        return float(x1)

    xv = -b / (2.0 * a)
    if xv < min(x0, x2) or xv > max(x0, x2):
        return float(x1)
    return float(xv)


def detect_absorption_band(
    wavelengths_um: np.ndarray,
    cont_removed_reflectance: np.ndarray,
    *,
    search_region_um: tuple[float, float],
    join_region_um: tuple[float, float] | None = (1.325, 1.355),
    baseline_level: float = 1.0,
    return_to_baseline_tol: float = 0.01,
) -> dict[str, float]:
    """Detect an absorption band inside a continuum-removed spectrum.

    This is a lightweight, dependency-free detector intended for M3-style
    spectra where the major mafic absorptions are near ~1 µm and ~2 µm.

    The algorithm is intentionally simple:
        1) Find the wavelength of minimum continuum-removed reflectance.
        2) Walk left/right to find where the spectrum returns near the baseline.
        3) Return the detected bounds and refined center estimate.

    Args:
        wavelengths_um: Wavelength array in microns.
        cont_removed_reflectance: Continuum-removed spectrum (R / C).
        search_region_um: Detection search window.
        join_region_um: Optional interval to ignore during detection.
        baseline_level: Expected continuum-removed baseline (usually 1.0).
        return_to_baseline_tol: Tolerance around baseline to define band edges.

    Returns:
        Dict with keys:
            - band_left_um
            - band_right_um
            - band_center_um
            - band_min_rc
            - band_depth
    """

    w = np.asarray(wavelengths_um, dtype=float)
    rc = np.asarray(cont_removed_reflectance, dtype=float)

    mask = (w >= float(search_region_um[0])) & (w <= float(search_region_um[1])) & np.isfinite(rc)
    if join_region_um is not None:
        mask &= ~((w >= float(join_region_um[0])) & (w <= float(join_region_um[1])))

    if np.count_nonzero(mask) < 5:
        return {
            "band_left_um": float("nan"),
            "band_right_um": float("nan"),
            "band_center_um": float("nan"),
            "band_min_rc": float("nan"),
            "band_depth": float("nan"),
        }

    w2 = w[mask]
    rc2 = rc[mask]
    idx_min = int(np.argmin(rc2))

    center = _parabolic_minimum(w2, rc2, idx_min)
    rc_min = float(np.min(rc2))
    depth = float(max(0.0, baseline_level - rc_min))

    left_idx = idx_min
    while left_idx > 0:
        if abs(float(rc2[left_idx]) - baseline_level) <= return_to_baseline_tol:
            break
        left_idx -= 1

    right_idx = idx_min
    while right_idx < len(w2) - 1:
        if abs(float(rc2[right_idx]) - baseline_level) <= return_to_baseline_tol:
            break
        right_idx += 1

    left = float(w2[left_idx])
    right = float(w2[right_idx])
    if right <= left:
        left = float(search_region_um[0])
        right = float(search_region_um[1])

    return {
        "band_left_um": left,
        "band_right_um": right,
        "band_center_um": center,
        "band_min_rc": rc_min,
        "band_depth": depth,
    }


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

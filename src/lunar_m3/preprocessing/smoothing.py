from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def savgol_smooth(
    reflectance: np.ndarray,
    *,
    window_length: int = 9,
    polyorder: int = 2,
) -> np.ndarray:
    """Smooth a spectrum with a Savitzky–Golay filter.

    Args:
        reflectance: Reflectance array shaped (bands,).
        window_length: Odd integer window size.
        polyorder: Polynomial order.

    Returns:
        Smoothed reflectance.
    """

    r = np.asarray(reflectance, dtype=float)
    if r.size < 3:
        return r.copy()

    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    wl = min(wl, r.size if r.size % 2 == 1 else r.size - 1)
    wl = max(wl, 3)
    po = min(int(polyorder), wl - 1)

    if not np.all(np.isfinite(r)):
        r = np.where(np.isfinite(r), r, np.nanmedian(r[np.isfinite(r)]))

    return savgol_filter(r, window_length=wl, polyorder=po, mode="interp")

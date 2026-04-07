from __future__ import annotations

import numpy as np


def mitigate_instrument_join(
    wavelengths_um: np.ndarray,
    reflectance: np.ndarray,
    *,
    join_region_um: tuple[float, float] = (1.325, 1.355),
    mode: str = "interpolate",
) -> tuple[np.ndarray, np.ndarray]:
    """Mitigate the ~1.34 µm instrument join discontinuity.

    The M3 VNIR/SWIR join near ~1.34 µm can show a step or spike that can
    contaminate band fitting and slope estimation. This helper returns:

    - a modified reflectance spectrum that is safer for feature extraction
    - a boolean mask indicating which bands fall inside the join interval

    Modes:
        - "interpolate": replace the join samples using linear interpolation
          between the nearest samples outside the join.
        - "mask": leave reflectance unchanged; only return the join mask.

    Args:
        wavelengths_um: Wavelength array in microns.
        reflectance: Reflectance array shaped (bands,).
        join_region_um: Wavelength interval treated as the join region.
        mode: One of {"interpolate", "mask"}.

    Returns:
        (reflectance_out, join_mask)
    """

    w = np.asarray(wavelengths_um, dtype=float)
    r = np.asarray(reflectance, dtype=float)
    join_lo, join_hi = float(join_region_um[0]), float(join_region_um[1])
    join_mask = (w >= join_lo) & (w <= join_hi)

    if mode not in {"interpolate", "mask"}:
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "mask" or not np.any(join_mask):
        return r.copy(), join_mask

    out = r.copy()

    left_mask = (w < join_lo) & np.isfinite(out)
    right_mask = (w > join_hi) & np.isfinite(out)
    if not np.any(left_mask) or not np.any(right_mask):
        return out, join_mask

    i_left = int(np.max(np.nonzero(left_mask)[0]))
    i_right = int(np.min(np.nonzero(right_mask)[0]))
    if i_right <= i_left:
        return out, join_mask

    wl = float(w[i_left])
    wr = float(w[i_right])
    rl = float(out[i_left])
    rr = float(out[i_right])
    if not (np.isfinite(rl) and np.isfinite(rr)):
        return out, join_mask

    denom = (wr - wl)
    if denom <= 0.0:
        return out, join_mask

    out[join_mask] = rl + (rr - rl) * (w[join_mask] - wl) / denom
    return out, join_mask


from __future__ import annotations

import numpy as np
import pandas as pd


def check_band_centers_plausible(
    features: pd.DataFrame,
    *,
    bc_1um_range: tuple[float, float] = (0.85, 1.30),
    bc_2um_range: tuple[float, float] = (1.70, 2.50),
    max_fraction_outside: float = 0.15,
) -> dict[str, float]:
    """Scientific sanity check: band centers should be within expected regions.

    Returns:
        Fractions of samples outside each expected range.
    """

    bc1 = np.asarray(features["bc_1um"], dtype=float)
    bc2 = np.asarray(features["bc_2um"], dtype=float)

    bc1_ok = (bc1 >= bc_1um_range[0]) & (bc1 <= bc_1um_range[1])
    bc2_ok = (bc2 >= bc_2um_range[0]) & (bc2 <= bc_2um_range[1])

    frac1 = float(1.0 - np.mean(bc1_ok[np.isfinite(bc1_ok)]))
    frac2 = float(1.0 - np.mean(bc2_ok[np.isfinite(bc2_ok)]))

    if frac1 > max_fraction_outside or frac2 > max_fraction_outside:
        raise ValueError(
            "Band center plausibility check failed: "
            f"frac_outside_1um={frac1:.3f} frac_outside_2um={frac2:.3f}"
        )

    return {"frac_outside_1um": frac1, "frac_outside_2um": frac2}

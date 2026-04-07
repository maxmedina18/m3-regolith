from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lunar_m3.preprocessing.continuum_removal import continuum_remove_linear
from lunar_m3.preprocessing.denoising import clip_invalid_reflectance
from lunar_m3.preprocessing.normalization import normalize_by_reference_window
from lunar_m3.preprocessing.smoothing import savgol_smooth

from .band_features import band_area, band_center, band_depth
from .spectral_indices import reflectance_ratio, spectral_slope


@dataclass(frozen=True)
class FeatureConfig:
    region_1um: tuple[float, float] = (0.85, 1.30)
    region_2um: tuple[float, float] = (1.70, 2.50)


def extract_feature_table(
    cube_data: np.ndarray,
    wavelengths_um: np.ndarray,
    *,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Extract a feature table for every pixel in a cube.

    Args:
        cube_data: Reflectance array shaped (rows, cols, bands).
        wavelengths_um: Wavelength array shaped (bands,) in microns.
        config: Feature extraction configuration.

    Returns:
        DataFrame with one row per pixel and feature columns.
    """

    cfg = config or FeatureConfig()
    rows, cols, _ = cube_data.shape

    records: list[dict[str, float | int]] = []
    for y in range(rows):
        for x in range(cols):
            r0 = cube_data[y, x, :]
            r_clean = clip_invalid_reflectance(r0)
            ref_window = (1.45, 1.55)
            ref_mask = (wavelengths_um >= ref_window[0]) & (wavelengths_um <= ref_window[1])
            if np.any(ref_mask):
                brightness_ref = float(np.nanmedian(r_clean[ref_mask]))
            else:
                brightness_ref = float(np.nanmedian(r_clean))

            r = normalize_by_reference_window(wavelengths_um, r_clean, window_um=ref_window)
            r = savgol_smooth(r)

            r1_cr, _ = continuum_remove_linear(wavelengths_um, r, region_um=cfg.region_1um)
            r2_cr, _ = continuum_remove_linear(wavelengths_um, r, region_um=cfg.region_2um)

            rec: dict[str, float | int] = {
                "x": x,
                "y": y,
                "brightness_ref": brightness_ref,
                "slope_visnir": spectral_slope(wavelengths_um, r, region_um=(0.70, 1.30)),
                "slope_swir": spectral_slope(wavelengths_um, r, region_um=(1.70, 2.50)),
                "ratio_950_750": reflectance_ratio(wavelengths_um, r, numerator_um=0.95, denominator_um=0.75),
                "ratio_2000_1500": reflectance_ratio(wavelengths_um, r, numerator_um=2.00, denominator_um=1.50),
                "bd_1um": band_depth(wavelengths_um, r1_cr, region_um=cfg.region_1um),
                "bc_1um": band_center(wavelengths_um, r1_cr, region_um=cfg.region_1um),
                "ba_1um": band_area(wavelengths_um, r1_cr, region_um=cfg.region_1um),
                "bd_2um": band_depth(wavelengths_um, r2_cr, region_um=cfg.region_2um),
                "bc_2um": band_center(wavelengths_um, r2_cr, region_um=cfg.region_2um),
                "ba_2um": band_area(wavelengths_um, r2_cr, region_um=cfg.region_2um),
            }
            records.append(rec)

    return pd.DataFrame.from_records(records)

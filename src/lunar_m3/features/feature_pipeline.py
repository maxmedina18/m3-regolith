from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lunar_m3.preprocessing.continuum_removal import continuum_remove_linear
from lunar_m3.preprocessing.denoising import clip_invalid_reflectance
from lunar_m3.preprocessing.instrument_join import mitigate_instrument_join
from lunar_m3.preprocessing.normalization import normalize_by_reference_window
from lunar_m3.preprocessing.smoothing import savgol_smooth

from .band_features import band_area, band_center, band_depth, detect_absorption_band
from .spectral_indices import reflectance_ratio, spectral_slope


@dataclass(frozen=True)
class FeatureConfig:
    band1_search_um: tuple[float, float] = (0.85, 1.30)
    band2_search_um: tuple[float, float] = (1.70, 2.50)
    join_region_um: tuple[float, float] = (1.325, 1.355)
    join_downweight: float = 0.2
    ref_window_um: tuple[float, float] = (1.45, 1.55)


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

            # 1) Basic cleanup to keep downstream math stable.
            #    This step is meant to remove NaNs/Infs and clip reflectance into
            #    a physically reasonable range.
            r_clean = clip_invalid_reflectance(r0)

            # 2) Track an absolute brightness proxy before normalization.
            #    This lets the model use both composition-like features
            #    (band shapes) and albedo-like information.
            ref_mask = (wavelengths_um >= cfg.ref_window_um[0]) & (wavelengths_um <= cfg.ref_window_um[1])
            if np.any(ref_mask):
                brightness_ref = float(np.nanmedian(r_clean[ref_mask]))
            else:
                brightness_ref = float(np.nanmedian(r_clean))

            # 3) Normalize and smooth for robust feature extraction.
            #    Normalization reduces brightness-only variation so absorption
            #    metrics are more comparable across pixels.
            r = normalize_by_reference_window(wavelengths_um, r_clean, window_um=cfg.ref_window_um)
            r = savgol_smooth(r)

            # 4) Mitigate the ~1.34 µm VNIR/SWIR join. We do two things:
            #    - interpolate across the join for band metrics and ratios
            #    - downweight it for slope fitting
            r, join_mask = mitigate_instrument_join(
                wavelengths_um,
                r,
                join_region_um=cfg.join_region_um,
                mode="interpolate",
            )

            # Per-band weights used by weighted slope fitting.
            wts = np.ones_like(r, dtype=float)
            wts[join_mask] *= float(cfg.join_downweight)

            # 5) Continuum removal happens before band measurement.
            #    We do a seed continuum removal over broad search windows to
            #    enable automated detection of the absorption band location.
            r1_cr_seed, _ = continuum_remove_linear(wavelengths_um, r, region_um=cfg.band1_search_um)
            r2_cr_seed, _ = continuum_remove_linear(wavelengths_um, r, region_um=cfg.band2_search_um)

            # 6) Automated detection of ~1 µm and ~2 µm absorptions.
            #    The detector returns approximate band edges; we then re-run
            #    continuum removal over those edges to measure depth/center/area.
            b1 = detect_absorption_band(
                wavelengths_um,
                r1_cr_seed,
                search_region_um=cfg.band1_search_um,
                join_region_um=cfg.join_region_um,
            )
            b2 = detect_absorption_band(
                wavelengths_um,
                r2_cr_seed,
                search_region_um=cfg.band2_search_um,
                join_region_um=cfg.join_region_um,
            )

            # 7) Convert detector outputs to safe regions (fallback to defaults).
            b1_region = (
                float(b1["band_left_um"]) if np.isfinite(float(b1["band_left_um"])) else cfg.band1_search_um[0],
                float(b1["band_right_um"]) if np.isfinite(float(b1["band_right_um"])) else cfg.band1_search_um[1],
            )
            b2_region = (
                float(b2["band_left_um"]) if np.isfinite(float(b2["band_left_um"])) else cfg.band2_search_um[0],
                float(b2["band_right_um"]) if np.isfinite(float(b2["band_right_um"])) else cfg.band2_search_um[1],
            )

            # 8) Final continuum removal for the detected band extents.
            r1_cr, _ = continuum_remove_linear(wavelengths_um, r, region_um=b1_region)
            r2_cr, _ = continuum_remove_linear(wavelengths_um, r, region_um=b2_region)

            rec: dict[str, float | int] = {
                "x": x,
                "y": y,
                "brightness_ref": brightness_ref,

                # Spectral slopes help capture reddening trends linked to
                # maturity/space weathering, while the band features capture
                # absorption physics linked to mineralogy.
                "slope_visnir": spectral_slope(
                    wavelengths_um,
                    r,
                    region_um=(0.70, 1.30),
                    weights=wts,
                    downweight_regions_um=[cfg.join_region_um],
                    downweight_factor=cfg.join_downweight,
                ),
                "slope_swir": spectral_slope(
                    wavelengths_um,
                    r,
                    region_um=(1.70, 2.50),
                    weights=wts,
                    downweight_regions_um=[cfg.join_region_um],
                    downweight_factor=cfg.join_downweight,
                ),
                "slope_global": spectral_slope(
                    wavelengths_um,
                    r,
                    region_um=(0.70, 2.50),
                    weights=wts,
                    downweight_regions_um=[cfg.join_region_um],
                    downweight_factor=cfg.join_downweight,
                ),
                "ratio_950_750": reflectance_ratio(wavelengths_um, r, numerator_um=0.95, denominator_um=0.75),
                "ratio_2000_1500": reflectance_ratio(wavelengths_um, r, numerator_um=2.00, denominator_um=1.50),

                # Band geometry columns are emitted so you can debug/QA the
                # detector and understand how measurements were bounded.
                "band1_left_um": b1_region[0],
                "band1_right_um": b1_region[1],
                "bd_1um": band_depth(wavelengths_um, r1_cr, region_um=b1_region),
                "bc_1um": band_center(wavelengths_um, r1_cr, region_um=b1_region),
                "ba_1um": band_area(wavelengths_um, r1_cr, region_um=b1_region),
                "band2_left_um": b2_region[0],
                "band2_right_um": b2_region[1],
                "bd_2um": band_depth(wavelengths_um, r2_cr, region_um=b2_region),
                "bc_2um": band_center(wavelengths_um, r2_cr, region_um=b2_region),
                "ba_2um": band_area(wavelengths_um, r2_cr, region_um=b2_region),
            }
            records.append(rec)

    return pd.DataFrame.from_records(records)

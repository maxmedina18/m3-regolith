import numpy as np

from lunar_m3.features import band_depth, detect_absorption_band
from lunar_m3.preprocessing import continuum_remove_linear, mitigate_instrument_join
from lunar_m3.features.spectral_indices import spectral_slope


def test_absorption_band_detection_known_1um_center_and_depth() -> None:
    w = np.linspace(0.7, 1.4, 701)
    true_center = 1.01
    true_depth = 0.12
    rc = 1.0 - true_depth * np.exp(-0.5 * ((w - true_center) / 0.045) ** 2)

    out = detect_absorption_band(w, rc, search_region_um=(0.85, 1.30), join_region_um=(1.325, 1.355))
    assert abs(out["band_center_um"] - true_center) < 0.015
    assert abs(out["band_depth"] - true_depth) < 0.02


def test_continuum_removal_flat_spectrum_yields_near_zero_band_depth() -> None:
    w = np.linspace(0.8, 1.3, 101)
    r = np.full_like(w, 0.33)
    r_cr, _ = continuum_remove_linear(w, r, region_um=(0.85, 1.25))
    bd = band_depth(w, r_cr, region_um=(0.85, 1.25))
    assert bd < 1e-3


def test_instrument_join_mitigation_removes_discontinuity() -> None:
    w = np.linspace(1.25, 1.45, 101)
    r = 0.3 + 0.02 * (w - w.min())
    join_mask = (w >= 1.325) & (w <= 1.355)
    r2 = r.copy()
    r2[join_mask] += 0.10

    out, join_mask_out = mitigate_instrument_join(w, r2, join_region_um=(1.325, 1.355), mode="interpolate")
    assert np.any(join_mask_out)

    left = out[w < 1.325][-1]
    right = out[w > 1.355][0]
    assert abs(float(left) - float(right)) < 0.02


def test_slope_estimation_matches_known_linear_trend() -> None:
    w = np.linspace(0.7, 1.3, 121)
    m_true = 0.18
    b_true = 0.25
    r = b_true + m_true * w
    m_hat = spectral_slope(w, r, region_um=(0.7, 1.3))
    assert abs(m_hat - m_true) < 1e-6


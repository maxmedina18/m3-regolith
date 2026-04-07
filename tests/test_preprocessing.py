import numpy as np

from lunar_m3.preprocessing import clip_invalid_reflectance, continuum_remove_linear, normalize_by_reference_window, savgol_smooth


def test_clip_invalid_reflectance_replaces_nonfinite() -> None:
    r = np.array([0.1, np.nan, np.inf, 0.2])
    out = clip_invalid_reflectance(r)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


def test_continuum_remove_linear_preserves_shape() -> None:
    w = np.linspace(0.6, 2.6, 21)
    r = 0.3 + 0.05 * (w - w.min())
    r_cr, cont = continuum_remove_linear(w, r, region_um=(0.8, 1.3))
    assert r_cr.shape == r.shape
    assert cont.shape == r.shape
    assert np.all(np.isfinite(r_cr))


def test_normalize_and_smooth_run() -> None:
    w = np.linspace(0.6, 2.6, 21)
    r = 0.3 + 0.01 * np.sin(5 * w)
    r2 = normalize_by_reference_window(w, r, window_um=(1.4, 1.6))
    r3 = savgol_smooth(r2, window_length=7, polyorder=2)
    assert r3.shape == r.shape

import numpy as np

from lunar_m3.features import band_area, band_center, band_depth, extract_feature_table


def test_band_metrics_basic() -> None:
    w = np.linspace(0.8, 1.3, 51)
    r_cr = 1.0 - 0.1 * np.exp(-0.5 * ((w - 1.02) / 0.04) ** 2)
    bc = band_center(w, r_cr, region_um=(0.85, 1.25))
    bd = band_depth(w, r_cr, region_um=(0.85, 1.25))
    ba = band_area(w, r_cr, region_um=(0.85, 1.25))
    assert 0.95 < bc < 1.10
    assert bd > 0.05
    assert ba > 0.0


def test_extract_feature_table_shape() -> None:
    w = np.linspace(0.62, 2.60, 20)
    cube = np.zeros((3, 4, 20), dtype=float)
    cube[:] = 0.3
    df = extract_feature_table(cube, w)
    assert df.shape[0] == 12
    assert "bd_1um" in df.columns
    assert "bc_2um" in df.columns

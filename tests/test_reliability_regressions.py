import numpy as np
import pytest
from lunar_m3.features import extract_feature_table
from lunar_m3.features.spectral_indices import spectral_slope,reflectance_ratio
from lunar_m3.preprocessing import continuum_remove_linear

@pytest.mark.parametrize('value',[np.nan,-999.,0.])
def test_invalid_spectrum_never_becomes_absorption(value):
    w=np.linspace(.62,2.6,85)
    f=extract_feature_table(np.full((1,1,85),value),w,nodata=-999)
    assert f[['bd_1um','bd_2um','bc_1um','bc_2um']].isna().all().all()

def test_off_grid_linear_continuum_is_flat():
    w=np.arange(.8,1.31,.1);r=.2+.1*w
    cr,_=continuum_remove_linear(w,r,region_um=(.85,1.25))
    np.testing.assert_allclose(cr[(w>=.85)&(w<=1.25)],1,atol=1e-14)

def test_slope_does_not_modify_weights():
    w=np.linspace(1.2,1.5,31);r=np.sin(w*20);weights=np.ones_like(w)
    kw=dict(region_um=(1.2,1.5),weights=weights,downweight_regions_um=[(1.325,1.355)])
    a=spectral_slope(w,r,**kw);b=spectral_slope(w,r,**kw)
    assert a==b
    np.testing.assert_array_equal(weights,1)

def test_ratio_rejects_outside_coverage():
    assert np.isnan(reflectance_ratio(np.array([1.,1.1,1.2]),np.ones(3),numerator_um=.95,denominator_um=.75))

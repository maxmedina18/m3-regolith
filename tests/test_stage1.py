import json
from pathlib import Path
import numpy as np
import pytest
from lunar_m3.dev.stage1_fixture import make_fixture
from lunar_m3.stage1 import ProductSet, export
from lunar_m3.data_loading.pds3_envi import read_envi_header,read_envi_image

@pytest.mark.parametrize('interleave',['bsq','bil','bip'])
@pytest.mark.parametrize('byte_order',[0,1])
def test_source_layout_precision_and_masks(tmp_path,interleave,byte_order):
    config,expected=make_fixture(tmp_path,interleave,byte_order)
    p=ProductSet(config)
    for role in expected:
        np.testing.assert_array_equal(p.arrays[role],expected[role])
        assert not p.arrays[role].flags.writeable
    assert not p.headers['RFL'].band_good[0]
    radius=expected['LOC'][:,:,2];e=p.plane('LOC',3,elevation=True)
    np.testing.assert_array_equal(e[radius!=-999],radius[radius!=-999]-1737400)
    assert e[-1,-1]==-999

@pytest.mark.parametrize('format',['tsv','npy'])
def test_99_exports_roi_and_provenance(tmp_path,format):
    config,expected=make_fixture(tmp_path/'inputs')
    p=ProductSet(config);out=tmp_path/'output'
    m=export(p,out,roi=(1,3,1,4),format=format,chunk_lines=1)
    assert len(m['outputs'])==99
    assert not (out/'run_incomplete.json').exists()
    for rec in m['outputs']:
        a=np.load(out/rec['file']) if format=='npy' else np.loadtxt(out/rec['file'],ndmin=2)
        source=p.plane(rec['role'],rec['band_1based'],(1,3,1,4),elevation=rec['derived'])
        np.testing.assert_array_equal(a.astype(source.dtype),source)
    assert m['roi']==[1,3,1,4]
    assert m['validation']['nasaview']=='not_performed'
    with pytest.raises(FileExistsError):export(p,out)

@pytest.mark.parametrize('change',[('byte order = 0','byte order = 2'),('bands = 85','bands = 86'),('interleave = bil','interleave = xyz')])
def test_reject_invalid_header(tmp_path,change):
    config,_=make_fixture(tmp_path);h=Path(config['rfl_hdr']);h.write_text(h.read_text().replace(*change))
    with pytest.raises(ValueError):ProductSet(config)

def test_reject_size_and_missing_units(tmp_path):
    config,_=make_fixture(tmp_path);h=Path(config['rfl_hdr']);p=h.with_suffix('.IMG')
    with p.open('ab') as f:f.write(b'X')
    with pytest.raises(ValueError,match='byte count'):ProductSet(config)
    h.write_text(h.read_text().replace('wavelength units = Nanometers\n',''))
    with pytest.raises(ValueError,match='Missing wavelength units'):read_envi_header(h)
    assert read_envi_header(h,wavelength_unit='nm').wavelength_unit_source.startswith('caller')

def test_vrt_scaling_and_version_warning(tmp_path):
    config,_=make_fixture(tmp_path);label=tmp_path/'label.LBL';label.write_text('CH1:PIXEL_LOCATION_FILE_NAME = "OTHER_V03_LOC.IMG"')
    config['rfl_label']=str(label)
    assert any('OTHER_V03' in w for w in ProductSet(config).warnings)
    vrt=Path(config['rfl_vrt']);vrt.write_text(vrt.read_text().replace('</SimpleSource>','<ScaleRatio>2</ScaleRatio></SimpleSource>'))
    with pytest.raises(ValueError,match='transformation'):ProductSet(config)

def test_dry_run_does_not_create_outputs(tmp_path):
    config,_=make_fixture(tmp_path/'inputs');out=tmp_path/'absent'
    assert export(ProductSet(config),out,dry_run=True)['planned_tables']==99
    assert not out.exists()

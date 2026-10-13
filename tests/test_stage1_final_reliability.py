from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest

from lunar_m3.data_loading import M3Cube, load_m3_cube
from lunar_m3.data_loading.pds3_envi import EnviHeader
from lunar_m3.features import extract_feature_table
from lunar_m3.preprocessing import continuum_remove_linear


@pytest.mark.parametrize('nodata', [None, -999.0, np.nan])
def test_pixel_validity_optional_nodata_preserves_finite_and_bbl(nodata):
    data = np.array([[[.2, np.nan, np.inf, .3, -999.]]])
    header = EnviHeader(1, 1, 5, 'bip', 5, 0, nodata=nodata,
                        band_good=np.array([True, True, True, False, True]))
    cube = M3Cube(data, np.arange(5.), header=header)
    np.testing.assert_array_equal(cube.pixel_validity(0, 0),
                                  [True, False, False, False, nodata != -999.])


def test_feature_optional_nodata_and_positive_sentinel():
    w = np.linspace(.62, 2.6, 85)
    data = np.full((1, 1, 85), .25)
    assert np.isfinite(extract_feature_table(data, w, nodata=None).brightness_ref.iloc[0])
    assert extract_feature_table(data, w, nodata=.25).drop(columns=['x', 'y']).isna().all().all()


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('index', [0, 2, 4])
def test_continuum_nonfinite_anywhere_invalidates_entire_result(bad, index):
    w = np.linspace(.5, 1.5, 5)
    r = np.full(5, .2)
    r[index] = bad
    before = r.copy()
    cr, continuum = continuum_remove_linear(w, r, region_um=(.7, 1.3))
    assert np.isnan(cr).all() and np.isnan(continuum).all()
    np.testing.assert_array_equal(r, before)


def test_loader_to_feature_cli_preserves_metadata_and_precision(tmp_path, monkeypatch):
    # Positive NoData distinguishes metadata masking from generic negative-value rejection.
    w = np.linspace(620, 2600, 85)
    data = np.full((1, 2, 85), .200000000123, dtype='<f8')
    data[:, :, 0] = np.nan  # Bad band must be excluded, not imputed.
    data[0, 1, 40] = .25
    img = tmp_path / 'sample.IMG'
    data.tofile(img)
    hdr = img.with_suffix('.HDR')
    hdr.write_text('ENVI\nsamples = 2\nlines = 1\nbands = 85\ninterleave = bip\n'
                   'data type = 5\nbyte order = 0\ndata ignore value = 0.25\n'
                   'wavelength = {' + ','.join(map(str, w)) + '}\n'
                   'bbl = {' + ','.join(['0'] + ['1'] * 84) + '}\n')
    cube = load_m3_cube(hdr, wavelength_unit='nm')
    assert cube.data.dtype == np.dtype('float64')
    np.testing.assert_array_equal(cube.data, data)
    with pytest.raises(ValueError, match='Missing wavelength units'):
        load_m3_cube(hdr)
    script = Path(__file__).resolve().parents[1] / 'scripts/run_feature_extraction.py'
    spec = importlib.util.spec_from_file_location('feature_cli_test', script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    captured = {}
    def extract(*args, **kwargs):
        captured.update(kwargs)
        table = extract_feature_table(*args, **kwargs)
        assert np.isfinite(table.brightness_ref.iloc[0])
        assert table.drop(columns=['x', 'y']).iloc[1].isna().all()
        return table
    monkeypatch.setattr(cli, 'extract_feature_table', extract)
    monkeypatch.setattr(sys, 'argv', [str(script), '--input', str(hdr),
                        '--wavelength-unit', 'nm', '--output', str(tmp_path / 'features.parquet')])
    cli.main()
    assert captured['nodata'] == .25
    np.testing.assert_array_equal(captured['band_good'], cube.band_good)


@pytest.mark.parametrize('unit', [None, 'um', 'nm'])
def test_historical_comparison_requires_explicit_protocol_units(unit):
    script = Path(__file__).resolve().parents[1] / 'scripts/compare_original_extractors.py'
    spec = importlib.util.spec_from_file_location('comparison_unit_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if unit == 'nm':
        module.require_snapshot_wavelength_unit({'wavelength_unit': unit})
    else:
        with pytest.raises(ValueError, match='explicit wavelength_unit=nm'):
            module.require_snapshot_wavelength_unit({'wavelength_unit': unit})

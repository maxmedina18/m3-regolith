"""Arithmetic/layout tests, not scientific validation of lunar chemistry."""
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lunar_m3.feo import feo_from_reflectance, read_cube
CFG=json.loads((Path(__file__).resolve().parents[1]/'configs/original_calibration.json').read_text())

class EquationTests(unittest.TestCase):
    def test_known_angles_in_all_quadrants(self):
        # Choose positive reflectances with translated x/y in every quadrant.
        # Below-origin x must be small so R750 remains positive.
        x=np.array([.02,-.02,-.02,.02]);y=np.array([.02,.02,-.02,-.02])
        a=x+CFG['x_origin'];b=(y+CFG['y_origin'])*a
        result=feo_from_reflectance(a,b,CFG)
        expected=np.array([-math.pi/4,-3*math.pi/4,3*math.pi/4,math.pi/4])
        np.testing.assert_allclose(result['theta']['signed'],expected,rtol=1e-13)
        np.testing.assert_allclose(result['feo']['signed'],CFG['A']*expected+CFG['B'])

    def test_wrap_order_changes_predictions_by_two_pi_shift(self):
        r=feo_from_reflectance([.5,.5],[.8,.4],CFG)
        signed=r['feo']['signed'];shift=CFG['A']*2*math.pi
        self.assertAlmostEqual(r['feo']['wrap_after'][0]-signed[0],shift)
        self.assertAlmostEqual(r['feo']['wrap_before'][1]-signed[1],-shift)
        self.assertAlmostEqual(r['feo']['wrap_after'][1],signed[1])
        self.assertAlmostEqual(r['feo']['wrap_before'][0],signed[0])

    def test_zero_negative_nonfinite_masked(self):
        a=[0,-.2,np.nan,.5,.5,.5];b=[.2,.2,.2,np.inf,0,-.1]
        r=feo_from_reflectance(a,b,CFG)
        self.assertFalse(r['valid'].any())
        self.assertTrue(np.isnan(r['feo']['signed']).all())

    def test_explicit_mask_and_above_one_retention(self):
        r=feo_from_reflectance([1.3,1.3],[1.1,1.1],CFG,[True,False])
        np.testing.assert_array_equal(r['valid'],[True,False])
        self.assertTrue(np.isfinite(r['feo']['signed'][0]))

    def test_exact_origin_rejected(self):
        r=feo_from_reflectance([CFG['x_origin']],[CFG['x_origin']*CFG['y_origin']],CFG)
        self.assertFalse(r['valid'][0])

    def test_matches_independent_scalar_math(self):
        rng=np.random.default_rng(27)
        a=rng.uniform(.005,2,100);b=rng.uniform(.005,2,100)
        r=feo_from_reflectance(a,b,CFG)
        for i in range(100):
            phi=math.atan2(b[i]/a[i]-CFG['y_origin'],a[i]-CFG['x_origin'])
            for key,theta in [('signed',-phi),('wrap_after',(-phi)%(2*math.pi)),('wrap_before',-(phi%(2*math.pi)))]:
                self.assertAlmostEqual(r['feo'][key][i],CFG['A']*theta+CFG['B'],places=11)

    def test_negative_prediction_not_clipped(self):
        r=feo_from_reflectance([.5722007751464844],[.6027820706367493],CFG)
        self.assertAlmostEqual(r['feo']['signed'][0],-2.8487148365081003)

class ReaderTests(unittest.TestCase):
    def fixture(self,d,offset=16):
        product='TEST';nline,nband,nsample=3,4,2
        hdr=f'''samples = {nsample}\nlines = {nline}\nbands = {nband}\nheader offset = {offset}\ninterleave = bil\ndata type = 4\nbyte order = 0\ndata ignore value = -999\nwavelength = {{600,750,950,1100}}\nbbl = {{1,1,1,1}}\n'''
        lbl=f'''OBJECT = RFL_IMAGE\nLINES = {nline}\nBANDS = {nband}\nLINE_SAMPLES = {nsample}\nSAMPLE_BITS = 32\nSAMPLE_TYPE = PC_REAL\nBAND_STORAGE_TYPE = LINE_INTERLEAVED\nINVALID_CONSTANT = -999\nEND_OBJECT = RFL_IMAGE\nCH1:THERMAL_CORRECTION_FLAG = "Y"\nCH1:PHOTOMETRY_CORR_FILE_NAME = "file.tab"\nCH1:GROUND_TRUTH_CORR_FILE_NAME = "N/A"\nCH1:PIXEL_LOCATION_FILE_NAME = "LOC.IMG"\nCH1:OBS_GEOMETRY_FILE_NAME = "OBS.IMG"\n'''
        (d/'TEST_RFL.HDR').write_text(hdr);(d/'TEST_L2.LBL').write_text(lbl)
        # Construct bytes with independent nested loops, not transpose/tobytes.
        with (d/'TEST_RFL.IMG').open('wb') as f:
            f.write(b'X'*offset)
            for row in range(nline):
                for band in range(nband):
                    for col in range(nsample):f.write(struct.pack('<f',100*row+10*band+col))
        vrt='<VRTDataset>'+''.join(f'<VRTRasterBand><Description>{w} nm</Description></VRTRasterBand>' for w in [600,750,950,1100])+'</VRTDataset>'
        (d/'TEST_RFL.IMG.vrt').write_text(vrt)
        return product

    def test_bil_axes_and_nonzero_offset(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);p=self.fixture(d);cube,*_=read_cube(d,p)
            for r in range(3):
                for c in range(2):
                    for k in range(4):self.assertEqual(cube[r,c,k],100*r+10*k+c)

    def test_truncation_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);p=self.fixture(d);f=d/'TEST_RFL.IMG';f.write_bytes(f.read_bytes()[:-4])
            with self.assertRaisesRegex(ValueError,'size'):read_cube(d,p)

    def test_wrong_endianness_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);p=self.fixture(d);f=d/'TEST_RFL.HDR';f.write_text(f.read_text().replace('byte order = 0','byte order = 1'))
            with self.assertRaisesRegex(ValueError,'little-endian'):read_cube(d,p)

    def test_wavelength_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);p=self.fixture(d);f=d/'TEST_RFL.IMG.vrt';f.write_text(f.read_text().replace('750 nm','751 nm'))
            with self.assertRaisesRegex(ValueError,'wavelength'):read_cube(d,p)

if __name__=='__main__':unittest.main(verbosity=2)

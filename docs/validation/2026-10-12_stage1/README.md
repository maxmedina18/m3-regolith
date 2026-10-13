# M3 Stage 1 Extraction Validation

## Observation

M3G20090107T011405

## Objective

Validate the numerical extraction of Moon Mineralogy Mapper (M3)
reflectance, location, and observation-geometry products using
independent binary reads and two extraction backends.

## Source products

Seven original source files were preserved:

- V01_RFL.IMG
- V01_RFL.HDR
- V01_RFL.IMG.vrt
- V02_LOC.IMG
- V02_LOC.HDR
- V02_OBS.IMG
- V02_OBS.HDR

All filenames include the observation prefix.

## Data layout

- Lines: 11935
- Samples: 304
- Reflectance bands: 85
- Location bands: 3
- Observation geometry bands: 10
- Interleave: BIL
- Byte order: little-endian
- RFL: Float32
- LOC: Float64
- OBS: Float32

The extraction preserves source-native spatial orientation.

## Extraction

The advisor-supplied Stage 1 extractor was executed using:

1. GDAL array backend
2. NumPy memmap backend

Each execution produced 99 TSV data tables, a data dictionary,
and a JSON manifest.

The 99 TSV files were compared using SHA-256 hashes.

Result: 99/99 identical.

## Independent numerical validation

A separate Python validation script reads selected values directly
from the original IMG binary files using independently calculated
BIL byte offsets.

The script uses a fixed random seed and tests boundary coordinates
and 100 pseudorandom coordinates per selected band.

Results:

- RFL: 840/840 passed
- LOC: 315/315 passed
- OBS: 1050/1050 passed
- Derived elevation: 105/105 passed

Total: 2310/2310 passed.

RFL Bands 1 and 2 were tested for expected bad-band masking.
Other selected bands were compared numerically against source data.

## Derived elevation

Elevation is calculated as:

    elevation_m = radius_m - 1737400

All 105 sampled elevation comparisons passed.

## Reproducibility

Run the validation script from the repository root:

    python tests/validation/validate_stage1_source.py

The script writes:

    docs/validation/2026-10-12_stage1/source_pixel_validation.json

The validation directory also contains available execution
metadata and environment records.

## Interpretation

The results establish strong agreement between the extraction
backends and independently sampled source-binary values.

They do not constitute exhaustive verification of every pixel
or independent confirmation of the scientific calibration.

The tests also do not independently establish geographic
alignment between the V01 RFL product and V02 LOC/OBS products.

## Remaining scientific validation

1. Confirm V01/V02 ancillary spatial alignment or obtain the
   corresponding V03 ancillary products.
2. Complete an independent NASAView numerical comparison.
3. Validate downstream spectral and geochemical interpretations
   separately.

## Conclusion

Stage 1 numerical extraction validation passed for the tested
observation and selected pixels.

The extraction workflow is suitable for continued development,
subject to the documented limitations.

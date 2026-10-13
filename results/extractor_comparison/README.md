# M³ original/advisor extractor comparison

## Objective and method

Compare the original repository reader at commit
`f176272f81c0307eddd273ba45c7ce7e2a3797fe` with the unchanged advisor
`M3_stage1_extractor_v2.py`. Two independent Python processes invoked the
respective reader APIs using Python 3.12.14 and NumPy 2.1.3. Each sampled the
same source-native pixels and all bands; no resampling or spectral processing.
This is a bounded reader comparison, not a run of the full 99-table advisor CLI.
The advisor source is not distributed; its required SHA-256 is in provenance.
The original reader source is preserved in `original_source/`.

## Inputs and coordinates

Snapshot `M3G20090107T011405`: V01 RFL IMG/HDR/VRT and L2 label; V02 LOC
and OBS IMG/HDR and L1B label. All original file sizes and SHA-256 hashes are
in `provenance.json`. No raw cubes or source labels are included.
RFL dimensions: 11935 × 304 × 85; LOC: 11935 × 304 × 3; OBS: 11935 × 304 × 10.

Zero-based (row, column): (0,0), (0,303), (11934,0), (11934,303),
(5967,152), (2983,76), (2983,228), (8950,76), (8950,228), (1,1),
(11933,302). Spectral band numbers are one-based 1–85.

## Results

| Quantity | Pairs | Max absolute error | MAE | RMSE | Exact mismatches |
|---|---:|---:|---:|---:|---:|
| RFL | 935 | 0 | 0 | 0 | 0 |
| Wavelength (µm) | 85 | 0 | 0 | 0 | 0 |
| OBS | 110 | 0 | 0 | 0 | 0 |
| LOC longitude (degrees) | 11 | 5.177306e-7 | 2.970289e-7 | 3.341968e-7 | 11 |
| LOC latitude (degrees) | 11 | 9.292047e-7 | 3.894589e-7 | 5.068706e-7 | 11 |
| LOC radius (m) | 11 | 0.05976345 | 0.03482966 | 0.03963314 | 11 |

The original reader unconditionally converts arrays to Float32. Advisor LOC
retains source Float64; casting its LOC samples to Float32 exactly reproduces
the original results. Mixed-unit aggregate LOC errors in the historical JSON
are retained for traceability but should not be interpreted physically.

All 22 sampled RFL NoData values (-999) agree; 913 valid reflectance pairs also
agree exactly. The original public header object drops NoData/BBL metadata;
the advisor retains it. Comparator validity uses the source/advisor ignore
value for both outputs. Plots hide bad bands and NoData; CSVs retain raw values.

## Reproduce and inspect

The 2026-10-13 UTC extraction is preserved. `comparison_spectra.csv`,
`ancillary_comparison.csv`, separate reader NPY arrays, metadata, logs, metrics,
and PNG/PDF overlays are the numerical evidence. `result_inventory.json` lists
relative paths and hashes of published artifacts, excluding itself.

Run from the repository root with NumPy installed:

```sh
python -B scripts/validate_extractor_comparison.py results/extractor_comparison
```

For a new extraction, obtain the exact hash-matching original input products
and advisor script separately. Copy `input_config.json` to a local file and
replace `${M3_DATA_ROOT}` with your data directory containing V01/ and V02/.
The config reader does not expand placeholders. Use an extraction interpreter
with Python 3.12.14 / NumPy 2.1.3 and an analysis interpreter with NumPy and
Matplotlib (historical analysis: Python 3.11.9, NumPy 2.4.4, Matplotlib 3.10.8).
Do not reuse an existing output directory.

```sh
python -B scripts/compare_original_extractors.py \
  --config /path/to/local-config.json \
  --advisor /path/to/M3_stage1_extractor_v2.py \
  --python /path/to/extraction-python \
  --output results/extractor_comparison_new
python -B scripts/prepare_comparison_publication.py results/extractor_comparison_new
python -B scripts/validate_extractor_comparison.py results/extractor_comparison_new
```

Historical provenance commands have host prefixes replaced by explicit
`${REPO_ROOT}`, `${M3_DATA_ROOT}`, `${EXTRACTION_PYTHON}` and
`${ANALYSIS_PYTHON}` placeholders. Source/input hashes are unchanged; the
inventory reflects sanitized publication bytes. The historical comparison
script is preserved unchanged. Other research results are untouched.

## Limitations and next validation

- Eleven pixels cannot establish whole-cube equality. Both readers use NumPy
  memmap, so separate processes do not provide independent decoding technology.
- Wavelength agreement checks metadata only. Source wavelength units are
  implicit; the original heuristic and protocol-based nm conversion agree here.
- NASAView/GDAL numerical comparisons, elevation and full-scene TSV
  serialization have not been validated by this run.
- V01 RFL's label references V03 ancillary files; prescribed V02 LOC/OBS were
  used. Matching dimensions do not prove spatial alignment across versions.
- No FeO calibration, chemical ground truth or mineral identification is tested.
- Next: independent NASAView/GDAL pixel checks; resolve ancillary versions;
  preserve Float64 LOC and source metadata, then validate serialized exports.

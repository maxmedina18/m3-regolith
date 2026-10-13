# Original vs advisor M³ extraction comparison

Original source: `f176272f81c0307eddd273ba45c7ce7e2a3797fe`; advisor unchanged SHA-256 `d190296d0ab7c7d61754c32857354b5342e4622738077e5637738ab2215ba855`.

Both extractors ran successfully in independent Python processes using their original reader APIs. The advisor CLI's full 99-table export was not run here. Wrappers select the same 11 source-native, zero-based row/column coordinates; all bands are retained. No resampling, normalization, or spectral preprocessing was applied.

Extraction environment: Python 3.12.14, NumPy 2.1.3. Exact executable, actual versions, commands, runtimes, source/input hashes and paths are in provenance.json and each reader's metadata.json.

## Numerical results

|Quantity|Values|Max absolute error|MAE|RMSE|Exact mismatches|
|---|---:|---:|---:|---:|---:|
|RFL|935|0|0|0|0|
|LOC|33|0.0597634476144|0.0116101163995|0.0228822028055|33|
|OBS|110|0|0|0|0|

LOC summary aggregates different physical units; use these per-plane errors:

- longitude degrees: maximum absolute error 5.17730605054e-07, MAE 2.970288781e-07, RMSE 3.34196844109e-07, mismatches 11/11.
- latitude degrees: maximum absolute error 9.29204652778e-07, MAE 3.8945890559e-07, RMSE 5.06870594311e-07, mismatches 11/11.
- radius metres: maximum absolute error 0.0597634476144, MAE 0.0348296627106, RMSE 0.0396331378436, mismatches 11/11.

Wavelength comparison (µm): {"selected_count": 85, "finite_pair_count": 85, "mismatched_values": 0, "max_absolute_error": 0.0, "mean_absolute_error": 0.0, "rmse": 0.0, "criterion": "Exact equality; matching NaNs accepted; errors calculated on finite pairs."}

## Interpretation and limitations

- Both readers report RFL shape [11935, 304, 85], LOC [11935, 304, 3], OBS [11935, 304, 10]. Layout/type codes, byte order and offsets agree. Original LOC output is float32; advisor preserves source float64. The original reader's unconditional float32 conversion causes the observed LOC precision loss.
- Original metadata omits NoData and BBL; advisor retains them. Invalid masks in this comparison use advisor/source-header NoData for both outputs, not a capability of the original reader. Valid-only and per-band errors are recorded separately in the metrics JSON.
- The original wavelength magnitude heuristic and explicit nm-to-µm conversion agree here. The header lacks an explicit unit; wavelength agreement is metadata consistency, not independent pixel validation.
- Only the listed pixels were extracted and compared. No claim of whole-cube equality, NASAView validation, FeO validation, or full Stage-1 compliance is made. Both implementations use NumPy memmap; independent process execution does not make them fully independent decoding technologies.
- V01 RFL label references V03 ancillary products; this run uses the advisor-prescribed V02 LOC/OBS. Equal dimensions do not establish cross-version spatial alignment.
- The original extractor has no elevation export. Elevation and full-scene TSV serialization were not compared in this run.

## Artifacts

See result_inventory.json for the relative path, byte size and SHA-256 of every generated artifact (except the inventory itself). Original and advisor outputs are isolated in their respective subdirectories. comparison_spectra.csv contains all 935 raw reflectance pairs, including sentinel values; ancillary_comparison.csv contains all 143 ancillary pairs.

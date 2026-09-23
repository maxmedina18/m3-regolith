# Concrete proposed architecture

## Decision

Use a deterministic Python pipeline as the scientific engine. Use **Codex as the supervised development/run operator now**. Later implement one **M3 Research Runner using the OpenAI Agents SDK for Python** to call explicitly defined tools. Do not start with multiple agents or ML abundance prediction. The agent service, downloading and geolocation stages below are proposals, not implemented capabilities.

## Tools and data flow

1. `discover_products(query)`: query NASA PDS metadata; return candidate observation IDs, versions, archive URLs and ancillary relationships. A human verifies initial study footprint.
2. `download_products(manifest)`: download selected RFL/HDR/LBL, referenced LOC/OBS and relevant calibration files into immutable per-observation folders. Record URL, retrieval time, byte length, SHA-256 and published checksum result where supplied. Resume interrupted transfers without silently replacing raw inputs.
3. `validate_products(manifest)`: check metadata, sizes, missing values, corrections and associated versions. Compare the current NumPy reader against GDAL/Rasterio reading the VRT: dimensions, sample values, full-array discrepancy summaries and nodata. Investigate differences before interpreting chemistry. This independent reader is the next implementation target.
4. `run_feo_experiment(config)`: call the tested Python engine. All constants, angle conventions and band choices live in versioned JSON. Retain full outputs and failure flags. Agent may choose an approved configuration, never quietly fit or change constants to obtain plausible colors.
5. `geolocate_swath(manifest)`: read label-referenced LOC/OBS; verify row/column alignment and reference frame; attach longitude/latitude per valid pixel. Check footprint against a lunar basemap. Use GDAL/Rasterio and a documented lunar CRS for a projected raster only after alignment is established. Publish located points/swath before mosaicking.
6. `validate_calibration(reference_table)`: reproduce the specified published convention and revised calibration when available. Match Apollo/reference footprints and units, fit only on designated training references, hold out independent sites, evaluate residuals/bias and spatial leakage. Calibration-domain masks and empirically supported uncertainty are deliverables.
7. `build_report(run_id)`: assemble fixed figures, source IDs, code/config hashes, limitations and validation state. Agent summarizes these structured results with direct references. Never upgrade a diagnostic into a validated product through wording.

## Stack

- Current science: Python, NumPy, Matplotlib; JSON configs/manifests, CSV selected pixels, compressed NPZ arrays; unittest tests.
- Next geospatial/reader layer: GDAL with Rasterio, matched PDS LOC/OBS; lunar CRS and GeoTIFF when genuinely georeferenced.
- Versioning: Git for code/configs; immutable raw files and per-run output directories. Record large-data checksums rather than committing binaries.
- Later operator: single OpenAI Agents SDK Python agent with explicit tool schemas and traceable tool results. Model configurable; scientific logic stays outside the model.
- Later ML: scikit-learn random-forest baseline only after a validated reference set and deterministic baseline exist. Use held-out sites, uncertainty evaluation and a separate Al target/reference branch. Do not train on our unvalidated equation outputs and call them ground truth.

## Why this agent

The SDK supports a code-first Python agent that calls application-owned tools. That fits a numerical Python workflow: the agent handles discovery, sequencing and summaries; code handles numbers; the researcher decides scientific assumptions. Codex is already the supervised operator used to build/run this prototype. No new agent has been deployed or benchmarked; SDK adoption is a concrete proposed next architecture, not a claim of superior scientific accuracy.

External documents are evidence, not executable instructions. Download tool paths and archive destinations must be constrained. Scientific configuration changes require review and a new run, not mutation of historical outputs.

Official architecture references:
https://developers.openai.com/api/docs/guides/agents/sdk
https://learn.chatgpt.com/docs/non-interactive-mode

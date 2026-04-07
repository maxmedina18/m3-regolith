# Roadmap

## Phase 1 (this scaffold)

- Synthetic cube generation to enable end-to-end development.
- Minimal `.npz` loader.
- Preprocessing + feature extraction + baseline modeling + plots.

## Phase 2 (real M³ ingestion)

- Implement M³ PDS product ingestion.
- Store georeferencing metadata if available (projection, lat/lon grids).
- Standardize reflectance units and bandpass definitions.

## Phase 3 (spectroscopy improvements)

- Convex-hull continuum removal and uncertainty propagation.
- Robust outlier handling and per-band quality masks.
- Optional photometric corrections if applicable.

## Phase 4 (labels and scientific validation)

- Literature-driven mineral classes and reference spectra.
- Cross-region validation: test class stability across illumination/terrain.
- Quantify feature sensitivity to smoothing and continuum choices.

## Phase 5 (publication-quality outputs)

- Reproducible figure generation scripts.
- Map outputs with consistent colorbars and metadata.
- Structured experiment tracking (config files, saved artifacts).

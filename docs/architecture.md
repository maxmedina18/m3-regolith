# Architecture

## Design goals

- Scientific correctness and interpretability over complexity.
- Modular separation: loading → preprocessing → features → models → visualization → validation.
- Reproducible execution via scripts and fixed seeds.

## Core data model

`M3Cube` is the only object shared across stages.

- `cube.data`: `(rows, cols, bands)` reflectance
- `cube.wavelengths`: `(bands,)` wavelengths in microns
- `cube.get_pixel_spectrum(x, y)`: retrieve spectrum for a pixel

## Pipeline stages

1. **Data loading** (`lunar_m3.data_loading`)
   - Development loader supports `.npz` and synthetic cubes.
   - Real M³ parsing is isolated to this module.

2. **Preprocessing** (`lunar_m3.preprocessing`)
   - Normalization, smoothing, denoising, and continuum removal.
   - Functions are pure (array-in/array-out) to ease testing.

3. **Features** (`lunar_m3.features`)
   - Physically motivated features: band depth/center/area, slopes, ratios.
   - Feature definitions are explicit and testable.

4. **Models** (`lunar_m3.models`)
   - Baseline interpretable classifiers with clear evaluation.
   - Input is a feature table; output is predicted labels + metrics.

5. **Visualization** (`lunar_m3.visualization`)
   - Publication-style spectra plots and map visualizations.

6. **Validation** (`lunar_m3.validation`)
   - Engineering checks (finite values, shape integrity).
   - Scientific sanity checks (band centers in plausible ranges).

## Interfaces and boundaries

- Data loading returns an `M3Cube`; no other stage reads raw files.
- Scripts call library functions; notebooks should call the same library.

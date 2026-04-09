# Lunar M³ Project

Research-grade Python project scaffold for analyzing NASA Moon Mineralogy Mapper (M³) hyperspectral reflectance data.

This repository is organized around a real-data lunar spectroscopy workflow for NASA Moon Mineralogy Mapper (M³) reflectance cubes.

## Quickstart

```bash
cd lunar_m3_project
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
pip install -e ".[dev]"
```

## Real-data workflow

The intended workflow is:

1) Load real M³ reflectance cube (`.IMG/.HDR`)
2) Preprocess spectra (cleanup → normalization → smoothing → join mitigation)
3) Extract physics-informed spectral features (continuum removal → band detection → band metrics → slope)
4) Generate feature maps
5) Run unsupervised clustering (GMM) in feature space
6) Interpret clusters physically (mafic strength, feldspathic candidates, maturity proxy)

### 1–3) Extract features from a real M³ cube

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --input /path/to/M3..._RFL.IMG \
  --output-dir artifacts
```

This writes the canonical per-pixel table to:
- `artifacts/features/features.csv`

### 3) QA: plot detected ~1 µm and ~2 µm bands

```bash
PYTHONPATH=src python scripts/qa_plot_band_detection.py \
  --input /path/to/M3..._RFL.IMG \
  --output-dir artifacts/qa_spectra \
  --num-samples 12
```

### 4) Generate feature maps

```bash
PYTHONPATH=src python scripts/make_feature_maps.py \
  --features-csv artifacts/features/features.csv \
  --output-dir artifacts/feature_maps
```

### 5) Run unsupervised clustering (GMM)

```bash
PYTHONPATH=src python scripts/run_unsupervised_clustering.py \
  --features-csv artifacts/features/features.csv \
  --output-dir artifacts/clustering \
  --n-clusters 4
```

### 6) Physical interpretation (how to read the outputs)

- High `bd_1um` + high `bd_2um` clusters are consistent with stronger mafic absorptions (Fe-bearing phases).
- High `brightness_ref` + weak band depths clusters are consistent with more feldspathic/plagioclase-rich candidates.
- High `slope_global` with suppressed band depths is consistent with a maturity/space-weathering proxy.

## Outputs (what gets generated and where)

The primary pipeline entrypoint is `scripts/run_pipeline.py`.

The project uses a canonical research artifacts layout under `artifacts/`:

- `artifacts/features/`
  - `features.csv`: one row per pixel with extracted spectral features
- `artifacts/qa_spectra/`
  - `qa_band_detection_*.png`: per-pixel QA plots with detected band bounds and centers
  - `spectrum_example_*.png`: example raw vs processed spectrum
- `artifacts/feature_maps/`
  - `map_bd_1um.png`, `map_bc_1um.png`, `map_bd_2um.png`, `map_bc_2um.png`, `map_slope_global.png`, `map_brightness_ref.png`
- `artifacts/clustering/`
  - `cluster_map.png`: spatial map of argmax cluster label
  - `cluster_prob_maps.png`: probability panels `P(cluster=k)` for uncertainty inspection
  - `cluster_summary.csv`: cluster means/stds + pixel counts + interpretation scaffold
  - `cluster_feature_means.csv`: compact mean table for quick comparison

Interpretation tips (aligned with our project goals):

- `bd_1um`, `bc_1um`, `ba_1um` capture the strength/position/area of the ~1 µm mafic absorption.
- `bd_2um`, `bc_2um`, `ba_2um` capture the strength/position/area of the ~2 µm pyroxene-related absorption.
- `slope_*` features capture spectral reddening and are useful for accounting for maturity/space weathering.
- `band1_left_um`, `band1_right_um`, `band2_left_um`, `band2_right_um` are emitted for QA/debugging: they show what wavelength interval was actually used for the band measurements.

## Project layout

- `src/lunar_m3/` contains the library code (import as `lunar_m3`).
- `scripts/` contains runnable entrypoints.
- `docs/` contains architecture, methodology, and roadmap.
- `tests/` contains unit tests.

## Using real data

The supported input is PDS Imaging Node ENVI-style `.HDR` + `.IMG` reflectance products (e.g., `*_RFL.IMG`).

If you have files like `*_RDN.HDR` and `*_RDN.IMG`, you can load them directly (note: radiance products are not ideal for band-depth mineral analysis without additional calibration):

```python
from lunar_m3.data_loading import load_m3_cube

cube = load_m3_cube("/path/to/M3..._RDN.IMG")
```

To plug in other real M³ products, implement parsing in `src/lunar_m3/data_loading/m3_loader.py` and return an `M3Cube` instance with:

- `data`: `float32` array shaped `(rows, cols, bands)`
- `wavelengths`: `float64` array shaped `(bands,)` in microns

Downstream modules only require the `M3Cube` API.

## Reproducibility

- Feature extraction and clustering are deterministic given a fixed cube and seed.

## Next steps

See:
- `docs/architecture.md`
- `docs/methodology.md`
- `docs/roadmap.md`

# Lunar M³ Project

Research-grade Python project scaffold for analyzing NASA Moon Mineralogy Mapper (M³) hyperspectral reflectance data.

This repository starts with a synthetic development pipeline (so you can run everything immediately) and is structured so real M³ ingestion can replace the synthetic loader without changing downstream code.

## Quickstart

```bash
cd lunar_m3_project
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
pip install -e ".[dev]"
```

Run the end-to-end baseline pipeline (synthetic cube):

```bash
python scripts/run_pipeline.py --output-dir artifacts

# Fe + Al-proxy report (synthetic cube)
python scripts/run_fe_al_report.py --output-dir artifacts_fe_al
```

## Outputs (what gets generated and where)

The primary pipeline entrypoint is `scripts/run_pipeline.py`.

Given `--output-dir <DIR>`, it writes:

- `<DIR>/features.csv`: one row per pixel with extracted spectral features
- `<DIR>/classification_report.txt`: model metrics (synthetic mode only)
- `<DIR>/predicted_map.png`: per-pixel predicted class map (synthetic mode only)
- `<DIR>/spectrum_example.png`: example raw vs processed spectrum

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

Phase 1 uses a synthetic cube and a simple `.npz` loader. It also supports many PDS Imaging Node ENVI-style `.HDR` + `.IMG` products.

If you have files like `*_RDN.HDR` and `*_RDN.IMG`, you can load them directly:

```python
from lunar_m3.data_loading import load_m3_cube

cube = load_m3_cube("/path/to/M3..._RDN.IMG")
```

To plug in other real M³ products, implement parsing in `src/lunar_m3/data_loading/m3_loader.py` and return an `M3Cube` instance with:

- `data`: `float32` array shaped `(rows, cols, bands)`
- `wavelengths`: `float64` array shaped `(bands,)` in microns

Downstream modules only require the `M3Cube` API.

## Reproducibility

- Deterministic synthetic data generation using fixed random seeds.
- Modeling uses fixed seeds where applicable.

## Next steps

See:
- `docs/architecture.md`
- `docs/methodology.md`
- `docs/roadmap.md`

# Lunar M³ Spectroscopy Mapper

Research software for analyzing Chandrayaan-1 Moon Mineralogy Mapper (M³) hyperspectral reflectance data, with an initial focus on reproducing and validating FeO estimation in the Apollo 17 region.

## Current Research Objective

The current objective is to establish a reproducible, physically defensible baseline before attempting mineralogical classification or machine-learning interpretation.

The working research chain is:

**M³ orbital reflectance → verified data ingestion → quality diagnostics → location and observing-geometry recovery → spectral observables → FeO calibration reproduction → validation against reference chemistry → georeferenced maps → later mineralogical and machine-learning extensions → eventual ISRU assessment**

The immediate target is reproduction of a published Apollo 17 FeO workflow using M³ Level 2 reflectance data.

At the current stage, generated FeO values are **provisional diagnostic outputs**, not validated iron-abundance measurements.

---

## Current Status — October 2026

**Stage-1 extraction validation has advanced substantially; independent NASAView checks and ancillary-product alignment remain open.** The current scientific focus is trustworthy data ingestion, not validated FeO mapping.

### Stage-1 Extraction Validation (October 12, 2026)

The tested observation is `M3G20090107T011405` (11,935 lines × 304 samples; 85 RFL bands, 3 LOC planes, and 10 OBS planes).

Two related but distinct comparisons are documented:

**1. Original project reader vs. advisor-supplied reader**

Both implementations were executed in separate processes at 11 shared pixel coordinates.

Across those samples:

- **935 RFL values matched exactly.**
- **110 OBS values matched exactly.**
- **85 wavelengths matched exactly.**
- LOC values did not match bit-for-bit.

The historical project reader converted source Float64 LOC data to Float32, resulting in maximum absolute differences of approximately:

| Quantity | Maximum Absolute Difference |
|---|---:|
| Longitude | 5.18 × 10⁻⁷ degrees |
| Latitude | 9.29 × 10⁻⁷ degrees |
| Lunar radius | 0.0598 metres |

Both readers use NumPy memmap, so this is a cross-implementation check, **not a comparison of fully independent decoding technologies**.

The advisor extractor's full 99-table CLI export was not part of this particular comparison.

**2. Advisor Stage-1 extractor: GDAL-array vs. NumPy-memmap**

Each backend produced **99 TSV tables**, with **99/99 identical output hashes**.

Separately, sampled checks using calculated source-binary offsets passed **2,310/2,310 comparisons**.

| Product | Successful Checks |
|---|---:|
| RFL | 840/840 |
| LOC | 315/315 |
| OBS | 1,050/1,050 |
| Derived elevation | 105/105 |
| **Total** | **2,310/2,310** |

These checks cover selected pixels and bands, not every source pixel.

### Validation Evidence

The following files preserve the numerical comparisons, procedures, and validation results:

**Original vs. advisor extractor comparison**

- [`comparison_summary.md`](results/extractor_comparison/comparison_summary.md) — Readable summary of the comparison.
- [`comparison_metrics.json`](results/extractor_comparison/comparison_metrics.json) — Detailed numerical comparison metrics.
- [`comparison_spectra.csv`](results/extractor_comparison/comparison_spectra.csv) — Sampled spectral values.
- [`results/extractor_comparison/README.md`](results/extractor_comparison/README.md) — Procedures, artifacts, and provenance.

**GDAL vs. NumPy memmap validation**

- [`Stage-1 Validation Documentation`](docs/validation/2026-10-12_stage1/README.md) — Validation methodology.
- [`validation_results.json`](docs/validation/2026-10-12_stage1/validation_results.json) — Output-hash comparisons.
- [`source_pixel_validation.json`](docs/validation/2026-10-12_stage1/source_pixel_validation.json) — Direct source-binary checks.

### Outstanding Validation

The following steps remain before Stage-1 validation can be considered complete:

- Independently compare numerical pixel values from original PDS products using NASAView.
- Confirm NASAView indexing conventions, pixel orientation, and band numbering.
- Resolve ancillary-product compatibility. The V01 RFL label references V03 ancillary products, whereas the recorded extraction uses V02 LOC/OBS.
- Finalize source provenance, checksums, environment records, and unresolved warnings.
- Independently validate FeO calibration, scene matching, reference chemistry, and mapping.

Matching image dimensions alone does not establish geographic alignment between different product versions.

The September FeO prototype and earlier feature/GMM tools remain available as exploratory research, not validated mineralogical or geochemical products.

---

## Scientific Workflow

### Phase 1 — Data Integrity

1. Identify the exact M³ observation and product version.
2. Verify binary layout, dimensions, wavelengths, units, and metadata.
3. Compare implementations and decoding backends, recording numerical coverage, precision, and limitations.
4. Characterize invalid values, striping, and other data-quality issues.

### Phase 2 — Geometry and Geolocation

Recover and validate the associated M³ location and observation products.

The reflectance cube alone is not sufficient for a scientifically defensible lunar map.

LOC/OBS information is required to associate image pixels with lunar coordinates and observing geometry.

### Phase 3 — FeO Reproduction

The provisional FeO calculation uses reflectances near 750 nm and 950 nm.

For the original calibration candidate:

```text
x = R750 - 0.08
y = R950 / R750 - 1.19
theta = -atan2(y, x)

FeO (wt%) = 17.427 * theta - 7.565
```

These constants and the angle convention are currently under audit.

The repository preserves diagnostic behavior rather than clipping or silently correcting physically implausible outputs. This allows failures in the reproduction attempt to remain visible and testable.

### Phase 4 — Validation

Before interpreting FeO estimates as lunar chemistry:

1. Reproduce the intended published calibration.
2. Resolve angle, sign, and wrapping conventions.
3. Obtain the required geolocation and observing-geometry products.
4. Match orbital observations to appropriate reference sample regions.
5. Compare estimates with reference chemistry using quantitative error metrics.
6. Document uncertainty and calibration sensitivity.

Only after these gates are satisfied should the project produce a validated georeferenced FeO product.

---

## Spectral Features and Machine Learning

The repository also contains earlier infrastructure for:

- Continuum removal
- ~1 µm and ~2 µm absorption features
- Band depth, center, and area
- Spectral slopes
- Feature maps
- Gaussian mixture model (GMM) clustering

These components are retained as experimental tools for later stages of the project.

They are **not currently treated as validated mineral-abundance or compositional classifiers**.

Machine-learning methods will be reconsidered after the deterministic data, calibration, geolocation, and validation pipeline is established.

---

## Installation

Python **3.11 or newer** is required.

Clone the repository:

```bash
git clone https://github.com/maxmedina18/m3-regolith.git
cd m3-regolith
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
python -m pip install -e .
```

For development and testing:

```bash
python -m pip install pytest
python -m pytest -v
```

---

## Project Layout

```text
m3-regolith/
├── configs/                          # FeO calibration configuration
├── docs/
│   ├── research/                     # September prototype and research plans
│   └── validation/
│       └── 2026-10-12_stage1/        # October Stage-1 validation evidence
├── results/
│   ├── 2026-09-18_feo_prototype/      # Provisional FeO results
│   └── extractor_comparison/         # Original vs. advisor reader evidence
├── scripts/                          # Processing and validation entry points
├── src/
│   └── lunar_m3/
│       ├── stage1.py                 # Stage-1 functionality
│       ├── data_loading/            # Active readers
│       └── feo.py                    # Provisional FeO implementation
└── tests/                             # Unit and source-validation tests
```

### Important Validation Scripts

- `scripts/compare_original_extractors.py`
- `scripts/validate_gdal_sources.py`
- `scripts/validate_extractor_comparison.py`
- `tests/validation/validate_stage1_source.py`

Historical reader source under `results/extractor_comparison/original_source/` is retained as comparison provenance. It is **not the active package implementation**.

The advisor-supplied standalone extractor is not distributed here as a tracked source file; its recorded checksum and execution metadata are preserved in the comparison artifacts.

---

## FeO Prototype

The current prototype implementation is located at:

```text
src/lunar_m3/feo.py
```

The associated calibration configuration is stored at:

```text
configs/original_calibration.json
```

Prototype tests are located at:

```text
tests/test_feo_prototype.py
```

A documented prototype run is preserved under:

```text
results/2026-09-18_feo_prototype/
```

This includes diagnostic figures, selected-pixel information, run metadata, and summary information.

These outputs document the behavior of the provisional implementation and should **not** be interpreted as a validated lunar FeO map.

---

## Research Documentation

Research documentation is available under `docs/research/`.

Important files include:

- `research_plan.md` — Initial research plan and validation gates.
- `equation_register.md` — Equations, definitions, assumptions, and provenance.
- `prototype_architecture.md` — Architecture of the provisional FeO workflow.
- `M3_Feo_prototype_report_2026-09-18.pdf` — Documented prototype run and findings.
- `agent_operator_prompt.md` — Proposed future automation/operator design.

Automation is not considered a substitute for the deterministic scientific pipeline and is deferred until the underlying reader, calibration, validation, and mapping workflow is established.

---

## Earlier Spectroscopy Pipeline

Earlier development produced infrastructure for:

- Spectral preprocessing
- Feature extraction
- Feature mapping
- Continuum removal
- Absorption-band analysis
- Unsupervised GMM clustering

Existing scripts include tools for:

```text
run_pipeline.py
run_preprocessing.py
run_feature_extraction.py
make_feature_maps.py
qa_plot_band_detection.py
run_unsupervised_clustering.py
```

This work is retained because it may become useful after the FeO reproduction and validation stages.

However, spectral clusters and absorption features are currently treated as **exploratory observables**, not direct measurements of mineral abundance or bulk chemistry.

---

## Data Management

Raw M³ archive products are intentionally excluded from Git.

The repository should store information required to identify and reproduce a dataset rather than duplicating large planetary-science archive products.

Data provenance should include, where available:

- Product identifiers
- Observation identifiers
- Archive source
- Product version
- Filenames
- Checksums
- Acquisition metadata
- Processing level
- Associated LOC/OBS products
- Role of each product in the analysis

Local raw data belong under:

```text
data/raw/
```

and are ignored by Git.

---

## Reproducibility

The project aims for deterministic and auditable scientific runs.

A reproducible run should preserve:

- Input product identifiers
- Input hashes where appropriate
- Calibration constants
- Configuration
- Code version
- Quality masks
- Processing choices
- Diagnostic outputs
- Run metadata

The current prototype records diagnostic outputs and run metadata so calculations can be reproduced and audited.

Historical tests were reported passing under Python 3.11.9 at an earlier checkpoint.

The full integrated test suite has not been rerun as part of this README update; the October validation records document their own specific checks.

---

## Scientific Caution

Reflectance, spectral absorption behavior, FeO concentration, mineral abundance, and engineering material suitability are **distinct physical quantities**.

A spectral feature or cluster should not be interpreted as a chemical abundance measurement without an appropriate calibration and validation chain.

Likewise, an FeO estimate alone is insufficient to establish construction-material or ISRU suitability. Those interpretations require additional mineralogical, chemical, physical, and engineering evidence.

---

## Near-Term Roadmap

1. Complete NASAView pixel-level verification and document the Stage-1 audit.
2. Resolve V01 RFL versus V02/V03 LOC/OBS compatibility and verify spatial alignment.
3. Resolve the published FeO calibration implementation.
4. Validate FeO estimates against appropriately matched reference chemistry.
5. Generate a quality-masked, georeferenced FeO swath.
6. Investigate additional spectral and mineralogical observables.
7. Evaluate whether machine learning adds validated predictive value.
8. Assess implications for lunar ISRU only after compositional validation.

See `docs/research/research_plan.md` for the detailed research plan.

---

## Research Direction

The goal of this repository is not simply to generate visually plausible lunar maps.

The goal is to build a chain in which every major scientific inference can be traced from the original M³ observation through processing, calibration, validation, and uncertainty assessment.

The long-term research direction is:

**M³ reflectance → validated compositional information → mineralogical interpretation → spatial mapping → conditional lunar ISRU assessment**

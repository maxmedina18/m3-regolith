# Lunar M³ Spectroscopy Mapper

Research software for analyzing Chandrayaan-1 Moon Mineralogy Mapper (M³) hyperspectral reflectance data, with an initial focus on reproducing and validating FeO estimation in the Apollo 17 region.

## Current Research Objective

The current objective is to establish a reproducible, physically defensible baseline before attempting mineralogical classification or machine-learning interpretation.

The working research chain is:

**M³ orbital reflectance → verified data ingestion → quality diagnostics → location and observing-geometry recovery → spectral observables → FeO calibration reproduction → validation against reference chemistry → georeferenced maps → later mineralogical and machine-learning extensions → eventual ISRU assessment**

The immediate target is reproduction of a published Apollo 17 FeO workflow using M³ Level 2 reflectance data.

At the current stage, generated FeO values are **provisional diagnostic outputs**, not validated iron-abundance measurements.

---

## Current Status

The repository currently includes:

- real-data M³ reflectance loading and inspection
- reproducible cube diagnostics
- 750 nm and 950 nm reflectance extraction
- a provisional implementation of the published FeO relation
- explicit testing of angle/sign conventions
- data-quality diagnostics
- reproducible prototype outputs and run metadata
- unit tests for the FeO prototype
- earlier spectral-feature and GMM infrastructure retained for later experiments

The current FeO implementation is intentionally treated as a **reproduction candidate rather than a validated abundance model**.

Calibration transfer, angle convention, scene matching, geolocation, and comparison with reference chemistry remain active validation tasks.

---

## Scientific Workflow

### Phase 1 — Data Integrity

1. Identify the exact M³ observation and product version.
2. Verify binary layout, dimensions, wavelengths, units, and metadata.
3. Compare the current reader against an independently configured compatible reader.
4. Characterize invalid values, striping, and other data-quality issues.

### Phase 2 — Geometry and Geolocation

Recover and validate the associated M³ location and observation products.

The reflectance cube alone is not sufficient for a scientifically defensible lunar map. LOC/OBS information is required to associate image pixels with lunar coordinates and observing geometry.

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

- continuum removal
- ~1 µm and ~2 µm absorption features
- band depth, center, and area
- spectral slopes
- feature maps
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
├── configs/
│   └── original_calibration.json
├── data/
│   ├── manifests/
│   ├── reference/
│   └── raw/                  # ignored by Git
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   ├── roadmap.md
│   └── research/
│       ├── research_plan.md
│       ├── equation_register.md
│       ├── prototype_architecture.md
│       ├── agent_operator_prompt.md
│       └── M3_Feo_prototype_report_2026-09-18.pdf
├── notebooks/
├── results/
│   └── 2026-09-18_feo_prototype/
├── scripts/
├── src/
│   └── lunar_m3/
│       └── feo.py
└── tests/
    └── test_feo_prototype.py
```

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

Current research documentation is available under `docs/research/`.

Important files include:

- `research_plan.md` — first-four-weeks research plan and validation gates
- `equation_register.md` — equations, definitions, assumptions, and provenance
- `prototype_architecture.md` — architecture of the provisional FeO workflow
- `M3_Feo_prototype_report_2026-09-18.pdf` — documented prototype run and current findings
- `agent_operator_prompt.md` — proposed future automation/operator design

Automation is not considered a substitute for the deterministic scientific pipeline and is deferred until the underlying reader, calibration, validation, and mapping workflow is established.

---

## Earlier Spectroscopy Pipeline

Earlier development produced infrastructure for:

- spectral preprocessing
- feature extraction
- feature mapping
- continuum removal
- absorption-band analysis
- unsupervised GMM clustering

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

- product identifiers
- observation identifiers
- archive source
- product version
- filenames
- checksums
- acquisition metadata
- processing level
- associated LOC/OBS products
- role of each product in the analysis

Local raw data belong under:

```text
data/raw/
```

and are ignored by Git.

---

## Reproducibility

The project aims for deterministic and auditable scientific runs.

A reproducible run should preserve:

- input product identifiers
- input hashes where appropriate
- calibration constants
- configuration
- code version
- quality masks
- processing choices
- diagnostic outputs
- run metadata

The current prototype records diagnostic outputs and run metadata so calculations can be reproduced and audited.

At the current development checkpoint, the integrated project test suite passes under Python 3.11.9.

---

## Scientific Caution

Reflectance, spectral absorption behavior, FeO concentration, mineral abundance, and engineering material suitability are **distinct physical quantities**.

A spectral feature or cluster should not be interpreted as a chemical abundance measurement without an appropriate calibration and validation chain.

Likewise, an FeO estimate alone is insufficient to establish construction-material or ISRU suitability. Those interpretations require additional mineralogical, chemical, physical, and engineering evidence.

---

## Near-Term Roadmap

1. Independently verify M³ reading and data quality.
2. Obtain and align the appropriate LOC/OBS products.
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

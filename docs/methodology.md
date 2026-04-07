# Methodology (Spectroscopy Logic)

This project focuses on interpretable, physically motivated analysis of reflectance spectra from NASA Moon Mineralogy Mapper (M³) data.

## Wavelength conventions

- Wavelengths are represented in **microns** (µm).
- M³ reflectance spectra are treated as per-pixel vectors `R(λ)`.

## Preprocessing steps

### 1) Quality screening

- Reject spectra with non-finite values.
- Optionally reject pixels with negative reflectance or extremely low signal.

### 2) Normalization

Normalization reduces brightness effects so absorption behavior is comparable.

Implemented options:
- Unit scaling by median reflectance within a reference window.

### 3) Smoothing (Savitzky–Golay)

Smoothing reduces high-frequency noise while preserving band shapes. A Savitzky–Golay filter is used because it is linear and locally polynomial, making it easy to reason about.

### 4) Continuum removal

Continuum removal isolates absorption band shapes by dividing the spectrum by a fitted continuum over a wavelength interval.

In Phase 1+, continuum is approximated using a straight-line continuum between band endpoints. This is a development-friendly method that is transparent and testable.

Key update:
- Continuum removal now happens **before** band measurements, and band endpoints are detected automatically (see below) rather than always using fixed windows.

### 5) Instrument join mitigation (~1.34 µm)

M³ often exhibits an instrument join discontinuity around ~1.34 µm. This can distort band fitting and slope estimation.

The pipeline mitigates this by:
- Identifying wavelengths in a configurable join window (default `1.325–1.355 µm`).
- Replacing join samples via **linear interpolation** between the nearest samples outside the join.
- Downweighting join wavelengths during slope fitting (default factor `0.2`).

## Features

Features are extracted around two physically motivated absorptions:

- **~1 µm band** (iron-related absorptions; mafic minerals): search window `0.85–1.30 µm`
- **~2 µm band** (iron-related absorptions; pyroxene-related): search window `1.70–2.50 µm`

### Automated band detection

Within each search window, the pipeline:

1) Performs a seed continuum removal over the search window.
2) Detects the band minimum and refines the band center via a local parabola fit.
3) Estimates left/right band edges by finding where continuum-removed reflectance returns near baseline (`~1.0 ± tol`).
4) Re-runs continuum removal using the detected edges.
5) Measures band depth/center/area within the detected band interval.

The detector ignores the instrument join interval during band detection.

Implemented feature families:

- **Band center**: wavelength of minimum continuum-removed reflectance in a window
- **Band depth**: `1 - min(R_c)` where `R_c` is continuum-removed reflectance
- **Band area**: numerical integral of `(1 - R_c)` across the window
- **Spectral slope**: weighted linear fit slope `dR/dλ` in a window (with join downweighting)
- **Reflectance ratios**: simple ratios at diagnostically useful wavelengths

Additional band geometry outputs (per pixel):
- Detected band bounds: `band1_left_um`, `band1_right_um`, `band2_left_um`, `band2_right_um`

## Baseline modeling

A first-pass classifier is trained on feature vectors.

Supported models:
- Logistic Regression (interpretable linear decision surface)
- Random Forest (nonlinear baseline, feature importance)
- SVM (margin-based baseline)

Phase 1 uses synthetic labels to prove plumbing. Real labels should be derived from literature-based spectral classes or expert annotation.

## Validation philosophy

- Engineering validation: tests for deterministic outputs, shape handling, numerical stability.
- Scientific sanity checks: confirm band centers fall within plausible windows and band depths are non-negative.

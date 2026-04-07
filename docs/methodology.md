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

In Phase 1, continuum is approximated using a straight-line continuum between the interval endpoints. This is a development-friendly method that is transparent and testable. Later phases can replace this with convex-hull continuum removal if needed.

## Features

Features are extracted from two physically motivated regions:

- **~1 µm region** (iron-related absorptions; mafic minerals): nominal window `0.85–1.30 µm`
- **~2 µm region** (iron-related absorptions; pyroxene-related): nominal window `1.70–2.50 µm`

Implemented feature families:

- **Band center**: wavelength of minimum continuum-removed reflectance in a window
- **Band depth**: `1 - min(R_c)` where `R_c` is continuum-removed reflectance
- **Band area**: numerical integral of `(1 - R_c)` across the window
- **Spectral slope**: linear fit slope `dR/dλ` in a window
- **Reflectance ratios**: simple ratios at diagnostically useful wavelengths

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

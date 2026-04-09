from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lunar_m3.data_loading.m3_loader import M3Cube


def generate_synthetic_cube(
    *,
    rows: int = 50,
    cols: int = 60,
    bands: int = 86,
    seed: int = 0,
    wavelengths_um: Optional[np.ndarray] = None,
) -> M3Cube:
    rng = np.random.default_rng(seed)

    if wavelengths_um is None:
        wavelengths_um = np.linspace(0.62, 2.60, bands, dtype=np.float64)

    labels = rng.integers(0, 3, size=(rows, cols), dtype=np.int32)

    base = np.zeros((rows, cols, bands), dtype=np.float32)
    for y in range(rows):
        for x in range(cols):
            label = int(labels[y, x])
            spectrum = _synthetic_spectrum(wavelengths_um, label=label, rng=rng)
            base[y, x, :] = spectrum.astype(np.float32)

    cube = M3Cube(data=base, wavelengths=wavelengths_um)
    cube.synthetic_labels = labels
    return cube


def _synthetic_spectrum(wavelengths_um: np.ndarray, *, label: int, rng: np.random.Generator) -> np.ndarray:
    w = wavelengths_um

    if label == 0:
        continuum = 0.35 + 0.08 * (w - w.min()) / (w.max() - w.min())
        absorptions = 0.015 * _gaussian(w, mu=1.05, sigma=0.08) + 0.01 * _gaussian(w, mu=2.05, sigma=0.10)
    elif label == 1:
        continuum = 0.25 + 0.10 * (w - w.min()) / (w.max() - w.min())
        absorptions = 0.11 * _gaussian(w, mu=1.02, sigma=0.07) + 0.08 * _gaussian(w, mu=2.12, sigma=0.11)
    elif label == 2:
        continuum = 0.48 + 0.02 * (w - w.min()) / (w.max() - w.min())
        absorptions = 0.02 * _gaussian(w, mu=1.25, sigma=0.10)
    else:
        raise ValueError(f"Unknown synthetic label: {label}")

    reflectance = continuum * (1.0 - absorptions)
    noise = rng.normal(loc=0.0, scale=0.004, size=w.shape)
    reflectance = reflectance + noise
    return np.clip(reflectance, 0.0, 1.5)


def _gaussian(w: np.ndarray, *, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((w - mu) / sigma) ** 2)


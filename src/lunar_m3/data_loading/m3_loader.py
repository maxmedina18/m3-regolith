from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .pds3_envi import read_envi_header, read_envi_image


@dataclass
class M3Cube:
    """Development-friendly abstraction for an M³ reflectance cube.

    Attributes:
        data: Reflectance array shaped (rows, cols, bands).
        wavelengths: Wavelength centers in microns shaped (bands,).
    """

    data: np.ndarray
    wavelengths: np.ndarray

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"data must be 3D (rows, cols, bands); got shape={self.data.shape}")
        if self.wavelengths.ndim != 1:
            raise ValueError(f"wavelengths must be 1D; got shape={self.wavelengths.shape}")
        if self.data.shape[2] != self.wavelengths.shape[0]:
            raise ValueError(
                "bands dimension mismatch: "
                f"data.shape[2]={self.data.shape[2]} wavelengths.shape[0]={self.wavelengths.shape[0]}"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    def get_pixel_spectrum(self, x: int, y: int) -> np.ndarray:
        """Return a copy of the spectrum at pixel (x, y)."""

        return np.asarray(self.data[y, x, :], dtype=float).copy()


def load_m3_cube(path: str | Path, *, allow_synthetic_fallback: bool = True) -> M3Cube:
    """Load an M³ cube.

    Currently supported:
    - `.npz` files containing `data` and `wavelengths` arrays.

    If parsing fails and `allow_synthetic_fallback=True`, a synthetic cube is
    generated to keep downstream development unblocked.
    """

    path_obj = Path(path)
    if path_obj.suffix.lower() == ".npz":
        return load_m3_cube_npz(path_obj)

    if path_obj.suffix.lower() in {".hdr", ".img"}:
        hdr_path = path_obj if path_obj.suffix.lower() == ".hdr" else path_obj.with_suffix(".HDR")
        img_path = path_obj if path_obj.suffix.lower() == ".img" else path_obj.with_suffix(".IMG")
        header = read_envi_header(hdr_path)
        data = read_envi_image(img_path, header)
        wavelengths = header.wavelengths_um
        if wavelengths is None:
            raise ValueError(f"No wavelengths found in header: {hdr_path}")
        return M3Cube(data=data, wavelengths=wavelengths)

    if allow_synthetic_fallback:
        return generate_synthetic_cube(seed=0)

    raise ValueError(f"Unsupported M3 input path: {path_obj}")


def load_m3_cube_npz(path: str | Path) -> M3Cube:
    """Load a cube stored as a `.npz` with arrays `data` and `wavelengths`."""

    path_obj = Path(path)
    with np.load(path_obj, allow_pickle=False) as npz:
        data = np.asarray(npz["data"], dtype=np.float32)
        wavelengths = np.asarray(npz["wavelengths"], dtype=np.float64)
    return M3Cube(data=data, wavelengths=wavelengths)


def generate_synthetic_cube(
    *,
    rows: int = 50,
    cols: int = 60,
    bands: int = 86,
    seed: int = 0,
    wavelengths_um: Optional[np.ndarray] = None,
) -> M3Cube:
    """Generate a synthetic reflectance cube with simple mineral-like signatures.

    This is a placeholder for development and testing.

    Synthetic classes:
    - 0: featureless / highlands-like (weak absorptions)
    - 1: mafic-like (absorption near ~1 and ~2 µm)
    - 2: plagioclase-like (high albedo, weak absorptions, distinct slope)

    Returns:
        M3Cube with `data` and `wavelengths`.
    """

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

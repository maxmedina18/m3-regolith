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


def load_m3_cube(path: str | Path, *, allow_synthetic_fallback: bool = False) -> M3Cube:
    """Load an M³ cube.

    Currently supported:
    - `.npz` files containing `data` and `wavelengths` arrays.

    If parsing fails and `allow_synthetic_fallback=True`, a synthetic cube is
    generated. This is intended for development and tests; real-data workflows
    should keep the default `allow_synthetic_fallback=False`.
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
        from lunar_m3.dev.synthetic import generate_synthetic_cube

        return generate_synthetic_cube(seed=0)

    raise ValueError(f"Unsupported M3 input path: {path_obj}")


def load_m3_cube_npz(path: str | Path) -> M3Cube:
    """Load a cube stored as a `.npz` with arrays `data` and `wavelengths`."""

    path_obj = Path(path)
    with np.load(path_obj, allow_pickle=False) as npz:
        data = np.asarray(npz["data"], dtype=np.float32)
        wavelengths = np.asarray(npz["wavelengths"], dtype=np.float64)
    return M3Cube(data=data, wavelengths=wavelengths)

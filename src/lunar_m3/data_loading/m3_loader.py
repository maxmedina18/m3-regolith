from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .pds3_envi import EnviHeader, read_envi_header, read_envi_image


@dataclass
class M3Cube:
    """Development-friendly abstraction for an M³ reflectance cube.

    Attributes:
        data: Reflectance array shaped (rows, cols, bands).
        wavelengths: Wavelength centers in microns shaped (bands,).
    """

    data: np.ndarray
    wavelengths: np.ndarray
    header: EnviHeader | None = None
    source_path: str | None = None

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

        if not (0 <= x < self.shape[1] and 0 <= y < self.shape[0]):
            raise IndexError("Pixel outside source bounds")
        return np.asarray(self.data[y, x, :], dtype=float).copy()

    @property
    def nodata(self) -> float | None:
        return self.header.nodata if self.header else None

    @property
    def band_good(self) -> np.ndarray:
        return self.header.band_good if self.header is not None else np.ones(self.shape[2], dtype=bool)

    def pixel_validity(self, x: int, y: int) -> np.ndarray:
        r = self.get_pixel_spectrum(x, y)
        valid = np.isfinite(r) & self.band_good
        if self.nodata is not None:
            valid &= r != self.nodata
        return valid


def load_m3_cube(path: str | Path, *, wavelength_unit: str | None = None) -> M3Cube:
    """Load an M³ cube.

    Supported inputs:
    - ENVI-style `.HDR` + `.IMG` products (common for PDS Imaging Node M³ cubes)
    """

    path_obj = Path(path)
    if path_obj.suffix.lower() not in {".hdr", ".img"}:
        raise ValueError(
            "Real-data-first workflow requires an ENVI cube (.IMG/.HDR). "
            f"Got: {path_obj}"
        )

    hdr_path = path_obj if path_obj.suffix.lower() == ".hdr" else path_obj.with_suffix(".HDR")
    img_path = path_obj if path_obj.suffix.lower() == ".img" else path_obj.with_suffix(".IMG")
    if not hdr_path.exists():
        hdr_path = hdr_path.with_suffix(".hdr")
    if not img_path.exists():
        img_path = img_path.with_suffix(".img")
    header = read_envi_header(hdr_path, wavelength_unit=wavelength_unit)
    data = read_envi_image(img_path, header)
    wavelengths = header.wavelengths_um
    if wavelengths is None:
        raise ValueError(f"No wavelengths found in header: {hdr_path}")
    return M3Cube(data=data, wavelengths=wavelengths, header=header, source_path=str(img_path.resolve()))


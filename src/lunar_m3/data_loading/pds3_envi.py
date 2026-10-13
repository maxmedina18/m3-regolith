from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class EnviHeader:
    samples: int
    lines: int
    bands: int
    interleave: str
    data_type: int
    byte_order: int
    header_offset: int = 0
    wavelengths_um: Optional[np.ndarray] = None
    nodata: float | None = None
    band_good: Optional[np.ndarray] = None
    metadata: dict[str, object] = field(default_factory=dict)
    wavelength_unit_source: str | None = None


def read_envi_header(path: str | Path, *, wavelength_unit: str | None = None) -> EnviHeader:
    """Parse a minimal ENVI-style `.HDR`.

    This is sufficient for many PDS Imaging Node M³ `.IMG` products where the
    `.HDR` contains ENVI-like keys.

    Supported keys:
    - samples, lines, bands
    - interleave (bsq, bil, bip)
    - data type (ENVI numeric type code)
    - byte order (0 little, 1 big)
    - header offset
    - wavelength, explicit units, bad-band flags and NoData

    Missing wavelength units require an explicit caller override; no magnitude
    heuristic is used. All original fields are retained in metadata.
    """

    text = Path(path).read_text(encoding="utf-8")
    kv = _parse_envi_kv(text)

    samples = int(kv["samples"])
    lines = int(kv["lines"])
    bands = int(kv["bands"])
    interleave = str(kv["interleave"]).strip().lower()
    data_type = int(kv["data type"])
    byte_order = int(kv["byte order"])
    header_offset = int(kv.get("header offset", 0))

    if min(samples, lines, bands) < 1 or header_offset < 0:
        raise ValueError("Dimensions must be positive and header offset nonnegative")
    _envi_dtype(data_type, byte_order)
    if interleave not in {"bsq", "bil", "bip"}:
        raise ValueError(f"Unsupported interleave: {interleave}")
    wavelengths_um: Optional[np.ndarray] = None
    unit_source = None
    if "wavelength" in kv:
        wl = kv["wavelength"]
        if isinstance(wl, list):
            wavelengths_um = np.asarray([float(x) for x in wl], dtype=np.float64)
        else:
            wavelengths_um = np.asarray([float(x) for x in str(wl).split(",") if x.strip()], dtype=np.float64)

        if len(wavelengths_um) != bands or not np.all(np.isfinite(wavelengths_um)) or not np.all(np.diff(wavelengths_um) > 0) or np.any(wavelengths_um <= 0):
            raise ValueError("Wavelengths must match bands and be positive, finite, increasing")
        units = {"nm": 0.001, "nanometers": 0.001, "nanometres": 0.001,
                 "um": 1.0, "µm": 1.0, "microns": 1.0, "micrometers": 1.0, "micrometres": 1.0}
        declared = str(kv.get("wavelength units", "")).strip().lower()
        override = (wavelength_unit or "").lower()
        if declared and declared not in units:
            raise ValueError(f"Unsupported declared wavelength units: {declared}")
        if override and override not in units:
            raise ValueError(f"Unsupported wavelength override: {override}")
        if declared and override and units[declared] != units[override]:
            raise ValueError("Wavelength override conflicts with header")
        chosen = declared or override
        if not chosen:
            raise ValueError("Missing wavelength units; supply an explicit wavelength_unit")
        wavelengths_um *= units[chosen]
        unit_source = "header" if declared else f"caller override: {override}"

    good = np.ones(bands, dtype=bool)
    if "bbl" in kv:
        flags = np.asarray(kv["bbl"], dtype=float)
        if flags.shape != (bands,) or not np.all(np.isin(flags, [0, 1])):
            raise ValueError("bbl must contain one 0/1 flag per band")
        good = flags.astype(bool)

    return EnviHeader(
        samples=samples,
        lines=lines,
        bands=bands,
        interleave=interleave,
        data_type=data_type,
        byte_order=byte_order,
        header_offset=header_offset,
        wavelengths_um=wavelengths_um,
        nodata=float(kv["data ignore value"]) if "data ignore value" in kv else None,
        band_good=good,
        metadata=kv,
        wavelength_unit_source=unit_source,
    )


def read_envi_image(path: str | Path, header: EnviHeader) -> np.ndarray:
    """Read-only source-dtype view shaped (lines, samples, bands); no value conversion."""

    dtype = _envi_dtype(header.data_type, header.byte_order)
    img_path = Path(path)
    if min(header.lines, header.samples, header.bands) < 1 or header.header_offset < 0:
        raise ValueError("Invalid dimensions or header offset")
    expected = header.header_offset + header.lines * header.samples * header.bands * dtype.itemsize
    if img_path.stat().st_size != expected:
        raise ValueError(f"Image byte count mismatch: expected {expected}, got {img_path.stat().st_size}")

    if header.interleave not in {"bsq", "bil", "bip"}:
        raise ValueError(f"Unsupported interleave: {header.interleave}")

    if header.interleave == "bsq":
        raw_shape = (header.bands, header.lines, header.samples)
    elif header.interleave == "bil":
        raw_shape = (header.lines, header.bands, header.samples)
    else:
        raw_shape = (header.lines, header.samples, header.bands)

    mm = np.memmap(img_path, dtype=dtype, mode="r", offset=header.header_offset, shape=raw_shape)

    if header.interleave == "bsq":
        data = np.transpose(mm, (1, 2, 0))
    elif header.interleave == "bil":
        data = np.transpose(mm, (0, 2, 1))
    else:
        data = np.asarray(mm)

    return np.asarray(data)


def _envi_dtype(data_type: int, byte_order: int) -> np.dtype:
    if byte_order not in (0, 1):
        raise ValueError("byte order must be 0 or 1")
    endian = "<" if int(byte_order) == 0 else ">"

    mapping: dict[int, str] = {
        1: "u1",
        2: "i2",
        3: "i4",
        4: "f4",
        5: "f8",
        12: "u2",
        13: "u4",
        14: "i8",
        15: "u8",
    }
    if data_type not in mapping:
        raise ValueError(f"Unsupported ENVI data type code: {data_type}")

    return np.dtype(endian + mapping[data_type])


def _parse_envi_kv(text: str) -> dict[str, object]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith(";")]
    kv: dict[str, object] = {}

    i = 0
    while i < len(lines):
        ln = lines[i]
        if "=" not in ln:
            i += 1
            continue

        key, rest = ln.split("=", 1)
        key = " ".join(key.strip().lower().split())
        if key in kv:
            raise ValueError(f"Duplicate ENVI field: {key}")
        value = rest.strip()

        if value.startswith("{") and not value.endswith("}"):
            chunks = [value]
            i += 1
            while i < len(lines) and not lines[i].strip().endswith("}"):
                chunks.append(lines[i])
                i += 1
            if i < len(lines):
                chunks.append(lines[i])
            value = " ".join(chunks)
            if not value.endswith("}"):
                raise ValueError(f"Unclosed ENVI field: {key}")

        if value.startswith("{") and value.endswith("}"):
            inner = value[1:-1].strip()
            items = [x.strip() for x in inner.replace("\n", " ").split(",")]
            kv[key] = [x for x in items if x]
        else:
            kv[key] = value

        i += 1

    return kv

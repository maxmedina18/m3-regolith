from __future__ import annotations

from dataclasses import dataclass
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


def read_envi_header(path: str | Path) -> EnviHeader:
    """Parse a minimal ENVI-style `.HDR`.

    This is sufficient for many PDS Imaging Node M³ `.IMG` products where the
    `.HDR` contains ENVI-like keys.

    Supported keys:
    - samples, lines, bands
    - interleave (bsq, bil, bip)
    - data type (ENVI numeric type code)
    - byte order (0 little, 1 big)
    - header offset
    - wavelength (list; assumed microns)
    """

    text = Path(path).read_text(errors="ignore")
    kv = _parse_envi_kv(text)

    samples = int(kv["samples"])
    lines = int(kv["lines"])
    bands = int(kv["bands"])
    interleave = str(kv.get("interleave", "bsq")).strip().lower()
    data_type = int(kv["data type"])
    byte_order = int(kv.get("byte order", 0))
    header_offset = int(kv.get("header offset", 0))

    wavelengths_um: Optional[np.ndarray] = None
    if "wavelength" in kv:
        wl = kv["wavelength"]
        if isinstance(wl, list):
            wavelengths_um = np.asarray([float(x) for x in wl], dtype=np.float64)
        else:
            wavelengths_um = np.asarray([float(x) for x in str(wl).split(",") if x.strip()], dtype=np.float64)

        if wavelengths_um.size > 0 and float(np.nanmedian(wavelengths_um)) > 50.0:
            wavelengths_um = wavelengths_um / 1000.0

    return EnviHeader(
        samples=samples,
        lines=lines,
        bands=bands,
        interleave=interleave,
        data_type=data_type,
        byte_order=byte_order,
        header_offset=header_offset,
        wavelengths_um=wavelengths_um,
    )


def read_envi_image(path: str | Path, header: EnviHeader) -> np.ndarray:
    """Read an ENVI-style `.IMG` into a float32 cube shaped (lines, samples, bands)."""

    dtype = _envi_dtype(header.data_type, header.byte_order)
    img_path = Path(path)

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

    return np.asarray(data, dtype=np.float32)


def _envi_dtype(data_type: int, byte_order: int) -> np.dtype:
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
        key = key.strip().lower()
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

        if value.startswith("{") and value.endswith("}"):
            inner = value[1:-1].strip()
            items = [x.strip() for x in inner.replace("\n", " ").split(",")]
            kv[key] = [x for x in items if x]
        else:
            kv[key] = value

        i += 1

    return kv

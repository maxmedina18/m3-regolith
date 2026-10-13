"""Independent extraction-only workflow; does not import the advisor implementation.

Source values and native line/sample order are preserved. Quality masks are
metadata/derived layers, never silent replacements of stored reflectance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data_loading.pds3_envi import read_envi_header, read_envi_image

OBS = [
    ("to_sun_azimuth", "degrees"), ("to_sun_zenith", "degrees"),
    ("to_m3_azimuth", "degrees"), ("to_m3_zenith", "degrees"),
    ("phase_angle", "degrees"), ("to_sun_path_offset", "AU"),
    ("to_m3_path", "m"), ("local_slope", "degrees"),
    ("local_aspect", "degrees"), ("cosine_local_incidence", "dimensionless"),
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def valid_values(values, nodata):
    return np.isfinite(values) & (values != nodata)


def parse_bands(text, count):
    if text == "all":
        return list(range(1, count + 1))
    result = set()
    for part in text.split(","):
        ends = [int(x) for x in part.strip().split("-")]
        if len(ends) == 1:
            result.add(ends[0])
        elif len(ends) == 2 and ends[0] <= ends[1]:
            result.update(range(ends[0], ends[1] + 1))
        else:
            raise ValueError("Invalid band range")
    if not result or min(result) < 1 or max(result) > count:
        raise ValueError("Band selection outside source range")
    return sorted(result)


def validate_vrt(path, header, image):
    """Accept only one full identity SimpleSource per band; no transformations."""
    root = ET.parse(path).getroot()
    if root.tag != "VRTDataset" or (int(root.get("rasterXSize", 0)), int(root.get("rasterYSize", 0))) != (header.samples, header.lines):
        raise ValueError("VRT dimensions disagree with header")
    bands = root.findall("VRTRasterBand")
    if len(bands) != header.bands:
        raise ValueError("VRT band count disagrees with header")
    differences = []
    for k, band in enumerate(bands, 1):
        if band.get("band") != str(k) or band.get("dataType") != "Float32" or band.get("subClass"):
            raise ValueError("Unsupported VRT band mapping or type")
        allowed = {"Description", "NoDataValue", "SimpleSource", "ColorInterp", "Metadata", "Unit", "UnitType"}
        if any(child.tag not in allowed for child in band) or len(band.findall("SimpleSource")) != 1:
            raise ValueError("VRT must contain one untransformed SimpleSource per band")
        source = band.find("SimpleSource")
        if any(c.tag not in {"SourceFilename", "SourceBand", "SourceProperties", "SrcRect", "DstRect"} for c in source):
            raise ValueError("VRT contains unsupported source transformation")
        filename = source.find("SourceFilename")
        if filename is None or filename.get("relativeToVRT") != "1" or (Path(path).parent / (filename.text or "")).resolve() != Path(image).resolve():
            raise ValueError("VRT references a different source")
        if source.findtext("SourceBand") != str(k):
            raise ValueError("VRT source band mismatch")
        for name in ("SrcRect", "DstRect"):
            rect = source.find(name)
            if rect is None or tuple(int(rect.get(key, -1)) for key in ("xOff", "yOff", "xSize", "ySize")) != (0, 0, header.samples, header.lines):
                raise ValueError("VRT resamples or crops the source")
        description = band.findtext("Description", "")
        match = re.fullmatch(r"\s*([0-9.eE+\-]+)\s+nm\s*", description)
        if match is None or not np.isfinite(float(match[1])):
            raise ValueError("VRT wavelength description must specify finite nm")
        differences.append(abs(float(match[1]) - header.wavelengths_um[k-1] * 1000))
    if max(differences) > 0.01 + 1e-10:
        raise ValueError("Header/VRT wavelength disagreement exceeds 0.01 nm")
    return {"sha256": sha256(path), "max_wavelength_difference_nm": max(differences),
            "mapping": "full source identity", "pixel_validation": False}


class ProductSet:
    def __init__(self, config):
        self.config = dict(config)
        self.prefix = config.get("prefix", "M3G20090107T011405")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.prefix):
            raise ValueError("Unsafe product prefix")
        self.headers, self.arrays, self.paths = {}, {}, {}
        self.warnings = []
        for role in ("RFL", "LOC", "OBS"):
            hdr = Path(config[role.lower() + "_hdr"]).expanduser().resolve()
            img = Path(config.get(role.lower() + "_img", hdr.with_suffix(".IMG"))).expanduser().resolve()
            self.paths[role] = (hdr, img)
            h = read_envi_header(hdr, wavelength_unit=config.get("wavelength_unit") if role == "RFL" else None)
            self.headers[role] = h
            self.arrays[role] = read_envi_image(img, h)
        r = self.headers["RFL"]
        if r.wavelengths_um is None:
            raise ValueError("RFL wavelength metadata required")
        self.shape = (r.lines, r.samples)
        for role, count, dtype in [("RFL", config.get("expect_rfl_bands", 85), 4), ("LOC", 3, 5), ("OBS", 10, 4)]:
            h = self.headers[role]
            if (h.lines, h.samples) != self.shape or h.bands != count or h.data_type != dtype:
                raise ValueError(f"{role}: dimensions/bands/type disagree with Stage-1 contract")
        for name, actual in [("expect_lines", r.lines), ("expect_samples", r.samples)]:
            if config.get(name) is not None and config[name] != actual:
                raise ValueError(f"{name} mismatch")
        self.vrt_path = Path(config.get("rfl_vrt", str(self.paths["RFL"][1]) + ".vrt")).expanduser().resolve()
        self.vrt = validate_vrt(self.vrt_path, r, self.paths["RFL"][1])
        self.labels = {}
        for role in ("RFL", "LOC", "OBS"):
            label = config.get(role.lower() + "_label")
            if label:
                p = Path(label).expanduser().resolve()
                self.labels[role] = {"path": str(p), "sha256": sha256(p), "text": p.read_text()}
        if "RFL" in self.labels:
            for role, key in [("LOC", "PIXEL_LOCATION_FILE_NAME"), ("OBS", "OBS_GEOMETRY_FILE_NAME")]:
                m = re.search(key + r'\s*=\s*"([^"]+)"', self.labels["RFL"]["text"])
                if m and m[1] != self.paths[role][1].name:
                    self.warnings.append(f"RFL label references {m[1]}, but selected {self.paths[role][1].name}; spatial alignment UNVERIFIED")
        else:
            self.warnings.append("RFL PDS label not supplied; ancillary version association UNVERIFIED")
        self.warnings.append("Matching dimensions do not verify geographic alignment. NASAView and chemistry validation not performed.")

    def region(self, roi=None):
        roi = tuple(roi) if roi is not None else (0, self.shape[0], 0, self.shape[1])
        r0, r1, c0, c1 = roi
        if not (0 <= r0 < r1 <= self.shape[0] and 0 <= c0 < c1 <= self.shape[1]):
            raise ValueError("ROI must be nonempty source bounds [row_start,row_stop,col_start,col_stop]")
        return roi

    def plane(self, role, band, roi=None, elevation=False, radius=1737400.0):
        r0, r1, c0, c1 = self.region(roi)
        h = self.headers[role]
        if not 1 <= band <= h.bands:
            raise ValueError("Band index outside source")
        values = self.arrays[role][r0:r1, c0:c1, band-1]
        if elevation:
            if role != "LOC" or band != 3 or not np.isfinite(radius) or radius <= 0:
                raise ValueError("Elevation requires LOC radius and a finite positive reference radius")
            values = np.array(values, dtype=np.float64, copy=True)
            mask = valid_values(values, h.nodata)
            values[mask] -= radius
        return values

    def manifest(self, hash_inputs=False):
        inputs = {}
        for role, (hdr, img) in self.paths.items():
            h = self.headers[role]
            inputs[role] = {"header": str(hdr), "image": str(img), "header_sha256": sha256(hdr),
                            "image_bytes": img.stat().st_size, "image_mtime_ns": img.stat().st_mtime_ns,
                            "image_sha256": sha256(img) if hash_inputs else None,
                            "dtype": self.arrays[role].dtype.str, "metadata": h.metadata,
                            "wavelength_unit_source": h.wavelength_unit_source,
                            "nodata": str(h.nodata), "band_good": h.band_good.tolist()}
        root = Path(__file__).resolve().parents[2]
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
        return {"schema": "lunar-m3-stage1-1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "prefix": self.prefix, "source_shape": list(self.shape), "inputs": inputs,
                "labels": self.labels, "vrt": self.vrt, "config": self.config,
                "source_sha256": {str(p.relative_to(root)): sha256(p) for p in [Path(__file__), root/'src/lunar_m3/data_loading/pds3_envi.py']},
                "git_commit": revision.stdout.strip(), "git_dirty": bool(dirty.stdout.strip()),
                "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                "orientation": "source-native rows/columns; zero-based ROI and pixels; one-based source bands",
                "processing": "no smoothing, normalization, interpolation, calibration or reprojection",
                "validation": {"layout_checked": True, "geographic_alignment": "unverified", "nasaview": "not_performed", "chemistry": "not_performed"},
                "warnings": self.warnings}


def export(product, output_dir, *, bands="all", roi=None, format="npy", chunk_lines=128,
           radius=1737400.0, hash_inputs=False, dry_run=False):
    if format not in ("npy", "tsv") or chunk_lines < 1 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid export format, chunk size or radius")
    roi = product.region(roi)
    selected = parse_bands(bands, product.headers["RFL"].bands)
    specs = [(f"band_{b}", "RFL", b, "reflectance", "dimensionless", False) for b in selected]
    specs += [(name, "LOC", b, meaning, unit, False) for b, name, meaning, unit in [(1,"lon","longitude east","degrees"),(2,"lat","latitude","degrees"),(3,"radius","radius at pixel centre","m")]]
    specs += [("elev", "LOC", 3, "elevation from mean radius", "m", True)]
    specs += [(f"OBS_{b}", "OBS", b, name, unit, False) for b,(name,unit) in enumerate(OBS,1)]
    manifest = product.manifest(hash_inputs)
    manifest.update(roi=list(roi), format=format, selected_rfl_bands=selected, chunk_lines=chunk_lines,
                    mean_lunar_radius_m=radius, dry_run=dry_run, planned_tables=len(specs), outputs=[])
    if dry_run:
        return manifest
    out = Path(output_dir).resolve()
    # Creating a new directory protects both input files and previous runs.
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    dictionary = []
    shape = (roi[1]-roi[0], roi[3]-roi[2])
    write_json(out/'run_incomplete.json', {"status":"incomplete", "roi":list(roi)})
    for suffix, role, band, meaning, units, derived in specs:
        name = f"{product.prefix}_{suffix}.{format}"
        path = out/name
        h = product.headers[role]
        dtype = np.dtype('float64') if derived else product.arrays[role].dtype
        dest = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape) if format == 'npy' else path.open('w')
        valid_count = 0
        try:
            for start in range(roi[0], roi[1], chunk_lines):
                stop = min(start + chunk_lines, roi[1])
                block = product.plane(role, band, (start,stop,roi[2],roi[3]), elevation=derived, radius=radius)
                valid_count += int(valid_values(block, h.nodata).sum())
                if format == 'npy':
                    dest[start-roi[0]:stop-roi[0]] = block
                else:
                    np.savetxt(dest, block, delimiter='\t', fmt='%.17g' if dtype.itemsize==8 else '%.9g')
        finally:
            if format == 'npy':
                dest.flush()
                del dest
            else:
                dest.close()
        row = dict(file=name, role=role, band_1based=band, meaning=meaning, units=units,
                   derived=derived, formula=f"radius - {radius}" if derived else "",
                   dtype=dtype.str, nodata=str(h.nodata), valid_count=valid_count,
                   band_good=bool(h.band_good[band-1]), shape=list(shape), sha256=sha256(path), bytes=path.stat().st_size)
        if role == 'RFL':
            row['wavelength_nm'] = float(h.wavelengths_um[band-1]*1000)
        manifest['outputs'].append(row)
        dictionary.append({k:row[k] for k in ('file','role','band_1based','meaning','units','derived','formula','nodata','band_good')})
    with (out/'data_dictionary.tsv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dictionary[0]),delimiter='\t');writer.writeheader();writer.writerows(dictionary)
    manifest['data_dictionary_sha256'] = sha256(out/'data_dictionary.tsv')
    manifest['elapsed_seconds'] = time.perf_counter()-started
    manifest['output_bytes'] = sum(o['bytes'] for o in manifest['outputs'])
    manifest['status'] = 'complete'
    write_json(out/'manifest.json',manifest)
    (out/'run_incomplete.json').unlink()
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--bands',default='all')
    parser.add_argument('--roi',nargs=4,type=int,metavar=('R0','R1','C0','C1'))
    parser.add_argument('--format',choices=['npy','tsv'],default='npy')
    parser.add_argument('--chunk-lines',type=int,default=128)
    parser.add_argument('--hash-inputs',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    result=export(ProductSet(config),args.output_dir,bands=args.bands,roi=args.roi,format=args.format,
                  chunk_lines=args.chunk_lines,hash_inputs=args.hash_inputs,dry_run=args.dry_run)
    print(json.dumps(result if args.dry_run else {"status":result['status'],"tables":len(result['outputs']),"output":str(args.output_dir)},indent=2))

if __name__ == '__main__':
    main()

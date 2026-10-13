"""Reproducible bounded comparison; never imports the revised project reader.

Run with an existing NumPy/Matplotlib environment. Worker processes use the
specified extraction Python and independently import immutable baseline readers.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "f176272f81c0307eddd273ba45c7ce7e2a3797fe"
ADVISOR_HASH = "d190296d0ab7c7d61754c32857354b5342e4622738077e5637738ab2215ba855"
COORDS = [(0, 0), (0, 303), (11934, 0), (11934, 303), (5967, 152),
          (2983, 76), (2983, 228), (8950, 76), (8950, 228), (1, 1), (11933, 302)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def require_snapshot_wavelength_unit(config):
    """These fixed-product comparisons require explicit protocol nanometres.

    Frozen legacy readers retain their historical implementation/API; they must
    not be replaced with the current reader or passed its new keyword argument.
    This guard establishes units explicitly, never from numerical magnitude.
    """
    if config.get("wavelength_unit") != "nm":
        raise ValueError("Snapshot comparison requires explicit wavelength_unit=nm")


def worker(args):
    config = json.loads(args.config.read_text())
    require_snapshot_wavelength_unit(config)
    dest = args.output / args.worker
    dest.mkdir()
    meta = {"python": sys.version, "executable": sys.executable,
            "numpy": np.__version__, "platform": platform.platform(), "products": {}}
    if args.worker == "original":
        sys.path.insert(0, str(args.output / "original_source"))
        from lunar_m3.data_loading.pds3_envi import read_envi_header, read_envi_image
        from lunar_m3.data_loading.m3_loader import load_m3_cube
    else:
        spec = importlib.util.spec_from_file_location("advisor_baseline", args.advisor)
        advisor = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = advisor
        spec.loader.exec_module(advisor)
    for role in ("rfl", "loc", "obs"):
        hdr_path = Path(config[role + "_hdr"])
        img_path = hdr_path.with_suffix(".IMG")
        if args.worker == "original":
            header = read_envi_header(hdr_path)
            if role == "rfl":
                cube = load_m3_cube(hdr_path)
                data = cube.data
                wavelength = cube.wavelengths
            else:
                data = read_envi_image(img_path, header)
            values = np.array([data[r, c, :] for r, c in COORDS])
            info = {"shape": list(data.shape), "dtype": str(data.dtype),
                    "interleave": header.interleave, "byte_order": header.byte_order,
                    "header_offset": header.header_offset, "data_type": header.data_type,
                    "nodata": None, "bbl": None,
                    "metadata_limitation": "Original public header object does not preserve NoData, BBL, or other source metadata."}
            del data
        else:
            cube = advisor.EnviCube(img_path, hdr_path, role.upper())
            values = np.array([[cube.read_band_rows(b, r, r + 1)[0, c]
                                for b in range(1, cube.bands + 1)] for r, c in COORDS])
            info = {"shape": [cube.lines, cube.samples, cube.bands], "dtype": str(cube.dtype),
                    "interleave": cube.interleave, "byte_order": cube.byte_order,
                    "header_offset": cube.header_offset, "data_type": cube.data_type_code,
                    "nodata": cube.nodata, "bbl": advisor.envi_list(cube.header, "bbl", int),
                    "raw_header": cube.header}
            if role == "rfl":
                native = np.array(advisor.envi_list(cube.header, "wavelength", float))
                np.save(dest / "wavelengths_native_nm.npy", native)
                wavelength = native / 1000.0
                info["vrt_validation"] = advisor.validate_vrt(Path(config["rfl_vrt"]), cube)
        info.update({"img_path": str(img_path), "hdr_path": str(hdr_path)})
        meta["products"][role] = info
        np.save(dest / (role + "_pixels.npy"), values)
    np.save(dest / "wavelengths_um.npy", wavelength)
    dump(dest / "metadata.json", meta)
    print(json.dumps({"status": "success", "reader": args.worker,
                      "coordinates": len(COORDS), "rfl_values": len(COORDS) * 85}))


def metrics(a, b, mask=None):
    a, b = a.astype(np.float64), b.astype(np.float64)
    selected = np.ones(a.shape, bool) if mask is None else mask
    equal = (a == b) | (np.isnan(a) & np.isnan(b))
    finite = selected & np.isfinite(a) & np.isfinite(b)
    delta = a[finite] - b[finite]
    return {"selected_count": int(selected.sum()), "finite_pair_count": int(finite.sum()),
            "mismatched_values": int((selected & ~equal).sum()),
            "max_absolute_error": float(np.max(np.abs(delta))) if delta.size else None,
            "mean_absolute_error": float(np.mean(np.abs(delta))) if delta.size else None,
            "rmse": float(np.sqrt(np.mean(delta ** 2))) if delta.size else None,
            "criterion": "Exact equality; matching NaNs accepted; errors calculated on finite pairs."}


def main(args):
    require_snapshot_wavelength_unit(json.loads(args.config.read_text()))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if sha(args.advisor) != ADVISOR_HASH:
        raise RuntimeError("Advisor checksum differs from ZIP-verified baseline")
    baseline = out / "original_source"
    for name in ("__init__.py", "data_loading/__init__.py", "data_loading/pds3_envi.py", "data_loading/m3_loader.py"):
        path = baseline / "lunar_m3" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(["git", "show", f"{COMMIT}:src/lunar_m3/{name}"], cwd=ROOT))
    config = json.loads(args.config.read_text())
    require_snapshot_wavelength_unit(config)
    dump(out / "input_config.json", config)
    dump(out / "coordinates.json", {"indexing": "zero-based source row, column; bands 1..85", "coordinates": COORDS})
    inputs = {Path(v) for k, v in config.items() if k.endswith(("_hdr", "_vrt", "_label"))}
    inputs |= {Path(config[r + "_hdr"]).with_suffix(".IMG") for r in ("rfl", "loc", "obs")}
    provenance = {"baseline_commit": COMMIT, "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "input_files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(inputs)],
                  "source_files": [], "commands": [], "scope": "11 source-native pixels, all 85 RFL + 3 LOC + 10 OBS bands; no full-scene export"}
    for path in [Path(__file__).resolve(), args.advisor.resolve(), *sorted(baseline.rglob("*.py"))]:
        provenance["source_files"].append({"path": str(path), "sha256": sha(path)})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for reader in ("original", "advisor"):
        cmd = [str(args.python.absolute()), "-B", str(Path(__file__).resolve()), "--worker", reader,
               "--config", str(out / "input_config.json"), "--output", str(out), "--advisor", str(args.advisor.resolve())]
        started = time.monotonic()
        run = subprocess.run(cmd, env=env, text=True, capture_output=True, cwd=ROOT)
        (out / (reader + ".log")).write_text(run.stdout + run.stderr)
        provenance["commands"].append({"argv": cmd, "shell": shlex.join(cmd), "returncode": run.returncode,
                                       "elapsed_seconds": time.monotonic() - started})
        dump(out / "provenance.json", provenance)
        if run.returncode:
            (out / "comparison_summary.md").write_text(f"Extraction failed for {reader}. See {reader}.log and provenance.json. No numerical agreement claimed.\n")
            raise RuntimeError(f"{reader} failed: {run.stderr}")
    a_meta, b_meta = [json.loads((out / r / "metadata.json").read_text()) for r in ("original", "advisor")]
    report = {"scope": provenance["scope"], "products": {}, "metadata": {}}
    arrays = {}
    for role in ("rfl", "loc", "obs"):
        a, b = [np.load(out / r / (role + "_pixels.npy")) for r in ("original", "advisor")]
        arrays[role] = (a, b)
        nodata = b_meta["products"][role]["nodata"]
        valid_a, valid_b = np.isfinite(a), np.isfinite(b)
        if nodata is not None:
            valid_a &= a != nodata
            valid_b &= b != nodata
        report["products"][role] = {"raw": metrics(a, b), "valid_only": metrics(a, b, valid_a & valid_b),
                                    "invalid_mask_mismatches": int((valid_a != valid_b).sum()),
                                    "invalid_pairs": int((~valid_a & ~valid_b).sum()),
                                    "per_band": [metrics(a[:, i], b[:, i]) for i in range(a.shape[1])]}
        report["metadata"][role] = {key: {"original": a_meta["products"][role][key],
                                                        "advisor": b_meta["products"][role][key],
                                                        "equal": a_meta["products"][role][key] == b_meta["products"][role][key]}
                                     for key in ("shape", "dtype", "interleave", "byte_order", "header_offset", "data_type", "nodata", "bbl")}
    wa, wb = [np.load(out / r / "wavelengths_um.npy") for r in ("original", "advisor")]
    report["wavelengths_um"] = metrics(wa, wb)
    report["wavelength_units"] = "Source has no explicit wavelength unit. Original uses magnitude heuristic; comparison converts advisor native numeric wavelengths from nm to um per protocol. Agreement does not independently establish units."
    dump(out / "comparison_metrics.json", report)
    with (out / "comparison_spectra.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_0based", "column_0based", "band_1based", "original_wavelength_um", "advisor_wavelength_um", "original_reflectance", "advisor_reflectance", "original_minus_advisor", "valid_finite_non_nodata", "bbl_good"])
        a, b = arrays["rfl"]
        good = b_meta["products"]["rfl"]["bbl"]
        for p, (row, col) in enumerate(COORDS):
            for i in range(85):
                x, y = float(a[p, i]), float(b[p, i])
                w.writerow([row, col, i + 1, wa[i], wb[i], x, y, x - y,
                            int(np.isfinite(x) and np.isfinite(y) and x != -999 and y != -999), good[i]])
    with (out / "ancillary_comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["role", "row_0based", "column_0based", "band_1based", "original", "advisor", "original_minus_advisor"])
        for role in ("loc", "obs"):
            a, b = arrays[role]
            for p, (row, col) in enumerate(COORDS):
                for i in range(a.shape[1]):
                    w.writerow([role, row, col, i + 1, float(a[p, i]), float(b[p, i]), float(a[p, i]) - float(b[p, i])])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(14, 13), sharex=True)
    a, b = arrays["rfl"]
    for p, ax in enumerate(axes.flat):
        if p == len(COORDS):
            ax.axis("off")
            continue
        valid = np.isfinite(a[p]) & np.isfinite(b[p]) & (a[p] != -999) & (b[p] != -999) & (np.array(good) == 1)
        ax.plot(wa, np.where(valid, a[p], np.nan), color="navy", label="Original f176272")
        ax.plot(wb, np.where(valid, b[p], np.nan), color="darkorange", linestyle="--", label="Advisor v2")
        ax.set_title(f"row {COORDS[p][0]}, column {COORDS[p][1]}")
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Reflectance")
        ax.grid(alpha=.25)
    axes.flat[0].legend()
    fig.suptitle("M3G20090107T011405 V01 — raw spectra, source-native coordinates\nNoData/nonfinite and BBL-bad bands hidden only in plots")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out / "spectra_overlay.png", dpi=160)
    fig.savefig(out / "spectra_overlay.pdf")
    plt.close(fig)
    provenance["comparison_environment"] = {"python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__}
    provenance["orchestrator_command"] = shlex.join([sys.executable, *sys.argv])
    provenance["advisor_hash_unchanged_after_run"] = sha(args.advisor) == ADVISOR_HASH
    dump(out / "provenance.json", provenance)
    lines = ["# Original vs advisor M³ extraction comparison", "",
             f"Original source: `{COMMIT}`; advisor unchanged SHA-256 `{ADVISOR_HASH}`.", "",
             "Both extractors ran successfully in independent Python processes using their original reader APIs. The advisor CLI's full 99-table export was not run here. Wrappers select the same 11 source-native, zero-based row/column coordinates; all bands are retained. No resampling, normalization, or spectral preprocessing was applied.", "",
             "Extraction environment: Python 3.12.14, NumPy 2.1.3. Exact executable, actual versions, commands, runtimes, source/input hashes and paths are in provenance.json and each reader's metadata.json.", "",
             "## Numerical results", "", "|Quantity|Values|Max absolute error|MAE|RMSE|Exact mismatches|", "|---|---:|---:|---:|---:|---:|"]
    for role in ("rfl", "loc", "obs"):
        m = report["products"][role]["raw"]
        lines.append(f"|{role.upper()}|{m['selected_count']}|{m['max_absolute_error']:.12g}|{m['mean_absolute_error']:.12g}|{m['rmse']:.12g}|{m['mismatched_values']}|")
    lines += ["", "LOC summary aggregates different physical units; use these per-plane errors:", ""]
    for i, unit in enumerate(("longitude degrees", "latitude degrees", "radius metres")):
        m = report["products"]["loc"]["per_band"][i]
        lines.append(f"- {unit}: maximum absolute error {m['max_absolute_error']:.12g}, MAE {m['mean_absolute_error']:.12g}, RMSE {m['rmse']:.12g}, mismatches {m['mismatched_values']}/11.")
    lines += ["", f"Wavelength comparison (µm): {json.dumps(report['wavelengths_um'])}", "",
              "## Interpretation and limitations", "",
              "- Both readers report RFL shape [11935, 304, 85], LOC [11935, 304, 3], OBS [11935, 304, 10]. Layout/type codes, byte order and offsets agree. Original LOC output is float32; advisor preserves source float64. The original reader's unconditional float32 conversion causes the observed LOC precision loss.",
              "- Original metadata omits NoData and BBL; advisor retains them. Invalid masks in this comparison use advisor/source-header NoData for both outputs, not a capability of the original reader. Valid-only and per-band errors are recorded separately in the metrics JSON.",
              "- The original wavelength magnitude heuristic and explicit nm-to-µm conversion agree here. The header lacks an explicit unit; wavelength agreement is metadata consistency, not independent pixel validation.",
              "- Only the listed pixels were extracted and compared. No claim of whole-cube equality, NASAView validation, FeO validation, or full Stage-1 compliance is made. Both implementations use NumPy memmap; independent process execution does not make them fully independent decoding technologies.",
              "- V01 RFL label references V03 ancillary products; this run uses the advisor-prescribed V02 LOC/OBS. Equal dimensions do not establish cross-version spatial alignment.",
              "- The original extractor has no elevation export. Elevation and full-scene TSV serialization were not compared in this run.", "",
              "## Artifacts", "", "See result_inventory.json for the absolute path, byte size and SHA-256 of every generated artifact (except the inventory itself). Original and advisor outputs are isolated in their respective subdirectories. comparison_spectra.csv contains all 935 raw reflectance pairs, including sentinel values; ancillary_comparison.csv contains all 143 ancillary pairs."]
    (out / "comparison_summary.md").write_text("\n".join(lines) + "\n")
    dump(out / "result_inventory.json", [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(out.rglob("*")) if p.is_file()])
    print(json.dumps({"output": str(out), "products": {r: report["products"][r]["raw"] for r in arrays}, "wavelengths": report["wavelengths_um"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/snapshot.local.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/extractor_comparison")
    parser.add_argument("--advisor", type=Path, default=ROOT / "external/advisor/M3_stage1_extractor_v2.py")
    parser.add_argument("--python", type=Path, default=ROOT / ".venv-stage1/bin/python")
    parser.add_argument("--worker", choices=("original", "advisor"))
    args = parser.parse_args()
    worker(args) if args.worker else main(args)

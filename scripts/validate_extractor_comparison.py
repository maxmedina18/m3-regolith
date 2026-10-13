"""Check published comparison integrity and recompute errors without raw cubes."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from compare_original_extractors import metrics, COORDS


def validate(folder):
    report = json.loads((folder / "comparison_metrics.json").read_text())
    for item in json.loads((folder / "result_inventory.json").read_text()):
        path = folder / item["path"]
        assert not Path(item["path"]).is_absolute()
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    for role in ("rfl", "loc", "obs"):
        a, b = [np.load(folder / reader / (role + "_pixels.npy")) for reader in ("original", "advisor")]
        assert metrics(a, b) == report["products"][role]["raw"]
        for i in range(a.shape[1]):
            assert metrics(a[:, i], b[:, i]) == report["products"][role]["per_band"][i]
        if role == "loc":
            assert np.array_equal(a, b.astype(np.float32))
        if role == "rfl":
            rows = list(csv.DictReader((folder / "comparison_spectra.csv").open()))
            assert len(rows) == a.size == 935
            valid = np.isfinite(a) & np.isfinite(b) & (a != -999) & (b != -999)
            assert metrics(a, b, valid) == report["products"][role]["valid_only"]
            for k, row in enumerate(rows):
                p, band = divmod(k, 85)
                assert (int(row['row_0based']), int(row['column_0based'])) == COORDS[p]
                assert int(row['band_1based']) == band + 1
                assert float(row['original_reflectance']) == float(a[p, band])
                assert float(row['advisor_reflectance']) == float(b[p, band])
    wa, wb = [np.load(folder / reader / "wavelengths_um.npy") for reader in ("original", "advisor")]
    assert metrics(wa, wb) == report["wavelengths_um"]
    for path in folder.rglob("*"):
        if path.suffix in (".json", ".md", ".csv", ".log", ".py"):
            text = path.read_text()
            assert not any(s in text for s in ("/Users/", "/Volumes/", "/private/", "/home/")), path
    print("PASS: hashes, sampled numerical metrics, CSV coordinates/values, wavelength equality, LOC cast explanation and path hygiene")


if __name__ == "__main__":
    validate(Path(sys.argv[1] if len(sys.argv) > 1 else "results/extractor_comparison"))

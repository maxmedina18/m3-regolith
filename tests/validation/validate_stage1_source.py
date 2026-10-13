#!/usr/bin/env python3

import csv
import json
import math
import random
import struct
from datetime import datetime, timezone
from pathlib import Path

PREFIX = "M3G20090107T011405"

INPUT = Path("data/m3_stage1_inputs")
OUTPUT = Path("results/stage1_memmap")
REPORT = Path(
    "docs/validation/2026-10-12_stage1/"
    "source_pixel_validation.json"
)

LINES = 11935
SAMPLES = 304
SEED = 20261012
RANDOM_POINTS = 100
MEAN_LUNAR_RADIUS = 1737400.0

PRODUCTS = {
    "RFL": {
        "file": f"{PREFIX}_V01_RFL.IMG",
        "bands": 85,
        "dtype": "<f",
        "bytes": 4,
    },
    "LOC": {
        "file": f"{PREFIX}_V02_LOC.IMG",
        "bands": 3,
        "dtype": "<d",
        "bytes": 8,
    },
    "OBS": {
        "file": f"{PREFIX}_V02_OBS.IMG",
        "bands": 10,
        "dtype": "<f",
        "bytes": 4,
    },
}

RFL_BANDS = [1, 2, 3, 9, 19, 40, 60, 85]
LOC_BANDS = [1, 2, 3]
OBS_BANDS = list(range(1, 11))

rng = random.Random(SEED)

points = [
    (0, 0),
    (0, SAMPLES - 1),
    (LINES - 1, 0),
    (LINES - 1, SAMPLES - 1),
    (LINES // 2, SAMPLES // 2),
]

points += [
    (rng.randrange(LINES), rng.randrange(SAMPLES))
    for _ in range(RANDOM_POINTS)
]


def read_binary(role, band, line, sample):
    p = PRODUCTS[role]

    offset = (
        line * p["bands"] * SAMPLES
        + (band - 1) * SAMPLES
        + sample
    ) * p["bytes"]

    with (INPUT / p["file"]).open("rb") as f:
        f.seek(offset)
        raw = f.read(p["bytes"])

    if len(raw) != p["bytes"]:
        raise IOError("Incomplete binary read")

    return struct.unpack(p["dtype"], raw)[0]


def table_name(role, band):
    if role == "RFL":
        return f"{PREFIX}_band_{band}.tsv"

    if role == "OBS":
        return f"{PREFIX}_OBS_{band}.tsv"

    return {
        1: f"{PREFIX}_lon.tsv",
        2: f"{PREFIX}_lat.tsv",
        3: f"{PREFIX}_radius.tsv",
    }[band]


def read_table_points(path, coordinates):
    wanted = {}
    by_line = {}

    for line, sample in coordinates:
        by_line.setdefault(line, set()).add(sample)

    with path.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")

        for line, row in enumerate(reader):
            if line in by_line:
                for sample in by_line[line]:
                    wanted[(line, sample)] = float(row[sample])

    if len(wanted) != len(set(coordinates)):
        raise ValueError(f"Missing coordinates in {path}")

    return wanted


def matches(source, extracted):
    if math.isnan(source):
        return math.isnan(extracted)

    if math.isinf(source):
        return source == extracted

    return math.isclose(
        source,
        extracted,
        rel_tol=1e-7,
        abs_tol=1e-7,
    )


results = []
failures = []

print("\n=== STAGE 1 INDEPENDENT SOURCE VALIDATION ===")
print("Random seed:", SEED)
print("Coordinates per band:", len(points))
print()

for role, bands in (
    ("RFL", RFL_BANDS),
    ("LOC", LOC_BANDS),
    ("OBS", OBS_BANDS),
):
    for band in bands:
        filename = table_name(role, band)
        extracted = read_table_points(OUTPUT / filename, points)

        passed = 0
        max_difference = 0.0

        for line, sample in points:
            original = read_binary(role, band, line, sample)
            actual = extracted[(line, sample)]

            # RFL bands 1 and 2 are masked bad bands.
            if role == "RFL" and band in (1, 2):
                expected = -999.0
            else:
                expected = original

            ok = matches(expected, actual)

            if math.isfinite(expected) and math.isfinite(actual):
                difference = abs(expected - actual)
                max_difference = max(max_difference, difference)

            if ok:
                passed += 1
            else:
                failures.append({
                    "role": role,
                    "band": band,
                    "line": line,
                    "sample": sample,
                    "source_value": original,
                    "expected_value": expected,
                    "extracted_value": actual,
                })

        result = {
            "role": role,
            "band": band,
            "file": filename,
            "tested": len(points),
            "passed": passed,
            "max_absolute_difference": max_difference,
        }

        results.append(result)

        status = "PASS" if passed == len(points) else "FAIL"

        print(
            f"{role:3s} band {band:2d}: "
            f"{passed:3d}/{len(points)} "
            f"{status} "
            f"max_diff={max_difference:.3e}"
        )


print("\n=== DERIVED ELEVATION VALIDATION ===")

elevation_file = OUTPUT / f"{PREFIX}_elev.tsv"
elevations = read_table_points(elevation_file, points)

elevation_passed = 0
elevation_max_difference = 0.0

for line, sample in points:
    radius = read_binary("LOC", 3, line, sample)
    expected = radius - MEAN_LUNAR_RADIUS
    actual = elevations[(line, sample)]

    ok = matches(expected, actual)

    if math.isfinite(expected) and math.isfinite(actual):
        difference = abs(expected - actual)
        elevation_max_difference = max(
            elevation_max_difference,
            difference
        )

    if ok:
        elevation_passed += 1
    else:
        failures.append({
            "role": "DERIVED",
            "band": "elevation",
            "line": line,
            "sample": sample,
            "expected_value": expected,
            "extracted_value": actual,
        })

print(
    f"Elevation: {elevation_passed}/{len(points)} "
    f"{'PASS' if elevation_passed == len(points) else 'FAIL'}"
)

total_tests = sum(r["tested"] for r in results) + len(points)
total_passed = sum(r["passed"] for r in results) + elevation_passed

report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "observation": PREFIX,
    "random_seed": SEED,
    "coordinates_per_band": len(points),
    "source_validation": results,
    "elevation": {
        "tested": len(points),
        "passed": elevation_passed,
        "max_absolute_difference": elevation_max_difference,
    },
    "total_tests": total_tests,
    "total_passed": total_passed,
    "failures": failures,
    "limitations": [
        "Sampled pixel validation, not exhaustive full-scene validation.",
        "RFL bad bands are checked against expected masking.",
        "Does not independently verify RFL/V02 geospatial alignment.",
        "Does not independently validate physical calibration.",
    ],
}

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(json.dumps(report, indent=2))

print("\n=== FINAL RESULT ===")
print(f"Passed: {total_passed}/{total_tests}")
print(f"Report: {REPORT}")

if failures:
    raise SystemExit("VALIDATION FAILED")

print("ALL TESTS PASSED")

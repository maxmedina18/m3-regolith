"""Replace host paths in comparison metadata; regenerate a relative inventory.

Numerical artifacts and extractor source are never edited. Run after extraction
and before publishing. Historical source/input hashes remain unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def prepare(folder):
    p = json.loads((folder / "provenance.json").read_text())
    if "publication" not in p:
        repo = str(Path(p["source_files"][0]["path"]).parent.parent)
        data = os.path.commonpath([str(Path(x["path"]).parent) for x in p["input_files"]])
        replacements = {
            p["commands"][0]["argv"][0]: "${EXTRACTION_PYTHON}",
            p["orchestrator_command"].split()[0]: "${ANALYSIS_PYTHON}",
            data: "${M3_DATA_ROOT}", repo: "${REPO_ROOT}",
        }
        for path in folder.rglob("*.json"):
            if path.name == "result_inventory.json":
                continue
            text = path.read_text()
            for old, new in sorted(replacements.items(), key=lambda x: -len(x[0])):
                text = text.replace(old, new)
            path.write_text(text)
        p = json.loads((folder / "provenance.json").read_text())
        p["publication"] = {
            "path_policy": "Host prefixes replaced with explicit placeholders; historical commands are templates, not literal original paths.",
            "unchanged": "Numerical arrays, CSVs, metrics, plots, logs, source snapshots and historical input/source hashes.",
            "inventory": "Paths relative to results directory; hashes describe published bytes.",
        }
        (folder / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")
    items = [{"path": str(f.relative_to(folder)), "bytes": f.stat().st_size,
              "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
             for f in sorted(folder.rglob("*")) if f.is_file() and f.name != "result_inventory.json" and not f.name.startswith(".")]
    (folder / "result_inventory.json").write_text(json.dumps(items, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    prepare(parser.parse_args().folder)

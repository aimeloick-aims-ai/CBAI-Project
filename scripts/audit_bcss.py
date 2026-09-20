"""Validate acquired BCSS pairs and measure annotated areas; no model training."""

import argparse
import csv
import hashlib
import html
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def area_fractions(mask, known_codes, ignored_codes):
    if mask.ndim != 2:
        raise ValueError("Expected a single-channel semantic mask")
    codes, counts = np.unique(mask, return_counts=True)
    if set(codes) - set(known_codes):
        raise ValueError("Unknown annotation code")
    valid_count = sum(
        int(n) for code, n in zip(codes, counts, strict=True) if code not in ignored_codes
    )
    if not valid_count:
        raise ValueError("No evaluable annotated pixels")
    return {
        int(code): int(n) / valid_count
        for code, n in zip(codes, counts, strict=True)
        if code not in ignored_codes
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bcss_acquisition.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw = ROOT / config["raw_directory"]
    with (raw / "metadata/gtruth_codes.tsv").open(encoding="utf-8", newline="") as stream:
        labels = {
            int(row["GT_code"]): row["label"] for row in csv.DictReader(stream, delimiter="\t")
        }
    with (ROOT / "data/bcss_smoke_manifest.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError("Empty manifest")
    groups = {}
    for record in records:
        path = (ROOT / record["path"]).resolve()
        if not path.is_relative_to(raw.resolve()):
            raise ValueError("Manifest path outside acquisition directory")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        if record["kind"] in groups.setdefault(record["roi_id"], {}):
            raise ValueError("Duplicate file kind for region")
        groups[record["roi_id"]][record["kind"]] = record
    results = []
    for roi_id, pair in groups.items():
        if set(pair) != {"image", "mask"}:
            raise ValueError(f"Incomplete image-mask pair: {roi_id}")
        with Image.open(ROOT / pair["image"]["path"]) as image:
            image.load()
            image_size = image.size
        with Image.open(ROOT / pair["mask"]["path"]) as image:
            mask = np.asarray(image)
        fractions = area_fractions(mask, labels, config["ignored_codes"])
        bounds = json.loads(pair["mask"]["bounds_base_pixels"])
        expected = (bounds["ymax"] - bounds["ymin"], bounds["xmax"] - bounds["xmin"])
        if mask.shape != expected:
            raise ValueError(f"Mask dimensions do not match base-resolution ROI: {roi_id}")
        results.append(
            {
                "roi_id": roi_id,
                "patient_id": pair["image"]["patient_id"],
                "image_size": image_size,
                "mask_shape_native": list(mask.shape),
                "fractions_native_mask": {labels[code]: value for code, value in fractions.items()},
                "ignored_codes": config["ignored_codes"],
                "status": "pass",
            }
        )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "real_data_acquisition_qc_only",
        "regions": results,
        "verified_files": len(records),
        "patient_count": len({row["patient_id"] for row in results}),
        "bytes": sum(int(row["bytes"]) for row in records),
        "model_trained": False,
        "external_test_opened": False,
        "limitation": "First metadata rows only; not representative and not a diagnostic cohort.",
    }
    out = ROOT / "reports/bcss_data_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    body = html.escape(json.dumps(report, ensure_ascii=False, indent=2))
    (ROOT / "reports/data_audit.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>BCSS acquisition audit</title>'
        "<h1>BCSS : contrôle des données réelles</h1>"
        "<p>Lecture et annotations uniquement ; aucun modèle entraîné.</p><pre>" + body + "</pre>",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

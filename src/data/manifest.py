"""Manifest readers for authorized pilot data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RoiRecord:
    dataset: str
    patient_id: str
    image_id: str
    label: str
    split: str
    path: Path
    width: int
    height: int


def read_bracs_smoke_manifest(path: Path | None = None) -> list[RoiRecord]:
    """Read real BRACS RoI pilot records without changing official test data."""
    source = path or ROOT / "data/bracs_smoke_manifest.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = []
    for row in rows:
        records.append(
            RoiRecord(
                dataset="BRACS",
                patient_id=row["patient_id"],
                image_id=row["image_id"],
                label=row["roi_label"],
                split=row["official_split"],
                path=ROOT / row["path"],
                width=int(row["width"]),
                height=int(row["height"]),
            )
        )
    return records

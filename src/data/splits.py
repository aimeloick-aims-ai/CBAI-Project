"""Patient-level split checks."""

from __future__ import annotations

from collections import defaultdict

from src.data.manifest import RoiRecord


def assert_no_patient_overlap(records: list[RoiRecord]) -> dict[str, list[str]]:
    """Return patient-to-splits mapping after rejecting split leakage."""
    by_patient: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_patient[record.patient_id].add(record.split)
    overlap = {patient: sorted(splits) for patient, splits in by_patient.items() if len(splits) > 1}
    if overlap:
        raise ValueError(f"Patient overlap across splits: {overlap}")
    return {patient: sorted(splits) for patient, splits in by_patient.items()}


def label_index(records: list[RoiRecord]) -> dict[str, int]:
    labels = sorted({record.label for record in records})
    return {label: idx for idx, label in enumerate(labels)}

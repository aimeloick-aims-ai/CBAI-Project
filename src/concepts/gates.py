"""Validation gates for concept experiments."""


def require_concept_annotations(path: str | None) -> None:
    if not path:
        raise ValueError(
            "Concept probing requires an independent annotation table; "
            "pilot image labels are not concept supervision."
        )


def require_held_out_concept_evaluation(train_patients: set[str], test_patients: set[str]) -> None:
    """Reject circular concept evaluation on the same patients used for fitting."""
    overlap = train_patients & test_patients
    if overlap:
        raise ValueError(f"Concept probe patient leakage: {sorted(overlap)}")

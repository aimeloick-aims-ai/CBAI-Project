from dataclasses import replace

import pytest

from src.evaluation import AuditEvidence, classify_evidence


def valid():
    return AuditEvidence(True, True, True, True, True, -0.02, 0.03, 0.01, True)


def test_nonsignificance_is_not_nonuse():
    assert classify_evidence(valid()) == "encoded_with_uncertain_effect"
    assert (
        classify_evidence(replace(valid(), effect_lower=-0.005, effect_upper=0.005))
        == "encoded_with_negligible_excess_effect_under_tested_intervention"
    )


def test_validity_precedes_effect_and_posthoc_margin_is_rejected():
    positive = replace(valid(), effect_lower=0.1, effect_upper=0.2)
    assert classify_evidence(positive) == "encoded_with_supported_target_specific_effect"
    assert (
        classify_evidence(replace(positive, independent_annotations=False)) == "concept_unvalidated"
    )
    assert (
        classify_evidence(replace(positive, intervention_selective=False))
        == "intervention_inconclusive"
    )
    assert (
        classify_evidence(replace(valid(), margin_fixed_before_evaluation=False))
        == "effect_threshold_not_prespecified"
    )


def test_invalid_intervals():
    for kwargs in (
        {"effect_lower": float("nan")},
        {"effect_upper": -0.1},
        {"equivalence_margin": 0},
    ):
        with pytest.raises(ValueError):
            classify_evidence(replace(valid(), **kwargs))

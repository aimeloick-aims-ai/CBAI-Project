"""Conservative interpretation of independently validated concept interventions."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AuditEvidence:
    independent_annotations: bool
    held_out_probe_passed: bool
    intervention_selective: bool
    plausibility_validated: bool
    matched_controls_valid: bool
    effect_lower: float
    effect_upper: float
    equivalence_margin: float
    margin_fixed_before_evaluation: bool


def classify_evidence(evidence: AuditEvidence) -> str:
    """An uncertain effect is not evidence of non-use; labels are audit-specific."""
    if not all(
        math.isfinite(x)
        for x in (evidence.effect_lower, evidence.effect_upper, evidence.equivalence_margin)
    ):
        raise ValueError("Effect bounds and margin must be finite")
    if evidence.effect_lower > evidence.effect_upper or evidence.equivalence_margin <= 0:
        raise ValueError("Invalid interval or equivalence margin")
    if not evidence.independent_annotations:
        return "concept_unvalidated"
    if not evidence.held_out_probe_passed:
        return "encoding_not_established"
    if not (
        evidence.intervention_selective
        and evidence.plausibility_validated
        and evidence.matched_controls_valid
    ):
        return "intervention_inconclusive"
    if not evidence.margin_fixed_before_evaluation:
        return "effect_threshold_not_prespecified"
    margin = evidence.equivalence_margin
    if evidence.effect_lower > margin:
        return "encoded_with_supported_target_specific_effect"
    if evidence.effect_lower > -margin and evidence.effect_upper < margin:
        return "encoded_with_negligible_excess_effect_under_tested_intervention"
    if evidence.effect_upper < -margin:
        return "encoded_with_reverse_target_control_effect"
    return "encoded_with_uncertain_effect"

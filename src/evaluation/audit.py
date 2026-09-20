"""Records for concept-use audits, deliberately separate from model accuracy."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.concepts.probes import ProbeResult
from src.evaluation.metrics import specificity


@dataclass(frozen=True)
class ConceptAudit:
    concept: str
    probe: ProbeResult
    target_probability_shift: float
    control_probability_shifts: list[float]
    validity_passed: bool

    def summary(self) -> dict[str, float | int | str | bool]:
        """Serialize an auditable result without asserting causal meaning."""
        result = asdict(self)
        result["specificity"] = specificity(
            self.target_probability_shift, self.control_probability_shifts
        )
        return result

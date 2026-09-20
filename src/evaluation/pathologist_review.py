"""Pathologist Review & Clinical Validation Framework for GCIA.

Generates structured clinical evaluation sheets, visual overlays of cell graphs
(before vs after intervention), and standardized JSON forms for pathologist scoring.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class ClinicalScoringForm:
    case_id: str
    patient_id: str
    target_diagnosis: str
    fidelity_score: float
    histological_plausibility_score: int  # 1 (unrealistic artifact) to 5 (fully plausible)
    cell_morphology_alignment: int  # 1 to 5
    counterfactual_validity: int  # 1 to 5
    pathologist_comments: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class PathologistReviewGenerator:
    """Exports structured visual review dossiers for anatomopathologist evaluation."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_review_sheet(
        self,
        case_id: str,
        patient_id: str,
        predicted_class: str,
        confidence: float,
        node_positions: torch.Tensor,
        node_importance: torch.Tensor,
        top_k_indices: list[int],
        fidelity_score: float,
    ) -> Path:
        """Reject biased graph-only dossiers; no clinical scores are generated."""
        raise RuntimeError(
            "Legacy review disabled: lacks H&E and leaks predictions. "
            "Use scripts/prepare_expert_round.py for blinded real-image discovery. "
            "Intervention plausibility review additionally requires genuine before/after images."
        )

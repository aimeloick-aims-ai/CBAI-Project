"""Unit tests for pathologist review and clinical validation framework (Axe 3)."""

import json
from pathlib import Path

import pytest
import torch

from src.evaluation.pathologist_review import ClinicalScoringForm, PathologistReviewGenerator


def test_clinical_scoring_form_serialization():
    form = ClinicalScoringForm(
        case_id="case_001",
        patient_id="patient_A",
        target_diagnosis="Invasive",
        fidelity_score=0.45,
        histological_plausibility_score=4,
        cell_morphology_alignment=5,
        counterfactual_validity=4,
        pathologist_comments="Ablation d'un amas de noyaux hyperchromatiques.",
    )
    json_str = form.to_json()
    data = json.loads(json_str)

    assert data["case_id"] == "case_001"
    assert data["histological_plausibility_score"] == 4
    assert data["cell_morphology_alignment"] == 5


def test_pathologist_review_generator(tmp_path: Path):
    generator = PathologistReviewGenerator(output_dir=tmp_path)

    node_pos = torch.rand(20, 2) * 500.0
    node_importance = torch.rand(20)
    top_k = [0, 3, 5]

    with pytest.raises(RuntimeError, match="Legacy review disabled"):
        generator.generate_html_review_sheet(
            case_id="test_case",
            patient_id="pt_123",
            predicted_class="Invasive",
            confidence=0.92,
            node_positions=node_pos,
            node_importance=node_importance,
            top_k_indices=top_k,
            fidelity_score=0.38,
        )
    assert not list(tmp_path.iterdir())

import pytest
import torch

from src.concepts.gates import require_held_out_concept_evaluation
from src.concepts.probes import ProbeResult, evaluate_binary_probe, fit_binary_probe
from src.evaluation.audit import ConceptAudit
from src.interventions.feature_edits import prediction_shift
from src.interventions.graph_edits import drop_undirected_edge, edge_subset
from src.interventions.patch_edits import replace_masked_rgb


def test_probe_is_evaluated_on_held_out_examples():
    torch.manual_seed(2)
    embeddings = torch.randn(20, 3)
    labels = (embeddings[:, 0] > 0).long()
    probe = fit_binary_probe(embeddings[:12], labels[:12], steps=100)
    result = evaluate_binary_probe(probe, embeddings[12:], labels[12:])
    assert result.n_examples == 8
    assert result.balanced_accuracy >= 0.5


def test_patient_leakage_is_rejected_for_concept_probes():
    with pytest.raises(ValueError, match="leakage"):
        require_held_out_concept_evaluation({"p1"}, {"p1", "p2"})


def test_concept_audit_requires_controls_and_reports_specificity():
    before = torch.tensor([[1.0, 0.0]])
    after = torch.tensor([[1.0, 1.0]])
    shift = prediction_shift(before, after, 1)
    audit = ConceptAudit("c", ProbeResult(1.0, 1.0, 1), shift, [0.0], True)
    assert audit.summary()["specificity"] > 0


def test_graph_and_patch_edits_preserve_declared_support_and_shape():
    edges = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]])
    edited_edges = drop_undirected_edge(edges, 0, 1)
    assert edited_edges.numel() == 0
    assert edge_subset(edited_edges, edges)
    image = torch.zeros((2, 2, 3), dtype=torch.uint8).numpy()
    replacement = torch.full((2, 2, 3), 255, dtype=torch.uint8).numpy()
    edited_image = replace_masked_rgb(
        image, replacement, torch.tensor([[1, 0], [0, 1]]).bool().numpy()
    )
    assert edited_image[0, 0, 0] == 255
    assert edited_image[0, 1, 0] == 0

"""Unit tests for probe robustness and statistical controls (Axe 2)."""

import torch

from src.concepts.probes import LinearConceptProbe
from src.evaluation.probe_robustness import (
    NullProbeEvaluator,
    ProbeStabilityEvaluator,
    RandomDirectionControlEvaluator,
)
from src.models.gnn import GraphClassifier


def test_null_probe_evaluator():
    torch.manual_seed(42)
    # Generate linearly separable data
    pos_samples = torch.randn(50, 8) + 2.0
    neg_samples = torch.randn(50, 8) - 2.0
    embeddings = torch.cat([pos_samples, neg_samples], dim=0)
    labels = torch.cat([torch.ones(50), torch.zeros(50)], dim=0)

    evaluator = NullProbeEvaluator(num_permutations=5, steps=50)
    result = evaluator.evaluate(embeddings, labels)

    assert result.true_accuracy >= 0.8
    assert result.null_accuracy_mean < result.true_accuracy
    assert result.selectivity_score > 0.1
    assert 0.0 <= result.p_value <= 1.0


def test_random_direction_control_evaluator():
    torch.manual_seed(42)
    model = GraphClassifier("GCN", in_channels=8, hidden_channels=16, out_channels=2)
    x = torch.randn(20, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    batch = torch.zeros(20, dtype=torch.long)

    concept_dir = torch.randn(8)

    evaluator = RandomDirectionControlEvaluator(num_random_directions=5, seed=42)
    result = evaluator.evaluate(model, x, edge_index, batch, concept_dir, alpha=1.0)

    assert isinstance(result.concept_prob_drop, float)
    assert isinstance(result.random_prob_drop_mean, float)
    assert isinstance(result.probe_specificity_score, float)


def test_probe_stability_evaluator():
    p1 = LinearConceptProbe(in_features=8)
    p2 = LinearConceptProbe(in_features=8)

    # Make them similar
    p1.classifier.weight.data = torch.ones(1, 8)
    p2.classifier.weight.data = torch.ones(1, 8) * 0.9

    result = ProbeStabilityEvaluator.evaluate([p1, p2])
    assert result.mean_cosine_similarity > 0.99

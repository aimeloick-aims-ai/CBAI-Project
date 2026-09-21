"""Critical scientific invariants of the canonical study."""

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation import cluster_interval, effect_interpretation, load_config, probe_analysis
from src.graph import synthetic_label, synthetic_split, verify_patient_splits
from src.interventions import (
    edge_set,
    matched_interventions,
    population_edit,
    relation_fraction,
    relational_edit,
)
from src.models import StudyModel

CONFIG = load_config(Path(__file__).resolve().parents[1])


def graphs():
    return synthetic_split(2, 77, "unit", CONFIG["synthetic"])


def test_patient_overlap_fails():
    with pytest.raises(ValueError, match="overlap"):
        verify_patient_splits(
            [{"patient_id": "a", "split": "train"}, {"patient_id": "a", "split": "test"}]
        )


def test_relational_pairs_identical_features_and_counts():
    gs = graphs()
    for original, paired in zip(gs[::2], gs[1::2], strict=True):
        assert torch.equal(original.x, paired.x)
        assert original.num_nodes == paired.num_nodes
        assert not torch.equal(original.edge_index, paired.edge_index)


def test_unused_absent_from_label_function_and_paired_labels():
    assert list(inspect.signature(synthetic_label).parameters) == ["used", "relational", "flip"]
    gs = graphs()
    for start in (0, 8):
        for used in (0, 1):
            for rel in (0, 1):
                assert torch.equal(gs[start + used * 4 + rel].y, gs[start + used * 4 + 2 + rel].y)


def test_deepsets_edge_only_invariance():
    g = graphs()[0]
    changed = relational_edit(g)
    model = StudyModel("DeepSets", input_dim=3, classes=2).eval()
    batch = torch.zeros(g.num_nodes, dtype=torch.long)
    assert torch.equal(model(g.x, g.edge_index, batch), model(changed.x, changed.edge_index, batch))


def test_interventions_preserve_original_and_match_budget():
    rng = np.random.default_rng(11)
    for graph in graphs()[:8]:
        original = graph.clone()
        for concept in range(3):
            target, controls, valid, audit = matched_interventions(
                graph, concept, "synthetic", CONFIG, rng
            )
            assert valid
            assert audit["budget"] == pytest.approx(audit["control_budget"])
            assert torch.equal(graph.x, original.x)
            assert torch.equal(graph.edge_index, original.edge_index)
            assert torch.equal(graph.y, original.y)
            for edited in [target, *controls]:
                assert edited.num_nodes == graph.num_nodes
                edges = edited.edge_index
                assert not (edges[0] == edges[1]).any()
                assert len(set(map(tuple, edges.T.tolist()))) == edited.num_edges
                if concept == 2:
                    assert torch.equal(graph.x, edited.x)
                    assert torch.equal(
                        torch.bincount(graph.edge_index[0]), torch.bincount(edges[0])
                    )
                    assert len(edge_set(graph) - edge_set(edited)) == audit["budget"]
                    visited, pending = set(), [0]
                    while pending:
                        node = pending.pop()
                        if node not in visited:
                            visited.add(node)
                            pending.extend(edges[1, edges[0] == node].tolist())
                    assert len(visited) == graph.num_nodes
            if concept == 2:
                assert relation_fraction(target) != relation_fraction(graph)
                assert all(relation_fraction(c) == relation_fraction(graph) for c in controls)


def test_population_requires_trusted_labels():
    with pytest.raises(ValueError, match="trustworthy"):
        population_edit(graphs()[0], [0])


def test_small_excess_is_not_negligible_target():
    assert effect_interpretation(True, -0.001, 0.001, 0.5, CONFIG) == "inconclusive"
    assert effect_interpretation(True, -0.001, 0.001, 0.002, CONFIG).startswith("negligible")
    assert effect_interpretation(False, 0.2, 0.3, 0.4, CONFIG).startswith("inconclusive")


def test_cluster_uncertainty_not_inflated_by_repeated_observations():
    a = cluster_interval([0.1, 0.3], ["a", "b"], CONFIG, 3)
    b = cluster_interval([0.1] * 8 + [0.3] * 8, ["a"] * 8 + ["b"] * 8, CONFIG, 3)
    assert np.allclose(a, b)


def test_probe_training_scaler_and_heldout_guard():
    h = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 1, 1])
    probe, _, _, _ = probe_analysis(
        h, labels, h + 10, labels, ["a", "b", "c", "d"], ["e", "f", "g", "h"], 11, CONFIG, False
    )
    assert probe.named_steps["standardscaler"].mean_[0] == 1.5
    with pytest.raises(ValueError, match="overlap"):
        probe_analysis(
            h, labels, h, labels, ["a", "b", "c", "d"], ["a", "f", "g", "h"], 11, CONFIG, False
        )


def test_permutation_null_breaks_heldout_association_by_group():
    # Exact visible encoding should exceed a patient-group permutation null.
    # Random training coefficients alone can still perfectly rank true test labels.
    labels = np.tile([0, 1], 32)
    h = labels.astype(float).reshape(-1, 1)
    train_groups = np.repeat([f"train_{i}" for i in range(32)], 2)
    test_groups = np.repeat([f"test_{i}" for i in range(32)], 2)
    _, metrics, null, pvalue = probe_analysis(
        h, labels, h, labels, train_groups, test_groups, 11, CONFIG, False
    )
    assert metrics["auroc"] == 1
    assert 0.4 < null < 0.6
    assert pvalue <= CONFIG["probe"]["alpha"]


def test_nonpositive_area_control_fails_without_dropping_graph():
    graph = graphs()[0]
    graph.x[:, 0] = 0
    _, controls, valid, audit = matched_interventions(
        graph,
        0,
        "bracs",
        CONFIG,
        np.random.default_rng(11),
        raw_scale=torch.ones(3),
        raw_mean=torch.full((3,), 0.05),
    )
    assert len(controls) == CONFIG["matched_controls"]
    assert not valid
    assert "nonpositive nuclear area" in audit["invalid_reason"]

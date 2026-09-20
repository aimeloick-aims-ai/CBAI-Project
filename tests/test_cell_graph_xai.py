"""Unit tests for cell-level graph XAI and counterfactual interventions."""

import pytest
import torch

from src.interventions.cell_xai import (
    CellGNNExplainer,
    CellGraphInterventions,
    ConceptCellAligner,
)
from src.models.gnn import GraphClassifier


@pytest.fixture
def sample_cell_graph():
    num_nodes = 20
    in_channels = 8
    g = torch.Generator().manual_seed(42)
    x = torch.randn(num_nodes, in_channels, generator=g)

    # 4-regular KNN synthetic edge_index
    sources = torch.arange(num_nodes).repeat_interleave(4)
    targets = torch.roll(sources, shifts=1)
    edge_index = torch.stack([sources, targets], dim=0)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    return x, edge_index, batch


@pytest.fixture
def trained_classifier(sample_cell_graph):
    x, edge_index, batch = sample_cell_graph
    model = GraphClassifier("GCN", in_channels=8, hidden_channels=16, out_channels=2)
    model.eval()
    return model


def test_cell_gnn_explainer_masks(sample_cell_graph, trained_classifier):
    x, edge_index, batch = sample_cell_graph
    explainer = CellGNNExplainer(trained_classifier, epochs=20, learning_rate=0.05)

    res = explainer.explain_graph(x, edge_index, batch)

    assert res.node_mask.shape == (x.shape[0],)
    assert res.edge_mask.shape == (edge_index.shape[1],)
    assert (res.node_mask >= 0.0).all() and (res.node_mask <= 1.0).all()
    assert (res.edge_mask >= 0.0).all() and (res.edge_mask <= 1.0).all()
    assert len(res.top_node_indices) == x.shape[0]


def test_concept_cell_aligner(sample_cell_graph):
    x, edge_index, batch = sample_cell_graph
    node_mask = torch.rand(x.shape[0])
    feature_names = [f"feat_{i}" for i in range(x.shape[1])]

    aligner = ConceptCellAligner(feature_names=feature_names)
    corrs = aligner.compute_feature_correlations(node_mask, x)

    assert len(corrs) == x.shape[1]
    for name, val in corrs.items():
        assert -1.0 <= val <= 1.0


def test_cell_graph_interventions(sample_cell_graph):
    x, edge_index, batch = sample_cell_graph

    # Ablation test
    nodes_to_ablate = [0, 2, 4]
    x_zero, _ = CellGraphInterventions.ablate_nodes(
        x, edge_index, batch, nodes_to_ablate, mode="zero"
    )
    assert (x_zero[nodes_to_ablate] == 0.0).all()

    x_mean, _ = CellGraphInterventions.ablate_nodes(
        x, edge_index, batch, nodes_to_ablate, mode="mean"
    )
    assert not (x_mean[nodes_to_ablate] == 0.0).any()

    # Rewiring test
    edge_index_rewired = CellGraphInterventions.rewire_edges(
        edge_index, num_nodes=x.shape[0], rewire_ratio=0.5
    )
    assert edge_index_rewired.shape == edge_index.shape


def test_cell_intervention_fidelity(sample_cell_graph, trained_classifier):
    x, edge_index, batch = sample_cell_graph
    top_nodes = torch.tensor([0, 1, 2, 3, 4])

    fid_res = CellGraphInterventions.evaluate_cell_intervention_fidelity(
        trained_classifier, x, edge_index, batch, top_nodes, top_k=3, random_control_count=5
    )

    assert isinstance(fid_res.target_prob_drop, float)
    assert isinstance(fid_res.control_prob_drop_mean, float)
    assert fid_res.fidelity_score == pytest.approx(
        fid_res.target_prob_drop - fid_res.control_prob_drop_mean
    )

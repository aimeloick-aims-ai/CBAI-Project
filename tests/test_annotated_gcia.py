import torch

from scripts.run_annotated_gcia import components
from src.interventions.graph_edits import drop_undirected_edge


def test_graph_audit_detects_disconnection_without_changing_node_count():
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    assert components(edges, 3) == 1
    changed = drop_undirected_edge(edges, 1, 2)
    assert components(changed, 3) == 2
    assert components(changed, 4) == 3
    assert edges.shape[1] == 4


def test_graph_audit_preserves_connectivity_for_redundant_edge():
    edges = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]])
    assert components(drop_undirected_edge(edges, 0, 1), 3) == 1

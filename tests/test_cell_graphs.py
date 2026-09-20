import numpy as np
import pytest
import torch

from src.graphs.cells import cell_graph
from src.models.gnn import GraphClassifier
from src.segmentation.nuclei import instance_metrics, watershed_baseline


def example():
    rgb = np.full((20, 30, 3), 180, dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=np.int32)
    mask[2:6, 2:6] = 7
    mask[2:6, 10:14] = 42
    mask[14:18, 24:28] = 90
    return rgb, mask


def test_instance_metrics_ignore_id_values_and_penalize_missing_objects():
    _, mask = example()
    predicted = mask.copy()
    predicted[predicted == 7] = 123
    assert instance_metrics(predicted, mask)["panoptic_quality"] == 1
    predicted[predicted == 42] = 0
    result = instance_metrics(predicted, mask)
    assert result["true_positive"] == 2
    assert result["false_negative"] == 1
    assert result["panoptic_quality"] == 0.8


def test_merged_instances_do_not_get_double_credit():
    _, mask = example()
    merged = (mask > 0).astype(np.int32)
    result = instance_metrics(merged, mask)
    assert result["true_positive"] == 0
    assert result["false_positive"] == 1
    assert result["false_negative"] == 3


def test_empty_and_all_foreground_masks():
    empty = np.zeros((5, 5), dtype=np.int32)
    assert instance_metrics(empty, empty)["panoptic_quality"] is None
    assert instance_metrics(empty + 1, empty + 8)["panoptic_quality"] == 1
    assert not watershed_baseline(np.full((20, 20, 3), 255, dtype=np.uint8)).any()


def test_cell_geometry_units_and_radius():
    rgb, mask = example()
    g = cell_graph(rgb, mask, mpp=0.5, radius=5)
    assert g.instance_id.tolist() == [7, 42, 90]
    assert g.x[0, 0] == 4
    assert g.coordinate_units == "micrometers"
    assert {tuple(e) for e in g.edge_index.T.tolist()} == {(0, 1), (1, 0)}
    assert torch.allclose(g.pos[0], torch.tensor([1.75, 1.75]))
    with pytest.raises(ValueError):
        cell_graph(rgb, mask[:10])


@pytest.mark.parametrize("architecture", ["GCN", "GraphSAGE", "GATv2"])
def test_existing_architectures_accept_cell_graphs_and_backpropagate(architecture):
    rgb, mask = example()
    graph = cell_graph(rgb, mask)
    graph.x.requires_grad_(True)
    model = GraphClassifier(architecture, graph.x.shape[1], 8, 2)
    logits = model(graph.x, graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long))
    assert logits.shape == (1, 2)
    logits.square().sum().backward()
    assert torch.isfinite(graph.x.grad).all()

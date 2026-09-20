import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from scripts.audit_cell_xai import Ensemble, enlarge_nuclei
from src.graphs.cells import cell_graph
from src.models.gnn import GraphClassifier


def test_area_intervention_preserves_axis_ratio_and_other_features():
    x = torch.tensor([[100.0, 0.7, 0.9, 15.0, 10.0, 0.2, 0.3, 0.4]])
    changed = enlarge_nuclei(x, 1.05)
    assert torch.allclose(changed[:, 3] / changed[:, 4], x[:, 3] / x[:, 4])
    assert torch.allclose(changed[:, 3] * changed[:, 4], 1.05 * x[:, 3] * x[:, 4])
    assert torch.equal(changed[:, [1, 2, 5, 6, 7]], x[:, [1, 2, 5, 6, 7]])
    assert x[0, 0] == 100


@pytest.mark.parametrize("kind", ["GCN", "GraphSAGE", "GATv2"])
def test_ensemble_matches_original_batched_classifier(kind):
    mask = np.zeros((16, 16), dtype=np.int32)
    mask[1:5, 1:5], mask[8:12, 8:12] = 1, 2
    graph = cell_graph(np.full((16, 16, 3), 150, dtype=np.uint8), mask)
    models = [GraphClassifier(kind, 8, 16, 2).eval() for _ in range(3)]
    wrapper = Ensemble(models)
    batch = Batch.from_data_list([graph])
    expected = torch.stack(
        [m(batch.x, batch.edge_index, batch.batch).softmax(-1)[:, 1] for m in models]
    ).mean(0)
    assert torch.allclose(wrapper(graph.x, graph.edge_index), expected)
    assert wrapper.representation(graph.x, graph.edge_index).shape == (1, 48)

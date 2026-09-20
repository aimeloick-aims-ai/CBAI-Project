import numpy as np
import torch
from torch_geometric.data import Batch

from scripts.run_bracs_three_xai import BinaryModel, Ensemble, constrained_direction
from src.graphs.build import make_graph


def test_ensemble_matches_batched_binary_classifier():
    torch.manual_seed(11)
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1]])
    graph = make_graph(np.ones((5, 6)), coords, 0)
    models = [BinaryModel("gcn", 6).eval() for _ in range(3)]
    wrapper = Ensemble(models)
    expected = torch.stack([m(Batch.from_data_list([graph])).sigmoid() for m in models]).mean(0)
    assert torch.allclose(wrapper(graph.x, graph.edge_index), expected)


def test_constrained_direction_spares_other_linear_concepts():
    coef = np.random.default_rng(11).normal(size=(3, 10))
    for target in range(3):
        direction = constrained_direction(coef, target)
        assert np.isclose(np.linalg.norm(direction), 1)
        assert np.allclose(np.delete(coef, target, axis=0) @ direction, 0, atol=1e-10)
        assert coef[target] @ direction > 0

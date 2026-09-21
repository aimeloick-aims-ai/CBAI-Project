import numpy as np
import pytest
import torch

from src.interventions import matched_edge_indices, paired_edges, remove_nodes, unique_pairs


def test_node_removal_reindexes_and_preserves_features():
    x = torch.arange(12).reshape(4, 3)
    edges = paired_edges([[0, 1], [1, 2], [2, 3], [0, 3]])
    new_x, new_edges = remove_nodes(x, edges, [1])
    assert torch.equal(new_x, x[[0, 2, 3]])
    assert unique_pairs(new_edges).tolist() == [[0, 2], [1, 2]]
    with pytest.raises(ValueError, match="empty"):
        remove_nodes(x, edges, [0, 1, 2, 3])


def test_edge_matching_exact_strata_without_duplicates():
    strata = [(1, 4, 5), (1, 4, 5), (2, 5, 5), (2, 5, 5), (3, 4, 4)]
    target = [0, 2, 3]
    selected = matched_edge_indices(target, strata, np.random.default_rng(3))
    assert len(set(selected)) == 3
    assert sorted(strata[i] for i in selected) == sorted(strata[i] for i in target)
    assert paired_edges([]).shape == (2, 0)

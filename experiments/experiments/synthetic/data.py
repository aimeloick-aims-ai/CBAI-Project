"""Factorial graph data; label dependence is not learned-model dependence."""
import itertools
import numpy as np
import torch
from torch_geometric.data import Data
from src.graph import ring_edges

def graph(concepts, rng, identifier):
    used, unused, relational, _irrelevant = concepts
    permutation = rng.permutation(12)
    x = torch.zeros(12, 3)
    x[permutation, 0] = torch.tensor([-1.0] * 6 + [1.0] * 6)
    x[:, 1], x[:, 2] = 2 * used - 1, 2 * unused - 1
    return Data(
        x=x,
        edge_index=ring_edges(relational, permutation),
        y=torch.tensor([2 * used + relational]),
        concepts=torch.tensor([concepts], dtype=torch.float32),
        permutation=torch.tensor(permutation),
        sample_id=identifier,
    )


def make_split(repetitions, seed, split):
    rng = np.random.default_rng(seed)
    values = list(itertools.product((0, 1), repeat=4)) * repetitions
    rng.shuffle(values)
    return [graph(c, rng, f"{split}_{i}") for i, c in enumerate(values)]


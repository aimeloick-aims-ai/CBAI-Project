"""Factorial synthetic graphs. Generator dependence is NOT learned-model dependence."""

import itertools

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, global_mean_pool


def ring_edges(relational, permutation):
    # Same 12 nodes, six nodes of each type, degree two, one connected cycle.
    order = list(range(12)) if relational == 0 else [j for i in range(6) for j in (i, i + 6)]
    pairs = [(int(permutation[order[i]]), int(permutation[order[(i + 1) % 12]])) for i in range(12)]
    return torch.tensor(pairs + [(b, a) for a, b in pairs], dtype=torch.long).T.contiguous()


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


def intervene(data, concept):
    changed = data.clone()
    changed.concepts = data.concepts.clone()
    changed.concepts[0, concept] = 1 - changed.concepts[0, concept]
    if concept in (0, 1):
        changed.x[:, concept + 1] *= -1
    elif concept == 2:
        changed.edge_index = ring_edges(int(changed.concepts[0, 2]), data.permutation.numpy())
    elif concept != 3:
        raise ValueError("Unknown concept")
    # Irrelevant is deliberately unobserved: changing it must not alter the graph.
    changed.y = (2 * changed.concepts[:, 0] + changed.concepts[:, 2]).long()
    return changed


class StudyModel(nn.Module):
    def __init__(self, architecture, hidden=32, layers=3):
        super().__init__()
        self.architecture = architecture
        conv = {"GCN": GCNConv, "GraphSAGE": SAGEConv, "GATv2": GATv2Conv}
        self.layers = nn.ModuleList(
            [
                nn.Linear(3 if i == 0 else hidden, hidden)
                if architecture == "DeepSets"
                else conv[architecture](3 if i == 0 else hidden, hidden)
                for i in range(layers)
            ]
        )
        self.head = nn.Linear(hidden, 4)

    def representations(self, x, edges, batch):
        result = [global_mean_pool(x, batch)]
        for layer in self.layers:
            x = (layer(x) if self.architecture == "DeepSets" else layer(x, edges)).relu()
            result.append(global_mean_pool(x, batch))
        return result

    def forward(self, x, edge_index, batch):
        return self.head(self.representations(x, edge_index, batch)[-1])

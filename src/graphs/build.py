"""Graph builders for patch/node features."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


def knn_edges(coordinates: np.ndarray, k: int = 4) -> torch.Tensor:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2:
        raise ValueError("coordinates must contain at least two nodes")
    if not 1 <= k < coordinates.shape[0]:
        raise ValueError("k must be between 1 and n_nodes - 1")
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.sum(delta * delta, axis=2)
    np.fill_diagonal(distance, np.inf)
    rows = []
    cols = []
    for src, neighbours in enumerate(np.argsort(distance, axis=1)[:, :k]):
        for dst in neighbours:
            rows.append(src)
            cols.append(int(dst))
    edge_index = torch.tensor([rows + cols, cols + rows], dtype=torch.long)
    return torch.unique(edge_index, dim=1)


def make_graph(features: np.ndarray, coordinates: np.ndarray, y: int, k: int = 4) -> Data:
    if features.shape[0] != coordinates.shape[0]:
        raise ValueError("features and coordinates disagree on node count")
    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=knn_edges(coordinates, k=k),
        pos=torch.tensor(coordinates, dtype=torch.float32),
        y=torch.tensor([y], dtype=torch.long),
    )

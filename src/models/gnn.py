"""Small graph classifiers used by smoke and future training scripts."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, global_mean_pool


class GraphClassifier(nn.Module):
    def __init__(
        self, architecture: str, in_channels: int, hidden_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        convs = {"GCN": GCNConv, "GraphSAGE": SAGEConv, "GATv2": GATv2Conv}
        if architecture not in convs:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.conv = convs[architecture](in_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, out_channels)

    def encode(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Return graph embeddings used by probes without changing model weights."""
        hidden = self.conv(x, edge_index).relu()
        return global_mean_pool(hidden, batch)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        return self.head(self.encode(x, edge_index, batch))

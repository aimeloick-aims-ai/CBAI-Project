"""Local graph interventions with explicit admissibility checks."""

from __future__ import annotations

import torch


def drop_undirected_edge(edge_index: torch.Tensor, source: int, target: int) -> torch.Tensor:
    """Remove both directions of one existing edge without altering node coordinates."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, n_edges]")
    forward = (edge_index[0] == source) & (edge_index[1] == target)
    reverse = (edge_index[0] == target) & (edge_index[1] == source)
    if not forward.any() or not reverse.any():
        raise ValueError("Both directions of the requested edge must exist")
    return edge_index[:, ~(forward | reverse)]


def edge_subset(candidate: torch.Tensor, admissible: torch.Tensor) -> bool:
    """Check that an edited graph introduces no edges outside a predefined support."""
    permitted = {tuple(edge) for edge in admissible.t().tolist()}
    return all(tuple(edge) in permitted for edge in candidate.t().tolist())

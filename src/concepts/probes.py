"""Frozen-representation concept probes and held-out evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ProbeResult:
    """Metrics from a probe evaluated on examples excluded from fitting."""

    accuracy: float
    balanced_accuracy: float
    n_examples: int


class LinearConceptProbe(nn.Module):
    """A linear binary probe; representations must come from a frozen GNN."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_features, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings).squeeze(-1)


def fit_binary_probe(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int = 200,
    learning_rate: float = 0.05,
) -> LinearConceptProbe:
    """Fit a probe only; callers must supply training-fold embeddings and labels."""
    if embeddings.ndim != 2 or labels.ndim != 1 or len(embeddings) != len(labels):
        raise ValueError("Embeddings must be [n, d] and binary labels must be [n]")
    if len(torch.unique(labels)) != 2:
        raise ValueError("A binary concept probe requires both concept values")
    probe = LinearConceptProbe(embeddings.shape[1])
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(probe(embeddings), labels.float())
        loss.backward()
        optimizer.step()
    return probe.eval()


@torch.no_grad()
def evaluate_binary_probe(
    probe: LinearConceptProbe, embeddings: torch.Tensor, labels: torch.Tensor
) -> ProbeResult:
    """Evaluate a frozen probe on a held-out fold."""
    if len(embeddings) == 0 or len(embeddings) != len(labels):
        raise ValueError("Held-out embeddings and labels must be non-empty and aligned")
    prediction = (torch.sigmoid(probe(embeddings)) >= 0.5).long()
    labels = labels.long()
    accuracy = float((prediction == labels).float().mean())
    recalls = []
    for value in (0, 1):
        subset = labels == value
        if subset.any():
            recalls.append(float((prediction[subset] == value).float().mean()))
    return ProbeResult(accuracy, sum(recalls) / len(recalls), len(labels))

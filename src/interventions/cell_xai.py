"""Cell-level Graph XAI & Concept-Interventional Engine for GCIA.

Provides topological cell subgraph explanation, morphological concept alignment,
and selective counterfactual interventions on cell graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from src.concepts.probes import LinearConceptProbe, fit_binary_probe


@dataclass
class ExplanationResult:
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    top_node_indices: torch.Tensor
    loss: float


@dataclass
class InterventionFidelityResult:
    target_prob_drop: float
    control_prob_drop_mean: float
    control_prob_drop_std: float
    fidelity_score: float  # target_prob_drop - control_prob_drop_mean


class CellGNNExplainer:
    """Optimizes node and edge masks for explaining cell-graph predictions."""

    def __init__(
        self,
        model: nn.Module,
        epochs: int = 100,
        learning_rate: float = 0.05,
        edge_size_reg: float = 0.005,
        edge_ent_reg: float = 0.1,
    ) -> None:
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.edge_size_reg = edge_size_reg
        self.edge_ent_reg = edge_ent_reg

    def explain_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        target_class: int | None = None,
    ) -> ExplanationResult:
        """Find important nodes and edges for a given cell graph prediction."""
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        self.model.eval()

        with torch.no_grad():
            orig_out = self.model(x, edge_index, batch)
            if target_class is None:
                target_class = int(orig_out.argmax(dim=-1).item())

        num_nodes, num_edges = x.shape[0], edge_index.shape[1]

        # Learnable masks initialized near 1
        node_mask_param = nn.Parameter(torch.randn(num_nodes, device=x.device) * 0.1 + 1.0)
        edge_mask_param = nn.Parameter(torch.randn(num_edges, device=x.device) * 0.1 + 1.0)

        optimizer = torch.optim.Adam([node_mask_param, edge_mask_param], lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        target_tensor = torch.tensor([target_class], device=x.device)

        for _ in range(self.epochs):
            optimizer.zero_grad()

            node_mask = torch.sigmoid(node_mask_param)
            edge_mask = torch.sigmoid(edge_mask_param)

            # Apply node mask
            masked_x = x * node_mask.unsqueeze(-1)

            out = self.model(masked_x, edge_index, batch)
            pred_loss = criterion(out, target_tensor)

            # Size and entropy regularization
            size_loss = (
                self.edge_size_reg * edge_mask.mean() + self.edge_size_reg * node_mask.mean()
            )
            ent_loss = -self.edge_ent_reg * (
                (edge_mask * torch.log(edge_mask + 1e-8)).mean()
                + (node_mask * torch.log(node_mask + 1e-8)).mean()
            )

            total_loss = pred_loss + size_loss + ent_loss
            total_loss.backward()
            optimizer.step()

        node_mask = torch.sigmoid(node_mask_param).detach()
        edge_mask = torch.sigmoid(edge_mask_param).detach()

        top_indices = torch.argsort(node_mask, descending=True)

        return ExplanationResult(
            node_mask=node_mask,
            edge_mask=edge_mask,
            top_node_indices=top_indices,
            loss=float(total_loss.item()),
        )


class ConceptCellAligner:
    """Aligns cell-graph node importance with morphological & spatial concepts."""

    def __init__(self, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names

    def compute_feature_correlations(
        self, node_mask: torch.Tensor, node_features: torch.Tensor
    ) -> dict[str, float]:
        """Correlation between node importance mask and individual morphological features."""
        correlations = {}
        mask_np = node_mask.detach().cpu().numpy()
        feat_np = node_features.detach().cpu().numpy()

        import numpy as np

        for idx in range(feat_np.shape[1]):
            name = self.feature_names[idx] if self.feature_names else f"feature_{idx}"
            col = feat_np[:, idx]
            if col.std() > 1e-6 and mask_np.std() > 1e-6:
                corr = float(np.corrcoef(mask_np, col)[0, 1])
            else:
                corr = 0.0
            correlations[name] = corr
        return correlations

    def fit_cell_concept_probe(
        self,
        embeddings: torch.Tensor,
        concept_labels: torch.Tensor,
        steps: int = 150,
        lr: float = 0.05,
    ) -> LinearConceptProbe:
        """Fit a linear concept probe on cell node embeddings."""
        return fit_binary_probe(embeddings, concept_labels, steps=steps, learning_rate=lr)


class CellGraphInterventions:
    """Provides targeted ablation and spatial edge interventions on cell graphs."""

    @staticmethod
    def ablate_nodes(
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_indices: torch.Tensor | list[int],
        mode: Literal["zero", "mean"] = "zero",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ablate specified cell nodes in the graph by zeroing or replacing with mean."""
        x_perturbed = x.clone()
        if isinstance(node_indices, list):
            node_indices = torch.tensor(node_indices, dtype=torch.long, device=x.device)

        if mode == "zero":
            x_perturbed[node_indices] = 0.0
        elif mode == "mean":
            mean_feat = x.mean(dim=0, keepdim=True)
            x_perturbed[node_indices] = mean_feat

        return x_perturbed, edge_index

    @staticmethod
    def rewire_edges(
        edge_index: torch.Tensor,
        num_nodes: int,
        rewire_ratio: float = 0.2,
        seed: int = 42,
    ) -> torch.Tensor:
        """Randomly rewire a fraction of cell-cell spatial connections to test topological causality."""
        g = torch.Generator().manual_seed(seed)
        num_edges = edge_index.shape[1]
        num_rewire = int(num_edges * rewire_ratio)

        if num_rewire == 0:
            return edge_index.clone()

        rewire_indices = torch.randperm(num_edges, generator=g)[:num_rewire]
        edge_index_perturbed = edge_index.clone()

        new_targets = torch.randint(
            0, num_nodes, (num_rewire,), generator=g, device=edge_index.device
        )
        edge_index_perturbed[1, rewire_indices] = new_targets

        return edge_index_perturbed

    @staticmethod
    @torch.no_grad()
    def evaluate_cell_intervention_fidelity(
        model: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        top_node_indices: torch.Tensor,
        top_k: int = 5,
        random_control_count: int = 20,
        seed: int = 42,
    ) -> InterventionFidelityResult:
        """Evaluate how much target class probability drops under top-cell ablation vs random control."""
        model.eval()
        orig_probs = model(x, edge_index, batch).softmax(dim=-1)[0]
        target_class = int(orig_probs.argmax().item())
        orig_p = float(orig_probs[target_class].item())

        # 1. Targeted ablation of top K nodes
        top_k_indices = top_node_indices[:top_k]
        x_target, _ = CellGraphInterventions.ablate_nodes(
            x, edge_index, batch, top_k_indices, mode="zero"
        )
        target_p = float(model(x_target, edge_index, batch).softmax(dim=-1)[0, target_class].item())
        target_drop = orig_p - target_p

        # 2. Random control ablations of equal size
        g = torch.Generator().manual_seed(seed)
        num_nodes = x.shape[0]
        control_drops = []

        for _ in range(random_control_count):
            rand_indices = torch.randperm(num_nodes, generator=g)[:top_k]
            x_ctrl, _ = CellGraphInterventions.ablate_nodes(
                x, edge_index, batch, rand_indices, mode="zero"
            )
            ctrl_p = float(model(x_ctrl, edge_index, batch).softmax(dim=-1)[0, target_class].item())
            control_drops.append(orig_p - ctrl_p)

        control_mean = float(torch.tensor(control_drops).mean().item())
        control_std = (
            float(torch.tensor(control_drops).std().item()) if len(control_drops) > 1 else 0.0
        )

        return InterventionFidelityResult(
            target_prob_drop=target_drop,
            control_prob_drop_mean=control_mean,
            control_prob_drop_std=control_std,
            fidelity_score=target_drop - control_mean,
        )

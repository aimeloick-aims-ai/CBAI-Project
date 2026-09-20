"""Probe Robustness & Artifact Detection Engine for GCIA.

Provides statistical control tests to distinguish genuine concept encoding
from linear probe artifacts (e.g. overfitting, spurious correlation, probe sensitivity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.concepts.probes import LinearConceptProbe, evaluate_binary_probe, fit_binary_probe


@dataclass
class NullProbeResult:
    true_accuracy: float
    null_accuracy_mean: float
    null_accuracy_std: float
    selectivity_score: float  # true_accuracy - null_accuracy_mean
    p_value: float


@dataclass
class RandomDirectionResult:
    concept_prob_drop: float
    random_prob_drop_mean: float
    random_prob_drop_std: float
    probe_specificity_score: float  # concept_prob_drop - random_prob_drop_mean


@dataclass
class ProbeStabilityResult:
    mean_cosine_similarity: float
    std_cosine_similarity: float
    pairwise_similarities: list[float]


class NullProbeEvaluator:
    """Evaluates whether probe performance significantly exceeds random label permutations."""

    def __init__(self, num_permutations: int = 10, steps: int = 100, lr: float = 0.05) -> None:
        self.num_permutations = num_permutations
        self.steps = steps
        self.lr = lr

    def evaluate(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        test_embeddings: torch.Tensor | None = None,
        test_labels: torch.Tensor | None = None,
        *,
        train_patients: list[str] | None = None,
        test_patients: list[str] | None = None,
    ) -> NullProbeResult:
        if test_embeddings is None or test_labels is None:
            raise ValueError("Explicit held-out embeddings and labels are required")
        if test_embeddings.data_ptr() == embeddings.data_ptr():
            raise ValueError("Training embeddings cannot be the evaluation set")
        if self.num_permutations < 1:
            raise ValueError("At least one permutation is required")

        if train_patients is None or test_patients is None:
            raise ValueError("Patient identifiers required for disjoint patient-level evaluation")
        if len(train_patients) != len(embeddings) or len(test_patients) != len(test_embeddings):
            raise ValueError("Patient identifiers must align with rows")
        if len(set(train_patients)) != len(train_patients) or len(set(test_patients)) != len(
            test_patients
        ):
            raise ValueError("Provide one aggregated representation and concept label per patient")
        if set(train_patients) & set(test_patients):
            raise ValueError("Patient leakage between training and evaluation")
        # Rows are now independent patient units; labels are shuffled at that level.
        # Fit ground truth probe
        true_probe = fit_binary_probe(embeddings, labels, steps=self.steps, learning_rate=self.lr)
        true_res = evaluate_binary_probe(true_probe, test_embeddings, test_labels)
        true_acc = true_res.accuracy if hasattr(true_res, "accuracy") else float(true_res)

        # Null probes with permuted labels
        null_accs = []
        for seed in range(self.num_permutations):
            g = torch.Generator().manual_seed(1000 + seed)
            perm_indices = torch.randperm(labels.shape[0], generator=g)
            null_labels = labels[perm_indices]

            null_probe = fit_binary_probe(
                embeddings, null_labels, steps=self.steps, learning_rate=self.lr
            )
            null_res = evaluate_binary_probe(null_probe, test_embeddings, test_labels)
            null_acc = null_res.accuracy if hasattr(null_res, "accuracy") else float(null_res)
            null_accs.append(null_acc)

        null_mean = float(np.mean(null_accs))
        null_std = float(np.std(null_accs))
        selectivity = true_acc - null_mean

        # Empirical p-value
        count_exceed = sum(1 for acc in null_accs if acc >= true_acc)
        p_val = (count_exceed + 1.0) / (self.num_permutations + 1.0)

        return NullProbeResult(
            true_accuracy=true_acc,
            null_accuracy_mean=null_mean,
            null_accuracy_std=null_std,
            selectivity_score=selectivity,
            p_value=p_val,
        )


class RandomDirectionControlEvaluator:
    """Evaluates whether interventions along concept vector produce specific effects vs random directions."""

    def __init__(self, num_random_directions: int = 20, seed: int = 42) -> None:
        self.num_random_directions = num_random_directions
        self.seed = seed

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        concept_direction: torch.Tensor,
        alpha: float = 1.0,
    ) -> RandomDirectionResult:
        """Measures target probability drop under concept vector shift vs random unit vector shifts."""
        model.eval()
        orig_out = model(x, edge_index, batch).softmax(dim=-1)[0]
        target_class = int(orig_out.argmax().item())
        orig_p = float(orig_out[target_class].item())

        # Normalize concept direction
        c_dir = concept_direction.detach().squeeze()
        if c_dir.norm() > 1e-6:
            c_dir = c_dir / c_dir.norm()

        # Shift along concept direction
        x_concept = x + alpha * c_dir.unsqueeze(0)
        p_concept = float(
            model(x_concept, edge_index, batch).softmax(dim=-1)[0, target_class].item()
        )
        concept_drop = orig_p - p_concept

        # Shift along random unit directions
        g = torch.Generator().manual_seed(self.seed)
        random_drops = []
        feature_dim = x.shape[1]

        for _ in range(self.num_random_directions):
            rand_dir = torch.randn(feature_dim, generator=g, device=x.device)
            rand_dir = rand_dir / rand_dir.norm()

            x_rand = x + alpha * rand_dir.unsqueeze(0)
            p_rand = float(model(x_rand, edge_index, batch).softmax(dim=-1)[0, target_class].item())
            random_drops.append(orig_p - p_rand)

        rand_mean = float(np.mean(random_drops))
        rand_std = float(np.std(random_drops))

        return RandomDirectionResult(
            concept_prob_drop=concept_drop,
            random_prob_drop_mean=rand_mean,
            random_prob_drop_std=rand_std,
            probe_specificity_score=concept_drop - rand_mean,
        )


class ProbeStabilityEvaluator:
    """Measures cross-fold / cross-seed consistency of concept vector weights."""

    @staticmethod
    def evaluate(probes: list[LinearConceptProbe]) -> ProbeStabilityResult:
        if len(probes) < 2:
            raise ValueError("At least two probes are required to measure stability")

        weights = []
        for p in probes:
            if hasattr(p, "classifier"):
                w = p.classifier.weight.detach().squeeze()
            elif hasattr(p, "weight"):
                w = p.weight.detach().squeeze()
            else:
                raise ValueError("Probe object does not have weight attribute")
            weights.append(w)

        similarities = []
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                w1, w2 = weights[i], weights[j]
                norm1, norm2 = w1.norm(), w2.norm()
                if norm1 > 1e-6 and norm2 > 1e-6:
                    sim = float((w1 @ w2) / (norm1 * norm2))
                else:
                    sim = 0.0
                similarities.append(sim)

        return ProbeStabilityResult(
            mean_cosine_similarity=float(np.mean(similarities)),
            std_cosine_similarity=float(np.std(similarities)),
            pairwise_similarities=similarities,
        )

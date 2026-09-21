"""Directional sensitivity of a linear classification head; not a significance test."""

import numpy as np
import torch


def concept_direction(probe):
    scaler = probe.named_steps["standardscaler"]
    estimator = probe.steps[-1][1]
    coefficient = np.asarray(estimator.coef_).reshape(-1) / scaler.scale_
    norm = np.linalg.norm(coefficient)
    return torch.tensor(coefficient / max(norm, 1e-12), dtype=torch.float32), norm


def positive_fraction(head, predicted, direction, norm):
    if norm <= 1e-10:
        return None
    return float(((head.weight[predicted] @ direction) > 1e-8).float().mean())


def heldout_sensitivity(model, embeddings, probe, tolerance):
    """Derivative of fixed class-1 probability along increasing-concept CAV.

    Only the final pooled representation is a classifier input. Earlier pooled
    layer probes are reported separately; no fictitious intermediate TCAV is used.
    """
    direction, norm = concept_direction(probe)
    if norm <= tolerance:
        return float("nan")
    h = embeddings.detach().clone().requires_grad_(True)
    probability = model.head(h).softmax(-1)[:, 1]
    gradient = torch.autograd.grad(probability.sum(), h)[0]
    return float(((gradient @ direction) > tolerance).float().mean())

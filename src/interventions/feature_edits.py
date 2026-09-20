"""Bounded feature interventions for audit experiments."""

from __future__ import annotations

import torch


def clamp_feature_shift(
    x: torch.Tensor, feature: int, delta: float, lower: float = 0.0, upper: float = 1.0
):
    edited = x.clone()
    edited[:, feature] = torch.clamp(edited[:, feature] + delta, lower, upper)
    return edited


def prediction_shift(
    before_logits: torch.Tensor, after_logits: torch.Tensor, target_class: int
) -> float:
    """Mean probability change for a pre-specified target class after an edit."""
    if before_logits.shape != after_logits.shape:
        raise ValueError("Logits must have the same shape before and after an intervention")
    if not 0 <= target_class < before_logits.shape[-1]:
        raise ValueError("target_class is outside the logits")
    before = torch.softmax(before_logits, dim=-1)[:, target_class]
    after = torch.softmax(after_logits, dim=-1)[:, target_class]
    return float((after - before).mean().detach())

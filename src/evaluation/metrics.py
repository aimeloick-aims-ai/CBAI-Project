"""Metrics safe for tiny smoke fixtures and full cohorts."""

from __future__ import annotations

import numpy as np


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Predictions and labels must be non-empty and equally sized")
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def specificity(target_change: float, control_changes: list[float]) -> float:
    """Target effect minus mean matched-control effect.

    A positive score is only an audit statistic; validity of the image or graph
    edit must be established independently before it is interpreted.
    """
    if not control_changes:
        raise ValueError("At least one matched control intervention is required")
    return float(target_change - np.mean(control_changes))

"""Patch-level edits used only after an independent realism review."""

from __future__ import annotations

import numpy as np


def replace_masked_rgb(image: np.ndarray, replacement: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace pixels in a binary mask while preserving image shape and range."""
    if image.shape != replacement.shape or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image and replacement must be RGB arrays with the same shape")
    if mask.shape != image.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("mask must be a boolean array matching image height and width")
    edited = image.copy()
    edited[mask] = replacement[mask]
    return edited

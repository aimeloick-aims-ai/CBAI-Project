"""Small deterministic image-to-node extractor for engineering smoke tests.

This is not a replacement for the planned HoVer-Net validation. It creates a
graph-like set of patch nodes so downstream code can be exercised on real image
files before the supervised nucleus segmentation gate is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PatchNodes:
    features: np.ndarray
    coordinates: np.ndarray


def image_patch_nodes(path: Path, grid: int = 4, resize: int = 256) -> PatchNodes:
    if grid < 2:
        raise ValueError("grid must be at least 2")
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((resize, resize))
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    step = resize // grid
    features = []
    coordinates = []
    for row in range(grid):
        for column in range(grid):
            patch = pixels[row * step : (row + 1) * step, column * step : (column + 1) * step]
            mean = patch.mean(axis=(0, 1))
            std = patch.std(axis=(0, 1))
            features.append(np.concatenate([mean, std]).astype(np.float32))
            coordinates.append([float(column), float(row)])
    return PatchNodes(np.vstack(features), np.asarray(coordinates, dtype=np.float32))

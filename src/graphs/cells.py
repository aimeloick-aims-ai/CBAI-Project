"""Build radius-limited cellular graphs from nuclear instance maps."""

import numpy as np
import torch
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from torch_geometric.data import Data

from src.segmentation.nuclei import validate_instances

FEATURE_NAMES = [
    "area",
    "eccentricity",
    "solidity",
    "major_axis",
    "minor_axis",
    "mean_red",
    "mean_green",
    "mean_blue",
]


def cell_graph(rgb, instances, *, mpp=None, radius=50.0, k=8, min_area=4):
    """Coordinates/lengths use micrometers when mpp is known, otherwise pixels.

    No tissue or cell-type labels are inferred from shape or RGB features.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB")
    validate_instances(instances, rgb.shape[:2])
    if (
        (mpp is not None and (not np.isfinite(mpp) or mpp <= 0))
        or not np.isfinite(radius)
        or radius <= 0
        or k < 1
        or min_area < 1
    ):
        raise ValueError("Invalid scale, radius, neighbour count or area")
    factor = mpp if mpp is not None else 1.0
    features, positions, ids = [], [], []
    for region in regionprops(instances, intensity_image=rgb):
        if region.area < min_area:
            continue
        features.append(
            [
                region.area * factor**2,
                region.eccentricity,
                region.solidity,
                region.axis_major_length * factor,
                region.axis_minor_length * factor,
                *list(region.intensity_mean / 255.0),
            ]
        )
        positions.append([region.centroid[1] * factor, region.centroid[0] * factor])
        ids.append(region.label)
    if len(ids) < 2:
        raise ValueError("At least two retained nuclear instances are needed; no patch fallback")
    coords = np.asarray(positions)
    tree = cKDTree(coords)
    distances, neighbours = tree.query(coords, k=min(k + 1, len(ids)))
    edges = set()
    for src, (ds, ns) in enumerate(zip(distances, neighbours, strict=True)):
        for distance, dst in zip(ds, ns, strict=True):
            if src != dst and distance <= radius:
                edges.update([(src, int(dst)), (int(dst), src)])
    edge_index = (
        torch.tensor(sorted(edges), dtype=torch.long).T.contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    graph = Data(
        x=torch.tensor(np.asarray(features), dtype=torch.float32),
        pos=torch.tensor(coords, dtype=torch.float32),
        edge_index=edge_index,
        instance_id=torch.tensor(ids, dtype=torch.long),
    )
    graph.coordinate_units = "micrometers" if mpp is not None else "pixels"
    graph.feature_names = FEATURE_NAMES.copy()
    graph.segmentation_validated = False
    return graph

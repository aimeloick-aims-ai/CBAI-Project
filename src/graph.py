"""Build radius-limited cellular graphs from nuclear instance maps."""

import csv
import hashlib
import itertools
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from torch_geometric.data import Data

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


def validate_instances(mask, shape=None):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
        raise ValueError("Instance map must be a nonnegative 2D integer array; zero is background")
    if shape is not None and mask.shape != tuple(shape):
        raise ValueError("Instance map and image must have identical pixel dimensions")
    return mask


def ring_edges(relational, permutation):
    # Same 12 nodes, six nodes of each type, degree two, one connected cycle.
    order = list(range(12)) if relational == 0 else [j for i in range(6) for j in (i, i + 6)]
    pairs = [(int(permutation[order[i]]), int(permutation[order[(i + 1) % 12]])) for i in range(12)]
    return torch.tensor(pairs + [(b, a) for a, b in pairs], dtype=torch.long).T.contiguous()


def synthetic_label(used, relational, flip=False):
    """Ground truth is binary OR plus independent observation noise; no unused input."""
    return int(bool(used or relational) != bool(flip))


def synthetic_split(repetitions, seed, split, config):
    """Eight factorial states per independent replicate; paired relations share x exactly.

    Replicates are the uncertainty clusters. Noise and node identities are shared
    within a replicate, so factorial variants must never be treated as independent.
    """
    rng = np.random.default_rng(seed)
    graphs = []
    for replicate in range(repetitions):
        permutation = rng.permutation(12)
        noise = rng.normal(0, config["feature_noise"], (12, 3))
        flip = rng.random() < config["label_flip_probability"]
        for used, unused, relational in itertools.product((0, 1), repeat=3):
            x = noise.copy()
            x[permutation, 0] += np.array([-1.0] * 6 + [1.0] * 6)
            x[:, 1] += 2 * used - 1
            x[:, 2] += 2 * unused - 1
            graphs.append(
                Data(
                    x=torch.tensor(x, dtype=torch.float32),
                    edge_index=ring_edges(relational, permutation),
                    concepts=torch.tensor([[used, unused, relational]], dtype=torch.float32),
                    y=torch.tensor([synthetic_label(used, relational, flip)]),
                    permutation=torch.tensor(permutation),
                    patient_id=f"{split}_{replicate}",
                    sample_id=f"{split}_{replicate}_{used}{unused}{relational}",
                )
            )
    return graphs


def verify_patient_splits(rows):
    groups = {}
    for row in rows:
        patient, split = str(row["patient_id"]).strip(), row["split"]
        if not patient or patient.lower() in {"nan", "none"}:
            raise ValueError("Missing patient ID")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if patient in groups and groups[patient] != split:
            raise ValueError(f"Patient overlap: {patient}")
        groups[patient] = split
    if set(groups.values()) != {"train", "validation", "test"}:
        raise ValueError("All three patient partitions are required")


def load_bracs(root, config):
    """Reuse graph tensors; verify checksums, source images and independent patient mapping."""
    paths = {
        k: (Path(root) / config[k]).resolve()
        for k in ("manifest", "source_manifest", "patient_metadata", "cohort_protocol", "excluded")
    }

    def read_csv(path):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    rows = read_csv(paths["manifest"])
    verify_patient_splits(rows)
    if len({r["patient_id"] for r in rows}) != len(rows):
        raise ValueError("This cohort expects one graph per patient")
    sources = {r["patient_id"]: r for r in read_csv(paths["source_manifest"])}
    slides = {r["slide_id"]: r["patient_id"] for r in read_csv(paths["patient_metadata"])}
    splits = {s: [] for s in ("train", "validation", "test")}
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}
    units, validated = set(), []
    for row in rows:
        source = sources[row["patient_id"]]
        slide = "_".join(source["image_id"].split("_")[:2])
        if slides.get(slide) != row["patient_id"]:
            raise ValueError("Patient identity disagrees with slide metadata")
        if {"internal_test": "test"}.get(source["proposed_split"], source["proposed_split"]) != row[
            "split"
        ]:
            raise ValueError("Source split mismatch")
        if int(row["label"]) != int(source["roi_label"] == "IC"):
            raise ValueError("Source target mismatch")
        path = paths["manifest"].parent / row["graph_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Graph checksum mismatch: {path}")
        hashes[str(path)] = digest
        graph = Data(**torch.load(path, weights_only=True))
        if graph.source_image_sha256 != source["sha256"]:
            raise ValueError("Graph source image mismatch")
        if list(graph.feature_names) != FEATURE_NAMES:
            raise ValueError("Unexpected feature schema")
        if (
            graph.num_nodes < 2
            or not torch.isfinite(graph.x).all()
            or not torch.isfinite(graph.pos).all()
        ):
            raise ValueError("Invalid graph tensors")
        if graph.edge_index.numel() and (
            graph.edge_index.min() < 0 or graph.edge_index.max() >= graph.num_nodes
        ):
            raise ValueError("Invalid edge endpoint")
        units.add(graph.coordinate_units)
        validated.append(bool(graph.segmentation_validated))
        graph.y = torch.tensor([int(row["label"])])
        graph.concepts = graph.x[:, config["feature_index"]].mean().reshape(1, 1)
        graph.patient_id = row["patient_id"]
        graph.sample_id = row["patient_id"]
        splits[row["split"]].append(graph)
    if len(units) != 1:
        raise ValueError("Incompatible physical scales")
    return splits, {
        "files_sha256": hashes,
        "coordinate_units": sorted(units),
        "physical_scale_validated": False,
        "segmentation_validated": all(validated),
        "patients": {s: [g.patient_id for g in gs] for s, gs in splits.items()},
        "excluded": read_csv(paths["excluded"]),
        "concept_scope": "mean segmented nuclear area in pixels squared; technical positive control",
    }

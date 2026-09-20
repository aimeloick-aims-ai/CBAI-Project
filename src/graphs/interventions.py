"""Structural perturbations; masks must not be mistaken for clinical annotations."""

import numpy as np
import torch


def remove_nodes(x, edges, selected):
    keep = torch.ones(len(x), dtype=torch.bool, device=x.device)
    keep[selected] = False
    if not keep.any():
        raise ValueError("Cannot evaluate an empty graph")
    mapping = torch.full((len(x),), -1, dtype=torch.long, device=x.device)
    mapping[keep] = torch.arange(int(keep.sum()), device=x.device)
    retained = keep[edges[0]] & keep[edges[1]]
    return x[keep], mapping[edges[:, retained]]


def unique_pairs(edges):
    return np.unique(np.sort(edges.cpu().numpy().T, axis=1), axis=0)


def paired_edges(pairs):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return torch.from_numpy(np.concatenate([pairs, pairs[:, ::-1]], axis=0).T.copy())


def edge_strata(pairs, coordinates, degrees):
    lengths = np.sum((coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]]) ** 2, axis=1)
    return [
        (float(length), *sorted(degrees[pair].tolist()))
        for length, pair in zip(lengths, pairs, strict=True)
    ]


def matched_edge_indices(target, strata, rng):
    chosen = []
    for key in sorted({strata[i] for i in target}):
        pool = [i for i, value in enumerate(strata) if value == key]
        count = sum(strata[i] == key for i in target)
        chosen.extend(rng.choice(pool, count, replace=False).tolist())
    return np.array(chosen, dtype=np.int64)

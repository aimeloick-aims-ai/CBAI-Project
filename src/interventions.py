"""Structural perturbations; masks must not be mistaken for clinical annotations."""

import numpy as np
import torch

from src.graph import ring_edges


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


def intervene(data, concept):
    changed = data.clone()
    changed.concepts = data.concepts.clone()
    changed.concepts[0, concept] = 1 - changed.concepts[0, concept]
    if concept in (0, 1):
        changed.x[:, concept + 1] *= -1
    elif concept == 2:
        changed.edge_index = ring_edges(int(changed.concepts[0, 2]), data.permutation.numpy())
    elif concept != 3:
        raise ValueError("Unknown concept")
    # Irrelevant is deliberately unobserved: changing it must not alter the graph.
    changed.y = (2 * changed.concepts[:, 0] + changed.concepts[:, 2]).long()
    return changed


def feature_edit(data, column, delta):
    """Edit one measured coordinate; no claim of biological plausibility."""
    changed = data.clone()
    changed.x[:, column] += torch.as_tensor(delta, dtype=data.x.dtype)
    return changed


def population_edit(data, selected, *, trustworthy_labels=False):
    if not trustworthy_labels:
        raise ValueError("Population intervention requires trustworthy population labels")
    changed = data.clone()
    keep = torch.ones(data.num_nodes, dtype=torch.bool)
    keep[selected] = False
    changed.x, changed.edge_index = remove_nodes(data.x, data.edge_index, selected)
    for key in ("pos", "instance_id"):
        if key in data:
            changed[key] = data[key][keep].clone()
    return changed


def edge_set(graph):
    return {tuple(pair) for pair in unique_pairs(graph.edge_index).tolist()}


def relational_edit(data):
    changed = data.clone()
    changed.edge_index = ring_edges(1 - int(data.concepts[0, 2]), data.permutation.numpy())
    return changed


def relation_fraction(data):
    groups = torch.zeros(data.num_nodes, dtype=torch.bool)
    groups[data.permutation[6:]] = True
    edges = data.edge_index
    return float((groups[edges[0]] != groups[edges[1]]).float().mean())


def relation_control(data, target, rng, attempts):
    budget = len(edge_set(data) - edge_set(target))
    for _ in range(attempts):
        order = data.permutation.numpy().copy()
        order[:6] = rng.permutation(order[:6])
        order[6:] = rng.permutation(order[6:])
        control = data.clone()
        control.edge_index = ring_edges(int(data.concepts[0, 2]), order)
        if len(edge_set(data) - edge_set(control)) == budget:
            return control
    raise ValueError("Cannot match relational edit budget without changing the concept")


def structural_validity(original, edited):
    edges = edited.edge_index
    degree = lambda g: torch.bincount(g.edge_index[0], minlength=g.num_nodes)
    # All synthetic outputs are generated as single cycles, hence connected.
    return bool(
        original.num_nodes == edited.num_nodes
        and torch.equal(original.x, edited.x)
        and original.num_edges == edited.num_edges
        and torch.equal(degree(original), degree(edited))
        and not (edges[0] == edges[1]).any()
        and len(set(map(tuple, edges.T.tolist()))) == edited.num_edges
    )


def matched_interventions(data, concept, stage, config, rng, raw_scale=None, raw_mean=None):
    """Return target, controls and measured validity; every control is nontrivial.

    Feature controls edit the SAME column and node count with equal L2 budget,
    distributing signs across nodes rather than coherently shifting the mean.
    Synthetic relation controls preserve mixing fraction and exactly match the
    number of replaced edges, degree sequence and connected-cycle constraint.
    """
    tolerance = config["numerical_tolerance"]
    if stage == "synthetic" and concept == 2:
        target = relational_edit(data)
        controls = [
            relation_control(data, target, rng, config["control_attempts"])
            for _ in range(config["matched_controls"])
        ]
        budget = len(edge_set(data) - edge_set(target))
        valid = structural_validity(data, target) and relation_fraction(data) != relation_fraction(
            target
        )
        valid = valid and all(
            structural_validity(data, c)
            and relation_fraction(c) == relation_fraction(data)
            and len(edge_set(data) - edge_set(c)) == budget
            for c in controls
        )
        return (
            target,
            controls,
            bool(valid),
            {
                "budget": budget,
                "control_budget": budget,
                "concept_shift": abs(relation_fraction(target) - relation_fraction(data)),
                "control_concept_shift": 0.0,
            },
        )
    column = concept + 1 if stage == "synthetic" else config["bracs"]["feature_index"]
    strength = config[stage]["feature_strength"]
    delta = (
        strength * (1 - 2 * float(data.concepts[0, concept])) if stage == "synthetic" else strength
    )
    target = feature_edit(data, column, delta)
    controls = []
    for _ in range(config["matched_controls"]):
        signs = np.ones(data.num_nodes)
        signs[: data.num_nodes // 2] = -1
        # Odd counts have one extra sign; allowed only within configured ratio.
        rng.shuffle(signs)
        controls.append(feature_edit(data, column, torch.tensor(signs * delta, dtype=data.x.dtype)))
    target_norm = float(torch.linalg.vector_norm(target.x - data.x))
    norms = [float(torch.linalg.vector_norm(c.x - data.x)) for c in controls]
    target_shift = abs(float((target.x[:, column] - data.x[:, column]).mean()))
    control_shift = max(abs(float((c.x[:, column] - data.x[:, column]).mean())) for c in controls)
    reasons = []
    if not all(abs(n - target_norm) <= tolerance * max(1, target_norm) for n in norms):
        reasons.append("feature L2 budget mismatch")
    ratio = config["bracs"]["control_mean_shift_ratio"] if stage == "bracs" else tolerance
    if control_shift > ratio * target_shift + tolerance:
        reasons.append("control changes the concept mean beyond allowed ratio")
    # Positive measured area is necessary but not sufficient for plausibility.
    if stage == "bracs" and not all(
        bool((g.x[:, column] * raw_scale[column] + raw_mean[column] > 0).all())
        for g in [target, *controls]
    ):
        reasons.append("nonpositive nuclear area introduced by matched control")
    return (
        target,
        controls,
        not reasons,
        {
            "budget": target_norm,
            "control_budget": float(np.mean(norms)),
            "concept_shift": target_shift,
            "control_concept_shift": control_shift,
            "invalid_reason": "; ".join(reasons),
        },
    )

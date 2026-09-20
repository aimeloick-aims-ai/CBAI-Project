"""Exploratory BCSS tissue-probe and perturbation audit of BRACS-trained GNNs.

No downloads. No diagnostic evaluation of BCSS. Mask-derived tissue fractions
are independent probe targets; perturbations are NOT validated concept edits.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, global_mean_pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.manifest import read_bracs_smoke_manifest
from src.data.splits import assert_no_patient_overlap, label_index
from src.graphs.build import make_graph

GRID = 8
SIZE = 256
SEEDS = [11, 23, 37, 53, 71]


def features(rgb):
    """8x8 spatial grid, with mean/std RGB per node."""
    patches = rgb.reshape(GRID, SIZE // GRID, GRID, SIZE // GRID, 3)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(GRID * GRID, -1, 3)
    return np.concatenate([patches.mean(1), patches.std(1)], axis=1).astype("float32")


def graph(rgb, label=0):
    coords = np.array([[c, r] for r in range(GRID) for c in range(GRID)])
    result = make_graph(features(rgb), coords, label)
    result.edge_index = torch.unique(result.edge_index, dim=1)
    return result


def read_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB").resize((SIZE, SIZE)), dtype=np.float32) / 255


class LayerGNN(nn.Module):
    def __init__(self, architecture, classes):
        super().__init__()
        conv = {"GCN": GCNConv, "GraphSAGE": SAGEConv, "GATv2": GATv2Conv}[architecture]
        self.layers = nn.ModuleList([conv(6, 16), conv(16, 16)])
        self.head = nn.Linear(16, classes)

    def representations(self, g):
        xs = [g.x]
        for layer in self.layers:
            xs.append(layer(xs[-1], g.edge_index).relu())
        return xs

    def forward(self, g):
        return self.head(global_mean_pool(self.representations(g)[-1], g.batch))


def ridge_probe(train_x, train_y, test_x, test_y):
    """Train-only scaling and fixed ridge penalty; no validation tuning."""
    mean = train_x.mean(0)
    scale = train_x.std(0).clip(1e-6)
    a = np.column_stack([(train_x - mean) / scale, np.ones(len(train_x))])
    b = np.column_stack([(test_x - mean) / scale, np.ones(len(test_x))])
    penalty = np.eye(a.shape[1])
    penalty[-1, -1] = 0
    coef = np.linalg.solve(a.T @ a + penalty, a.T @ train_y)
    mse = float(np.mean((b @ coef - test_y) ** 2))
    return mse


def load_bcss(manifest=None, expected_patients=2):
    with (manifest or ROOT / "data/bcss_smoke_manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    pairs = {}
    for row in rows:
        path = ROOT / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("BCSS checksum mismatch")
        pair = pairs.setdefault(row["patient_id"], {})
        if row["kind"] in pair:
            raise ValueError("Multiple regions per patient need explicit grouping")
        pair[row["kind"]] = row
    result = []
    for patient, pair in sorted(pairs.items()):
        rgb = read_rgb(ROOT / pair["image"]["path"])
        with Image.open(ROOT / pair["mask"]["path"]) as im:
            mask = np.array(im.resize((SIZE, SIZE), Image.Resampling.NEAREST))
        # Fractions are computed only among annotated, non-excluded pixels.
        valid = (mask != 0) & (mask != 7)
        tiles = lambda a: a.reshape(GRID, 32, GRID, 32).transpose(0, 2, 1, 3).reshape(64, -1)
        counts = tiles(valid).sum(1)
        usable = counts >= 512
        tumor = tiles(mask == 1).sum(1) / counts.clip(1)
        result.append((patient, rgb, mask, tumor, usable))
    if len(result) != expected_patients:
        raise ValueError(f"Expected {expected_patients} distinct BCSS patients")
    return result


def variants(rgb, mask, seed):
    """Mask-guided occlusion and equal-budget random controls; no realism claim."""
    rng = np.random.default_rng(seed)
    base = Batch.from_data_list([graph(rgb)])
    tumor = mask == 1
    valid = (mask != 0) & (mask != 7)
    scores = tumor.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, -1).mean(1)
    selected = np.argsort(scores)[-8:]
    all_edges = base.edge_index
    pairs = sorted({tuple(sorted(e)) for e in all_edges.T.tolist()})
    selected_set = set(selected.tolist())
    target_pairs = [p for p in pairs if p[0] in selected_set or p[1] in selected_set]
    for level in ("attributes", "pixels", "edges"):
        for control in range(-1, 10):
            g = base.clone()
            if level == "attributes":
                nodes = selected if control == -1 else rng.choice(64, 8, replace=False)
                g.x[nodes] = base.x.mean(0)
                budget = 8
            elif level == "pixels":
                chosen = tumor.copy()
                if control != -1:
                    chosen[:] = False
                    idx = rng.choice(np.flatnonzero(valid), int(tumor.sum()), replace=False)
                    chosen.flat[idx] = True
                edited = rgb.copy()
                edited[chosen] = rgb[valid].mean(0)
                g = Batch.from_data_list([graph(edited)])
                budget = int(chosen.sum())
            else:
                remove = (
                    target_pairs
                    if control == -1
                    else [
                        pairs[i] for i in rng.choice(len(pairs), len(target_pairs), replace=False)
                    ]
                )
                remove = set(remove)
                keep = [tuple(sorted(e)) not in remove for e in all_edges.T.tolist()]
                g.edge_index = all_edges[:, keep]
                budget = len(remove)
            yield level, control, budget, g


def main():
    torch.set_num_threads(2)
    records = read_bracs_smoke_manifest()
    assert_no_patient_overlap(records)
    labels = label_index(records)
    groups = {"train": [], "val": []}
    with (ROOT / "data/bracs_smoke_manifest.csv").open(newline="") as f:
        hashes = {row["image_id"]: row["sha256"] for row in csv.DictReader(f)}
    for row in records:
        if row.split in groups:
            if hashlib.sha256(row.path.read_bytes()).hexdigest() != hashes[row.image_id]:
                raise ValueError("BRACS checksum mismatch")
            groups[row.split].append(graph(read_rgb(row.path), labels[row.label]))
    train, val = [Batch.from_data_list(groups[s]) for s in ("train", "val")]
    bcss = load_bcss()
    out = ROOT / "reports/real_xai"
    out.mkdir(parents=True, exist_ok=True)
    runs, probes, effects = [], [], []
    for architecture in ("GCN", "GraphSAGE", "GATv2"):
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = LayerGNN(architecture, len(labels))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
            for _ in range(60):
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(train), train.y)
                loss.backward()
                optimizer.step()
            model.eval().requires_grad_(False)
            torch.save(model.state_dict(), out / f"{architecture}_{seed}.pt")
            runs.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "validation_accuracy": float((model(val).argmax(1) == val.y).float().mean()),
                }
            )
            reps = []
            for patient, rgb, mask, target, usable in bcss:
                g = Batch.from_data_list([graph(rgb)])
                reps.append([x.numpy() for x in model.representations(g)])
                probabilities = model(g).softmax(1)
                fixed_class = int(probabilities.argmax(1))
                for level, control, budget, edited in variants(rgb, mask, seed):
                    delta = float((model(edited).softmax(1) - probabilities)[0, fixed_class])
                    effects.append(
                        {
                            "architecture": architecture,
                            "seed": seed,
                            "patient": patient,
                            "level": level,
                            "control": control,
                            "budget": budget,
                            "baseline_class": fixed_class,
                            "probability_change": delta,
                        }
                    )
            for test in (0, 1):
                fit = 1 - test
                ytrain, mtrain = bcss[fit][3:]
                ytest, mtest = bcss[test][3:]
                for layer in range(3):
                    a, b = reps[fit][layer][mtrain], reps[test][layer][mtest]
                    y, z = ytrain[mtrain], ytest[mtest]
                    shuffled = np.random.default_rng(seed).permutation(y)
                    probes.append(
                        {
                            "architecture": architecture,
                            "seed": seed,
                            "layer": layer,
                            "train_patient": bcss[fit][0],
                            "test_patient": bcss[test][0],
                            "train_nodes": len(y),
                            "test_nodes": len(z),
                            "mse": ridge_probe(a, y, b, z),
                            "shuffled_mse": ridge_probe(a, shuffled, b, z),
                            "constant_mse": float(np.mean((y.mean() - z) ** 2)),
                        }
                    )
            print(f"{architecture} seed {seed}: finished", flush=True)
    for name, rows in [("classification", runs), ("probes", probes), ("interventions", effects)]:
        with (out / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "scope": "exploratory_patch_graph_audit_not_validated_GCIA",
        "bcss_patients": 2,
        "runs": len(runs),
        "probe_evaluations": len(probes),
        "intervention_evaluations": len(effects),
        "seeds": SEEDS,
        "protocol": "60 epochs; 2 GNN layers; RGB 8x8 grid; ridge alpha=1; 10 random controls",
        "probe_target": "BCSS tumor code 1 fraction among valid pixels; masks nearest-resized",
        "unit_of_independence": "patient, not node or seed",
        "limitations": [
            "BCSS is out-of-domain for the BRACS classifier; probabilities are not diagnoses.",
            "Two BCSS patients cannot support population inference or reliable bootstrap CIs.",
            "Controls match edit counts, not spatial structure or all non-target concepts.",
            "Occlusion is not a realistic concept removal; no encoded/used classification is justified.",
            "Cell segmentation, NuCLS validation, concept validity and BACH evaluation remain missing.",
        ],
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "data/bracs_smoke_manifest.csv",
                ROOT / "data/bcss_smoke_manifest.csv",
                Path(__file__),
            ]
        },
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

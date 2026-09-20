"""Fixed exploratory extension on six BCSS patients, reusing frozen checkpoints."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_real_xai import LayerGNN, graph, load_bcss, ridge_probe


def tissue_targets(mask, code):
    valid = (mask != 0) & (mask != 7)

    def tiles(a):
        return a.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, -1)

    count = tiles(valid).sum(1)
    return tiles(mask == code).sum(1) / count.clip(1), count >= 512


def saliency_order(model, g, target):
    """Absolute gradient x displacement from the actual deletion baseline."""
    edited = g.clone()
    edited.x = g.x.detach().clone().requires_grad_(True)
    prob = model(edited).softmax(1)[0, target]
    grad = torch.autograd.grad(prob, edited.x)[0]
    score = (grad * (edited.x - g.x.mean(0))).abs().sum(1)
    return score.detach().argsort(descending=True).numpy()


def save_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(2)
    protocol_path = ROOT / "configs/extended_xai_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    manifest = ROOT / "data/bcss_extended_manifest.csv"
    patients = load_bcss(manifest, expected_patients=protocol["patients"])
    out = ROOT / "reports/extended_xai"
    out.mkdir(parents=True, exist_ok=True)
    probes, effects, weights = [], [], {}
    for architecture in ("GCN", "GraphSAGE", "GATv2"):
        for seed in (11, 23, 37, 53, 71):
            checkpoint = ROOT / f"reports/real_xai/{architecture}_{seed}.pt"
            weights[checkpoint.name] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            model = LayerGNN(architecture, 7)
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            model.eval().requires_grad_(False)
            graphs = [Batch.from_data_list([graph(p[1])]) for p in patients]
            representations = [[x.numpy() for x in model.representations(g)] for g in graphs]
            for concept, code in protocol["concept_codes"].items():
                targets = [tissue_targets(p[2], code) for p in patients]
                for test in range(len(patients)):
                    fit = [i for i in range(len(patients)) if i != test]
                    z, valid_test = targets[test]
                    if not valid_test.any():
                        raise ValueError("No valid test nodes")
                    rng = np.random.default_rng(seed)
                    y = np.concatenate([targets[i][0][targets[i][1]] for i in fit])
                    shuffled = np.concatenate(
                        [rng.permutation(targets[i][0][targets[i][1]]) for i in fit]
                    )
                    for layer in range(3):
                        a = np.concatenate([representations[i][layer][targets[i][1]] for i in fit])
                        b = representations[test][layer][valid_test]
                        probes.append(
                            {
                                "architecture": architecture,
                                "seed": seed,
                                "concept": concept,
                                "layer": layer,
                                "test_patient": patients[test][0],
                                "train_patients": "|".join(patients[i][0] for i in fit),
                                "mse": ridge_probe(a, y, b, z[valid_test]),
                                "shuffled_mse": ridge_probe(a, shuffled, b, z[valid_test]),
                                "constant_mse": float(np.mean((y.mean() - z[valid_test]) ** 2)),
                            }
                        )
            for i, g in enumerate(graphs):
                base = model(g).softmax(1)
                target = int(base.argmax(1))
                gradient_nodes = saliency_order(model, g, target)[:8]
                tumor, _ = tissue_targets(patients[i][2], 1)
                tumor_nodes = np.argsort(tumor)[-8:]
                rng = np.random.default_rng(seed)
                selections = [
                    ("tumor_mask", -1, tumor_nodes),
                    ("gradient_input", -1, gradient_nodes),
                ]
                selections += [("random", j, rng.choice(64, 8, replace=False)) for j in range(10)]
                for method, control, selected in selections:
                    edited = g.clone()
                    edited.x[selected] = g.x.mean(0)
                    delta = float((model(edited).softmax(1) - base)[0, target])
                    effects.append(
                        {
                            "architecture": architecture,
                            "seed": seed,
                            "patient": patients[i][0],
                            "method": method,
                            "control": control,
                            "probability_change": delta,
                            "nodes": "|".join(map(str, selected)),
                        }
                    )
            print(f"{architecture} seed {seed}: complete", flush=True)
    save_csv(out / "probes.csv", probes)
    save_csv(out / "deletion_baselines.csv", effects)
    summary = {
        "scope": "exploratory_not_final_GCIA",
        "patients": len(patients),
        "probe_evaluations": len(probes),
        "deletion_evaluations": len(effects),
        "checkpoint_hashes": weights,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "No nuclear segmentation or NuCLS validation.",
            "No BACH external diagnosis evaluation.",
            "Deletion realism and non-target preservation unvalidated.",
            "Six convenience-sampled patients; no population claim.",
            "Frozen BRACS models are out of domain on BCSS.",
            "Simple gradient baseline is not TCAV or GNNExplainer.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

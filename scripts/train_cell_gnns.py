"""Fixed three-architecture cell-graph comparison with strict patient partitions."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.gnn import GraphClassifier


def check_rows(rows):
    if not rows or len({r["patient_id"] for r in rows}) != len(rows):
        raise ValueError("Exactly one graph per distinct patient is required")
    if {r["split"] for r in rows} != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test partitions")
    for split in ("train", "validation", "test"):
        if {r["label"] for r in rows if r["split"] == split} != {"0", "1"}:
            raise ValueError("Each partition needs both binary classes")


def scores(y, p):
    labels = (p >= 0.5).astype(int)
    return {
        "patients": len(y),
        "accuracy": float(accuracy_score(y, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y, labels)),
        "auc": float(roc_auc_score(y, p)),
        "confusion": confusion_matrix(y, labels, labels=[0, 1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = list(csv.DictReader(args.manifest.open()))
    check_rows(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "architectures": ["GCN", "GraphSAGE", "GATv2"],
        "seeds": [11, 23, 37],
        "epochs": 120,
        "hidden": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "scope": "exploratory unless independent segmentation and cohort validation are documented",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "test_policy": "all predefined architectures, seed-mean probabilities, threshold .5; no selection or tuning on test",
    }
    with (args.output / "protocol.json").open("x") as f:
        json.dump(protocol, f, indent=2)
    signature = None

    def load(split):
        nonlocal signature
        graphs = []
        for row in rows:
            if row["split"] != split:
                continue
            path = args.manifest.parent / row["graph_path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Graph checksum mismatch")
            g = Data(**torch.load(path, weights_only=True))
            current = (tuple(g.feature_names), g.coordinate_units)
            if signature is not None and current != signature:
                raise ValueError("Incompatible feature schema or physical units")
            signature = current
            if g.num_nodes < 2 or not torch.isfinite(g.x).all() or not torch.isfinite(g.pos).all():
                raise ValueError("Invalid graph")
            g.y = torch.tensor([int(row["label"])])
            graphs.append(g)
        return graphs

    training, validation = load("train"), load("validation")
    combined = torch.cat([g.x for g in training])
    mean, scale = combined.mean(0), combined.std(0, unbiased=False).clamp_min(1e-6)

    def batch(graphs):
        copies = [g.clone() for g in graphs]
        for g in copies:
            g.x = (g.x - mean) / scale
        return Batch.from_data_list(copies)

    train, val = batch(training), batch(validation)
    models, results = {}, {"protocol": protocol, "validation": {}, "test": {}}
    for architecture in protocol["architectures"]:
        models[architecture], probabilities = [], []
        for seed in protocol["seeds"]:
            torch.manual_seed(seed)
            model = GraphClassifier(architecture, train.x.shape[1], 16, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
            for _ in range(120):
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(
                    model(train.x, train.edge_index, train.batch), train.y
                )
                loss.backward()
                optimizer.step()
            model.eval().requires_grad_(False)
            models[architecture].append(model)
            probabilities.append(model(val.x, val.edge_index, val.batch).softmax(-1)[:, 1].numpy())
            torch.save(
                {
                    "weights": model.state_dict(),
                    "mean": mean,
                    "scale": scale,
                    "feature_names": list(signature[0]),
                    "coordinate_units": signature[1],
                },
                args.output / f"{architecture}_{seed}.pt",
            )
        results["validation"][architecture] = scores(val.y.numpy(), np.mean(probabilities, axis=0))
    (args.output / "validation.json").write_text(json.dumps(results, indent=2))
    with (args.output / "test_opened.json").open("x") as f:
        json.dump(
            {
                "validation_sha256": hashlib.sha256(
                    (args.output / "validation.json").read_bytes()
                ).hexdigest()
            },
            f,
        )
    test = batch(load("test"))
    prediction_rows = []
    for architecture in protocol["architectures"]:
        p = np.mean(
            [
                m(test.x, test.edge_index, test.batch).softmax(-1)[:, 1].numpy()
                for m in models[architecture]
            ],
            axis=0,
        )
        results["test"][architecture] = scores(test.y.numpy(), p)
        for row, prob in zip([r for r in rows if r["split"] == "test"], p, strict=True):
            prediction_rows.append(
                {
                    "patient_id": row["patient_id"],
                    "architecture": architecture,
                    "label": row["label"],
                    "probability": float(prob),
                }
            )
    with (args.output / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

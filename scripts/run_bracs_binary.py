"""Fixed binary diagnostic pilot; choose on validation before opening internal test."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from tiatoolbox.models.architecture.vanilla import CNNModel
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_extended_xai import save_csv
from src.graphs.build import make_graph


class BinaryModel(nn.Module):
    def __init__(self, kind, channels):
        super().__init__()
        self.kind = kind
        if kind == "gcn":
            self.first = GCNConv(channels, 16)
            self.second = GCNConv(16, 16)
        else:
            self.first = nn.Linear(channels, 16)
            self.second = nn.Linear(16, 16)
        self.head = nn.Linear(16, 1)

    def forward(self, g):
        if self.kind == "gcn":
            x = self.second(self.first(g.x, g.edge_index).relu(), g.edge_index).relu()
            x = global_mean_pool(x, g.batch)
        else:
            x = self.second(self.first(global_mean_pool(g.x, g.batch)).relu()).relu()
        return self.head(x).flatten()


def metrics(labels, probabilities):
    labels, probabilities = np.array(labels), np.array(probabilities)
    pred = (probabilities >= 0.5).astype(int)
    return {
        "patients": len(labels),
        "accuracy": float((labels == pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix_N_IC": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
    }


def check_manifest(rows):
    from collections import Counter

    if len(rows) != 40 or len({r["patient_id"] for r in rows}) != 40:
        raise ValueError("Expected 40 distinct patients")
    counts = Counter((r["proposed_split"], r["roi_label"]) for r in rows)
    for split, n in [("train", 10), ("validation", 5), ("internal_test", 5)]:
        if any(counts[(split, label)] != n for label in ("N", "IC")):
            raise ValueError("Unexpected class balance")


def extract(rows, encoder):
    features = []
    for row in rows:
        path = ROOT / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Image checksum mismatch")
        with Image.open(path) as im:
            rgb = (
                np.array(
                    im.convert("RGB").resize((768, 768), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
                / 255
            )
        tiles = rgb.reshape(8, 96, 8, 96, 3).transpose(0, 2, 1, 3, 4).reshape(64, 96, 96, 3)
        embeddings = []
        with torch.no_grad():
            for start in (0, 32):
                x = torch.from_numpy(tiles[start : start + 32]).permute(0, 3, 1, 2)
                embeddings.append(encoder.pool(encoder.feat_extract(x)).flatten(1).numpy())
        features.append(np.concatenate(embeddings))
        print(f"Encoded {row['proposed_split']}: {row['image_id']}", flush=True)
    return features


def batch_graphs(rows, features, mean, scale):
    coords = np.array([[x, y] for y in range(8) for x in range(8)])
    return Batch.from_data_list(
        [
            make_graph((x - mean) / scale, coords, int(row["roi_label"] == "IC"))
            for row, x in zip(rows, features, strict=True)
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bracs_binary_protocol.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/bracs_binary_manifest.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/bracs_binary")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config_path = args.config
    config = json.loads(config_path.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "test_opened.json").exists():
        raise FileExistsError("Internal test already opened: do not rerun or tune on it")
    manifest = args.manifest
    if (
        config.get("manifest_sha256")
        and hashlib.sha256(manifest.read_bytes()).hexdigest() != config["manifest_sha256"]
    ):
        raise ValueError("Manifest checksum mismatch")
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if "custom_patient_split" not in config:
        check_manifest(rows)
    else:
        if len({r["patient_id"] for r in rows}) != len(rows):
            raise ValueError("Patient overlap")
        for split, count in config["splits"].items():
            subset = [r for r in rows if r["proposed_split"] == split]
            if len(subset) != count or {r["roi_label"] for r in subset} != {"N", "IC"}:
                raise ValueError("Unexpected allocation or missing class")
        if sum(config["splits"].values()) != len(rows):
            raise ValueError("Unexpected partition")
    groups = {
        s: [r for r in rows if r["proposed_split"] == s]
        for s in ("train", "validation", "internal_test")
    }
    weights = ROOT / "data/pretrained/mobilenet_v3_small-pcam.pth"
    encoder_config = json.loads((ROOT / "configs/encoder_protocol.json").read_text())
    if hashlib.sha256(weights.read_bytes()).hexdigest() != encoder_config["weight_sha256"]:
        raise ValueError("Encoder checksum mismatch")
    encoder = CNNModel("mobilenet_v3_small", num_classes=2)
    encoder.load_state_dict(torch.load(weights, weights_only=True))
    encoder.eval().requires_grad_(False)
    train_x = extract(groups["train"], encoder)
    val_x = extract(groups["validation"], encoder)
    combined = np.concatenate(train_x)
    mean, scale = combined.mean(0), combined.std(0).clip(1e-6)
    train = batch_graphs(groups["train"], train_x, mean, scale)
    val = batch_graphs(groups["validation"], val_x, mean, scale)
    validation, models, prediction_rows = {}, {}, []
    for kind in config["models"]:
        models[kind], probabilities = [], []
        for seed in config["seeds"]:
            torch.manual_seed(seed)
            model = BinaryModel(kind, combined.shape[1])
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
            )
            for _ in range(config["epochs"]):
                optimizer.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(model(train), train.y.float())
                loss.backward()
                optimizer.step()
            model.eval().requires_grad_(False)
            prob = model(val).sigmoid().numpy()
            probabilities.append(prob)
            models[kind].append(model)
            torch.save(
                {
                    "weights": model.state_dict(),
                    "mean": torch.tensor(mean),
                    "scale": torch.tensor(scale),
                },
                out / f"{kind}_{seed}.pt",
            )
            for row, p in zip(groups["validation"], prob, strict=True):
                prediction_rows.append(
                    {
                        "split": "validation",
                        "model": kind,
                        "seed": str(seed),
                        "patient": row["patient_id"],
                        "image": row["image_id"],
                        "label": row["roi_label"],
                        "probability_IC": float(p),
                    }
                )
        ensemble = np.mean(probabilities, axis=0)
        validation[kind] = metrics(val.y.tolist(), ensemble)
    # Sorted ties prefer mean_mlp. Selection is persisted before test image decoding.
    selected = max(
        config["models"], key=lambda k: (validation[k]["balanced_accuracy"], k == "mean_mlp")
    )
    selection = {
        "selected_model": selected,
        "validation": validation,
        "criterion": config["selection"],
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Validation selection committed: {selected}", flush=True)
    with (out / "test_opened.json").open("x") as f:
        json.dump(
            {"selection_sha256": hashlib.sha256((out / "selection.json").read_bytes()).hexdigest()},
            f,
        )
    test_x = extract(groups["internal_test"], encoder)
    test = batch_graphs(groups["internal_test"], test_x, mean, scale)
    ensemble = np.mean([m(test).sigmoid().numpy() for m in models[selected]], axis=0)
    for row, p in zip(groups["internal_test"], ensemble, strict=True):
        prediction_rows.append(
            {
                "split": "internal_test",
                "model": selected,
                "seed": "ensemble",
                "patient": row["patient_id"],
                "image": row["image_id"],
                "label": row["roi_label"],
                "probability_IC": float(p),
            }
        )
    save_csv(out / "predictions.csv", prediction_rows)
    result = {
        "scope": config["scope"],
        "selected_model": selected,
        "validation": validation,
        "internal_test": metrics(test.y.tolist(), ensemble),
        "baseline_balanced_accuracy": 0.5,
        "protocol": config,
        "encoder_sha256": encoder_config["weight_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            f"Only {len(groups['internal_test'])} test patients; no clinical claim.",
            "Small-file sampling bias; internal split not official test.",
            "Unknown physical scale of BRACS RoI after resize; PCam domain mismatch.",
            "Binary extremes do not represent intermediate lesions or full diagnosis.",
        ],
    }
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

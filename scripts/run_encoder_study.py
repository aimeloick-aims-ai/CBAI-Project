"""Fixed RGB versus frozen PCam representation comparison; no downloads."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tiatoolbox.models.architecture.vanilla import CNNModel
from torch import nn
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_extended_xai import save_csv
from scripts.run_focused_study import FractionGCN, load_patients
from src.graphs.build import make_graph


class PatchMLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, g):
        return self.net(g.x).squeeze(-1)


def build_graph(patient, feature_set, mean, scale):
    g = make_graph((patient[feature_set] - mean) / scale, patient["coords"], 0)
    g.target = torch.tensor(patient["fractions"][:, 1], dtype=torch.float32)
    g.valid = torch.tensor(patient["valid"])
    return g


def main():
    torch.set_num_threads(2)
    cfg_path = ROOT / "configs/encoder_protocol.json"
    cfg = json.loads(cfg_path.read_text())
    weights = ROOT / "data/pretrained" / (cfg["encoder"] + ".pth")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != cfg["weight_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    out = ROOT / "reports/encoder_study"
    out.mkdir(parents=True, exist_ok=True)
    # Preserve a once-only new-patient result rather than silently overwrite it.
    if (out / "prediction.csv").exists():
        raise FileExistsError(
            "Results already exist; use a separately versioned protocol for another run"
        )
    patients = load_patients(cfg, ROOT / "data/bcss_encoder_manifest.csv")
    development_ids = {p["patient"] for p in load_patients(cfg)}
    dev = [i for i, p in enumerate(patients) if p["patient"] in development_ids]
    heldout = [i for i, p in enumerate(patients) if p["patient"] not in development_ids]
    if len(dev) != 6 or len(heldout) != 2:
        raise ValueError("Expected six development and two new patients")
    encoder = CNNModel("mobilenet_v3_small", num_classes=2)
    encoder.load_state_dict(torch.load(weights, weights_only=True), strict=True)
    encoder.eval().requires_grad_(False)
    for p in patients:
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(p["patches"]), 32):
                # TIAToolbox PCam preprocessing is ToTensor only (RGB /255).
                x = torch.from_numpy(np.stack(p["patches"][start : start + 32])).permute(0, 3, 1, 2)
                h = encoder.pool(encoder.feat_extract(x)).flatten(1)
                embeddings.append(h.numpy())
        p["pcam"] = np.concatenate(embeddings)
        p["rgb"] = p["features"][:, :6]
        np.savez_compressed(out / f"{p['patient']}_features.npz", pcam=p["pcam"], rgb=p["rgb"])
    rows = []
    splits = [("development", [i], [j for j in dev if j != i]) for i in dev]
    splits.append(("held_out", heldout, dev))
    for fold, (split, test_ids, train_ids) in enumerate(splits):
        train_patients = [patients[i] for i in train_ids]
        for feature_set in cfg["feature_sets"]:
            x = np.concatenate([p[feature_set][p["valid"]] for p in train_patients])
            mean, scale = x.mean(0), x.std(0).clip(1e-6)
            train = Batch.from_data_list(
                [build_graph(p, feature_set, mean, scale) for p in train_patients]
            )
            constant = float(
                np.mean([p["fractions"][p["valid"], 1].mean() for p in train_patients])
            )
            for kind in cfg["models"]:
                for seed in cfg["seeds"]:
                    torch.manual_seed(seed)
                    model = (PatchMLP if kind == "mlp" else FractionGCN)(x.shape[1])
                    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
                    for _ in range(cfg["epochs"]):
                        opt.zero_grad()
                        predicted = model(train)
                        losses = [
                            (
                                predicted[(train.batch == j) & train.valid]
                                - train.target[(train.batch == j) & train.valid]
                            )
                            .square()
                            .mean()
                            for j in range(len(train_ids))
                        ]
                        torch.stack(losses).mean().backward()
                        opt.step()
                    model.eval().requires_grad_(False)
                    for idx in test_ids:
                        p = patients[idx]
                        g = build_graph(p, feature_set, mean, scale)
                        pred = model(g)[g.valid].numpy()
                        truth = g.target[g.valid].numpy()
                        rows.append(
                            {
                                "split": split,
                                "patient": p["patient"],
                                "features": feature_set,
                                "model": kind,
                                "seed": seed,
                                "patches": len(pred),
                                "patch_mse": float(np.mean((pred - truth) ** 2)),
                                "region_mae": float(abs(pred.mean() - truth.mean())),
                                "constant_patch_mse": float(np.mean((constant - truth) ** 2)),
                                "constant_region_mae": float(abs(constant - truth.mean())),
                            }
                        )
                    torch.save(
                        {
                            "weights": model.state_dict(),
                            "mean": torch.tensor(mean),
                            "scale": torch.tensor(scale),
                            "train_patients": [patients[j]["patient"] for j in train_ids],
                        },
                        out / f"{fold}_{feature_set}_{kind}_{seed}.pt",
                    )
        print(f"{split} fold {fold}: complete", flush=True)
    save_csv(out / "prediction.csv", rows)
    summary = {
        "scope": "exploratory_representation_comparison_not_GCIA_validation",
        "development_patients": sorted(development_ids),
        "new_patients": [patients[i]["patient"] for i in heldout],
        "encoder_frozen": True,
        "encoder_dimensions": patients[0]["pcam"].shape[1],
        "protocol": cfg,
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [cfg_path, weights, Path(__file__), ROOT / "data/bcss_encoder_manifest.csv"]
        },
        "limits": [
            "Only two new convenience-sampled patients; no population inference.",
            "BCSS breast tissue differs from PCam pretraining domain.",
            "No expert validation of annotations or intervention realism.",
            "Masks have finite resolution; whole-slide diagnosis not evaluated.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

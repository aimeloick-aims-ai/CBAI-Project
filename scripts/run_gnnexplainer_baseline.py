"""Adapted GNNExplainer regression baseline on existing frozen PCam/GCN models."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from torch import nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_extended_xai import save_csv
from scripts.run_focused_study import FractionGCN, load_patients
from src.graphs.build import make_graph


class RegionalModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, valid):
        return self.model(Data(x=x, edge_index=edge_index))[valid].mean().reshape(1)


def deletion_prediction(model, x, edge_index, valid, nodes):
    edited = x.clone()
    edited[nodes] = 0
    with torch.no_grad():
        return float(model(edited, edge_index, valid)[0])


def visualize(patient, scores, out):
    coords = patient["coords"].astype(int)
    cols, rows = coords.max(0) + 1
    size = patient["patches"][0].shape[0]
    rgb = np.zeros((rows * size, cols * size, 3))
    heat = np.zeros((rows, cols))
    chosen = np.zeros((rows, cols))
    top = set(np.argsort(scores)[-max(1, int(np.ceil(0.1 * len(scores)))) :].tolist())
    for i, ((x, y), patch) in enumerate(zip(coords, patient["patches"], strict=True)):
        rgb[y * size : (y + 1) * size, x * size : (x + 1) * size] = patch
        heat[y, x] = scores[i]
        chosen[y, x] = i in top
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Original patches (cropped borders)")
    im = axes[1].imshow(heat, vmin=0, vmax=1, cmap="viridis")
    axes[1].set_title("GNNExplainer node mask, seed 11")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(chosen, vmin=0, vmax=1, cmap="gray")
    axes[2].set_title("Top 10% nodes (white)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(patient["patient"])
    fig.tight_layout()
    fig.savefig(out / (patient["patient"] + ".png"), dpi=120)
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    config_path = ROOT / "configs/gnnexplainer_protocol.json"
    config = json.loads(config_path.read_text())
    source = ROOT / "reports/encoder_study"
    encoder_cfg = json.loads((ROOT / "configs/encoder_protocol.json").read_text())
    previous = json.loads((source / "summary.json").read_text())
    patients = load_patients(encoder_cfg, ROOT / "data/bcss_encoder_manifest.csv")
    out = ROOT / "reports/gnnexplainer_baseline"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "deletion.csv").exists():
        raise FileExistsError("Preserve existing results; version the protocol before another run")
    rows, explanations, provenance = [], [], {}
    for patient in patients:
        pid = patient["patient"]
        dev = previous["development_patients"]
        fold = dev.index(pid) if pid in dev else 6
        split = "development" if pid in dev else "previously_examined_holdout"
        features_path = source / f"{pid}_features.npz"
        features = np.load(features_path)["pcam"]
        provenance[features_path.name] = hashlib.sha256(features_path.read_bytes()).hexdigest()
        for seed in config["seeds"]:
            torch.manual_seed(seed)
            checkpoint = source / f"{fold}_pcam_gcn_{seed}.pt"
            state = torch.load(checkpoint, weights_only=True)
            if pid in state["train_patients"]:
                raise ValueError("Patient leakage in checkpoint")
            provenance[checkpoint.name] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            model = FractionGCN(features.shape[1])
            model.load_state_dict(state["weights"])
            model.eval().requires_grad_(False)
            original = {k: v.clone() for k, v in model.state_dict().items()}
            wrapper = RegionalModel(model).eval()
            x = (torch.from_numpy(features) - state["mean"]) / state["scale"]
            g = make_graph(x.numpy(), patient["coords"], 0)
            valid = torch.tensor(patient["valid"])
            baseline = float(wrapper(g.x, g.edge_index, valid)[0])
            explainer = Explainer(
                model=wrapper,
                algorithm=GNNExplainer(epochs=config["epochs"], lr=config["learning_rate"]),
                explanation_type="model",
                node_mask_type="object",
                edge_mask_type="object",
                model_config={"mode": "regression", "task_level": "graph", "return_type": "raw"},
            )
            explanation = explainer(g.x, g.edge_index, valid=valid)
            scores = explanation.node_mask.detach().flatten().numpy()
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite explanation")
            if not all(torch.equal(original[k], v) for k, v in model.state_dict().items()):
                raise ValueError("Explainer changed model weights")
            after = float(wrapper(g.x, g.edge_index, valid)[0])
            if abs(after - baseline) > 1e-7:
                raise ValueError("Explainer did not clear its masks")
            np.savez_compressed(
                out / f"{pid}_{seed}_masks.npz",
                node=scores,
                edge=explanation.edge_mask.detach().numpy(),
                edge_index=g.edge_index.numpy(),
            )
            explanations.append(
                {
                    "patient": pid,
                    "seed": seed,
                    "nodes": len(scores),
                    "baseline_prediction": baseline,
                    "score_min": float(scores.min()),
                    "score_max": float(scores.max()),
                    "weights_unchanged": True,
                }
            )
            order = np.argsort(-scores, kind="stable")
            rng = np.random.default_rng(seed)
            for fraction in config["fractions"]:
                count = max(1, int(np.ceil(len(scores) * fraction)))
                selections = [("gnnexplainer", -1, order[:count])]
                selections += [
                    ("random", j, rng.choice(len(scores), count, replace=False))
                    for j in range(config["random_controls"])
                ]
                for method, control, selected in selections:
                    prediction = deletion_prediction(wrapper, g.x, g.edge_index, valid, selected)
                    rows.append(
                        {
                            "patient": pid,
                            "split": split,
                            "seed": seed,
                            "fraction": fraction,
                            "count": count,
                            "method": method,
                            "control": control,
                            "signed_change": prediction - baseline,
                            "absolute_change": abs(prediction - baseline),
                        }
                    )
            if seed == 11:
                visualize(patient, scores, out)
            print(f"{pid} seed {seed}: explained and checked", flush=True)
    save_csv(out / "deletion.csv", rows)
    save_csv(out / "explanations.csv", explanations)
    summary = {
        "scope": config["scope"],
        "explanations": len(explanations),
        "deletion_evaluations": len(rows),
        "patients": len(patients),
        "torch_geometric_version": torch_geometric.__version__,
        "default_regularizers": GNNExplainer.default_coeffs,
        "sources": provenance,
        "protocol": config,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "limitations": [
            "Adaptation to regression; not exact replication of original benchmarks.",
            "Nodes and edges optimized jointly; node-only deletion tests one projection.",
            "Feature replacement is not physical tissue removal.",
            "Eight patients already examined; no new external validation.",
            "High sensitivity does not establish concept specificity or biological causality.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

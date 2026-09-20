"""Fixed exploratory XAI audit; adaptations are not full clinical GCIA validation."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from tiatoolbox.models.architecture.vanilla import CNNModel
from torch import nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_bracs_binary import BinaryModel, extract
from scripts.run_extended_xai import save_csv
from src.graphs.build import make_graph

OUT = ROOT / "reports/bracs_three_xai"
CONCEPTS = ["brightness_proxy", "red_minus_blue_proxy", "intensity_sd_proxy"]


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, edge_index):
        graph = Data(x=x, edge_index=edge_index, batch=torch.zeros(len(x), dtype=torch.long))
        return torch.stack([m(graph).sigmoid() for m in self.models]).mean(0)


def activations(models, x, edges):
    first = [m.first(x, edges).relu() for m in models]
    second = [m.second(h, edges).relu() for m, h in zip(models, first, strict=True)]
    return [torch.cat(first, dim=1), torch.cat(second, dim=1)]


def proxy_values(row):
    with Image.open(ROOT / row["path"]) as im:
        rgb = (
            np.asarray(im.convert("RGB").resize((768, 768), Image.Resampling.BILINEAR), dtype=float)
            / 255
        )
    patches = rgb.reshape(8, 96, 8, 96, 3).transpose(0, 2, 1, 3, 4).reshape(64, 96, 96, 3)
    gray = patches.mean(-1)
    return np.stack(
        [gray.mean((1, 2)), (patches[..., 0] - patches[..., 2]).mean((1, 2)), gray.std((1, 2))],
        axis=1,
    )


def constrained_direction(coef, target):
    other = np.delete(coef, target, axis=0).astype(float)
    direction = coef[target].astype(float)
    direction -= other.T @ np.linalg.pinv(other @ other.T) @ (other @ direction)
    norm = np.linalg.norm(direction)
    return direction / max(norm, 1e-12)


def paired_interval(values):
    values = np.asarray(values)
    rng = np.random.default_rng(2026)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
    return {
        "mean": float(values.mean()),
        "patient_bootstrap_95": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True, parents=True)
    if (OUT / "started.json").exists():
        raise FileExistsError("Preserve audit: already started. Inspect logs; do not tune on test.")
    protocol = {
        "scope": "exploratory analysis of already examined internal test; no external validation",
        "gnnexplainer": "PyG official; exact frozen 3-seed probability ensemble; 100 epochs; seeds 11,23,37; node and edge masks; node deletion only",
        "deletion": "5,10,20% nodes replaced with normalized training mean; 20 matched random subsets; predicted-class probability decrease",
        "gcexplainer": "adaptation: k=5 kmeans on concatenated last-layer node activations; train-only clusters; histogram + depth3 decision tree imitates ensemble labels; no human review or graph-edit-distance purity",
        "gcia": "technical proxy audit only: image-derived brightness, red-blue, intensity SD; train-only ridge probes at input/layer1/layer2; bounded input-feature edits orthogonal to non-target input probes",
        "gcia_edits": "both signs, node L2=1, all nodes; 20 isotropic equal-norm controls; input probe target >=0.1 train SD, collateral <=0.05 SD; feature-range support diagnostic only",
        "gcia_clinical_validity": False,
        "seeds": [11, 23, 37],
        "sources": [
            "https://arxiv.org/abs/2107.11889",
            "https://github.com/CharlotteMagister/GCExplainer",
        ],
    }
    manifest = ROOT / "data/bracs_expanded_study_manifest.csv"
    protocol["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    protocol["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "started.json").write_text(json.dumps(protocol, indent=2))
    rows = list(csv.DictReader(manifest.open(newline="")))
    if len({r["patient_id"] for r in rows}) != len(rows):
        raise ValueError("Patient overlap")
    expected = json.loads((ROOT / "configs/bracs_expanded_protocol.json").read_text())
    if protocol["manifest_sha256"] != expected["manifest_sha256"]:
        raise ValueError("Manifest modified")
    train_ids = [i for i, r in enumerate(rows) if r["proposed_split"] == "train"]
    test_ids = [i for i, r in enumerate(rows) if r["proposed_split"] == "internal_test"]
    models, provenance = [], {}
    for seed in protocol["seeds"]:
        checkpoint = ROOT / f"reports/bracs_expanded/gcn_{seed}.pt"
        provenance[checkpoint.name] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        state = torch.load(checkpoint, weights_only=True)
        model = BinaryModel("gcn", 576)
        model.load_state_dict(state["weights"])
        models.append(model.eval().requires_grad_(False))
    wrapper = Ensemble(models).eval()
    original = {k: v.clone() for k, v in wrapper.state_dict().items()}
    weights = ROOT / "data/pretrained/mobilenet_v3_small-pcam.pth"
    enc_cfg = json.loads((ROOT / "configs/encoder_protocol.json").read_text())
    if hashlib.sha256(weights.read_bytes()).hexdigest() != enc_cfg["weight_sha256"]:
        raise ValueError("Encoder modified")
    encoder = CNNModel("mobilenet_v3_small", num_classes=2)
    encoder.load_state_dict(torch.load(weights, weights_only=True))
    encoder.eval().requires_grad_(False)
    features = extract(rows, encoder)
    coords = np.array([[x, y] for y in range(8) for x in range(8)])
    graphs = [
        make_graph(
            ((torch.from_numpy(f) - state["mean"]) / state["scale"]).numpy(),
            coords,
            int(r["roi_label"] == "IC"),
        )
        for r, f in zip(rows, features, strict=True)
    ]
    probs = np.array([float(wrapper(g.x, g.edge_index)[0]) for g in graphs])
    old = list(csv.DictReader((ROOT / "reports/bracs_expanded/predictions.csv").open()))
    old = {r["patient"]: float(r["probability_IC"]) for r in old if r["split"] == "internal_test"}
    if any(abs(probs[i] - old[rows[i]["patient_id"]]) > 1e-5 for i in test_ids):
        raise ValueError("Frozen ensemble predictions differ from published results")
    hs = [activations(models, g.x, g.edge_index) for g in graphs]
    # Reusable cached representations, never fitted on evaluation patients.
    np.savez_compressed(
        OUT / "representations.npz",
        x=np.stack([g.x.numpy() for g in graphs]),
        layer1=np.stack([h[0].numpy() for h in hs]),
        layer2=np.stack([h[1].numpy() for h in hs]),
        patients=np.array([r["patient_id"] for r in rows]),
    )
    targets = np.stack([proxy_values(r) for r in rows])
    target_mean = targets[train_ids].reshape(-1, 3).mean(0)
    target_scale = targets[train_ids].reshape(-1, 3).std(0).clip(1e-6)
    targets = (targets - target_mean) / target_scale
    representations = [np.stack([g.x.numpy() for g in graphs])] + [
        np.stack([h[layer].numpy() for h in hs]) for layer in (0, 1)
    ]
    probes, probe_rows = [], []
    for layer, values in enumerate(representations):
        probe = Ridge(alpha=10).fit(
            values[train_ids].reshape(-1, values.shape[-1]), targets[train_ids].reshape(-1, 3)
        )
        probes.append(probe)
        prediction = probe.predict(values[test_ids].reshape(-1, values.shape[-1])).reshape(
            len(test_ids), 64, 3
        )
        for c, concept in enumerate(CONCEPTS):
            probe_rows.append(
                {
                    "layer": layer,
                    "concept": concept,
                    "patient_mean_r2": float(
                        r2_score(targets[test_ids, :, c].mean(1), prediction[:, :, c].mean(1))
                    ),
                }
            )
    save_csv(OUT / "probes.csv", probe_rows)
    # GCExplainer-inspired global surrogate, using no diagnostic test labels for fitting.
    kmeans = KMeans(n_clusters=5, random_state=11, n_init=10).fit(
        representations[2][train_ids].reshape(-1, 48)
    )
    clusters = [kmeans.predict(v) for v in representations[2]]
    hist = np.array([np.bincount(v, minlength=5) / 64 for v in clusters])
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=11).fit(
        hist[train_ids], probs[train_ids] >= 0.5
    )
    imitation = tree.predict(hist[test_ids])
    gc = {
        "adaptation": True,
        "test_agreement_with_gcn": float(np.mean(imitation == (probs[test_ids] >= 0.5))),
        "constant_train_majority_agreement": float(
            np.mean((probs[test_ids] >= 0.5) == (np.mean(probs[train_ids] >= 0.5) >= 0.5))
        ),
        "patients": len(test_ids),
        "clinical_concept_purity": "not measured",
        "human_review": "not performed",
    }
    save_csv(
        OUT / "gcexplainer_predictions.csv",
        [
            {
                "patient": rows[i]["patient_id"],
                "gcn_probability": probs[i],
                "surrogate_class": int(pred),
                **{f"cluster_{j}": float(hist[i, j]) for j in range(5)},
            }
            for i, pred in zip(test_ids, imitation, strict=True)
        ],
    )
    np.savez_compressed(
        OUT / "gcexplainer_clusters.npz",
        centers=kmeans.cluster_centers_,
        assignments=np.array(clusters),
    )
    deletions, edits = [], []
    rng = np.random.default_rng(2026)
    lower = representations[0][train_ids].min((0, 1))
    upper = representations[0][train_ids].max((0, 1))
    for i in test_ids:
        g, pid = graphs[i], rows[i]["patient_id"]
        baseline = probs[i]
        sign = 1 if baseline >= 0.5 else -1
        for seed in protocol["seeds"]:
            torch.manual_seed(seed)
            explainer = Explainer(
                model=wrapper,
                algorithm=GNNExplainer(epochs=100, lr=0.01),
                explanation_type="model",
                node_mask_type="object",
                edge_mask_type="object",
                model_config={
                    "mode": "binary_classification",
                    "task_level": "graph",
                    "return_type": "probs",
                },
            )
            explanation = explainer(g.x, g.edge_index)
            scores = explanation.node_mask.detach().flatten().numpy()
            if (
                not np.isfinite(scores).all()
                or abs(float(wrapper(g.x, g.edge_index)[0]) - baseline) > 1e-7
            ):
                raise ValueError("Invalid mask or uncleared explainer state")
            np.savez_compressed(
                OUT / f"{pid}_{seed}_masks.npz",
                node=scores,
                edge=explanation.edge_mask.detach().numpy(),
                edge_index=g.edge_index.numpy(),
            )
            order = np.argsort(-scores, kind="stable")
            for fraction in [0.05, 0.1, 0.2]:
                count = int(np.ceil(64 * fraction))
                for repeat in range(-1, 20):
                    selected = (
                        order[:count] if repeat == -1 else rng.choice(64, count, replace=False)
                    )
                    changed = g.x.clone()
                    changed[selected] = 0
                    after = float(wrapper(changed, g.edge_index)[0])
                    deletions.append(
                        {
                            "patient": pid,
                            "seed": seed,
                            "fraction": fraction,
                            "method": "gnnexplainer" if repeat == -1 else "random",
                            "repeat": repeat,
                            "predicted_class_drop": sign * (baseline - after),
                        }
                    )
        before_probe = probes[0].predict(g.x.numpy()).mean(0)
        for c, concept in enumerate(CONCEPTS):
            direction = constrained_direction(probes[0].coef_, c)
            for polarity in [-1, 1]:
                for repeat in range(-1, 20):
                    delta = direction.copy() if repeat == -1 else rng.normal(size=576)
                    delta /= max(np.linalg.norm(delta), 1e-12)
                    changed = g.x + torch.tensor(polarity * delta, dtype=g.x.dtype)
                    after_probe = probes[0].predict(changed.numpy()).mean(0)
                    shifts = after_probe - before_probe
                    collateral = float(np.max(np.abs(np.delete(shifts, c))))
                    support = float(
                        np.mean((changed.numpy() >= lower) & (changed.numpy() <= upper))
                    )
                    h = activations(models, changed, g.edge_index)[1].numpy()
                    layer2_shift = probes[2].predict(h).mean(0) - probes[2].predict(
                        hs[i][1].numpy()
                    ).mean(0)
                    edits.append(
                        {
                            "patient": pid,
                            "concept": concept,
                            "polarity": polarity,
                            "method": "gcia_proxy" if repeat == -1 else "random",
                            "repeat": repeat,
                            "probe_target_change_sd": float(shifts[c]),
                            "max_collateral_sd": collateral,
                            "layer2_target_change_sd": float(layer2_shift[c]),
                            "probability_IC_change": float(wrapper(changed, g.edge_index)[0])
                            - baseline,
                            "feature_support_fraction": support,
                            "numerical_gate": bool(
                                polarity * shifts[c] >= 0.1
                                and collateral <= 0.05
                                and support >= 0.99
                            ),
                            "clinical_validity": False,
                        }
                    )
        print(f"Three-method exploratory audit completed: patient {pid}", flush=True)
    if not all(torch.equal(original[k], v) for k, v in wrapper.state_dict().items()):
        raise ValueError("Frozen weights changed")
    save_csv(OUT / "deletion.csv", deletions)
    save_csv(OUT / "gcia_proxy_interventions.csv", edits)
    summary = {
        "protocol": protocol,
        "checkpoint_sha256": provenance,
        "weights_unchanged": True,
        "gnnexplainer": {},
        "gcexplainer_adaptation": gc,
        "gcia_proxy": {},
    }
    for fraction in [0.05, 0.1, 0.2]:
        differences, method_means = [], {"gnnexplainer": [], "random": []}
        for i in test_ids:
            pid = rows[i]["patient_id"]
            values = {
                method: np.mean(
                    [
                        r["predicted_class_drop"]
                        for r in deletions
                        if r["patient"] == pid
                        and r["fraction"] == fraction
                        and r["method"] == method
                    ]
                )
                for method in method_means
            }
            differences.append(values["gnnexplainer"] - values["random"])
            for method, collected in method_means.items():
                collected.append(values[method])
        summary["gnnexplainer"][str(fraction)] = {
            **{k: float(np.mean(v)) for k, v in method_means.items()},
            "paired_difference": paired_interval(differences),
        }
    for concept in CONCEPTS:
        subset = [r for r in edits if r["concept"] == concept and r["method"] == "gcia_proxy"]
        summary["gcia_proxy"][concept] = {
            "numerical_gate_rate": float(np.mean([r["numerical_gate"] for r in subset])),
            "clinical_validity": False,
            "mean_absolute_probability_change": float(
                np.mean([abs(r["probability_IC_change"]) for r in subset])
            ),
            "layer2_patient_mean_r2": next(
                r["patient_mean_r2"]
                for r in probe_rows
                if r["concept"] == concept and r["layer"] == 2
            ),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

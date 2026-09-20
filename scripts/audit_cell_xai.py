"""Frozen cellular GNN explanations and a bounded morphology intervention audit."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.gnn import GraphClassifier

OUT = ROOT / "reports/cell_xai"


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, edge_index):
        batch = torch.zeros(len(x), dtype=torch.long, device=x.device)
        return torch.stack([m(x, edge_index, batch).softmax(-1)[:, 1] for m in self.models]).mean(0)

    def representation(self, x, edges):
        batch = torch.zeros(len(x), dtype=torch.long, device=x.device)
        return torch.cat([m.encode(x, edges, batch) for m in self.models], dim=1)


def enlarge_nuclei(x, factor):
    if factor <= 0:
        raise ValueError("Area multiplier must be positive")
    result = x.clone()
    result[:, 0] *= factor
    result[:, 3:5] *= np.sqrt(factor)
    return result


def save_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values):
    a = np.asarray(values)
    rng = np.random.default_rng(2026)
    means = a[rng.integers(len(a), size=(2000, len(a)))].mean(1)
    return {
        "mean": float(a.mean()),
        "patient_bootstrap_95": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "reports/cell_cohort"
    manifest = source / "manifest.csv"
    rows = list(csv.DictReader(manifest.open()))
    if len({r["patient_id"] for r in rows}) != len(rows):
        raise ValueError("Patient overlap")
    protocol = {
        "architectures": ["GCN", "GraphSAGE", "GATv2"],
        "model_seeds": [11, 23, 37],
        "explanation_seeds": [11, 23, 37],
        "gnnexplainer_epochs": 100,
        "budgets": [0.05, 0.1, 0.2],
        "random_controls": 20,
        "scope": "exploratory audit on previously examined internal test; not external validation",
        "gnnexplainer": "official PyG joint node/edge masks, evaluate node feature replacement with train mean; exact ensemble probabilities",
        "gcia": "mean nuclear area, eccentricity, solidity probes from frozen hidden graph representations; training patients only",
        "intervention": "all predicted nuclei area multiplied by 0.95 or 1.05; both axes multiplied by sqrt(factor); other features and graph unchanged",
        "controls": "20 random isotropic changes with matched per-node norm in standardized feature space; numerical sensitivity controls, not histologically plausible edits",
        "limitations": "predicted morphology, unknown physical scale; no pixel/contour edits; no spatial overlap check; no expert validity; area and axes change jointly",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with (OUT / "protocol.json").open("x") as f:
        json.dump(protocol, f, indent=2)
    graphs = []
    for row in rows:
        path = source / row["graph_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Graph checksum mismatch")
        graphs.append(Data(**torch.load(path, weights_only=True)))
    train = [i for i, r in enumerate(rows) if r["split"] == "train"]
    test = [i for i, r in enumerate(rows) if r["split"] == "test"]
    historical = list(csv.DictReader((ROOT / "reports/cell_gnn_comparison/predictions.csv").open()))
    reference = {(r["architecture"], r["patient_id"]): float(r["probability"]) for r in historical}
    all_deletions, all_edits, probes_rows, summary = [], [], [], {}
    for architecture in protocol["architectures"]:
        models, hashes = [], {}
        for seed in protocol["model_seeds"]:
            path = ROOT / f"reports/cell_gnn_comparison/{architecture}_{seed}.pt"
            state = torch.load(path, weights_only=True)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            model = GraphClassifier(architecture, 8, 16, 2)
            model.load_state_dict(state["weights"])
            models.append(model.eval().requires_grad_(False))
            if seed == 11:
                mean, scale = state["mean"], state["scale"]
            elif not torch.equal(mean, state["mean"]) or not torch.equal(scale, state["scale"]):
                raise ValueError("Ensemble normalization mismatch")
        wrapper = Ensemble(models).eval()
        frozen = {k: v.clone() for k, v in wrapper.state_dict().items()}
        xs = [(g.x - mean) / scale for g in graphs]
        h = np.concatenate(
            [
                wrapper.representation(x, g.edge_index).numpy()
                for x, g in zip(xs, graphs, strict=True)
            ]
        )
        traits = np.array([g.x[:, [0, 1, 2]].mean(0).numpy() for g in graphs])
        tm, ts = traits[train].mean(0), traits[train].std(0).clip(1e-6)
        hm, hs = h[train].mean(0), h[train].std(0).clip(1e-6)
        probe = Ridge(alpha=10).fit((h[train] - hm) / hs, (traits[train] - tm) / ts)
        predicted = probe.predict((h[test] - hm) / hs)
        r2 = r2_score((traits[test] - tm) / ts, predicted, multioutput="raw_values")
        np.savez_compressed(
            OUT / f"{architecture}_probe.npz",
            coef=probe.coef_,
            intercept=probe.intercept_,
            hidden_mean=hm,
            hidden_scale=hs,
            target_mean=tm,
            target_scale=ts,
            train_patients=np.array([rows[i]["patient_id"] for i in train]),
        )
        for c, name in enumerate(["area", "eccentricity", "solidity"]):
            probes_rows.append(
                {"architecture": architecture, "concept": name, "test_patient_r2": float(r2[c])}
            )
        lower = torch.cat([xs[i] for i in train]).min(0).values
        upper = torch.cat([xs[i] for i in train]).max(0).values
        rng = np.random.default_rng(2026)
        for i in test:
            g, x, pid = graphs[i], xs[i], rows[i]["patient_id"]
            baseline = float(wrapper(x, g.edge_index)[0])
            if abs(baseline - reference[(architecture, pid)]) > 1e-5:
                raise ValueError("Classifier predictions differ from saved results")
            direction = 1 if baseline >= 0.5 else -1
            for seed in protocol["explanation_seeds"]:
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
                explanation = explainer(x, g.edge_index)
                scores = explanation.node_mask.detach().flatten().numpy()
                if (
                    not np.isfinite(scores).all()
                    or abs(float(wrapper(x, g.edge_index)[0]) - baseline) > 1e-7
                ):
                    raise ValueError("Invalid explanation or mask cleanup failure")
                np.savez_compressed(
                    OUT / f"{architecture}_{pid}_{seed}_masks.npz",
                    node=scores,
                    edge=explanation.edge_mask.detach().numpy(),
                    edge_index=g.edge_index.numpy(),
                )
                order = np.argsort(-scores, kind="stable")
                for fraction in protocol["budgets"]:
                    count = max(1, int(np.ceil(len(x) * fraction)))
                    for repeat in range(-1, 20):
                        selected = (
                            order[:count]
                            if repeat == -1
                            else rng.choice(len(x), count, replace=False)
                        )
                        edited = x.clone()
                        edited[selected] = 0
                        drop = direction * (baseline - float(wrapper(edited, g.edge_index)[0]))
                        all_deletions.append(
                            {
                                "architecture": architecture,
                                "patient": pid,
                                "seed": seed,
                                "fraction": fraction,
                                "method": "gnnexplainer" if repeat == -1 else "random",
                                "repeat": repeat,
                                "predicted_class_probability_drop": drop,
                            }
                        )
            before_probe = probe.predict((h[i : i + 1] - hm) / hs)[0]
            for factor in [0.95, 1.05]:
                target = (enlarge_nuclei(g.x, factor) - mean) / scale
                norms = (target - x).norm(dim=1, keepdim=True)
                for repeat in range(-1, 20):
                    if repeat == -1:
                        edited = target
                    else:
                        delta = torch.tensor(rng.normal(size=x.shape), dtype=x.dtype)
                        edited = (
                            x + delta / delta.norm(dim=1, keepdim=True).clamp_min(1e-12) * norms
                        )
                    after_h = wrapper.representation(edited, g.edge_index).numpy()
                    shift = probe.predict((after_h - hm) / hs)[0] - before_probe
                    all_edits.append(
                        {
                            "architecture": architecture,
                            "patient": pid,
                            "area_factor": factor,
                            "method": "gcia_morphology" if repeat == -1 else "random",
                            "repeat": repeat,
                            "probability_IC_change": float(wrapper(edited, g.edge_index)[0])
                            - baseline,
                            "area_probe_shift_sd": float(shift[0]),
                            "eccentricity_probe_shift_sd": float(shift[1]),
                            "solidity_probe_shift_sd": float(shift[2]),
                            "feature_support_fraction": float(
                                ((edited >= lower) & (edited <= upper)).float().mean()
                            ),
                            "clinical_validity": False,
                        }
                    )
            print(f"{architecture}: patient {pid} explained and intervened", flush=True)
        if not all(torch.equal(frozen[k], v) for k, v in wrapper.state_dict().items()):
            raise ValueError("Classifier weights changed")
        summary[architecture] = {
            "checkpoint_sha256": hashes,
            "weights_unchanged": True,
            "probe_r2": dict(zip(["area", "eccentricity", "solidity"], r2.tolist(), strict=True)),
            "gnnexplainer": {},
            "gcia": {},
        }
        for fraction in protocol["budgets"]:
            per_patient = []
            means = {"gnnexplainer": [], "random": []}
            for i in test:
                for method, values in means.items():
                    values.append(
                        np.mean(
                            [
                                r["predicted_class_probability_drop"]
                                for r in all_deletions
                                if r["architecture"] == architecture
                                and r["patient"] == rows[i]["patient_id"]
                                and r["method"] == method
                                and r["fraction"] == fraction
                            ]
                        )
                    )
                per_patient.append(means["gnnexplainer"][-1] - means["random"][-1])
            summary[architecture]["gnnexplainer"][str(fraction)] = {
                **{k: float(np.mean(v)) for k, v in means.items()},
                "difference": interval(per_patient),
            }
        for factor in [0.95, 1.05]:
            values = {
                method: [
                    r
                    for r in all_edits
                    if r["architecture"] == architecture
                    and r["area_factor"] == factor
                    and r["method"] == method
                ]
                for method in ["gcia_morphology", "random"]
            }
            summary[architecture]["gcia"][str(factor)] = {
                method: float(np.mean([abs(r["probability_IC_change"]) for r in selected]))
                for method, selected in values.items()
            }
            summary[architecture]["gcia"][str(factor)]["probe_direction_success_rate"] = float(
                np.mean(
                    [(factor - 1) * r["area_probe_shift_sd"] > 0 for r in values["gcia_morphology"]]
                )
            )
        save_csv("deletions.csv", all_deletions)
        save_csv("interventions.csv", all_edits)
        save_csv("probes.csv", probes_rows)
        (OUT / "summary_partial.json").write_text(json.dumps(summary, indent=2))
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "patients": len(test),
                "explanations": len(test) * 9,
                "results": summary,
                "protocol": protocol,
            },
            indent=2,
        )
    )
    print("All cellular XAI audits complete", flush=True)


if __name__ == "__main__":
    main()

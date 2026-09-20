"""Annotation-grounded exploratory audit of a frozen BRACS classifier on BCSS."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import Ridge
from tiatoolbox.models.architecture.vanilla import CNNModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_bracs_three_xai import BinaryModel, Ensemble, activations
from scripts.run_extended_xai import save_csv
from scripts.run_focused_study import load_patients
from src.graphs.build import make_graph
from src.interventions.graph_edits import drop_undirected_edge

OUT = ROOT / "reports/annotated_gcia"
NAMES = ["tumor", "stroma", "lymphocytic_infiltrate"]


def encode(encoder, patches):
    result = []
    with torch.no_grad():
        for start in range(0, len(patches), 32):
            x = torch.from_numpy(np.stack(patches[start : start + 32])).permute(0, 3, 1, 2)
            result.append(encoder.pool(encoder.feat_extract(x)).flatten(1).numpy())
    return np.concatenate(result)


def components(edge_index, nodes):
    neighbours = [set() for _ in range(nodes)]
    for a, b in edge_index.t().tolist():
        neighbours[a].add(b)
    visited, count = set(), 0
    for root in range(nodes):
        if root in visited:
            continue
        count += 1
        stack = [root]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(neighbours[node] - visited)
    return count


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "scope": "exploratory frozen BRACS classifier audit on eight previously explored annotated BCSS patients; not external diagnostic validation",
        "concept_codes": {"tumor": 1, "stroma": 2, "lymphocytic_infiltrate": 3},
        "probes": "ridge alpha10; leave-one-patient-out; inverse patch-count patient weighting; input/layer1/layer2; constant comparator from training patients",
        "patch_edits": "up to five >=80% tumor recipients; nearest RGB-mean same-patient >=80% stroma donor; same-patient >=80% tumor control distinct from recipient; re-encode actual substituted RGB patch",
        "graph_edits": "remove up to five tumor-stroma edges with both endpoints >=80% pure; 20 random matched-count and matched-geometric-length deletion controls; record connectivity; node attributes unchanged",
        "validity": "joint tumor-to-stroma substitution, not isolated concept manipulation; no human plausibility approval; diagnostic performance on BCSS unknown",
        "downloads": 0,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with (OUT / "protocol.json").open("x") as f:
        json.dump(protocol, f, indent=2)
    cfg = json.loads((ROOT / "configs/encoder_protocol.json").read_text())
    patients = load_patients(cfg, ROOT / "data/bcss_encoder_manifest.csv")
    weights = ROOT / "data/pretrained/mobilenet_v3_small-pcam.pth"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != cfg["weight_sha256"]:
        raise ValueError("Encoder checksum mismatch")
    encoder = CNNModel("mobilenet_v3_small", num_classes=2)
    encoder.load_state_dict(torch.load(weights, weights_only=True))
    encoder.eval().requires_grad_(False)
    models, sources = [], {}
    for seed in (11, 23, 37):
        path = ROOT / f"reports/bracs_expanded/gcn_{seed}.pt"
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        state = torch.load(path, weights_only=True)
        model = BinaryModel("gcn", 576)
        model.load_state_dict(state["weights"])
        models.append(model.eval().requires_grad_(False))
    wrapper = Ensemble(models).eval()
    frozen = {k: v.clone() for k, v in wrapper.state_dict().items()}
    mean, scale = state["mean"], state["scale"]
    for patient in patients:
        features = encode(encoder, patient["patches"])
        g = make_graph(((torch.from_numpy(features) - mean) / scale).numpy(), patient["coords"], 0)
        patient["graph"] = g
        patient["representations"] = [g.x.numpy()] + [
            h.numpy() for h in activations(models, g.x, g.edge_index)
        ]
        patient["probability"] = float(wrapper(g.x, g.edge_index)[0])
    probe_rows, edits, graph_rows, coverage = [], [], [], []
    rng = np.random.default_rng(2026)
    for index, patient in enumerate(patients):
        pid, g, valid = patient["patient"], patient["graph"], patient["valid"]
        train = [p for j, p in enumerate(patients) if j != index]
        target = patient["fractions"][:, 1:4]
        y = np.concatenate([p["fractions"][p["valid"], 1:4] for p in train])
        weight = np.concatenate([np.full(sum(p["valid"]), 1 / sum(p["valid"])) for p in train])
        weight *= len(weight) / weight.sum()
        constant = np.mean([p["fractions"][p["valid"], 1:4].mean(0) for p in train], axis=0)
        probes = []
        for layer in range(3):
            x = np.concatenate([p["representations"][layer][p["valid"]] for p in train])
            probe = Ridge(alpha=10).fit(x, y, sample_weight=weight)
            probes.append(probe)
            prediction = probe.predict(patient["representations"][layer])
            for c, name in enumerate(NAMES):
                probe_rows.append(
                    {
                        "patient": pid,
                        "layer": layer,
                        "concept": name,
                        "patch_mse": float(np.mean((prediction[valid, c] - target[valid, c]) ** 2)),
                        "constant_mse": float(np.mean((constant[c] - target[valid, c]) ** 2)),
                        "observed_mean": float(target[valid, c].mean()),
                        "predicted_mean": float(prediction[valid, c].mean()),
                    }
                )
            np.savez_compressed(
                OUT / f"{pid}_probe_layer{layer}.npz",
                coef=probe.coef_,
                intercept=probe.intercept_,
                train_patients=np.array([p["patient"] for p in train]),
            )
        tumor = np.flatnonzero(valid & (target[:, 0] >= 0.8))
        stroma = np.flatnonzero(valid & (target[:, 1] >= 0.8))
        coverage.append(
            {
                "patient": pid,
                "valid_patches": int(sum(valid)),
                "tumor_patches": len(tumor),
                "stroma_patches": len(stroma),
                "probability_IC_out_of_domain": patient["probability"],
            }
        )
        rgb_means = np.array([p.mean((0, 1)) for p in patient["patches"]])
        before_probe = probes[2].predict(patient["representations"][2])[valid].mean(0)
        if len(tumor) >= 2 and len(stroma):
            for recipient in tumor[:5]:
                donors = {"target": stroma, "control": tumor[tumor != recipient]}
                for arm, candidates in donors.items():
                    donor = int(
                        candidates[
                            np.argmin(
                                np.sum((rgb_means[candidates] - rgb_means[recipient]) ** 2, axis=1)
                            )
                        ]
                    )
                    altered = g.x.clone()
                    # Actual donor pixels are encoded again, not an arbitrary feature shift.
                    donor_feature = encode(encoder, [patient["patches"][donor]])[0]
                    altered[recipient] = (torch.from_numpy(donor_feature) - mean) / scale
                    error = float((altered[recipient] - g.x[donor]).abs().max())
                    if error > 1e-3:
                        raise ValueError("Pixel/embedding substitution mismatch")
                    after_h = activations(models, altered, g.edge_index)[1].numpy()
                    shift = probes[2].predict(after_h)[valid].mean(0) - before_probe
                    observed_shift = (target[donor] - target[recipient]) / sum(valid)
                    row = {
                        "patient": pid,
                        "recipient": int(recipient),
                        "donor": donor,
                        "arm": arm,
                        "probability_IC_change": float(wrapper(altered, g.edge_index)[0])
                        - patient["probability"],
                        "reencoding_max_error": error,
                        "histological_realism_validated": False,
                    }
                    for c, name in enumerate(NAMES):
                        row[f"observed_{name}_change"] = float(observed_shift[c])
                        row[f"probe_{name}_change"] = float(shift[c])
                    edits.append(row)
                    if recipient == tumor[0]:
                        canvas = np.concatenate(
                            [patient["patches"][recipient], patient["patches"][donor]], axis=1
                        )
                        Image.fromarray(np.uint8(np.clip(canvas * 255, 0, 255))).save(
                            OUT / f"{pid}_{arm}_recipient_donor.png"
                        )
        edges = [(a, b) for a, b in g.edge_index.t().tolist() if a < b]
        target_edges = [
            (a, b) for a, b in edges if (a in tumor and b in stroma) or (b in tumor and a in stroma)
        ][:5]
        if target_edges:
            lengths = [
                float(np.sum((patient["coords"][a] - patient["coords"][b]) ** 2)) for a, b in edges
            ]
            for repeat in range(-1, 20):
                selected = target_edges if repeat == -1 else []
                if repeat != -1:
                    for a, b in target_edges:
                        length = float(np.sum((patient["coords"][a] - patient["coords"][b]) ** 2))
                        options = [
                            edge
                            for edge, d in zip(edges, lengths, strict=True)
                            if np.isclose(d, length) and edge not in selected
                        ]
                        selected.append(options[int(rng.integers(len(options)))])
                changed_edges = g.edge_index.clone()
                for a, b in selected:
                    changed_edges = drop_undirected_edge(changed_edges, a, b)
                after_h = activations(models, g.x, changed_edges)[1].numpy()
                shifts = probes[2].predict(after_h)[valid].mean(0) - before_probe
                graph_rows.append(
                    {
                        "patient": pid,
                        "arm": "target" if repeat == -1 else "random",
                        "repeat": repeat,
                        "edges_deleted": len(selected),
                        "components_before": components(g.edge_index, len(g.x)),
                        "components_after": components(changed_edges, len(g.x)),
                        "probability_IC_change": float(wrapper(g.x, changed_edges)[0])
                        - patient["probability"],
                        **{
                            f"probe_{name}_change": float(shifts[c]) for c, name in enumerate(NAMES)
                        },
                        "observed_tissue_fractions_change": 0,
                        "histological_realism_validated": False,
                    }
                )
        print(f"Annotated audit completed: {pid}", flush=True)
    if not all(torch.equal(frozen[k], v) for k, v in wrapper.state_dict().items()):
        raise ValueError("Classifier weights changed")
    save_csv(OUT / "probes.csv", probe_rows)
    save_csv(OUT / "coverage.csv", coverage)
    if edits:
        save_csv(OUT / "patch_interventions.csv", edits)
    if graph_rows:
        save_csv(OUT / "graph_interventions.csv", graph_rows)
    summary = {
        "patients": len(patients),
        "patch_intervention_rows": len(edits),
        "graph_intervention_rows": len(graph_rows),
        "weights_unchanged": True,
        "sources": sources,
        "protocol": protocol,
        "layer2_probes": {},
    }
    for name in NAMES:
        selected = [r for r in probe_rows if r["layer"] == 2 and r["concept"] == name]
        summary["layer2_probes"][name] = {
            key: float(np.mean([r[key] for r in selected])) for key in ["patch_mse", "constant_mse"]
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

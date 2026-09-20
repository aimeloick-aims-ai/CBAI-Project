"""Supplementary structural audit of frozen patch clusters, not typed cells."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_candidate_clusters import degree_matched
from scripts.run_bracs_binary import BinaryModel
from scripts.run_bracs_three_xai import Ensemble, activations, paired_interval
from scripts.run_extended_xai import save_csv
from src.graphs.build import knn_edges
from src.graphs.interventions import (
    edge_strata,
    matched_edge_indices,
    paired_edges,
    remove_nodes,
    unique_pairs,
)


def main():
    torch.set_num_threads(2)
    out = ROOT / "reports/cluster_structure_audit"
    out.mkdir(exist_ok=False)
    source = ROOT / "reports/bracs_three_xai"
    files = [
        ROOT / "data/bracs_expanded_study_manifest.csv",
        source / "representations.npz",
        source / "gcexplainer_clusters.npz",
    ]
    files += [ROOT / f"reports/bracs_expanded/gcn_{seed}.pt" for seed in (11, 23, 37)]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    protocol = {
        "scope": "supplementary exploratory patch-cluster analysis after inspecting feature occlusion; unvalidated AI labels; already examined test",
        "node_removal": "all nodes originally assigned to cluster; induced surviving graph; reject empty graph; controls exact count and original degree histogram",
        "boundary_removal": "all undirected edges with exactly one endpoint in target cluster; symmetric removal; controls exact count, squared pixel-grid length and unordered original endpoint-degree pair",
        "controls": 20,
        "control_overlap_allowed": True,
        "seed": 2026,
        "limitations": "controls do not match connectedness, spatial contiguity or post-removal degree sequence; no cell typing or biological plausibility established",
        "outcome": "baseline predicted-class probability decrease; paired patient bootstrap2000 unadjusted",
        "hashes": hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    assert (
        hashes[str(files[0].relative_to(ROOT))]
        == json.loads((source / "started.json").read_text())["manifest_sha256"]
    )
    rows = list(csv.DictReader(files[0].open()))
    cache = np.load(files[1])
    groups = np.load(files[2])
    assert cache["patients"].tolist() == [r["patient_id"] for r in rows]
    assert len(set(cache["patients"])) == len(rows)
    models = []
    for file in files[3:]:
        model = BinaryModel("gcn", 576)
        model.load_state_dict(torch.load(file, weights_only=True)["weights"])
        models.append(model.eval().requires_grad_(False))
    model = Ensemble(models).eval()
    coordinates = np.array([[x, y] for y in range(8) for x in range(8)])
    edges = knn_edges(coordinates)
    degrees = np.bincount(edges[0].numpy(), minlength=64)
    pairs = unique_pairs(edges)
    assert torch.equal(paired_edges(pairs).sort(dim=1).values, edges.sort(dim=1).values)
    strata = edge_strata(pairs, coordinates, degrees)
    old = {
        r["patient"]: float(r["gcn_probability"])
        for r in csv.DictReader((source / "gcexplainer_predictions.csv").open())
    }
    records, exclusions = [], []
    rng = np.random.default_rng(2026)
    with torch.no_grad():
        for i, row in enumerate(rows):
            if row["proposed_split"] != "internal_test":
                continue
            pid = row["patient_id"]
            x = torch.from_numpy(cache["x"][i])
            base = float(model(x, edges)[0])
            assert abs(base - old[pid]) < 1e-5
            hidden = activations(models, x, edges)[1].numpy()
            assigned = np.linalg.norm(hidden[:, None] - groups["centers"][None], axis=2).argmin(1)
            assert np.array_equal(assigned, groups["assignments"][i])
            for c in range(5):
                members = assigned == c
                for mode in ("node_removal", "boundary_removal"):
                    target = (
                        np.flatnonzero(members)
                        if mode == "node_removal"
                        else np.flatnonzero(members[pairs[:, 0]] != members[pairs[:, 1]])
                    )
                    if not len(target) or (mode == "node_removal" and len(target) == len(x)):
                        exclusions.append(
                            {
                                "patient": pid,
                                "cluster": c,
                                "mode": mode,
                                "reason": "no_target_or_empty_graph",
                            }
                        )
                        continue
                    for repeat in range(21):
                        selected = target
                        if repeat:
                            selected = (
                                degree_matched(target, degrees, rng)
                                if mode == "node_removal"
                                else matched_edge_indices(target, strata, rng)
                            )
                        assert len(selected) == len(set(selected)) == len(target)
                        if mode == "node_removal":
                            assert np.array_equal(
                                np.sort(degrees[selected]), np.sort(degrees[target])
                            )
                            changed, new_edges = remove_nodes(x, edges, selected)
                        else:
                            assert sorted(strata[j] for j in selected) == sorted(
                                strata[j] for j in target
                            )
                            retained = np.ones(len(pairs), dtype=bool)
                            retained[selected] = False
                            changed, new_edges = x, paired_edges(pairs[retained])
                        prob = float(model(changed, new_edges)[0])
                        records.append(
                            {
                                "patient": pid,
                                "cluster": c,
                                "mode": mode,
                                "kind": "target" if repeat == 0 else "random",
                                "repeat": repeat,
                                "removed": len(selected),
                                "selected_indices": json.dumps(selected.tolist()),
                                "remaining_nodes": len(changed),
                                "remaining_directed_edges": new_edges.shape[1],
                                "baseline_p_IC": base,
                                "perturbed_p_IC": prob,
                                "drop": (1 if base >= 0.5 else -1) * (base - prob),
                            }
                        )
            print(f"Structural audit: {pid}", flush=True)
    save_csv(out / "interventions.csv", records)
    save_csv(out / "exclusions.csv", exclusions)
    summary = []
    for mode in ("node_removal", "boundary_removal"):
        for c in range(5):
            targeted = [
                r
                for r in records
                if r["mode"] == mode and r["cluster"] == c and r["kind"] == "target"
            ]
            a, b = [], []
            for row in targeted:
                controls = [
                    r["drop"]
                    for r in records
                    if r["mode"] == mode
                    and r["cluster"] == c
                    and r["patient"] == row["patient"]
                    and r["kind"] == "random"
                ]
                assert len(controls) == 20
                a.append(row["drop"])
                b.append(np.mean(controls))
            summary.append(
                {
                    "mode": mode,
                    "cluster": c,
                    "patients": len(a),
                    "target": float(np.mean(a)) if a else None,
                    "control": float(np.mean(b)) if b else None,
                    "difference": paired_interval(np.array(a) - b) if a else None,
                }
            )
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Audit structurel exploratoire des clusters de patches",
        "",
        "Trois GCN gelés, 22 patients internes déjà examinés. Concepts nommés par IA, non validés. Aucun type cellulaire n'est attribué.",
        "",
        "Suppression de nœuds : retrait du groupe et de ses arêtes incidentes. Suppression de frontière : retrait des arêtes reliant le groupe aux autres ; les nœuds restent identiques. Les témoins sont appariés selon protocol.json. Aucune intervention ne garantit la plausibilité biologique.",
        "",
        "Différences en points de probabilité de la classe initialement prédite. IC bootstrap patients descriptifs, sans correction des dix comparaisons. Effectifs variables : les lignes ne forment pas un classement.",
        "",
        "| Intervention | Cluster | Patients | Ciblé | Témoin | Différence [IC 95 %] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mode in zip(axes, ("node_removal", "boundary_removal"), strict=True):
        for s in summary:
            if s["mode"] != mode or not s["patients"]:
                continue
            lo, hi = np.array(s["difference"]["patient_bootstrap_95"]) * 100
            avg = s["difference"]["mean"] * 100
            lines.append(
                f"| {mode} | {s['cluster']} | {s['patients']} | {100 * s['target']:.2f} | {100 * s['control']:.2f} | {avg:.2f} [{lo:.2f}, {hi:.2f}] |"
            )
            ax.plot([lo, hi], [s["cluster"]] * 2, color="steelblue")
            ax.plot(avg, s["cluster"], "o", color="steelblue")
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_title(mode)
        ax.set_yticks(range(5))
        ax.set_xlabel("Target minus control (pp)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"effects.{ext}", dpi=180)
    plt.close(fig)
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Completed structural audit", flush=True)


if __name__ == "__main__":
    main()

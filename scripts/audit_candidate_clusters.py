"""Exploratory frozen patch-GCN cluster occlusion; no clinical validation."""

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
from scripts.run_bracs_binary import BinaryModel
from scripts.run_bracs_three_xai import Ensemble, activations, paired_interval
from scripts.run_extended_xai import save_csv
from src.graphs.build import knn_edges


def degree_matched(indices, degrees, rng):
    """Sample without replacement with exactly the target degree histogram."""
    return np.concatenate(
        [
            rng.choice(
                np.flatnonzero(degrees == degree),
                int(sum(degrees[indices] == degree)),
                replace=False,
            )
            for degree in np.unique(degrees[indices])
        ]
    )


def main():
    torch.set_num_threads(2)
    out = ROOT / "reports/candidate_cluster_audit"
    out.mkdir(exist_ok=False)
    source = ROOT / "reports/bracs_three_xai"
    paths = [
        ROOT / "data/bracs_expanded_study_manifest.csv",
        source / "representations.npz",
        source / "gcexplainer_clusters.npz",
        ROOT / "reports/cluster_review/CONCEPTS_PROVISOIRES.json",
    ]
    checkpoints = [ROOT / f"reports/bracs_expanded/gcn_{seed}.pt" for seed in (11, 23, 37)]
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths + checkpoints
    }
    protocol = {
        "scope": "exploratory already-examined internal test; patch graphs only; AI names unvalidated",
        "intervention": "replace every node assigned to target cluster with training mean input; edges unchanged",
        "controls": "20 random subsets per patient/cluster, exact same count and degree histogram; overlap allowed; norm and spatial contiguity not matched",
        "outcome": "decrease in probability of baseline predicted class; paired patient bootstrap 2000; descriptive unadjusted intervals",
        "absent_clusters": "record presence=0; exclude from conditional effect estimate",
        "seed": 2026,
        "hashes": hashes,
        "clinical_validation": False,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows = list(csv.DictReader(paths[0].open(newline="")))
    assert (
        hashlib.sha256(paths[0].read_bytes()).hexdigest()
        == json.loads((source / "started.json").read_text())["manifest_sha256"]
    )
    assert len({r["patient_id"] for r in rows}) == len(rows)
    cache = np.load(paths[1])
    assert list(cache["patients"]) == [r["patient_id"] for r in rows]
    groups = np.load(paths[2])
    centers, assignments = groups["centers"], groups["assignments"]
    registry = json.loads(paths[3].read_text(encoding="utf-8"))["concepts"]
    train = [i for i, r in enumerate(rows) if r["proposed_split"] == "train"]
    test = [i for i, r in enumerate(rows) if r["proposed_split"] == "internal_test"]
    mean = torch.from_numpy(cache["x"][train].mean((0, 1)))
    models = []
    for path in checkpoints:
        model = BinaryModel("gcn", 576)
        model.load_state_dict(torch.load(path, weights_only=True)["weights"])
        models.append(model.eval().requires_grad_(False))
    wrapper = Ensemble(models).eval()
    edges = knn_edges(np.array([[x, y] for y in range(8) for x in range(8)]))
    degrees = np.bincount(edges[0].numpy(), minlength=64)
    old = {
        r["patient"]: float(r["gcn_probability"])
        for r in csv.DictReader((source / "gcexplainer_predictions.csv").open())
    }
    rng = np.random.default_rng(2026)
    records, presence = [], []
    with torch.no_grad():
        for i in test:
            pid = rows[i]["patient_id"]
            x = torch.from_numpy(cache["x"][i])
            baseline = float(wrapper(x, edges)[0])
            assert abs(baseline - old[pid]) < 1e-5
            hidden = activations(models, x, edges)[1].numpy()
            np.testing.assert_allclose(hidden, cache["layer2"][i], rtol=1e-4, atol=1e-5)
            predict_cluster = lambda h: np.linalg.norm(h[:, None] - centers[None], axis=2).argmin(1)
            assert np.array_equal(predict_cluster(hidden), assignments[i])
            original_hist = np.bincount(assignments[i], minlength=5) / 64
            for c in range(5):
                target = np.flatnonzero(assignments[i] == c)
                presence.append({"patient": pid, "cluster": c, "nodes": len(target)})
                if not len(target):
                    continue
                for repeat in range(21):
                    chosen = target if repeat == 0 else degree_matched(target, degrees, rng)
                    assert len(chosen) == len(np.unique(chosen)) == len(target)
                    assert np.array_equal(np.sort(degrees[chosen]), np.sort(degrees[target]))
                    changed = x.clone()
                    changed[chosen] = mean
                    probability = float(wrapper(changed, edges)[0])
                    post = predict_cluster(activations(models, changed, edges)[1].numpy())
                    histogram = np.bincount(post, minlength=5) / 64
                    records.append(
                        {
                            "patient": pid,
                            "cluster": c,
                            "kind": "target" if repeat == 0 else "random",
                            "repeat": repeat,
                            "nodes": len(chosen),
                            "selected_nodes": json.dumps(chosen.tolist()),
                            "baseline_p_IC": baseline,
                            "perturbed_p_IC": probability,
                            "predicted_class_drop": (1 if baseline >= 0.5 else -1)
                            * (baseline - probability),
                            "total_l2": float(torch.linalg.vector_norm(changed - x)),
                            **{
                                f"cluster_{j}_fraction_change": float(
                                    histogram[j] - original_hist[j]
                                )
                                for j in range(5)
                            },
                        }
                    )
            print(f"Audited patient {pid}", flush=True)
    save_csv(out / "interventions.csv", records)
    save_csv(out / "presence.csv", presence)
    summary = []
    for c in range(5):
        targeted = [r for r in records if r["cluster"] == c and r["kind"] == "target"]
        pairs = []
        for row in targeted:
            controls = [
                r
                for r in records
                if r["cluster"] == c and r["patient"] == row["patient"] and r["kind"] == "random"
            ]
            pairs.append(
                (
                    row["predicted_class_drop"],
                    np.mean([r["predicted_class_drop"] for r in controls]),
                )
            )
        if not pairs:
            summary.append({"cluster": c, "patients": 0})
            continue
        values = np.array(pairs)
        summary.append(
            {
                "cluster": c,
                "name": registry[c]["label_fr"],
                "patients": len(pairs),
                "target_drop": float(values[:, 0].mean()),
                "control_drop": float(values[:, 1].mean()),
                "difference": paired_interval(values[:, 0] - values[:, 1]),
                "mean_nodes": float(np.mean([r["nodes"] for r in targeted])),
                "mean_target_cluster_fraction_change": float(
                    np.mean([r[f"cluster_{c}_fraction_change"] for r in targeted])
                ),
            }
        )
    assert all(
        hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha for p, sha in hashes.items()
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Audit exploratoire des cinq clusters candidats",
        "",
        "GCN de patches gelé, ensemble de trois initialisations, 22 patients du test déjà examiné. Noms proposés par IA, sans validation médicale.",
        "",
        "Tous les patches du cluster sont remplacés par la moyenne des embeddings d'apprentissage. Vingt témoins par patient ont le même nombre de patches et la même distribution de degrés. Les arêtes restent inchangées.",
        "",
        "Baisse de confiance dans la classe initialement prédite, en points de pourcentage. Valeur négative : augmentation de confiance. Patients sans le cluster exclus de son estimation conditionnelle.",
        "",
        "| Cluster | Patients présents / 22 | Patches moyens | Ciblé | Témoin | Différence [IC 95 %] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        if s["patients"]:
            lo, hi = s["difference"]["patient_bootstrap_95"]
            lines.append(
                f"| {s['cluster']} | {s['patients']} | {s['mean_nodes']:.1f} | {100 * s['target_drop']:.2f} | {100 * s['control_drop']:.2f} | {100 * s['difference']['mean']:.2f} [{100 * lo:.2f}, {100 * hi:.2f}] |"
            )
    lines += [
        "",
        "## Interprétation et limites",
        "",
        "Une différence positive indique une sensibilité plus forte aux patches de ce cluster qu'aux témoins. Elle ne démontre pas une dépendance au concept médical nommé. Les effets de différents clusters ne sont pas directement comparables : effectifs et tailles diffèrent.",
        "",
        "L'occlusion ne préserve ni la plausibilité histologique ni les autres concepts. Les changements des cinq fractions de clusters sont enregistrés dans interventions.csv ; ce sont des réponses du clustering, pas des annotations cliniques. Les contrôles ne sont appariés ni en norme de perturbation ni en contiguïté spatiale. Les normes sont enregistrées.",
        "",
        "IC descriptifs par bootstrap apparié sur les patients, sans correction des cinq comparaisons. Les graines des modèles ne sont pas des patients supplémentaires. Pas de nouveau probe clinique, d'édition de pixels ou d'intervention sur des cellules typées. Aucun réentraînement ni téléchargement.",
        "",
        "Fichiers : protocol.json, summary.json, presence.csv, interventions.csv et effects.png/pdf.",
    ]
    lines += [
        "",
        "## Vérification du sens de l'intervention",
        "",
        "Variation moyenne de la fraction de patches affectés au cluster ciblé après recalcul du GNN. Une hausse signifie que l'occlusion ne supprime pas le cluster dans la représentation ; elle ne peut pas être interprétée comme suppression du concept.",
        "",
    ]
    for s in summary:
        if s["patients"]:
            lines.append(
                f"- Cluster {s['cluster']} : {100 * s['mean_target_cluster_fraction_change']:+.2f} points."
            )
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in summary:
        if s["patients"]:
            avg = 100 * s["difference"]["mean"]
            lo, hi = np.array(s["difference"]["patient_bootstrap_95"]) * 100
            ax.plot([lo, hi], [s["cluster"]] * 2, color="steelblue")
            ax.plot(avg, s["cluster"], "o", color="steelblue")
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_yticks(range(5), [f"Cluster {c}" for c in range(5)])
    ax.set_xlabel("Target minus control confidence drop (percentage points)")
    ax.set_title("Exploratory patch-cluster occlusion — unvalidated concept names")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"effects.{ext}", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

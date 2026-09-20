"""Generate baseline report and a descriptive deletion curve without retraining."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "reports/gnnexplainer_baseline"
    with (out / "deletion.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], float(row["fraction"]), row["method"])].append(
            float(row["absolute_change"])
        )
    lines = [
        "# Baseline GNNExplainer executee",
        "",
        "Adaptation sur BCSS, encodeur PCam gele et GCN : regression de proportion tumorale.",
        "Huit patients, trois graines, 24 explications, 1512 evaluations de perturbation.",
        "Aucun nouveau telechargement. Aucun poids du GNN modifie.",
        "",
        "## Protocole",
        "",
        "GNNExplainer officiel PyG : 100 epochs, lr=0,01, regularisation par defaut.",
        "Masques de noeuds et aretes optimises conjointement. Evaluation du classement",
        "des noeuds seulement : remplacement par la moyenne des features du train,",
        "topologie conservee. Vingt controles aleatoires de meme taille par budget.",
        "La cible est la prediction du modele, sans usage des masques tissulaires pour",
        "optimiser les explications. Les annotations definissent les patches valides.",
        "",
        "## Resultats descriptifs",
        "",
        "Variation absolue moyenne de proportion predite, en points de pourcentage.",
        "Plus elevee indique une plus grande sensibilite a cette perturbation,",
        "pas automatiquement une meilleure explication clinique.",
        "",
        "| Ensemble | Patches remplaces | GNNExplainer | Aleatoire |",
        "|---|---:|---:|---:|",
    ]
    splits = ["development", "previously_examined_holdout"]
    fractions = [0.05, 0.1, 0.2]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, split in zip(axes, splits, strict=True):
        for fraction in fractions:
            a = 100 * np.mean(groups[(split, fraction, "gnnexplainer")])
            b = 100 * np.mean(groups[(split, fraction, "random")])
            lines.append(f"| {split} | {100 * fraction:.0f}% | {a:.3f} | {b:.3f} |")
        for method in ("gnnexplainer", "random"):
            values = [100 * np.mean(groups[(split, f, method)]) for f in fractions]
            ax.plot(np.array(fractions) * 100, values, marker="o", label=method)
        ax.set_title(
            "Six development patients"
            if split == "development"
            else "Two previously examined patients"
        )
        ax.set_xlabel("Replaced nodes (%)")
        ax.legend()
    axes[0].set_ylabel("Absolute prediction change (percentage points)")
    fig.tight_layout()
    fig.savefig(out / "deletion_curve.png", dpi=160)
    plt.close(fig)
    lines += [
        "",
        "## Interpretation",
        "",
        "GNNExplainer ne depasse pas systematiquement les controles aleatoires.",
        "A 20%, son effet moyen est superieur ; a 10%, il est inferieur dans les deux groupes.",
        "Aucun test de significativite ou gain clinique n est revendique.",
        "Les graines et controles ne constituent pas des patients independants.",
        "",
        "Les heatmaps sont presentees sur une echelle absolue 0..1 ; les valeurs proches",
        "ne doivent pas etre presentees comme des separations fortes. Le top 10% est",
        "un classement de budget fixe, pas une annotation de tumeur.",
        "",
        "## Figures et fichiers",
        "",
        "- deletion_curve.png : courbes descriptives sans intervalle de confiance.",
        "- TCGA-*.png : image, masque de scores et top 10%, graine 11 fixee.",
        "- *_masks.npz : scores de noeuds et aretes pour chaque graine.",
        "- deletion.csv : tous les effets signes et absolus, sans selection.",
        "- explanations.csv : plages de scores et controles de poids inchanges.",
        "- summary.json : version PyG, coefficients, protocole et empreintes.",
        "",
        "Reproduction : scripts/run_gnnexplainer_baseline.py (refuse d ecraser les resultats).",
        "Synthese seule : scripts/report_gnnexplainer.py.",
        "",
        "## Limites et source",
        "",
        "Il s agit d une baseline adaptee de regression, pas d une reproduction exacte",
        "d un article diagnostique. Remplacer des embeddings ne garantit aucune",
        "plausibilite histologique. Les masques d aretes sont sauvegardes, sans test",
        "separe de leur qualite. Ce resultat ne demontre pas la contribution GCIA.",
        "",
        "[GNNExplainer, Ying et al.](https://arxiv.org/abs/1903.03894) ;",
        "[implementation PyG](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.explain.algorithm.GNNExplainer.html).",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[19:35]))


if __name__ == "__main__":
    main()

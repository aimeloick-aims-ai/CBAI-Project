"""Report all three exploratory audits without selecting or tuning their results."""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/bracs_three_xai"


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    lines = [
        "# Trois audits XAI sur le GCN BRACS gelé",
        "",
        "Analyse exploratoire des 22 patients du test interne déjà consulté. Aucun réentraînement",
        "du classifieur, aucun téléchargement. Ses probabilités ont été vérifiées contre les résultats",
        "précédents et ses poids sont inchangés. Les initialisations ne sont pas des patients indépendants.",
        "",
        "## 1. GNNExplainer : implémentation PyG officielle",
        "",
        "66 explications (22 patients × 3 initialisations), 100 époques chacune, sur l'ensemble",
        "exact des trois GCN gelés. Masques de nœuds et d'arêtes optimisés conjointement ;",
        "seul le classement des nœuds est évalué. Remplacement des embeddings par la moyenne",
        "d'apprentissage, avec 20 contrôles aléatoires de même taille par patient, graine et budget.",
        "",
        "Baisse moyenne de probabilité de la **classe initialement prédite**, en points de pourcentage.",
        "Une valeur négative signifie une augmentation de confiance. Les IC sont descriptifs,",
        "bootstrap apparié sur les patients après moyenne des graines, sans correction de multiplicité.",
        "",
        "| Budget de nœuds | GNNExplainer | Aléatoire | Différence [IC 95 %] |",
        "|---|---:|---:|---:|",
    ]
    for fraction, values in summary["gnnexplainer"].items():
        diff = values["paired_difference"]
        lo, hi = diff["patient_bootstrap_95"]
        lines.append(
            f"| {float(fraction):.0%} | {100 * values['gnnexplainer']:.2f} | {100 * values['random']:.2f} | {100 * diff['mean']:.2f} [{100 * lo:.2f}, {100 * hi:.2f}] |"
        )
    gc = summary["gcexplainer_adaptation"]
    lines += [
        "",
        "## 2. GCExplainer : adaptation automatique, sans revue humaine",
        "",
        "K-means à cinq groupes sur les activations finales des nœuds concaténées entre les trois",
        "GCN. Ajustement sur les 34 patients d'apprentissage uniquement. Histogramme des groupes",
        "par région puis arbre de décision de profondeur 3 imitant les classes prédites du GCN.",
        "Cette adaptation n'est pas une reproduction complète de l'article : pas de validation",
        "humaine ni de mesure de pureté par distance d'édition des graphes.",
        "",
        f"- Accord avec le GCN sur les 22 patients : **{gc['test_agreement_with_gcn']:.1%}**.",
        f"- Accord du prédicteur constant majoritaire d'apprentissage : {gc['constant_train_majority_agreement']:.1%}.",
        "- Les groupes sont des concepts candidats sans nom histopathologique validé.",
        "",
        "Méthode inspirée de [Magister et al., GCExplainer](https://arxiv.org/abs/2107.11889).",
        "L'accord mesure une imitation des décisions, pas une preuve d'utilisation des concepts.",
        "",
        "## 3. GCIA : pilote technique sur trois propriétés visuelles",
        "",
        "Cibles mesurées sur les pixels : luminosité, rouge moins bleu, écart-type d'intensité.",
        "Ce ne sont pas des annotations pathologiques. Probes ridge ajustées uniquement sur",
        "l'apprentissage aux entrées et aux deux couches ; R² ci-dessous calculé sur les moyennes",
        "par patient du test. Un R² négatif indique une faible capacité de décodage.",
        "",
        "Interventions dans les embeddings d'entrée : norme L2 par nœud égale à 1, deux sens,",
        "direction orthogonale aux deux autres probes d'entrée. Le contrôle numérique exige",
        "un changement ciblé ≥0,1 écart-type, des changements collatéraux ≤0,05 et ≥99 %",
        "des valeurs dans les bornes observées en apprentissage. Cela ne prouve pas la plausibilité",
        "histologique : aucune édition de pixels ni de structure n'est évaluée ici.",
        "",
        "| Propriété proxy | R² couche 2 | Contrôle numérique réussi | Variation absolue P(IC), points |",
        "|---|---:|---:|---:|",
    ]
    for concept, scores in summary["gcia_proxy"].items():
        lines.append(
            f"| {concept} | {scores['layer2_patient_mean_r2']:.3f} | {scores['numerical_gate_rate']:.1%} | {100 * scores['mean_absolute_probability_change']:.3f} |"
        )
    edits = list(csv.DictReader((OUT / "gcia_proxy_interventions.csv").open()))
    lines += [
        "",
        "Contrôles isotropes de même norme (20 par intervention) :",
        "",
        "| Propriété | Variation absolue P(IC) aléatoire, points |",
        "|---|---:|",
    ]
    for concept in summary["gcia_proxy"]:
        values = [
            abs(float(r["probability_IC_change"]))
            for r in edits
            if r["concept"] == concept and r["method"] == "random"
        ]
        lines.append(f"| {concept} | {100 * np.mean(values):.3f} |")
    lines += [
        "",
        "**GCIA clinique reste non validé.** Les variations de probes sur embeddings modifiés",
        "peuvent refléter une exploitation du probe. Aucun de ces résultats n'autorise à classer",
        "un concept pathologique comme « encodé et utilisé ». Les contrôles numériques sont",
        "des diagnostics partiels, pas des critères de validité biologique.",
        "",
        "## Fichiers reproductibles",
        "",
        "- `started.json` : protocole fixé avant calcul et empreintes.",
        "- `summary.json`, `deletion.csv` : synthèse et toutes les perturbations GNNExplainer.",
        "- `gcexplainer_predictions.csv`, `gcexplainer_clusters.npz` : imitation et groupes.",
        "- `probes.csv`, `gcia_proxy_interventions.csv` : décodage et toutes les interventions, y compris échecs.",
        "- `representations.npz`, `*_masks.npz` : représentations et masques sauvegardés.",
        "- `gnnexplainer_example.png` : premier patient selon l'ordre du manifeste, sans sélection visuelle.",
        "- `gcexplainer_prototypes.png` : patch d'apprentissage le plus proche de chaque centre, sans nom clinique attribué.",
        "",
        "Ces trois métriques ne forment pas un classement entre méthodes. La petite cohorte",
        "sélectionnée par taille des fichiers et son test déjà examiné limitent la généralisation.",
        "L'abstract multi-cohortes sur graphes cellulaires et concepts pathologiques reste à valider.",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = list(csv.DictReader((ROOT / "data/bracs_expanded_study_manifest.csv").open()))
    row = next(r for r in manifest if r["proposed_split"] == "internal_test")
    masks = [
        np.load(OUT / f"{row['patient_id']}_{seed}_masks.npz")["node"] for seed in (11, 23, 37)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    with Image.open(ROOT / row["path"]) as im:
        axes[0].imshow(im.convert("RGB").resize((768, 768)))
    axes[0].set_title(f"Patient {row['patient_id']}")
    heat = axes[1].imshow(np.mean(masks, axis=0).reshape(8, 8), vmin=0, vmax=1, cmap="viridis")
    axes[1].set_title("Mean GNNExplainer score (3 seeds)")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(heat, ax=axes[1], shrink=0.7)
    fig.savefig(OUT / "gnnexplainer_example.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    representations = np.load(OUT / "representations.npz")["layer2"]
    centers = np.load(OUT / "gcexplainer_clusters.npz")["centers"]
    train_ids = [i for i, r in enumerate(manifest) if r["proposed_split"] == "train"]
    train_nodes = representations[train_ids].reshape(-1, representations.shape[-1])
    fig, axes = plt.subplots(1, len(centers), figsize=(12, 3))
    for cluster, (center, ax) in enumerate(zip(centers, axes, strict=True)):
        closest = int(np.argmin(np.sum((train_nodes - center) ** 2, axis=1)))
        region, node = divmod(closest, 64)
        candidate = manifest[train_ids[region]]
        y, x = divmod(node, 8)
        with Image.open(ROOT / candidate["path"]) as im:
            rgb = im.convert("RGB").resize((768, 768), Image.Resampling.BILINEAR)
            ax.imshow(rgb.crop((96 * x, 96 * y, 96 * (x + 1), 96 * (y + 1))))
        ax.set_title(f"Cluster {cluster}\nPatient {candidate['patient_id']}, patch {node}")
        ax.axis("off")
    fig.savefig(OUT / "gcexplainer_prototypes.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

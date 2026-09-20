"""Generate a complete cellular XAI report and a preselected example figure."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/cell_xai"


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    interventions = list(csv.DictReader((OUT / "interventions.csv").open()))
    lines = [
        "# GNNExplainer et GCIA sur les GNN cellulaires",
        "",
        "22 patients du test historique déjà exploré. Trois architectures gelées, chacune",
        "évaluée comme moyenne des probabilités des trois graines 11/23/37. Les probabilités",
        "de départ sont vérifiées contre les résultats sauvegardés et les poids restent inchangés.",
        "Aucun nouveau téléchargement ni ajustement du classifieur.",
        "",
        "## GNNExplainer",
        "",
        "198 explications avec PyG officiel, 100 époques chacune. Masques de nœuds et d'arêtes",
        "optimisés ensemble, puis évaluation du classement des nœuds uniquement. Les attributs",
        "des nœuds sélectionnés sont remplacés par la moyenne d'apprentissage ; topologie inchangée.",
        "Vingt contrôles aléatoires de même nombre de nœuds par budget, patient et graine.",
        "",
        "Baisse de probabilité de la classe initialement prédite, en points de pourcentage.",
        "Les différences et intervalles bootstrap sont calculés par patient après moyenne",
        "des graines. Ce sont des intervalles descriptifs, sans correction de multiplicité.",
        "Une différence positive favorise GNNExplainer pour ce test de perturbation uniquement.",
        "",
        "| Modèle | Budget | GNNExplainer | Aléatoire | Différence [IC 95 %] |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in summary["results"].items():
        for fraction, score in result["gnnexplainer"].items():
            diff = score["difference"]
            lo, hi = diff["patient_bootstrap_95"]
            lines.append(
                f"| {name} | {float(fraction):.0%} | {100 * score['gnnexplainer']:.2f} | {100 * score['random']:.2f} | {100 * diff['mean']:.2f} [{100 * lo:.2f}, {100 * hi:.2f}] |"
            )
    lines += [
        "",
        "## GCIA : audit morphologique partiel",
        "",
        "Les concepts sont les moyennes d'aire, d'excentricité et de solidité des noyaux",
        "prédits par HoVer-Net. Ce ne sont pas des annotations indépendantes de pathologiste.",
        "Une probe ridge est ajustée sur les représentations cachées des patients d'apprentissage",
        "uniquement. Ces GNN ont une seule couche de convolution : aucun résultat multi-couche",
        "n'est revendiqué. Un R² négatif indique un décodage inférieur à la référence moyenne de test.",
        "",
        "| Modèle | R² aire | R² excentricité | R² solidité |",
        "|---|---:|---:|---:|",
    ]
    for name, result in summary["results"].items():
        r = result["probe_r2"]
        lines.append(
            f"| {name} | {r['area']:.3f} | {r['eccentricity']:.3f} | {r['solidity']:.3f} |"
        )
    lines += [
        "",
        "L'intervention multiplie l'aire de tous les noyaux par 0,95 ou 1,05 et les deux",
        "axes par la racine carrée du facteur. Les autres attributs et les connexions sont",
        "conservés. Les contours et pixels ne sont pas modifiés et les collisions ne sont pas",
        "vérifiées : cette édition numérique n'est pas une intervention histologiquement validée.",
        "Les contrôles sont isotropes dans l'espace normalisé, avec même norme par nœud.",
        "",
        "Variations **absolues** de P(IC), en points, distinctes de la métrique GNNExplainer.",
        "",
        "| Modèle | Aire | Intervention | Aléatoire | Probe dans le sens attendu |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in summary["results"].items():
        for factor, score in result["gcia"].items():
            lines.append(
                f"| {name} | {(float(factor) - 1) * 100:+.0f}% | {100 * score['gcia_morphology']:.3f} | {100 * score['random']:.3f} | {score['probe_direction_success_rate']:.1%} |"
            )
    lines += [
        "",
        "Changements collatéraux des probes non ciblées (moyenne des valeurs absolues en écarts-types",
        "d'apprentissage, les deux sens réunis). La proportion dans les bornes d'apprentissage",
        "est un diagnostic numérique, pas un test de plausibilité histologique.",
        "",
        "| Modèle | Probe excentricité | Probe solidité | Attributs dans les bornes du train |",
        "|---|---:|---:|---:|",
    ]
    for name in summary["results"]:
        chosen = [
            r
            for r in interventions
            if r["architecture"] == name and r["method"] == "gcia_morphology"
        ]
        ecc, sol = [
            np.mean([abs(float(r[key])) for r in chosen])
            for key in ("eccentricity_probe_shift_sd", "solidity_probe_shift_sd")
        ]
        support = np.mean([float(r["feature_support_fraction"]) for r in chosen])
        lines.append(f"| {name} | {ecc:.3f} | {sol:.3f} | {support:.1%} |")
    coherent = json.loads((OUT / "coherent_summary.json").read_text())
    lines += [
        "",
        "## Contrôle supplémentaire de cohérence entre noyaux",
        "",
        "Analyse de robustesse ajoutée après le premier audit, sans modifier les modèles ni les interventions.",
        "Les directions aléatoires sont ici communes à tous les noyaux, en conservant la norme de",
        "perturbation de chaque nœud. Cela réduit l'avantage artificiel lié à l'annulation des bruits",
        "indépendants lors de l'agrégation. Résultats en points de probabilité, toutes conditions rapportées.",
        "",
        "| Modèle | Aire | Contrôle cohérent | Intervention moins contrôle [IC 95 %] |",
        "|---|---:|---:|---:|",
    ]
    for name, conditions in coherent.items():
        for factor, score in conditions.items():
            diff = score["target_minus_coherent_control"]
            lo, hi = diff["patient_bootstrap_95"]
            lines.append(
                f"| {name} | {(float(factor) - 1) * 100:+.0f}% | {100 * score['coherent_control_mean_absolute_change']:.3f} | {100 * diff['mean']:.3f} [{100 * lo:.3f}, {100 * hi:.3f}] |"
            )
    lines += [
        "",
        "## Conclusion et portée",
        "",
        "Ces expériences testent l'influence des attributs sur les nouveaux modèles cellulaires.",
        "Elles ne permettent pas d'affirmer qu'un concept pathologique est encodé et utilisé :",
        "les caractéristiques dépendent d'une segmentation BRACS non validée, les probes peuvent",
        "être sensibles à des changements hors distribution et les contrôles ne sont pas des tissus réels.",
        "La prédiction des concepts ne suffit pas à démontrer leur utilisation. Les axes covarient",
        "avec l'aire : l'effet n'est pas attribuable à l'aire seule. L'échelle physique est inconnue.",
        "Le test contient seulement 22 patients et a déjà servi à d'autres analyses exploratoires.",
        "",
        "## Fichiers",
        "",
        "- `protocol.json` et `summary.json` : paramètres, métriques et empreintes des neuf checkpoints.",
        "- `deletions.csv` : toutes les perturbations GNNExplainer et aléatoires.",
        "- `interventions.csv` : toutes les interventions morphologiques, contrôles et changements de probes.",
        "- `coherent_control_protocol.json`, `coherent_controls.csv`, `coherent_summary.json` : analyse de robustesse supplémentaire.",
        "- `probes.csv`, `*_probe.npz` : décodage et paramètres des probes.",
        "- `*_masks.npz` : masques de nœuds/arêtes pour les 198 explications.",
        "- `example.png` : premier patient du test dans le manifeste, moyenne des trois graines, échelle 0..1.",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = list(csv.DictReader((ROOT / "reports/cell_cohort/manifest.csv").open()))
    row = next(r for r in manifest if r["split"] == "test")
    pid = row["patient_id"]
    graph = torch.load(ROOT / "reports/cell_cohort" / row["graph_path"], weights_only=True)
    source_rows = list(csv.DictReader((ROOT / "data/bracs_expanded_study_manifest.csv").open()))
    source = next(r for r in source_rows if r["patient_id"] == pid)
    quality = list(csv.DictReader((ROOT / "reports/cell_cohort/quality.csv").open()))
    q = next(r for r in quality if r["patient_id"] == pid)
    x, y = graph["crop_origin_xy"]
    with Image.open(ROOT / source["path"]) as image:
        rgb = np.array(
            image.crop((x, y, x + int(q["crop_width"]), y + int(q["crop_height"]))).convert("RGB")
        )
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    axes[0].imshow(rgb)
    axes[0].set_title(f"Patient {pid}, original")
    pos = graph["pos"].numpy()
    for ax, name in zip(axes[1:], summary["results"], strict=True):
        masks = [np.load(OUT / f"{name}_{pid}_{seed}_masks.npz")["node"] for seed in (11, 23, 37)]
        ax.imshow(rgb)
        colors = ax.scatter(
            pos[:, 0], pos[:, 1], c=np.mean(masks, axis=0), s=12, cmap="viridis", vmin=0, vmax=1
        )
        ax.set_title(name)
    for ax in axes:
        ax.axis("off")
    fig.colorbar(colors, ax=list(axes[1:]), shrink=0.65, label="GNNExplainer score")
    fig.savefig(OUT / "example.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()

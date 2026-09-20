"""Report complete segmentation measurements and constructed candidate cell graphs."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "reports/segmentation_evaluation"
    summary = json.loads((out / "summary.json").read_text())
    cohort = ROOT / "reports/cell_cohort"
    rows = list(csv.DictReader((cohort / "manifest.csv").open()))
    quality = list(csv.DictReader((cohort / "quality.csv").open()))
    excluded_path = cohort / "excluded.csv"
    excluded = list(csv.DictReader(excluded_path.open())) if excluded_path.exists() else []
    lines = [
        "# Évaluation de segmentation et graphes cellulaires candidats",
        "",
        "## Évaluation effectivement réalisée",
        "",
        "14 images complètes du test officiel MoNuSeg, sans sélection d'image ni ajustement",
        "du modèle. HoVer-Net fast préentraîné sur PanNuke comparé à notre baseline watershed.",
        "Les contours XML sont rasterisés à la résolution native et les instances appariées",
        "une à une avec IoU strictement supérieur à 0,5. Moyennes par image ci-dessous ;",
        "les valeurs de chaque image sont conservées dans metrics.csv.",
        "",
        "| Méthode | Dice premier plan | F1 détection | SQ | PQ |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, scores in summary["methods"].items():
        lines.append(
            f"| {method} | {scores['foreground_dice']:.3f} | {scores['detection_f1']:.3f} | {scores['segmentation_quality']:.3f} | {scores['panoptic_quality']:.3f} |"
        )
    lines += [
        "",
        "Un Dice élevé ne suffit pas : SQ et PQ pénalisent les frontières et les fusions/scissions.",
        "Aucun seuil de réussite n'a été choisi après observation des scores. Aucun score n'est",
        "présenté comme une validation clinique. Le recouvrement éventuel des lames TCGA avec",
        "le préentraînement PanNuke n'est pas exclu : l'indépendance externe n'est pas établie.",
        "",
        "Source : [données officielles MoNuSeg](https://monuseg.grand-challenge.org/Data/).",
        "Citer Kumar et al., A Multi-organ Nucleus Segmentation Challenge, IEEE TMI 2019.",
        "Les règles du challenge ne sont pas revendiquées : nous évaluons un modèle préentraîné externe.",
        "",
        "## Graphes BRACS construits après l'évaluation",
        "",
        f"- {len(rows)} graphes, un par patient ; {len(excluded)} exclusions explicites dans excluded.csv (moins de deux noyaux retenus).",
        "- Politique d'exclusion amendée après l'arrêt initial du calcul, avant tout entraînement cellulaire ; biais de sélection possible.",
        f"- Apprentissage : {sum(r['split'] == 'train' for r in rows)} ; validation : {sum(r['split'] == 'validation' for r in rows)} ; test historique déjà exploré : {sum(r['split'] == 'test' for r in rows)}.",
        f"- {sum(int(r['nodes']) for r in quality)} nœuds candidats ; médiane par graphe : {np.median([int(r['nodes']) for r in quality]):.0f}.",
        f"- {sum(int(r['undirected_edges']) for r in quality)} connexions non orientées ; {sum(int(r['isolated_nodes']) for r in quality)} nœuds isolés.",
        "- Crop central jusqu'à 512×512 pixels natifs ; pas de redimensionnement ni de sélection guidée par les scores.",
        "- Morphologie et RGB par instance ; voisinage k=8 symétrisé, rayon maximal 50 pixels.",
        "",
        "**Ces graphes restent candidats.** MoNuSeg n'est pas BRACS. L'échelle physique de BRACS",
        "n'est pas documentée ici, alors que le modèle attend 0,25 µm/pixel. Les coordonnées",
        "et aires BRACS sont donc en pixels, pas en micromètres. Les crops héritent du diagnostic",
        "de la RoI complète : leur représentativité locale n'a pas été revue par un pathologiste.",
        "Les performances BRACS de segmentation et la plausibilité clinique restent à établir.",
        "",
        "## Fichiers",
        "",
        "- `reports/segmentation_evaluation/metrics.csv` : tous les scores ; `*_comparison.png` : toutes les planches.",
        "- `reports/cell_cohort/manifest.csv` : patients, partitions, labels, chemins et empreintes des graphes.",
        "- `reports/cell_cohort/quality.csv` : contrôles par graphe ; `.pt` : graphes ; `_instances.npy` : segmentations.",
        "- Les protocoles, empreintes et poids sont conservés. Aucun nouveau GNN n'a été entraîné dans cette étape.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "segmentation": summary["methods"],
                "graphs": len(rows),
                "nodes": sum(int(r["nodes"]) for r in quality),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

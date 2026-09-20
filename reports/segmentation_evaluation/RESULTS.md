---
noteId: "d1d9d550b4f411f1ac794713d9e3a04f"
tags: []

---

# Évaluation de segmentation et graphes cellulaires candidats

## Évaluation effectivement réalisée

14 images complètes du test officiel MoNuSeg, sans sélection d'image ni ajustement
du modèle. HoVer-Net fast préentraîné sur PanNuke comparé à notre baseline watershed.
Les contours XML sont rasterisés à la résolution native et les instances appariées
une à une avec IoU strictement supérieur à 0,5. Moyennes par image ci-dessous ;
les valeurs de chaque image sont conservées dans metrics.csv.

| Méthode | Dice premier plan | F1 détection | SQ | PQ |
|---|---:|---:|---:|---:|
| hovernet | 0.831 | 0.850 | 0.773 | 0.657 |
| watershed | 0.762 | 0.288 | 0.708 | 0.204 |

Un Dice élevé ne suffit pas : SQ et PQ pénalisent les frontières et les fusions/scissions.
Aucun seuil de réussite n'a été choisi après observation des scores. Aucun score n'est
présenté comme une validation clinique. Le recouvrement éventuel des lames TCGA avec
le préentraînement PanNuke n'est pas exclu : l'indépendance externe n'est pas établie.

Source : [données officielles MoNuSeg](https://monuseg.grand-challenge.org/Data/).
Citer Kumar et al., A Multi-organ Nucleus Segmentation Challenge, IEEE TMI 2019.
Les règles du challenge ne sont pas revendiquées : nous évaluons un modèle préentraîné externe.

## Graphes BRACS construits après l'évaluation

- 65 graphes, un par patient ; 1 exclusions explicites dans excluded.csv (moins de deux noyaux retenus).
- Politique d'exclusion amendée après l'arrêt initial du calcul, avant tout entraînement cellulaire ; biais de sélection possible.
- Apprentissage : 33 ; validation : 10 ; test historique déjà exploré : 22.
- 8154 nœuds candidats ; médiane par graphe : 122.
- 19701 connexions non orientées ; 314 nœuds isolés.
- Crop central jusqu'à 512×512 pixels natifs ; pas de redimensionnement ni de sélection guidée par les scores.
- Morphologie et RGB par instance ; voisinage k=8 symétrisé, rayon maximal 50 pixels.

**Ces graphes restent candidats.** MoNuSeg n'est pas BRACS. L'échelle physique de BRACS
n'est pas documentée ici, alors que le modèle attend 0,25 µm/pixel. Les coordonnées
et aires BRACS sont donc en pixels, pas en micromètres. Les crops héritent du diagnostic
de la RoI complète : leur représentativité locale n'a pas été revue par un pathologiste.
Les performances BRACS de segmentation et la plausibilité clinique restent à établir.

## Fichiers

- `reports/segmentation_evaluation/metrics.csv` : tous les scores ; `*_comparison.png` : toutes les planches.
- `reports/cell_cohort/manifest.csv` : patients, partitions, labels, chemins et empreintes des graphes.
- `reports/cell_cohort/quality.csv` : contrôles par graphe ; `.pt` : graphes ; `_instances.npy` : segmentations.
- Les protocoles, empreintes et poids sont conservés. Aucun nouveau GNN n'a été entraîné dans cette étape.

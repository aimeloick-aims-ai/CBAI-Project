---
noteId: "860cc2d0b4f011f1ac794713d9e3a04f"
tags: []

---

# Évaluation MoNuSeg terminée

14 images complètes du test officiel, 6 697 instances de référence.
Paramètres fixés avant le calcul, sans ajustement du modèle.

| Méthode | Dice | F1 détection | SQ | PQ |
|---|---:|---:|---:|---:|
| HoVer-Net PanNuke | 0,831 | 0,850 | 0,773 | 0,657 |
| Watershed | 0,762 | 0,288 | 0,708 | 0,204 |

Moyennes par image. Sources numériques : [summary.json](summary.json) et
[metrics.csv](metrics.csv). Les 14 planches `*_comparison.png` sont conservées.

HoVer-Net obtient de meilleurs scores sur cet ensemble. Cela ne constitue pas une
validation de la segmentation BRACS, ni une validation externe indépendante : le
recouvrement éventuel des lames TCGA avec le préentraînement PanNuke n'est pas exclu.
Source : [jeu officiel MoNuSeg](https://monuseg.grand-challenge.org/Data/).

La construction BRACS s'est arrêtée après 39 graphes, sur un crop avec moins de deux
instances retenues. La reprise conserve les graphes existants et consigne les
exclusions dans `reports/cell_cohort/excluded.csv`, sans inventer de noyaux ni changer
les crops pour obtenir un résultat favorable.

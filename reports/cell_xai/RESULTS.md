# GNNExplainer et GCIA sur les GNN cellulaires

22 patients du test historique déjà exploré. Trois architectures gelées, chacune
évaluée comme moyenne des probabilités des trois graines 11/23/37. Les probabilités
de départ sont vérifiées contre les résultats sauvegardés et les poids restent inchangés.
Aucun nouveau téléchargement ni ajustement du classifieur.

## GNNExplainer

198 explications avec PyG officiel, 100 époques chacune. Masques de nœuds et d'arêtes
optimisés ensemble, puis évaluation du classement des nœuds uniquement. Les attributs
des nœuds sélectionnés sont remplacés par la moyenne d'apprentissage ; topologie inchangée.
Vingt contrôles aléatoires de même nombre de nœuds par budget, patient et graine.

Baisse de probabilité de la classe initialement prédite, en points de pourcentage.
Les différences et intervalles bootstrap sont calculés par patient après moyenne
des graines. Ce sont des intervalles descriptifs, sans correction de multiplicité.
Une différence positive favorise GNNExplainer pour ce test de perturbation uniquement.

| Modèle | Budget | GNNExplainer | Aléatoire | Différence [IC 95 %] |
|---|---:|---:|---:|---:|
| GCN | 5% | 2.33 | 0.67 | 1.67 [0.88, 2.46] |
| GCN | 10% | 3.76 | 1.29 | 2.48 [1.27, 3.78] |
| GCN | 20% | 6.02 | 2.45 | 3.58 [1.83, 5.55] |
| GraphSAGE | 5% | 3.32 | 0.79 | 2.53 [1.69, 3.33] |
| GraphSAGE | 10% | 5.43 | 1.52 | 3.91 [2.62, 5.21] |
| GraphSAGE | 20% | 8.21 | 2.87 | 5.35 [3.55, 7.41] |
| GATv2 | 5% | 2.17 | 0.54 | 1.63 [0.81, 2.48] |
| GATv2 | 10% | 3.36 | 1.07 | 2.29 [1.05, 3.56] |
| GATv2 | 20% | 5.14 | 2.07 | 3.07 [1.48, 4.87] |

## GCIA : audit morphologique partiel

Les concepts sont les moyennes d'aire, d'excentricité et de solidité des noyaux
prédits par HoVer-Net. Ce ne sont pas des annotations indépendantes de pathologiste.
Une probe ridge est ajustée sur les représentations cachées des patients d'apprentissage
uniquement. Ces GNN ont une seule couche de convolution : aucun résultat multi-couche
n'est revendiqué. Un R² négatif indique un décodage inférieur à la référence moyenne de test.

| Modèle | R² aire | R² excentricité | R² solidité |
|---|---:|---:|---:|
| GCN | 0.988 | 0.839 | 0.820 |
| GraphSAGE | 0.980 | 0.780 | 0.854 |
| GATv2 | 0.983 | 0.789 | 0.752 |

L'intervention multiplie l'aire de tous les noyaux par 0,95 ou 1,05 et les deux
axes par la racine carrée du facteur. Les autres attributs et les connexions sont
conservés. Les contours et pixels ne sont pas modifiés et les collisions ne sont pas
vérifiées : cette édition numérique n'est pas une intervention histologiquement validée.
Les contrôles sont isotropes dans l'espace normalisé, avec même norme par nœud.

Variations **absolues** de P(IC), en points, distinctes de la métrique GNNExplainer.

| Modèle | Aire | Intervention | Aléatoire | Probe dans le sens attendu |
|---|---:|---:|---:|---:|
| GCN | -5% | 1.766 | 0.103 | 100.0% |
| GCN | +5% | 1.812 | 0.101 | 100.0% |
| GraphSAGE | -5% | 2.091 | 0.115 | 100.0% |
| GraphSAGE | +5% | 2.139 | 0.108 | 100.0% |
| GATv2 | -5% | 1.469 | 0.119 | 100.0% |
| GATv2 | +5% | 1.502 | 0.110 | 100.0% |

Changements collatéraux des probes non ciblées (moyenne des valeurs absolues en écarts-types
d'apprentissage, les deux sens réunis). La proportion dans les bornes d'apprentissage
est un diagnostic numérique, pas un test de plausibilité histologique.

| Modèle | Probe excentricité | Probe solidité | Attributs dans les bornes du train |
|---|---:|---:|---:|
| GCN | 0.013 | 0.042 | 100.0% |
| GraphSAGE | 0.026 | 0.044 | 100.0% |
| GATv2 | 0.022 | 0.024 | 100.0% |

## Contrôle supplémentaire de cohérence entre noyaux

Analyse de robustesse ajoutée après le premier audit, sans modifier les modèles ni les interventions.
Les directions aléatoires sont ici communes à tous les noyaux, en conservant la norme de
perturbation de chaque nœud. Cela réduit l'avantage artificiel lié à l'annulation des bruits
indépendants lors de l'agrégation. Résultats en points de probabilité, toutes conditions rapportées.

| Modèle | Aire | Contrôle cohérent | Intervention moins contrôle [IC 95 %] |
|---|---:|---:|---:|
| GCN | -5% | 0.854 | 0.911 [0.660, 1.162] |
| GCN | +5% | 0.802 | 1.011 [0.765, 1.255] |
| GraphSAGE | -5% | 0.923 | 1.168 [0.879, 1.463] |
| GraphSAGE | +5% | 0.884 | 1.255 [0.994, 1.494] |
| GATv2 | -5% | 0.737 | 0.731 [0.540, 0.916] |
| GATv2 | +5% | 0.693 | 0.809 [0.625, 0.979] |

## Conclusion et portée

Ces expériences testent l'influence des attributs sur les nouveaux modèles cellulaires.
Elles ne permettent pas d'affirmer qu'un concept pathologique est encodé et utilisé :
les caractéristiques dépendent d'une segmentation BRACS non validée, les probes peuvent
être sensibles à des changements hors distribution et les contrôles ne sont pas des tissus réels.
La prédiction des concepts ne suffit pas à démontrer leur utilisation. Les axes covarient
avec l'aire : l'effet n'est pas attribuable à l'aire seule. L'échelle physique est inconnue.
Le test contient seulement 22 patients et a déjà servi à d'autres analyses exploratoires.

## Fichiers

- `protocol.json` et `summary.json` : paramètres, métriques et empreintes des neuf checkpoints.
- `deletions.csv` : toutes les perturbations GNNExplainer et aléatoires.
- `interventions.csv` : toutes les interventions morphologiques, contrôles et changements de probes.
- `coherent_control_protocol.json`, `coherent_controls.csv`, `coherent_summary.json` : analyse de robustesse supplémentaire.
- `probes.csv`, `*_probe.npz` : décodage et paramètres des probes.
- `*_masks.npz` : masques de nœuds/arêtes pour les 198 explications.
- `example.png` : premier patient du test dans le manifeste, moyenne des trois graines, échelle 0..1.

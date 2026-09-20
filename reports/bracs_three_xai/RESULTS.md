# Trois audits XAI sur le GCN BRACS gelé

Analyse exploratoire des 22 patients du test interne déjà consulté. Aucun réentraînement
du classifieur, aucun téléchargement. Ses probabilités ont été vérifiées contre les résultats
précédents et ses poids sont inchangés. Les initialisations ne sont pas des patients indépendants.

## 1. GNNExplainer : implémentation PyG officielle

66 explications (22 patients × 3 initialisations), 100 époques chacune, sur l'ensemble
exact des trois GCN gelés. Masques de nœuds et d'arêtes optimisés conjointement ;
seul le classement des nœuds est évalué. Remplacement des embeddings par la moyenne
d'apprentissage, avec 20 contrôles aléatoires de même taille par patient, graine et budget.

Baisse moyenne de probabilité de la **classe initialement prédite**, en points de pourcentage.
Une valeur négative signifie une augmentation de confiance. Les IC sont descriptifs,
bootstrap apparié sur les patients après moyenne des graines, sans correction de multiplicité.

| Budget de nœuds | GNNExplainer | Aléatoire | Différence [IC 95 %] |
|---|---:|---:|---:|
| 5% | 5.61 | 0.75 | 4.86 [3.09, 6.75] |
| 10% | 10.60 | 1.33 | 9.27 [6.07, 12.52] |
| 20% | 19.15 | 2.62 | 16.53 [11.61, 21.36] |

## 2. GCExplainer : adaptation automatique, sans revue humaine

K-means à cinq groupes sur les activations finales des nœuds concaténées entre les trois
GCN. Ajustement sur les 34 patients d'apprentissage uniquement. Histogramme des groupes
par région puis arbre de décision de profondeur 3 imitant les classes prédites du GCN.
Cette adaptation n'est pas une reproduction complète de l'article : pas de validation
humaine ni de mesure de pureté par distance d'édition des graphes.

- Accord avec le GCN sur les 22 patients : **100.0%**.
- Accord du prédicteur constant majoritaire d'apprentissage : 36.4%.
- Les groupes sont des concepts candidats sans nom histopathologique validé.

Méthode inspirée de [Magister et al., GCExplainer](https://arxiv.org/abs/2107.11889).
L'accord mesure une imitation des décisions, pas une preuve d'utilisation des concepts.

## 3. GCIA : pilote technique sur trois propriétés visuelles

Cibles mesurées sur les pixels : luminosité, rouge moins bleu, écart-type d'intensité.
Ce ne sont pas des annotations pathologiques. Probes ridge ajustées uniquement sur
l'apprentissage aux entrées et aux deux couches ; R² ci-dessous calculé sur les moyennes
par patient du test. Un R² négatif indique une faible capacité de décodage.

Interventions dans les embeddings d'entrée : norme L2 par nœud égale à 1, deux sens,
direction orthogonale aux deux autres probes d'entrée. Le contrôle numérique exige
un changement ciblé ≥0,1 écart-type, des changements collatéraux ≤0,05 et ≥99 %
des valeurs dans les bornes observées en apprentissage. Cela ne prouve pas la plausibilité
histologique : aucune édition de pixels ni de structure n'est évaluée ici.

| Propriété proxy | R² couche 2 | Contrôle numérique réussi | Variation absolue P(IC), points |
|---|---:|---:|---:|
| brightness_proxy | 0.197 | 95.5% | 2.253 |
| red_minus_blue_proxy | 0.371 | 95.5% | 2.491 |
| intensity_sd_proxy | -0.043 | 95.5% | 3.753 |

Contrôles isotropes de même norme (20 par intervention) :

| Propriété | Variation absolue P(IC) aléatoire, points |
|---|---:|
| brightness_proxy | 0.954 |
| red_minus_blue_proxy | 0.924 |
| intensity_sd_proxy | 0.915 |

**GCIA clinique reste non validé.** Les variations de probes sur embeddings modifiés
peuvent refléter une exploitation du probe. Aucun de ces résultats n'autorise à classer
un concept pathologique comme « encodé et utilisé ». Les contrôles numériques sont
des diagnostics partiels, pas des critères de validité biologique.

## Fichiers reproductibles

- `started.json` : protocole fixé avant calcul et empreintes.
- `summary.json`, `deletion.csv` : synthèse et toutes les perturbations GNNExplainer.
- `gcexplainer_predictions.csv`, `gcexplainer_clusters.npz` : imitation et groupes.
- `probes.csv`, `gcia_proxy_interventions.csv` : décodage et toutes les interventions, y compris échecs.
- `representations.npz`, `*_masks.npz` : représentations et masques sauvegardés.
- `gnnexplainer_example.png` : premier patient selon l'ordre du manifeste, sans sélection visuelle.
- `gcexplainer_prototypes.png` : patch d'apprentissage le plus proche de chaque centre, sans nom clinique attribué.

Ces trois métriques ne forment pas un classement entre méthodes. La petite cohorte
sélectionnée par taille des fichiers et son test déjà examiné limitent la généralisation.
L'abstract multi-cohortes sur graphes cellulaires et concepts pathologiques reste à valider.

# Baseline GNNExplainer executee

Adaptation sur BCSS, encodeur PCam gele et GCN : regression de proportion tumorale.
Huit patients, trois graines, 24 explications, 1512 evaluations de perturbation.
Aucun nouveau telechargement. Aucun poids du GNN modifie.

## Protocole

GNNExplainer officiel PyG : 100 epochs, lr=0,01, regularisation par defaut.
Masques de noeuds et aretes optimises conjointement. Evaluation du classement
des noeuds seulement : remplacement par la moyenne des features du train,
topologie conservee. Vingt controles aleatoires de meme taille par budget.
La cible est la prediction du modele, sans usage des masques tissulaires pour
optimiser les explications. Les annotations definissent les patches valides.

## Resultats descriptifs

Variation absolue moyenne de proportion predite, en points de pourcentage.
Plus elevee indique une plus grande sensibilite a cette perturbation,
pas automatiquement une meilleure explication clinique.

| Ensemble | Patches remplaces | GNNExplainer | Aleatoire |
|---|---:|---:|---:|
| development | 5% | 0.925 | 1.469 |
| development | 10% | 1.904 | 1.940 |
| development | 20% | 3.322 | 2.845 |
| previously_examined_holdout | 5% | 0.905 | 0.834 |
| previously_examined_holdout | 10% | 1.242 | 1.347 |
| previously_examined_holdout | 20% | 2.105 | 1.727 |

## Interpretation

GNNExplainer ne depasse pas systematiquement les controles aleatoires.
A 20%, son effet moyen est superieur ; a 10%, il est inferieur dans les deux groupes.
Aucun test de significativite ou gain clinique n est revendique.
Les graines et controles ne constituent pas des patients independants.

Les heatmaps sont presentees sur une echelle absolue 0..1 ; les valeurs proches
ne doivent pas etre presentees comme des separations fortes. Le top 10% est
un classement de budget fixe, pas une annotation de tumeur.

## Figures et fichiers

- deletion_curve.png : courbes descriptives sans intervalle de confiance.
- TCGA-*.png : image, masque de scores et top 10%, graine 11 fixee.
- *_masks.npz : scores de noeuds et aretes pour chaque graine.
- deletion.csv : tous les effets signes et absolus, sans selection.
- explanations.csv : plages de scores et controles de poids inchanges.
- summary.json : version PyG, coefficients, protocole et empreintes.

Reproduction : scripts/run_gnnexplainer_baseline.py (refuse d ecraser les resultats).
Synthese seule : scripts/report_gnnexplainer.py.

## Limites et source

Il s agit d une baseline adaptee de regression, pas d une reproduction exacte
d un article diagnostique. Remplacer des embeddings ne garantit aucune
plausibilite histologique. Les masques d aretes sont sauvegardes, sans test
separe de leur qualite. Ce resultat ne demontre pas la contribution GCIA.

[GNNExplainer, Ying et al.](https://arxiv.org/abs/1903.03894) ;
[implementation PyG](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.explain.algorithm.GNNExplainer.html).

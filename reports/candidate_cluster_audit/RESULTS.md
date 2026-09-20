# Audit exploratoire des cinq clusters candidats

GCN de patches gelé, ensemble de trois initialisations, 22 patients du test déjà examiné. Noms proposés par IA, sans validation médicale.

Tous les patches du cluster sont remplacés par la moyenne des embeddings d'apprentissage. Vingt témoins par patient ont le même nombre de patches et la même distribution de degrés. Les arêtes restent inchangées.

Baisse de confiance dans la classe initialement prédite, en points de pourcentage. Valeur négative : augmentation de confiance. Patients sans le cluster exclus de son estimation conditionnelle.

| Cluster | Patients présents / 22 | Patches moyens | Ciblé | Témoin | Différence [IC 95 %] |
|---|---:|---:|---:|---:|---:|
| 0 | 17 | 37.9 | 39.42 | 22.15 | 17.27 [10.36, 24.05] |
| 1 | 8 | 15.6 | 4.93 | 4.69 | 0.24 [-6.78, 5.43] |
| 2 | 13 | 23.0 | 4.21 | 8.97 | -4.76 [-8.92, -0.64] |
| 3 | 6 | 7.3 | 10.39 | 1.75 | 8.64 [-0.70, 18.52] |
| 4 | 18 | 16.4 | 2.29 | 6.31 | -4.03 [-6.86, -1.72] |

## Interprétation et limites

Une différence positive indique une sensibilité plus forte aux patches de ce cluster qu'aux témoins. Elle ne démontre pas une dépendance au concept médical nommé. Les effets de différents clusters ne sont pas directement comparables : effectifs et tailles diffèrent.

L'occlusion ne préserve ni la plausibilité histologique ni les autres concepts. Les changements des cinq fractions de clusters sont enregistrés dans interventions.csv ; ce sont des réponses du clustering, pas des annotations cliniques. Les contrôles ne sont appariés ni en norme de perturbation ni en contiguïté spatiale. Les normes sont enregistrées.

IC descriptifs par bootstrap apparié sur les patients, sans correction des cinq comparaisons. Les graines des modèles ne sont pas des patients supplémentaires. Pas de nouveau probe clinique, d'édition de pixels ou d'intervention sur des cellules typées. Aucun réentraînement ni téléchargement.

Fichiers : protocol.json, summary.json, presence.csv, interventions.csv et effects.png/pdf.

## Verification du sens de l'intervention

Variation moyenne de la fraction du cluster cible apres recalcul du GNN. Une hausse invalide son interpretation comme suppression du concept.

- Cluster 0 : -59.28 points.
- Cluster 1 : -24.41 points.
- Cluster 2 : +5.53 points.
- Cluster 3 : -11.46 points.
- Cluster 4 : -24.31 points.

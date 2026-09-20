# BRACS : expérience binaire élargie

Étude pilote N (normal) contre IC (carcinome invasif), une région par patient.
34 patients d'apprentissage, 10 de validation et 22 nouveaux patients de test interne.
Encodeur PCam gelé, 120 époques, graines 11/23/37 et moyenne des probabilités.

## Validation

| Modèle | Balanced accuracy | AUC |
|---|---:|---:|
| mean_mlp | 58.3% | 0.750 |
| gcn | 79.2% | 0.875 |

Modèle retenu avant ouverture du test : **gcn**.

## Test interne

- Exactitude : 90.9% ; intervalle de Wilson à 95 % : [72.2%, 97.5%].
- Balanced accuracy : 90.2%.
- AUC : 0.955.
- Matrice de confusion (lignes réelles, colonnes prédites ; ordre N, IC) : [[13, 1], [1, 7]].

## Portée scientifique

Ce résultat teste un socle de classification, pas la contribution interventionnelle GCIA.
Les graphes représentent des patches, pas des cellules segmentées. Les expériences
multi-cohortes, l'ancrage pathologique des concepts et la fidélité des interventions
ne sont pas établis par cette expérience. L'abstract complet n'est donc pas validé.

Le recrutement privilégie les petits fichiers et le test ne contient que 22 patients.
La résolution physique est inconnue après redimensionnement. Il ne s'agit pas du test
officiel BRACS. Certains anciens patients de test sont maintenant en apprentissage ;
le nouveau test est distinct et ne permet pas de comparaison directe avec l'ancien score.
Aucun ajustement ne doit être guidé par ce nouveau test désormais consulté.

Traçabilité : results.json, selection.json, predictions.csv et test_opened.json
dans ce dossier ; protocole configs/bracs_expanded_protocol.json.

---
noteId: "170ddba0b53e11f1b1f059eca98e70d3"
tags: []

---

# Statut GCIA

Généré par `scripts/generate_status.py`. L'inventaire vérifie la présence et la lisibilité des résultats, pas leur validité clinique.

| Expérience pilote | Fichier de résultats |
|---|---|
| Segmentation MoNuSeg | [reports/segmentation_evaluation/summary.json](reports/segmentation_evaluation/summary.json) |
| Classification cellulaire | [reports/cell_gnn_comparison/results.json](reports/cell_gnn_comparison/results.json) |
| XAI cellulaire sur modèles gelés | [reports/cell_xai/summary.json](reports/cell_xai/summary.json) |
| Occlusion des clusters de patches | [reports/candidate_cluster_audit/summary.json](reports/candidate_cluster_audit/summary.json) |
| Audit des frontières de patches | [reports/cluster_structure_audit/summary.json](reports/cluster_structure_audit/summary.json) |

## Démonstrateurs retirés de la voie scientifique

`scripts/acquire_bach.py` refuse une fausse acquisition ; générateur explicitement synthétique dans `examples/toy_demo/`. Les anciens master et run_cell_graph_xai sont bloqués. Leurs sources sont archivées en texte non exécutable. Le wrapper CellGNNExplainer utilise désormais PyG officiel. Le rewiring non contraint et le formulaire clinique sans H&E sont désactivés.

La revue d'images réelle reste disponible via `scripts/prepare_expert_round.py`. Les dossiers historiques préremplis ne sont pas des avis de spécialistes. Les annotations reçues sont produites par IA ; une confirmation par spécialiste est rapportée par l'utilisateur et conservée séparément. Elle n'est pas une validation indépendante en aveugle.

## Non terminé

- Vraie acquisition/évaluation BACH et test externe verrouillé.
- Concepts indépendamment annotés, typage cellulaire et échelle physique vérifiés.
- Interventions cellulaires sélectives, plausibilité et rewiring spatial contraint.
- Benchmark synthétique à vérité connue avec modèles réellement entraînés ; les fixtures label-codées ne le remplacent pas.
- DeepSets, GNN multicouches, TCAV/LEACE sur protocole commun, nouvelle cohorte et statistiques confirmatoires.

Les contrôles de suppression d'arêtes historiques autorisent explicitement le chevauchement avec les cibles. Ils sont préservés pour reproductibilité ; ils ne représentent pas des contrôles exclusivement non-cibles. Le null-probe exige maintenant une ligne par patient et des patients disjoints ; son score de permutations des labels d'apprentissage reste un diagnostic exploratoire, pas une procédure confirmatoire complète.

Voir [corrections et limites](reports/CODE_REVIEW_REMEDIATION.md).

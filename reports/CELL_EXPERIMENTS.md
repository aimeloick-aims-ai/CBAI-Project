---
noteId: "dad184a0b4f411f1ac794713d9e3a04f"
tags: []

---

# Bilan des expériences cellulaires

Patients traités : 66 ; graphes exploitables : 65 ; exclusions explicites : 1.

Trois architectures, trois graines chacune, 120 époques fixes. Aucun réglage sur le test.
Le test est historique et déjà exploré. Les graphes restent candidats, à échelle physique inconnue.

| Architecture | Exactitude | Balanced accuracy | AUC | Patients de test |
|---|---:|---:|---:|---:|
| GCN | 72.7% | 73.2% | 0.741 | 22 |
| GraphSAGE | 72.7% | 73.2% | 0.741 | 22 |
| GATv2 | 72.7% | 75.9% | 0.741 | 22 |

## Ce que cette exécution termine

Évaluation des 14 images MoNuSeg, traitement de tous les patients BRACS disponibles, exclusions tracées, comparaison des trois GNN cellulaires et sauvegarde des poids/prédictions.

## Ce qui n'est pas démontré

La validation de segmentation BRACS, la validation externe BACH, les audits GCIA des nouveaux GNN cellulaires et la revue anatomopathologique ne sont pas achevés par cette exécution. Les audits XAI antérieurs concernent les modèles de patches. Aucune réussite clinique n'est déduite automatiquement des scores.

Voir `segmentation_evaluation/RESULTS.md`, `cell_cohort/excluded.csv` et `cell_gnn_comparison/results.json`.
